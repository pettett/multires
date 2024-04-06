use std::{sync::Arc, time};

use ash::vk::{self};

use crate::{
    app::{frame_measure::RollingMeasure, mesh_data::MeshData, renderer::Renderer, scene::Scene},
    core::Core,
    utility::{
        buffer::{AsBuffer, TBuffer},
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::{CommandBuffer, CommandPool},
            descriptor_pool::{
                DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
                DescriptorWriteData,
            },
            query_pool::QueryPool,
        },
        render_pass::RenderPass,
        screen::Screen,
        ComputePipeline,
    },
    VkHandle,
};

use super::{
    indirect_tasks::MeshInvocationsQueryResults, render_multires::RenderMultires,
    render_multires_meshlets::RenderMultiresMeshlets, BufferRange, DrawPipeline,
};

pub struct ExpandingComputeCulledMesh {
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,

    cluster_draw_buffer: Arc<TBuffer<u32>>,
    indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
    range_buffer: Arc<TBuffer<BufferRange>>,

    last_sample: time::Instant,
    mesh_invocations: RollingMeasure<u32, 60>,
    query_pool: Arc<QueryPool<MeshInvocationsQueryResults>>,
    query: bool,
    render_meshlets: RenderMultiresMeshlets,
}

impl ExpandingComputeCulledMesh {
    pub fn new(
        core: Arc<Core>,
        renderer: &Renderer,
        screen: &Screen,
        scene: &Scene,
        mesh_data: &MeshData,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(&core);

        let should_draw_pipeline = ComputePipeline::create_compute_pipeline(
            &core,
            include_bytes!("../../shaders/spv/expanding_should_draw.comp"),
            ubo_layout.clone(),
            "Should Draw Pipeline",
        );

        let cluster_draw_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![1; (scene.instances.max(1) as u32 * mesh_data.cluster_count / 2) as _],
            "Should Cull Buffer",
        );

        let task_indirect_data = vec![
            vk::DrawMeshTasksIndirectCommandEXT {
                group_count_x: 8,
                group_count_y: 1,
                group_count_z: 1,
            };
            scene.instances.max(1)
        ];

        let indirect_task_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
            "Indirect Task Buffer",
        );

        let task_range_data = vec![BufferRange { start: 0, end: 0 }; scene.instances + 1];

        let range_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_range_data,
            "Indirect Range Buffer",
        );

        let descriptor_sets = create_compute_culled_meshes_descriptor_sets(
            &core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &cluster_draw_buffer,
            &indirect_task_buffer,
            &range_buffer,
            &scene,
            &mesh_data,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        let query_pool = QueryPool::new(core.device.clone(), 1);

        let render_meshlets = RenderMultiresMeshlets::new(
            &core,
            renderer,
            screen,
            scene,
            mesh_data,
            indirect_task_buffer.clone(),
            range_buffer.clone(),
            cluster_draw_buffer.clone(),
        );

        Self {
            descriptor_sets,
            screen: None,
            cluster_draw_buffer,
            should_draw_pipeline,
            indirect_task_buffer,
            range_buffer,
            render_meshlets,
            query_pool,
            query: renderer.query,
            last_sample: time::Instant::now(),
            mesh_invocations: Default::default(),
        }
    }
}

impl DrawPipeline for ExpandingComputeCulledMesh {
    fn draw(
        &self,
        frame_index: usize,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> &CommandBuffer {
        self.screen
            .as_ref()
            .unwrap()
            .command_buffers
            .get(frame_index)
    }
    fn init_swapchain(&mut self, core: &Core, screen: &Screen, render_pass: &RenderPass) {
        self.screen = Some(ScreenData::create_command_buffers(
            &self,
            core,
            screen,
            render_pass,
        ));
    }

    fn stats_gui(&mut self, ui: &mut egui::Ui, image_index: usize) {
        if image_index == 0 && self.last_sample.elapsed() > time::Duration::from_secs_f32(0.01) {
            if let Some(results) = self.query_pool.get_results(0) {
                assert!(results.avail > 0);

                self.mesh_invocations.tick(results.mesh);
            }

            self.last_sample = time::Instant::now();
        }

        self.mesh_invocations.gui("Mesh Invocations", ui);
        //self.task_invocations.gui("Task Invocations", ui);
    }
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: Arc<CommandBufferGroup>,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &ExpandingComputeCulledMesh,
        core: &Core,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> Self {
        let device = core.device.clone();

        let mut command_buffers = CommandBufferGroup::new(
            core.command_pool.clone(),
            screen.swapchain_framebuffers.len() as _,
        );

        for (i, mut command_buffer) in command_buffers.iter_to_fill().enumerate() {
            let descriptor_sets_to_bind = [*core_draw.descriptor_sets[i]];

            // let compute_to_compute_barriers = [vk::BufferMemoryBarrier2::default()
            //     .buffer(core_draw.range_buffer.handle())
            //     .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            //     .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            //     .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            //     .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            //     .size(vk::WHOLE_SIZE)
            //     ];

            // let compute_to_compute_dependency_info =
            //     vk::DependencyInfo::default().buffer_memory_barriers(&compute_to_compute_barriers);

            let compute_to_task_barriers = [
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.indirect_task_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .size(vk::WHOLE_SIZE)
                    ,
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.cluster_draw_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
                    .size(vk::WHOLE_SIZE)
                    ,
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.range_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
                    .size(vk::WHOLE_SIZE)
                    ,
            ];

            let compute_to_task_dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&compute_to_task_barriers);

            unsafe {
                // core_draw
                //     .cluster_draw_buffer
                //     .get_buffer()
                //     .fill(command_buffer, 0);

                let q = i as _;
                let query = core_draw.query && q == 0;
                if query {
                    core_draw.query_pool.reset(*command_buffer, q);
                }

                core_draw.range_buffer.get_buffer().fill(*command_buffer, 0);

                device.cmd_bind_pipeline(
                    *command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *core_draw.should_draw_pipeline,
                );

                device.cmd_bind_descriptor_sets(
                    *command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout().handle(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                // for i in 0..core_draw.indirect_task_buffer.len() {
                //     println!("{i}");
                //     // Push this instance's clusters to the buffer, then do the next one
                //     device.cmd_dispatch_base(command_buffer, 0, i as _, 0, 1, 1, 1);

                //     device
                //         .cmd_pipeline_barrier2(command_buffer, &compute_to_compute_dependency_info);
                // }

                device.cmd_dispatch(
                    *command_buffer,
                    1,
                    core_draw.indirect_task_buffer.len() as _,
                    1,
                );

                // Mark barrier between compute write and task read, and between indirect write and indirect read
                device.cmd_pipeline_barrier2(*command_buffer, &compute_to_task_dependency_info);

                core_draw.render_meshlets.render(
                    &mut command_buffer,
                    &core.device,
                    screen,
                    render_pass,
                    i,
                );
            }
        }

        Self {
            command_buffers: Arc::new(command_buffers),
            device,
            command_pool: core.command_pool.clone(),
        }
    }
}

fn create_descriptor_set_layout(core: &Core) -> Arc<DescriptorSetLayout> {
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
    ];

    Arc::new(DescriptorSetLayout::new(
        core,
        bindings,
        "expanding culling mesh compute layout",
    ))
}

fn create_compute_culled_meshes_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    cluster_draw_buffer: &Arc<impl AsBuffer>,
    indirect_data_buffer: &Arc<impl AsBuffer>,
    range_buffer: &Arc<impl AsBuffer>,

    scene: &Scene,
    mesh_data: &MeshData,
    swapchain_images_size: usize,
) -> Vec<DescriptorSet> {
    descriptor_pool.alloc(descriptor_set_layout, swapchain_images_size, |i| {
        vec![
            DescriptorWriteData::Buffer {
                buf: scene.uniform_transform_buffer.buffer(),
            },
            DescriptorWriteData::Buffer {
                buf: cluster_draw_buffer.buffer(),
            },
            DescriptorWriteData::Buffer {
                buf: mesh_data.cluster_buffer.buffer(), //
            },
            DescriptorWriteData::Buffer {
                buf: scene.uniform_camera_buffers[i].buffer(),
            },
            DescriptorWriteData::Empty,
            DescriptorWriteData::Empty,
            DescriptorWriteData::Buffer {
                buf: indirect_data_buffer.buffer(), //
            },
            DescriptorWriteData::Buffer {
                buf: range_buffer.buffer(), //
            },
        ]
    })
}
