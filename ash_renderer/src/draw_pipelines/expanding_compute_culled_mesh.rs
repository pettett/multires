use std::{
    ffi::CString,
    mem,
    sync::{Arc, Mutex},
    time,
};

use ash::vk::{self};

use common::MeshVert;

use gpu_allocator::vulkan::Allocator;

use crate::{
    app::{
        frame_measure::RollingMeasure,
        mesh_data::MeshData,
        renderer::Renderer,
        scene::{ModelUniformBufferObject, Scene},
    },
    core::Core,
    screen::Screen,
    utility::{
        buffer::{AsBuffer, TBuffer},
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::CommandPool,
            descriptor_pool::{
                DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
                DescriptorWriteData,
            },
            query_pool::QueryPool,
        },
        render_pass::RenderPass,
        ComputePipeline, GraphicsPipeline, ShaderModule,
    },
    VkHandle, CLEAR_COL, CLEAR_VALUES, TASK_GROUP_SIZE,
};

use super::{
    indirect_tasks::MeshInvocationsQueryResults, init_color_blend_attachment_states,
    init_depth_state_create_info, init_multisample_state_create_info,
    init_rasterization_statue_create_info, BufferRange, DrawPipeline,
};

pub struct ExpandingComputeCulledMesh {
    graphics_pipeline: GraphicsPipeline,
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    cluster_draw_buffer: Arc<TBuffer<u32>>,
    indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
    range_buffer: Arc<TBuffer<BufferRange>>,

    last_sample: time::Instant,
    mesh_invocations: RollingMeasure<u32, 60>,
    query_pool: Arc<QueryPool<MeshInvocationsQueryResults>>,
    query: bool,
}

impl ExpandingComputeCulledMesh {
    pub fn new(
        core: Arc<Core>,
        renderer: &Renderer,
        screen: &Screen,
        scene: &Scene,
        mesh_data: &MeshData,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: &RenderPass,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let should_draw_pipeline = ComputePipeline::create_compute_pipeline(
            &core,
            include_bytes!("../../shaders/spv/expanding_should_draw.comp"),
            ubo_layout.clone(),
            "Should Draw Pipeline",
        );

        let cluster_draw_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            graphics_queue,
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
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
            "Indirect Task Buffer",
        );

        let task_range_data = vec![BufferRange { start: 0, end: 0 }; scene.instances + 1];

        let range_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_range_data,
            "Indirect Range Buffer",
        );

        let descriptor_sets = create_compute_culled_meshes_descriptor_sets(
            &core.device,
            &descriptor_pool,
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

        Self {
            graphics_pipeline,
            descriptor_sets,
            screen: None,
            core,
            descriptor_pool,
            cluster_draw_buffer,
            should_draw_pipeline,
            indirect_task_buffer,
            range_buffer,

            query_pool,
            query: renderer.query,
            last_sample: time::Instant::now(),
            mesh_invocations: Default::default(),
        }
    }
}

impl DrawPipeline for ExpandingComputeCulledMesh {
    fn draw(&self, frame_index: usize) -> vk::CommandBuffer {
        self.screen.as_ref().unwrap().command_buffers[frame_index]
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
    command_buffers: CommandBufferGroup,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &ExpandingComputeCulledMesh,
        core: &Core,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> Self {
        let device = core.device.clone();

        let command_buffers = CommandBufferGroup::new(
            core.command_pool.clone(),
            screen.swapchain_framebuffers.len() as _,
        );

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder(); //.flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass.handle())
                .framebuffer(screen.swapchain_framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                })
                .clear_values(&CLEAR_VALUES);

            let descriptor_sets_to_bind = [*core_draw.descriptor_sets[i]];

            // let compute_to_compute_barriers = [vk::BufferMemoryBarrier2::builder()
            //     .buffer(core_draw.range_buffer.handle())
            //     .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            //     .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
            //     .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            //     .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            //     .size(vk::WHOLE_SIZE)
            //     .build()];

            // let compute_to_compute_dependency_info =
            //     vk::DependencyInfo::builder().buffer_memory_barriers(&compute_to_compute_barriers);

            let compute_to_task_barriers = [
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.indirect_task_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .size(vk::WHOLE_SIZE)
                    .build(),
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.cluster_draw_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
                    .size(vk::WHOLE_SIZE)
                    .build(),
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.range_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
                    .size(vk::WHOLE_SIZE)
                    .build(),
            ];

            let compute_to_task_dependency_info =
                vk::DependencyInfo::builder().buffer_memory_barriers(&compute_to_task_barriers);

            unsafe {
                // core_draw
                //     .cluster_draw_buffer
                //     .get_buffer()
                //     .fill(command_buffer, 0);

                let q = i as _;
                let query = core_draw.query && q == 0;
                if query {
                    core_draw.query_pool.reset(command_buffer, q);
                }

                core_draw.range_buffer.get_buffer().fill(command_buffer, 0);

                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    *core_draw.should_draw_pipeline,
                );

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout(),
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
                    command_buffer,
                    1,
                    core_draw.indirect_task_buffer.len() as _,
                    1,
                );

                // Mark barrier between compute write and task read, and between indirect write and indirect read
                device.cmd_pipeline_barrier2(command_buffer, &compute_to_task_dependency_info);

                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                device.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: screen.swapchain().extent,
                    }],
                );
                device.cmd_set_viewport(
                    command_buffer,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: screen.swapchain().extent.width as _,
                        height: screen.swapchain().extent.height as _,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );

                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.handle(),
                );
                {
                    let _qry = if query {
                        Some(core_draw.query_pool.begin_query(command_buffer, q))
                    } else {
                        None
                    };

                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        core_draw.graphics_pipeline.layout(),
                        0,
                        &descriptor_sets_to_bind,
                        &[],
                    );

                    core_draw
                        .indirect_task_buffer
                        .draw_tasks_indirect(command_buffer);
                }

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        Self {
            command_buffers,
            device,
            command_pool: core.command_pool.clone(),
        }
    }
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> GraphicsPipeline {
    let task_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!(
            "../../shaders/spv/mesh_shader_compute_cull.task"
        )),
    );
    let mesh_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.mesh")),
    );
    let frag_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/frag_colour.frag")),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .module(task_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::TASK_EXT)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(mesh_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(frag_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_extent.width as f32,
        height: swapchain_extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_extent,
    }];

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .scissors(&scissors)
        .viewports(&viewports)
        .build();

    let rasterization_statue_create_info = init_rasterization_statue_create_info();

    let multisample_state_create_info = init_multisample_state_create_info();

    let depth_state_create_info = init_depth_state_create_info();

    let color_blend_attachment_states = init_color_blend_attachment_states();

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .attachments(&color_blend_attachment_states)
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let set_layouts = [ubo_set_layout.handle()];

    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]);

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Failed to create pipeline layout!")
    };

    let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .rasterization_state(&rasterization_statue_create_info)
        .viewport_state(&viewport_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .depth_stencil_state(&depth_state_create_info)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout)
        .render_pass(render_pass.handle())
        .build()];

    let graphics_pipelines = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    GraphicsPipeline::new(
        device,
        graphics_pipelines[0],
        pipeline_layout,
        ubo_set_layout,
    )
}

fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::MESH_EXT
                | vk::ShaderStageFlags::TASK_EXT
                | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::TASK_EXT,
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, bindings))
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
    let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
    for _ in 0..swapchain_images_size {
        layouts.push(descriptor_set_layout.handle());
    }

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool.handle())
        .set_layouts(&layouts);

    let vk_descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .expect("Failed to allocate descriptor sets!")
    };

    let descriptor_sets: Vec<_> = vk_descriptor_sets
        .into_iter()
        .enumerate()
        .map(|(i, set)| {
            DescriptorSet::new(
                set,
                descriptor_pool.clone(),
                descriptor_set_layout.clone(),
                device.clone(),
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
                    DescriptorWriteData::Buffer {
                        buf: mesh_data.meshlet_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: mesh_data.vertex_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: indirect_data_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: range_buffer.buffer(), //
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}
