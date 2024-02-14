use std::{
    ffi::CString,
    ptr,
    sync::{Arc, Mutex},
    time,
};

use ash::vk::{self};
use bevy_ecs::system::Query;
use common::MeshVert;
use common_renderer::components::transform::Transform;
use gpu_allocator::vulkan::Allocator;

use crate::{
    app::{
        frame_measure::RollingMeasure, mesh_data::MeshDataBuffers, scene::ModelUniformBufferObject,
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
        GraphicsPipeline, ShaderModule,
    },
    VkHandle,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info, DrawPipeline,
};
#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct IndirectTasksQueryResults {
    task: u32,
    mesh: u32,
    avail: u32,
}

pub struct DrawMeshTasksIndirect {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
    pub offset: u32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeshShaderMode {
    TriangleList,
    TriangleStrip,
}

pub struct IndirectTasks {
    graphics_pipeline: GraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    indirect_task_buffer: Arc<TBuffer<DrawMeshTasksIndirect>>,

    // Evaluation data
    last_sample: time::Instant,
    mesh_invocations: RollingMeasure<u32, 60>,
    task_invocations: RollingMeasure<u32, 60>,
    query_pool: Arc<QueryPool<IndirectTasksQueryResults>>,
    query: bool,
}

impl IndirectTasks {
    pub fn new(
        core: Arc<Core>,
        screen: &Screen,
        transforms: Query<&Transform>,
        mesh_data: &MeshDataBuffers,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: &RenderPass,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
        uniform_transform_buffer: Arc<TBuffer<ModelUniformBufferObject>>,
        uniform_camera_buffers: &[Arc<impl AsBuffer>],
        _cluster_count: u32,
        query: bool,
        mode: MeshShaderMode,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
            mode,
        );
        let mut task_indirect_data = Vec::new();

        for _e in transforms.iter() {
            task_indirect_data.push(DrawMeshTasksIndirect {
                group_count_x: 8,
                group_count_y: 1,
                group_count_z: 1,
                offset: 0,
            });
        }

        let indirect_task_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
            "Indirect Task Buffer",
        );

        let descriptor_sets = create_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &mesh_data.vertex_buffer,
            match mode {
                MeshShaderMode::TriangleList => &mesh_data.meshlet_buffer,
                MeshShaderMode::TriangleStrip => &mesh_data.stripped_meshlet_buffer,
            },
            &mesh_data.cluster_buffer,
            &indirect_task_buffer,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        let query_pool = QueryPool::new(core.device.clone(), 1);

        Self {
            graphics_pipeline,
            descriptor_sets,
            descriptor_pool,
            screen: None,
            indirect_task_buffer,
            query_pool,
            query,
            core,
            last_sample: time::Instant::now(),
            mesh_invocations: Default::default(),
            task_invocations: Default::default(),
        }
    }
}

impl DrawPipeline for IndirectTasks {
    fn draw(&self, frame_index: usize) -> vk::CommandBuffer {
        self.screen.as_ref().unwrap().command_buffers[frame_index]
    }
    fn init_swapchain(
        &mut self,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
    ) {
        self.screen = Some(ScreenData::create_command_buffers(
            &self,
            core,
            screen,
            submesh_count,
            instance_count,
            render_pass,
            &self.indirect_task_buffer,
        ));
    }

    fn stats_gui(&mut self, ui: &mut egui::Ui, image_index: usize) {
        if image_index == 0 && self.last_sample.elapsed() > time::Duration::from_secs_f32(0.01) {
            if let Some(results) = self.query_pool.get_results(0) {
                assert!(results.avail > 0);

                self.mesh_invocations.tick(results.mesh);
                self.task_invocations.tick(results.task);
            }

            self.last_sample = time::Instant::now();
        }

        self.mesh_invocations.gui("Mesh Invocations", ui);
        self.task_invocations.gui("Task Invocations", ui);
    }
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: CommandBufferGroup,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &IndirectTasks,
        core: &Core,
        screen: &Screen,
        _submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
        indirect_task_buffer: &TBuffer<DrawMeshTasksIndirect>,
    ) -> Self {
        let device = core.device.clone();
        let command_buffers = CommandBufferGroup::new(
            core.command_pool.clone(),
            screen.swapchain_framebuffers.len() as _,
        );

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    // clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass.handle())
                .framebuffer(screen.swapchain_framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                })
                .clear_values(&clear_values);

            unsafe {
                let q = i as _;
                let query = core_draw.query && q == 0;
                if query {
                    core_draw.query_pool.reset(command_buffer, q);
                }
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

                // ---------- Pipeline bound, use queries
                if query {
                    device.cmd_begin_query(
                        command_buffer,
                        core_draw.query_pool.handle(),
                        q,
                        vk::QueryControlFlags::empty(),
                    );
                }
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.handle(),
                );

                let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.fn_mesh_shader.cmd_draw_mesh_tasks_indirect(
                    command_buffer,
                    indirect_task_buffer.handle(),
                    0,
                    instance_count,
                    indirect_task_buffer.stride() as _,
                );
                if query {
                    device.cmd_end_query(command_buffer, core_draw.query_pool.handle(), q);
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

fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
        },
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT,
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, bindings))
}

fn create_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<TBuffer<ModelUniformBufferObject>>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
    vertex_buffer: &Arc<TBuffer<MeshVert>>,
    meshlet_buffer: &Arc<impl AsBuffer>,
    cluster_buffer: &Arc<impl AsBuffer>,
    indirect_draw_array_buffer: &Arc<impl AsBuffer>,
    //texture: &Image,
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
                        //  0
                        buf: uniform_transform_buffer.buffer(),
                    },
                    DescriptorWriteData::Empty,
                    DescriptorWriteData::Buffer {
                        //  2
                        buf: cluster_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 3
                        buf: uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        //  4
                        buf: meshlet_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        //  5
                        buf: vertex_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        //  6
                        buf: indirect_draw_array_buffer.buffer(),
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
    mode: MeshShaderMode,
) -> GraphicsPipeline {
    let task_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.task")),
    );

    let mesh_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(match mode {
            MeshShaderMode::TriangleList => include_bytes!("../../shaders/spv/mesh-shader.mesh"),
            MeshShaderMode::TriangleStrip => {
                include_bytes!("../../shaders/spv/mesh_tri_strip.mesh")
            }
        }),
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

    // let binding_description = VertexV3::get_binding_descriptions();
    // let attribute_description = VertexV3::get_attribute_descriptions();

    // let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
    //     s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    //     p_next: ptr::null(),
    //     flags: vk::PipelineVertexInputStateCreateFlags::empty(),
    //     vertex_attribute_description_count: attribute_description.len() as u32,
    //     p_vertex_attribute_descriptions: attribute_description.as_ptr(),
    //     vertex_binding_description_count: binding_description.len() as u32,
    //     p_vertex_binding_descriptions: binding_description.as_ptr(),
    // };
    // let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
    //     s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    //     flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
    //     p_next: ptr::null(),
    //     primitive_restart_enable: vk::FALSE,
    //     topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    // };

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

    //  {
    //     stage_count: shader_stages.len() as u32,
    //     p_stages: shader_stages.as_ptr(),
    //     p_viewport_state: &viewport_state_create_info,
    //     p_rasterization_state: &rasterization_statue_create_info,
    //     p_multisample_state: &multisample_state_create_info,
    //     p_depth_stencil_state: &depth_state_create_info,
    //     p_color_blend_state: &color_blend_state,
    //     layout: pipeline_layout,
    //     render_pass,
    //     subpass: 0,
    //     base_pipeline_handle: vk::Pipeline::null(),
    //     base_pipeline_index: -1,
    //     ..Default::default()
    // };

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
