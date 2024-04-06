use std::{ffi::CString, sync::Arc, time};

use ash::vk::{self};
use bevy_ecs::system::Query;
use common::MeshVert;
use common_renderer::components::transform::Transform;

use crate::{
    app::{
        frame_measure::RollingMeasure, mesh_data::MeshData, renderer::Renderer,
        scene::ModelUniformBufferObject,
    },
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
            query_pool::{QueryPool, QueryResult},
        },
        render_pass::RenderPass,
        screen::Screen,
        GraphicsPipeline, PipelineLayout, ShaderModule,
    },
    VkHandle, CLEAR_VALUES,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info, DrawPipeline,
};
#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct MeshInvocationsQueryResults {
    pub mesh: u32,
    pub avail: u32,
}

impl QueryResult for MeshInvocationsQueryResults {
    fn flags() -> vk::QueryPipelineStatisticFlags {
        vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT
    }
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
    indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,

    // Evaluation data
    last_sample: time::Instant,
    mesh_invocations: RollingMeasure<u32, 60>,
    //task_invocations: RollingMeasure<u32, 60>,
    query_pool: Arc<QueryPool<MeshInvocationsQueryResults>>,
    query: bool,
}

impl IndirectTasks {
    pub fn new(
        core: Arc<Core>,
        renderer: &Renderer,
        screen: &Screen,
        transforms: Query<&Transform>,
        mesh_data: &MeshData,
        uniform_transform_buffer: Arc<TBuffer<ModelUniformBufferObject>>,
        uniform_camera_buffers: &[Arc<impl AsBuffer>],
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(&core);

        let graphics_pipeline = create_graphics_pipeline(
            &core,
            &renderer.render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
            renderer.fragment(),
            renderer.mesh_mode,
        );
        let mut task_indirect_data = Vec::new();

        for _e in transforms.iter() {
            task_indirect_data.push(vk::DrawMeshTasksIndirectCommandEXT {
                group_count_x: 8,
                group_count_y: 1,
                group_count_z: 1,
            });
        }

        let indirect_task_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
            "Indirect Task Buffer",
        );

        let descriptor_sets = create_descriptor_sets(
            &core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &mesh_data.vertex_buffer,
            match &renderer.mesh_mode {
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
            descriptor_pool: renderer.descriptor_pool.clone(),
            screen: None,
            indirect_task_buffer,
            query_pool,
            query: renderer.query,
            core,
            last_sample: time::Instant::now(),
            mesh_invocations: Default::default(),
            //task_invocations: Default::default(),
        }
    }
}

impl DrawPipeline for IndirectTasks {
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
            &self.indirect_task_buffer,
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
        core_draw: &IndirectTasks,
        core: &Core,
        screen: &Screen,
        render_pass: &RenderPass,
        _indirect_task_buffer: &TBuffer<vk::DrawMeshTasksIndirectCommandEXT>,
    ) -> Self {
        let device = core.device.clone();
        let mut command_buffers = CommandBufferGroup::new(
            core.command_pool.clone(),
            screen.swapchain_framebuffers.len() as _,
        );

        for (i, mut command_buffer) in command_buffers.iter_to_fill().enumerate() {
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass.handle())
                .framebuffer(screen.swapchain_framebuffers[i].handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                })
                .clear_values(&CLEAR_VALUES);

            unsafe {
                let q = i as _;
                let query = core_draw.query && q == 0;
                if query {
                    core_draw.query_pool.reset(*command_buffer, q);
                }

                device.cmd_begin_render_pass(
                    *command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                command_buffer.set_dynamic_screen(screen);

                // ---------- Pipeline bound, use queries
                {
                    let _qry = if query {
                        Some(core_draw.query_pool.begin_query(*command_buffer, q))
                    } else {
                        None
                    };

                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        core_draw.graphics_pipeline.handle(),
                    );

                    let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

                    device.cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        core_draw.graphics_pipeline.layout().handle(),
                        0,
                        &descriptor_sets_to_bind,
                        &[],
                    );

                    core_draw
                        .indirect_task_buffer
                        .draw_tasks_indirect(*command_buffer);
                }

                device.cmd_end_render_pass(*command_buffer);
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

    Arc::new(DescriptorSetLayout::new(
        core,
        bindings,
        "indirect tasks layout",
    ))
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
    descriptor_pool.alloc(descriptor_set_layout, swapchain_images_size, |i| {
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
        ]
    })
}

fn create_graphics_pipeline(
    core: &Core,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
    frag_shader_module: &ShaderModule,
    mode: MeshShaderMode,
) -> GraphicsPipeline {
    let task_shader_module = ShaderModule::new(
        core.device.clone(),
        include_bytes!("../../shaders/spv/mesh-shader.task"),
    );

    let mesh_shader_module = ShaderModule::new(
        core.device.clone(),
        match mode {
            MeshShaderMode::TriangleList => include_bytes!("../../shaders/spv/mesh-shader.mesh"),
            MeshShaderMode::TriangleStrip => {
                include_bytes!("../../shaders/spv/mesh_tri_strip.mesh")
            }
        },
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(task_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::TASK_EXT)
            ,
        vk::PipelineShaderStageCreateInfo::default()
            .module(mesh_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::MESH_EXT)
            ,
        vk::PipelineShaderStageCreateInfo::default()
            .module(frag_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            ,
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

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
        .scissors(&scissors)
        .viewports(&viewports)
        ;

    let rasterization_statue_create_info = init_rasterization_statue_create_info();

    let multisample_state_create_info = init_multisample_state_create_info();

    let depth_state_create_info = init_depth_state_create_info();

    let color_blend_attachment_states = init_color_blend_attachment_states();

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&color_blend_attachment_states)
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]);

    let pipeline_layout = PipelineLayout::new(core.device.clone(), ubo_set_layout);

    let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .rasterization_state(&rasterization_statue_create_info)
        .viewport_state(&viewport_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .depth_stencil_state(&depth_state_create_info)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout.handle())
        .render_pass(render_pass.handle())
        ];

    let graphics_pipelines = unsafe {
        core.device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    core.name_object("Indirect Tasks", graphics_pipelines[0]);

    GraphicsPipeline::new_raw(core.device.clone(), graphics_pipelines[0], pipeline_layout)
}
