use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

use ash::vk::{self};

use common::MeshVert;

use gpu_allocator::vulkan::Allocator;

use crate::{
    app::{mesh_data::MeshDataBuffers, scene::ModelUniformBufferObject},
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
        },
        render_pass::RenderPass,
        ComputePipeline, GraphicsPipeline, ShaderModule,
    },
    VkHandle,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info, DrawPipeline,
};

pub struct ComputeCulledMesh {
    graphics_pipeline: GraphicsPipeline,
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    should_cull_buffer: Arc<TBuffer<u32>>,
}

impl ComputeCulledMesh {
    pub fn new(
        core: Arc<Core>,
        screen: &Screen,

        mesh_data: &MeshDataBuffers,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: &RenderPass,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
        uniform_transform_buffer: Arc<TBuffer<ModelUniformBufferObject>>,
        uniform_camera_buffers: &[Arc<impl AsBuffer>],
        cluster_count: u32,
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
            include_bytes!("../../shaders/spv/should_draw.comp"),
            ubo_layout.clone(),
            "Should Draw Pipeline",
        );

        let should_cull_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![1; cluster_count as _],
            "Should Cull Buffer",
        );

        let descriptor_sets = create_compute_culled_meshes_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            uniform_camera_buffers,
            &should_cull_buffer,
            &mesh_data.vertex_buffer,
            &mesh_data.meshlet_buffer,
            &mesh_data.cluster_buffer,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            descriptor_sets,
            screen: None,
            core,
            descriptor_pool,
            should_cull_buffer,
            should_draw_pipeline,
        }
    }
}

impl DrawPipeline for ComputeCulledMesh {
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
        ));
    }

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: CommandBufferGroup,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &ComputeCulledMesh,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
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

            let descriptor_sets_to_bind = [*core_draw.descriptor_sets[i]];

            unsafe {
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

                device.cmd_dispatch(command_buffer, submesh_count, 1, 1);

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

                //let vertex_buffers = [vertex_buffer];
                //let offsets = [0_u64];

                //device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                //device.cmd_bind_index_buffer(
                //    command_buffer,
                //    index_buffer,
                //    0,
                //    vk::IndexType::UINT32,
                //);
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.fn_mesh_shader.cmd_draw_mesh_tasks(
                    command_buffer,
                    submesh_count,
                    instance_count,
                    1,
                );
                // device.cmd_draw_indexed(
                //     command_buffer,
                //     RECT_TEX_COORD_INDICES_DATA.len() as u32,
                //     1,
                //     0,
                //     0,
                //     0,
                // );

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

fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, bindings))
}

fn create_compute_culled_meshes_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<TBuffer<ModelUniformBufferObject>>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
    should_draw_buffer: &Arc<impl AsBuffer>,
    vertex_buffer: &Arc<TBuffer<MeshVert>>,
    meshlet_buffer: &Arc<impl AsBuffer>,
    submesh_buffer: &Arc<impl AsBuffer>,
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
                        buf: uniform_transform_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: submesh_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: should_draw_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: meshlet_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: vertex_buffer.buffer(), //
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}
