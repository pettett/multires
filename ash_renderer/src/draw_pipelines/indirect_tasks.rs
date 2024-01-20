use std::{ffi::CString, ptr, sync::Arc};

use ash::vk::{self};
use bevy_ecs::world::World;
use common_renderer::components::transform::Transform;

use crate::{
    core::Core,
    screen::Screen,
    utility::{
        buffer::{AsBuffer, Buffer, TypedBuffer},
        command_pool::CommandPool,
        descriptor_pool::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
        device::Device,
        pipeline::{Pipeline, ShaderModule},
        render_pass::RenderPass,
    },
    VkHandle, TASK_GROUP_SIZE,
};

use super::DrawPipeline;

pub struct IndirectTasks {
    graphics_pipeline: Pipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<IndirectTasksScreen>,
    indirect_task_buffer: Arc<TypedBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
}

impl IndirectTasks {
    pub fn new(
        core: Arc<Core>,
        screen: &Screen,
        world: &mut World,
        render_pass: &RenderPass,
        swapchain_extent: vk::Extent2D,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
        uniform_transform_buffer: Arc<Buffer>,
        uniform_camera_buffers: &[impl AsBuffer],
        vertex_buffer: Arc<Buffer>,
        meshlet_buffer: Arc<Buffer>,
        submesh_buffer: Arc<Buffer>,
        cluster_count: u32,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            render_pass,
            swapchain_extent,
            ubo_layout.clone(),
        );
        let mut task_indirect_data = Vec::new();

        for e in world.query::<&Transform>().iter(world) {
            task_indirect_data.push(vk::DrawMeshTasksIndirectCommandEXT {
                group_count_x: cluster_count.div_ceil(TASK_GROUP_SIZE) / 4,
                group_count_y: 1,
                group_count_z: 1,
            });
        }
        let indirect_task_buffer = TypedBuffer::new_filled(
            core.device.clone(),
            core.allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
        );

        let descriptor_sets = DescriptorSet::create_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &vertex_buffer,
            &meshlet_buffer,
            &submesh_buffer,
            &indirect_task_buffer,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            descriptor_sets,
            screen: None,
            indirect_task_buffer,
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
        self.screen = Some(IndirectTasksScreen::create_command_buffers(
            &self,
            core,
            screen,
            submesh_count,
            instance_count,
            render_pass,
            &self.indirect_task_buffer,
        ));
    }
}

pub struct IndirectTasksScreen {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
}
impl Drop for IndirectTasksScreen {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .free_command_buffers(self.command_pool.handle, &self.command_buffers);
        }
    }
}
impl IndirectTasksScreen {
    pub fn create_command_buffers(
        core_draw: &IndirectTasks,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
        indirect_task_buffer: &TypedBuffer<vk::DrawMeshTasksIndirectCommandEXT>,
    ) -> Self {
        let device = core.device.clone();
        let command_buffers = core
            .command_pool
            .allocate_group(screen.swapchain_framebuffers.len() as _);

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .handle
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
                device.handle.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                device.handle.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: screen.swapchain().extent,
                    }],
                );
                device.handle.cmd_set_viewport(
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

                device.handle.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.handle(),
                );

                //let vertex_buffers = [vertex_buffer];
                //let offsets = [0_u64];
                let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

                //device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                //device.cmd_bind_index_buffer(
                //    command_buffer,
                //    index_buffer,
                //    0,
                //    vk::IndexType::UINT32,
                //);
                device.handle.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.fn_mesh_shader.cmd_draw_mesh_tasks_indirect(
                    command_buffer,
                    indirect_task_buffer.buffer(),
                    0,
                    instance_count,
                    indirect_task_buffer.stride() as _,
                );
                // device.cmd_draw_indexed(
                //     command_buffer,
                //     RECT_TEX_COORD_INDICES_DATA.len() as u32,
                //     1,
                //     0,
                //     0,
                //     0,
                // );

                device.handle.cmd_end_render_pass(command_buffer);

                device
                    .handle
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

pub fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let ubo_layout_bindings = [
        vk::DescriptorSetLayoutBinding {
            // transform uniform
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // sampler uniform
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // sampler uniform
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::TASK_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // camera uniform
            binding: 5,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // indirect draw params buffer array
            binding: 6,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::TASK_EXT,
            p_immutable_samplers: ptr::null(),
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, &ubo_layout_bindings))
}

pub fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> Pipeline {
    let task_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.task")),
    );
    let mesh_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.mesh")),
    );
    let frag_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.frag")),
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

    let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL)
        .rasterizer_discard_enable(false)
        .depth_bias_enable(false)
        .build();

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
        ..Default::default()
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
        ..Default::default()
    };

    let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: vk::TRUE,
        depth_write_enable: vk::TRUE,
        depth_compare_op: vk::CompareOp::LESS,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
        ..Default::default()
    };

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        ..Default::default()
    }];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_blend_attachment_states.len() as u32,
        p_attachments: color_blend_attachment_states.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };

    let set_layouts = [ubo_set_layout.handle()];

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        ..Default::default()
    };

    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&[vk::DynamicState::SCISSOR, vk::DynamicState::VIEWPORT]);

    let pipeline_layout = unsafe {
        device
            .handle
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
            .handle
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    Pipeline::new(
        device,
        graphics_pipelines[0],
        pipeline_layout,
        ubo_set_layout,
    )
}