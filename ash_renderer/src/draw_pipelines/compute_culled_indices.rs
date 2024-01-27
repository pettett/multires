use std::{
    ffi::CString,
    ptr,
    sync::{Arc, Mutex},
};

use ash::vk::{self};
use bevy_ecs::world::World;
use common::MeshVert;
use common_renderer::components::transform::Transform;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    screen::Screen,
    utility::{
        buffer::{AsBuffer, Buffer, TBuffer},
        device::Device,
        pooled::command_pool::CommandPool,
        pooled::descriptor_pool::{
            DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorWriteData,
        },
        render_pass::RenderPass,
        {ComputePipeline, GraphicsPipeline, ShaderModule},
    },
    vertex::Vertex,
    VkHandle, TASK_GROUP_SIZE,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info, DrawPipeline,
};

pub struct ComputeCulledIndices {
    graphics_pipeline: GraphicsPipeline,
    should_draw_pipeline: ComputePipeline,
    compact_indices_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    should_cull_buffer: Arc<TBuffer<u32>>,
    result_indices_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    indices_buffer: Arc<TBuffer<u32>>,
}

impl ComputeCulledIndices {
    pub fn new(
        core: Arc<Core>,
        screen: &Screen,
        world: &mut World,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: &RenderPass,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
        uniform_transform_buffer: Arc<Buffer>,
        uniform_camera_buffers: &[Arc<impl AsBuffer>],
        vertex_buffer: Arc<TBuffer<MeshVert>>,
        meshlet_buffer: Arc<Buffer>,
        submesh_buffer: Arc<Buffer>,
        indices_buffer: Arc<TBuffer<u32>>,
        instance_count: usize,
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

        let compact_indices_pipeline = ComputePipeline::create_compute_pipeline(
            &core,
            include_bytes!("../../shaders/spv/compact_indices.comp"),
            ubo_layout.clone(),
            "Compact Indices Pipeline",
        );

        let should_cull_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![0; cluster_count as _],
            "Should Cull Buffer",
        );

        let result_indices_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &vec![0; (indices_buffer.size() * 2) as usize],
            "Result Indices Buffer",
        );

        let mut draw_indexed_commands = Vec::with_capacity(instance_count);

        for i in 0..instance_count {
            draw_indexed_commands.push(vk::DrawIndexedIndirectCommand {
                index_count: indices_buffer.item_len() as _,
                instance_count: 1,
                first_index: (indices_buffer.item_len() * i) as _,
                vertex_offset: 0,
                first_instance: i as _,
            });
        }

        let draw_indexed_indirect_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &draw_indexed_commands,
            "Draw Indexed Indirect Buffer",
        );

        let descriptor_sets = create_compute_culled_indices_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &should_cull_buffer,
            &meshlet_buffer,
            &submesh_buffer,
            //&texture_image,
            &result_indices_buffer,
            &indices_buffer,
            &draw_indexed_indirect_buffer,
            screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            compact_indices_pipeline,
            descriptor_sets,
            screen: None,
            core,
            descriptor_pool,
            should_cull_buffer,
            vertex_buffer,
            should_draw_pipeline,
            result_indices_buffer,
            draw_indexed_indirect_buffer,
            indices_buffer,
        }
    }
}

impl DrawPipeline for ComputeCulledIndices {
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

    fn stats_gui(&mut self, ui: &mut egui::Ui, image_index: usize) {}
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &ComputeCulledIndices,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
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

            let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

            let should_draw_buffer_barriers = [
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.should_cull_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE)
                    .build(),
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.draw_indexed_indirect_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE)
                    .build(),
            ];

            let should_draw_dependency_info =
                vk::DependencyInfo::builder().buffer_memory_barriers(&should_draw_buffer_barriers);

            let result_indices_buffer_barriers = [
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.result_indices_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                    .size(vk::WHOLE_SIZE)
                    .build(),
                vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.draw_indexed_indirect_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .size(vk::WHOLE_SIZE)
                    .build(),
            ];

            let result_indices_dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(&result_indices_buffer_barriers);

            unsafe {
                device.handle.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                for instance in 0..instance_count {
                    device.handle.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        core_draw.should_draw_pipeline.handle(),
                    );

                    device.handle.cmd_dispatch_base(
                        command_buffer,
                        0,
                        instance,
                        0,
                        submesh_count,
                        1,
                        1,
                    );

                    // Force previous compute shader to be complete before this one
                    device
                        .handle
                        .cmd_pipeline_barrier2(command_buffer, &should_draw_dependency_info);

                    device.handle.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        core_draw.compact_indices_pipeline.handle(),
                    );

                    device.handle.cmd_dispatch_base(
                        command_buffer,
                        0,
                        instance,
                        0,
                        submesh_count.div_ceil(16),
                        1,
                        1,
                    );

                    // Force result indices to be complete before continuing.
                    // Because we re-bind the pipelines every time, we need to specify this dependency for all
                    // Otherwise, only the last instance will have correct info
                    device
                        .handle
                        .cmd_pipeline_barrier2(command_buffer, &result_indices_dependency_info);
                }

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

                device.handle.cmd_bind_index_buffer(
                    command_buffer,
                    core_draw.result_indices_buffer.handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                device.handle.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[core_draw.vertex_buffer.handle()],
                    &[0],
                );
                // device.handle.cmd_draw_indexed(
                //     command_buffer,
                //     core_draw.indices_buffer.item_len() as _,
                //     instance_count,
                //     0,
                //     0,
                //     0,
                // );

                // Each instance has their own indirect drawing buffer, tracing out their position in the result buffer
                device.handle.cmd_draw_indexed_indirect(
                    command_buffer,
                    core_draw.draw_indexed_indirect_buffer.handle(),
                    0,
                    instance_count,
                    core_draw.draw_indexed_indirect_buffer.stride() as _,
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

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> GraphicsPipeline {
    let vert_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/vert.vert")),
    );
    let frag_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/frag_pbr.frag")),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .module(vert_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::VERTEX)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(frag_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let binding_description = MeshVert::get_binding_descriptions();
    let attribute_description = MeshVert::get_attribute_descriptions();

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_description)
        .vertex_attribute_descriptions(&attribute_description);

    let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
        primitive_restart_enable: vk::FALSE,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

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
        .vertex_input_state(&vertex_input_state_create_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
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

    GraphicsPipeline::new(
        device,
        graphics_pipelines[0],
        pipeline_layout,
        ubo_set_layout,
    )
}

fn create_descriptor_set_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let ubo_layout_bindings = [
        // layout (binding = 0) readonly buffer ModelUniformBufferObject {
        // 	mat4 models[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(
                vk::ShaderStageFlags::MESH_EXT
                    | vk::ShaderStageFlags::COMPUTE
                    | vk::ShaderStageFlags::VERTEX,
            )
            .build(),
        // layout(binding = 1) buffer Clusters {
        // 	ClusterData clusters[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout(binding = 2) buffer ShouldDraw {
        // 	uint should_draw[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout (binding = 3) uniform CameraUniformBufferObject {
        // 	CameraUniformObject ubo;
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX)
            .build(),
        // layout (std430, binding = 4) buffer InputBufferI {
        // 	s_meshlet meshlets[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT)
            .build(),
        // layout (std430, binding = 5) buffer InputBufferV {
        // 	Vertex verts[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(7)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];

    Arc::new(DescriptorSetLayout::new(device, &ubo_layout_bindings))
}

fn create_compute_culled_indices_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<Buffer>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
    should_draw_buffer: &Arc<impl AsBuffer>,
    meshlet_buffer: &Arc<Buffer>,
    submesh_buffer: &Arc<Buffer>,
    result_indices_buffer: &Arc<impl AsBuffer>,
    indices_buffer: &Arc<impl AsBuffer>,
    draw_indexed_indirect_buffer: &Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    //texture: &Image,
    swapchain_images_size: usize,
) -> Vec<DescriptorSet> {
    let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
    for _ in 0..swapchain_images_size {
        layouts.push(descriptor_set_layout.handle());
    }

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool.handle)
        .set_layouts(&layouts);

    let vk_descriptor_sets = unsafe {
        device
            .handle
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
                        // 0
                        buf: uniform_transform_buffer.clone(),
                    },
                    DescriptorWriteData::Buffer {
                        // 1
                        buf: submesh_buffer.clone(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 2
                        buf: should_draw_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 3
                        buf: uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 4
                        buf: meshlet_buffer.clone(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 5
                        buf: result_indices_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 6
                        buf: indices_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 6
                        buf: draw_indexed_indirect_buffer.buffer(), //
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}
