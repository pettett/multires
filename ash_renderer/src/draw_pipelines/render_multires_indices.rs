// Draw indices from a should draw buffer

use std::{ffi, sync::Arc};

use crate::{
    app::scene::Scene,
    core::Core,
    utility::{
        buffer::{AsBuffer, TBuffer},
        pooled::{
            command_buffer_writer::CommandBufferWriter,
            descriptor_pool::{
                DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
                DescriptorWriteData,
            },
        },
        ComputePipeline, GraphicsPipeline, ShaderModule,
    },
    vertex::Vertex,
    VkHandle,
};
use ash::vk;
use common::MeshVert;

use crate::{
    screen::Screen,
    utility::{device::Device, render_pass::RenderPass},
    CLEAR_VALUES,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info,
    render_multires::RenderMultires,
};

pub struct RenderMultiresIndices {
    graphics_pipeline: GraphicsPipeline,
    compact_indices_pipeline: ComputePipeline,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    result_indices_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl RenderMultiresIndices {
    pub fn new(
        core: &Core,
        screen: &Screen,
        render_pass: &RenderPass,
        vertex_buffer: Arc<TBuffer<MeshVert>>,
        result_indices_buffer: Arc<TBuffer<u32>>,
        draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
        descriptor_pool: Arc<DescriptorPool>,
        scene: &Scene,
        compact_indices_pipeline: ComputePipeline,
    ) -> Self {
        let ubo_layout = create_traditional_graphics_descriptor_set_layout(core);

        let graphics_pipeline = create_traditional_graphics_pipeline(
            &core,
            render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
        );

        // let compact_indices_pipeline = ComputePipeline::create_compute_pipeline(
        //     &core,
        //     include_bytes!("../../shaders/spv/compact_indices.comp"),
        //     ubo_layout.clone(),
        //     "Compact Indices Pipeline",
        // );

        let descriptor_sets = create_traditional_graphics_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &scene.uniform_transform_buffer,
            &scene.uniform_camera_buffers,
            screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            vertex_buffer,
            result_indices_buffer,
            draw_indexed_indirect_buffer,
            descriptor_sets,
            compact_indices_pipeline,
        }
    }

    pub fn compact_indices(
        &self,
        cmd: vk::CommandBuffer,
        device: &Device,
        instance: usize,
        cluster_count: usize,
    ) {
        let result_indices_buffer_barriers = [
            *vk::BufferMemoryBarrier2::builder()
                .buffer(self.result_indices_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                .size(vk::WHOLE_SIZE),
            *vk::BufferMemoryBarrier2::builder()
                .buffer(self.draw_indexed_indirect_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .size(vk::WHOLE_SIZE),
        ];

        let result_indices_dependency_info =
            vk::DependencyInfo::builder().buffer_memory_barriers(&result_indices_buffer_barriers);

        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.compact_indices_pipeline.handle(),
            );

            device.cmd_dispatch_base(
                cmd,
                0,
                instance as _,
                0,
                cluster_count.div_ceil(16) as _,
                1,
                1,
            );

            // Force result indices to be complete before continuing.
            // Because we re-bind the pipelines every time, we need to specify this dependency for all
            // Otherwise, only the last instance will have correct info
            device.cmd_pipeline_barrier2(cmd, &result_indices_dependency_info);
        }
    }
}
impl RenderMultires for RenderMultiresIndices {
    fn render(
        &self,
        cmd: &mut CommandBufferWriter,
        device: &Device,
        screen: &Screen,
        render_pass: &RenderPass,
        frame: usize,
    ) {
        let descriptor_sets_to_bind = [self.descriptor_sets[frame].handle()];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.handle())
            .framebuffer(screen.swapchain_framebuffers[frame])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: screen.swapchain().extent,
            })
            .clear_values(&CLEAR_VALUES);

        unsafe {
            device.cmd_begin_render_pass(
                **cmd,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            cmd.set_dynamic_screen(screen);

            device.cmd_bind_pipeline(
                **cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.handle(),
            );

            //let vertex_buffers = [vertex_buffer];
            //let offsets = [0_u64];

            //device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
            //device.cmd_bind_index_buffer(
            //    cmd,
            //    index_buffer,
            //    0,
            //    vk::IndexType::UINT32,
            //);
            device.cmd_bind_descriptor_sets(
                **cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.layout(),
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            device.cmd_bind_index_buffer(
                **cmd,
                self.result_indices_buffer.handle(),
                0,
                vk::IndexType::UINT32,
            );

            device.cmd_bind_vertex_buffers(**cmd, 0, &[self.vertex_buffer.handle()], &[0]);

            // Each instance has their own indirect drawing buffer, tracing out their position in the result buffer
            device.cmd_draw_indexed_indirect(
                **cmd,
                self.draw_indexed_indirect_buffer.handle(),
                0,
                self.draw_indexed_indirect_buffer.len() as _,
                self.draw_indexed_indirect_buffer.stride() as _,
            );

            device.cmd_end_render_pass(**cmd);
        }
    }
}

pub fn create_traditional_graphics_pipeline(
    core: &Core,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> GraphicsPipeline {
    let vert_shader_module = ShaderModule::new(
        core.device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/vert.vert")),
    );
    let frag_shader_module = ShaderModule::new(
        core.device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/frag_pbr.frag")),
    );

    let main_function_name = ffi::CString::new("main").unwrap(); // the beginning function name in shader code.

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
        core.device
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
        core.device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    core.name_object("Compute Culled Indices", graphics_pipelines[0]);

    GraphicsPipeline::new(
        core.device.clone(),
        graphics_pipelines[0],
        pipeline_layout,
        ubo_set_layout,
    )
}

pub fn create_traditional_graphics_descriptor_set_layout(core: &Core) -> Arc<DescriptorSetLayout> {
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::VERTEX,
        },
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::VERTEX,
        },
    ];

    Arc::new(DescriptorSetLayout::new(core, bindings, "indices layout"))
}

pub fn create_traditional_graphics_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<impl AsBuffer>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
    swapchain_images_size: usize,
) -> Vec<DescriptorSet> {
    descriptor_pool.alloc(descriptor_set_layout, swapchain_images_size, |i| {
        vec![
            DescriptorWriteData::Buffer {
                // 0
                buf: uniform_transform_buffer.buffer(),
            },
            DescriptorWriteData::Empty,
            DescriptorWriteData::Empty,
            DescriptorWriteData::Buffer {
                // 3
                buf: uniform_camera_buffers[i].buffer(),
            },
        ]
    })
}
