// Draw indices from a should draw buffer

use std::{ffi, sync::Arc};

use crate::{
    app::{
        material::{Material, MAIN_FUNCTION_NAME},
        renderer::Renderer,
        scene::Scene,
    },
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
        ComputePipeline, GraphicsPipeline, PipelineLayout, ShaderModule,
    },
    vertex::Vertex,
    VkHandle,
};
use ash::vk;
use common::MeshVert;

use crate::{
    utility::{device::Device, render_pass::RenderPass, screen::Screen},
    CLEAR_VALUES,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, render_multires::RenderMultires,
};

pub struct RenderMultiresIndices {
    graphics_pipeline: GraphicsPipeline,
    compact_indices_pipeline: ComputePipeline,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    result_indices_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    /// Buffer that will control the number of compaction invocations
    indirect_compute_buffer: Arc<TBuffer<vk::DispatchIndirectCommand>>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl RenderMultiresIndices {
    pub fn new(
        core: &Core,
        renderer: &Renderer,
        vertex_buffer: Arc<TBuffer<MeshVert>>,
        result_indices_buffer: Arc<TBuffer<u32>>,
        draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
        indirect_compute_buffer: Arc<TBuffer<vk::DispatchIndirectCommand>>,
        scene: &Scene,
        compact_indices_pipeline: ComputePipeline,
    ) -> Self {
        let ubo_layout = create_traditional_graphics_descriptor_set_layout(core);

        let graphics_pipeline = create_primitive_graphics_pipeline(
            &core,
            &renderer.render_pass,
            renderer.screen.swapchain().extent,
            ubo_layout.clone(),
            renderer.fragment(),
        );

        // let compact_indices_pipeline = ComputePipeline::create_compute_pipeline(
        //     &core,
        //     include_bytes!("../../shaders/spv/compact_indices.comp"),
        //     ubo_layout.clone(),
        //     "Compact Indices Pipeline",
        // );

        let descriptor_sets = create_traditional_graphics_descriptor_sets(
            &core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &scene.uniform_transform_buffer,
            &scene.uniform_camera_buffers,
            renderer.screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            vertex_buffer,
            result_indices_buffer,
            draw_indexed_indirect_buffer,
            descriptor_sets,
            indirect_compute_buffer,
            compact_indices_pipeline,
        }
    }

    pub fn compact_indices(&self, cmd: vk::CommandBuffer, device: &Device, instance: usize) {
        self.bind_compact(cmd, device);
        self.dispatch_compact(cmd, device, instance);
    }

    pub fn bind_compact(&self, cmd: vk::CommandBuffer, device: &Device) {
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.compact_indices_pipeline.handle(),
            );
        }
    }

    pub fn dispatch_compact(&self, cmd: vk::CommandBuffer, device: &Device, instance: usize) {
        let result_indices_buffer_barriers = [
            vk::BufferMemoryBarrier2::default()
                .buffer(self.result_indices_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::INDEX_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::INDEX_INPUT)
                .size(vk::WHOLE_SIZE),
            vk::BufferMemoryBarrier2::default()
                .buffer(self.draw_indexed_indirect_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                .size(vk::WHOLE_SIZE),
        ];

        let result_indices_dependency_info =
            vk::DependencyInfo::default().buffer_memory_barriers(&result_indices_buffer_barriers);

        self.indirect_compute_buffer
            .dispatch_indirect(cmd, instance);

        unsafe {
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

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(screen.swapchain_framebuffers[frame].handle())
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

            device.cmd_bind_descriptor_sets(
                **cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.layout().handle(),
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
            self.draw_indexed_indirect_buffer.draw_indexed_indirect(
                **cmd,
                1,
                self.draw_indexed_indirect_buffer.len() - 1,
            );

            device.cmd_end_render_pass(**cmd);
        }
    }
}

pub fn create_primitive_graphics_pipeline(
    core: &Core,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
    frag_shader_module: &Material,
) -> GraphicsPipeline {
    let vert_shader_module = ShaderModule::new(
        core.device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/vert.vert")),
    );

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(vert_shader_module.handle())
            .name(&MAIN_FUNCTION_NAME)
            .stage(vk::ShaderStageFlags::VERTEX),
        frag_shader_module.shader_stage_create_info(),
    ];

    let binding_description = MeshVert::get_binding_descriptions();
    let attribute_description = MeshVert::get_attribute_descriptions();

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::default()
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

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
        .scissors(&scissors)
        .viewports(&viewports);

    let rasterization_statue_create_info = frag_shader_module.rasterization_statue_create_info();

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
        .vertex_input_state(&vertex_input_state_create_info)
        .input_assembly_state(&vertex_input_assembly_state_info)
        .layout(pipeline_layout.handle())
        .render_pass(render_pass.handle())];

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

    GraphicsPipeline::new_raw(core.device.clone(), graphics_pipelines[0], pipeline_layout)
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
