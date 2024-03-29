use std::sync::Arc;

use ash::vk::{self};

use common::MeshVert;

use crate::{
    app::{mesh_data::MeshData, renderer::Renderer, scene::Scene},
    core::Core,
    screen::Screen,
    utility::{
        buffer::TBuffer,
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::{CommandBuffer, CommandPool},
            descriptor_pool::{DescriptorPool, DescriptorSet},
        },
        render_pass::RenderPass,
        GraphicsPipeline,
    },
    VkHandle, CLEAR_VALUES,
};

use super::{
    render_multires_indices::{
        create_traditional_graphics_descriptor_set_layout,
        create_traditional_graphics_descriptor_sets, create_traditional_graphics_pipeline,
    },
    DrawPipeline,
};

pub struct DrawLODChain {
    graphics_pipeline: GraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    index_buffers: Vec<Arc<TBuffer<u32>>>,
    query: bool,
    command_buffers: Vec<CommandBuffer>,
    instance_count: usize,
}

impl DrawLODChain {
    pub fn new(renderer: &Renderer, mesh_data: &MeshData, scene: &Scene) -> Self {
        let ubo_layout =
            create_traditional_graphics_descriptor_set_layout(renderer.core.device.clone());

        let graphics_pipeline = create_traditional_graphics_pipeline(
            &renderer.core,
            &renderer.render_pass,
            renderer.screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let instance_count = scene.uniform_transform_buffer.len();

        let descriptor_sets = create_traditional_graphics_descriptor_sets(
            &renderer.core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &scene.uniform_transform_buffer,
            &scene.uniform_camera_buffers,
            renderer.screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            descriptor_sets,
            core: renderer.core.clone(),
            descriptor_pool: renderer.descriptor_pool.clone(),
            query: renderer.query,
            vertex_buffer: mesh_data.vertex_buffer.clone(),
            index_buffers: mesh_data.lod_index_buffers.clone(),
            command_buffers: Vec::new(),
            instance_count,
        }
    }
}

impl DrawPipeline for DrawLODChain {
    fn draw(
        &self,
        frame_index: usize,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> &CommandBuffer {
        let core = &self.core;
        let device = core.device.clone();

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.handle())
            .framebuffer(screen.swapchain_framebuffers[frame_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: screen.swapchain().extent,
            })
            .clear_values(&CLEAR_VALUES);

        {
            let mut command_buffer_writer = self.command_buffers[frame_index].reset_and_write();

            unsafe {
                device.cmd_begin_render_pass(
                    *command_buffer_writer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                command_buffer_writer.set_dynamic_screen(screen);

                device.cmd_bind_pipeline(
                    *command_buffer_writer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline.handle(),
                );

                let descriptor_sets_to_bind = [self.descriptor_sets[frame_index].handle()];
                device.cmd_bind_descriptor_sets(
                    *command_buffer_writer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.cmd_bind_index_buffer(
                    *command_buffer_writer,
                    self.index_buffers[0].handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_bind_vertex_buffers(
                    *command_buffer_writer,
                    0,
                    &[self.vertex_buffer.handle()],
                    &[0],
                );

                for i in 0..self.instance_count {
                    // Each instance has their own indirect drawing buffer, tracing out their position in the result buffer
                    device.cmd_draw_indexed(
                        *command_buffer_writer,
                        self.index_buffers[0].len() as _,
                        1,
                        0,
                        0,
                        i as _,
                    );
                }

                device.cmd_end_render_pass(*command_buffer_writer);
            }
        }

        &self.command_buffers[frame_index]
    }
    fn init_swapchain(&mut self, core: &Core, screen: &Screen, render_pass: &RenderPass) {
        // only ever top-up our commands - we dont need to reset on resize
        while screen.swapchain_framebuffers.len() >= self.command_buffers.len() {
            self.command_buffers
                .push(core.command_pool.begin_one_shot_command());
        }
    }

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}
}
