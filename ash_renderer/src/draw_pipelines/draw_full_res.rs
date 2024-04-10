use std::sync::Arc;

use ash::vk::{self};

use common::MeshVert;

use crate::{
    app::{mesh_data::MeshData, renderer::Renderer, scene::Scene},
    core::Core,
    utility::{
        buffer::TBuffer,
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::{CommandBuffer, CommandPool},
            descriptor_pool::{DescriptorPool, DescriptorSet},
        },
        render_pass::RenderPass,
        screen::Screen,
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

pub struct DrawFullRes {
    graphics_pipeline: GraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    //draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    index_buffer: Arc<TBuffer<u32>>,
    query: bool,
    instance_count: usize,
}

impl DrawFullRes {
    pub fn new(renderer: &Renderer, mesh_data: &MeshData, scene: &Scene) -> Self {
        let ubo_layout = create_traditional_graphics_descriptor_set_layout(&renderer.core);

        let graphics_pipeline = create_traditional_graphics_pipeline(
            &renderer.core,
            &renderer.render_pass,
            renderer.screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let index_buffer = mesh_data.lod_chain[0].index_buffer.clone();

        let instance_count = scene.uniform_transform_buffer.len();

        // let draw_indexed_commands = [vk::DrawIndexedIndirectCommand {
        //     index_count: index_buffer.len() as _,
        //     instance_count: instance_count as _,
        //     first_index: 0,
        //     vertex_offset: 0,
        //     first_instance: 0,
        // }];

        // let draw_indexed_indirect_buffer = TBuffer::new_filled(
        //     &renderer.core,
        //     renderer.allocator.clone(),
        //     renderer.graphics_queue,
        //     vk::BufferUsageFlags::INDIRECT_BUFFER,
        //     &draw_indexed_commands,
        //     "Draw Indexed Indirect Buffer",
        // );

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
            screen: None,
            core: renderer.core.clone(),
            descriptor_pool: renderer.descriptor_pool.clone(),
            query: renderer.query,
            vertex_buffer: mesh_data.vertex_buffer.clone(),
            //    draw_indexed_indirect_buffer,
            index_buffer,
            instance_count,
        }
    }
}

impl DrawPipeline for DrawFullRes {
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
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: Arc<CommandBufferGroup>,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &DrawFullRes,
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
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass.handle())
                .framebuffer(screen.swapchain_framebuffers[i].handle())
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                })
                .clear_values(&CLEAR_VALUES);

            unsafe {
                device.cmd_begin_render_pass(
                    *command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                command_buffer.set_dynamic_screen(screen);

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

                device.cmd_bind_index_buffer(
                    *command_buffer,
                    core_draw.index_buffer.handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_bind_vertex_buffers(
                    *command_buffer,
                    0,
                    &[core_draw.vertex_buffer.handle()],
                    &[0],
                );

                // Each instance has their own indirect drawing buffer, tracing out their position in the result buffer
                device.cmd_draw_indexed(
                    *command_buffer,
                    core_draw.index_buffer.len() as _,
                    core_draw.instance_count as _,
                    0,
                    0,
                    0,
                );

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
