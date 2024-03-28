use std::{ffi::CString, sync::Arc};

use ash::vk::{self};

use common::MeshVert;

use crate::{
    app::{
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
        },
        render_pass::RenderPass,
        GraphicsPipeline, ShaderModule,
    },
    vertex::Vertex,
    VkHandle, CLEAR_VALUES,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info,
    render_multires_indices::create_traditional_graphics_pipeline, DrawPipeline,
};

pub struct DrawIndirect {
    graphics_pipeline: GraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    index_buffer: Arc<TBuffer<u32>>,
    query: bool,
}

impl DrawIndirect {
    pub fn new(renderer: &Renderer, mesh_data: &MeshData, scene: &Scene) -> Self {
        let ubo_layout = create_descriptor_set_layout(renderer.core.device.clone());

        let graphics_pipeline = create_traditional_graphics_pipeline(
            &renderer.core,
            &renderer.render_pass,
            renderer.screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let instance_count = scene.uniform_transform_buffer.len();

        let mut draw_indexed_commands = Vec::with_capacity(instance_count);

        draw_indexed_commands.push(vk::DrawIndexedIndirectCommand {
            index_count: mesh_data.index_buffer.len() as _,
            instance_count: instance_count as _,
            first_index: 0,
            vertex_offset: 0,
            first_instance: 0,
        });

        let draw_indexed_indirect_buffer = TBuffer::new_filled(
            &renderer.core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &draw_indexed_commands,
            "Draw Indexed Indirect Buffer",
        );

        let descriptor_sets = create_compute_culled_indices_descriptor_sets(
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
            draw_indexed_indirect_buffer,
            index_buffer: mesh_data.index_buffer.clone(),
        }
    }
}

impl DrawPipeline for DrawIndirect {
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

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: CommandBufferGroup,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &DrawIndirect,
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

            let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

            unsafe {
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

                device.cmd_bind_index_buffer(
                    command_buffer,
                    core_draw.index_buffer.handle(),
                    0,
                    vk::IndexType::UINT32,
                );

                device.cmd_bind_vertex_buffers(
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
                device.cmd_draw_indexed_indirect(
                    command_buffer,
                    core_draw.draw_indexed_indirect_buffer.handle(),
                    0,
                    1,
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
            vis: vk::ShaderStageFlags::VERTEX,
        },
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::VERTEX,
        },
    ];
    Arc::new(DescriptorSetLayout::new(device, bindings))
}

fn create_compute_culled_indices_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<TBuffer<ModelUniformBufferObject>>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
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
                        // 0
                        buf: uniform_transform_buffer.buffer(),
                    },
                    DescriptorWriteData::Empty,
                    DescriptorWriteData::Empty,
                    DescriptorWriteData::Buffer {
                        // 3
                        buf: uniform_camera_buffers[i].buffer(),
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}
