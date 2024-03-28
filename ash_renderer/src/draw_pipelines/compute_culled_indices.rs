use std::{
    sync::{Arc},
};

use ash::vk::{self};





use crate::{
    app::{
        mesh_data::MeshData,
        renderer::Renderer,
        scene::{Scene},
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
        ComputePipeline,
    },
    VkHandle,
};

use super::{
    render_multires::RenderMultires, render_multires_indices::RenderMultiresIndices, DrawPipeline,
};

pub struct ComputeCulledIndices {
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    should_cull_buffer: Arc<TBuffer<u32>>,
    result_indices_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    render_indices: RenderMultiresIndices,
}

impl ComputeCulledIndices {
    pub fn new(core: Arc<Core>, renderer: &Renderer, scene: &Scene, mesh_data: &MeshData) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

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
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![0; mesh_data.cluster_count as _],
            "Should Cull Buffer",
        );

        let result_indices_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &vec![0; (mesh_data.meshlet_index_buffer.size() * 2) as usize],
            "Result Indices Buffer",
        );

        let instance_count = scene.uniform_transform_buffer.len();

        let mut draw_indexed_commands = Vec::with_capacity(instance_count);

        for i in 0..instance_count {
            draw_indexed_commands.push(vk::DrawIndexedIndirectCommand {
                index_count: mesh_data.meshlet_index_buffer.len() as _,
                instance_count: 1,
                first_index: (mesh_data.meshlet_index_buffer.len() * i) as _,
                vertex_offset: 0,
                first_instance: i as _,
            });
        }

        let draw_indexed_indirect_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &draw_indexed_commands,
            "Draw Indexed Indirect Buffer",
        );

        let descriptor_sets = create_compute_culled_indices_descriptor_sets(
            &core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &scene.uniform_transform_buffer,
            &scene.uniform_camera_buffers,
            &should_cull_buffer,
            &mesh_data.meshlet_buffer,
            &mesh_data.cluster_buffer,
            //&texture_image,
            &result_indices_buffer,
            &mesh_data.meshlet_index_buffer,
            &draw_indexed_indirect_buffer,
            renderer.screen.swapchain().images.len(),
        );

        let render_indices = RenderMultiresIndices::new(
            &core,
            &renderer.screen,
            &renderer.render_pass,
            mesh_data.vertex_buffer.clone(),
            result_indices_buffer.clone(),
            draw_indexed_indirect_buffer.clone(),
            renderer.descriptor_pool.clone(),
            scene,
            compact_indices_pipeline,
        );

        Self {
            //    compact_indices_pipeline,
            descriptor_sets,
            screen: None,
            should_cull_buffer,
            should_draw_pipeline,
            result_indices_buffer,
            draw_indexed_indirect_buffer,
            render_indices,
        }
    }
}

impl DrawPipeline for ComputeCulledIndices {
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
        core_draw: &ComputeCulledIndices,
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

            let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

            let should_draw_buffer_barriers = [
                *vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.should_cull_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE),
                *vk::BufferMemoryBarrier2::builder()
                    .buffer(core_draw.draw_indexed_indirect_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE),
            ];

            let should_draw_dependency_info =
                vk::DependencyInfo::builder().buffer_memory_barriers(&should_draw_buffer_barriers);

            unsafe {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                for instance in 0..core_draw.draw_indexed_indirect_buffer.len() {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        core_draw.should_draw_pipeline.handle(),
                    );

                    device.cmd_dispatch_base(
                        command_buffer,
                        0,
                        instance as _,
                        0,
                        core_draw.should_cull_buffer.len() as _,
                        1,
                        1,
                    );

                    // Force previous compute shader to be complete before this one
                    device.cmd_pipeline_barrier2(command_buffer, &should_draw_dependency_info);

                    core_draw.render_indices.compact_indices(
                        command_buffer,
                        &core.device,
                        instance,
                        core_draw.should_cull_buffer.len(),
                    );
                }

                core_draw.render_indices.render(
                    command_buffer,
                    &core.device,
                    screen,
                    render_pass,
                    i,
                );

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
            vis: vk::ShaderStageFlags::MESH_EXT
                | vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::VERTEX,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE,
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, bindings))
}

fn create_compute_culled_indices_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    uniform_transform_buffer: &Arc<impl AsBuffer>,
    uniform_camera_buffers: &[Arc<impl AsBuffer>],
    should_draw_buffer: &Arc<impl AsBuffer>,
    meshlet_buffer: &Arc<impl AsBuffer>,
    cluster_buffer: &Arc<impl AsBuffer>,
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
                    DescriptorWriteData::Buffer {
                        // 1
                        buf: should_draw_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 2
                        buf: cluster_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 3
                        buf: uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 4
                        buf: meshlet_buffer.buffer(), //
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
