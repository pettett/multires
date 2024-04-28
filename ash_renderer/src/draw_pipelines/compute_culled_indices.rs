use std::sync::Arc;

use ash::vk::{self};

use crate::{
    app::{mesh_data::MeshData, renderer::Renderer, scene::Scene},
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
            query_pool::TypelessQueryPool,
        },
        render_pass::RenderPass,
        screen::Screen,
        ComputePipeline, PipelineLayout,
    },
    VkHandle,
};

use super::{
    render_multires::RenderMultires, render_multires_indices::RenderMultiresIndices, DrawPipeline,
};

pub struct ComputeCulledIndices {
    allocate_indices_pipeline: ComputePipeline,
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    should_cull_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    indirect_compute_buffer: Arc<TBuffer<vk::DispatchIndirectCommand>>,
    render_indices: RenderMultiresIndices,
    timestamp_query_pool: Option<Arc<TypelessQueryPool>>,
}

impl ComputeCulledIndices {
    pub fn new(core: Arc<Core>, renderer: &Renderer, scene: &Scene, mesh_data: &MeshData) -> Self {
        let ubo_layout = create_descriptor_set_layout(&core);

        let pipeline_layout = PipelineLayout::new_single_push::<u32>(
            core.device.clone(),
            ubo_layout.clone(),
            vk::ShaderStageFlags::COMPUTE,
        );

        // This pipeline pushes the instance index
        let should_draw_pipeline = ComputePipeline::create_compute_with_layout(
            &core,
            include_bytes!("../../shaders/spv/should_draw.comp"),
            pipeline_layout.clone(),
            "Should Draw Pipeline",
        );

        let compact_indices_pipeline = ComputePipeline::create_compute_with_layout(
            &core,
            include_bytes!("../../shaders/spv/compact_indices.comp"),
            pipeline_layout.clone(),
            "Compact Indices Pipeline",
        );

        let allocate_indices_pipeline = ComputePipeline::create_compute_with_layout(
            &core,
            include_bytes!("../../shaders/spv/allocate_index_ranges.comp"),
            pipeline_layout,
            "Allocate Indices Pipeline",
        );

        let should_cull_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![0; (1 + mesh_data.cluster_count) as _],
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
        let mut compute_commands = Vec::with_capacity(instance_count);

        draw_indexed_commands.push(vk::DrawIndexedIndirectCommand::default());

        for i in 0..instance_count {
            draw_indexed_commands.push(vk::DrawIndexedIndirectCommand::default());

            // This will get fixed in post (adaptive dispatch counts)
            compute_commands.push(vk::DispatchIndirectCommand { x: 1, y: 1, z: 1 })
        }

        let draw_indexed_indirect_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &draw_indexed_commands,
            "Draw Indexed Indirect Buffer",
        );

        let indirect_compute_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &compute_commands,
            "Indirect Compute Buffer",
        );

        let descriptor_sets = renderer.descriptor_pool.alloc(
            &ubo_layout,
            renderer.screen.swapchain().images.len(),
            |i| {
                vec![
                    DescriptorWriteData::Buffer {
                        // 0
                        buf: scene.uniform_transform_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 1
                        buf: should_cull_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 2
                        buf: mesh_data.cluster_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 3
                        buf: scene.uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        // 4
                        buf: indirect_compute_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 5
                        buf: result_indices_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 6
                        buf: mesh_data.meshlet_index_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 6
                        buf: draw_indexed_indirect_buffer.buffer(), //
                    },
                ]
            },
        );

        let render_indices = RenderMultiresIndices::new(
            &core,
            &renderer,
            mesh_data.vertex_buffer.clone(),
            result_indices_buffer.clone(),
            draw_indexed_indirect_buffer.clone(),
            indirect_compute_buffer.clone(),
            scene,
            compact_indices_pipeline,
        );

        Self {
            //    compact_indices_pipeline,
            descriptor_sets,
            screen: None,
            should_cull_buffer,
            should_draw_pipeline,
            draw_indexed_indirect_buffer,
            indirect_compute_buffer,
            allocate_indices_pipeline,
            render_indices,
            timestamp_query_pool: renderer.get_timestamp_query(),
        }
    }
}

impl DrawPipeline for ComputeCulledIndices {
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
        core_draw: &ComputeCulledIndices,
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
            let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];
            let should_cull_buffer_barrier = [vk::BufferMemoryBarrier2::default()
                .buffer(core_draw.should_cull_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .size(vk::WHOLE_SIZE)];

            let should_cull_buffer_dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&should_cull_buffer_barrier);

            let draw_indirect_buffer_barrier = [vk::BufferMemoryBarrier2::default()
                .buffer(core_draw.draw_indexed_indirect_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                )
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
                .size(vk::WHOLE_SIZE)];

            let draw_indirect_buffer_dependency_info =
                vk::DependencyInfo::default().buffer_memory_barriers(&draw_indirect_buffer_barrier);

            unsafe {
                if i == 0 {
                    core_draw
                        .timestamp_query_pool
                        .as_ref()
                        .map(|pool| pool.write_timestamp_top(*command_buffer));
                }

                device.cmd_bind_descriptor_sets(
                    *command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout().handle(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                // Fill indirect draw with 0s
                device.cmd_fill_buffer(
                    *command_buffer,
                    core_draw.draw_indexed_indirect_buffer.handle(),
                    0,
                    core_draw.draw_indexed_indirect_buffer.size(),
                    0,
                );

                // Instance 0 is intermediate data / the entire scene
                for instance in 0..core_draw.indirect_compute_buffer.len() {
                    // reset the count value to 1 (first index, as 0 reserved for counter)
                    device.cmd_fill_buffer(
                        *command_buffer,
                        core_draw.should_cull_buffer.handle(),
                        0,
                        4,
                        2,
                    );
                    //reset tri-count to 0.
                    device.cmd_fill_buffer(
                        *command_buffer,
                        core_draw.should_cull_buffer.handle(),
                        4,
                        4,
                        0,
                    );

                    // All the shaders share this layout, so we only need to push once
                    core_draw
                        .allocate_indices_pipeline
                        .layout()
                        .push_single_constant(0, *command_buffer, instance as u32);

                    {
                        // Generate cluster selection
                        device.cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            core_draw.should_draw_pipeline.handle(),
                        );

                        core_draw
                            .indirect_compute_buffer
                            .dispatch_indirect(*command_buffer, instance);

                        // Force previous compute shader to be complete before this one
                        device.cmd_pipeline_barrier2(
                            *command_buffer,
                            &should_cull_buffer_dependency_info,
                        );
                    }
                    {
                        // Initialise indirect draw arguments

                        device.cmd_bind_pipeline(
                            *command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            core_draw.allocate_indices_pipeline.handle(),
                        );

                        device.cmd_push_constants(
                            *command_buffer,
                            core_draw.allocate_indices_pipeline.layout().handle(),
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            bytemuck::cast_slice(&[instance as u32]),
                        );

                        device.cmd_dispatch(*command_buffer, 1, 1, 1);

                        device.cmd_pipeline_barrier2(
                            *command_buffer,
                            &draw_indirect_buffer_dependency_info,
                        );
                    }

                    {
                        // Fill into indirect draw arguments
                        core_draw.render_indices.compact_indices(
                            *command_buffer,
                            &core.device,
                            instance,
                        );
                    }
                    // device.cmd_pipeline_barrier2(
                    //     *command_buffer,
                    //     &should_cull_buffer_dependency_info,
                    // );
                    // device.cmd_pipeline_barrier2(
                    //     *command_buffer,
                    //     &draw_indirect_buffer_dependency_info,
                    // );
                }

                core_draw.render_indices.render(
                    &mut command_buffer,
                    &core.device,
                    screen,
                    render_pass,
                    i,
                );

                if i == 0 {
                    core_draw
                        .timestamp_query_pool
                        .as_ref()
                        .map(|pool| pool.write_timestamp_bottom(*command_buffer));
                }
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
            vis: vk::ShaderStageFlags::COMPUTE,
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

    Arc::new(DescriptorSetLayout::new(
        core,
        bindings,
        "compute culled indices layout",
    ))
}
