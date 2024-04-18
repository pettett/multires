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
        },
        render_pass::RenderPass,
        screen::Screen,
        ComputePipeline, PipelineLayout,
    },
    VkHandle,
};

use super::{
    render_multires::RenderMultires, render_multires_indices::RenderMultiresIndices, BufferRange,
    DrawPipeline,
};

pub struct ExpandingComputeCulledIndices {
    should_draw_pipeline: ComputePipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    selected_cluster_buffer: Arc<TBuffer<u32>>,
    draw_indexed_indirect_buffer: Arc<TBuffer<vk::DrawIndexedIndirectCommand>>,
    render_indices: RenderMultiresIndices,
    indirect_compute_buffer: Arc<TBuffer<vk::DispatchIndirectCommand>>,
    range_buffer: Arc<TBuffer<BufferRange>>,
    push_pipeline_layout: Arc<PipelineLayout>,
}

impl ExpandingComputeCulledIndices {
    pub fn new(core: Arc<Core>, renderer: &Renderer, scene: &Scene, mesh_data: &MeshData) -> Self {
        let bindings = vec![
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX,
            },
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::COMPUTE,
            },
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::COMPUTE,
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
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::COMPUTE,
            },
        ];

        let ubo_layout = Arc::new(DescriptorSetLayout::new(
            &core,
            bindings,
            "expanding compute culled indices layout",
        ));

        let pipeline_layout = PipelineLayout::new(core.device.clone(), ubo_layout.clone());
        let push_pipeline_layout = PipelineLayout::new_single_push::<u32>(
            core.device.clone(),
            ubo_layout.clone(),
            vk::ShaderStageFlags::COMPUTE,
        );

        // This pipeline pushes the instance index
        let should_draw_pipeline = ComputePipeline::create_compute_with_layout(
            &core,
            include_bytes!("../../shaders/spv/expanding_should_draw_indices.comp"),
            pipeline_layout.clone(),
            "Should Draw Pipeline",
        );

        let compact_indices_pipeline = ComputePipeline::create_compute_with_layout(
            &core,
            include_bytes!("../../shaders/spv/expanding_compact_indices.comp"),
            push_pipeline_layout.clone(),
            "Compact Indices Pipeline",
        );

        let task_range_data = vec![BufferRange { start: 0, end: 0 }; scene.instances + 1];

        let range_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_range_data,
            "Indirect Range Buffer",
        );

        let instance_count = scene.uniform_transform_buffer.len();

        let result_indices_allocation = (mesh_data.meshlet_index_buffer.size()
            * f32::log2(1.0 + instance_count as f32) as u64)
            as usize
            / 2;

        // Roughly 200 tris per cluster
        let selected_cluster_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![0; (result_indices_allocation / 600) as _],
            "Should Cull Buffer",
        );

        let result_indices_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &vec![0; result_indices_allocation],
            "Result Indices Buffer",
        );

        println!("Allowing up to {result_indices_allocation} result indices");

        let mut draw_indexed_commands = Vec::with_capacity(instance_count);
        let mut compute_commands = Vec::with_capacity(instance_count);

        draw_indexed_commands.push(vk::DrawIndexedIndirectCommand::default());

        for i in 0..instance_count {
            draw_indexed_commands.push(vk::DrawIndexedIndirectCommand::default());

            // This will get fixed in post (adaptive dispatch counts)
            compute_commands.push(vk::DispatchIndirectCommand { x: 250, y: 1, z: 1 })
        }

        let indirect_compute_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &compute_commands,
            "Indirect Compute Buffer",
        );

        let draw_indexed_indirect_buffer = TBuffer::new_filled(
            &core,
            renderer.allocator.clone(),
            renderer.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            &draw_indexed_commands,
            "Draw Indexed Indirect Buffer",
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
                        buf: selected_cluster_buffer.buffer(),
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
                        buf: draw_indexed_indirect_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 5
                        buf: result_indices_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 6
                        buf: indirect_compute_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 7
                        buf: range_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        // 8
                        buf: mesh_data.meshlet_index_buffer.buffer(), //
                    },
                ]
            },
        );

        let render_indices = RenderMultiresIndices::new(
            &core,
            &renderer.screen,
            &renderer.render_pass,
            mesh_data.vertex_buffer.clone(),
            result_indices_buffer.clone(),
            draw_indexed_indirect_buffer.clone(),
            indirect_compute_buffer.clone(),
            renderer.descriptor_pool.clone(),
            scene,
            compact_indices_pipeline,
        );

        Self {
            //    compact_indices_pipeline,
            descriptor_sets,
            screen: None,
            selected_cluster_buffer,
            push_pipeline_layout,
            should_draw_pipeline,
            draw_indexed_indirect_buffer,
            indirect_compute_buffer,
            range_buffer,
            render_indices,
        }
    }
}

impl DrawPipeline for ExpandingComputeCulledIndices {
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
        core_draw: &ExpandingComputeCulledIndices,
        core: &Core,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> Self {
        let device = core.device.clone();
        let mut command_buffers = CommandBufferGroup::new(
            core.command_pool.clone(),
            screen.swapchain_framebuffers.len() as _,
        );

        for (frame, mut command_buffer) in command_buffers.iter_to_fill().enumerate() {
            let descriptor_sets_to_bind = [core_draw.descriptor_sets[frame].handle()];
            let selected_cluster_buffer_barrier = [
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.selected_cluster_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_STORAGE_READ
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    )
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.range_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_STORAGE_READ
                            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    )
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .buffer(core_draw.indirect_compute_buffer.handle())
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::DRAW_INDIRECT
                            | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    )
                    .size(vk::WHOLE_SIZE),
            ];

            let selected_cluster_buffer_dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(&selected_cluster_buffer_barrier);

            let draw_indirect_buffer_barrier = [vk::BufferMemoryBarrier2::default()
                .buffer(core_draw.draw_indexed_indirect_buffer.handle())
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::INDIRECT_COMMAND_READ,
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
                core_draw.range_buffer.get_buffer().fill(*command_buffer, 0);

                {
                    // Select clusters with ranges for each instance.
                    // Also outputs an indirect buffer for the number of clusters per instance

                    //TODO: can we only wipe the working buffer?
                    device.cmd_fill_buffer(
                        *command_buffer,
                        core_draw.draw_indexed_indirect_buffer.handle(),
                        0,
                        core_draw.draw_indexed_indirect_buffer.size(),
                        0,
                    );

                    device.cmd_bind_pipeline(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        *core_draw.should_draw_pipeline,
                    );

                    device.cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        core_draw.should_draw_pipeline.layout().handle(),
                        0,
                        &descriptor_sets_to_bind,
                        &[],
                    );

                    device.cmd_dispatch(
                        *command_buffer,
                        1,
                        core_draw.indirect_compute_buffer.len() as _,
                        1,
                    );

                    device.cmd_pipeline_barrier2(
                        *command_buffer,
                        &selected_cluster_buffer_dependency_info,
                    );
                    device.cmd_pipeline_barrier2(
                        *command_buffer,
                        &draw_indirect_buffer_dependency_info,
                    );
                }

                {
                    // Selected clusters + ranges + destinations index ranges -> triangles
                    core_draw
                        .render_indices
                        .bind_compact(*command_buffer, &core.device);

                    device.cmd_bind_descriptor_sets(
                        *command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        core_draw.push_pipeline_layout.handle(),
                        0,
                        &descriptor_sets_to_bind,
                        &[],
                    );

                    // Run all compaction at once
                    for instance in 0..core_draw.indirect_compute_buffer.len() {
                        core_draw.push_pipeline_layout.push_single_constant(
                            0,
                            *command_buffer,
                            instance as u32,
                        );

                        core_draw.render_indices.dispatch_compact(
                            *command_buffer,
                            &core.device,
                            instance,
                        );
                    }

                    device.cmd_pipeline_barrier2(
                        *command_buffer,
                        &draw_indirect_buffer_dependency_info,
                    );
                }
                core_draw.render_indices.render(
                    &mut command_buffer,
                    &core.device,
                    screen,
                    render_pass,
                    frame,
                );

                // Mark barrier between compute write and task read, and between indirect write and indirect read
                // device.cmd_pipeline_barrier2(*command_buffer, &compute_to_task_dependency_info);

                // device.cmd_bind_descriptor_sets(
                //     *command_buffer,
                //     vk::PipelineBindPoint::COMPUTE,
                //     core_draw.should_draw_pipeline.layout().handle(),
                //     0,
                //     &descriptor_sets_to_bind,
                //     &[],
                // );

                // // Fill indirect draw with 0s
                // device.cmd_fill_buffer(
                //     *command_buffer,
                //     core_draw.draw_indexed_indirect_buffer.handle(),
                //     0,
                //     core_draw.draw_indexed_indirect_buffer.size(),
                //     0,
                // );

                // Instance 0 is intermediate data / the entire scene

                // reset the count value to 1 (first index, as 0 reserved for counter)
                // device.cmd_fill_buffer(
                //     *command_buffer,
                //     core_draw.should_cull_buffer.handle(),
                //     0,
                //     4,
                //     2,
                // );
                // //reset tri-count to 0.
                // device.cmd_fill_buffer(
                //     *command_buffer,
                //     core_draw.should_cull_buffer.handle(),
                //     4,
                //     4,
                //     0,
                // );

                // All the shaders share this layout, so we only need to push once
                // core_draw
                //     .allocate_indices_pipeline
                //     .layout()
                //     .push_single_constant(0, *command_buffer, instance as u32);

                // {
                //     // Generate cluster selection
                //     device.cmd_bind_pipeline(
                //         *command_buffer,
                //         vk::PipelineBindPoint::COMPUTE,
                //         core_draw.should_draw_pipeline.handle(),
                //     );

                //     core_draw
                //         .indirect_compute_buffer
                //         .dispatch_indirect(*command_buffer, instance);

                //     // Force previous compute shader to be complete before this one
                //     device.cmd_pipeline_barrier2(
                //         *command_buffer,
                //         &should_cull_buffer_dependency_info,
                //     );
                // }

                // {
                //     // Initialise indirect draw arguments

                //     device.cmd_bind_pipeline(
                //         *command_buffer,
                //         vk::PipelineBindPoint::COMPUTE,
                //         core_draw.allocate_indices_pipeline.handle(),
                //     );

                //     device.cmd_push_constants(
                //         *command_buffer,
                //         core_draw.allocate_indices_pipeline.layout().handle(),
                //         vk::ShaderStageFlags::COMPUTE,
                //         0,
                //         bytemuck::cast_slice(&[instance as u32]),
                //     );

                //     device.cmd_dispatch(*command_buffer, 1, 1, 1);

                //     device.cmd_pipeline_barrier2(
                //         *command_buffer,
                //         &draw_indirect_buffer_dependency_info,
                //     );
                // }

                // {
                //     // Fill into indirect draw arguments
                //     core_draw.render_indices.compact_indices(
                //         *command_buffer,
                //         &core.device,
                //         instance,
                //     );
                // }
                // device.cmd_pipeline_barrier2(
                //     *command_buffer,
                //     &should_cull_buffer_dependency_info,
                // );
                // device.cmd_pipeline_barrier2(
                //     *command_buffer,
                //     &draw_indirect_buffer_dependency_info,
                // );

                // core_draw.render_indices.render(
                //     &mut command_buffer,
                //     &core.device,
                //     screen,
                //     render_pass,
                //     i,
                // );
            }
        }

        Self {
            command_buffers: Arc::new(command_buffers),
            device,
            command_pool: core.command_pool.clone(),
        }
    }
}
