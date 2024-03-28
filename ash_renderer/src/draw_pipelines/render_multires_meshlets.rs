// Draw meshlets from a should draw buffer

use std::{ffi, sync::Arc};

use ash::vk;

use crate::{
    app::{mesh_data::MeshData, renderer::Renderer, scene::Scene},
    core::Core,
    screen::Screen,
    utility::{
        buffer::{AsBuffer, TBuffer},
        device::Device,
        pooled::{
            descriptor_pool::{
                DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
                DescriptorWriteData,
            },
            query_pool::QueryPool,
        },
        render_pass::RenderPass,
        GraphicsPipeline, ShaderModule,
    },
    CLEAR_VALUES,
};

use super::{
    indirect_tasks::MeshInvocationsQueryResults, init_color_blend_attachment_states,
    init_depth_state_create_info, init_multisample_state_create_info,
    init_rasterization_statue_create_info, render_multires::RenderMultires, BufferRange,
};
use crate::VkHandle;

pub struct RenderMultiresMeshlets {
    graphics_pipeline: GraphicsPipeline,
    indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
    range_buffer: Arc<TBuffer<BufferRange>>,
    descriptor_sets: Vec<DescriptorSet>,
    query_pool: Arc<QueryPool<MeshInvocationsQueryResults>>,
    query: bool,
}

impl RenderMultiresMeshlets {
    pub fn new(
        core: &Core,
        renderer: &Renderer,
        screen: &Screen,
        scene: &Scene,
        mesh_data: &MeshData,
        indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
        range_buffer: Arc<TBuffer<BufferRange>>,
        cluster_draw_buffer: Arc<TBuffer<u32>>,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            &renderer.render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let descriptor_sets = create_compute_culled_meshes_descriptor_sets(
            &core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &cluster_draw_buffer,
            &range_buffer,
            &scene,
            &mesh_data,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        let query_pool = QueryPool::new(core.device.clone(), 1);

        Self {
            graphics_pipeline,
            indirect_task_buffer,
            range_buffer,
            descriptor_sets,
            query_pool,
            query: renderer.query,
        }
    }

    pub fn query_pool(&self) -> &QueryPool<MeshInvocationsQueryResults> {
        &self.query_pool
    }
}

impl RenderMultires for RenderMultiresMeshlets {
    fn render(
        &self,
        cmd: ash::vk::CommandBuffer,
        device: &crate::utility::device::Device,
        screen: &crate::screen::Screen,
        render_pass: &crate::utility::render_pass::RenderPass,
        frame: usize,
    ) {
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.handle())
            .framebuffer(screen.swapchain_framebuffers[frame])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: screen.swapchain().extent,
            })
            .clear_values(&CLEAR_VALUES);

        let descriptor_sets_to_bind = [*self.descriptor_sets[frame]];

        unsafe {
            let query = self.query && frame == 0;
            if query {
                self.query_pool.reset(cmd, frame as _);
            }

            device.cmd_begin_render_pass(cmd, &render_pass_begin_info, vk::SubpassContents::INLINE);

            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                }],
            );
            device.cmd_set_viewport(
                cmd,
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
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.handle(),
            );
            {
                let _qry = if query {
                    Some(self.query_pool.begin_query(cmd, frame as _))
                } else {
                    None
                };

                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                self.indirect_task_buffer.draw_tasks_indirect(cmd);
            }

            device.cmd_end_render_pass(cmd);
        }
    }
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> GraphicsPipeline {
    let task_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!(
            "../../shaders/spv/mesh_shader_compute_cull.task"
        )),
    );
    let mesh_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/mesh-shader.mesh")),
    );
    let frag_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/frag_colour.frag")),
    );

    let main_function_name = ffi::CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(task_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::TASK_EXT),
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(mesh_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::MESH_EXT),
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(frag_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::FRAGMENT),
    ];

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
        .viewports(&viewports);

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

    let graphics_pipelines = unsafe {
        device
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
    let bindings = vec![
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Uniform {
            vis: vk::ShaderStageFlags::MESH_EXT
                | vk::ShaderStageFlags::TASK_EXT
                | vk::ShaderStageFlags::COMPUTE,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::MESH_EXT,
        },
        DescriptorSetLayoutBinding::None,
        DescriptorSetLayoutBinding::Storage {
            vis: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::TASK_EXT,
        },
    ];

    Arc::new(DescriptorSetLayout::new(device, bindings))
}

fn create_compute_culled_meshes_descriptor_sets(
    device: &Arc<Device>,
    descriptor_pool: &Arc<DescriptorPool>,
    descriptor_set_layout: &Arc<DescriptorSetLayout>,
    cluster_draw_buffer: &Arc<impl AsBuffer>,
    range_buffer: &Arc<impl AsBuffer>,

    scene: &Scene,
    mesh_data: &MeshData,
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
                        buf: scene.uniform_transform_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: cluster_draw_buffer.buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: mesh_data.cluster_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: scene.uniform_camera_buffers[i].buffer(),
                    },
                    DescriptorWriteData::Buffer {
                        buf: mesh_data.meshlet_buffer.buffer(), //
                    },
                    DescriptorWriteData::Buffer {
                        buf: mesh_data.vertex_buffer.buffer(), //
                    },
                    DescriptorWriteData::Empty,
                    DescriptorWriteData::Buffer {
                        buf: range_buffer.buffer(), //
                    },
                ],
            )
        })
        .collect();

    descriptor_sets
}
