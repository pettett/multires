// Draw meshlets from a should draw buffer

use std::{ffi, sync::Arc};

use ash::vk;

use crate::{
    app::{
        material::{Material, MAIN_FUNCTION_NAME},
        mesh_data::MeshData,
        renderer::Renderer,
        scene::Scene,
    },
    core::Core,
    utility::{
        buffer::{AsBuffer, TBuffer},
        device::Device,
        pooled::{
            command_buffer_writer::CommandBufferWriter,
            descriptor_pool::{
                DescriptorPool, DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding,
                DescriptorWriteData,
            },
            query_pool::{MeshInvocationsQueryResults, QueryPool, TypelessQueryPool},
        },
        render_pass::RenderPass,
        screen::Screen,
        GraphicsPipeline, PipelineLayout, ShaderModule,
    },
    CLEAR_VALUES,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, render_multires::RenderMultires, BufferRange,
};
use crate::VkHandle;

pub struct RenderMultiresMeshlets {
    graphics_pipeline: GraphicsPipeline,
    indirect_task_buffer: Arc<TBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,
    range_buffer: Arc<TBuffer<BufferRange>>,
    descriptor_sets: Vec<DescriptorSet>,
    query_pool: Option<Arc<TypelessQueryPool>>,
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
        query_pool: Option<Arc<TypelessQueryPool>>,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core);

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            &renderer.render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
            renderer.fragment(),
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

        Self {
            graphics_pipeline,
            indirect_task_buffer,
            range_buffer,
            descriptor_sets,
            query_pool,
        }
    }

    pub fn query_pool(&self) -> Option<&Arc<TypelessQueryPool>> {
        self.query_pool.as_ref()
    }
}

impl RenderMultires for RenderMultiresMeshlets {
    fn render(
        &self,
        cmd: &mut CommandBufferWriter,
        device: &crate::utility::device::Device,
        screen: &Screen,
        render_pass: &crate::utility::render_pass::RenderPass,
        frame: usize,
    ) {
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(screen.swapchain_framebuffers[frame].handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: screen.swapchain().extent,
            })
            .clear_values(&CLEAR_VALUES);

        let descriptor_sets_to_bind = [*self.descriptor_sets[frame]];

        unsafe {
            let query = frame == 0;

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
            {
                let _qry = if query {
                    self.query_pool
                        .as_ref()
                        .map(|pool| pool.begin_query(**cmd, frame as _))
                } else {
                    None
                };

                device.cmd_bind_descriptor_sets(
                    **cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline.layout().handle(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                self.indirect_task_buffer.draw_tasks_indirect(**cmd);
            }

            device.cmd_end_render_pass(**cmd);
        }
    }
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
    frag_shader_module: &Material,
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

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .module(task_shader_module.handle())
            .name(&MAIN_FUNCTION_NAME)
            .stage(vk::ShaderStageFlags::TASK_EXT),
        vk::PipelineShaderStageCreateInfo::default()
            .module(mesh_shader_module.handle())
            .name(&MAIN_FUNCTION_NAME)
            .stage(vk::ShaderStageFlags::MESH_EXT),
        frag_shader_module.shader_stage_create_info(),
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

    let pipeline_layout = PipelineLayout::new(device.clone(), ubo_set_layout);

    let graphic_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .rasterization_state(&rasterization_statue_create_info)
        .viewport_state(&viewport_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .depth_stencil_state(&depth_state_create_info)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state_info)
        .layout(pipeline_layout.handle())
        .render_pass(render_pass.handle());

    GraphicsPipeline::new(device, graphic_pipeline_create_info, pipeline_layout)
}

fn create_descriptor_set_layout(core: &Core) -> Arc<DescriptorSetLayout> {
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

    Arc::new(DescriptorSetLayout::new(core, bindings, "meshlets layout"))
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
    descriptor_pool.alloc(descriptor_set_layout, swapchain_images_size, |i| {
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
        ]
    })
}
