use std::{
    ffi::CString,
    ptr,
    sync::{Arc, Mutex},
};

use ash::vk::{self};
use bevy_ecs::world::World;
use common_renderer::components::transform::Transform;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    screen::Screen,
    utility::{
        buffer::{AsBuffer, Buffer, TypedBuffer},
        command_pool::CommandPool,
        descriptor_pool::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
        device::Device,
        pipeline::{Pipeline, ShaderModule},
        render_pass::RenderPass,
    },
    VkHandle, TASK_GROUP_SIZE,
};

use super::{
    init_color_blend_attachment_states, init_depth_state_create_info,
    init_multisample_state_create_info, init_rasterization_statue_create_info, DrawPipeline,
};

pub struct ComputeCulledMesh {
    graphics_pipeline: Pipeline,
    should_draw_pipeline: Pipeline,
    descriptor_sets: Vec<DescriptorSet>,
    screen: Option<ScreenData>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    should_cull_buffer: Arc<TypedBuffer<u32>>,
}

impl ComputeCulledMesh {
    pub fn new(
        core: Arc<Core>,
        screen: &Screen,
        world: &mut World,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: &RenderPass,
        graphics_queue: vk::Queue,
        descriptor_pool: Arc<DescriptorPool>,
        uniform_transform_buffer: Arc<Buffer>,
        uniform_camera_buffers: &[impl AsBuffer],
        vertex_buffer: Arc<Buffer>,
        meshlet_buffer: Arc<Buffer>,
        submesh_buffer: Arc<Buffer>,
        cluster_count: u32,
    ) -> Self {
        let ubo_layout = create_descriptor_set_layout(core.device.clone());

        let graphics_pipeline = create_graphics_pipeline(
            core.device.clone(),
            render_pass,
            screen.swapchain().extent,
            ubo_layout.clone(),
        );
        let should_draw_pipeline = create_compute_pipeline(core.device.clone(), ubo_layout.clone());

        let should_cull_buffer = TypedBuffer::new_filled(
            core.device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vec![1; cluster_count as _],
        );

        let descriptor_sets = DescriptorSet::create_compute_culled_descriptor_sets(
            &core.device,
            &descriptor_pool,
            &ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &should_cull_buffer,
            &vertex_buffer,
            &meshlet_buffer,
            &submesh_buffer,
            //&texture_image,
            screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            descriptor_sets,
            screen: None,
            core,
            descriptor_pool,
            should_cull_buffer,
            should_draw_pipeline,
        }
    }
}

impl DrawPipeline for ComputeCulledMesh {
    fn draw(&self, frame_index: usize) -> vk::CommandBuffer {
        self.screen.as_ref().unwrap().command_buffers[frame_index]
    }
    fn init_swapchain(
        &mut self,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
    ) {
        self.screen = Some(ScreenData::create_command_buffers(
            &self,
            core,
            screen,
            submesh_count,
            instance_count,
            render_pass,
        ));
    }
}

struct ScreenData {
    device: Arc<Device>,
    command_pool: Arc<CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl ScreenData {
    pub fn create_command_buffers(
        core_draw: &ComputeCulledMesh,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
    ) -> Self {
        let device = core.device.clone();
        let command_buffers = core
            .command_pool
            .allocate_group(screen.swapchain_framebuffers.len() as _);

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .handle
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    // clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass.handle())
                .framebuffer(screen.swapchain_framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                })
                .clear_values(&clear_values);

            let descriptor_sets_to_bind = [core_draw.descriptor_sets[i].handle()];

            unsafe {
                device.handle.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.handle(),
                );

                device.handle.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    core_draw.should_draw_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device
                    .handle
                    .cmd_dispatch(command_buffer, submesh_count, 1, 1);

                device.handle.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                device.handle.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: screen.swapchain().extent,
                    }],
                );
                device.handle.cmd_set_viewport(
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

                device.handle.cmd_bind_pipeline(
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
                device.handle.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    core_draw.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.fn_mesh_shader.cmd_draw_mesh_tasks(
                    command_buffer,
                    submesh_count,
                    instance_count,
                    1,
                );
                // device.cmd_draw_indexed(
                //     command_buffer,
                //     RECT_TEX_COORD_INDICES_DATA.len() as u32,
                //     1,
                //     0,
                //     0,
                //     0,
                // );

                device.handle.cmd_end_render_pass(command_buffer);

                device
                    .handle
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
    let ubo_layout_bindings = [
        // layout (binding = 0) readonly buffer ModelUniformBufferObject {
        // 	mat4 models[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout(binding = 1) buffer Clusters {
        // 	ClusterData clusters[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout(binding = 2) buffer ShouldDraw {
        // 	uint should_draw[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout (binding = 3) uniform CameraUniformBufferObject {
        // 	CameraUniformObject ubo;
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE)
            .build(),
        // layout (std430, binding = 4) buffer InputBufferI {
        // 	s_meshlet meshlets[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT)
            .build(),
        // layout (std430, binding = 5) buffer InputBufferV {
        // 	Vertex verts[];
        // };
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT)
            .build(),
    ];

    Arc::new(DescriptorSetLayout::new(device, &ubo_layout_bindings))
}

fn create_graphics_pipeline(
    device: Arc<Device>,
    render_pass: &RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: Arc<DescriptorSetLayout>,
) -> Pipeline {
    let task_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!(
            "../../shaders/spv/mesh_shader_compute_cull.task"
        )),
    );
    let mesh_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!(
            "../../shaders/spv/mesh_shader_compute_cull.mesh"
        )),
    );
    let frag_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!(
            "../../shaders/spv/mesh_shader_compute_cull.frag"
        )),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .module(task_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::TASK_EXT)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(mesh_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .module(frag_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    // let binding_description = VertexV3::get_binding_descriptions();
    // let attribute_description = VertexV3::get_attribute_descriptions();

    // let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
    //     s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    //     p_next: ptr::null(),
    //     flags: vk::PipelineVertexInputStateCreateFlags::empty(),
    //     vertex_attribute_description_count: attribute_description.len() as u32,
    //     p_vertex_attribute_descriptions: attribute_description.as_ptr(),
    //     vertex_binding_description_count: binding_description.len() as u32,
    //     p_vertex_binding_descriptions: binding_description.as_ptr(),
    // };
    // let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
    //     s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
    //     flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
    //     p_next: ptr::null(),
    //     primitive_restart_enable: vk::FALSE,
    //     topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    // };

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
        device
            .handle
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
        device
            .handle
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!.")
    };

    Pipeline::new(
        device,
        graphics_pipelines[0],
        pipeline_layout,
        ubo_set_layout,
    )
}

fn create_compute_pipeline(device: Arc<Device>, ubo_layout: Arc<DescriptorSetLayout>) -> Pipeline {
    let comp_shader_module = ShaderModule::new(
        device.clone(),
        bytemuck::cast_slice(include_bytes!("../../shaders/spv/should_draw.comp")),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
        .module(comp_shader_module.handle())
        .name(&main_function_name)
        .stage(vk::ShaderStageFlags::COMPUTE)
        .build();

    let set_layouts = [ubo_layout.handle()];

    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

    let pipeline_layout = unsafe {
        device
            .handle
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Failed to create pipeline layout!")
    };

    let graphic_pipeline_create_infos = [vk::ComputePipelineCreateInfo::builder()
        .stage(shader_stage)
        .layout(pipeline_layout)
        .build()];

    let compute_pipelines = unsafe {
        device
            .handle
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &graphic_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Compute Pipeline!.")
    };

    Pipeline::new(device, compute_pipelines[0], pipeline_layout, ubo_layout)
}

impl DescriptorSet {
    fn create_compute_culled_descriptor_sets(
        device: &Arc<Device>,
        descriptor_pool: &Arc<DescriptorPool>,
        descriptor_set_layout: &Arc<DescriptorSetLayout>,
        uniform_transform_buffer: &Arc<Buffer>,
        uniform_camera_buffers: &[impl AsBuffer],
        should_draw_buffer: &Arc<impl AsBuffer>,
        vertex_buffer: &Arc<Buffer>,
        meshlet_buffer: &Arc<Buffer>,
        submesh_buffer: &Arc<Buffer>,
        //texture: &Image,
        swapchain_images_size: usize,
    ) -> Vec<DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout.handle());
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool.handle)
            .set_layouts(&layouts);

        let vk_descriptor_sets = unsafe {
            device
                .handle
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        let descriptor_sets: Vec<_> = vk_descriptor_sets
            .into_iter()
            .map(|set| {
                DescriptorSet::new(
                    set,
                    descriptor_pool.clone(),
                    device.clone(),
                    vec![
                        vertex_buffer.clone(),
                        uniform_transform_buffer.clone(),
                        meshlet_buffer.clone(),
                        submesh_buffer.clone(),
                    ],
                )
            })
            .collect();

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_transform_buffer_infos =
                [uniform_transform_buffer.full_range_descriptor()];
            let descriptor_camera_buffer_infos =
                [uniform_camera_buffers[i].full_range_descriptor()];

            let should_draw_buffer_infos = [should_draw_buffer.full_range_descriptor()];
            let vertex_buffer_infos = [vertex_buffer.full_range_descriptor()];
            let meshlet_buffer_infos = [meshlet_buffer.full_range_descriptor()];
            let cluster_buffer_infos = [submesh_buffer.full_range_descriptor()];

            // let descriptor_image_infos = [vk::DescriptorImageInfo {
            //     sampler: texture.sampler(),
            //     image_view: texture.image_view(),
            //     image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            // }];

            let descriptor_write_sets = [
                // layout (binding = 0) readonly buffer ModelUniformBufferObject {
                // 	mat4 models[];
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: descriptor_transform_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // layout(binding = 1) buffer Clusters {
                // 	ClusterData clusters[];
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: cluster_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // layout(binding = 2) buffer ShouldDraw {
                // 	uint should_draw[];
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: should_draw_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // layout (binding = 3) uniform CameraUniformBufferObject {
                // 	CameraUniformObject ubo;
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,

                    p_buffer_info: descriptor_camera_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // layout (std430, binding = 4) buffer InputBufferI {
                // 	s_meshlet meshlets[];
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 4,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: meshlet_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // layout (std430, binding = 5) buffer InputBufferV {
                // 	Vertex verts[];
                // };
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set.handle(),
                    dst_binding: 5,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: vertex_buffer_infos.as_ptr(),
                    ..Default::default()
                },
            ];

            unsafe {
                device
                    .handle
                    .update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }
}
