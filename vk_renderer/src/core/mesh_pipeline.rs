use std::{ffi::CString, mem, ptr, sync::Arc};

use ash::{vk, RawPtr};
use vulkano::VulkanObject;

pub struct MeshPipeline {
    device: Arc<vulkano::device::Device>,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl MeshPipeline {
    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}
impl Drop for MeshPipeline {
    fn drop(&mut self) {
        unsafe {
            (self.device.fns().v1_0.destroy_pipeline)(
                self.device.handle(),
                self.pipeline,
                None.as_raw_ptr(),
            );
            (self.device.fns().v1_0.destroy_pipeline_layout)(
                self.device.handle(),
                self.layout,
                None.as_raw_ptr(),
            );
        }
    }
}
fn create_descriptor_set_layout(device: &vulkano::device::Device) -> vk::DescriptorSetLayout {
    let ubo_layout_bindings = [
        vk::DescriptorSetLayoutBinding {
            // transform uniform
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: None.as_raw_ptr(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: None.as_raw_ptr(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: None.as_raw_ptr(),
        },
        vk::DescriptorSetLayoutBinding {
            // sampler uniform
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: None.as_raw_ptr(),
        },
    ];

    let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: None.as_raw_ptr(),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: ubo_layout_bindings.len() as u32,
        p_bindings: ubo_layout_bindings.as_ptr(),
    };

    unsafe {
        let mut layout = mem::zeroed();
        (device.fns().v1_0.create_descriptor_set_layout)(
            device.handle(),
            &ubo_layout_create_info,
            None.as_raw_ptr(),
            &mut layout,
        )
        .result_with_success(layout)
        .expect("Failed to create Descriptor Set Layout!")
    }
}

fn create_shader_module(device: &vulkano::device::Device, code: Vec<u8>) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: None.as_raw_ptr(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
    };

    unsafe {
        let mut shader = mem::zeroed();
        (device.fns().v1_0.create_shader_module)(
            device.handle(),
            &shader_module_create_info,
            ptr::null(),
            &mut shader,
        )
        .result_with_success(shader)
        .expect("Failed to create Shader Module!")
    }
}

pub fn create_graphics_pipeline(
    device: Arc<vulkano::device::Device>,
    render_pass: &vulkano::render_pass::RenderPass,
    swapchain_extent: [u32; 2],
) -> Arc<MeshPipeline> {
    let swapchain_extent = vk::Extent2D {
        width: swapchain_extent[0],
        height: swapchain_extent[1],
    };
    let ubo_set_layout: vk::DescriptorSetLayout = create_descriptor_set_layout(&device);

    let mesh_shader_module = create_shader_module(
        &device,
        include_bytes!("../../../ash_renderer/shaders/spv/mesh-shader.mesh").to_vec(),
    );
    let frag_shader_module = create_shader_module(
        &device,
        include_bytes!("../../../ash_renderer/shaders/spv/mesh-shader.frag").to_vec(),
    );

    let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            // Vertex Shader
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: mesh_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::MESH_EXT,
        },
        vk::PipelineShaderStageCreateInfo {
            // Fragment Shader
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: frag_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::FRAGMENT,
        },
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

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
    };

    let rasterization_statue_create_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_clamp: 0.0,
        depth_bias_constant_factor: 0.0,
        depth_bias_enable: vk::FALSE,
        depth_bias_slope_factor: 0.0,
    };

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        p_next: ptr::null(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };

    let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: vk::TRUE,
        depth_write_enable: vk::TRUE,
        depth_compare_op: vk::CompareOp::LESS,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
    };

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
    }];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_blend_attachment_states.len() as u32,
        p_attachments: color_blend_attachment_states.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };

    let set_layouts = [ubo_set_layout];

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };

    let pipeline_layout = unsafe {
        let mut pipeline_layout = mem::zeroed();
        (device.fns().v1_0.create_pipeline_layout)(
            device.handle(),
            &pipeline_layout_create_info,
            ptr::null(),
            &mut pipeline_layout,
        )
        .result_with_success(pipeline_layout)
        .expect("Failed to create Pipeline Layout!.")
    };

    let graphic_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: ptr::null(),
        p_input_assembly_state: ptr::null(),
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state_create_info,
        p_rasterization_state: &rasterization_statue_create_info,
        p_multisample_state: &multisample_state_create_info,
        p_depth_stencil_state: &depth_state_create_info,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass: render_pass.handle(),
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
    }];

    let graphics_pipelines = unsafe {
        let mut pipelines = Vec::with_capacity(graphic_pipeline_create_infos.len());
        let err_code = (device.fns().v1_0.create_graphics_pipelines)(
            device.handle(),
            vk::PipelineCache::null(),
            graphic_pipeline_create_infos.len() as u32,
            graphic_pipeline_create_infos.as_ptr(),
            ptr::null(),
            pipelines.as_mut_ptr(),
        );
        pipelines.set_len(graphic_pipeline_create_infos.len());
        match err_code {
            vk::Result::SUCCESS => Ok(pipelines),
            _ => Err((pipelines, err_code)),
        }
        .expect("Failed to create Graphics Pipeline!.")
    };

    unsafe {
        (device.fns().v1_0.destroy_shader_module)(device.handle(), mesh_shader_module, ptr::null());
        (device.fns().v1_0.destroy_shader_module)(device.handle(), frag_shader_module, ptr::null());
    }

    Arc::new(MeshPipeline {
        device,
        pipeline: graphics_pipelines[0],
        layout: pipeline_layout,
    })
}