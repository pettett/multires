use std::{ffi::CString, mem, sync::Arc};

use ash::vk::{self, PushConstantRange};

use crate::{core::Core, VkHandle};

use super::{
    device::Device,
    macros::{vk_device_drop, vk_device_owned_wrapper, vk_handle_wrapper, vk_handle_wrapper_const},
    pooled::descriptor_pool::DescriptorSetLayout,
};

pub struct PipelineLayout {
    handle: vk::PipelineLayout,
    device: Arc<Device>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

vk_handle_wrapper!(PipelineLayout);
vk_device_drop!(PipelineLayout, destroy_pipeline_layout);

impl PipelineLayout {
    /// Create a new pipeline layout with no push constants
    pub fn new(device: Arc<Device>, descriptor_set_layout: Arc<DescriptorSetLayout>) -> Arc<Self> {
        Self::new_push_constants(device, descriptor_set_layout, &[])
    }

    /// Create a pipeline layout with a single push constant with layout P
    pub fn new_single_push<P>(
        device: Arc<Device>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
        stage: vk::ShaderStageFlags,
    ) -> Arc<Self> {
        Self::new_push_constants(
            device,
            descriptor_set_layout,
            &[PushConstantRange::default()
                .offset(0)
                .stage_flags(stage)
                .size(mem::size_of::<P>() as u32)],
        )
    }

    /// Create a new pipeline layout with push constants from push_constant_ranges
    pub fn new_push_constants(
        device: Arc<Device>,
        descriptor_set_layout: Arc<DescriptorSetLayout>,
        push_constant_ranges: &[PushConstantRange],
    ) -> Arc<Self> {
        let set_layouts = [descriptor_set_layout.handle()];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let handle = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };
        Arc::new(Self {
            handle,
            device,
            descriptor_set_layout,
            push_constant_ranges: push_constant_ranges.to_vec(),
        })
    }

    pub fn push_single_constant<P: bytemuck::Pod>(
        &self,
        range_index: usize,
        cmd: vk::CommandBuffer,
        constants: P,
    ) {
        // safety - data must be the right size
        assert_eq!(
            self.push_constant_ranges[range_index].size as usize,
            mem::size_of::<P>()
        );
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.handle(),
                self.push_constant_ranges[range_index].stage_flags,
                self.push_constant_ranges[range_index].offset,
                bytemuck::cast_slice(&[constants]),
            )
        }
    }
}

pub struct Pipeline<const T: bool> {
    device: Arc<Device>,
    handle: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

pub type GraphicsPipeline = Pipeline<true>;
pub type ComputePipeline = Pipeline<false>;

vk_handle_wrapper_const!(Pipeline, bool);

impl<const T: bool> Pipeline<T> {
    pub fn new_raw(device: Arc<Device>, handle: vk::Pipeline, layout: Arc<PipelineLayout>) -> Self {
        Self {
            device,
            handle,
            layout,
        }
    }

    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
}

impl GraphicsPipeline {
    pub fn new(
        device: Arc<Device>,
        pipeline_create_info: vk::GraphicsPipelineCreateInfo<'_>,
        layout: Arc<PipelineLayout>,
    ) -> Self {
        let handle = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        }
        .expect("Failed to create graphics pipeline")[0];

        Self {
            device,
            handle,
            layout,
        }
    }
}

impl ComputePipeline {
    pub fn create_compute_with_layout(
        core: &Core,
        shader: &[u8],
        pipeline_layout: Arc<PipelineLayout>,
        name: &str,
    ) -> ComputePipeline {
        let comp_shader_module =
            ShaderModule::new(core.device.clone(), bytemuck::cast_slice(shader));

        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .module(comp_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let pipeline_create_infos = [
            vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage)
                .layout(pipeline_layout.handle())
                .flags(vk::PipelineCreateFlags::DISPATCH_BASE), // Allow non-0 bases, for applying this to instances
        ];

        let compute_pipelines = unsafe {
            core.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .expect("Failed to create Compute Pipeline!.")
        };

        core.name_object(name, compute_pipelines[0]);

        ComputePipeline::new_raw(core.device.clone(), compute_pipelines[0], pipeline_layout)
    }

    pub fn create_compute_pipeline(
        core: &Core,
        shader: &[u8],
        ubo_layout: Arc<DescriptorSetLayout>,
        name: &str,
    ) -> ComputePipeline {
        let comp_shader_module =
            ShaderModule::new(core.device.clone(), bytemuck::cast_slice(shader));

        let main_function_name = CString::new("main").unwrap(); // the beginning function name in shader code.

        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .module(comp_shader_module.handle())
            .name(&main_function_name)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let pipeline_layout = PipelineLayout::new(core.device.clone(), ubo_layout);

        let pipeline_create_infos = [
            vk::ComputePipelineCreateInfo::default()
                .stage(shader_stage)
                .layout(pipeline_layout.handle())
                .flags(vk::PipelineCreateFlags::DISPATCH_BASE), // Allow non-0 bases, for applying this to instances
        ];

        let compute_pipelines = unsafe {
            core.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .expect("Failed to create Compute Pipeline!.")
        };

        core.name_object(name, compute_pipelines[0]);

        ComputePipeline::new_raw(core.device.clone(), compute_pipelines[0], pipeline_layout)
    }
}

impl<const T: bool> Drop for Pipeline<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

vk_device_owned_wrapper!(ShaderModule, destroy_shader_module);

impl ShaderModule {
    pub fn new(device: Arc<Device>, code: &[u8]) -> Self {
        assert_eq!(code.len() % 4, 0);

        // Make an uninitialised vector the correct size for our program
        let mut new_code = Vec::<u32>::with_capacity(code.len() / 4);
        unsafe { new_code.set_len(new_code.capacity()) }

        // Cast our vec to u8s, and copy code into it
        // This ensures vectors are aligned to u32, but is fast
        bytemuck::cast_slice_mut(&mut new_code[..]).copy_from_slice(code);

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&new_code);

        let handle = unsafe {
            device
                .create_shader_module(&shader_module_create_info, None)
                .expect("Failed to create Shader Module!")
        };

        ShaderModule { handle, device }
    }
}
