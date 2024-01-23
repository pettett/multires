use std::{ffi::CString, ops::Deref, ptr, sync::Arc};

use ash::vk;

use crate::{core::Core, VkHandle};

use super::{descriptor_pool::DescriptorSetLayout, device::Device};

pub struct Pipeline<const T: bool> {
    device: Arc<Device>,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    pub ubo_layout: Arc<DescriptorSetLayout>,
}

pub type GraphicsPipeline = Pipeline<true>;
pub type ComputePipeline = Pipeline<false>;

impl<const T: bool> Pipeline<T> {
    pub fn new(
        device: Arc<Device>,
        handle: vk::Pipeline,
        layout: vk::PipelineLayout,
        ubo_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        Self {
            device,
            handle,
            layout,
            ubo_layout,
        }
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}
impl ComputePipeline {
    pub fn create_compute_pipeline(
        core: &Core,
        shader: &[u8],
        ubo_layout: Arc<DescriptorSetLayout>,
        name: &str,
    ) -> ComputePipeline {
        let comp_shader_module =
            ShaderModule::new(core.device.clone(), bytemuck::cast_slice(shader));

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
            core.device
                .handle
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        let pipeline_create_infos = [vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage)
            .layout(pipeline_layout)
            .flags(vk::PipelineCreateFlags::DISPATCH_BASE) // Allow non-0 bases, for applying this to instances
            .build()];

        let compute_pipelines = unsafe {
            core.device
                .handle
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .expect("Failed to create Compute Pipeline!.")
        };

        core.name_object(name, compute_pipelines[0]);

        ComputePipeline::new(
            core.device.clone(),
            compute_pipelines[0],
            pipeline_layout,
            ubo_layout,
        )
    }
}

impl<const T: bool> VkHandle for Pipeline<T> {
    type VkItem = vk::Pipeline;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl<const T: bool> Drop for Pipeline<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
            self.device
                .handle
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct ShaderModule {
    device: Arc<Device>,
    handle: vk::ShaderModule,
}

impl VkHandle for ShaderModule {
    type VkItem = vk::ShaderModule;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl ShaderModule {
    pub fn new(device: Arc<Device>, code: &[u32]) -> Self {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(code);

        ShaderModule {
            handle: unsafe {
                device
                    .handle
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create Shader Module!")
            },
            device,
        }
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_shader_module(self.handle, None);
        }
    }
}
