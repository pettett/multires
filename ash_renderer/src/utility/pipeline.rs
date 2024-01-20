use std::{ffi::CString, ops::Deref, ptr, sync::Arc};

use ash::vk;

use crate::VkHandle;

use super::{descriptor_pool::DescriptorSetLayout, device::Device};

pub struct Pipeline {
    device: Arc<Device>,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    pub ubo_layout: Arc<DescriptorSetLayout>,
}

impl Pipeline {
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

impl VkHandle for Pipeline {
    type VkItem = vk::Pipeline;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl Drop for Pipeline {
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
