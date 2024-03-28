use std::{ffi::CString, sync::Arc};

use ash::vk;

use crate::{core::Core, VkHandle};

use super::{
    device::Device,
    macros::{vk_device_owned_wrapper, vk_handle_wrapper_const},
    pooled::descriptor_pool::DescriptorSetLayout,
};

pub struct Pipeline<const T: bool> {
    device: Arc<Device>,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    pub ubo_layout: Arc<DescriptorSetLayout>,
}

pub type GraphicsPipeline = Pipeline<true>;
pub type ComputePipeline = Pipeline<false>;

vk_handle_wrapper_const!(Pipeline, bool);

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

impl<const T: bool> Drop for Pipeline<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.handle, None);
            self.device.destroy_pipeline_layout(self.layout, None);
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


		let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&new_code);

        ShaderModule {
            handle: unsafe {
                device
                    .create_shader_module(&shader_module_create_info, None)
                    .expect("Failed to create Shader Module!")
            },
            device,
        }
    }
}
