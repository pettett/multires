use std::{ffi, sync::Arc};

use ash::vk;

use crate::VkHandle;

use super::{
    instance::Instance,
    physical_device::{DeviceFeatureSet, PhysicalDevice},
    queue_family_indices::QueueFamilyIndices,
    surface::Surface,
};

pub struct Device {
    instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,

    pub features: DeviceFeatureSet,
    pub fn_mesh_shader: ash::extensions::ext::MeshShader,
    pub fn_swapchain: ash::extensions::khr::Swapchain,
    handle: ash::Device,
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_device(None);
        }
    }
}

impl Device {
    pub fn wait_device_idle(&self) {
        unsafe {
            self.device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    pub fn create_logical_device(
        instance: Arc<Instance>,
        physical_device: Arc<PhysicalDevice>,
        validation: &super::debug::ValidationInfo,
        surface: &Surface,
    ) -> (Arc<Self>, QueueFamilyIndices) {
        let indices = instance.find_queue_family(physical_device.handle(), surface);

        use std::collections::HashSet;
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        for &queue_family in unique_queue_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo {
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: queue_family,
                p_queue_priorities: queue_priorities.as_ptr(),
                queue_count: queue_priorities.len() as u32,
                ..Default::default()
            };
            queue_create_infos.push(queue_create_info);
        }

        let requred_validation_layer_raw_names: Vec<ffi::CString> = validation
            .required_validation_layers
            .iter()
            .map(|layer_name| ffi::CString::new(*layer_name).unwrap())
            .collect();
        let _enable_layer_names: Vec<*const ffi::c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        // Just go ahead and enable everything we have
        let mut physical_device_features = physical_device.get_features();
        // FIXME: Do these on a case by case basis for each pipeline

        let feature_set = physical_device_features.feature_set();

        assert!(feature_set.maintenance4);
        assert!(feature_set.synchronization2);
        assert!(feature_set.buffer_device_address);

        if feature_set.mesh_shader {
            println!("Mesh shaders are supported!");
            assert!(feature_set.mesh_queries);
            assert!(feature_set.task_shader);
        }

        println!("{:?}", feature_set);

        let enable_extension_names = physical_device.extensions.get_extensions_raw_names();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .push_next(&mut physical_device_features.device) // The rest will already have been pushed on
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enable_extension_names);

        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device.handle(), &device_create_info, None)
                .expect("Failed to create logical Device!")
        };

        let fn_mesh_shader = ash::extensions::ext::MeshShader::new(&instance, &device);
        let fn_swapchain = ash::extensions::khr::Swapchain::new(&instance, &device);

        (
            Arc::new(Self {
                instance,
                physical_device,
                fn_mesh_shader,
                fn_swapchain,
                features: feature_set,
                handle: device,
            }),
            indices,
        )
    }
}
