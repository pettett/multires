use std::{ffi::CString, ptr, sync::Arc};

use ash::vk::{self, PhysicalDeviceMaintenance4Features, PhysicalDeviceMeshShaderFeaturesEXT};
use winapi::ctypes::c_char;

use crate::utility::share::find_queue_family;

use super::structures::{DeviceExtension, QueueFamilyIndices, SurfaceStuff};

pub struct Device {
    pub device: ash::Device,
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
impl Device {
    pub fn wait_device_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    pub fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        validation: &super::debug::ValidationInfo,
        device_extensions: &DeviceExtension,
        surface_stuff: &SurfaceStuff,
    ) -> (Arc<Self>, QueueFamilyIndices) {
        let indices = find_queue_family(instance, physical_device, surface_stuff);

        use std::collections::HashSet;
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

        let queue_priorities = [1.0_f32];
        let mut queue_create_infos = vec![];
        for &queue_family in unique_queue_families.iter() {
            let queue_create_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                queue_family_index: queue_family,
                p_queue_priorities: queue_priorities.as_ptr(),
                queue_count: queue_priorities.len() as u32,
            };
            queue_create_infos.push(queue_create_info);
        }

        let physical_device_features = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE, // enable anisotropy device feature from Chapter-24.
            ..Default::default()
        };

        let requred_validation_layer_raw_names: Vec<CString> = validation
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let enable_layer_names: Vec<*const c_char> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let enable_extension_names = device_extensions.get_extensions_raw_names();

        let mesh_shader = Box::new(
            PhysicalDeviceMeshShaderFeaturesEXT::builder()
                .mesh_shader(true)
                .build(),
        );

        let mut man4 = Box::new(
            PhysicalDeviceMaintenance4Features::builder()
                .maintenance4(true)
                .build(),
        );

        man4.p_next = Box::into_raw(mesh_shader).cast();

        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_next: Box::into_raw(man4).cast(),
            flags: vk::DeviceCreateFlags::empty(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),

            enabled_extension_count: enable_extension_names.len() as u32,
            pp_enabled_extension_names: enable_extension_names.as_ptr(),
            p_enabled_features: &physical_device_features,
            ..Default::default()
        };

        let device: ash::Device = unsafe {
            instance
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical Device!")
        };

        (Arc::new(Self { device }), indices)
    }
}