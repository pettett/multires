use std::{ptr, sync::Arc};

use ash::vk;

use crate::VkHandle;

use super::instance::Instance;

pub struct PhysicalDevice {
    handle: vk::PhysicalDevice,
    instance: Arc<Instance>,
}

impl VkHandle for PhysicalDevice {
    type VkItem = vk::PhysicalDevice;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Hash)]
pub struct PhysicalDeviceSubgroupProperties {
    pub subgroup_size: u32,
    pub supported_stages: vk::ShaderStageFlags,
    pub supported_operations: vk::SubgroupFeatureFlags,
    pub quad_operations_in_all_stages: vk::Bool32,
}

impl PhysicalDevice {
    pub fn new(handle: vk::PhysicalDevice, instance: Arc<Instance>) -> Self {
        Self { handle, instance }
    }
    pub fn get_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
                .handle
                .get_physical_device_memory_properties(self.handle)
        }
    }

    pub fn get_subgroup_properties(&self) -> PhysicalDeviceSubgroupProperties {
        unsafe {
            let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties {
                s_type: vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                p_next: ptr::null_mut(),
                ..Default::default()
            };

            let mut physical_device_properties = vk::PhysicalDeviceProperties2 {
                s_type: vk::StructureType::PHYSICAL_DEVICE_PROPERTIES_2,
                p_next: &mut subgroup_properties as *mut _ as *mut std::ffi::c_void,
                ..Default::default()
            };

            self.instance
                .handle
                .get_physical_device_properties2(self.handle, &mut physical_device_properties);

            PhysicalDeviceSubgroupProperties {
                subgroup_size: subgroup_properties.subgroup_size,
                supported_stages: subgroup_properties.supported_stages,
                supported_operations: subgroup_properties.supported_operations,
                quad_operations_in_all_stages: subgroup_properties.quad_operations_in_all_stages,
            }
        }
    }

    pub fn get_features(&self) -> vk::PhysicalDeviceFeatures2 {
        let mut features = vk::PhysicalDeviceFeatures2::default();
        unsafe {
            self.instance
                .handle
                .get_physical_device_features2(self.handle, &mut features)
        };

        features
    }
}
