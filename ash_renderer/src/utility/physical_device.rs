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

#[derive(Default)]
pub struct PhysicalRelevantFeatureSupport {
    pub device: vk::PhysicalDeviceFeatures2,
    buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeatures,
    maintenance4: vk::PhysicalDeviceMaintenance4Features,
    mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT,
    shader_draw_param: vk::PhysicalDeviceShaderDrawParametersFeatures,
    sync2: vk::PhysicalDeviceSynchronization2Features,
}

pub struct DeviceFeatureSet {
    pub mesh_shader: bool,
    pub task_shader: bool,
    pub synchronization2: bool,
    pub shader_draw_parameters: bool,
    pub maintenance4: bool,
    pub buffer_device_address: bool,
}

impl PhysicalRelevantFeatureSupport {
    pub fn feature_set(&self) -> DeviceFeatureSet {
        DeviceFeatureSet {
            mesh_shader: self.mesh_shader.mesh_shader > 0,
            task_shader: self.mesh_shader.task_shader > 0,
            synchronization2: self.sync2.synchronization2 > 0,
            shader_draw_parameters: self.shader_draw_param.shader_draw_parameters > 0,
            maintenance4: self.maintenance4.maintenance4 > 0,
            buffer_device_address: self.buffer_device_address.buffer_device_address > 0,
        }
    }
}

impl Into<DeviceFeatureSet> for PhysicalRelevantFeatureSupport {
    fn into(self) -> DeviceFeatureSet {
        self.feature_set()
    }
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

    pub fn get_features(&self) -> Box<PhysicalRelevantFeatureSupport> {
        let mut relevant_features = Box::new(PhysicalRelevantFeatureSupport::default());

        // Get support info on all the features we want
        relevant_features.device = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut relevant_features.buffer_device_address)
            .push_next(&mut relevant_features.mesh_shader)
            .push_next(&mut relevant_features.maintenance4)
            .push_next(&mut relevant_features.sync2)
            .push_next(&mut relevant_features.shader_draw_param)
            .build();

        unsafe {
            self.instance
                .handle
                .get_physical_device_features2(self.handle, &mut relevant_features.device)
        };

        // Disable everything we don't need

        relevant_features
            .mesh_shader
            .primitive_fragment_shading_rate_mesh_shader = 0;

        relevant_features.mesh_shader.multiview_mesh_shader = 0;

        relevant_features
            .buffer_device_address
            .buffer_device_address_capture_replay = 0;
        relevant_features
            .buffer_device_address
            .buffer_device_address_multi_device = 0;

        relevant_features
    }
}
