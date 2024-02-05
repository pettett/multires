use std::{ptr, sync::Arc};

use ash::{extensions, vk};

use crate::VkHandle;

use super::{extensions::Extensions, instance::Instance, macros::vk_handle_wrapper};

pub struct PhysicalDevice {
    instance: Arc<Instance>,
    handle: vk::PhysicalDevice,
    pub extensions: Extensions,
}

vk_handle_wrapper!(PhysicalDevice);

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

#[derive(Debug)]
pub struct DeviceFeatureSet {
    pub mesh_shader: bool,
    pub task_shader: bool,
    pub mesh_queries: bool,
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
            mesh_queries: self.mesh_shader.mesh_shader_queries > 0,

            synchronization2: self.sync2.synchronization2 > 0,
            shader_draw_parameters: self.shader_draw_param.shader_draw_parameters > 0,
            maintenance4: self.maintenance4.maintenance4 > 0,
            buffer_device_address: self.buffer_device_address.buffer_device_address > 0,
        }
    }

    pub fn init() -> Box<Self> {
        let mut relevant_features = Box::<PhysicalRelevantFeatureSupport>::default();

        // Get support info on all the features we want
        relevant_features.device = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut relevant_features.buffer_device_address)
            .push_next(&mut relevant_features.mesh_shader)
            .push_next(&mut relevant_features.maintenance4)
            .push_next(&mut relevant_features.sync2)
            .push_next(&mut relevant_features.shader_draw_param)
            .build();
        relevant_features
    }
}

impl From<PhysicalRelevantFeatureSupport> for DeviceFeatureSet {
    fn from(val: PhysicalRelevantFeatureSupport) -> Self {
        val.feature_set()
    }
}

impl PhysicalDevice {
    pub fn new(
        handle: vk::PhysicalDevice,
        instance: Arc<Instance>,
        extensions: Extensions,
    ) -> Self {
        println!("Attaching to device with extensions: {extensions:?}");

        Self {
            handle,
            instance,
            extensions,
        }
    }
    pub fn get_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance
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
        let mut all_features = PhysicalRelevantFeatureSupport::init();

        unsafe {
            self.instance
                .get_physical_device_features2(self.handle, &mut all_features.device)
        };

        // Disable everything we don't need

        let mut relevant_features = PhysicalRelevantFeatureSupport::init();

        relevant_features.mesh_shader.mesh_shader = all_features.mesh_shader.mesh_shader;
        relevant_features.mesh_shader.task_shader = all_features.mesh_shader.task_shader;
        relevant_features.mesh_shader.mesh_shader_queries =
            all_features.mesh_shader.mesh_shader_queries;

        relevant_features.sync2.synchronization2 = all_features.sync2.synchronization2;
        relevant_features
            .buffer_device_address
            .buffer_device_address = all_features.buffer_device_address.buffer_device_address;
        relevant_features.maintenance4.maintenance4 = all_features.maintenance4.maintenance4;
        relevant_features.shader_draw_param.shader_draw_parameters =
            all_features.shader_draw_param.shader_draw_parameters;

        relevant_features.device.features.multi_draw_indirect =
            all_features.device.features.multi_draw_indirect;

        relevant_features.device.features.pipeline_statistics_query =
            all_features.device.features.pipeline_statistics_query;

        relevant_features
    }
}
