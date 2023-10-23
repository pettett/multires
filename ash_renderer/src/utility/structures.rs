use ash::vk;

use glam::Mat4;
use memoffset::offset_of;

pub struct DeviceExtension {
    pub names: [&'static str; 2],
    //    pub raw_names: [*const i8; 1],
}

pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view_proj: Mat4,
}
