use ash::vk;

use glam::Mat4;
use memoffset::offset_of;

pub struct DeviceExtension {}

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
#[derive(Clone, Debug, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniformBufferObject {
    pub model: glam::Mat4,
    pub inv_model: glam::Mat4,
}

#[repr(C)]
#[derive(Clone, Debug, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformBufferObject {
    pub view_proj: glam::Mat4,
    pub cam_pos: glam::Vec3,
    pub target_error: f32,
}
