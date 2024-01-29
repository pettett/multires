use crate::utility::debug::ValidationInfo;
use crate::utility::structures::DeviceExtension;
use ash::vk::make_api_version;

use std::os::raw::c_char;

pub const APPLICATION_VERSION: u32 = make_api_version(0, 1, 0, 0);
pub const ENGINE_VERSION: u32 = make_api_version(0, 1, 0, 0);
pub const API_VERSION: u32 = make_api_version(0, 1, 3, 251);

pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;
pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {};

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

impl DeviceExtension {
    pub fn get_extensions_raw_names(&self) -> [*const c_char; 2] {
        [
            // ash::extensions::ext::MeshShader::name().as_ptr(),
            ash::extensions::khr::Swapchain::name().as_ptr(),
            ash::extensions::khr::BufferDeviceAddress::name().as_ptr(),
        ]
    }
}
