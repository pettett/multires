use std::sync::Arc;

use ash::vk;
use raw_window_handle::{
    HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle, HasWindowHandle,
};

use crate::VkHandle;

use super::{
    instance::Instance, macros::vk_handle_wrapper, physical_device::PhysicalDevice,
    swapchain::SwapChainSupportDetail,
};

pub struct Surface {
    instance: Arc<Instance>,
    handle: vk::SurfaceKHR,
    screen_width: u32,
    screen_height: u32,
}

vk_handle_wrapper!(Surface, SurfaceKHR);

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance.fn_surface.destroy_surface(self.handle, None);
        }
    }
}
impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: Arc<Instance>,
        window: &winit::window::Window,
        screen_width: u32,
        screen_height: u32,
    ) -> Arc<Surface> {
        let handle = unsafe {
            ash_window::create_surface(
                entry,
                &instance,
                window.display_handle().unwrap().into(),
                window.window_handle().unwrap().into(),
                None,
            )
            .expect("Failed to create surface.")
        };

        Arc::new(Surface {
            instance,
            handle,
            screen_width,
            screen_height,
        })
    }

    pub fn swapchain_support(&self, physical_device: vk::PhysicalDevice) -> SwapChainSupportDetail {
        unsafe {
            let capabilities = self
                .instance
                .fn_surface
                .get_physical_device_surface_capabilities(physical_device, self.handle())
                .expect("Failed to query for surface capabilities.");
            let formats = self
                .instance
                .fn_surface
                .get_physical_device_surface_formats(physical_device, self.handle())
                .expect("Failed to query for surface formats.");
            let present_modes = self
                .instance
                .fn_surface
                .get_physical_device_surface_present_modes(physical_device, self.handle())
                .expect("Failed to query for surface present mode.");

            SwapChainSupportDetail::new(capabilities, formats, present_modes)
        }
    }
}
