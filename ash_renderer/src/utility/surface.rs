use std::sync::Arc;

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle, HasRawWindowHandle, HasWindowHandle};

use super::{instance::Instance, macros::vk_handle_wrapper};

pub struct Surface {
    pub instance: Arc<Instance>,
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
        let surface = unsafe {
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
            handle: surface,
            screen_width,
            screen_height,
        })
    }
}
