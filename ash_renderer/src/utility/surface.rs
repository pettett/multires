use std::sync::Arc;

use ash::vk;

use super::{instance::Instance, macros::vk_handle_wrapper, platforms};

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
            platforms::create_surface(entry, &instance.handle, window)
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
