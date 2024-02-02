use std::sync::Arc;

use ash::vk::{self};

use crate::VkHandle;

use super::{
    device::Device, macros::vk_handle_wrapper, physical_device::PhysicalDevice,
    structures::QueueFamilyIndices, surface::Surface,
};
pub struct SwapChainSupportDetail {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetail {
    pub fn query(physical_device: vk::PhysicalDevice, surface: &Surface) -> Self {
        unsafe {
            let capabilities = surface
                .instance
                .fn_surface
                .get_physical_device_surface_capabilities(physical_device, surface.handle())
                .expect("Failed to query for surface capabilities.");
            let formats = surface
                .instance
                .fn_surface
                .get_physical_device_surface_formats(physical_device, surface.handle())
                .expect("Failed to query for surface formats.");
            let present_modes = surface
                .instance
                .fn_surface
                .get_physical_device_surface_present_modes(physical_device, surface.handle())
                .expect("Failed to query for surface present mode.");

            Self {
                capabilities,
                formats,
                present_modes,
            }
        }
    }
    pub fn choose_swapchain_format(&self) -> vk::SurfaceFormatKHR {
        for available_format in &self.formats {
            if available_format.format == vk::Format::B8G8R8A8_SRGB
                && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *available_format;
            }
        }

        return *self.formats.first().unwrap();
    }
}

pub struct Swapchain {
    device: Arc<Device>,
    // surface is the surface onto which the swapchain will present images. If the creation succeeds, the swapchain becomes associated with surface.
    surface: Arc<Surface>,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub surface_format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
}

vk_handle_wrapper!(Swapchain, SwapchainKHR);

impl Drop for Swapchain {
    fn drop(&mut self) {
        // TODO: We should really only destroy this when we are 'finished' using it, whenever that is
        unsafe {
            self.device
                .fn_swapchain
                .destroy_swapchain(self.handle, None);
        };
    }
}
impl Swapchain {
    pub fn new(
        device: Arc<Device>,
        physical_device: &PhysicalDevice,
        window: &winit::window::Window,
        surface: Arc<Surface>,
        queue_family: &QueueFamilyIndices,
        old_swapchain: Option<&Swapchain>,
    ) -> Swapchain {
        let swapchain_support = SwapChainSupportDetail::query(physical_device.handle(), &surface);

        let surface_format = swapchain_support.choose_swapchain_format();
        let present_mode = Self::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swapchain_extent(&swapchain_support.capabilities, window);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
            if queue_family.graphics_family != queue_family.present_family {
                (
                    vk::SharingMode::CONCURRENT,
                    2,
                    vec![
                        queue_family.graphics_family.unwrap(),
                        queue_family.present_family.unwrap(),
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, vec![])
            };

        // Actual swapchain will be dropped when object lost, but this retires it now
        let old_swapchain = old_swapchain
            .map(|s| s.handle)
            .unwrap_or(vk::SwapchainKHR::null());

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            surface: surface.handle(),
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            image_array_layers: 1,
            ..Default::default()
        };
        let swapchain = unsafe {
            device
                .fn_swapchain
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create Swapchain!")
        };

        let images = unsafe {
            device
                .fn_swapchain
                .get_swapchain_images(swapchain)
                .expect("Failed to get Swapchain Images.")
        };
        Swapchain {
            surface,
            device,
            handle: swapchain,
            surface_format,
            extent,
            images,
        }
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        for &available_present_mode in available_present_modes {
            if available_present_mode == vk::PresentModeKHR::MAILBOX {
                return available_present_mode;
            }
        }

        vk::PresentModeKHR::FIFO
    }

    fn choose_swapchain_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &winit::window::Window,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            use num::clamp;

            let window_size = window.inner_size();
            println!(
                "\t\tInner Window Size: ({}, {})",
                window_size.width, window_size.height
            );

            vk::Extent2D {
                width: clamp(
                    window_size.width,
                    capabilities.min_image_extent.width.max(1),
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    window_size.height,
                    capabilities.min_image_extent.height.max(1),
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }
}
