use std::{ptr, sync::Arc};

use ash::{vk, RawPtr};

use super::{device::Device, instance::Instance, structures::QueueFamilyIndices, surface::Surface};
pub struct SwapChainSupportDetail {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct Swapchain {
    device: Arc<Device>,
    // surface is the surface onto which the swapchain will present images. If the creation succeeds, the swapchain becomes associated with surface.
    surface: Arc<Surface>,
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Drop for Swapchain {
    fn drop(&mut self) {}
}
impl Swapchain {
    pub fn new(
        device: Arc<Device>,
        physical_device: vk::PhysicalDevice,
        window: &winit::window::Window,
        surface: Arc<Surface>,
        queue_family: &QueueFamilyIndices,
        old_swapchain: Option<&Swapchain>,
    ) -> Swapchain {
        let swapchain_support = query_swapchain_support(physical_device, &surface);

        let surface_format = choose_swapchain_format(&swapchain_support.formats);
        let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = choose_swapchain_extent(&swapchain_support.capabilities, window);

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
        let old_swapchain = old_swapchain
            .map(|s| s.swapchain)
            .unwrap_or(vk::SwapchainKHR::null());

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface.handle,
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

        // TODO: We should really only destroy this when we are 'finished' using it, whenever that is
        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe {
                device.fn_swapchain.destroy_swapchain(old_swapchain, None);
            };
        }

        Swapchain {
            surface,
            device,
            swapchain,
            format: surface_format.format,
            extent,
            images,
        }
    }
}

fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for available_format in available_formats {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return available_format.clone();
        }
    }

    return available_formats.first().unwrap().clone();
}

fn choose_swapchain_present_mode(
    available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
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
                window_size.width as u32,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                window_size.height as u32,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

pub fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> SwapChainSupportDetail {
    unsafe {
        let capabilities = surface
            .instance
            .fn_surface
            .get_physical_device_surface_capabilities(physical_device, surface.handle)
            .expect("Failed to query for surface capabilities.");
        let formats = surface
            .instance
            .fn_surface
            .get_physical_device_surface_formats(physical_device, surface.handle)
            .expect("Failed to query for surface formats.");
        let present_modes = surface
            .instance
            .fn_surface
            .get_physical_device_surface_present_modes(physical_device, surface.handle)
            .expect("Failed to query for surface present mode.");

        SwapChainSupportDetail {
            capabilities,
            formats,
            present_modes,
        }
    }
}
