use std::{
    ffi::{c_void, CString},
    ptr,
    sync::Arc,
};

use ash::vk;
use raw_window_handle::HasRawDisplayHandle;

use crate::{
    utility::{
        constants::{API_VERSION, APPLICATION_VERSION, ENGINE_VERSION, VALIDATION},
        debug,
    },
    VkHandle,
};

use super::{
    extensions::Extensions, physical_device::PhysicalDevice,
    queue_family_indices::QueueFamilyIndices, surface::Surface, swapchain::SwapChainSupportDetail,
};

pub struct Instance {
    handle: ash::Instance,
    pub fn_surface: ash::extensions::khr::Surface,
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Instance {
    pub fn new(
        entry: &ash::Entry,
        window_title: &str,
        is_enable_debug: bool,
        window: &winit::window::Window,
        required_instance_extensions: Extensions,
        required_validation_layers: &Vec<&str>,
    ) -> Arc<Self> {
        if is_enable_debug
            && !debug::check_validation_layer_support(entry, required_validation_layers)
        {
            panic!("Validation layers requested, but not available!");
        }

        let app_name = CString::new(window_title).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: ENGINE_VERSION,
            api_version: API_VERSION,
            ..Default::default()
        };

        // This create info used to debug issues in vk::createInstance and vk::destroyInstance.
        let debug_utils_create_info = debug::populate_debug_messenger_create_info();

        // VK_EXT debug report has been requested here.

        let mut extension_names = required_instance_extensions.get_extensions_raw_names();

        extension_names.extend_from_slice(
            ash_window::enumerate_required_extensions(window.raw_display_handle()).unwrap(),
        );

        let requred_validation_layer_raw_names: Vec<CString> = required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let layer_names: Vec<*const i8> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo {
            p_next: if VALIDATION.is_enable {
                &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const c_void
            } else {
                ptr::null()
            },
            p_application_info: &app_info,
            pp_enabled_layer_names: if is_enable_debug {
                layer_names.as_ptr()
            } else {
                ptr::null()
            },
            enabled_layer_count: if is_enable_debug {
                layer_names.len()
            } else {
                0
            } as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            ..Default::default()
        };

        println!("Creating instance");
        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        let fn_surface = ash::extensions::khr::Surface::new(entry, &instance);

        Arc::new(Self {
            handle: instance,
            fn_surface,
        })
    }

    pub fn find_depth_format(&self, physical_device: vk::PhysicalDevice) -> vk::Format {
        self.find_supported_format(
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    pub fn find_supported_format(
        &self,
        physical_device: vk::PhysicalDevice,
        candidate_formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties = unsafe {
                self.handle
                    .get_physical_device_format_properties(physical_device, format)
            };
            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features)
                || tiling == vk::ImageTiling::OPTIMAL
                    && format_properties.optimal_tiling_features.contains(features)
            {
                return format;
            }
        }

        panic!("Failed to find supported format!")
    }

    pub fn check_mipmap_support(&self, physical_device: &PhysicalDevice, image_format: vk::Format) {
        let format_properties = unsafe {
            self.handle
                .get_physical_device_format_properties(physical_device.handle(), image_format)
        };

        let is_sample_image_filter_linear_support = format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

        if !is_sample_image_filter_linear_support {
            panic!("Texture Image format does not support linear blitting!")
        }
    }
    pub fn pick_physical_device(
        self: &Arc<Self>,
        surface: &Surface,
        required_device_extensions: &Extensions,
        optional_device_extensions: &Extensions,
    ) -> Arc<PhysicalDevice> {
        let physical_devices = unsafe {
            self.enumerate_physical_devices()
                .expect("Failed to enumerate Physical Devices!")
        };

        let result = physical_devices.iter().find_map(|physical_device| {
            self.is_physical_device_suitable(
                *physical_device,
                surface,
                required_device_extensions,
                optional_device_extensions,
            )
            .map(|e| (physical_device, e))
        });

        match result {
            Some((p_physical_device, supported_extensions)) => Arc::new(PhysicalDevice::new(
                *p_physical_device,
                self.clone(),
                supported_extensions,
            )),
            None => panic!("Failed to find a suitable GPU!"),
        }
    }

    pub fn is_physical_device_suitable(
        &self,
        physical_device: vk::PhysicalDevice,
        surface_stuff: &Surface,
        required_device_extensions: &Extensions,
        optional_device_extensions: &Extensions,
    ) -> Option<Extensions> {
        let device_features = unsafe { self.handle.get_physical_device_features(physical_device) };

        let indices = self.find_queue_family(physical_device, surface_stuff);

        let is_queue_family_supported = indices.is_complete();
        let present_required_extensions =
            required_device_extensions.get_device_extension_support(physical_device, self);
        let present_optional_extensions =
            optional_device_extensions.get_device_extension_support(physical_device, self);

        let is_swapchain_supported = if present_required_extensions == *required_device_extensions {
            let swapchain_support = SwapChainSupportDetail::query(physical_device, surface_stuff);
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
        } else {
            false
        };
        let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

        (is_queue_family_supported
            && present_required_extensions == *required_device_extensions
            && is_swapchain_supported
            && is_support_sampler_anisotropy)
            .then(|| present_required_extensions.union(&present_optional_extensions))
    }

    pub fn find_queue_family(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> QueueFamilyIndices {
        let queue_families = unsafe {
            self.handle
                .get_physical_device_queue_family_properties(physical_device)
        };

        let mut queue_family_indices = QueueFamilyIndices::new();

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(index as _);
            }

            let is_present_support = unsafe {
                self.fn_surface.get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface.handle(),
                )
            }
            .unwrap();

            if queue_family.queue_count > 0 && is_present_support {
                queue_family_indices.present_family = Some(index as _);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        queue_family_indices
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_instance(None);
        }
    }
}
