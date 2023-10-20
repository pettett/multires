//! Split reduplicate functions in this share module

pub mod v1;
pub mod v2;

use ash::extensions::khr::Maintenance4;
use ash::vk;
use ash::vk::PhysicalDeviceMaintenance4Features;
use ash::vk::PhysicalDeviceMeshShaderFeaturesEXT;

use std::ffi::CString;
use std::os::raw::c_char;
use std::os::raw::c_void;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use crate::utility::constants::*;
use crate::utility::debug;
use crate::utility::platforms;
use crate::utility::structures::*;

use super::device::Device;
use super::instance::Instance;
use super::surface::Surface;
use super::swapchain::query_swapchain_support;

pub fn pick_physical_device(
    instance: &Instance,
    surface_stuff: &Surface,
    required_device_extensions: &DeviceExtension,
) -> vk::PhysicalDevice {
    let physical_devices = unsafe {
        instance
            .handle
            .enumerate_physical_devices()
            .expect("Failed to enumerate Physical Devices!")
    };

    let result = physical_devices.iter().find(|physical_device| {
        let is_suitable = is_physical_device_suitable(
            instance,
            **physical_device,
            surface_stuff,
            required_device_extensions,
        );

        // if is_suitable {
        //     let device_properties = instance.get_physical_device_properties(**physical_device);
        //     let device_name = super::tools::vk_to_string(&device_properties.device_name);
        //     println!("Using GPU: {}", device_name);
        // }

        is_suitable
    });

    match result {
        Some(p_physical_device) => *p_physical_device,
        None => panic!("Failed to find a suitable GPU!"),
    }
}

pub fn is_physical_device_suitable(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &Surface,
    required_device_extensions: &DeviceExtension,
) -> bool {
    let device_features = unsafe {
        instance
            .handle
            .get_physical_device_features(physical_device)
    };

    let indices = find_queue_family(instance, physical_device, surface_stuff);

    let is_queue_family_supported = indices.is_complete();
    let is_device_extension_supported = check_device_extension_support(
        &instance.handle,
        physical_device,
        required_device_extensions,
    );
    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = query_swapchain_support(physical_device, surface_stuff);
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };
    let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

    return is_queue_family_supported
        && is_device_extension_supported
        && is_swapchain_supported
        && is_support_sampler_anisotropy;
}

pub fn find_queue_family(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> QueueFamilyIndices {
    let queue_families = unsafe {
        instance
            .handle
            .get_physical_device_queue_family_properties(physical_device)
    };

    let mut queue_family_indices = QueueFamilyIndices::new();

    let mut index = 0;
    for queue_family in queue_families.iter() {
        if queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some(index);
        }

        let is_present_support = unsafe {
            instance.fn_surface.get_physical_device_surface_support(
                physical_device,
                index as u32,
                surface.handle,
            )
        }
        .unwrap();

        if queue_family.queue_count > 0 && is_present_support {
            queue_family_indices.present_family = Some(index);
        }

        if queue_family_indices.is_complete() {
            break;
        }

        index += 1;
    }

    queue_family_indices
}

pub fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device_extensions: &DeviceExtension,
) -> bool {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties.")
    };

    let mut available_extension_names = vec![];

    for extension in available_extensions.iter() {
        let extension_name = super::tools::vk_to_string(&extension.extension_name);

        available_extension_names.push(extension_name);
    }

    use std::collections::HashSet;
    let mut required_extensions = HashSet::new();
    for extension in device_extensions.names.iter() {
        required_extensions.insert(extension.to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    return required_extensions.is_empty();
}

pub fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
    };

    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Failed to create Shader Module!")
    }
}

pub fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return i as u32;
        }
    }

    panic!("Failed to find suitable memory type!")
}

pub fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}

pub fn find_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::Format {
    find_supported_format(
        instance,
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
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    candidate_formats: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> vk::Format {
    for &format in candidate_formats.iter() {
        let format_properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        if tiling == vk::ImageTiling::LINEAR
            && format_properties.linear_tiling_features.contains(features)
        {
            return format.clone();
        } else if tiling == vk::ImageTiling::OPTIMAL
            && format_properties.optimal_tiling_features.contains(features)
        {
            return format.clone();
        }
    }

    panic!("Failed to find supported format!")
}

pub fn check_mipmap_support(
    instance: &ash::Instance,
    physcial_device: vk::PhysicalDevice,
    image_format: vk::Format,
) {
    let format_properties =
        unsafe { instance.get_physical_device_format_properties(physcial_device, image_format) };

    let is_sample_image_filter_linear_support = format_properties
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

    if is_sample_image_filter_linear_support == false {
        panic!("Texture Image format does not support linear blitting!")
    }
}
