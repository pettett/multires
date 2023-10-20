use std::{
    ffi::{c_void, CString},
    ptr,
    sync::Arc,
};

use ash::vk;

use crate::utility::{
    constants::{API_VERSION, APPLICATION_VERSION, ENGINE_VERSION, VALIDATION},
    debug, platforms,
};

pub struct Instance {
    pub instance: ash::Instance,
}
impl Instance {
    pub fn new(
        entry: &ash::Entry,
        window_title: &str,
        is_enable_debug: bool,
        required_validation_layers: &Vec<&str>,
    ) -> Arc<Self> {
        if is_enable_debug
            && debug::check_validation_layer_support(entry, required_validation_layers) == false
        {
            panic!("Validation layers requested, but not available!");
        }

        let app_name = CString::new(window_title).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: ptr::null(),
            application_version: APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: ENGINE_VERSION,
            api_version: API_VERSION,
        };

        // This create info used to debug issues in vk::createInstance and vk::destroyInstance.
        let debug_utils_create_info = debug::populate_debug_messenger_create_info();

        // VK_EXT debug report has been requested here.
        let extension_names = platforms::required_extension_names();

        let requred_validation_layer_raw_names: Vec<CString> = required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let layer_names: Vec<*const i8> = requred_validation_layer_raw_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_next: if VALIDATION.is_enable {
                &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                    as *const c_void
            } else {
                ptr::null()
            },
            flags: vk::InstanceCreateFlags::empty(),
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
        };

        println!("Creating instance");
        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        Arc::new(Self { instance })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
