use std::{ffi, sync::Arc};

use crate::{
    utility::{
        // the mod define some fixed functions that have been learned before.
        constants::*,
        debug::*,
        instance::Instance,
        pooled::{command_pool::CommandPool},
        structures::*,
    },
    VkHandle, TASK_GROUP_SIZE, WINDOW_TITLE,
};

use crate::utility::{device::Device, physical_device::PhysicalDevice, surface::Surface};
use ash::vk::{self, Handle};

use winit::event_loop::EventLoop;

pub struct Core {
    pub window: Arc<winit::window::Window>,
    pub device: Arc<Device>,
    pub physical_device: Arc<PhysicalDevice>,
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface>,
    pub queue_family: QueueFamilyIndices,
    pub command_pool: Arc<CommandPool>,

    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
}
impl Core {
    pub fn new(event_loop: &EventLoop<()>) -> Arc<Self> {
        println!("initing window");
        let window =
            crate::app::window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        // init vulkan stuff
        println!("initing vulkan");
        let entry = ash::Entry::linked();
        let instance = Instance::new(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers.to_vec(),
        );

        println!("initing surface");
        let surface = Surface::new(
            &entry,
            instance.clone(),
            &window,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        );

        let (debug_utils_loader, debug_messenger) =
            setup_debug_utils(VALIDATION.is_enable, &entry, &instance.handle);
        let physical_device = instance.pick_physical_device(&surface, &DEVICE_EXTENSIONS);

        //let physical_device_memory_properties = physical_device.get_memory_properties();

        let physical_device_subgroup_properties = physical_device.get_subgroup_properties();

        let (device, queue_family) = Device::create_logical_device(
            instance.clone(),
            physical_device.clone(),
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface,
        );

        // Features required for subgroupMax to work in task shader
        assert!(TASK_GROUP_SIZE <= physical_device_subgroup_properties.subgroup_size);
        if device.features.task_shader {
            assert!(physical_device_subgroup_properties
                .supported_stages
                .contains(vk::ShaderStageFlags::TASK_EXT));
        }
        assert!(physical_device_subgroup_properties
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC));

        let command_pool = CommandPool::new(device.clone(), queue_family.graphics_family.unwrap());

        Arc::new(Core {
            window,
            device,
            physical_device,
            instance,
            surface,
            queue_family,
            command_pool,
            debug_utils_loader,
            debug_messenger,
        })
    }

    /// Name the object, if a debug util loader is attached
    /// Otherwise this function will do nothing
    pub fn name_object<T: Handle + Default>(&self, name: &str, object: T) {
        let raw = object.as_raw();
        if raw == T::default().as_raw() {
            // Null pointer
            return;
        }

        let name_c = ffi::CString::new(name).unwrap();

        let object_name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(T::TYPE)
            .object_handle(raw)
            .object_name(&name_c);
        unsafe {
            self.debug_utils_loader
                .set_debug_utils_object_name(self.device.handle.handle(), &object_name_info)
                .unwrap();
        }
    }
}
impl Drop for Core {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}
