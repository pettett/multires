use std::sync::Arc;

use ash::vk;

use super::device::Device;

pub struct DescriptorPool {
    device: Arc<Device>,
    pub descriptor_pool: vk::DescriptorPool,
}

pub struct CommandPool {
    device: Arc<Device>,
    pub pool: vk::CommandPool,
}

impl CommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
        };

        let pool = unsafe {
            device
                .device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        };

        Self { device, pool }
    }
}
impl Drop for CommandPool {
    fn drop(&mut self) {
        //TODO: Command buffers will be freed when the pool is dropped, so any buffers created from this pool must be invalidated
        unsafe {
            self.device.device.destroy_command_pool(self.pool, None);
        }
    }
}
