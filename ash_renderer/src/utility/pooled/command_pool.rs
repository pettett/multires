use std::sync::Arc;

use ash::vk;

use crate::{
    utility::macros::{vk_device_owned_wrapper, vk_handle_wrapper, vk_handle_wrapper_lifetime},
    VkHandle,
};

use super::{
    super::device::Device, command_buffer_group::CommandBufferGroup,
    command_buffer_writer::CommandBufferWriter, instant_command_buffer::InstantCommandBuffer,
};

pub struct CommandBuffer {
    pool: Arc<CommandPool>,
    handle: vk::CommandBuffer,
}

vk_handle_wrapper!(CommandBuffer);

impl CommandBuffer {
    pub fn new(pool: Arc<CommandPool>, handle: vk::CommandBuffer) -> Self {
        Self { pool, handle }
    }

    pub fn reset_and_write<'a>(&'a self) -> CommandBufferWriter<'a> {
        unsafe {
            self.pool
                .device
                .reset_command_buffer(self.handle, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer for writing");
        }
        CommandBufferWriter::new(
            &self.pool.device,
            self.handle,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )
    }
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        // free commands as part of a group ideally, but we do it one by one here as it doesn't really matter for this

        unsafe {
            self.pool
                .device
                .free_command_buffers(self.pool.handle, &[self.handle]);
        }
    }
}

vk_device_owned_wrapper!(CommandPool, destroy_command_pool);

impl CommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> Arc<Self> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let pool = unsafe {
            device
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        };

        Arc::new(Self {
            device,
            handle: pool,
        })
    }

    pub fn begin_instant_command(
        self: &Arc<Self>,
        submit_queue: vk::Queue,
    ) -> InstantCommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: self.handle,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffer = unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        InstantCommandBuffer::new(self.clone(), command_buffer, submit_queue)
    }

    pub fn begin_one_shot_command(self: &Arc<Self>) -> CommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: self.handle,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let handle = unsafe {
            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        CommandBuffer {
            pool: self.clone(),
            handle,
        }
    }
}
