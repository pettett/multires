use std::{ptr, sync::Arc};

use ash::vk;

use crate::{utility::macros::vk_device_owned_wrapper, VkHandle};

use super::super::device::Device;

pub struct SingleTimeCommandBuffer {
    pool: Arc<CommandPool>,
    pub handle: vk::CommandBuffer,
    submit_queue: vk::Queue,
}

impl Drop for SingleTimeCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device
                .handle
                .end_command_buffer(self.handle)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let buffers_to_submit = [self.handle];

        let sumbit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            self.pool
                .device
                .handle
                .queue_submit(self.submit_queue, &sumbit_infos, vk::Fence::null())
                .expect("Failed to Queue Submit!");
            self.pool
                .device
                .handle
                .queue_wait_idle(self.submit_queue)
                .expect("Failed to wait Queue idle!");
            self.pool
                .device
                .handle
                .free_command_buffers(self.pool.handle, &buffers_to_submit);
        }
    }
}

pub struct OneShotCommandBuffer {
    pool: Arc<CommandPool>,
    pub handle: vk::CommandBuffer,
}
impl OneShotCommandBuffer {
    pub fn end(&self) {
        unsafe {
            self.pool
                .device
                .handle
                .end_command_buffer(self.handle)
                .expect("Failed to record Command Buffer at Ending!");
        }
    }
}
impl Drop for OneShotCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device
                .handle
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
                .handle
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        };

        Arc::new(Self {
            device,
            handle: pool,
        })
    }

    pub fn allocate_group(&self, command_buffer_count: u32) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(command_buffer_count)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.handle);

        unsafe {
            self.device
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }
    }

    pub fn begin_single_time_command(
        self: &Arc<Self>,
        submit_queue: vk::Queue,
    ) -> SingleTimeCommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: self.handle,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffer = unsafe {
            self.device
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .handle
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        SingleTimeCommandBuffer {
            pool: self.clone(),
            handle: command_buffer,
            submit_queue,
        }
    }

    pub fn begin_one_shot_command(self: &Arc<Self>) -> OneShotCommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            command_buffer_count: 1,
            command_pool: self.handle,
            level: vk::CommandBufferLevel::PRIMARY,
            ..Default::default()
        };

        let command_buffer = unsafe {
            self.device
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .handle
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        OneShotCommandBuffer {
            pool: self.clone(),
            handle: command_buffer,
        }
    }
}
