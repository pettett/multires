use std::sync::Arc;

use super::command_pool::CommandPool;
use crate::VkHandle;
use ash::vk;

pub struct InstantCommandBuffer {
    pool: Arc<CommandPool>,
    pub handle: vk::CommandBuffer,
    submit_queue: vk::Queue,
}

impl InstantCommandBuffer {
    pub fn new(pool: Arc<CommandPool>, handle: vk::CommandBuffer, submit_queue: vk::Queue) -> Self {
        Self {
            pool,
            handle,
            submit_queue,
        }
    }
}

impl Drop for InstantCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .parent()
                .end_command_buffer(self.handle)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let buffers_to_submit = [self.handle];

        let sumbit_infos = [vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,

            ..Default::default()
        }];

        unsafe {
            self.pool
                .parent()
                .queue_submit(self.submit_queue, &sumbit_infos, vk::Fence::null())
                .expect("Failed to Queue Submit!");
            self.pool
                .parent()
                .queue_wait_idle(self.submit_queue)
                .expect("Failed to wait Queue idle!");
            self.pool
                .parent()
                .free_command_buffers(self.pool.handle(), &buffers_to_submit);
        }
    }
}
