use std::{ops::Index, sync::Arc};

use ash::vk;

use super::command_pool::CommandPool;

pub struct CommandBufferGroup {
    pool: Arc<CommandPool>,
    buffers: Vec<vk::CommandBuffer>,
}

impl CommandBufferGroup {
    pub fn new(command_pool: Arc<CommandPool>, group_count: usize) -> Self {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(group_count as _)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(**command_pool);

        let command_buffers = unsafe {
            command_pool
                .parent()
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        Self {
            pool: command_pool,
            buffers: command_buffers,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &vk::CommandBuffer> {
        self.buffers.iter()
    }
}

impl Index<usize> for CommandBufferGroup{
    type Output = vk::CommandBuffer;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffers[index]
    }
}


impl Drop for CommandBufferGroup {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .parent()
                .free_command_buffers(**self.pool, &self.buffers);
        }
    }
}
