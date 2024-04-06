use std::{ops::Index, sync::Arc};

use ash::vk;

use crate::VkHandle;

use super::{
    command_buffer_writer::CommandBufferWriter,
    command_pool::{CommandBuffer, CommandPool},
};

pub struct CommandBufferGroup {
    pool: Arc<CommandPool>,
    buffers: Vec<CommandBuffer>,
}

impl CommandBufferGroup {
    pub fn new(command_pool: Arc<CommandPool>, group_count: usize) -> Self {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
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
            buffers: command_buffers
                .into_iter()
                .map(|handle| CommandBuffer::new(command_pool.clone(), handle))
                .collect(),
            pool: command_pool,
        }
    }

    pub fn iter_to_fill<'a>(&'a mut self) -> impl Iterator<Item = CommandBufferWriter<'a>> + '_ {
        self.buffers.iter().map(|handle| {
            CommandBufferWriter::new(
                self.pool.parent(),
                handle.handle(),
                vk::CommandBufferUsageFlags::empty(),
            )
        })
    }

    pub fn iter_handles(&self) -> impl Iterator<Item = &CommandBuffer> {
        self.buffers.iter()
    }

    pub fn get(&self, index: usize) -> &CommandBuffer {
        &self.buffers[index]
    }
}
