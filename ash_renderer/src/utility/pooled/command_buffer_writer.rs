use std::{ops::Index, sync::Arc};

use ash::vk;

use crate::utility::{
    device::Device,
    macros::{vk_handle_wrapper, vk_handle_wrapper_lifetime},
    screen::Screen,
};

pub struct CommandBufferWriter<'a> {
    device: &'a Device,
    handle: vk::CommandBuffer,
}

vk_handle_wrapper_lifetime!(CommandBufferWriter, CommandBuffer);

impl<'a> CommandBufferWriter<'a> {
    pub fn new(
        device: &'a Device,
        handle: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> Self {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(flags);

        unsafe {
            device
                .begin_command_buffer(handle, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        Self { device, handle }
    }

    pub fn set_dynamic_screen(&mut self, screen: &Screen) {
        unsafe {
            self.device.cmd_set_scissor(
                self.handle,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: screen.swapchain().extent,
                }],
            );
            self.device.cmd_set_viewport(
                self.handle,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: screen.swapchain().extent.width as _,
                    height: screen.swapchain().extent.height as _,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
        }
    }
    pub fn start_render_pass(&mut self, render_pass_begin_info: vk::RenderPassBeginInfo<'_>) {
        unsafe {
            self.device.cmd_begin_render_pass(
                **self,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }
    }
}
impl<'a> Drop for CommandBufferWriter<'a> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .end_command_buffer(self.handle)
                .expect("Failed to record Command Buffer at Ending!");
        }
    }
}
