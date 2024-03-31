use ash::vk;

use crate::utility::{
    device::Device, pooled::command_buffer_writer::CommandBufferWriter, render_pass::RenderPass,
    screen::Screen,
};

pub trait RenderMultires {
    fn render(
        &self,
        cmd: &mut CommandBufferWriter,
        device: &Device,
        screen: &Screen,
        render_pass: &RenderPass,
        frame: usize,
    );
}
