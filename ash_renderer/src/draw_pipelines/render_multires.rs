use ash::vk;

use crate::{
    screen::Screen,
    utility::{device::Device, render_pass::RenderPass},
};

pub trait RenderMultires {
    fn render(
        &self,
        cmd: vk::CommandBuffer,
        device: &Device,
        screen: &Screen,
        render_pass: &RenderPass,
        frame: usize,
    );
}
