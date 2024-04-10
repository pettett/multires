use ash::vk;

use crate::utility::{
    pooled::command_pool::CommandBuffer, render_pass::RenderPass, screen::Screen,
};

use super::DrawPipeline;

pub struct Stub;

impl DrawPipeline for Stub {
    fn draw(
        &self,
        _frame_index: usize,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> &CommandBuffer {
        panic!("Stub should never reach drawing stage!")
    }

    fn init_swapchain(
        &mut self,
        _core: &crate::core::Core,
        _screen: &Screen,
        _render_pass: &crate::utility::render_pass::RenderPass,
    ) {
    }
}
