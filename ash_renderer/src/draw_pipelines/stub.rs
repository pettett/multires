use ash::vk;

use crate::{
    screen::Screen,
    utility::{pooled::command_pool::CommandBuffer, render_pass::RenderPass},
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
        _screen: &crate::screen::Screen,
        _render_pass: &crate::utility::render_pass::RenderPass,
    ) {
    }

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}
}
