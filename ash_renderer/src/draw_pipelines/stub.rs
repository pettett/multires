use ash::vk;

use super::DrawPipeline;

pub struct Stub;

impl DrawPipeline for Stub {
    fn draw(&self, _frame_index: usize) -> ash::vk::CommandBuffer {
        vk::CommandBuffer::null()
    }

    fn init_swapchain(
        &mut self,
        _core: &crate::core::Core,
        _screen: &crate::screen::Screen,
        _submesh_count: u32,
        _instance_count: u32,
        _render_pass: &crate::utility::render_pass::RenderPass,
    ) {
    }

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}
}
