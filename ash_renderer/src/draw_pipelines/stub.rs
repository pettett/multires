use ash::vk;

use super::DrawPipeline;

pub struct Stub;

impl DrawPipeline for Stub {
    fn draw(&self, frame_index: usize) -> ash::vk::CommandBuffer {
        vk::CommandBuffer::null()
    }

    fn init_swapchain(
        &mut self,
        core: &crate::core::Core,
        screen: &crate::screen::Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &crate::utility::render_pass::RenderPass,
    ) {
    }

    fn stats_gui(&mut self, ui: &mut egui::Ui, image_index: usize) {}
}
