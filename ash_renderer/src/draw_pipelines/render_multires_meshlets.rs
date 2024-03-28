// Draw meshlets from a should draw buffer

use crate::utility::GraphicsPipeline;

use super::render_multires::RenderMultires;

pub struct RenderMultiresMeshlets {
    graphics_pipeline: GraphicsPipeline,
}
impl RenderMultires for RenderMultiresMeshlets {
    fn render(
        &self,
        cmd: ash::vk::CommandBuffer,
        device: &crate::utility::device::Device,
        screen: &crate::screen::Screen,
        render_pass: &crate::utility::render_pass::RenderPass,
        frame: usize,
    ) {
        todo!()
    }
}
