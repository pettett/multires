use ash::vk;
use bevy_ecs::system::Commands;

use crate::{
    core::Core,
    utility::{pooled::command_pool::CommandBuffer, render_pass::RenderPass, screen::Screen},
};

pub mod compute_culled_indices;
pub mod draw_full_res;
pub mod draw_lod_chain;
pub mod expanding_compute_culled_mesh;
pub mod indirect_tasks;
pub mod render_multires;
pub mod render_multires_indices;
pub mod render_multires_meshlets;
pub mod stub;

#[derive(Clone)]
pub struct BufferRange {
    start: u32,
    end: u32,
}

pub trait DrawPipeline {
    fn draw(&self, frame_index: usize, screen: &Screen, render_pass: &RenderPass)
        -> &CommandBuffer;

    fn init_swapchain(&mut self, core: &Core, screen: &Screen, render_pass: &RenderPass);

    /// Notification that the scene has changed, so we need to rebind to model buffers / update for new instances
    fn on_scene_dirty(&mut self) {}

    /// Notification that the user has selected a new fragment shader, and we need to regenerate our pipelines
    fn on_fragment_changed(&mut self) {}

    fn cleanup(&mut self, commands: &mut Commands) {}
}

pub fn init_rasterization_statue_create_info() -> vk::PipelineRasterizationStateCreateInfo<'static>
{
    vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL)
        .rasterizer_discard_enable(false)
        .depth_bias_enable(false)
}
pub fn init_multisample_state_create_info() -> vk::PipelineMultisampleStateCreateInfo<'static> {
    vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false)
        .min_sample_shading(0.0)
}
pub fn init_stencil_op_state() -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
        ..Default::default()
    }
}

pub fn init_depth_state_create_info() -> vk::PipelineDepthStencilStateCreateInfo<'static> {
    let stencil_state = init_stencil_op_state();

    vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .front(stencil_state)
        .back(stencil_state)
        .max_depth_bounds(1.0)
        .min_depth_bounds(1.0)
}

pub fn init_color_blend_attachment_states() -> [vk::PipelineColorBlendAttachmentState; 1] {
    [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        ..Default::default()
    }]
}
