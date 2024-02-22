use ash::vk;
use bevy_ecs::schedule::Schedule;

use crate::{core::Core, screen::Screen, utility::render_pass::RenderPass};

pub mod compute_culled_indices;
pub mod compute_culled_mesh;
pub mod draw_indirect;
pub mod expanding_compute_culled_mesh;
pub mod indirect_tasks;
pub mod stub;

pub trait DrawPipeline {
    fn draw(&self, frame_index: usize) -> vk::CommandBuffer;

    fn init_swapchain(&mut self, core: &Core, screen: &Screen, render_pass: &RenderPass);

    /// Draw our stats UI
    fn stats_gui(&mut self, ui: &mut egui::Ui, frame_index: usize);

    /// Notification that the scene has changed, so we need to rebind to model buffers / update for new instances
    fn on_scene_dirty(&mut self) {}

    /// Notification that the user has selected a new fragment shader, and we need to regenerate our pipelines
    fn on_fragment_changed(&mut self) {}
}

pub fn init_rasterization_statue_create_info() -> vk::PipelineRasterizationStateCreateInfo {
    vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0)
        .polygon_mode(vk::PolygonMode::FILL)
        .rasterizer_discard_enable(false)
        .depth_bias_enable(false)
        .build()
}
pub fn init_multisample_state_create_info() -> vk::PipelineMultisampleStateCreateInfo {
    vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
        ..Default::default()
    }
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

pub fn init_depth_state_create_info() -> vk::PipelineDepthStencilStateCreateInfo {
    let stencil_state = init_stencil_op_state();

    vk::PipelineDepthStencilStateCreateInfo {
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: vk::TRUE,
        depth_write_enable: vk::TRUE,
        depth_compare_op: vk::CompareOp::LESS,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
        ..Default::default()
    }
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
