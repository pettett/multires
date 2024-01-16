use std::ops::Deref;

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};

use super::gpu_multi_res_mesh::MultiResMeshRenderer;

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Default, Copy, Clone, bytemuck::Zeroable, Component)]
pub struct CameraUniform {
    view_proj: glam::Mat4,
    camera_pos: glam::Vec3,
    part_highlight: i32,
    error: f32,
    mode: u32,
}
unsafe impl bytemuck::Pod for CameraUniform {}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY,

            part_highlight: -1,
            camera_pos: glam::Vec3::ZERO,
            error: 0.0,
            mode: 0,
        }
    }
}
pub fn update_view_proj(
    mut query: Query<(&mut CameraUniform, &Camera, &Transform)>,
    renderer: Res<MultiResMeshRenderer>,
) {
    for (mut u, c, t) in query.iter_mut() {
        u.view_proj = c.build_view_projection_matrix(t);
        u.camera_pos = (*t.get_pos()).into();
        u.part_highlight = c.part_highlight;
        u.error = renderer.error_target;
        u.mode = renderer.error_calc.mode();
    }
}
