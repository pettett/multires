use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};
use glam::Mat4;

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Default, Copy, Clone, bytemuck::Zeroable, Component)]
pub struct CameraUniform {
    view_proj: Mat4,
    part_highlight: i32,
}
unsafe impl bytemuck::Pod for CameraUniform {}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            part_highlight: -1,
        }
    }
}
pub fn update_view_proj(mut query: Query<(&mut CameraUniform, &Camera, &Transform)>) {
    for (mut u, c, t) in query.iter_mut() {
        u.view_proj = c.build_view_projection_matrix(t);
        u.part_highlight = c.part_highlight;
    }
}
