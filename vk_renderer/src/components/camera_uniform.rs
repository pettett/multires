use bevy_ecs::prelude::*;

use glam::Mat4;

use common_renderer::components::transform::Transform;

use common_renderer::components::camera::Camera;

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Component)]
pub struct CameraUniform {
    view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
        }
    }
}
pub fn update_view_proj(mut query: Query<(&mut CameraUniform, &Camera, &Transform)>) {
    for (mut u, c, t) in query.iter_mut() {
        u.view_proj = c.build_view_projection_matrix(t);
    }
}
