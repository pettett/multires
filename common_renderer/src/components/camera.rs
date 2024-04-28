use bevy_ecs::prelude::*;

use glam::Mat4;

use crate::components::transform::Transform;

#[derive(Component, Clone, Debug, PartialEq)]
pub struct Camera {
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    pub part_highlight: i32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(
    &[
		1.0, 0.0, 0.0, 0.0,
    	0.0, 1.0, 0.0, 0.0,
    	0.0, 0.0, 0.5, 0.5,
    	0.0, 0.0, 0.0, 1.0
	]
);

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            //transform: Transform::new(Vec3A::new(-8000.0, -2000.0, 1000.0), Quat::IDENTITY),
            aspect,
            fovy: 45.0_f32.to_radians(),
            znear: 0.01,
            zfar: 100.0,
            part_highlight: -1,
        }
    }

    pub fn on_resize(&mut self, size: &winit::dpi::PhysicalSize<u32>) {
        self.aspect = size.width as f32 / size.height as f32;
    }

    pub fn build_view_projection_matrix(&self, transform: &Transform) -> Mat4 {
        // 1.
        //let view = Mat4::look_at_rh(
        //    (*self.transform.get_pos()).into(),
        //    Vec3::new(-5603.8857, -3556.1858, -132.67264),
        //    Vec3::Z,
        //);
        // let view = Mat4::look_to_rh(
        //     (*transform.get_pos()).into(),
        //     transform.forward().into(),
        //     transform.up().into(),
        // );

        let view = transform.get_local_to_world().inverse();

        // 2.
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);

        // println!(
        //     "FOV: {}",
        //     (2.0 * f32::atan(1.0 / proj.x_axis.x)).to_degrees()
        // );

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }

    pub fn znear(&self) -> f32 {
        self.znear
    }
}
