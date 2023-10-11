use bevy_ecs::prelude::*;

use glam::Mat4;

use crate::components::transform::Transform;

#[derive(Component)]
pub struct Camera {
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
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
            fovy: 45.0,
            znear: 0.01,
            zfar: 100.0,
        }
    }

    pub fn build_view_projection_matrix(&self, transform: &Transform) -> Mat4 {
        // 1.
        //let view = Mat4::look_at_rh(
        //    (*self.transform.get_pos()).into(),
        //    Vec3::new(-5603.8857, -3556.1858, -132.67264),
        //    Vec3::Z,
        //);
        let view = Mat4::look_to_rh(
            (*transform.get_pos()).into(),
            transform.forward().into(),
            transform.up().into(),
        );

        // 2.
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}
