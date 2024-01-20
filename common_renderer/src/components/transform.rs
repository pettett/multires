use bevy_ecs::component::Component;
use glam::{Mat4, Quat, Vec3, Vec3A};

#[derive(Debug, Copy, Clone, Default, bytemuck::Zeroable, Component)]
pub struct Transform {
    pos: Vec3A,
    scale: Vec3A,
    rot: Quat,
}

impl Transform {
    pub fn new_scaled(pos: Vec3A, rot: Quat, scale: Vec3A) -> Self {
        Self { pos, rot, scale }
    }

    pub fn new(pos: Vec3A, rot: Quat) -> Self {
        Self {
            pos,
            rot,
            scale: Vec3A::ONE,
        }
    }

    pub fn new_pos(pos: Vec3A) -> Self {
        Self {
            pos,
            rot: Quat::IDENTITY,
            scale: Vec3A::ONE,
        }
    }

    pub fn get_pos(&self) -> &Vec3A {
        &self.pos
    }

    pub fn translate(&mut self, translation: Vec3A) {
        self.pos += translation
    }

    pub fn translate_local(&mut self, translation: Vec3A) {
        self.pos += self.rot.mul_vec3a(translation)
    }

    pub fn forward(&self) -> Vec3A {
        self.rot.mul_vec3a(Vec3A::X)
    }

    pub fn left(&self) -> Vec3A {
        self.rot.mul_vec3a(Vec3A::Y)
    }

    pub fn up(&self) -> Vec3A {
        self.rot.mul_vec3a(Vec3A::Z)
    }

    pub fn look_at(&mut self, target: Vec3A) {
        self.rot = Quat::from_rotation_arc(Vec3::X, (self.pos - target).normalize().into());
    }

    pub fn get_rot(&self) -> &Quat {
        &self.rot
    }
    pub fn get_rot_mut(&mut self) -> &mut Quat {
        &mut self.rot
    }
    pub fn get_local_to_world(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale.into(), self.rot, self.pos.into())
    }

    pub fn scale_mut(&mut self) -> &mut Vec3A {
        &mut self.scale
    }
}
