use crate::mesh::plane::Plane;


pub struct Line {
    s: glam::Vec3A,
    e: glam::Vec3A,
}

impl Line {
    pub fn new(s: glam::Vec3A, e: glam::Vec3A) -> Self {
        Self { s, e }
    }

    pub fn delta(&self) -> glam::Vec3A {
        self.e - self.s
    }
    pub fn lerp(&self, s: f32) -> glam::Vec3A {
        self.s.lerp(self.e, s)
    }

    pub fn project(&self, point: glam::Vec3A) -> f32 {
        (point - self.s).dot(self.delta()) / self.s.distance_squared(self.e)
    }

    pub fn plane_against(&self, plane: Plane) -> Plane {
        Plane::from_normal_and_point(plane.normal().cross(self.delta()), self.s)
    }
}
