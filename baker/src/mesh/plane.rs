use super::quadric::Quadric;

#[derive(Clone, Copy)]
pub struct Plane(pub glam::Vec4);

impl Plane {
    pub fn from_three_points(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> Self {
        let ab = b - a;
        let ac = c - a;

        let normal = glam::Vec3A::cross(ab, ac).normalize();

        Self::from_normal_and_point(normal, a)
    }

    pub fn from_normal_and_point(norm: glam::Vec3A, p: glam::Vec3A) -> Self {
        let d = -p.dot(norm);

        Plane(glam::vec4(norm.x, norm.y, norm.z, d))
    }

    /// The fundamental error quadric `K_p`, such that `v^T K_p v` = `sqr distance v <-> p`
    /// Properties: Additive, Symmetric.
    pub fn fundamental_error_quadric(self) -> Quadric {
        let p: glam::DVec4 = self.0.into();
        let (a, b, c, d) = p.into();

        // Do `p p^T`
        Quadric(glam::DMat4::from_cols(a * p, b * p, c * p, d * p))
    }

    pub fn normal(&self) -> glam::Vec3A {
        self.0.into()
    }
}
