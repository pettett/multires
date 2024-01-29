use std::ops;

/// Quadric type. Internally a DMat4.
#[derive(Default)]
#[repr(transparent)]
pub struct Quadric(pub glam::DMat4);

impl ops::Add for &Quadric {
    type Output = Quadric;

    fn add(self, rhs: Self) -> Self::Output {
        Quadric(self.0 + rhs.0)
    }
}

impl ops::Mul<f64> for Quadric {
    type Output = Quadric;

    fn mul(self, rhs: f64) -> Self::Output {
        Quadric(self.0 * rhs)
    }
}

impl ops::AddAssign<Self> for Quadric {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}
impl Quadric {
    /// Calculate error from Q and vertex, `v^T K_p v`
    pub fn quadric_error(&self, v: glam::Vec3A) -> f64 {
        let v: glam::Vec4 = (v, 1.0).into();
        let v: glam::DVec4 = v.into();
        v.dot(self.0 * v)
    }
}
