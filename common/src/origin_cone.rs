use crate::vec3::Vec3;

#[derive(Debug, Default, Clone, Copy, bincode::Decode, bincode::Encode, PartialEq)]
pub struct OriginCone {
    axis: Vec3,
    cutoff: f32,
}

impl OriginCone {
    pub fn add_axis(&mut self, axis: glam::Vec3) {
        self.axis.0 += axis;
    }
    pub fn axis(&self) -> glam::Vec3 {
        self.axis.0
    }
    pub fn normalise_axis(&mut self) {
        self.axis.0 = self.axis.0.normalize_or_zero()
    }
    pub fn packed(&self) -> glam::Vec4 {
        (self.axis.0, self.cutoff).into()
    }
    pub fn min_cutoff(&mut self, cutoff: f32) {
        self.cutoff = self.cutoff.min(cutoff);
    }
}

#[cfg(test)]
mod tests {
    use glam::{vec4, Vec3};

    use crate::OriginCone;

    #[test]
    fn general_test() {
        let mut c = OriginCone::default();

        c.add_axis(Vec3::Z);
        c.add_axis(Vec3::Z);

        c.normalise_axis();

        assert!(c.axis() == Vec3::Z);

        c.min_cutoff(1.0);

        assert_eq!(c.packed(), vec4(0.0, 0.0, 1.0, 0.0));
    }
}
