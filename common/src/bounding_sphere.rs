use crate::vec3::Vec3;

#[derive(Debug, Default, Clone, Copy, bincode::Decode, bincode::Encode, PartialEq)]
pub struct BoundingSphere {
    center: Vec3,
    radius: f32,
}

impl BoundingSphere {
    pub fn new(center: glam::Vec3, radius: f32) -> Self {
        Self {
            center: center.into(),
            radius,
        }
    }

    pub fn center(&self) -> glam::Vec3 {
        self.center.0
    }
    pub fn packed(&self) -> glam::Vec4 {
        (self.center.0, self.radius).into()
    }
    pub fn set_center(&mut self, center: glam::Vec3) {
        self.center = center.into();
    }
    pub fn translate(&mut self, offset: glam::Vec3) {
        self.center = (self.center() + offset).into();
    }
    pub fn normalise(&mut self, count: usize) {
        self.center = (self.center() / (count as f32)).into();
    }
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Wrapper around including a sphere with 0 radius
    pub fn include_point(&mut self, point: glam::Vec3) {
        self.include_sphere(&BoundingSphere::new(point, 0.0))
    }

    /// Shift this bounding sphere so it completely envelops `other`, with the minimal increase in volume.
    pub fn include_sphere(&mut self, other: &BoundingSphere) {
        // Include sphere while minimising radius increase
        let towards_other = other.center() - self.center();
        let distance_towards_other = towards_other.length();

        let furthest_point = distance_towards_other + other.radius();

        let increase_needed = furthest_point - self.radius();

        let test_before_edit = self.clone();

        if increase_needed > 0.0 {
            // Shift half this many units towards the other sphere's center, and increase our radius by half of this

            // Add a small error factor to ensure monotonicity despite floating points
            const ERROR: f32 = 0.001;

            let half_increase_needed = increase_needed / 2.0;
            let other_dir = towards_other / distance_towards_other;

            if distance_towards_other >= half_increase_needed {
                self.radius += half_increase_needed + ERROR;

                self.translate(other_dir * half_increase_needed);
            } else {
                // distance_towards_other < half_increase_needed
                // Shift all the way to the other center, and increase radius further
                let rad_increase_needed = increase_needed - distance_towards_other;

                self.radius += rad_increase_needed + ERROR;
                self.set_center(other.center());
            }
        }

        self.assert_contains_sphere(other);
        self.assert_contains_sphere(&test_before_edit);
    }

    pub fn assert_contains_sphere(&self, sphere: &BoundingSphere) {
        let max_dist = self.center().distance(sphere.center()) + sphere.radius();
        assert!(
            max_dist <= self.radius,
            "{self:?} {sphere:?} MAX DIST - {max_dist}"
        )
    }
}

#[cfg(test)]
pub mod test {
    use glam::{vec3, Vec3};

    use crate::bounding_sphere::BoundingSphere;

    #[test]
    fn test_sphere_include_0() {
        let mut s0 = BoundingSphere::new(Vec3::ONE, 1.0);
        let s1 = BoundingSphere::new(Vec3::ONE * 2.0, 1.0);

        s0.include_sphere(&s1);

        println!("{s0:?}")
    }

    #[test]
    fn test_point_include_0() {
        let mut s0 = BoundingSphere::new(Vec3::ONE, 0.0);

        s0.include_point(Vec3::ONE * 2.0);

        println!("{s0:?}")
    }

    #[test]
    fn test_sphere_include_1() {
        let mut s0 = BoundingSphere::new(Vec3::ONE, 1.0);
        let s1 = BoundingSphere::new(Vec3::ONE * 2.0, 3.0);

        s0.include_sphere(&s1);

        println!("{s0:?}")
    }
    #[test]
    fn test_sphere_include_2() {
        let mut s0 = BoundingSphere::new(Vec3::ONE, 1.0);
        let s1 = BoundingSphere::new(Vec3::ONE * 2.0, 2.0);

        s0.include_sphere(&s1);

        println!("{s0:?}")
    }

    #[test]
    fn test_sphere_include_3() {
        let mut s0 = BoundingSphere::new(vec3(6.6490693, -9.154039, 18.179909), 2.1208615);
        let s1 = BoundingSphere::new(vec3(6.9035482, -9.225378, 18.632265), 1.5969583);

        s0.include_sphere(&s1);

        println!("{s0:?}")
    }

    // #[test]
    // fn test_meshlet_creation() {
    //     let mut meshlet = Meshlet::default();

    //     meshlet.push_temp_tri([5, 3, 7]);

    //     assert_eq!(meshlet.local_indices(), [0, 1, 2]);

    //     meshlet.push_temp_tri([5, 3, 8]);

    //     assert_eq!(meshlet.local_indices(), [0, 1, 2, 0, 1, 3]);
    // }
}
