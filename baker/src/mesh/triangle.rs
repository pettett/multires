use super::{line::Line, plane::Plane};

#[derive(Clone, Debug, PartialEq)]
pub struct Triangle {
    a: glam::Vec3A,
    b: glam::Vec3A,
    c: glam::Vec3A,
}

impl Triangle {
    pub fn new(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> Self {
        Self { a, b, c }
    }

    pub fn square_dist_from_point(&self, point: glam::Vec3A) -> f32 {
        self.closest_to_point(point).distance_squared(point)
    }

    pub fn area(&self) -> f32 {
        (self.c - self.a).cross(self.c - self.b).length() / 2.0
    }

    /// Generate self.a random point on self.a triangle by sampling from self.a parallelogram and reflecting
    ///
    /// https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html
    pub fn sample_random_point(&self, rng: &mut impl rand::Rng) -> glam::Vec3A {
        let mut ua = rng.gen_range(0.0..1.0);
        let mut ub = rng.gen_range(0.0..1.0);

        if ua + ub > 1.0 {
            ua = 1.0 - ua;
            ub = 1.0 - ub;
        }

        let w = ua * (self.a - self.c) + ub * (self.b - self.c);

        self.c + w
    }

    pub fn closest_to_point(&self, point: glam::Vec3A) -> glam::Vec3A {
        let plane = Plane::from_three_points(self.a, self.b, self.c);

        let ab = Line::new(self.a, self.b);
        let bc = Line::new(self.b, self.c);
        let ca = Line::new(self.c, self.a);

        let on_ab = ab.project(point);
        let on_bc = bc.project(point);
        let on_ca = ca.project(point);

        if on_ca > 1.0 && on_ab < 0.0 {
            return self.a;
        }
        if on_ab > 1.0 && on_bc < 0.0 {
            return self.b;
        }
        if on_bc > 1.0 && on_ca < 0.0 {
            return self.c;
        }

        let pab = ab.plane_against(plane);
        let pbc = bc.plane_against(plane);
        let pca = ca.plane_against(plane);

        if (0.0..=1.0).contains(&on_ab) && !pab.is_above(point) {
            return ab.lerp(on_ab);
        }

        if (0.0..=1.0).contains(&on_bc) && !pbc.is_above(point) {
            return bc.lerp(on_bc);
        }

        if (0.0..=1.0).contains(&on_ca) && !pca.is_above(point) {
            return ca.lerp(on_ca);
        }

        let d = plane.signed_distance(point);

        point - plane.normal() * d
    }
}

#[cfg(test)]
pub mod tests {
    use glam::Vec3A;

    use super::*;

    #[test]
    fn test_closest_tri_point() {
        let tri = Triangle::new(Vec3A::ZERO, Vec3A::X, Vec3A::Y);
        assert_eq!(tri.closest_to_point(Vec3A::ZERO), Vec3A::ZERO);

        assert_eq!(tri.closest_to_point(Vec3A::X,), Vec3A::X);

        assert_eq!(tri.closest_to_point(Vec3A::Y), Vec3A::Y);

        assert_eq!(tri.closest_to_point(Vec3A::Y * 2.0), Vec3A::Y);
        assert_eq!(tri.closest_to_point(Vec3A::X * 2.0), Vec3A::X);
        assert_eq!(
            tri.closest_to_point(Vec3A::ZERO - Vec3A::X * 2.0),
            Vec3A::ZERO
        );

        assert_eq!(
            tri.closest_to_point(Vec3A::Y.lerp(Vec3A::X, 0.5),),
            Vec3A::Y.lerp(Vec3A::X, 0.5)
        );

        assert_eq!(
            tri.closest_to_point(Vec3A::ZERO.lerp(Vec3A::X, 0.5) - Vec3A::Y,),
            Vec3A::ZERO.lerp(Vec3A::X, 0.5)
        );

        assert_eq!(
            tri.closest_to_point((Vec3A::ZERO + Vec3A::X + Vec3A::Y) / 3.0 + Vec3A::Z,),
            (Vec3A::ZERO + Vec3A::X + Vec3A::Y) / 3.0
        );
    }

    #[test]
    fn test_sqr_dst() {
        let tri = Triangle::new(Vec3A::ZERO, Vec3A::X, Vec3A::Y);
        assert_eq!(tri.square_dist_from_point(Vec3A::ZERO), 0.0);

        assert_eq!(tri.square_dist_from_point(Vec3A::X), 0.0);

        assert_eq!(tri.square_dist_from_point(Vec3A::Y), 0.0);

        assert_eq!(tri.square_dist_from_point(Vec3A::Y + Vec3A::Z), 1.0);
        assert_eq!(tri.square_dist_from_point(Vec3A::Y + 2.0 * Vec3A::Z), 4.0);
    }

    #[test]
    fn test_random_point() {
        let tri = Triangle::new(Vec3A::ZERO, Vec3A::X, Vec3A::Y);
        let mut rand = rand::rngs::OsRng::default();
        for _ in 0..100 {
            assert_eq!(
                tri.square_dist_from_point(tri.sample_random_point(&mut rand)),
                0.0
            );
        }
    }
}
