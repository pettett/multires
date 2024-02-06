use std::collections::HashSet;

use bincode::{Decode, Encode};
use glam::Vec3;

use crate::asset;

pub const MAX_INDICES_PER_COLOUR: usize = 126 * 3;

#[derive(Debug, Default, Clone, Decode, Encode, PartialEq)]
pub struct BoundingSphere {
    center: [f32; 3],
    radius: f32,
}

#[derive(Debug, Clone, Decode, Encode)]
pub struct MeshCluster {
    indices: Vec<Vec<u32>>,
    // Bounding sphere for the submesh
    // TODO: bounding sphere radii
    // Will only be used for culling, so uncomment later
    //pub tight_sphere: BoundingSphere,
    // Similarly, the bounding sphere must be enlarged to enclose the bounding spheres of all its children in the DAG,
    // in order to ensure a monotonic view-dependent error function.
    pub saturated_sphere: BoundingSphere,
    pub lod: usize, //TODO: In future, we should not need this - group indexes should be consistent across LOD
    pub error: f32,
    pub info: ClusterInfo,
}

#[repr(C)]
#[derive(Clone, Decode, Encode, bytemuck::Pod, bytemuck::Zeroable, Copy)]
pub struct MeshVert {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}

#[derive(Clone, Decode, Encode)]
pub struct MultiResMesh {
    pub name: String,
    pub verts: Vec<MeshVert>,
    pub full_indices: Box<[u32]>,
    pub lods: Vec<MeshLevel>,
    pub clusters: Vec<MeshCluster>,
    pub group_count: usize,
}

/// Information for a partition on layer n
#[derive(Debug, Clone, Decode, Encode, PartialEq, Default)]
pub struct ClusterInfo {
    /// Group in the previous LOD layer (LOD`n-1`) we have been attached to. LOD0 will have none
    pub child_group_index: Option<usize>,
    /// Group in this layer. will be usize::MAX if not yet grouped, but always valid in a loaded asset
    pub group_index: usize,
    /// For culling purposes - smallest bounding sphere for the partition
    pub tight_bound: BoundingSphere,
    /// Number of colours within its triangles
    pub num_colours: usize,
}

#[derive(Clone, Decode, Encode)]
pub struct MeshLevel {
    pub partition_indices: Vec<usize>,
    pub group_indices: Vec<usize>,
    /// used by the layer below to tell what dependant tris means
    pub indices: Vec<u32>,
}

impl asset::Asset for MultiResMesh {}

impl MeshCluster {
    pub fn new(
        colours: usize,
        error: f32,
        center: Vec3,
        monotonic_radius: f32,
        _radius: f32,
        lod: usize,
        info: ClusterInfo,
    ) -> Self {
        Self {
            indices: vec![Vec::new(); colours],
            // tight_sphere: BoundingSphere {
            //     center: center.to_array(),
            //     radius: radius,
            // },
            saturated_sphere: BoundingSphere {
                center: center.to_array(),
                radius: monotonic_radius,
            },
            error,
            lod,
            info,
        }
    }
    pub fn push_tri(&mut self, colour: usize, tri: [usize; 3]) {
        for v in tri {
            self.indices[colour].push(v as _)
        }

        assert!(self.indices.len() <= MAX_INDICES_PER_COLOUR);
    }
    pub fn indices_for_colour(&self, colour: usize) -> &Vec<u32> {
        &self.indices[colour]
    }
    pub fn colour_count(&self) -> usize {
        self.indices.len()
    }
    pub fn index_count(&self) -> usize {
        self.indices.iter().map(|x| x.len()).sum()
    }
    pub fn colour_vert_count(&self, colour: usize) -> usize {
        self.indices[colour].iter().collect::<HashSet<_>>().len()
    }
}

impl BoundingSphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center: center.into(),
            radius,
        }
    }

    pub fn center(&self) -> Vec3 {
        Vec3::from_array(self.center)
    }
    pub fn set_center(&mut self, center: Vec3) {
        self.center = center.into();
    }
    pub fn translate(&mut self, offset: Vec3) {
        self.center = (self.center() + offset).into();
    }
    pub fn normalise(&mut self, count: usize) {
        self.center = (self.center() / (count as f32)).into();
    }
    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn include_point(&mut self, point: Vec3) {
        self.candidate_radius(self.center().distance(point))
    }

    /// Shift this bounding sphere so it completely envelops `other`, with the minimal increase in volume.
    pub fn include_sphere(&mut self, other: &BoundingSphere) {
        // Include sphere while minimising radius increase
        let towards_other = other.center() - self.center();
        let distance_towards_other = towards_other.length();

        let furthest_point = distance_towards_other + other.radius;

        let increase_needed = furthest_point - self.radius;

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

    fn candidate_radius(&mut self, radius: f32) {
        self.radius = self.radius.max(radius)
    }

    fn assert_contains_sphere(&self, sphere: &BoundingSphere) {
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

    use crate::BoundingSphere;

    #[test]
    fn test_sphere_include_0() {
        let mut s0 = BoundingSphere::new(Vec3::ONE, 1.0);
        let s1 = BoundingSphere::new(Vec3::ONE * 2.0, 1.0);

        s0.include_sphere(&s1);

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
}
