use std::collections::HashSet;

use bincode::{BorrowDecode, Decode, Encode};

use crate::asset;

pub const MAX_INDICES_PER_COLOUR: usize = 126 * 3;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Vec3(glam::Vec3);

impl Into<Vec3> for glam::Vec3 {
    fn into(self) -> Vec3 {
        Vec3(self)
    }
}
impl Into<glam::Vec3> for Vec3 {
    fn into(self) -> glam::Vec3 {
        self.0
    }
}

impl Encode for Vec3 {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.0.x, encoder)?;
        bincode::Encode::encode(&self.0.y, encoder)?;
        bincode::Encode::encode(&self.0.z, encoder)?;
        Ok(())
    }
}

impl Decode for Vec3 {
    fn decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self(glam::Vec3 {
            x: bincode::Decode::decode(decoder)?,
            y: bincode::Decode::decode(decoder)?,
            z: bincode::Decode::decode(decoder)?,
        }))
    }
}
impl BorrowDecode<'_> for Vec3 {
    fn borrow_decode<D: bincode::de::Decoder>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self(glam::Vec3 {
            x: bincode::Decode::decode(decoder)?,
            y: bincode::Decode::decode(decoder)?,
            z: bincode::Decode::decode(decoder)?,
        }))
    }
}

#[derive(Debug, Default, Clone, Copy, Decode, Encode, PartialEq)]
pub struct BoundingSphere {
    center: Vec3,
    radius: f32,
}

#[derive(Debug, Default, Clone, Copy, Decode, Encode, PartialEq)]
pub struct OriginCone {
    axis: Vec3,
    cutoff: f32,
}

#[derive(Debug, Clone, Decode, Encode, Default)]
pub struct Meshlet {
    indices: Vec<u32>,
    local_indices: Vec<u32>,
    local_strip_indices: Vec<u32>,
    verts: Vec<u32>,
}

#[derive(Debug, Clone, Decode, Encode)]
pub struct MeshCluster {
    meshlets: Vec<Meshlet>,
    // Bounding sphere for the submesh
    // Will only be used for culling, so uncomment later
    pub tight_bound: BoundingSphere,
    pub tight_cone: OriginCone,

    // Similarly, the bounding sphere must be enlarged to enclose the bounding spheres of all its children in the DAG,
    // in order to ensure a monotonic view-dependent error function.
    pub saturated_bound: BoundingSphere,
    pub lod: usize, //We should not need this - group indexes are consistent across LOD
    pub error: f32,
    pub group_index: usize,
    pub child_group_index: Option<usize>,
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

#[derive(Clone, Decode, Encode, Default)]
pub struct MeshLevel {
    pub partition_indices: Vec<usize>,
    pub group_indices: Vec<usize>,
    /// used by the layer below to tell what dependant tris means
    pub indices: Vec<u32>,
}

impl asset::Asset for MultiResMesh {}

impl Meshlet {
    //TODO: Generate  local, strip, verts, etc.
    pub fn from_indices(indices: Vec<u32>) -> Self {
        let mut m = Self {
            local_indices: Vec::with_capacity(indices.len()),
            verts: Vec::with_capacity(indices.len() / 2),
            local_strip_indices: Vec::with_capacity(indices.len() / 2),
            indices,
        };

        for i in 0..m.indices.len() {
            let id = m.global_to_local_vert_index(m.indices[i] as _);
            m.local_indices.push(id);
        }

        m
    }

    pub fn from_local_indices(indices: Vec<u32>, verts: Vec<u32>) -> Self {
        let mut m = Self {
            local_strip_indices: Vec::with_capacity(indices.len() / 2),
            indices: Vec::with_capacity(indices.len()),
            local_indices: indices,
            verts,
        };

        for i in 0..m.local_indices.len() {
            m.indices.push(m.verts[m.local_indices[i] as usize]);
        }

        m
    }

    pub fn vert_count(&self) -> usize {
        self.verts.len()
    }

    pub fn local_to_global_vert_index(&self, local_vert: u32) -> u32 {
        self.verts[local_vert as usize]
    }

    pub fn global_to_local_vert_index(&mut self, mesh_vert: u32) -> u32 {
        (match self.verts.iter().position(|&x| x == mesh_vert) {
            Some(idx) => idx,
            None => {
                self.verts.push(mesh_vert);

                self.verts.len() - 1
            }
        }) as _
    }

    pub fn push_tri(&mut self, tri: [usize; 3]) {
        for v in tri {
            let id = self.global_to_local_vert_index(v as _);
            self.local_indices.push(id);
            self.indices.push(v as _);
        }

        assert!(self.indices.len() <= MAX_INDICES_PER_COLOUR);
    }

    pub fn push_temp_tri(&mut self, tri: [usize; 3]) {
        for v in tri {
            self.indices.push(v as _);
        }
    }

    pub fn local_indices(&self) -> &[u32] {
        self.local_indices.as_ref()
    }

    pub fn local_indices_mut(&mut self) -> &mut Vec<u32> {
        self.local_indices.as_mut()
    }

    pub fn indices(&self) -> &[u32] {
        self.indices.as_ref()
    }

    pub fn strip_indices(&self) -> &[u32] {
        self.local_strip_indices.as_ref()
    }

    pub fn strip_indices_mut(&mut self) -> &mut Vec<u32> {
        &mut self.local_strip_indices
    }

    pub fn verts(&self) -> &[u32] {
        self.verts.as_ref()
    }
}

impl MeshCluster {
    pub fn new(
        colours: usize,
        error: f32,
        tight_bound: BoundingSphere,
        tight_cone: OriginCone,
        saturated_bound: BoundingSphere,
        lod: usize,
        group_index: usize,
        child_group_index: Option<usize>,
    ) -> Self {
        Self {
            meshlets: vec![Meshlet::default(); colours],
            tight_bound,
            tight_cone,

            saturated_bound,
            error,
            lod,
            group_index,
            child_group_index,
        }
    }

    pub fn new_raw_temp(indices: Vec<u32>, lod: usize) -> Self {
        Self {
            meshlets: vec![Meshlet::from_indices(indices); 1],
            tight_bound: Default::default(),
            tight_cone: Default::default(),
            saturated_bound: Default::default(),
            error: 0.0,
            lod,
            group_index: 0,
            child_group_index: None,
        }
    }

    pub fn reset_meshlets(&mut self) {
        self.meshlets.clear()
    }

    pub fn add_meshlet(&mut self, m: Meshlet) {
        self.meshlets.push(m)
    }

    pub fn meshlet_for_colour(&self, colour: usize) -> &Meshlet {
        &self.meshlets[colour]
    }

    pub fn meshlet_for_colour_mut(&mut self, colour: usize) -> &mut Meshlet {
        &mut self.meshlets[colour]
    }

    pub fn meshlets(&self) -> &[Meshlet] {
        &self.meshlets
    }

    pub fn colour_count(&self) -> usize {
        self.meshlets.len()
    }

    pub fn index_count(&self) -> usize {
        self.meshlets.iter().map(|x| x.indices.len()).sum()
    }

    pub fn stripped_index_count(&self) -> usize {
        self.meshlets
            .iter()
            .map(|x| x.local_strip_indices.len())
            .sum()
    }
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

    use crate::{BoundingSphere, Meshlet};

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

    #[test]
    fn test_meshlet_creation() {
        let mut meshlet = Meshlet::default();

        meshlet.push_temp_tri([5, 3, 7]);

        assert_eq!(meshlet.local_indices(), [0, 1, 2]);

        meshlet.push_temp_tri([5, 3, 8]);

        assert_eq!(meshlet.local_indices(), [0, 1, 2, 0, 1, 3]);
    }
}
