use std::collections::{BTreeSet, HashSet};

use bincode::{Decode, Encode};
use glam::Vec3;

use crate::asset;

pub const MAX_INDICES_PER_COLOUR: usize = 126 * 3;

#[derive(Debug, Default, Clone, Decode, Encode, PartialEq)]
pub struct BoundingSphere {
    center: [f32; 3],
    radius: f32,
}

impl BoundingSphere {
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
    pub fn include_sphere(&mut self, other: &BoundingSphere) {
        self.candidate_radius(self.center().distance(other.center()) + other.radius)
    }
    fn candidate_radius(&mut self, radius: f32) {
        self.radius = self.radius.max(radius)
    }
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
	pub lod : usize, //TODO: In future, we should not need this - group indexes should be consistent across LOD
    pub error: f32,
	pub info : ClusterInfo,
}

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
			info
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
    pub lods: Vec<MeshLevel>,
    pub group_count : usize
}

/// Information for a group on layer n
#[derive(Debug, Clone, Default, Decode, Encode, PartialEq)]
pub struct GroupInfo {
    /// Partitions in LOD`n-1` that we were created from. Will be empty in LOD0
    //pub child_partitions: Vec<usize>,
    // Partitions that we created by subdividing ourselves
    pub clusters: Vec<usize>,

    /// Indexes of all groups that touch this one and could be effected by an edge collapse in this group
    pub group_neighbours: BTreeSet<usize>,

    pub tris: usize,
    /// Monotonic bounds for error function of partitions. Includes bounds of all other partitions in the group,
    /// and all partitions we are children to
    pub monotonic_bound: BoundingSphere,
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
    pub clusters: Vec<MeshCluster>,
}

impl asset::Asset for MultiResMesh {}
