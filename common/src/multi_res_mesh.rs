use std::collections::HashMap;

use bincode::{config, Decode, Encode};
use glam::Vec3;

use crate::asset;

#[derive(Debug, Default, Clone, Decode, Encode)]
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

#[repr(C)]
#[derive(Debug, Copy, Clone, Decode, Encode)]
pub struct Meshlet {
    pub vertices: [u32; 64],
    pub indices: [u32; 378], // 126 triangles => 378 indices
    pub vertex_count: u32,
    pub index_count: u32,
}

#[derive(Debug, Clone, Decode, Encode)]
pub struct SubMesh {
    pub indices: Vec<u32>,
    // Bounding sphere for the submesh
    // Similarly, the bounding sphere must be enlarged to enclose the bounding spheres of all its children in the DAG,
    // in order to ensure a monotonic view-dependent error function.
    // TODO: bounding sphere radii
    pub tight_sphere: BoundingSphere,
    pub saturated_sphere: BoundingSphere,
    pub debug_group: usize,
    pub error: f32,
}

impl SubMesh {
    pub fn new(error: f32, center: Vec3, monotonic_radius: f32, radius: f32, group: usize) -> Self {
        Self {
            indices: Vec::new(),
            tight_sphere: BoundingSphere {
                center: center.to_array(),
                radius: monotonic_radius,
            },
            saturated_sphere: BoundingSphere {
                center: center.to_array(),
                radius: radius,
            },
            error,
            debug_group: group,
        }
    }
}

unsafe impl bytemuck::Zeroable for Meshlet {}
unsafe impl bytemuck::Pod for Meshlet {}

impl Default for Meshlet {
    fn default() -> Meshlet {
        Meshlet {
            vertices: [0; 64],
            indices: [0; 378],
            vertex_count: 0,
            index_count: 0,
        }
    }
}

#[derive(Clone, Decode, Encode)]
pub struct MultiResMesh {
    pub name: String,
    pub verts: Vec<[f32; 4]>,
    pub lods: Vec<MeshLevel>,
}

/// Information for a group on layer n
#[derive(Debug, Clone, Default, Decode, Encode)]
pub struct GroupInfo {
    /// Partitions in LOD[n-1] that we were created from. Will be empty in LOD0
    //pub child_partitions: Vec<usize>,
    // Partitions that we created by subdividing ourselves
    pub partitions: Vec<usize>,

    pub tris: usize,
    /// Monotonic bounds for error function of partitions. Includes bounds of all other partitions in the group,
    /// and all partitions we are children to
    pub monotonic_bound: BoundingSphere,
}

/// Information for a partition on layer n
#[derive(Debug, Clone, Decode, Encode)]
pub struct PartitionInfo {
    /// Group in the previous LOD layer (LOD[n-1]) we have been attached to. LOD0 will have none
    pub child_group_index: Option<usize>,
    /// Group in this layer. will be usize::MAX if not yet grouped, but always valid in a loaded asset
    pub group_index: usize,
    pub tight_bound: BoundingSphere,
}

#[derive(Clone, Decode, Encode)]
pub struct MeshLevel {
    pub partition_indices: Vec<usize>,
    pub group_indices: Vec<usize>,
    pub partitions: Vec<PartitionInfo>,
    pub groups: Vec<GroupInfo>,
    /// used by the layer below to tell what dependant tris means
    pub indices: Vec<u32>,
    pub meshlets: Vec<Meshlet>,
    pub submeshes: Vec<SubMesh>,
}

impl asset::Asset for MultiResMesh {}
