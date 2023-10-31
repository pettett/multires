use std::collections::HashMap;

use bincode::{config, Decode, Encode};
use glam::Vec3;

use crate::asset;

#[derive(Debug, Clone, Decode, Encode)]
pub struct BoundingSphere {
    center: [f32; 3],
    radius: f32,
}

impl BoundingSphere {
    pub fn center(&self) -> Vec3 {
        Vec3::from_array(self.center)
    }

    pub fn radius(&self) -> f32 {
        self.radius
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
    pub fn new(error: f32, center: Vec3, radius: f32, group: usize) -> Self {
        Self {
            indices: Vec::new(),
            tight_sphere: BoundingSphere {
                center: center.to_array(),
                radius,
            },
            saturated_sphere: BoundingSphere {
                center: center.to_array(),
                radius,
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
    pub layers: Vec<MeshLayer>,
}

#[derive(Clone, Decode, Encode)]
pub struct MeshLayer {
    pub partitions: Vec<usize>,
    pub groups: Vec<usize>,
    /// partition -> group in previous layers
    pub dependant_partitions: HashMap<usize, usize>,
    /// used by the layer below to tell what dependant tris means
    /// group -> partition
    pub partition_groups: Vec<Vec<usize>>,
    pub indices: Vec<u32>,
    pub meshlets: Vec<Meshlet>,
    pub submeshes: Vec<SubMesh>,
}

impl asset::Asset for MultiResMesh {}
