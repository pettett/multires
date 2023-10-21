pub mod asset;
use bincode::{config, Decode, Encode};
use glam::Vec3;

#[repr(C)]
#[derive(Debug, Copy, Clone, Decode, Encode)]
pub struct Meshlet {
    pub vertices: [u32; 64],
    pub indices: [u32; 378], // 126 triangles => 378 indices
    pub vertex_count: u32,
    pub index_count: u32,
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
    pub clusters: Vec<i32>,
    pub clusters2: Vec<i32>,
    pub verts: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub layer_1_indices: Vec<u32>,
    pub meshlets: Vec<Meshlet>,
}

impl asset::Asset for MultiResMesh {}
