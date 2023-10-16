pub mod asset;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct MultiResMesh {
    pub name: String,
    pub clusters: Vec<i32>,
    pub clusters2: Vec<i32>,
    pub verts: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl asset::Asset for MultiResMesh {}
