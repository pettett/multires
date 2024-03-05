use bincode::{BorrowDecode, Decode, Encode};

use crate::{asset, mesh_cluster::MeshCluster, mesh_vert::MeshVert};

#[derive(Clone, bincode::Decode, bincode::Encode)]
pub struct MultiResMesh {
    pub name: String,
    pub verts: Vec<MeshVert>,
    pub full_indices: Box<[u32]>,
    pub clusters: Vec<MeshCluster>,
    pub group_count: usize,
}

impl asset::Asset for MultiResMesh {}
