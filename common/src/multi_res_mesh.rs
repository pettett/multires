use bincode::{BorrowDecode, Decode, Encode};

use crate::{asset, mesh_cluster::MeshCluster, mesh_vert::MeshVert};

pub const MAX_INDICES_PER_COLOUR: usize = 126 * 3;

#[derive(Clone, bincode::Decode, bincode::Encode)]
pub struct MultiResMesh {
    pub name: String,
    pub verts: Vec<MeshVert>,
    pub full_indices: Box<[u32]>,
    pub clusters: Vec<MeshCluster>,
    pub group_count: usize,
}

impl asset::Asset for MultiResMesh {}
