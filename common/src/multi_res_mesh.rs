use std::mem;

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

impl MultiResMesh {
    pub fn vertex_adapter(&self) -> meshopt::VertexDataAdapter {
        meshopt::VertexDataAdapter::new(
            bytemuck::cast_slice(&self.verts),
            mem::size_of::<MeshVert>(),
            0,
        )
        .unwrap()
    }
}
