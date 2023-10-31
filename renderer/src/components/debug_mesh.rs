use std::sync::Arc;

use common::tri_mesh::TriMesh;
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance};

pub struct DebugMesh {
    pub verts: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,
}

impl DebugMesh {
    pub fn from_tris(instance: Arc<Instance>, mesh: &TriMesh) -> Self {
        let verts = instance
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.verts[..]),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let indices = instance
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh.indices[..]),
                usage: wgpu::BufferUsages::INDEX,
            });

        DebugMesh {
            verts,
            indices,
            index_count: mesh.indices.len() as _,
        }
    }
}
