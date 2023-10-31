use std::sync::Arc;

use common::tri_mesh::TriMesh;
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance};

pub struct DebugMesh {
    verts: wgpu::Buffer,
    indices: wgpu::Buffer,
    model: BufferGroup<1>,
    index_count: usize,
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

        let model = BufferGroup::create_single(
            &[Mat4::IDENTITY],
            wgpu::BufferUsages::UNIFORM,
            instance.device(),
            instance.model_bind_group_layout(),
            Some("Uniform Debug Model Buffer"),
        );

        DebugMesh {
            verts,
            indices,
            model,
            index_count: mesh.indices.len(),
        }
    }
}
