use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use bevy_ecs::component::Component;
use common::asset::Asset;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

struct ReMesh {
    indices: wgpu::Buffer,
    partitions: BufferGroup<2>,
    num_indices: u32,
}

#[derive(Component)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    remeshes: Vec<ReMesh>,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<'a>(&'a self, state: &'a Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
        // 1.

        render_pass.set_pipeline(state.render_pipeline());

        render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let remesh = &self.remeshes[state.mesh_index.min(self.remeshes.len() - 1)];

        render_pass.set_bind_group(1, remesh.partitions.bind_group(), &[]);
        render_pass.set_index_buffer(remesh.indices.slice(..), self.index_format);
        render_pass.draw_indexed(0..remesh.num_indices, 0, 0..1);
    }

    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let mut remeshes = Vec::new();

        for r in asset.layers {
            let partitions = BufferGroup::create_plural_storage(
                &[&r.partitions[..], &r.groups[..]],
                instance.device(),
                &instance.partition_bind_group_layout(),
                Some("Partition Buffer"),
            );

            let indices = instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U32 Index Buffer"),
                    contents: bytemuck::cast_slice(&r.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

            let num_indices = r.indices.len() as u32;

            remeshes.push(ReMesh {
                indices,
                partitions,
                num_indices,
            })
        }

        let vertex_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&asset.verts[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let index_format = wgpu::IndexFormat::Uint32;

        // Update the value stored in this mesh
        Mesh {
            vertex_buffer,
            remeshes,
            index_format,
        }
    }
}
