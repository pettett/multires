use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use bevy_ecs::component::Component;
use common::asset::Asset;
use gltf::mesh::util::ReadIndices;
use wgpu::util::DeviceExt;

use crate::core::{Buffer, Instance, Renderer};

#[derive(Component)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    partitions: Buffer,
    num_indices: u32,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<'a>(&'a self, state: &'a Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
        // 1.

        render_pass.set_pipeline(state.render_pipeline());

        render_pass.set_bind_group(0, state.camera_bind_group(), &[]);
        render_pass.set_bind_group(1, self.partitions.bind_group(), &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);

        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }

    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let partitions = Buffer::create_storage(
            &asset.clusters[..],
            instance.device(),
            Some("Partition Buffer"),
            &instance.partition_bind_group_layout(),
        );

        let (document, buffers, images) =
            gltf::import("../assets/torus_low.glb").expect("Torus import should work");

        let mesh = document.meshes().next().unwrap();
        let prim = mesh.primitives().next().unwrap();

        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

        let iter = reader.read_positions().unwrap();
        let verts: Vec<[f32; 3]> = iter.collect();

        let vertex_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&verts[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let (index_buffer, index_format, num_indices) = match reader.read_indices() {
            Some(ReadIndices::U16(iter)) => {
                let indicies: Vec<u16> = iter.collect();

                (
                    instance
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("U16 Index Buffer"),
                            contents: bytemuck::cast_slice(&indicies[..]),
                            usage: wgpu::BufferUsages::INDEX,
                        }),
                    wgpu::IndexFormat::Uint16,
                    indicies.len(),
                )
            }
            Some(ReadIndices::U32(iter)) => {
                let indicies: Vec<u32> = iter.collect();

                (
                    instance
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("U32 Index Buffer"),
                            contents: bytemuck::cast_slice(&indicies[..]),
                            usage: wgpu::BufferUsages::INDEX,
                        }),
                    wgpu::IndexFormat::Uint32,
                    indicies.len(),
                )
            }
            _ => panic!("No indices"),
        };
        // Update the value stored in this mesh
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices: num_indices as u32,
            partitions,
            index_format,
        }
    }
}
