use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use bevy_ecs::component::Component;
use common::asset::Asset;
use gltf::mesh::util::ReadIndices;
use vulkano::{
    buffer::{BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::AutoCommandBufferBuilder,
    memory::allocator::{AllocationCreateInfo, MemoryUsage},
};
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct Mesh {
    vertex_buffer: Subbuffer<[[f32; 3]]>,
    index_buffer: Subbuffer<[u32]>,
    //partitions: BufferGroup<2>,
    num_indices: u32,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<L>(&self, state: &Renderer, render_pass: &mut AutoCommandBufferBuilder<L>) {
        // 1.

        //render_pass.set_pipeline(state.render_pipeline());

        //render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        //render_pass.set_bind_group(1, self.partitions.bind_group(), &[]);

        render_pass
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .bind_index_buffer(self.index_buffer.clone())
            .draw_indexed(self.num_indices, 1, 0, 0, 0)
            .unwrap();
    }

    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        //let partitions = BufferGroup::create_plural_storage(
        //    &[&asset.clusters[..], &asset.clusters2[..]],
        //    instance.device(),
        //    &instance.partition_bind_group_layout(),
        //    Some("Partition Buffer"),
        //);

        let (document, buffers, images) =
            gltf::import(asset.name).expect("Torus import should work");

        let mesh = document.meshes().next().unwrap();
        let prim = mesh.primitives().next().unwrap();

        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

        let iter = reader.read_positions().unwrap();
        let verts: Vec<[f32; 3]> = iter.collect();

        let vertex_buffer = vulkano::buffer::Buffer::from_iter(
            instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            verts,
        )
        .unwrap();

        // let vertex_buffer =
        //     instance
        //         .device()
        //         .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //             label: Some("Vertex Buffer"),
        //             contents: bytemuck::cast_slice(&verts[..]),
        //             usage: wgpu::BufferUsages::VERTEX,
        //         });

        let indicies: Vec<u32> = match reader.read_indices() {
            Some(ReadIndices::U16(iter)) => iter.map(|i| i as _).collect(),
            Some(ReadIndices::U32(iter)) => iter.collect(),
            _ => panic!("No indices"),
        };
        let num_indices = indicies.len() as u32;
        let index_buffer = vulkano::buffer::Buffer::from_iter(
            instance.memory_allocator(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            indicies,
        )
        .unwrap();

        // Update the value stored in this mesh
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices, //partitions,
        }
    }
}
