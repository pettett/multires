use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use bevy_ecs::component::Component;
use common::asset::Asset;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    partitions: BufferGroup<2>,
    num_indices: u32,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<'a>(&'a self, state: &'a Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
        // 1.

        render_pass.set_pipeline(state.render_pipeline());

        render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        render_pass.set_bind_group(1, self.partitions.bind_group(), &[]);

        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);

        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }

    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let partitions = BufferGroup::create_plural_storage(
            &[&asset.clusters[..], &asset.clusters2[..]],
            instance.device(),
            &instance.partition_bind_group_layout(),
            Some("Partition Buffer"),
        );

        let vertex_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&asset.verts[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let index_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U32 Index Buffer"),
                    contents: bytemuck::cast_slice(&asset.layer_1_indices[..]),
                    usage: wgpu::BufferUsages::INDEX,
                });
        let index_format = wgpu::IndexFormat::Uint32;
        let num_indices = asset.layer_1_indices.len() as u32;

        // Update the value stored in this mesh
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices: num_indices,
            partitions,
            index_format,
        }
    }
}
