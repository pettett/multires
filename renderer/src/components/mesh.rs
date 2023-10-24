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
    index_buffer0: wgpu::Buffer,
    index_buffer1: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    partitions0: BufferGroup<2>,
    partitions1: BufferGroup<2>,
    num_indices0: u32,
    num_indices1: u32,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<'a>(&'a self, state: &'a Renderer, render_pass: &mut wgpu::RenderPass<'a>) {
        // 1.

        render_pass.set_pipeline(state.render_pipeline());

        render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        if state.apply_remesh {
            render_pass.set_bind_group(1, self.partitions0.bind_group(), &[]);
            render_pass.set_index_buffer(self.index_buffer0.slice(..), self.index_format);
            render_pass.draw_indexed(0..self.num_indices0, 0, 0..1);
        } else {
            render_pass.set_bind_group(1, self.partitions1.bind_group(), &[]);
            render_pass.set_index_buffer(self.index_buffer1.slice(..), self.index_format);
            render_pass.draw_indexed(0..self.num_indices1, 0, 0..1);
        }
    }

    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let partitions0 = BufferGroup::create_plural_storage(
            &[&asset.layers[0].partitions[..], &asset.layers[0].groups[..]],
            instance.device(),
            &instance.partition_bind_group_layout(),
            Some("Partition Buffer"),
        );

        let partitions1 = BufferGroup::create_plural_storage(
            &[&asset.layers[1].partitions[..], &asset.layers[1].groups[..]],
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

        let index_buffer0 =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U32 Index Buffer"),
                    contents: bytemuck::cast_slice(&asset.layers[0].indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

        let index_buffer1 =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U32 Index Buffer"),
                    contents: bytemuck::cast_slice(&asset.layers[1].indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

        let index_format = wgpu::IndexFormat::Uint32;
        let num_indices0 = asset.layers[0].indices.len() as u32;
        let num_indices1 = asset.layers[1].indices.len() as u32;

        // Update the value stored in this mesh
        Mesh {
            vertex_buffer,
            index_buffer0,
            index_buffer1,
            partitions0,
            partitions1,
            num_indices0,
            num_indices1,
            index_format,
        }
    }
}
