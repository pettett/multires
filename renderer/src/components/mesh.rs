use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use bevy_ecs::{component::Component, entity::Entity, query::Has, system::Query, world::World};
use common::{asset::Asset, MultiResMesh};
use wgpu::{util::DeviceExt, ShaderStages};

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct SubMesh {
    indices: wgpu::Buffer,
    partitions: BufferGroup<2>,
    num_indices: u32,
    pub layer: usize,
    pub part: i32,
    // Partitions in the layer above that this is not compatible with
    pub dependences: Vec<Entity>,
}
#[derive(Component)]

pub struct Visible();

#[derive(Component)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    pub submeshes: Vec<Entity>,
    asset: MultiResMesh,
    //pub remesh: usize,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn render_pass<'a>(
        &'a self,
        state: &'a Renderer,
        submeshes: &'a Query<&SubMesh>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        // 1.

        render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        for s in &self.submeshes {
            let submesh = submeshes.get(*s).unwrap();

            render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
            render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);

            render_pass.set_pipeline(state.render_pipeline());
            render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);

            //render_pass.set_pipeline(state.render_pipeline_wire());
            //render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
        }
    }

    pub fn load_mesh(instance: Arc<Instance>, world: &mut World) {
        let asset = common::MultiResMesh::load().unwrap();

        let mut vis = true;
        let mut submeshes = Vec::new();
        let mut all_ents: Vec<HashMap<i32, Entity>> = Vec::new();

        for (layer, r) in asset.layers.iter().enumerate() {
            let mut ents = HashMap::default();

            for (part, meshlet) in r.meshlets.iter().enumerate() {
                // Map index buffer to global vertex range
                let mut ind = meshlet.indices[..meshlet.index_count as _].to_vec();
                assert_eq!(ind.len() % 3, 0);
                for i in &mut ind {
                    *i = meshlet.vertices[*i as usize];
                }

                let indices =
                    instance
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("U32 Index Buffer"),
                            contents: bytemuck::cast_slice(&ind),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                let num_indices = ind.len() as u32;

                let part = part as i32;

                let partitions = BufferGroup::create_plural_storage(
                    &[&[part], &[0]],
                    instance.device(),
                    &instance.partition_bind_group_layout(),
                    Some("Partition Buffer"),
                );

                let dependences = if layer > 0 {
                    // unmap the grouping info from previous layer
                    if let Some(dep_group) = r.dependant_partitions.get(&(part as i32)) {
                        asset.layers[layer - 1].partition_groups[dep_group]
                            .iter()
                            .map(|part| all_ents[layer - 1][part])
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                let sub = SubMesh {
                    indices,
                    partitions,
                    num_indices,
                    layer,
                    part,

                    dependences,
                };
                let e = world.spawn(sub).id();
                if vis {
                    submeshes.push(e);
                }
                ents.insert(part, e);
            }
            all_ents.push(ents);
            vis = false;
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
        world.spawn(Mesh {
            vertex_buffer,
            index_format,
            asset,
            submeshes,
        });
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.asset
    }
}
