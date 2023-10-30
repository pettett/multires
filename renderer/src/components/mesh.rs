use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, system::Query, world::World};
use common::{asset::Asset, MultiResMesh};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct SubMeshComponent {
    indices: wgpu::Buffer,
    partitions: BufferGroup<2>,
    num_indices: u32,
    pub layer: usize,
    pub part: usize,
    pub error: f32,
    pub center: Vec3,
    // Partitions in the layer above that this is not compatible with
    pub dependences: Vec<Entity>,
    // Partitions in the layer below (lower resolution) this is not compatible with
    //pub dependants: Vec<Entity>,
    pub group: Vec<Entity>,
}

impl SubMeshComponent {
    pub fn error_within_bounds(&self, mesh: &Mesh) -> bool {
        let max_err = match &mesh.error_calc {
            ErrorMode::PointDistance {
                camera_point,
                error_falloff,
            } => {
                // Max error we can have before mesh is not suitable to draw
                self.center.distance_squared(*camera_point) / error_falloff
            }
            ErrorMode::MaxError { error } => *error,
        };

        self.error < max_err
    }
}
#[derive(Debug, PartialEq)]
pub enum ErrorMode {
    PointDistance {
        camera_point: Vec3,
        error_falloff: f32,
    },
    MaxError {
        error: f32,
    },
}
#[derive(Component)]
pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_format: wgpu::IndexFormat,
    pub submeshes: HashSet<Entity>,
    asset: MultiResMesh,
    pub error_calc: ErrorMode,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl Mesh {
    pub fn add_submesh(&mut self, ent: Entity, submeshes: &Query<&SubMeshComponent>) {
        self.remove_submesh_upwards(&ent, submeshes);
        self.remove_submesh_downwards(&ent, submeshes);

        let s = submeshes.get(ent).unwrap();

        self.submeshes.insert(ent);
        for g in &s.group {
            if self.submeshes.insert(*g) {
                println!("Added group member");
            }
        }

        // Need to remove them down the chain also

        // Other meshes in the group need to be enabled to fill any holes
    }

    // disable a submesh, by recursively disabling dependants above too
    fn remove_submesh_upwards(&mut self, ent: &Entity, submeshes: &Query<&SubMeshComponent>) {
        if self.submeshes.remove(ent) {
            println!("Disabled mesh above");
        }

        let s = submeshes.get(*ent).unwrap();
        todo!()
        //for dep in &s.dependences {
        //    // These must be disabled, recursively
        //    self.remove_submesh_upwards(dep, submeshes)
        //}
    }

    fn remove_submesh_downwards(&mut self, ent: &Entity, submeshes: &Query<&SubMeshComponent>) {
        if self.submeshes.remove(ent) {
            println!("Disabled mesh below");
        }

        let s = submeshes.get(*ent).unwrap();
        todo!()
        //for dep in &s.dependants {
        //    // These must be disabled, recursively
        //    self.remove_submesh_downwards(dep, submeshes)
        //}
    }

    pub fn render_pass<'a>(
        &'a self,
        state: &'a Renderer,
        submeshes: &'a Query<&SubMeshComponent>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        // 1.

        render_pass.set_bind_group(0, state.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        for submesh in submeshes.iter() {
            let draw_child = if let Some(dep) = submesh.dependences.get(0) {
                let child = submeshes.get(*dep).unwrap();

                for dep in &submesh.dependences {
                    let o = submeshes.get(*dep).unwrap();

                    assert_eq!(
                        o.center, child.center,
                        "All parents should be in the same group"
                    )
                }

                child.error_within_bounds(self)
            } else {
                false
            };
            //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
            // and we should draw the child
            if !submesh.error_within_bounds(self) && draw_child {
                //let submesh = submeshes.get(*s).unwrap();

                render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
                render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);

                render_pass.set_pipeline(state.render_pipeline_wire());
                render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
            }
            //render_pass.set_pipeline(state.render_pipeline_wire());
            //render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
        }
    }

    pub fn load_mesh(instance: Arc<Instance>, world: &mut World) {
        let asset = common::MultiResMesh::load().unwrap();

        let mut vis = true;
        let mut submeshes = HashSet::new();
        let mut partitions_per_layer: Vec<HashMap<usize, Entity>> = Vec::new();

        for (layer, r) in asset.layers.iter().enumerate() {
            let mut submeshes_per_partition = HashMap::default();

            for (part, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let indices =
                    instance
                        .device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("U32 Index Buffer"),
                            contents: bytemuck::cast_slice(&submesh.indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });

                let num_indices = submesh.indices.len() as u32;

                let part = part;

                let partitions = BufferGroup::create_plural_storage(
                    &[&[part as i32, layer as i32], &[0]],
                    instance.device(),
                    &instance.partition_bind_group_layout(),
                    Some("Partition Buffer"),
                );

                let sub = SubMeshComponent {
                    indices,
                    partitions,
                    num_indices,
                    layer,
                    part,
                    center: submesh.tight_sphere.center(),
                    error: submesh.error,
                    dependences: vec![],
                    //dependants: vec![],
                    group: vec![],
                };
                let e = world.spawn(sub).id();
                if vis {
                    submeshes.insert(e);
                }
                submeshes_per_partition.insert(part, e);
            }
            partitions_per_layer.push(submeshes_per_partition);
            vis = false;
        }

        // Search for [dependencies], group members, and dependants
        for (layer, ents) in partitions_per_layer.iter().enumerate() {
            for (part, ent) in ents.iter() {
                // dependancies
                if layer > 0 {
                    // unmap the grouping info from previous layer

                    if let Some(dep_group) = asset.layers[layer].dependant_partitions.get(part) {
                        let dependences: Vec<_> = asset.layers[layer - 1].partition_groups
                            [*dep_group]
                            .iter()
                            .map(|dep_part| partitions_per_layer[layer - 1][dep_part])
                            .collect();

                        //for d in &dependences {
                        //    world
                        //        .get_mut::<SubMeshComponent>(*d)
                        //        .unwrap()
                        //        .dependants
                        //        .push(*ent);
                        //}
                        world.get_mut::<SubMeshComponent>(*ent).unwrap().dependences = dependences;
                        let e = world.get::<SubMeshComponent>(*ent).unwrap();

                        for d in &e.dependences {
                            let dep = world.get::<SubMeshComponent>(*d).unwrap();
                            // A dependancy is a higher res mesh we are derived from, it should always have a lower error
                            assert!(dep.error < e.error);
                        }
                    }
                }
            }

            // Groups
            for (group, parts) in asset.layers[layer].partition_groups.iter().enumerate() {
                println!("{group} {parts:?}");

                for part0 in parts {
                    let ent0 = ents[part0];
                    for part1 in parts {
                        if part0 != part1 {
                            let ent1 = ents[part1];

                            world
                                .get_mut::<SubMeshComponent>(ent0)
                                .unwrap()
                                .group
                                .push(ent1);
                            world
                                .get_mut::<SubMeshComponent>(ent1)
                                .unwrap()
                                .group
                                .push(ent0);
                        }
                    }
                }
            }
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
            error_calc: ErrorMode::MaxError { error: 0.1 },
        });
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.asset
    }
}
