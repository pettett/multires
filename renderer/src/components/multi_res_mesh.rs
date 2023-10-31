use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, system::Query, world::World};
use common::{asset::Asset, MultiResMesh};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct SubMeshComponent {
    indices: wgpu::Buffer,
    partitions: BufferGroup<2>,
    index_count: u32,
    pub layer: usize,
    pub part: usize,
    pub error: f32,
    pub center: Vec3,
    // Partitions in the layer above that this is not compatible with
    pub dependences: Vec<Entity>,
    // Partitions in the layer below (lower resolution) this is not compatible with
    //pub dependants: Vec<Entity>,
    pub group: Vec<Entity>,
    pub model: BufferGroup<1>,
}

impl SubMeshComponent {
    pub fn error_within_bounds(&self, mesh: &MultiResMeshComponent) -> bool {
        match &mesh.error_calc {
            ErrorMode::PointDistance {
                camera_point,
                error_falloff,
            } => {
                // Max error we can have before mesh is not suitable to draw
                self.error < self.center.distance_squared(*camera_point) / error_falloff
            }
            ErrorMode::MaxError { error } => self.error < *error,
            ErrorMode::ExactLayer { layer } => self.layer == *layer,
        }
    }

    pub fn should_draw(
        &self,
        submeshes: &Query<&SubMeshComponent>,
        mesh: &MultiResMeshComponent,
    ) -> bool {
        let draw_child = if let Some(dep) = self.dependences.get(0) {
            let child = submeshes.get(*dep).unwrap();

            for dep in &self.dependences {
                let o = submeshes.get(*dep).unwrap();

                assert_eq!(
                    o.center, child.center,
                    "All parents should be in the same group"
                )
            }

            child.error_within_bounds(mesh)
        } else {
            false
        };
        //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
        // and we should draw the child
        return !self.error_within_bounds(mesh) && draw_child;
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
    ExactLayer {
        layer: usize,
    },
}
#[derive(Component)]
pub struct MultiResMeshComponent {
    vertex_buffer: wgpu::Buffer,
    model: BufferGroup<1>,
    index_format: wgpu::IndexFormat,
    pub submeshes: HashSet<Entity>,
    asset: MultiResMesh,
    pub error_calc: ErrorMode,
    pub focus_part: usize,
    //puffin_ui : puffin_imgui::ProfilerUi,
}
impl MultiResMeshComponent {
    pub fn render_pass<'a>(
        &'a self,
        renderer: &'a Renderer,
        submeshes: &'a Query<&SubMeshComponent>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        // 1.

        render_pass.set_bind_group(0, renderer.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        render_pass.set_bind_group(2, self.model.bind_group(), &[]);

        render_pass.set_pipeline(renderer.render_pipeline());

        for submesh in submeshes.iter() {
            if submesh.should_draw(submeshes, self) {
                //let submesh = submeshes.get(*s).unwrap();

                render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
                render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);

                render_pass.draw_indexed(0..submesh.index_count, 0, 0..1);
            }
            //render_pass.set_pipeline(state.render_pipeline_wire());
            //render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
        }

        // Draw bounds gizmos

        render_pass.set_vertex_buffer(0, renderer.sphere_gizmo.verts.slice(..));

        render_pass.set_pipeline(renderer.render_pipeline_wire());

        for submesh in submeshes.iter() {
            if submesh.part == self.focus_part {
                if submesh.should_draw(submeshes, self) {
                    render_pass.set_index_buffer(
                        renderer.sphere_gizmo.indices.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    render_pass.set_bind_group(2, submesh.model.bind_group(), &[]);
                    render_pass.draw_indexed(0..renderer.sphere_gizmo.index_count, 0, 0..1);
                }
            }
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

                let partitions = BufferGroup::create_plural_storage(
                    &[
                        &[part as i32, layer as i32, submesh.debug_group as i32],
                        &[0],
                    ],
                    instance.device(),
                    &instance.partition_bind_group_layout(),
                    Some("Partition Buffer"),
                );

                let model = BufferGroup::create_single(
                    &[Mat4::from_translation(submesh.tight_sphere.center())
                        * Mat4::from_scale(Vec3::ONE * submesh.tight_sphere.radius())],
                    wgpu::BufferUsages::UNIFORM,
                    instance.device(),
                    instance.model_bind_group_layout(),
                    Some("Uniform Debug Model Buffer"),
                );

                let sub = SubMeshComponent {
                    indices,
                    partitions,
                    index_count: num_indices,
                    layer,
                    part,
                    center: submesh.tight_sphere.center(),
                    error: submesh.error,
                    model,
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

        let model = BufferGroup::create_single(
            &[Mat4::IDENTITY],
            wgpu::BufferUsages::UNIFORM,
            instance.device(),
            instance.model_bind_group_layout(),
            Some("Uniform Model Buffer"),
        );

        // Update the value stored in this mesh
        world.spawn(MultiResMeshComponent {
            vertex_buffer,
            index_format,
            model,
            asset,
            submeshes,
            error_calc: ErrorMode::ExactLayer { layer: 0 },
            focus_part: 0,
        });
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.asset
    }
}
