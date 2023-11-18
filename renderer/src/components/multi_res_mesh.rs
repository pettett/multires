use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, system::Query, world::World};
use common::{asset::Asset, MultiResMesh};
use common_renderer::components::camera::Camera;
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
    pub radius: f32,
    // Partitions in the layer below (higher resolution) that this is not compatible with
    //pub children: Vec<Entity>,
    // Partitions in the layer above (lower resolution) this is not compatible with
    pub parents: Vec<Entity>,
    pub group: Vec<Entity>,
    pub model: BufferGroup<1>,
}

impl SubMeshComponent {
    pub fn error_within_bounds(&self, mesh: &MultiResMeshComponent) -> bool {
        match &mesh.error_calc {
            ErrorMode::PointDistance {
                camera_point,
                target_error,
                cam,
            } => {
                // Max error we can have before mesh is not suitable to draw

                let mut distance = self.center.distance(*camera_point);

                if distance < cam.znear() {
                    distance = cam.znear()
                }
                let error = self.error / distance;

                error < *target_error
            }
            ErrorMode::MaxError { error } => self.error < *error,
            ErrorMode::ExactLayer { layer } => self.layer == *layer, //FIXME:
        }
    }

    pub fn is_monotonic(&self, submeshes: &Query<&SubMeshComponent>, mesh: &MultiResMeshComponent) {
        if let Some(dep) = self.parents.get(0) {
            let parent = submeshes.get(*dep).unwrap();
            // for dep in &self.dependences {
            //     let o = submeshes.get(*dep).unwrap();
            //     // How can this be true?
            //     // Do we actually compare against the 'child'? that is a well formed item, but renders in wrong level
            //     // children should be spread between groups no?
            //     assert_eq!(
            //         o.center, child.center,
            //         "All children should be in the same group"
            //     )
            // }

            //if self.center.distance(parent.center) + self.radius > parent.radius {
            //    println!("WARNING: Child's bounds extends outside of parents");
            //}

            if self.radius > parent.radius {
                //println!("WARNING: Non monotonic const error detected - error within bounds when child's error is not, but child's abs error should be lower");
            }

            if !self.error_within_bounds(mesh) && parent.error_within_bounds(mesh) {
                //    println!("WARNING: Non monotonic error detected - error within bounds when child's error is not, but child's projected error should be lower");
            }

            parent.is_monotonic(submeshes, mesh);
        }
    }

    pub fn should_draw(
        &self,
        submeshes: &Query<&SubMeshComponent>,
        mesh: &MultiResMeshComponent,
    ) -> bool {
        //TODO: Give each partition a unique parent. This parent should be a

        // a partition is remeshed, then repartitioned inside the same group, so how can it have a parent?
        // remeshing can make two partitions where there was once one

        // when grouping, each partition is assigned to a unique group that is demeshed
        // so each partition has a parent group of partitions that form the same bound
        // this is computed as a group such that one of the original member was us,

        // Each demeshed item can look at a unique bound to compare agaisnt, maybe this is what we want

        // With no parents, assume infinite error
        let mut parent_error_too_large = self.parents.len() == 0;

        for &dep in &self.parents {
            if !submeshes.get(dep).unwrap().error_within_bounds(mesh) {
                parent_error_too_large = true;
            }
        }

        //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
        // and we should draw the child

        self.error_within_bounds(mesh) && parent_error_too_large
    }

    pub fn r_should_draw(
        &self,
        submeshes: &Query<&SubMeshComponent>,
        mesh: &MultiResMeshComponent,
    ) -> bool {
        let should_draw = self.should_draw(submeshes, mesh);

        for g in &self.group {
            if should_draw != submeshes.get(*g).unwrap().should_draw(submeshes, mesh) {
                println!("WARNING: Not all members of a group are drawing")
            }
        }

        self.is_monotonic(submeshes, mesh);

        should_draw
    }
}
#[derive(Debug, PartialEq)]
pub enum ErrorMode {
    PointDistance {
        camera_point: Vec3,
        target_error: f32,
        cam: Camera,
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

        for submesh in submeshes.iter() {
            if submesh.r_should_draw(submeshes, self) {
                //let submesh = submeshes.get(*s).unwrap();

                render_pass.set_pipeline(renderer.render_pipeline_wire());

                render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
                render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);

                render_pass.draw_indexed(0..submesh.index_count, 0, 0..1);

                render_pass.set_pipeline(renderer.render_pipeline());

                render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
                render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);

                render_pass.draw_indexed(0..submesh.index_count, 0, 0..1);
            }
            //render_pass.set_pipeline(state.render_pipeline_wire());
            //render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
        }

        // Draw bounds gizmos
        return;

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
        let mut partitions_per_lod: Vec<Vec<Entity>> = Vec::new();

        for (level, r) in asset.lods.iter().enumerate() {
            println!("Loading layer {level}:");
            let mut partitions = Vec::new();

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

                let info_buffer = BufferGroup::create_plural_storage(
                    &[
                        &[part as i32, level as i32, submesh.debug_group as i32],
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
                    partitions: info_buffer,
                    index_count: num_indices,
                    layer: level,
                    part,
                    center: submesh.saturated_sphere.center(),
                    error: submesh.tight_sphere.radius(),
                    model,
                    radius: submesh.saturated_sphere.radius(),
                    //    children: vec![],
                    parents: vec![],
                    group: vec![],
                };
                let e = world.spawn(sub).id();
                if vis {
                    submeshes.insert(e);
                }
                partitions.push(e);
            }
            partitions_per_lod.push(partitions);
            vis = false;
        }

        // Search for [dependencies], group members, and dependants
        for (level, partition_entities) in partitions_per_lod.iter().enumerate() {
            for (i_partition, partition) in partition_entities.iter().enumerate() {
                let i_partition_group = asset.lods[level].partitions[i_partition].group_index;

                assert!(asset.lods[level].groups[i_partition_group]
                    .partitions
                    .contains(&i_partition));

                let Some(i_partition_child_group) =
                    asset.lods[level].partitions[i_partition].child_group_index
                else {
                    continue;
                };

                let child_partitions: Vec<_> = asset.lods[level - 1].groups
                    [i_partition_child_group]
                    .partitions
                    .iter()
                    .map(|child_partition| partitions_per_lod[level - 1][*child_partition])
                    .collect();

                for child in &child_partitions {
                    // only the partitions with a shared boundary should be listed as dependants

                    world
                        .get_mut::<SubMeshComponent>(*child)
                        .unwrap()
                        .parents
                        .push(*partition);
                }

                //world.get_mut::<SubMeshComponent>(*ent).unwrap().dependences = dependences;
                //let e = world.get::<SubMeshComponent>(*ent).unwrap();

                //for d in &e.dependences {
                //    let dep = world.get::<SubMeshComponent>(*d).unwrap();
                //    // A dependancy is a higher res mesh we are derived from, it should always have a lower error
                //    assert!(dep.error < e.error);
                //}
            }

            // Groups TODO:
            //for (group, info) in asset.lods[layer].groups.iter().enumerate() {
            //    for part0 in &info.partitions {
            //        let ent0 = partition_entities[part0];
            //        for part1 in &info.partitions {
            //            if part0 != part1 {
            //                let ent1 = partition_entities[part1];
            //
            //                world
            //                    .get_mut::<SubMeshComponent>(ent0)
            //                    .unwrap()
            //                    .group
            //                    .push(ent1);
            //                world
            //                    .get_mut::<SubMeshComponent>(ent1)
            //                    .unwrap()
            //                    .group
            //                    .push(ent0);
            //            }
            //        }
            //    }
            //}
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
