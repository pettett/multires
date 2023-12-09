use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    sync::Arc,
};

use bevy_ecs::{component::Component, entity::Entity, system::Query, world::World};
use common::{asset::Asset, MultiResMesh};
use common_renderer::components::camera::Camera;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct ClusterComponent {
    // Range into the index array that this submesh resides
    id: usize,
    index_offset: u32,
    index_count: u32,
    partitions: BufferGroup<2>,
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
    pub co_parent: Option<Entity>,
    pub model: BufferGroup<1>,
}
#[repr(C)]
#[derive(crevice::std430::AsStd430, bytemuck::Pod, Clone, Copy, bytemuck::Zeroable)]
pub struct ClusterData {
    // Range into the index array that this submesh resides
    index_offset: u32,
    index_count: u32,
    error: f32,
    //center: Vec3,
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    parent0: i32,
    parent1: i32,
    co_parent: i32,
}

#[derive(PartialEq, Clone)]
pub enum ErrorMode {
    PointDistance { camera_point: Vec3, cam: Camera },
    MaxError,
    ExactLayer,
}
impl Debug for ErrorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PointDistance { .. } => f.debug_struct("PointDistance").finish(),
            Self::MaxError => f.debug_struct("MaxError").finish(),
            Self::ExactLayer => f.debug_struct("ExactLayer").finish(),
        }
    }
}

#[derive(Component)]
pub struct MultiResMeshComponent {
    index_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    cluster_count: u32,
    pub can_draw_buffer: BufferGroup<1>,
    pub cluster_data_buffer: BufferGroup<1>,
    pub debug_staging_buffer: wgpu::Buffer,
    pub staging_buffer_size: usize,
    model: BufferGroup<1>,
    index_format: wgpu::IndexFormat,
    asset: MultiResMesh,
    pub error_calc: ErrorMode,
    pub error_target: f32,
    pub focus_part: usize,
    pub freeze: bool,
}

impl ClusterComponent {
    pub fn co_error(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
    ) -> f32 {
        self.error(mesh).min(match self.co_parent {
            Some(co_parent) => submeshes.get(co_parent).unwrap().1.error(mesh),
            None => f32::MAX, // Leaf nodes have no co parent, as have no children
        })
    }

    pub fn error(&self, mesh: &MultiResMeshComponent) -> f32 {
        match &mesh.error_calc {
            ErrorMode::PointDistance {
                camera_point, cam, ..
            } => {
                // Max error we can have before mesh is not suitable to draw

                let distance = self.center.distance(*camera_point).max(cam.znear());

                self.error * self.radius / distance
            }
            ErrorMode::MaxError => self.error,
            ErrorMode::ExactLayer => self.layer as _, //FIXME:
        }
    }

    // Issue - parents of a group may dissagree on if to draw, if they have differing errors due to being in different groups.

    // Solution Idea - merge the parents into a single node after calculating view dependant error,
    // taking the smaller of the two's errors to ensure other things in the group can still be drawn at the exact same time.
    // (Group != siblings, but everything in a group and every sibling must *both* be in agreement on whether to draw)

    pub fn error_within_bounds(&self, mesh: &MultiResMeshComponent) -> bool {
        self.error(mesh) < mesh.error_target
    }

    pub fn should_draw(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
    ) -> bool {
        //TODO: Give each partition a unique parent. This parent should be a

        // a partition is remeshed, then repartitioned inside the same group, so how can it have a parent?
        // remeshing can make two partitions where there was once one

        // when grouping, each partition is assigned to a unique group that is demeshed
        // so each partition has a parent group of partitions that form the same bound
        // this is computed as a group such that one of the original member was us,

        // Each demeshed item can look at a unique bound to compare against, maybe this is what we want

        // With no parents, assume infinite error
        let mut parent_error_too_large = self.parents.len() != 0;

        // For each parent, if any are within bounds, they all will be
        for &dep in &self.parents {
            if submeshes.get(dep).unwrap().1.error_within_bounds(mesh) {
                parent_error_too_large = false;
            }
        }

        //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
        // and we should draw the child

        parent_error_too_large
            && (self.error_within_bounds(mesh)
                || match self.co_parent {
                    Some(co_parent) => submeshes
                        .get(co_parent)
                        .unwrap()
                        .1
                        .error_within_bounds(mesh),
                    None => true, // Leaf nodes have no co parent, as have no children
                })
    }

    pub fn r_should_draw(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
    ) -> bool {
        let should_draw = self.should_draw(submeshes, mesh);

        for g in &self.group {
            if should_draw != submeshes.get(*g).unwrap().1.should_draw(submeshes, mesh) {
                println!("WARNING: Not all members of a group are drawing")
            }
        }

        // FIXME: probably do this on the error graph
        //self.is_monotonic(submeshes, mesh);

        should_draw
    }
}

impl MultiResMeshComponent {
    pub fn compute_pass<'a>(
        &'a self,
        renderer: &'a Renderer,
        submeshes: &'a Query<(Entity, &ClusterComponent)>,
        render_pass: &mut wgpu::ComputePass<'a>,
    ) {
        render_pass.set_pipeline(&renderer.compute_pipeline);
        render_pass.set_bind_group(0, self.can_draw_buffer.bind_group(), &[]);
        render_pass.set_bind_group(1, self.cluster_data_buffer.bind_group(), &[]);
        render_pass.dispatch_workgroups(self.cluster_count, 1, 1);
    }

    pub fn render_pass<'a>(
        &'a self,
        renderer: &'a Renderer,
        submeshes: &'a Query<(Entity, &ClusterComponent)>,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        // 1.
        {
            for i in 0..self.cluster_count {
                let r = self.debug_staging_buffer.slice(..).get_mapped_range();
                print!(
                    "{}|",
                    000 + ((r[(i as usize) * 4 + 0] as u32) << 0)
                        + ((r[(i as usize) * 4 + 1] as u32) << 8)
                        + ((r[(i as usize) * 4 + 2] as u32) << 16)
                        + ((r[(i as usize) * 4 + 3] as u32) << 26)
                );
            }
            println!("");
        }

        render_pass.set_bind_group(0, renderer.camera_buffer().bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);

        render_pass.set_bind_group(2, self.model.bind_group(), &[]);

        render_pass.set_pipeline(renderer.render_pipeline());
        for (_, submesh) in submeshes.iter() {
            if submesh.r_should_draw(submeshes, self) {
                //let submesh = submeshes.get(*s).unwrap();

                //render_pass.set_pipeline(renderer.render_pipeline_wire());
                //
                //render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);
                //render_pass.set_index_buffer(submesh.indices.slice(..), self.index_format);
                //
                //render_pass.draw_indexed(0..submesh.index_count, 0, 0..1);

                render_pass.set_bind_group(1, submesh.partitions.bind_group(), &[]);

                render_pass.draw_indexed(
                    submesh.index_offset..submesh.index_offset + submesh.index_count,
                    0,
                    0..1,
                );
            }
            //render_pass.set_pipeline(state.render_pipeline_wire());
            //render_pass.draw_indexed(0..submesh.num_indices, 0, 0..1);
        }

        // Draw bounds gizmos
        return;

        render_pass.set_vertex_buffer(0, renderer.sphere_gizmo.verts.slice(..));

        render_pass.set_pipeline(renderer.render_pipeline_wire());

        for (_, submesh) in submeshes.iter() {
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

    pub fn submesh_error_graph(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
    ) -> petgraph::prelude::Graph<f32, ()> {
        let mut graph = petgraph::Graph::new();

        let mut nodes = HashMap::new();

        for (e, s) in submeshes.iter() {
            nodes.insert(e, graph.add_node(s.co_error(submeshes, &self)));
        }

        for (e, s) in submeshes.iter() {
            let n = nodes[&e];

            for p in &s.parents {
                let p_n = nodes[p];

                graph.add_edge(n, p_n, ());
            }

            // if let Some(co_parent) = &s.co_parent {
            //     graph.add_edge(n, nodes[co_parent], ());
            // }
        }
        graph
    }

    pub fn load_mesh(instance: Arc<Instance>, world: &mut World) {
        let asset = common::MultiResMesh::load().unwrap();

        let mut clusters_per_lod: Vec<Vec<Entity>> = Vec::new();

        let mut all_clusters = Vec::new();
        let mut all_clusters_data = Vec::new();
        let mut indices = Vec::new();

        for (level, r) in asset.lods.iter().enumerate() {
            println!("Loading layer {level}:");
            let mut clusters = Vec::new();

            for (part, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let index_count = submesh.indices.len() as u32;

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
                    &[Mat4::from_translation(submesh.saturated_sphere.center())
                        * Mat4::from_scale(Vec3::ONE * submesh.saturated_sphere.radius())],
                    wgpu::BufferUsages::UNIFORM,
                    instance.device(),
                    instance.model_bind_group_layout(),
                    Some("Uniform Debug Model Buffer"),
                );

                let cluster = ClusterComponent {
                    id: all_clusters_data.len(),
                    partitions: info_buffer,
                    index_offset: indices.len() as u32,
                    index_count,
                    layer: level,
                    part,
                    center: submesh.saturated_sphere.center(),
                    error: submesh.error,
                    model,
                    radius: submesh.saturated_sphere.radius(),
                    //    children: vec![],
                    parents: vec![],
                    group: vec![],
                    co_parent: None,
                };

                all_clusters_data.push(ClusterData {
                    index_offset: cluster.index_count,
                    index_count: cluster.index_count,
                    error: cluster.error,
                    parent0: -1,
                    parent1: -1,
                    co_parent: -1,
                });

                // Push to indices *after* recording the offset above
                indices.extend_from_slice(&submesh.indices);

                let e = world.spawn(cluster).id();

                clusters.push(e);
                all_clusters.push(e);
            }
            clusters_per_lod.push(clusters);
        }

        // Search for [dependencies], group members, and dependants
        for (level, partition_entities) in clusters_per_lod.iter().enumerate() {
            for (i_partition, &partition) in partition_entities.iter().enumerate() {
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
                    .map(|child_partition| clusters_per_lod[level - 1][*child_partition])
                    .collect();

                for &child in &child_partitions {
                    // only the partitions with a shared boundary should be listed as dependants

                    world
                        .get_mut::<ClusterComponent>(child)
                        .unwrap()
                        .parents
                        .push(partition);
                }
            }

            // Search for Co-parents
            for &cluster in &all_clusters {
                let parents = world
                    .get::<ClusterComponent>(cluster)
                    .unwrap()
                    .parents
                    .clone();

                if parents.len() == 2 {
                    // Set co-parent pointers to each other
                    let id0 = world.get::<ClusterComponent>(parents[0]).unwrap().id as _;
                    let id1 = world.get::<ClusterComponent>(parents[1]).unwrap().id as _;

                    let mut p1 = world.get_mut::<ClusterComponent>(parents[1]).unwrap();
                    p1.co_parent = Some(parents[0]);
                    all_clusters_data[p1.id].co_parent = id0;

                    let mut p0 = world.get_mut::<ClusterComponent>(parents[0]).unwrap();
                    p0.co_parent = Some(parents[1]);
                    all_clusters_data[p0.id].co_parent = id1;

                    // Set parent pointers for ourself

                    let this = world.get::<ClusterComponent>(cluster).unwrap();
                    all_clusters_data[this.id].parent0 = id0;
                    all_clusters_data[this.id].parent1 = id1;
                } else if parents.len() != 0 {
                    panic!("Non-binary parented DAG, not currently (or ever) supported");
                }
            }
        }

        let index_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("U32 Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

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

        // let mut sizer = crevice::std430::Sizer::new();
        // for _ in 0..all_clusters.len() {
        //     sizer.add::<ClusterData>();
        // }

        //let mut cluster_buffer_data = vec![0; sizer.len()];

        let cluster_data_buffer_size = std::mem::size_of_val(&all_clusters_data[..]);

        //let mut writer = crevice::std430::Writer::new(&mut cluster_buffer_data);
        //for data in &all_clusters_data {
        //    writer.write(data).unwrap();
        //}

        let cluster_can_draw = vec![1i32; all_clusters_data.len()];

        let cluster_data_buffer = BufferGroup::create_single(
            &all_clusters_data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            instance.device(),
            instance.read_compute_bind_group_layout(),
            Some("cluster_buffer_data"),
        );

        let compute_can_draw_buffer = BufferGroup::create_single(
            &cluster_can_draw,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            instance.device(),
            instance.write_compute_bind_group_layout(),
            Some("cluster_can_draw"),
        );

        let staging_buffer_size = std::mem::size_of_val(&cluster_can_draw[..]);
        let debug_staging_buffer = instance.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Buffer"),
            size: staging_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Update the value stored in this mesh
        world.spawn(MultiResMeshComponent {
            vertex_buffer,
            index_buffer,
            index_format,
            cluster_data_buffer,
            can_draw_buffer: compute_can_draw_buffer,
            debug_staging_buffer,
            staging_buffer_size,
            cluster_count: all_clusters_data.len() as _,
            model,
            asset,
            error_calc: ErrorMode::ExactLayer,
            error_target: 0.5,
            focus_part: 0,
            freeze: false,
        });
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.asset
    }
}
