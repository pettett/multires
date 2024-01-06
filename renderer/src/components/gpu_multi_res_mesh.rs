use std::{collections::HashMap, fmt::Debug, sync::Arc};

use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Query, Resource},
    world::World,
};
use common::{asset::Asset, MultiResMesh};
use common_renderer::components::{camera::Camera, transform::Transform};
use glam::{Mat4, Vec3};
use petgraph::visit::EdgeRef;
use wgpu::util::DeviceExt;

use crate::core::{BufferGroup, Instance, Renderer};

#[derive(Component)]
pub struct ClusterComponent {
    // Range into the index array that this submesh resides
    id: usize,
    index_offset: u32,
    index_count: u32,
    pub layer: usize,
    pub cluster_layer_idx: usize,
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
#[derive(
    crevice::std430::AsStd430, bytemuck::Pod, Clone, Copy, bytemuck::Zeroable, PartialEq, Debug,
)]
pub struct ClusterData {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    // Range into the index array that this submesh resides
    index_offset: u32,
    index_count: u32,
    error: f32,
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    parent0: i32,
    parent1: i32,
    co_parent: i32,

    // Pad alignment to 4 bytes
    radius: f32,
    _1: i32,
    _2: i32,

    _3: i32,
    _4: i32,
    _5: i32,
    _6: i32,
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

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct DrawData {
    model: Mat4,
    camera_pos: Vec3,
    error: f32,

    mode: u32,
    // Having a vec3 always brings fun antics like this
    _0: i32,
    _1: i32,
    _2: i32,
}

#[derive(Component)]
pub struct MultiResMeshComponent {
    name: String,
    cluster_count: u32,
    can_draw_buffer: BufferGroup<1>,
    //FIXME: This really should exist on the camera/part of uniform group
    draw_data_buffer: BufferGroup<1>,
    model: BufferGroup<1>,
    index_format: wgpu::IndexFormat,
    asset: Arc<MultiResMeshAsset>,
}

#[derive(Resource)]
pub struct MultiResMeshRenderer {
    pub error_calc: ErrorMode,
    pub error_target: f32,
    pub focus_part: usize,
    pub freeze: bool,
    pub show_wire: bool,
    pub show_solid: bool,
    pub show_bounds: bool,
}

/// Stores all immutable DAG data for a mesh, to be referenced by any number of instances.
pub struct MultiResMeshAsset {
    partition_buffer: BufferGroup<2>,
    vertex_buffer: wgpu::Buffer,
    cluster_count: u32,
    index_count: u32,
    cluster_data_real_error_group: BufferGroup<2>,
    cluster_data_layer_error_group: BufferGroup<2>,
    index_format: wgpu::IndexFormat,
    root_asset: MultiResMesh,
}

pub struct MultiResMeshDatabase {
    assets: HashMap<String, Arc<MultiResMeshAsset>>,
}

impl ClusterComponent {
    pub fn co_error(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
        renderer: &MultiResMeshRenderer,
    ) -> f32 {
        self.error(mesh, renderer).min(match self.co_parent {
            Some(co_parent) => submeshes.get(co_parent).unwrap().1.error(mesh, renderer),
            None => f32::MAX, // Leaf nodes have no co parent, as have no children
        })
    }

    pub fn error(&self, _mesh: &MultiResMeshComponent, renderer: &MultiResMeshRenderer) -> f32 {
        match &renderer.error_calc {
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

    // Issue - parents of a group may disagree on if to draw, if they have differing errors due to being in different groups.

    // Solution Idea - merge the parents into a single node after calculating view dependant error,
    // taking the smaller of the two's errors to ensure other things in the group can still be drawn at the exact same time.
    // (Group != siblings, but everything in a group and every sibling must *both* be in agreement on whether to draw)

    pub fn error_within_bounds(
        &self,
        mesh: &MultiResMeshComponent,
        renderer: &MultiResMeshRenderer,
    ) -> bool {
        self.error(mesh, renderer) < renderer.error_target
    }

    pub fn should_draw(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
        renderer: &MultiResMeshRenderer,
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
            if submeshes
                .get(dep)
                .unwrap()
                .1
                .error_within_bounds(mesh, renderer)
            {
                parent_error_too_large = false;
            }
        }

        //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
        // and we should draw the child

        parent_error_too_large
            && (self.error_within_bounds(mesh, renderer)
                || match self.co_parent {
                    Some(co_parent) => submeshes
                        .get(co_parent)
                        .unwrap()
                        .1
                        .error_within_bounds(mesh, renderer),
                    None => true, // Leaf nodes have no co parent, as have no children
                })
    }

    pub fn r_should_draw(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mesh: &MultiResMeshComponent,
        renderer: &MultiResMeshRenderer,
    ) -> bool {
        let should_draw = self.should_draw(submeshes, mesh, renderer);

        for g in &self.group {
            if should_draw
                != submeshes
                    .get(*g)
                    .unwrap()
                    .1
                    .should_draw(submeshes, mesh, renderer)
            {
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
        transform: &Transform,
        renderer: &'a Renderer,
        mesh_renderer: &MultiResMeshRenderer,
        camera_trans: &Transform,
        _submeshes: &'a Query<(Entity, &ClusterComponent)>,
        render_pass: &mut wgpu::ComputePass<'a>,
    ) {
        if mesh_renderer.freeze {
            return;
        }

        {
            renderer.queue().write_buffer(
                self.draw_data_buffer.buffer(),
                0,
                &bytemuck::cast_slice(&[DrawData {
                    model: transform.get_local_to_world(),
                    camera_pos: (*camera_trans.get_pos()).into(),
                    error: mesh_renderer.error_target,
                    mode: 0,
                    _0: 0,
                    _1: 0,
                    _2: 0,
                }]),
            );
        }

        render_pass.set_pipeline(&renderer.compute_pipeline);
        render_pass.set_bind_group(0, self.can_draw_buffer.bind_group(), &[]);

        let bind_1 = if mesh_renderer.error_calc == ErrorMode::ExactLayer {
            self.asset.cluster_data_layer_error_group.bind_group()
        } else {
            self.asset.cluster_data_real_error_group.bind_group()
        };

        render_pass.set_bind_group(1, bind_1, &[]);

        render_pass.set_bind_group(2, self.draw_data_buffer.bind_group(), &[]);

        render_pass.dispatch_workgroups(self.cluster_count, 1, 1);
    }

    pub fn render_pass<'a>(
        &'a self,
        renderer: &'a Renderer,
        submeshes: &'a Query<(Entity, &ClusterComponent)>,
        render_pass: &mut wgpu::RenderPass<'a>,
        mesh_renderer: &MultiResMeshRenderer,
    ) {
        //{
        //    for i in 0..self.staging_buffer_size {
        //        let r = self.debug_staging_buffer.slice(..).get_mapped_range();
        //        print!("{}|", r[i]);
        //    }
        //    println!("");
        //}

        render_pass.set_vertex_buffer(0, self.asset.vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.can_draw_buffer.buffer().slice(..),
            self.asset.index_format,
        );

        render_pass.set_bind_group(0, renderer.camera_buffer().bind_group(), &[]);
        render_pass.set_bind_group(1, self.asset.partition_buffer.bind_group(), &[]);
        render_pass.set_bind_group(2, self.model.bind_group(), &[]);

        if mesh_renderer.show_solid {
            render_pass.set_pipeline(renderer.render_pipeline());

            render_pass.draw_indexed(0..self.asset.index_count, 0, 0..1);
        }
        if mesh_renderer.show_wire {
            render_pass.set_pipeline(renderer.render_pipeline_wire());

            render_pass.draw_indexed(0..self.asset.index_count, 0, 0..1);
        }

        // Draw bounds gizmos
        if mesh_renderer.show_bounds {
            render_pass.set_vertex_buffer(0, renderer.sphere_gizmo.verts.slice(..));

            render_pass.set_pipeline(renderer.render_pipeline_wire());

            for (_, submesh) in submeshes.iter() {
                if submesh.cluster_layer_idx == mesh_renderer.focus_part {
                    if submesh.should_draw(submeshes, self, mesh_renderer) {
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
    }

    pub fn submesh_error_graph(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        renderer: &MultiResMeshRenderer,
    ) -> petgraph::prelude::Graph<f32, ()> {
        let mut graph = petgraph::Graph::new();

        let mut nodes = HashMap::new();

        for (e, s) in submeshes.iter() {
            nodes.insert(e, graph.add_node(s.co_error(submeshes, &self, renderer)));
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

    pub fn from_asset(
        name: String,
        instance: Arc<Instance>,
        world: &mut World,
        asset: Arc<MultiResMeshAsset>,
        trans: Transform,
    ) {
        let mut clusters_per_lod: Vec<Vec<Entity>> = Vec::new();

        let mut all_clusters = Vec::new();
        let mut indices = Vec::new();
        // Face indexed array
        let mut partitions = Vec::new();
        // Partition indexed array
        let mut groups = Vec::new();

        let mut cluster_idx = 0;

        for (level, r) in asset.asset().lods.iter().enumerate() {
            println!("Loading layer {level}:");
            let mut clusters = Vec::new();

            for (cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let index_count = submesh.indices.len() as u32;

                for _ in 0..(index_count / 3) {
                    partitions.push(cluster_idx as i32);
                }
                groups.push(submesh.debug_group as i32);

                cluster_idx += 1;

                let cluster_model = BufferGroup::create_single(
                    &[Mat4::from_translation(submesh.saturated_sphere.center())
                        * Mat4::from_scale(Vec3::ONE * submesh.saturated_sphere.radius())],
                    wgpu::BufferUsages::UNIFORM,
                    instance.device(),
                    &instance.model_bind_group_layout,
                    Some("Uniform Debug Model Buffer"),
                );

                let cluster = ClusterComponent {
                    id: all_clusters.len(),
                    //partitions: info_buffer,
                    index_offset: indices.len() as u32,
                    index_count,
                    layer: level,
                    cluster_layer_idx,
                    center: submesh.saturated_sphere.center(),
                    error: submesh.error,
                    model: cluster_model,
                    radius: submesh.saturated_sphere.radius(),
                    //    children: vec![],
                    parents: vec![],
                    group: vec![],
                    co_parent: None,
                };

                // Push to indices *after* recording the offset above
                indices.extend_from_slice(&submesh.indices);

                let e = world.spawn(cluster).id();

                clusters.push(e);
                all_clusters.push(e);
            }
            clusters_per_lod.push(clusters);
        }

        assert_eq!(partitions.len(), indices.len() / 3);
        // The last partition should be the largest
        assert_eq!(groups.len(), *partitions.last().unwrap() as usize + 1);

        // Search for [dependencies], group members, and dependants
        for (level, partition_entities) in clusters_per_lod.iter().enumerate() {
            for (i_partition, &partition) in partition_entities.iter().enumerate() {
                let i_partition_group =
                    asset.asset().lods[level].partitions[i_partition].group_index;

                assert!(asset.asset().lods[level].groups[i_partition_group]
                    .partitions
                    .contains(&i_partition));

                let Some(i_partition_child_group) =
                    asset.asset().lods[level].partitions[i_partition].child_group_index
                else {
                    continue;
                };

                let child_partitions: Vec<_> = asset.asset().lods[level - 1].groups
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
                match world.get::<ClusterComponent>(cluster).unwrap().parents[..] {
                    [p0, p1] => {
                        // Set co-parent pointers to each other

                        world.get_mut::<ClusterComponent>(p1).unwrap().co_parent = Some(p0);

                        world.get_mut::<ClusterComponent>(p0).unwrap().co_parent = Some(p1);
                    }
                    [] => (),
                    _ => panic!("Non-binary parented DAG, not currently (or ever) supported"),
                }
            }
        }

        let draw_data_buffer = BufferGroup::create_single(
            &[DrawData {
                model: trans.get_local_to_world(),
                camera_pos: Vec3::ZERO,
                error: 1.5,
                mode: 0,
                _0: 0,
                _1: 0,
                _2: 0,
            }],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            instance.device(),
            &instance.read_compute_bind_group_layout,
            Some("Draw Data Buffer"),
        );

        let index_format = wgpu::IndexFormat::Uint32;

        let model = BufferGroup::create_single(
            &[trans.get_local_to_world()],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
            instance.device(),
            &instance.model_bind_group_layout,
            Some("Mesh Uniform Model Buffer"),
        );

        // let mut sizer = crevice::std430::Sizer::new();
        // for _ in 0..all_clusters.len() {
        //     sizer.add::<ClusterData>();
        // }

        //let mut cluster_buffer_data = vec![0; sizer.len()];

        //let cluster_data_buffer_size = std::mem::size_of_val(&all_clusters_data[..]);

        //let mut writer = crevice::std430::Writer::new(&mut cluster_buffer_data);
        //for data in &all_clusters_data {
        //    writer.write(data).unwrap();
        //}

        let cluster_can_draw = vec![1i32; indices.len()];

        let compute_can_draw_buffer = BufferGroup::create_single(
            &cluster_can_draw,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::INDEX,
            instance.device(),
            &instance.write_compute_bind_group_layout,
            Some("cluster_can_draw"),
        );

        let staging_buffer_size = 100; // std::mem::size_of_val(&cluster_can_draw[..]);
        let _debug_staging_buffer = instance.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compute Buffer"),
            size: staging_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Update the value stored in this mesh
        world.spawn((
            MultiResMeshComponent {
                name,
                index_format,
                draw_data_buffer,
                can_draw_buffer: compute_can_draw_buffer,
                //debug_staging_buffer,
                //staging_buffer_size,
                cluster_count: all_clusters.len() as _,
                model,
                asset,
            },
            trans,
        ));
    }

    pub fn asset(&self) -> &MultiResMeshAsset {
        &self.asset
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

impl MultiResMeshAsset {
    pub fn load_mesh(instance: Arc<Instance>) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let mut clusters_per_lod = Vec::new();

        let mut all_clusters_data_real_error = Vec::new();
        let mut all_clusters_data_layer_error = Vec::new();
        let mut indices = Vec::new();
        // Face indexed array
        let mut partitions = Vec::new();
        // Partition indexed array
        let mut groups = Vec::new();

        let mut cluster_idx = 0;

        let mut dag = petgraph::Graph::new();

        for (level, r) in asset.lods.iter().enumerate() {
            println!("Loading layer {level}:");
            let mut cluster_nodes = Vec::new();

            for (_cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let index_count = submesh.indices.len() as u32;

                for _ in 0..(index_count / 3) {
                    partitions.push(cluster_idx as i32);
                }
                groups.push(submesh.debug_group as i32);

                cluster_idx += 1;

                // let model = BufferGroup::create_single(
                //     &[Mat4::from_translation(submesh.saturated_sphere.center())
                //         * Mat4::from_scale(Vec3::ONE * submesh.saturated_sphere.radius())],
                //     wgpu::BufferUsages::UNIFORM,
                //     instance.device(),
                //     instance.model_bind_group_layout(),
                //     Some("Uniform Debug Model Buffer"),
                // );

                all_clusters_data_real_error.push(ClusterData {
                    index_offset: indices.len() as u32,
                    index_count,
                    error: submesh.error,
                    center_x: submesh.saturated_sphere.center().x,
                    center_y: submesh.saturated_sphere.center().y,
                    center_z: submesh.saturated_sphere.center().z,
                    parent0: -1,
                    parent1: -1,
                    co_parent: -1,
                    radius: submesh.saturated_sphere.radius(),
                    _1: -1,
                    _2: -1,

                    _3: -1,
                    _4: -1,

                    _5: -1,
                    _6: -1,
                });

                cluster_nodes.push(dag.add_node(level));

                // Push to indices *after* recording the offset above
                indices.extend_from_slice(&submesh.indices);
            }
            clusters_per_lod.push(cluster_nodes);
        }

        assert_eq!(partitions.len(), indices.len() / 3);
        // The last partition should be the largest
        assert_eq!(groups.len(), *partitions.last().unwrap() as usize + 1);

        // Search for [dependencies], group members, and dependants
        for (level, cluster_nodes) in clusters_per_lod.iter().enumerate() {
            for (cluster_idx, &cluster_node_idx) in cluster_nodes.iter().enumerate() {
                let cluster_group_idx = asset.lods[level].partitions[cluster_idx].group_index;

                assert!(asset.lods[level].groups[cluster_group_idx]
                    .partitions
                    .contains(&cluster_idx));

                let Some(child_group_idx) =
                    asset.lods[level].partitions[cluster_idx].child_group_index
                else {
                    continue;
                };

                // To have a child group, level > 0

                let child_clusters: Vec<_> = asset.lods[level - 1].groups[child_group_idx]
                    .partitions
                    .iter()
                    .map(|&child_partition| clusters_per_lod[level - 1][child_partition])
                    .collect();

                // println!("{}", child_partitions.len());

                for &child in &child_clusters {
                    // only the partitions with a shared boundary should be listed as dependants

                    dag.add_edge(cluster_node_idx, child, ());
                }
            }
        }

        // petgraph_to_svg(
        //     &dag,
        //     "svg\\asset_dag.svg",
        //     &|_, _| String::new(),
        //     common::graph::GraphSVGRender::Directed {
        //         node_label: common::graph::Label::Weight,
        //     },
        // )
        // .unwrap();

        // Search for Co-parents
        for i in 0..all_clusters_data_real_error.len() {
            let parents = dag
                .neighbors_directed(
                    petgraph::graph::node_index(i),
                    petgraph::Direction::Incoming,
                )
                .collect::<Vec<_>>();

            match parents[..] {
                [p0, p1] => {
                    // Set co-parent pointers to each other. This work will be duplicated a lot of times, but it's convenient
                    let id0 = p0.index();
                    let id1 = p1.index();

                    all_clusters_data_real_error[id0].co_parent = id1 as _;
                    all_clusters_data_real_error[id1].co_parent = id0 as _;

                    // Set parent pointers for ourself
                    all_clusters_data_real_error[i].parent0 = (id0 as i32).min(id1 as i32);
                    all_clusters_data_real_error[i].parent1 = (id0 as i32).max(id1 as i32);
                }
                [] => (), // No parents is allowed. Indexes are already -1 by default.
                _ => {
                    for p in parents {
                        println!("{:?}", dag[p])
                    }

                    panic!("Non-binary parented DAG, not currently (or ever) supported");
                }
            };
        }

        // all_clusters_data_real_error is now completely valid
        for i in 0..all_clusters_data_real_error.len() {
            all_clusters_data_layer_error.push(ClusterData {
                error: *dag.node_weight(petgraph::graph::node_index(i)).unwrap() as _,
                ..all_clusters_data_real_error[i].clone()
            });
        }

        let index_buffer = Arc::new(instance.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("U32 Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
            },
        ));

        let vertex_buffer =
            instance
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&asset.verts[..]),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let index_format = wgpu::IndexFormat::Uint32;

        let cluster_data_real_error_buffer = Arc::new(instance.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("all_clusters_data_real_error"),
                contents: bytemuck::cast_slice(&all_clusters_data_real_error),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let cluster_data_layer_error_buffer = Arc::new(instance.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("all_clusters_data_layer_error"),
                contents: bytemuck::cast_slice(&all_clusters_data_layer_error),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));

        let cluster_data_real_error_group = BufferGroup::from_existing(
            [index_buffer.clone(), cluster_data_real_error_buffer.clone()],
            instance.device(),
            &instance.cluster_info_buffer_bind_group_layout,
            Some("all_clusters_data_real_error"),
        );
        let cluster_data_layer_error_group = BufferGroup::from_existing(
            [
                index_buffer.clone(),
                cluster_data_layer_error_buffer.clone(),
            ],
            instance.device(),
            &instance.cluster_info_buffer_bind_group_layout,
            Some("all_clusters_data_real_error"),
        );

        let partition_buffer = BufferGroup::create_plural_storage(
            &[&partitions, &groups],
            instance.device(),
            &instance.partition_bind_group_layout,
            Some("Partition Buffer"),
        );

        // Update the value stored in this mesh

        MultiResMeshAsset {
            vertex_buffer,
            partition_buffer,
            index_format,
            index_count: indices.len() as _,
            cluster_data_real_error_group,
            cluster_data_layer_error_group,
            cluster_count: all_clusters_data_real_error.len() as _,
            root_asset: asset,
        }
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.root_asset
    }
}
