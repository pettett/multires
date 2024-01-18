use std::{collections::HashMap, fmt::Debug, sync::Arc};

use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Query, Resource},
    world::World,
};
use common::{asset::Asset, MultiResMesh};
use common_renderer::components::{
    camera::Camera,
    gpu_mesh_util::{ClusterData, MultiResData},
    transform::Transform,
};
use glam::{Mat4, Vec3};
use petgraph::visit::EdgeRef;
use wgpu::util::{DeviceExt, DispatchIndirect};

use crate::core::{pipeline::make_render_pipeline, BufferGroup, Instance, Renderer};

const MAX_PARALLEL_INSTANCES: usize = 1;

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

#[derive(PartialEq, Clone)]
pub enum ErrorMode {
    PointDistance { camera_point: Vec3, cam: Camera },
    MaxError,
    ExactLayer,
}

impl ErrorMode {
    pub fn mode(&self) -> u32 {
        match self {
            ErrorMode::PointDistance { .. } => 0,
            ErrorMode::MaxError => 1,
            ErrorMode::ExactLayer => 2,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum DrawMode {
    Clusters,
    //Triangles,
    Pbr,
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
struct InstanceData {
    model: Mat4,
    //camera_pos: Vec3,
    //error: f32,

    //mode: u32,
    // Having a vec3 always brings fun antics like this
    current_count: u32,
    _0: i32,
    _1: i32,
    _2: i32,
}

#[derive(Component)]
pub struct MultiResMeshComponent {
    name: String,
    //cluster_count: u32,
    //FIXME: This really should exist on the camera/part of uniform group
    draw_data_buffer: BufferGroup<1>,
    model: BufferGroup<1>,
    index_format: wgpu::IndexFormat,
    asset: Arc<MultiResMeshAsset>,
    dirty: bool,
}

#[derive(Resource)]
pub struct MultiResMeshRenderer {
    pub error_calc: ErrorMode,
    pub error_target: f32,
    pub focus_part: usize,
    pub freeze: bool,
    pub draw_mode: DrawMode,
    pub show_wire: bool,
    pub show_solid: bool,
    pub show_bounds: bool,

    pub render_pipeline: wgpu::RenderPipeline,
    pub render_pipeline_pbr: wgpu::RenderPipeline,
    pub render_pipeline_wire: wgpu::RenderPipeline,
    pub culling_compute_pipeline: wgpu::ComputePipeline,
    pub compacting_compute_pipeline: wgpu::ComputePipeline,
    pub debug_staging_buffer: wgpu::Buffer,

    pub model_bind_group_layout: crate::core::BindGroupLayout<1>,
    pub partition_bind_group_layout: crate::core::BindGroupLayout<2>,
    pub write_compute_bind_group_layout: crate::core::BindGroupLayout<1>,
    pub cluster_info_buffer_bind_group_layout: crate::core::BindGroupLayout<2>,
    pub result_indices_buffer_bind_group_layout: crate::core::BindGroupLayout<2>,
    pub read_compute_bind_group_layout: crate::core::BindGroupLayout<1>,
}
/// Stores all immutable DAG data for a mesh, to be referenced by any number of instances.
pub struct MultiResMeshAsset {
    partition_buffer: BufferGroup<2>,
    vertex_buffer: wgpu::Buffer,
    cluster_count: u32,
    index_count: u32,
    cluster_data_real_error_group: BufferGroup<2>,
    index_format: wgpu::IndexFormat,
    root_asset: MultiResMesh,

    result_indices_buffer: BufferGroup<2>,
    write_can_draw_buffer: BufferGroup<1>,
    read_can_draw_buffer: BufferGroup<1>,
}

pub struct MultiResMeshDatabase {
    assets: HashMap<String, Arc<MultiResMeshAsset>>,
}

impl MultiResMeshRenderer {
    pub fn new(renderer: &Renderer) -> Self {
        let shader = renderer
            .device()
            .create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        let pbr_shader = renderer
            .device()
            .create_shader_module(wgpu::include_wgsl!("../shaders/pbr_shader.wgsl"));

        let cull_meshlets_shader = renderer
            .device()
            .create_shader_module(wgpu::include_wgsl!("../shaders/should_draw.wgsl"));

        let compact_indices_shader = renderer
            .device()
            .create_shader_module(wgpu::include_wgsl!("../shaders/compact_indices.wgsl"));

        let shader_wire = renderer
            .device()
            .create_shader_module(wgpu::include_wgsl!("../shaders/shader_wire.wgsl"));

        let model_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("model_bind_group_layout"),
        );

        let write_compute_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("writeable_compute_buffer_bind_group"),
        );
        let cluster_info_buffer_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            Some("cluster_info_buffer_bind_group_layout"),
        );

        let result_indices_buffer_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // wgpu::BindGroupLayoutEntry {
                // binding: 2,
                // visibility: wgpu::ShaderStages::COMPUTE,
                // ty: wgpu::BindingType::Buffer {
                // ty: wgpu::BufferBindingType::Storage { read_only: false },
                // has_dynamic_offset: false,
                // min_binding_size: None,
                // },
                // count: None,
                // },
            ],
            Some("indirect_draw_info_buffer_bind_group_layout"),
        );

        let read_compute_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("readable_compute_buffer_bind_group"),
        );

        let partition_bind_group_layout = crate::core::BindGroupLayout::create(
            renderer.device(),
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            Some("partition_bind_group_layout"),
        );

        let render_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        (&renderer.instance.camera_bind_group_layout).into(),
                        (&partition_bind_group_layout).into(),
                        (&model_bind_group_layout).into(),
                    ],
                    push_constant_ranges: &[],
                });

        let culling_compute_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[
                        // Reminder: Max of 4 for these, don't add any more
                        (&write_compute_bind_group_layout).into(),
                        (&cluster_info_buffer_bind_group_layout).into(),
                        (&read_compute_bind_group_layout).into(),
                        (&renderer.instance.camera_bind_group_layout).into(),
                    ],
                    push_constant_ranges: &[],
                });

        let compacting_compute_pipeline_layout =
            renderer
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compacting Compute Pipeline Layout"),
                    bind_group_layouts: &[
                        // Reminder: Max of 4 for these, don't add any more
                        (&result_indices_buffer_bind_group_layout).into(),
                        (&cluster_info_buffer_bind_group_layout).into(),
                        (&read_compute_bind_group_layout).into(),
                    ],
                    push_constant_ranges: &[],
                });

        let render_pipeline = make_render_pipeline(
            renderer.device(),
            &render_pipeline_layout,
            &shader,
            renderer.config().format,
            wgpu::PolygonMode::Fill,
            Some(wgpu::Face::Back),
        );

        let render_pipeline_pbr = make_render_pipeline(
            renderer.device(),
            &render_pipeline_layout,
            &pbr_shader,
            renderer.config().format,
            wgpu::PolygonMode::Fill,
            Some(wgpu::Face::Back),
        );
        let render_pipeline_wire = make_render_pipeline(
            renderer.device(),
            &render_pipeline_layout,
            &shader_wire,
            renderer.config().format,
            wgpu::PolygonMode::Line,
            None,
        );

        let culling_compute_pipeline =
            renderer
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Culling compute pipeline"),
                    layout: Some(&culling_compute_pipeline_layout),
                    module: &cull_meshlets_shader,
                    entry_point: "main",
                });

        let compacting_compute_pipeline =
            renderer
                .device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compacting compute pipeline"),
                    layout: Some(&compacting_compute_pipeline_layout),
                    module: &compact_indices_shader,
                    entry_point: "main",
                });
        let debug_staging_buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Staging Buffer"),
            size: 12248 as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        MultiResMeshRenderer {
            error_calc: crate::components::gpu_multi_res_mesh::ErrorMode::PointDistance {
                camera_point: Vec3::ZERO,
                cam: Camera::new(1.0),
            },
            draw_mode: DrawMode::Clusters,
            error_target: 0.5,
            focus_part: 0,
            freeze: false,
            show_wire: true,
            show_solid: true,
            show_bounds: false,
            render_pipeline,
            render_pipeline_pbr,
            render_pipeline_wire,
            culling_compute_pipeline,
            compacting_compute_pipeline,
            debug_staging_buffer,
            model_bind_group_layout,
            partition_bind_group_layout,
            write_compute_bind_group_layout,
            cluster_info_buffer_bind_group_layout,
            result_indices_buffer_bind_group_layout,
            read_compute_bind_group_layout,
        }
    }
}

// impl ClusterComponent {
//     pub fn co_error(
//         &self,
//         submeshes: &Query<(Entity, &ClusterComponent)>,
//         mesh: &MultiResMeshComponent,
//         renderer: &MultiResMeshRenderer,
//     ) -> f32 {
//         self.error(mesh, renderer).min(match self.co_parent {
//             Some(co_parent) => submeshes.get(co_parent).unwrap().1.error(mesh, renderer),
//             None => f32::MAX, // Leaf nodes have no co parent, as have no children
//         })
//     }

//     pub fn error(&self, _mesh: &MultiResMeshComponent, renderer: &MultiResMeshRenderer) -> f32 {
//         match &renderer.error_calc {
//             ErrorMode::PointDistance {
//                 camera_point, cam, ..
//             } => {
//                 // Max error we can have before mesh is not suitable to draw

//                 let distance = self.center.distance(*camera_point).max(cam.znear());

//                 self.error * self.radius / distance
//             }
//             ErrorMode::MaxError => self.error,
//             ErrorMode::ExactLayer => self.layer as _, //FIXME:
//         }
//     }

//     // Issue - parents of a group may disagree on if to draw, if they have differing errors due to being in different groups.

//     // Solution Idea - merge the parents into a single node after calculating view dependant error,
//     // taking the smaller of the two's errors to ensure other things in the group can still be drawn at the exact same time.
//     // (Group != siblings, but everything in a group and every sibling must *both* be in agreement on whether to draw)

//     pub fn error_within_bounds(
//         &self,
//         mesh: &MultiResMeshComponent,
//         renderer: &MultiResMeshRenderer,
//     ) -> bool {
//         self.error(mesh, renderer) < renderer.error_target
//     }

//     pub fn should_draw(
//         &self,
//         submeshes: &Query<(Entity, &ClusterComponent)>,
//         mesh: &MultiResMeshComponent,
//         renderer: &MultiResMeshRenderer,
//     ) -> bool {
//         //TODO: Give each partition a unique parent. This parent should be a

//         // a partition is remeshed, then repartitioned inside the same group, so how can it have a parent?
//         // remeshing can make two partitions where there was once one

//         // when grouping, each partition is assigned to a unique group that is demeshed
//         // so each partition has a parent group of partitions that form the same bound
//         // this is computed as a group such that one of the original member was us,

//         // Each demeshed item can look at a unique bound to compare against, maybe this is what we want

//         // With no parents, assume infinite error
//         let mut parent_error_too_large = self.parents.len() != 0;

//         // For each parent, if any are within bounds, they all will be
//         for &dep in &self.parents {
//             if submeshes
//                 .get(dep)
//                 .unwrap()
//                 .1
//                 .error_within_bounds(mesh, renderer)
//             {
//                 parent_error_too_large = false;
//             }
//         }

//         //TODO: This is messy - we are drawing if *we* have too high an error, but our child does not - this should be flipped,
//         // and we should draw the child

//         parent_error_too_large
//             && (self.error_within_bounds(mesh, renderer)
//                 || match self.co_parent {
//                     Some(co_parent) => submeshes
//                         .get(co_parent)
//                         .unwrap()
//                         .1
//                         .error_within_bounds(mesh, renderer),
//                     None => true, // Leaf nodes have no co parent, as have no children
//                 })
//     }

//     pub fn r_should_draw(
//         &self,
//         submeshes: &Query<(Entity, &ClusterComponent)>,
//         mesh: &MultiResMeshComponent,
//         renderer: &MultiResMeshRenderer,
//     ) -> bool {
//         let should_draw = self.should_draw(submeshes, mesh, renderer);

//         for g in &self.group {
//             if should_draw
//                 != submeshes
//                     .get(*g)
//                     .unwrap()
//                     .1
//                     .should_draw(submeshes, mesh, renderer)
//             {
//                 println!("WARNING: Not all members of a group are drawing")
//             }
//         }

//         // FIXME: probably do this on the error graph
//         //self.is_monotonic(submeshes, mesh);

//         should_draw
//     }
// }

impl MultiResMeshComponent {
    pub fn draw<'a>(
        meshes: &'a mut Query<(&mut MultiResMeshComponent, &Transform)>,
        renderer: &'a Renderer,
        mesh_renderer: &'a MultiResMeshRenderer,
        camera_trans: &Transform,
        _submeshes: &'a Query<(Entity, &ClusterComponent)>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let mut load_op = wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        });

        let mut load_op_depth = wgpu::LoadOp::Clear(1.0);

        for (mut mesh, transform) in meshes.iter_mut() {
            //    if !mesh_renderer.freeze
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                });

                let cam_dist = (*transform.get_pos() - *camera_trans.get_pos()).length_squared();

                // Quick conservative estimation of total clusters that could be drawn to test performance impact.
                //let current_count = (((mesh.asset.cluster_count as f32 / cam_dist) * 400.0
                //    / (mesh_renderer.error_target)) as u32)
                //    .min(mesh.asset.cluster_count)
                //    .max(4);

                let current_count = mesh.asset.cluster_count;

                if mesh.dirty {
                    renderer.queue().write_buffer(
                        mesh.draw_data_buffer.buffer(),
                        0,
                        &bytemuck::cast_slice(&[InstanceData {
                            model: transform.get_local_to_world(),
                            current_count,
                            _0: 0,
                            _1: 0,
                            _2: 0,
                        }]),
                    );
                    mesh.dirty = false;
                }

                let cluster_data = mesh.asset.cluster_data_real_error_group.bind_group();

                compute_pass.set_pipeline(&mesh_renderer.culling_compute_pipeline);

                compute_pass.set_bind_group(0, mesh.asset.write_can_draw_buffer.bind_group(), &[]);
                compute_pass.set_bind_group(1, cluster_data, &[]);
                compute_pass.set_bind_group(2, mesh.draw_data_buffer.bind_group(), &[]);
                compute_pass.set_bind_group(3, renderer.camera_buffer().bind_group(), &[]);

                //render_pass.set_bind_group(3, mesh.draw_data_buffer.bind_group(), &[]);

                //render_pass.set_bind_group(3, renderer.camera_buffer().bind_group(), &[]);

                compute_pass.dispatch_workgroups(current_count.div_ceil(64), 1, 1);

                compute_pass.set_pipeline(&mesh_renderer.compacting_compute_pipeline);

                compute_pass.set_bind_group(0, mesh.asset.result_indices_buffer.bind_group(), &[]);
                compute_pass.set_bind_group(1, cluster_data, &[]);
                compute_pass.set_bind_group(2, mesh.asset.read_can_draw_buffer.bind_group(), &[]);

                compute_pass.dispatch_workgroups(current_count.div_ceil(64), 1, 1);
            }

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mesh Render Pass"),
                    color_attachments: &[
                        // This is what @location(0) in the fragment shader targets
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: load_op,
                                store: true,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &renderer.depth_texture.view(),
                        depth_ops: Some(wgpu::Operations {
                            load: load_op_depth,
                            store: true,
                        }),
                        stencil_ops: None,
                    }),
                });

                render_pass.set_vertex_buffer(0, mesh.asset.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    mesh.asset.result_indices_buffer.get_buffer(0).slice(..),
                    mesh.asset.index_format,
                );

                render_pass.set_bind_group(0, renderer.camera_buffer().bind_group(), &[]);
                render_pass.set_bind_group(1, mesh.asset.partition_buffer.bind_group(), &[]);
                render_pass.set_bind_group(2, mesh.model.bind_group(), &[]);

                if mesh_renderer.show_solid {
                    let pipeline = match mesh_renderer.draw_mode {
                        DrawMode::Clusters => &mesh_renderer.render_pipeline,
                        DrawMode::Pbr => &mesh_renderer.render_pipeline_pbr,
                    };

                    render_pass.set_pipeline(pipeline);

                    render_pass
                        .draw_indexed_indirect(mesh.asset.result_indices_buffer.get_buffer(1), 0);
                }
                if mesh_renderer.show_wire {
                    render_pass.set_pipeline(&mesh_renderer.render_pipeline_wire);

                    render_pass
                        .draw_indexed_indirect(mesh.asset.result_indices_buffer.get_buffer(1), 0);
                }
            }
            load_op = wgpu::LoadOp::Load;
            load_op_depth = wgpu::LoadOp::Load;
        }
    }

    pub fn submesh_error_graph(
        &self,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        renderer: &MultiResMeshRenderer,
    ) -> petgraph::prelude::Graph<f32, ()> {
        let mut graph = petgraph::Graph::new();

        // let mut nodes = HashMap::new();

        // for (e, s) in submeshes.iter() {
        //     nodes.insert(e, graph.add_node(s.co_error(submeshes, &self, renderer)));
        // }

        // for (e, s) in submeshes.iter() {
        //     let n = nodes[&e];

        //     for p in &s.parents {
        //         let p_n = nodes[p];

        //         graph.add_edge(n, p_n, ());
        //     }

        //     // if let Some(co_parent) = &s.co_parent {
        //     //     graph.add_edge(n, nodes[co_parent], ());
        //     // }
        // }
        graph
    }

    pub fn from_asset(
        name: String,
        instance: Arc<Instance>,
        mesh_renderer: &MultiResMeshRenderer,
        world: &mut World,
        asset: Arc<MultiResMeshAsset>,
        trans: Transform,
    ) {
        //  let mut clusters_per_lod: Vec<Vec<Entity>> = Vec::new();

        //    let mut all_clusters = Vec::new();
        //    let mut indices = Vec::new();
        // Face indexed array
        //let mut partitions = Vec::new();
        // Partition indexed array
        //    let mut groups = Vec::new();

        let mut cluster_idx = 0;

        // for (level, r) in asset.asset().lods.iter().enumerate() {
        //     println!("Loading layer {level}:");
        //     let mut clusters = Vec::new();

        //     for (cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
        //         // Map index buffer to global vertex range

        //         let index_count = submesh.indices.len() as u32;

        //         for _ in 0..(index_count / 3) {
        //             partitions.push(cluster_idx as i32);
        //         }
        //         groups.push(submesh.debug_group as i32);

        //         cluster_idx += 1;

        //         let cluster_model = BufferGroup::create_single(
        //             &[Mat4::from_translation(submesh.saturated_sphere.center())
        //                 * Mat4::from_scale(Vec3::ONE * submesh.saturated_sphere.radius())],
        //             wgpu::BufferUsages::UNIFORM,
        //             instance.device(),
        //             &instance.model_bind_group_layout,
        //             Some("Uniform Debug Model Buffer"),
        //         );

        //         // let cluster = ClusterComponent {
        //         //     id: all_clusters.len(),
        //         //     //partitions: info_buffer,
        //         //     index_offset: indices.len() as u32,
        //         //     index_count,
        //         //     layer: level,
        //         //     cluster_layer_idx,
        //         //     center: submesh.saturated_sphere.center(),
        //         //     error: submesh.error,
        //         //     model: cluster_model,
        //         //     radius: submesh.saturated_sphere.radius(),
        //         //     //    children: vec![],
        //         //     parents: vec![],
        //         //     group: vec![],
        //         //     co_parent: None,
        //         // };

        //         // Push to indices *after* recording the offset above
        //         indices.extend_from_slice(&submesh.indices);

        //         let e = world.spawn(cluster).id();

        //         clusters.push(e);
        //         all_clusters.push(e);
        //     }
        //     clusters_per_lod.push(clusters);
        // }

        //assert_eq!(partitions.len(), indices.len() / 3);
        // The last partition should be the largest
        //assert_eq!(groups.len(), *partitions.last().unwrap() as usize + 1);

        // Search for [dependencies], group members, and dependants
        // for (level, partition_entities) in clusters_per_lod.iter().enumerate() {
        //     for (i_partition, &partition) in partition_entities.iter().enumerate() {
        //         let i_partition_group =
        //             asset.asset().lods[level].partitions[i_partition].group_index;

        //         assert!(asset.asset().lods[level].groups[i_partition_group]
        //             .partitions
        //             .contains(&i_partition));

        //         let Some(i_partition_child_group) =
        //             asset.asset().lods[level].partitions[i_partition].child_group_index
        //         else {
        //             continue;
        //         };

        //         let child_partitions: Vec<_> = asset.asset().lods[level - 1].groups
        //             [i_partition_child_group]
        //             .partitions
        //             .iter()
        //             .map(|child_partition| clusters_per_lod[level - 1][*child_partition])
        //             .collect();

        //         for &child in &child_partitions {
        //             // only the partitions with a shared boundary should be listed as dependants

        //             world
        //                 .get_mut::<ClusterComponent>(child)
        //                 .unwrap()
        //                 .parents
        //                 .push(partition);
        //         }
        //     }

        //     // Search for Co-parents
        //     for &cluster in &all_clusters {
        //         match world.get::<ClusterComponent>(cluster).unwrap().parents[..] {
        //             [p0, p1] => {
        //                 // Set co-parent pointers to each other

        //                 world.get_mut::<ClusterComponent>(p1).unwrap().co_parent = Some(p0);

        //                 world.get_mut::<ClusterComponent>(p0).unwrap().co_parent = Some(p1);
        //             }
        //             [] => (),
        //             _ => panic!("Non-binary parented DAG, not currently (or ever) supported"),
        //         }
        //     }
        // }

        let draw_data_buffer = BufferGroup::create_single(
            &[InstanceData {
                model: trans.get_local_to_world(),
                current_count: 2,
                _0: 0,
                _1: 0,
                _2: 0,
            }],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            instance.device(),
            &mesh_renderer.read_compute_bind_group_layout,
            Some("Draw Data Buffer"),
        );

        let index_format = wgpu::IndexFormat::Uint32;

        let model = BufferGroup::create_single(
            &[trans.get_local_to_world()],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
            instance.device(),
            &mesh_renderer.model_bind_group_layout,
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

        let staging_buffer_size = 100; // std::mem::size_of_val(&cluster_can_draw[..]);
                                       // let _debug_staging_buffer = instance.device().create_buffer(&wgpu::BufferDescriptor {
                                       //     label: Some("Compute Buffer"),
                                       //     size: staging_buffer_size as u64,
                                       //     usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                                       //     mapped_at_creation: false,
                                       // });

        // Update the value stored in this mesh
        world.spawn((
            MultiResMeshComponent {
                name,
                index_format,
                draw_data_buffer,

                //result_indices_buffer,
                //debug_staging_buffer,
                //staging_buffer_size,
                model,
                asset,
                dirty: true,
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
    pub fn load_mesh(instance: Arc<Instance>, mesh_renderer: &MultiResMeshRenderer) -> Self {
        let asset = common::MultiResMesh::load().unwrap();

        let all_clusters_data_real_error = asset.generate_cluster_data();

        let (indices, partitions, groups) = asset.indices_partitions_groups();

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

        let cluster_data_real_error_group = BufferGroup::from_existing(
            [index_buffer.clone(), cluster_data_real_error_buffer.clone()],
            instance.device(),
            &mesh_renderer.cluster_info_buffer_bind_group_layout,
            Some("all_clusters_data_real_error"),
        );

        let partition_buffer = BufferGroup::create_plural_storage(
            &[&partitions, &groups],
            instance.device(),
            &mesh_renderer.partition_bind_group_layout,
            &[Some("Partition Buffer"), Some("Group")],
        );

        let cluster_result_indices = vec![0i32; indices.len() * MAX_PARALLEL_INSTANCES];

        let result_indices_buffer = BufferGroup::create_plural(
            &[
                &cluster_result_indices,
                bytemuck::cast_slice(&[[indices.len() as i32, 1, 0, 0, 0]; MAX_PARALLEL_INSTANCES]),
                //    &cluster_can_draw,
            ],
            &[
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDEX,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::INDIRECT,
                //    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ],
            instance.device(),
            &mesh_renderer.result_indices_buffer_bind_group_layout,
            &[
                Some("cluster_result_indices"),
                Some("Indirect draw indexed buffer"),
                //    Some("Prefix scan buffer"),
            ],
        );

        let cluster_can_draw =
            vec![0i32; (all_clusters_data_real_error.len() + 1) * MAX_PARALLEL_INSTANCES];

        let can_draw_buffer = BufferGroup::create_single(
            &cluster_can_draw,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            instance.device(),
            &mesh_renderer.write_compute_bind_group_layout,
            Some("cluster_can_draw_cull_buffer"),
        );

        // Update the value stored in this mesh

        MultiResMeshAsset {
            vertex_buffer,
            partition_buffer,
            index_format,
            result_indices_buffer,
            read_can_draw_buffer: can_draw_buffer.alternate_bind_group(
                instance.device(),
                &mesh_renderer.read_compute_bind_group_layout,
                Some("read_cluster_can_draw"),
            ),
            write_can_draw_buffer: can_draw_buffer,
            index_count: indices.len() as _,
            cluster_data_real_error_group,
            cluster_count: all_clusters_data_real_error.len() as _,
            root_asset: asset,
        }
    }

    pub fn asset(&self) -> &MultiResMesh {
        &self.root_asset
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_scan_algorithm() {
        let mut data = [2, 0, 7, 3, 1, 0, 1];

        for i in 0..(f32::log2(data.len() as _).ceil() as u32) {
            let ex2 = 2usize.pow(i);
            let mut new_data = [2, 0, 0, 5, 6, 0, 0];

            for idx in 0..data.len() {
                new_data[idx] = if idx < ex2 {
                    data[idx]
                } else {
                    data[idx] + data[idx - ex2]
                };
            }
            println!("{data:?} {ex2}");
            data = new_data;
        }

        println!("{data:?}")
    }
}
