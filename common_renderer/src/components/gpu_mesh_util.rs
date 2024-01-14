use common::{asset::Asset, MultiResMesh};

#[repr(C)]
#[derive(
    crevice::std430::AsStd430,
    bytemuck::Pod,
    Clone,
    Copy,
    bytemuck::Zeroable,
    PartialEq,
    Debug,
    Default,
)]
pub struct ClusterData {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    // Range into the index array that this submesh resides
    pub index_offset: u32,
    pub index_count: u32,
    pub error: f32,
    //radius: f32,
    // All of these could be none (-1), if we are a leaf or a root node
    pub parent0: i32,
    pub parent1: i32,
    pub co_parent: i32,

    // Pad alignment to 4 bytes
    pub radius: f32,
    pub layer: i32,
    _2: i32,

    _3: i32,
    _4: i32,
    _5: i32,
    _6: i32,
}

pub fn cluster_data_from_asset(
    asset: &MultiResMesh,
) -> (Vec<ClusterData>, Vec<u32>, Vec<i32>, Vec<i32>) {
    let mut all_clusters_data_real_error = Vec::new();
    let mut indices = Vec::new();
    // Face indexed array
    let mut partitions = Vec::new();
    // Partition indexed array
    let mut groups = Vec::new();

    let mut cluster_idx = 0;

    let mut dag = petgraph::Graph::new();
    let mut clusters_per_lod = vec![Vec::new(); asset.lods.len()];

    for (level, r) in asset.lods.iter().enumerate().rev() {
        println!("Loading layer {level}:");
        let cluster_nodes = &mut clusters_per_lod[level];

        for (_cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
            // Map index buffer to global vertex range

            let index_count = submesh.index_count() as u32;

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
                layer: level as _,
                ..ClusterData::default()
            });

            cluster_nodes.push(dag.add_node(level));

            // Push to indices *after* recording the offset above
            for i in 0..submesh.colour_count() {
                indices.extend_from_slice(&submesh.indices_for_colour(i));
            }
        }
    }

    assert_eq!(partitions.len(), indices.len() / 3);
    // The last partition should be the largest
    assert_eq!(groups.len(), *partitions.last().unwrap() as usize + 1);

    // Search for [dependencies], group members, and dependants
    for (level, cluster_nodes) in clusters_per_lod.iter().enumerate() {
        for (cluster_idx, &cluster_node_idx) in cluster_nodes.iter().enumerate() {
            let cluster_group_idx = asset.lods[level].clusters[cluster_idx].group_index;

            assert!(asset.lods[level].groups[cluster_group_idx]
                .partitions
                .contains(&cluster_idx));

            let Some(child_group_idx) = asset.lods[level].clusters[cluster_idx].child_group_index
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

    (all_clusters_data_real_error, indices, partitions, groups)
}
