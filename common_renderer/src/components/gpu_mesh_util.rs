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
    pub layer: u32,
    pub max_child_index: i32,

    _3: i32,
    _4: i32,
    pub meshlet_start: u32,
    pub meshlet_count: u32,
}

impl ClusterData {
    /// Get indexes of both parents of this cluster in the asset array
    pub fn get_parents(&self) -> Option<(usize, usize)> {
        // If we have one parent, we must have to other, otherwise the function is invalid
        self.parent0
            .try_into()
            .ok()
            .map(|p0| (p0, self.parent1 as _))
    }

    /// Get index of co-parent of this cluster in the asset array
    pub fn get_co_parent(&self) -> Option<usize> {
        self.co_parent.try_into().ok()
    }
}

pub trait MultiResData {
    fn generate_cluster_data(&self) -> Vec<ClusterData>;
    fn indices_partitions_groups(&self) -> (Vec<u32>, Vec<i32>, Vec<i32>);
}

impl MultiResData for MultiResMesh {
    fn generate_cluster_data(&self) -> Vec<ClusterData> {
        let mut clusters = Vec::new();

        let mut dag = petgraph::Graph::new();
        let mut clusters_per_lod = vec![Vec::new(); self.lods.len()];

        let mut index_sum = 0;

        for (level, r) in self.lods.iter().enumerate().rev() {
            println!("Loading layer {level}:");
            let cluster_nodes = &mut clusters_per_lod[level];

            for (_cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let index_count = submesh.index_count() as u32;

                clusters.push(ClusterData {
                    index_offset: index_sum,
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

                index_sum += index_count;

                cluster_nodes.push(dag.add_node(level));
            }
        }

        // Search for [dependencies], group members, and dependants
        for (level, cluster_nodes) in clusters_per_lod.iter().enumerate() {
            for (cluster_idx, &cluster_node_idx) in cluster_nodes.iter().enumerate() {
                let cluster_group_idx = self.lods[level].clusters[cluster_idx].group_index;

                assert!(self.lods[level].groups[cluster_group_idx]
                    .partitions
                    .contains(&cluster_idx));

                let Some(child_group_idx) =
                    self.lods[level].clusters[cluster_idx].child_group_index
                else {
                    continue;
                };

                // To have a child group, level > 0

                let child_clusters: Vec<_> = self.lods[level - 1].groups[child_group_idx]
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
        for i in 0..clusters.len() {
            let parents = dag
                .neighbors_directed(
                    petgraph::graph::node_index(i),
                    petgraph::Direction::Incoming,
                )
                .collect::<Vec<_>>();

            match parents[..] {
                [p0, p1] => {
                    // Set co-parent pointers to each other. This work will be duplicated a lot of times, but it's convenient
                    let id0 = p0.index().min(p1.index());
                    let id1 = p0.index().max(p1.index());

                    clusters[id0].co_parent = id1 as _;
                    clusters[id1].co_parent = id0 as _;

                    // Set parent pointers for ourself
                    clusters[i].parent0 = (id0 as i32).min(id1 as i32);
                    clusters[i].parent1 = (id0 as i32).max(id1 as i32);
                }
                [] => (), // No parents is allowed. Indexes are already -1 by default.
                _ => {
                    for p in parents {
                        println!("{:?}", dag[p])
                    }

                    panic!("Non-binary parented DAG, not currently (or ever) supported");
                }
            };

            let max_child_idx = dag
                .neighbors_directed(
                    petgraph::graph::node_index(i),
                    petgraph::Direction::Outgoing,
                )
                .max()
                .map(|x| x.index() as i32)
                .unwrap_or(-1);

            clusters[i].max_child_index = max_child_idx;
        }

        clusters
    }

    fn indices_partitions_groups(&self) -> (Vec<u32>, Vec<i32>, Vec<i32>) {
        let mut indices = Vec::new();
        // Face indexed array
        let mut partitions = Vec::new();
        // Partition indexed array
        let mut groups = Vec::new();

        let mut cluster_idx = 0;

        for (level, r) in self.lods.iter().enumerate().rev() {
            println!("Loading layer {level}:");

            for (_cluster_layer_idx, submesh) in r.submeshes.iter().enumerate() {
                // Map index buffer to global vertex range

                let index_count = submesh.index_count() as u32;

                for _ in 0..(index_count / 3) {
                    partitions.push(cluster_idx as i32);
                }
                groups.push(submesh.debug_group as i32);

                cluster_idx += 1;

                // Push to indices *after* recording the offset above
                for i in 0..submesh.colour_count() {
                    indices.extend_from_slice(&submesh.indices_for_colour(i));
                }
            }
        }

        assert_eq!(partitions.len(), indices.len() / 3);
        // The last partition should be the largest
        assert_eq!(groups.len(), *partitions.last().unwrap() as usize + 1);

        (indices, partitions, groups)
    }
}

#[cfg(test)]
mod tests {
    use std::{path, str::FromStr};

    use common::{asset::Asset, MultiResMesh};

    use super::MultiResData;

    #[test]
    fn test_co_parents() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let clusters = mesh.generate_cluster_data();

        for c in &clusters {
            if let Some((p0id, p1id)) = c.get_parents() {
                assert!(p0id < p1id);

                let p0 = &clusters[p0id];
                let p1 = &clusters[p1id];

                assert_eq!(p0.get_co_parent().unwrap(), p1id);
                assert_eq!(p1.get_co_parent().unwrap(), p0id);
            }
        }
    }

    #[test]
    fn test_max_children() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let clusters = mesh.generate_cluster_data();

        for i in 0..clusters.len() {
            if let Some((p0id, p1id)) = clusters[i].get_parents() {
                let p0 = &clusters[p0id];
                let p1 = &clusters[p1id];

                assert!(p0.max_child_index >= i as i32);
                assert!(p1.max_child_index >= i as i32);
            }
        }
    }
}
