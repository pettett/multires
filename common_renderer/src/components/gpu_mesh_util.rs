use std::cmp;

use common::{graph::assert_graph_contiguous, MeshCluster, MultiResMesh};

#[repr(C)]
#[derive(crevice::std430::AsStd430, Clone, Copy, PartialEq, Debug)]
pub struct ClusterData {
    pub center: mint::Vector3<f32>,

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

    pub min_child_index: i32,
    _4: i32,
    pub meshlet_start: u32,
    pub meshlet_count: u32,

    tight_sphere: mint::Vector4<f32>,
    culling_cone: mint::Vector4<f32>,
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
    /// Generate the cluster ordering for the mesh. This indexing should be respected by all functions creating
    /// arrays of data due to clusters for the GPU, to ensure consistent indexing on whichever ordering this imposes
    fn order_clusters(&self) -> (Vec<&MeshCluster>, Vec<Vec<usize>>);
    fn generate_cluster_data(
        &self,
        cluster_order: &[&MeshCluster],
        cluster_groups: &Vec<Vec<usize>>,
    ) -> Vec<ClusterData>;
    fn indices_partitions_groups(
        &self,
        cluster_order: &[&MeshCluster],
    ) -> (Vec<u32>, Vec<i32>, Vec<i32>);
}

impl MultiResData for MultiResMesh {
    fn order_clusters(&self) -> (Vec<&MeshCluster>, Vec<Vec<usize>>) {
        let mut clusters: Vec<_> = self.clusters.iter().collect();

        // Sorting by group index allows us to represent all children with a min/max child pair
        // As they have also share some properties, this will also slightly improve GPU compute efficiency
        clusters.sort_unstable_by_key(|x| cmp::Reverse(x.group_index));

        let mut groups = vec![Vec::new(); self.group_count];
        for i in 0..clusters.len() {
            groups[clusters[i].group_index].push(i);
        }

        (clusters, groups)
    }

    fn generate_cluster_data(
        &self,
        cluster_order: &[&MeshCluster],
        cluster_groups: &Vec<Vec<usize>>,
    ) -> Vec<ClusterData> {
        let mut clusters = Vec::new();

        let mut dag = petgraph::Graph::new();
        let mut cluster_nodes = Vec::new();

        let mut index_sum = 0;

        for (i, cluster) in cluster_order.iter().enumerate() {
            // Map index buffer to global vertex range

            let index_count = cluster.index_count() as u32;

            clusters.push(ClusterData {
                index_offset: index_sum,
                index_count,
                error: cluster.error,
                center: cluster.saturated_bound.center().into(),

                parent0: -1,
                parent1: -1,
                co_parent: -1,
                radius: cluster.saturated_bound.radius(),
                layer: cluster.lod as _,
                max_child_index: -1,
                min_child_index: -1,
                _4: 0,
                meshlet_start: 0,
                meshlet_count: 0,
                tight_sphere: cluster.tight_bound.packed().into(),
                culling_cone: cluster.tight_cone.packed().into(),
            });

            index_sum += index_count;

            cluster_nodes.push(dag.add_node(i));
        }

        // Search for [dependencies], group members, and dependants

        for (cluster_idx, &cluster_node_idx) in cluster_nodes.iter().enumerate() {
            let i = *dag.node_weight(cluster_node_idx).unwrap();

            let cluster_group_idx = cluster_order[i].group_index;

            assert!(cluster_groups[cluster_group_idx].contains(&cluster_idx));

            let Some(child_group_idx) = cluster_order[i].child_group_index else {
                continue;
            };

            // To have a child group, level > 0

            let child_clusters: Vec<_> = cluster_groups[child_group_idx]
                .iter()
                .map(|&child_cluster| cluster_nodes[child_cluster])
                .collect();

            // println!("{}", child_partitions.len());

            for &child in &child_clusters {
                // only the partitions with a shared boundary should be listed as dependants

                dag.add_edge(cluster_node_idx, child, ());
            }
        }

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

            if let Some(child_group) = cluster_order[i].child_group_index {
                let min_child_index = *cluster_groups[child_group].iter().min().unwrap();
                let max_child_index = *cluster_groups[child_group].iter().max().unwrap();

                clusters[i].min_child_index = min_child_index as _;
                clusters[i].max_child_index = max_child_index as _;
            } else {
                clusters[i].min_child_index = -1;
                clusters[i].max_child_index = -1;
            }
        }

        clusters
    }

    fn indices_partitions_groups(
        &self,
        cluster_order: &[&MeshCluster],
    ) -> (Vec<u32>, Vec<i32>, Vec<i32>) {
        let mut indices = Vec::new();
        // Face indexed array
        let mut partitions = Vec::new();
        // Partition indexed array
        let mut groups = Vec::new();

        let mut cluster_idx = 0;

        for cluster in cluster_order {
            // Map index buffer to global vertex range

            let index_count = cluster.index_count();

            for _ in 0..(index_count / 3) {
                partitions.push(cluster_idx as i32);
            }
            groups.push(cluster.group_index as i32);

            cluster_idx += 1;

            // Push to indices *after* recording the offset above
            for i in 0..cluster.colour_count() {
                indices.extend_from_slice(&cluster.meshlet_for_colour(i).calc_indices());
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

    use std::collections::VecDeque;

    use common::{Asset, MultiResMesh};

    use super::MultiResData;

    #[test]
    fn test_co_parents() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

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
    fn test_children_range() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

        for i in 0..clusters.len() {
            if clusters[i].max_child_index >= 0 {
                for c in clusters[i].min_child_index..=clusters[i].max_child_index {
                    let (p0, p1) = clusters[c as usize].get_parents().unwrap();

                    assert!(p0 == i || p1 == i);
                }
            }
        }
    }

    #[test]
    fn test_no_co_parents_at_zero() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

        for i in 0..clusters.len() {
            if clusters[i].layer == 0 {
                assert_eq!(clusters[i].co_parent, -1);
            } else {
                assert_ne!(clusters[i].co_parent, -1);
            }
        }
    }

    #[test]
    fn test_no_parents_at_zero_top() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

        for i in 0..clusters.len() {
            if clusters[i].layer == clusters[0].layer {
                assert_eq!(clusters[i].parent0, -1);
                assert_eq!(clusters[i].parent1, -1);
            } else {
                assert_ne!(clusters[i].parent0, -1);
                assert_ne!(clusters[i].parent1, -1);
            }
        }
    }

    /// For expanding searches of the dag, starting at the top and
    #[test]
    fn test_dag_queue_traversal() {
        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();

        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

        let mut q = VecDeque::new();
        let mut seen = vec![false; clusters.len()];

        let starting = 32;

        for i in 0..starting {
            q.push_back(i);
        }

        while let Some(i) = q.pop_front() {
            assert!(!seen[i]);
            seen[i] = true;

            if (i as i32) < clusters[i].co_parent {
                for c in clusters[i].min_child_index..=clusters[i].max_child_index {
                    if c as usize >= starting {
                        q.push_back(c as _);
                    }
                }
            }
            println!("{}", q.len());
        }
        println!("Seen {}", seen.len());

        assert!(seen.iter().all(|&p| p))
    }

    /// For expanding searches of the dag, starting at the top and
    #[test]
    fn test_dag_buffer_queue_traversal() {
        const QUEUE_SIZE: usize = 3000;
        const STARTING: usize = 8;

        let mesh = MultiResMesh::load_from_cargo_manifest_dir().unwrap();
        let (cluster_order, groups) = mesh.order_clusters();
        let clusters = mesh.generate_cluster_data(&cluster_order, &groups);

        let mut q = vec![0; QUEUE_SIZE];

        let mut seen = vec![false; clusters.len()];

        for i in 0..STARTING {
            q[i] = i;
        }

        let mut queue_tail = STARTING;
        let mut queue_head = 0;
        let mut max_queue_size = 0;

        while queue_tail - queue_head > 0 {
            let i = q[queue_head % QUEUE_SIZE];
            queue_head += 1;

            assert!(!seen[i]);
            seen[i] = true;

            //let can_queue_children = clusters[i].max_child_index >= STARTING as _;

            if (i as i32) < clusters[i].co_parent {
                assert!((i as i32) < clusters[i].min_child_index);
                assert!((i as i32) < clusters[i].max_child_index);

                for c in clusters[i].min_child_index..=clusters[i].max_child_index {
                    if (c >= STARTING as _) && (queue_tail - queue_head < QUEUE_SIZE) {
                        q[queue_tail % QUEUE_SIZE] = c as _;
                        queue_tail += 1;
                    }
                }
            }
            //println!("{}", (queue_tail - queue_head));
            max_queue_size = max_queue_size.max(queue_tail - queue_head);
        }
        println!("Seen {}", seen.len());
        println!("Iterations: {}", queue_head);
        println!("Max queue size: {}", max_queue_size);

        assert!(seen.iter().all(|&p| p))
    }
}
