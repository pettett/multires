//use crate::MAX_TRIS_PER_CLUSTER;

use crate::mesh::{cluster_info::ClusterInfo, group_info::GroupInfo};

use super::winged_mesh::WingedMesh;
use common::{
    graph::{assert_graph_contiguous, petgraph_to_svg},
    BoundingSphere,
};
use glam::Vec4Swizzles;

impl WingedMesh {
    pub fn partition_contiguous(&mut self) -> Vec<usize> {
        println!("Wiping partitions");
        // Wipe partitions
        for (_, f) in self.iter_faces_mut() {
            f.cluster_idx = 0;
        }

        let mut p = 1;
        let mut search = Vec::new();
        let mut counts = Vec::new();

        loop {
            assert_eq!(search.len(), 0);
            // select a random face to start the search

            for (fid, f) in self.iter_faces() {
                if f.cluster_idx == 0 {
                    search.push(fid);
                    break;
                }
            }

            if search.len() == 0 {
                // end this with p-1 partitions
                break counts;
            }

            counts.push(0);

            while let Some(fid) = search.pop() {
                // Mark

                self.get_face_mut(fid).cluster_idx = p;
                counts[p - 1] += 1;

                // Search for unmarked
                let e0 = self.get_face(fid).edge;
                let e1 = self.get_edge(e0).edge_left_cw;
                let e2 = self.get_edge(e0).edge_left_ccw;

                for e in [e0, e1, e2] {
                    if let Some(t) = self.get_edge(e).twin {
                        let f = self.get_edge(t).face;
                        if self.get_face(f).cluster_idx == 0 {
                            // splitting by contiguous, we should not be able to access others

                            search.push(f);
                        } else {
                            assert_eq!(self.get_face(f).cluster_idx, p);
                        }
                    }
                }
            }
            p += 1;
        }
    }

    pub fn partition_full_mesh(
        &mut self,
        config: &metis::PartitioningConfig,
        partitions: u32,
        verts: &[glam::Vec4],
    ) -> Result<(), metis::PartitioningError> {
        println!("Partitioning into {partitions} partitions");

        let mesh_dual = self.generate_face_graph();

        #[cfg(test)]
        {
            //if config.force_contiguous_partitions {
            //        assert_contiguous_graph(&mesh_dual);
            //    }
        }

        let cluster_indexes = config.partition_from_graph(partitions, &mesh_dual)?;

        assert_eq!(cluster_indexes.len(), self.face_count());

        let cluster_count = *cluster_indexes.iter().max().unwrap() as usize + 1;

        let mut occupancies = vec![0; cluster_count];

        for i in 0..cluster_indexes.len() {
            let fid = *mesh_dual
                .node_weight(petgraph::graph::node_index(i))
                .unwrap();
            let face = self.get_face_mut(fid);

            face.cluster_idx = cluster_indexes[i] as usize;
            occupancies[face.cluster_idx] += 1;
        }

        // for (fid, face) in self.iter_faces_mut() {
        //     // Some faces will have already been removed
        //     face.cluster_idx = cluster_indexes[fid.0] as usize;
        //     occupancies[face.cluster_idx] += 1;
        // }

        //assert!(*occupancies.iter().max().unwrap() <= MAX_TRIS_PER_CLUSTER);

        println!(
            "Clustered full mesh, Min tris: {}, Max tris: {}",
            occupancies.iter().min().unwrap(),
            occupancies.iter().max().unwrap()
        );

        let mut new_clusters = vec![
            ClusterInfo {
                child_group_index: None,
                group_index: usize::MAX,
                tight_bound: Default::default(),
                tight_cone: Default::default(),
                num_tris: 0,
                num_colours: 0,
            };
            cluster_count
        ];

        self.construct_tight_cluster_bounds(&mut new_clusters, verts);

        self.clusters = new_clusters;

        Ok(())
    }

    /// Group clusters, recording the bounding boxes for all groups.
    pub fn group(
        &mut self,
        config: &metis::PartitioningConfig,
    ) -> Result<usize, metis::PartitioningError> {
        assert!(config.force_contiguous_partitions.unwrap());

        const EXACT_CLUSTERING: bool = true;

        let cluster_graph = self.generate_guided_cluster_graph();

        let group_count = {
            //std::mem::swap(&mut self.groups, &mut new_groups);

            let (cluster_partitioning, group_count) = if EXACT_CLUSTERING {
                config.exact_partition_onto_graph(4, &cluster_graph)?
            } else {
                let group_count = (cluster_graph.node_count() - 1).div_ceil(4) as _;

                let cluster_partitioning =
                    config.partition_onto_graph(group_count, &cluster_graph)?;

                (cluster_partitioning, group_count)
            };

            let group_count = group_count as usize;

            // let non_contig_even_clustering_config = &metis::PartitioningConfig {
            //     method: metis::PartitioningMethod::MultilevelRecursiveBisection,
            //     force_contiguous_partitions: Some(true),
            //     u_factor: Some(1),
            //     //objective_type: Some(metis::ObjectiveType::Volume),
            //     //    minimize_subgraph_degree: Some(true), // this will sometimes break contiguous partitions
            //     ..Default::default()
            // };

            {
                let mut occupancies = vec![0; group_count];
                for &group in cluster_partitioning.node_weights() {
                    occupancies[group as usize] += 1;
                }
                let m = occupancies.iter().max().unwrap();
                println!(
                    "Generated groups with sizes {}<->{}",
                    occupancies.iter().min().unwrap(),
                    m
                );

                let mut counts = vec![0usize; (*m + 1) as _];

                let sum = occupancies.len();

                for i in occupancies {
                    counts[i] += 1;
                }

                println!("Cnt | Occupancies (total {sum})");

                for (i, &c) in counts.iter().enumerate() {
                    println!("{i:<3} | {c:>5} {}", "@".repeat((c * 100) / sum))
                }

                //let sum = occupancies.iter().sum();
            }

            // Tell each partition what group they now belong to.
            if group_count != 1 {
                for (part, &group) in cluster_partitioning.node_weights().enumerate() {
                    // Offset for previous groups, to ensure references such as child group
                    // are correct across groupings
                    self.clusters[part].group_index = group as usize;
                }
            } else {
                for p in &mut self.clusters {
                    p.group_index = 0;
                }
            };

            group_count
        };

        println!(
            "Partitioning into {group_count} groups from {} partitions",
            self.clusters.len()
        );

        // create new array of groups, and remember the old groups
        let mut new_groups = vec![
            GroupInfo {
                saturated_bound: Default::default(),
                clusters: Vec::new(),
                //group_neighbours: BTreeSet::new(),
            };
            group_count
        ];

        // Record the partitions that each of these groups come from, and build a monotonic bounding sphere
        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            let cluster_group = &mut new_groups[cluster.group_index];

            if cluster_group.clusters.len() == 0 {
                //initialise the bounds
                cluster_group.saturated_bound = cluster.tight_bound;
            } else {
                // Expand the bounds
                cluster_group
                    .saturated_bound
                    .include_sphere(&cluster.tight_bound);
            }

            cluster_group.clusters.push(cluster_idx);
        }

        // println!(
        //     "Including child bounds with {} old groups",
        //     new_groups.len()
        // );

        for g in &mut new_groups {
            // Each group also must envelop all the groups it is descended from,
            // as our partitions must do the same, as we base them off group info

            for cluster in &g.clusters {
                if let Some(child_group_index) = self.clusters[*cluster].child_group_index {
                    let child_group = &self.groups[child_group_index];
                    // combine groups radius
                    g.saturated_bound
                        .include_sphere(&child_group.saturated_bound);
                }
            }
        }

        self.groups = new_groups;

        //#[cfg(test)]
        //{
        //Assert that we have made good groups
        //TODO: Split graph into contiguous segments beforehand

        let groups = self.generate_group_keyed_graphs();

        for g in &groups {
            assert_graph_contiguous(g);
            //super::graph::test::assert_contiguous_graph(g);
        }
        //}

        Ok(group_count)
    }

    /// Within each group, split triangles into two completely new partitions, so as not to preserve any old seams between ancient partitions
    /// Ensures the data structure is seamless with changing seams! Yippee!
    /// Will update the partitions list, but groups list will still refer to old partitions. To find out what group these should be in, before regrouping,
    /// look at `child_group_index`
    pub fn partition_within_groups(
        &mut self,
        config: &metis::PartitioningConfig,
        verts: &[glam::Vec4],
        parts_per_group: Option<u32>,
        tris_per_cluster: Option<u32>,
    ) -> Result<usize, metis::PartitioningError> {
        let graphs = self.generate_group_keyed_graphs();

        println!("Partitioning {} groups into sub-partitions", graphs.len());

        // Ungrouped partitions but with a dependence on an old group
        let mut new_clusters = Vec::new();

        for (group_idx, graph) in graphs.iter().enumerate() {
            // TODO: fine tune so we get 64/126 meshlets

            if graph.node_count() == 0 {
                println!("WARNING: Group {group_idx} face graph has no nodes!");
                continue;
            }

            let parts = if let Some(parts_per_group) = parts_per_group {
                parts_per_group
            } else {
                (graph.node_count() as u32).div_ceil(tris_per_cluster.unwrap())
            };

            assert_graph_contiguous(graph);

            let part = match config.partition_from_graph(parts, graph) {
                Ok(part) => part,
                e => {
                    petgraph_to_svg(
                        graph,
                        "error_partition_within_group.svg",
                        &|_, _| String::new(),
                        common::graph::GraphSVGRender::Undirected {
                            edge_label: common::graph::Label::None,
                            positions: false,
                        },
                    )
                    .unwrap();
                    e?
                }
            };

            // Each new part needs to register its dependence on the group we were a part of before
            let child_group = self.clusters[self
                .get_face(graph[petgraph::graph::node_index(0)])
                .cluster_idx]
                .group_index;

            assert_eq!(group_idx, child_group);

            // Update partitions of the actual triangles
            for x in graph.node_indices() {
                self.get_face_mut(graph[x]).cluster_idx =
                    new_clusters.len() + part[x.index()] as usize;
            }
            // If we have not been grouped yet,
            let child_group_index = if self.groups.len() == 0 {
                None
            } else {
                Some(group_idx)
            };

            let mut occupancies = vec![0; parts as usize];

            for p in part {
                occupancies[p as usize] += 1;
            }

            for _ in 0..parts {
                //    self.groups[group].partitions.push(new_partitions.len());

                new_clusters.push(ClusterInfo {
                    child_group_index,
                    group_index: usize::MAX,
                    tight_bound: Default::default(),
                    tight_cone: Default::default(),
                    num_colours: 0,
                    num_tris: 0,
                })
            }

            //let max = *occupancies.iter().max().unwrap();
            //assert!(
            //    max <= MAX_TRIS_PER_CLUSTER,
            //    "Too many triangles in cluster: {max}"
            //);
        }

        self.construct_tight_cluster_bounds(&mut new_clusters, verts);

        self.clusters = new_clusters;

        Ok(self.clusters.len())
        //Ok(groups)
    }

    /// Construct a tight bounding cone for each cluster based on vertices in the mesh
    fn construct_tight_cluster_bounds(&self, clusters: &mut [ClusterInfo], verts: &[glam::Vec4]) {
        for (_fid, f) in self.iter_faces() {
            let cluster = &mut clusters[f.cluster_idx];

            let point = verts[self.get_edge(f.edge).vert_origin.0].xyz();

            if cluster.num_tris == 0 {
                // Start the bounding sphere at this vertex
                cluster.tight_bound = BoundingSphere::new(point, 0.0);
            } else {
                // Expand the bounding sphere
                cluster.tight_bound.include_point(point);
            }

            let face_plane = f.plane(self, verts);

            cluster.tight_cone.add_axis(face_plane.normal().into());

            cluster.num_tris += 1;
        }

        // We now have a bounding sphere, we can make a safe space cone. We use the bounding center as our cone apex point, and have an axis for the cone.

        // We just need to normalise the axis, and force each cone to include every face plane (min(dot(face, axis) - 1)).
        for cluster in clusters.iter_mut() {
            cluster.tight_cone.normalise_axis();
        }

        for (_fid, f) in self.iter_faces() {
            let cluster = &mut clusters[f.cluster_idx];

            let face_plane = f.plane(self, verts);

            let plane_dot = face_plane.normal().dot(cluster.tight_cone.axis().into());

            // We are able to see a plane if the dot product of our view to its normal is over 0, so subtract 1
            cluster.tight_cone.min_cutoff(plane_dot - 1.0);
        }
    }

    /// Colour within clusters, for splitting too large clusters into smaller units for use in mesh shaders
    pub fn colour_within_clusters(
        &mut self,
        config: &metis::PartitioningConfig,
        tris_per_cluster: usize,
    ) -> Result<(), metis::PartitioningError> {
        let graphs = self.generate_cluster_keyed_graphs();
        // Ungrouped partitions but with a dependence on an old group

        for (cluster_idx, graph) in graphs.iter().enumerate() {
            if graph.node_count() == 0 {
                println!("WARNING: Cluster {cluster_idx} face graph has no nodes!");
                continue;
            }

            let parts = graph.node_count().div_ceil(tris_per_cluster);

            #[cfg(test)]
            {
                assert_graph_contiguous(graph);
            }

            let partitioning = match config.partition_from_graph(parts as u32, graph) {
                Ok(part) => part,
                e => {
                    petgraph_to_svg(
                        graph,
                        "error_partition_within_clusters.svg",
                        &|_, _| String::new(),
                        common::graph::GraphSVGRender::Undirected {
                            edge_label: common::graph::Label::None,
                            positions: false,
                        },
                    )
                    .unwrap();
                    e?
                }
            };

            let mut occupancies = vec![0; parts as usize];

            // Update colours of the actual triangles
            for x in graph.node_indices() {
                self.get_face_mut(graph[x]).colour = partitioning[x.index()] as usize;
                occupancies[self.get_face_mut(graph[x]).colour] += 1;
            }

            assert!(
                *occupancies.iter().max().unwrap() <= 126,
                "{:?}",
                occupancies
            );

            self.clusters[cluster_idx].num_colours = parts;
        }

        Ok(())
        //Ok(groups)
    }
}

#[cfg(test)]
pub mod tests {
    use std::{collections::HashSet, io::Write};

    use anyhow::Context;

    use crate::{
        mesh::winged_mesh::{
            test::{TEST_MESH_DRAGON, TEST_MESH_LOW},
            MeshError,
        },
        STARTING_CLUSTER_SIZE,
    };

    use super::*;

    #[test]
    pub fn test_group_sizes() -> anyhow::Result<()> {
        let test_config = &metis::MultilevelKWayPartitioningConfig {
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        }
        .into();
        let (mut mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_DRAGON);

        println!("{:?}", mesh.partition_contiguous());

        mesh.filter_tris_by_cluster(1).unwrap();

        mesh.partition_full_mesh(
            test_config,
            mesh.face_count().div_ceil(STARTING_CLUSTER_SIZE) as _,
            &tri_mesh.verts,
        )?;

        mesh.group(test_config)?;

        Ok(())
    }

    #[test]
    pub fn test_group_neighbours() -> anyhow::Result<()> {
        let test_config = &metis::MultilevelKWayPartitioningConfig {
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        }
        .into();
        let (mut mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_LOW);

        mesh.partition_full_mesh(
            test_config,
            mesh.face_count().div_ceil(STARTING_CLUSTER_SIZE) as _,
            &tri_mesh.verts,
        )?;
        mesh.group(test_config)?;

        // What does it mean for two groups to be neighbours?

        // - An edge collapse in a group can only effect edges in a group and its neighbours.

        // Define effect as changing the estimated error.

        // - An edge collapse will effect any edges where the quartics change, i.e. surrounding the output vertex.

        // This must be correct for the purposes of parallelisation, as we don't want to parallel collapse edges in neighbouring groups.

        for (_, vert) in mesh.iter_verts() {
            let mut groups = HashSet::new();

            for &out in vert.outgoing_edges() {
                groups.insert(
                    mesh.clusters[mesh.get_face(mesh.get_edge(out).face).cluster_idx].group_index,
                );
            }
            for &out in vert.incoming_edges() {
                groups.insert(
                    mesh.clusters[mesh.get_face(mesh.get_edge(out).face).cluster_idx].group_index,
                );
            }

            // for &g in &groups {
            //     for &g2 in &groups {
            //         if g != g2 {
            //             if !mesh.groups[g].group_neighbours.contains(&g2) {
            //                 println!("{:?}", mesh.groups[g].group_neighbours);
            //                 println!("{:?}", mesh.groups[g2].group_neighbours);
            //                 println!("{:?}", groups);

            //                 Err(MeshError::InvalidNeighbours(g, g2))
            //                     .context(MeshError::InvalidGroup(g))?;
            //             }
            //         }
            //     }
            // }
        }

        Ok(())
    }
}

// //std::mem::swap(&mut self.groups, &mut new_groups);

// let mut cluster_partitioning =
//     config.partition_onto_graph(group_count as u32, &cluster_graph)?;

// // Tell each partition what group they now belong to.
// if group_count != 1 {
//     let mut occupancies = vec![Vec::new(); group_count];

//     for (cluster, &group) in cluster_partitioning.node_weights().enumerate() {
//         occupancies[group].push(cluster);
//     }
//     let mut fail = 0;
//     let mut min_group = 10;
//     let mut max_group = 0;
//     for (group, clusters) in occupancies.into_iter().enumerate() {
//         let o = clusters.len();

//         if o >= 5 || (o <= 3 && o != 0) {
//             fail += 1;
//             min_group = min_group.min(o);
//             max_group = max_group.max(o);

//             if o >= 15 {
//                 // Split group into smaller sections
//                 let groups = o.div_ceil(4);

//                 // Take a snapshot of the graph for this

//                 let snapshot = cluster_partitioning.filter_map(
//                     |i, &n| if n == group { Some(i) } else { None },
//                     // Want a random clean grouping at this point, so ignore added edges
//                     |i, &e| Some(()),
//                 );

//                 let part = config.partition_from_graph(groups as _, &snapshot)?;

//                 let mut part_map = vec![group];
//                 for _ in 1..groups {
//                     part_map.push(group_count);
//                     group_count += 1;
//                 }

//                 println!("Split 1 large group into {groups}: {part:?}");

//                 // Re-apply grouping from subdivided group
//                 for (i, p) in part.into_iter().enumerate() {
//                     *cluster_partitioning
//                         .node_weight_mut(
//                             *snapshot
//                                 .node_weight(petgraph::graph::node_index(i))
//                                 .unwrap(),
//                         )
//                         .unwrap() = part_map[p as usize];
//                 }
//             }
//         } else {
//             let tri_count: i32 = clusters
//                 .iter()
//                 .map(|&i| {
//                     cluster_graph
//                         .node_weight(petgraph::graph::node_index(i))
//                         .unwrap()
//                 })
//                 .sum();

//             assert!(
//                 tri_count < MAX_TRIS_PER_CLUSTER as i32 * 4,
//                 "Too many triangles in group of {clusters:?}: {tri_count}"
//             );
//         }
//     }

//     if fail > 0 {
//         println!(
//             "Failed to group graph with {fail} invalid groups. {min_group}/{max_group}"
//         );

//         // petgraph_to_svg(
//         //     &cluster_partitioning,
//         //     "svg\\group_clustering_failure.svg",
//         //     &|_, (i, _)| {
//         //         format!(
//         //             "color={}",
//         //             common::graph::COLS[*cluster_partitioning.node_weight(i).unwrap()
//         //                 as usize
//         //                 % common::graph::COLS.len()]
//         //         )
//         //     },
//         //     common::graph::GraphSVGRender::Undirected {
//         //         positions: false,
//         //         edge_label: common::graph::Label::None,
//         //     },
//         // )
//         // .unwrap();

//         //panic!("Failed to group graph with {fail} invalid groups");
//     }

//     for (cluster, &group) in cluster_partitioning.node_weights().enumerate() {
//         self.clusters[cluster].group_index = group;
//     }
// } else {
//     for p in &mut self.clusters {
//         p.group_index = 0;
//     }
// };
