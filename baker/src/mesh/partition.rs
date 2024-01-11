use std::collections::BTreeSet;

use super::winged_mesh::WingedMesh;
use common::graph::petgraph_to_svg;
use glam::Vec4Swizzles;

impl WingedMesh {
    pub fn partition_contiguous(&mut self) -> usize {
        println!("Wiping partitions");
        // Wipe partitions
        for (_, f) in self.iter_faces_mut() {
            f.cluster_idx = 0;
        }

        let mut p = 1;
        let mut search = Vec::new();

        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.face_count() as _);

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
                #[cfg(feature = "progress")]
                bar.finish();

                // end this with p-1 partitions
                break p - 1;
            }

            while let Some(fid) = search.pop() {
                // Mark
                #[cfg(feature = "progress")]
                bar.inc(1);
                self.get_face_mut(fid).cluster_idx = p;

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

        for (fid, face) in self.iter_faces_mut() {
            // Some faces will have already been removed
            face.cluster_idx = cluster_indexes[fid.0] as usize;
            occupancies[face.cluster_idx] += 1;
        }

        //assert!(*occupancies.iter().max().unwrap() <= 126);

        self.clusters = vec![
            common::PartitionInfo {
                child_group_index: None,
                group_index: usize::MAX,
                tight_bound: Default::default()
            };
            cluster_count
        ];

        Ok(())
    }

    /// Group clusters, recording the bounding boxes for all groups.
    pub fn group(
        &mut self,
        config: &metis::PartitioningConfig,
        verts: &[glam::Vec4],
    ) -> Result<usize, metis::PartitioningError> {
        let group_count = self.clusters.len().div_ceil(4);

        assert!(config.force_contiguous_partitions);

        println!(
            "Partitioning into {group_count} groups from {} partitions",
            self.clusters.len()
        );

        let cluster_graph = self.generate_cluster_graph();

        // create new array of groups, and remember the old groups
        let mut new_groups = vec![
            common::GroupInfo {
                tris: 0,
                monotonic_bound: Default::default(),
                partitions: Vec::new(),
                group_neighbours: BTreeSet::new(),
            };
            group_count
        ];

        //std::mem::swap(&mut self.groups, &mut new_groups);

        // Tell each partition what group they now belong to.
        if group_count != 1 {
            for (part, &group) in config
                .partition_from_graph(group_count as u32, &cluster_graph)?
                .iter()
                .enumerate()
            {
                self.clusters[part].group_index = group as usize;
            }
        } else {
            for p in &mut self.clusters {
                p.group_index = 0;
            }
        };

        // Record the partitions that each of these groups come from
        for (part, info) in self.clusters.iter().enumerate() {
            new_groups[info.group_index].partitions.push(part);

            for n in cluster_graph.neighbors(petgraph::graph::node_index(part)) {
                new_groups[info.group_index]
                    .group_neighbours
                    .insert(self.clusters[n.index()].group_index);
            }
        }

        // MONOTONIC BOUND ------ get sums of positions
        for (_fid, f) in self.iter_faces() {
            let f_group_info = &mut new_groups[self.clusters[f.cluster_idx].group_index];
            f_group_info.tris += 1;
            f_group_info
                .monotonic_bound
                .translate(verts[self.get_edge(f.edge).vert_origin.0].xyz());
        }

        // Take averages
        for g in &mut new_groups {
            g.monotonic_bound.normalise(g.tris);
        }

        // Find radii of groups, now that they have accurate positions
        for (_fid, f) in self.iter_faces() {
            let f_group_info = &mut new_groups[self.clusters[f.cluster_idx].group_index];

            f_group_info
                .monotonic_bound
                .include_point(verts[self.get_edge(f.edge).vert_origin.0].xyz());
        }
        println!(
            "Including child bounds with {} old groups",
            new_groups.len()
        );

        for g in &mut new_groups {
            // SQRT each group
            //    g.monotonic_bound.radius = g.monotonic_bound.radius.sqrt();

            // Each group also must envelop all the groups it is descended from,
            // as our partitions must do the same, as we base them off group info

            for p in &g.partitions {
                if let Some(child_group_index) = self.clusters[*p].child_group_index {
                    let child_group = &self.groups[child_group_index];
                    // combine groups radius
                    g.monotonic_bound
                        .include_sphere(&child_group.monotonic_bound);
                }
            }
        }

        self.groups = new_groups;

        #[cfg(test)]
        {
            //Assert that we have made good groups
            //TODO: Split graph into contiguous segments beforehand

            let groups = self.generate_group_graphs();

            for g in &groups {
                super::graph::test::assert_contiguous_graph(g);
            }
        }

        Ok(group_count)
    }

    /// Within each group, split triangles into two completely new partitions, so as not to preserve any old seams between ancient partitions
    /// Ensures the data structure is seamless with changing seams! Yippee!
    /// Will update the partitions list, but groups list will still refer to old partitions. To find out what group these should be in, before regrouping,
    /// look at `child_group_index`
    pub fn partition_within_groups(
        &mut self,
        config: &metis::PartitioningConfig,
        parts_per_group: Option<u32>,
        tris_per_cluster: Option<u32>,
    ) -> Result<usize, metis::PartitioningError> {
        let graphs = self.generate_group_graphs();

        println!("Partitioning {} groups into sub-partitions", graphs.len());

        // Ungrouped partitions but with a dependence on an old group
        let mut new_partitions = Vec::new();

        for (i_group, graph) in graphs.iter().enumerate() {
            // TODO: fine tune so we get 64/126 meshlets

            if graph.node_count() == 0 {
                println!("WARNING: Group {i_group} face graph has no nodes!");
                continue;
            }

            let parts = if let Some(parts_per_group) = parts_per_group {
                parts_per_group
            } else {
                (graph.node_count() as u32).div_ceil(tris_per_cluster.unwrap())
            };

            #[cfg(test)]
            {
                super::graph::test::assert_contiguous_graph(graph);
            }

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

            assert_eq!(i_group, child_group);

            // Update partitions of the actual triangles
            for x in graph.node_indices() {
                self.get_face_mut(graph[x]).cluster_idx =
                    new_partitions.len() + part[x.index()] as usize;
            }
            // If we have not been grouped yet,
            let child_group_index = if self.groups.len() == 0 {
                None
            } else {
                Some(i_group)
            };

            for _ in 0..parts {
                //    self.groups[group].partitions.push(new_partitions.len());

                new_partitions.push(common::PartitionInfo {
                    child_group_index,
                    group_index: usize::MAX,
                    tight_bound: Default::default(), //TODO:
                })
            }
        }
        self.clusters = new_partitions;

        Ok(self.clusters.len())
        //Ok(groups)
    }
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashSet;

    use anyhow::Context;

    use crate::mesh::winged_mesh::{test::TEST_MESH_LOW, MeshError};

    use super::*;

    #[test]
    pub fn test_group_neighbours() -> anyhow::Result<()> {
        let test_config = &metis::PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_LOW);

        mesh.partition_full_mesh(test_config, 200)?;
        mesh.group(test_config, &verts)?;

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

            for &g in &groups {
                for &g2 in &groups {
                    if g != g2 {
                        if !mesh.groups[g].group_neighbours.contains(&g2) {
                            println!("{:?}", mesh.groups[g].group_neighbours);
                            println!("{:?}", mesh.groups[g2].group_neighbours);
                            println!("{:?}", groups);

                            Err(MeshError::InvalidNeighbours(g, g2))
                                .context(MeshError::InvalidGroup(g))?;
                        }
                    }
                }
            }
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
