use std::collections::BTreeSet;

use super::winged_mesh::WingedMesh;
use common::graph::petgraph_to_svg;
use glam::Vec4Swizzles;

impl WingedMesh {
    pub fn partition_contiguous(&mut self) -> usize {
        println!("Wiping partitions");
        // Wipe partitions
        for (_, f) in self.iter_faces_mut() {
            f.part = 0;
        }

        let mut p = 1;
        let mut search = Vec::new();

        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.face_count() as _);

        loop {
            assert_eq!(search.len(), 0);
            // select a random face to start the search

            for (fid, f) in self.iter_faces() {
                if f.part == 0 {
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
                self.get_face_mut(fid).part = p;

                // Search for unmarked
                let e0 = self.get_face(fid).edge;
                let e1 = self.get_edge(e0).edge_left_cw;
                let e2 = self.get_edge(e0).edge_left_ccw;

                for e in [e0, e1, e2] {
                    if let Some(t) = self.get_edge(e).twin {
                        let f = self.get_edge(t).face;
                        if self.get_face(f).part == 0 {
                            // splitting by contiguous, we should not be able to access others

                            search.push(f);
                        } else {
                            assert_eq!(self.get_face(f).part, p);
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

        let part = config.partition_from_graph(partitions, &mesh_dual)?;

        assert_eq!(part.len(), self.face_count());

        let mut max_part = 0;
        for (fid, face) in self.iter_faces_mut() {
            // Some faces will have already been removed
            face.part = part[fid.0] as usize;
            max_part = max_part.max(face.part)
        }

        self.partitions = vec![
            common::PartitionInfo {
                child_group_index: None,
                group_index: usize::MAX,
                tight_bound: Default::default()
            };
            max_part + 1
        ];

        Ok(())
    }

    /// Group clusters, recording the bounding boxes for all groups.
    pub fn group(
        &mut self,
        config: &metis::PartitioningConfig,
        verts: &[glam::Vec4],
    ) -> Result<usize, metis::PartitioningError> {
        let group_count = self.partitions.len().div_ceil(4);
        println!(
            "Partitioning into {group_count} groups from {} partitions",
            self.partitions.len()
        );

        let graph = self.generate_partition_graph();

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
                .partition_from_graph(group_count as u32, &graph)?
                .iter()
                .enumerate()
            {
                self.partitions[part].group_index = group as usize;
            }
        } else {
            for p in &mut self.partitions {
                p.group_index = 0;
            }
        };

        // Record the partitions that each of these groups come from
        for (part, info) in self.partitions.iter().enumerate() {
            new_groups[info.group_index].partitions.push(part);

            for n in graph.neighbors(petgraph::graph::node_index(part)) {
                new_groups[info.group_index]
                    .group_neighbours
                    .insert(self.partitions[n.index()].group_index);
            }
        }

        // MONOTONIC BOUND ------ get sums of positions
        for (_fid, f) in self.iter_faces() {
            let f_group_info = &mut new_groups[self.partitions[f.part].group_index];
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
            let f_group_info = &mut new_groups[self.partitions[f.part].group_index];

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
                if let Some(child_group_index) = self.partitions[*p].child_group_index {
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
            if config.force_contiguous_partitions {
                let groups = self.generate_group_graphs();

                for g in &groups {
                    super::graph::test::assert_contiguous_graph(g);
                }
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
                (graph.node_count() as u32).div_ceil(60)
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
                        "svg/error_partition_within_group.svg",
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
            let child_group = self.partitions
                [self.get_face(graph[petgraph::graph::node_index(0)]).part]
                .group_index;

            assert_eq!(i_group, child_group);

            // Update partitions of the actual triangles
            for x in graph.node_indices() {
                self.get_face_mut(graph[x]).part = new_partitions.len() + part[x.index()] as usize;
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
        self.partitions = new_partitions;

        Ok(self.partitions.len())
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
                    mesh.partitions[mesh.get_face(mesh.get_edge(out).face).part].group_index,
                );
            }
            for &out in vert.incoming_edges() {
                groups.insert(
                    mesh.partitions[mesh.get_face(mesh.get_edge(out).face).part].group_index,
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
