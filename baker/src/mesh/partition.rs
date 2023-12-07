use glam::Vec4Swizzles;

#[cfg(test)]
use crate::mesh::graph::test::assert_contiguous_graph;

use super::winged_mesh::WingedMesh;

impl WingedMesh {
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
        for (fid, mut face) in self.iter_faces_mut() {
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
                partitions: Vec::new()
            };
            group_count
        ];

        //std::mem::swap(&mut self.groups, &mut new_groups);

        // Partition -> Group
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

        for (part, info) in self.partitions.iter().enumerate() {
            new_groups[info.group_index].partitions.push(part);
        }

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

        // Find radii of groups
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

            let part = config.partition_from_graph(parts, graph)?;

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
