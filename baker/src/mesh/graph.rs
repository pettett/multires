use std::collections::{HashMap, HashSet};

use petgraph::visit::EdgeRef;

use super::winged_mesh::{FaceID, WingedMesh};

impl WingedMesh {
    /// Generates a graph that is the dual of this mesh - connections from each face to their neighbours
    pub fn generate_face_graph(&self) -> petgraph::graph::UnGraph<FaceID, ()> {
        //TODO: Give lower weight to grouping partitions that have not been recently grouped, to ensure we are
        // constantly overwriting old borders with remeshes

        let mut graph = petgraph::Graph::with_capacity(
            self.cluster_count(),
            // Estimate each partition hits roughly 3 other partitions
            self.cluster_count() * 3,
        );
        let mut ids = HashMap::with_capacity(self.face_count());
        for (fid, _f) in self.iter_faces() {
            // Each node should directly correspond to a partition
            ids.insert(fid, graph.add_node(fid));
        }

        for (fid, face) in self.iter_faces() {
            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self.get_edge(e).twin {
                    let other_face = &self.get_edge(twin).face;

                    graph.update_edge(ids[&fid], ids[other_face], ());
                }
            }
        }

        graph
    }

    /// `generate_face_graph`, but with one graph per group
    pub fn generate_group_graphs(&self) -> Vec<petgraph::Graph<FaceID, (), petgraph::Undirected>> {
        let group_count = self.groups.len().max(1);

        let mut graphs =
            vec![petgraph::graph::UnGraph::<FaceID, ()>::new_undirected(); group_count];

        // Stores vecs of face IDs, which should be associated with data in graphs
        let mut ids = vec![Vec::new(); group_count];
        // only needs a single face map, as each face can only be part of one group
        let mut faces = HashMap::new();

        // Give every triangle a node
        for (fid, face) in self.iter_faces() {
            let g = self.clusters[face.cluster_idx].group_index;
            let n = graphs[g].add_node(fid);

            // indexes should correspond
            assert_eq!(n.index(), ids[g].len());
            ids[g].push(fid);

            faces.insert(fid, n);
        }
        // Apply links between nodes
        for (fid, face) in self.iter_faces() {
            let g = self.clusters[face.cluster_idx].group_index;

            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self.get_edge(e).twin {
                    let other_face = self.get_edge(twin).face;
                    let o_g = self.clusters[self.get_face(other_face).cluster_idx].group_index;

                    if g == o_g {
                        graphs[g].update_edge(faces[&fid], faces[&other_face], ());
                    }
                }
            }
        }
        graphs
    }

    /// Generates a graph of all partitions and their neighbours.
    /// A partition neighbours another one iff there is some triangle in each that share an edge.
    /// We add an edge for each linking triangle, to record 'weights' for partitioning.
    pub fn generate_cluster_graph(&self) -> petgraph::graph::UnGraph<i32, ()> {
        //TODO: Give lower weight to grouping partitions that have not been recently grouped, to ensure we are
        // constantly overwriting old borders with remeshes

        let mut graph = petgraph::Graph::with_capacity(
            self.cluster_count(),
            // Estimate each partition hits roughly 3 other partitions
            self.cluster_count() * 3,
        );

        for p in 0..self.cluster_count() {
            // Each node should directly correspond to a partition
            assert_eq!(p, graph.add_node(0).index());
        }

        //let mut map: HashMap<usize, HashMap<usize, usize>> = HashMap::new();

        for (_fid, face) in self.iter_faces() {
            let n0 = petgraph::graph::NodeIndex::new(face.cluster_idx);

            *graph.node_weight_mut(n0).unwrap() += 1;

            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self.get_edge(e).twin {
                    let other_face = &self.get_face(self.get_edge(twin).face);

                    //let c0 = face.cluster_idx.min(other_face.cluster_idx);
                    //let c1 = face.cluster_idx.max(other_face.cluster_idx);

                    if face.cluster_idx != other_face.cluster_idx {
                        //*map.entry(c0).or_default().entry(c1).or_default() += 1;

                        let n1 = petgraph::graph::NodeIndex::new(other_face.cluster_idx);

                        // let w = graph
                        //     .find_edge(n0, n1)
                        //     .map(|e| graph.edge_weight(e))
                        //     .flatten();

                        graph.add_edge(n0, n1, ());
                    }
                }
            }
        }

        // // Add extra connections on the highest weighted neighbour for each node
        // for n in graph.node_indices() {
        //     let total: i32 = graph.edges(n).map(|x| x.weight()).sum();

        //     let edges = graph
        //         .edges(n)
        //         .map(|e| (e.source(), e.target(), *e.weight()))
        //         .collect::<Vec<_>>();

        //     let len = edges.len() as i32;

        //     let avg = total / len;

        //     for (src, tgt, w) in edges {
        //         if w > avg {
        //             graph.add_edge(src, tgt, 0);
        //         }
        //         if w * 2 > avg * 3 {
        //             graph.add_edge(src, tgt, 0);
        //         }
        //         if w > total * 2 {
        //             graph.add_edge(src, tgt, 0);
        //         }
        //     }
        // }

        // let mut min_count = 100000;
        // let mut max_count = 0;
        // for (c0, connectings) in &map {
        //     for (c1, &count) in connectings {
        //         min_count = min_count.min(count);
        //         max_count = max_count.max(count);
        //     }
        // }

        // println!("{min_count} - {max_count}");

        // let mid1 = (min_count + max_count * 3) / 4;
        // let mid2 = (min_count * 3 + max_count) / 4;

        // for (c0, connectings) in map {
        //     for (c1, count) in connectings {
        //         // Add an edge for *each* shared edge, recording how
        //         // linked the two partitions are (how much 'cruft' is shared)

        //         let num = if count > mid1 {
        //             2
        //         } else if count > mid2 {
        //             1
        //         } else {
        //             1
        //         };
        //         for _ in 0..num {
        //             graph.add_edge(
        //                 petgraph::graph::NodeIndex::new(c0),
        //                 petgraph::graph::NodeIndex::new(c1),
        //                 (),
        //             );
        //         }
        //     }
        // }

        graph
    }
}
#[cfg(test)]
pub mod test {
    pub const DOT_OUT: &str = "baker\\graph.gv";

    pub const HIERARCHY_SVG_OUT: &str = "baker\\hierarchy_graph.svg";
    pub const PART_SVG_OUT: &str = "baker\\part_graph.svg";

    pub const FACE_SVG_OUT: &str = "baker\\face_graph.svg";

    pub const FACE_SVG_OUT_2: &str = "baker\\face_graph2.svg";
    pub const ERROR_SVG_OUT: &str = "error.svg";

    const COLS: [&str; 10] = [
        "red",
        "blue",
        "green",
        "aqua",
        "cornflowerblue",
        "darkgoldenrod1",
        "deeppink",
        "indigo",
        "orchid",
        "peru",
    ];
    pub fn assert_contiguous_graph<V: std::fmt::Debug, E: std::fmt::Debug>(
        graph: &petgraph::Graph<V, E, petgraph::Undirected>,
    ) {
        let n0 = petgraph::graph::node_index(0);
        let mut dfs_space = petgraph::algo::DfsSpace::default();

        for i in graph.node_indices() {
            if !petgraph::algo::has_path_connecting(&graph, n0, i, Some(&mut dfs_space)) {
                println!("Graph is not contiguous, outputting error...");

                graph::petgraph_to_svg(
                    graph,
                    ERROR_SVG_OUT,
                    &|_, _| String::new(),
                    graph::GraphSVGRender::Directed {
                        node_label: common::graph::Label::None,
                    },
                )
                .unwrap();

                assert!(
                    false,
                    "Graph is not contiguous. Outputted error graph to {}",
                    ERROR_SVG_OUT
                );
            }
        }
    }
    #[test]
    pub fn generate_face_graph() -> Result<(), Box<dyn error::Error>> {
        let test_config = &metis::PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let mesh = TEST_MESH_PLANE_LOW;
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_full_mesh(test_config, 9)?;

        println!(
            "Faces: {}, Verts: {}, Partitions: {}",
            mesh.face_count(),
            mesh.vert_count(),
            mesh.clusters.len()
        );

        let mut graph = mesh.generate_face_graph();

        graph::petgraph_to_svg(
            &graph,
            FACE_SVG_OUT,
            &|_, (_n, &fid)| {
                let p = fid.center(&mesh, &verts);
                let part = mesh.get_face(fid).cluster_idx;
                format!(
                    "shape=point, color={}, pos=\"{},{}\"",
                    COLS[part % COLS.len()],
                    p.x * 200.0,
                    p.z * 200.0,
                )
            },
            graph::GraphSVGRender::Undirected {
                positions: true,
                edge_label: Default::default(),
            },
        )?;

        graph.retain_edges(|g, e| {
            let (v1, v2) = g.edge_endpoints(e).unwrap();

            let &p1 = g.node_weight(v1).unwrap();
            let &p2 = g.node_weight(v2).unwrap();

            p1 == p2
        });

        graph::petgraph_to_svg(
            &graph,
            FACE_SVG_OUT_2,
            &|_, (_n, &fid)| {
                let p = fid.center(&mesh, &verts);
                let part = mesh.get_face(fid).cluster_idx;
                format!(
                    "shape=point, color={}, pos=\"{},{}\"",
                    COLS[part % COLS.len()],
                    p.x * 200.0,
                    p.z * 200.0,
                )
            },
            graph::GraphSVGRender::Undirected {
                positions: true,
                edge_label: Default::default(),
            },
        )?;

        Ok(())
    }

    #[test]
    pub fn generate_partition_graph() -> Result<(), Box<dyn error::Error>> {
        let test_config = &metis::PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let mesh = TEST_MESH_PLANE_LOW;
        let (mut mesh, _verts) = WingedMesh::from_gltf(mesh);

        println!("Faces: {}, Verts: {}", mesh.face_count(), mesh.vert_count());

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_within_groups(test_config, None, Some(60))?;

        graph::petgraph_to_svg(
            &mesh.generate_cluster_graph(),
            PART_SVG_OUT,
            &|_, _| format!("shape=point"),
            graph::GraphSVGRender::Directed {
                node_label: common::graph::Label::None,
            },
        )?;

        Ok(())
    }

    #[test]
    pub fn generate_partition_hierarchy_graph() -> Result<(), Box<dyn error::Error>> {
        let test_config = &metis::PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let mesh = TEST_MESH_LOW;
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        println!("Faces: {}, Verts: {}", mesh.face_count(), mesh.vert_count());

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_within_groups(test_config, None, Some(60))?;

        mesh.group(test_config, &verts)?;
        let mut graph: petgraph::Graph<(), ()> = petgraph::Graph::new();

        let mut old_part_nodes: Vec<_> =
            mesh.clusters.iter().map(|_o| graph.add_node(())).collect();

        // Record a big colour map for node indexes, to show grouping
        let mut colouring = HashMap::new();

        let mut seen_groups = 0;

        loop {
            for (i, &n) in old_part_nodes.iter().enumerate() {
                colouring.insert(n, mesh.clusters[i].group_index + seen_groups);
            }

            seen_groups += mesh.groups.len();

            mesh.partition_within_groups(test_config, Some(2), None)?;

            let new_part_nodes: Vec<_> =
                mesh.clusters.iter().map(|_o| graph.add_node(())).collect();

            for (new_p_i, new_p) in mesh.clusters.iter().enumerate() {
                let g_i = new_p.child_group_index.unwrap();

                for &old_p_i in &mesh.groups[g_i].partitions {
                    graph.add_edge(old_part_nodes[old_p_i], new_part_nodes[new_p_i], ());
                }
            }

            old_part_nodes = new_part_nodes;

            mesh.group(test_config, &verts)?;

            if mesh.clusters.len() <= 2 {
                break;
            }
        }

        for (i, &n) in old_part_nodes.iter().enumerate() {
            colouring.insert(n, mesh.clusters[i].group_index + seen_groups);
        }

        graph::petgraph_to_svg(
            &graph,
            HIERARCHY_SVG_OUT,
            &|_, (n, _)| format!("shape=point, color={}", COLS[colouring[&n] % COLS.len()]),
            graph::GraphSVGRender::Directed {
                node_label: common::graph::Label::None,
            },
        )?;

        Ok(())
    }

    use std::{collections::HashMap, error};

    use common::graph;

    use crate::mesh::winged_mesh::{
        test::{TEST_MESH_LOW, TEST_MESH_PLANE_LOW},
        WingedMesh,
    };
}
