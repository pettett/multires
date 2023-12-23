use std::collections::HashMap;

use super::winged_mesh::{FaceID, WingedMesh};

impl WingedMesh {
    /// Generates a graph that is the dual of this mesh - connections from each face to their neighbours
    pub fn generate_face_graph(&self) -> petgraph::graph::UnGraph<FaceID, ()> {
        //TODO: Give lower weight to grouping partitions that have not been recently grouped, to ensure we are
        // constantly overwriting old borders with remeshes

        let mut graph = petgraph::Graph::with_capacity(
            self.partition_count(),
            // Estimate each partition hits roughly 3 other partitions
            self.partition_count() * 3,
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
            let g = self.partitions[face.part].group_index;
            let n = graphs[g].add_node(fid);

            // indexes should correspond
            assert_eq!(n.index(), ids[g].len());
            ids[g].push(fid);

            faces.insert(fid, n);
        }
        // Apply links between nodes
        for (fid, face) in self.iter_faces() {
            let g = self.partitions[face.part].group_index;

            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self.get_edge(e).twin {
                    let other_face = self.get_edge(twin).face;
                    let o_g = self.partitions[self.get_face(other_face).part].group_index;

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
    pub fn generate_partition_graph(&self) -> petgraph::graph::UnGraph<(), ()> {
        //TODO: Give lower weight to grouping partitions that have not been recently grouped, to ensure we are
        // constantly overwriting old borders with remeshes

        let mut graph = petgraph::Graph::with_capacity(
            self.partition_count(),
            // Estimate each partition hits roughly 3 other partitions
            self.partition_count() * 3,
        );

        for p in 0..self.partition_count() {
            // Each node should directly correspond to a partition
            assert_eq!(p, graph.add_node(()).index());
        }

        for (_fid, face) in self.iter_faces() {
            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self.get_edge(e).twin {
                    let other_face = &self.get_face(self.get_edge(twin).face);

                    if face.part != other_face.part {
                        // Add an edge for *each* shared edge, recording how
                        // linked the two partitions are (how much 'cruft' is shared)
                        graph.add_edge(
                            petgraph::graph::NodeIndex::new(face.part),
                            petgraph::graph::NodeIndex::new(other_face.part),
                            (),
                        );
                    }
                }
            }
        }

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
    pub const ERROR_SVG_OUT: &str = "baker\\error.svg";

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
            mesh.partitions.len()
        );

        let mut graph = mesh.generate_face_graph();

        graph::petgraph_to_svg(
            &graph,
            FACE_SVG_OUT,
            &|_, (_n, &fid)| {
                let p = fid.center(&mesh, &verts);
                let part = mesh.get_face(fid).part;
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
                let part = mesh.get_face(fid).part;
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
        mesh.partition_within_groups(test_config, None)?;

        graph::petgraph_to_svg(
            &mesh.generate_partition_graph(),
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
        mesh.partition_within_groups(test_config, None)?;

        mesh.group(test_config, &verts)?;
        let mut graph: petgraph::Graph<(), ()> = petgraph::Graph::new();

        let mut old_part_nodes: Vec<_> =
            mesh.partitions.iter().map(|_o| graph.add_node(())).collect();

        // Record a big colour map for node indexes, to show grouping
        let mut colouring = HashMap::new();

        let mut seen_groups = 0;

        loop {
            for (i, &n) in old_part_nodes.iter().enumerate() {
                colouring.insert(n, mesh.partitions[i].group_index + seen_groups);
            }

            seen_groups += mesh.groups.len();

            mesh.partition_within_groups(test_config, Some(2))?;

            let new_part_nodes: Vec<_> =
                mesh.partitions.iter().map(|_o| graph.add_node(())).collect();

            for (new_p_i, new_p) in mesh.partitions.iter().enumerate() {
                let g_i = new_p.child_group_index.unwrap();

                for &old_p_i in &mesh.groups[g_i].partitions {
                    graph.add_edge(old_part_nodes[old_p_i], new_part_nodes[new_p_i], ());
                }
            }

            old_part_nodes = new_part_nodes;

            mesh.group(test_config, &verts)?;

            if mesh.partitions.len() <= 2 {
                break;
            }
        }

        for (i, &n) in old_part_nodes.iter().enumerate() {
            colouring.insert(n, mesh.partitions[i].group_index + seen_groups);
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
