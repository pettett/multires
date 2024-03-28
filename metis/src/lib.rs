#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod partition;
pub use partition::*;

#[cfg(test)]
pub mod test {
    use common::graph;

    use super::*;

    #[test]
    fn test_triangle_plane() {
        let graph = graph::generate_triangle_plane::<12, 6>();

        graph::petgraph_to_svg(
            &graph,
            "..\\svg\\triangle_plane.svg",
            &|_, _| String::new(),
            graph::GraphSVGRender::Undirected {
                positions: false,
                edge_label: Default::default(),
            },
        )
        .unwrap();
    }

    #[test]
    fn test_unweighted_edges() {
        let graph = graph::generate_triangle_plane::<6, 4>();

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config.partition_from_graph(3, &graph).unwrap();

        graph::petgraph_to_svg(
            &graph,
            "..\\svg\\triangle_plane_unweighted_group.svg",
            &|_, (i, _)| {
                format!(
                    "color={}",
                    graph::COLS[p[i.index()] as usize % graph::COLS.len()]
                )
            },
            graph::GraphSVGRender::Undirected {
                positions: false,
                edge_label: common::graph::Label::None,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_weighted_edges() {
        let graph =
            graph::generate_triangle_plane_weighted::<6, 4, _>(|x, y| match (x.min(y), x.max(y)) {
                (0, 1) => 10,
                (0, 7) => 10,
                (10, 17) => 100,
                (10, 11) => 100,
                (8, 9) => 100,
                (9, 10) => 100,
                (14, 21) => 100,
                (14, 15) => 100,
                (2, 3) => 100,
                _ => 1,
            });

        let test_config: &PartitioningConfig = &MultilevelKWayPartitioningConfig {
            //    method: PartitioningMethod::MultilevelKWay,
            //objective_type: Some(ObjectiveType::Volume),
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            u_factor: Some(100),
            ..Default::default()
        }
        .into();

        let p = test_config
            .partition_from_edge_weighted_graph(2, &graph)
            .unwrap();

        println!(
            "{} {}",
            p.iter().filter(|&&x| x == 0).count(),
            p.iter().filter(|&&x| x == 1).count()
        );

        graph::petgraph_to_svg(
            &graph,
            "..\\svg\\triangle_plane_unweighted_group.svg",
            &|_, (i, _)| {
                format!(
                    "color={}",
                    graph::COLS[p[i.index()] as usize % graph::COLS.len()]
                )
            },
            graph::GraphSVGRender::Undirected {
                positions: false,
                edge_label: common::graph::Label::Weight,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_unweighted_duplicated_edges() {
        let mut graph = graph::generate_triangle_plane::<12, 8>();

        // Demonstrates that we can use weighting (that works, weights array breaks the contiguous graphs) by adding extra edges connecting nodes
        // We use this principle in generating the group graph

        for (s, e) in [
            (4, 11),
            (21, 22),
            (6, 13),
            (20, 19),
            (16, 23),
            (15, 14),
            (3, 4),
            (13, 12),
            (3, 2),
            (7, 8),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (7, 6),
            (2, 1),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (7, 0),
            (8, 15),
            (7, 0),
            (7, 0),
        ] {
            graph.add_edge(
                petgraph::graph::node_index(s),
                petgraph::graph::node_index(e),
                (),
            );
        }

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            u_factor: Some(100),
            //minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let parts = 4;

        let (p, _all_parts) = test_config
            .exact_partition_onto_graph(parts, &graph)
            .unwrap();

        //println!("{p:?}");
        println!(
            "{} {}",
            p.node_weights().filter(|&&x| x == 0).count(),
            p.node_weights().filter(|&&x| x == 1).count()
        );

        println!(
            "{} {}",
            p.node_weights().filter(|&&x| x == 0).count(),
            p.node_weights().filter(|&&x| x == 1).count()
        );

        graph::petgraph_to_svg(
            &graph,
            "..\\svg\\triangle_plane_weighted_group.svg",
            &|_, (i, _)| {
                format!(
                    "color={}",
                    graph::COLS[*p.node_weight(i).unwrap() as usize % graph::COLS.len()]
                )
            },
            graph::GraphSVGRender::Undirected {
                positions: false,
                edge_label: common::graph::Label::None,
            },
        )
        .unwrap();

        for g in graph::filter_nodes_by_weight(&p, 0..(parts as _)) {
            assert!(graph::graph_contiguous(&g));
        }
    }

    #[test]
    fn test_huge_exact() {
        // let graph = graph::generate_triangle_plane::<1000, 10000>();
        let N: usize = 1000;
        let M: usize = 1000;
        let mut graph = petgraph::Graph::with_capacity(N * M, N * M * 3);
        // Add nodes
        let mut nodes = vec![vec![petgraph::graph::node_index(0); N]; M];
        for m in 0..M {
            for n in 0..N {
                nodes[m][n] = graph.add_node(());
            }
        }

        // Add edges
        for m in 0..M {
            for n in 0..N {
                let a = nodes[m][n];
                for _i in 0..100 {
                    if n < N - 1 {
                        graph.update_edge(a, nodes[m][n + 1], ());

                        if m < M - 1 && n % 2 == 0 {
                            graph.update_edge(a, nodes[m + 1][n + 1], ());
                        }
                    }

                    if n > 0 {
                        graph.update_edge(a, nodes[m][n - 1], ());

                        if m > 0 && n % 2 == 1 {
                            graph.update_edge(a, nodes[m - 1][n - 1], ());
                        }
                    }
                }
            }
        }

        // Demonstrates that we can use weighting (that works, weights array breaks the contiguous graphs) by adding extra edges connecting nodes
        // We use this principle in generating the group graph

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            // u_factor: Some(100),
            //minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let parts = 4;

        let (p, _all_parts) = test_config
            .exact_partition_onto_graph(parts, &graph)
            .unwrap();

        for g in graph::filter_nodes_by_weight(&p, 0..(parts as _)) {
            assert!(graph::graph_contiguous(&g));
        }
    }

    #[test]
    fn test_2_node_graph() {
        let mut g = petgraph::Graph::new_undirected();

        let a = g.add_node(());

        let b = g.add_node(());

        g.add_edge(a, b, ());

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config.partition_from_graph(2, &g).unwrap();

        assert_ne!(p[0], p[1]);
    }

    #[test]
    fn test_pack_partitions() {
        let mut p = vec![1, 1, 2, 2, 4, 4, 4, 4, 1, 1];

        pack_partitioning(&mut p);

        println!("{:?}", p);
    }
}
