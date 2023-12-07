#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod partition;
pub use partition::*;

#[cfg(test)]
pub mod test {
    use common::graph::{generate_triangle_plane, petgraph_to_svg, GraphSVGRender, COLS};

    use super::*;

    #[test]
    fn test_triangle_plane() {
        let graph = generate_triangle_plane::<12, 6>();

        petgraph_to_svg(
            &graph,
            "svg\\triangle_plane.svg",
            &|_, _| String::new(),
            GraphSVGRender::Undirected {
                positions: false,
                edge_label: Default::default(),
            },
        )
        .unwrap();
    }

    #[test]
    fn test_unweighted_edges() {
        let graph = generate_triangle_plane::<6, 4>();

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config.partition_from_graph(3, &graph).unwrap();

        petgraph_to_svg(
            &graph,
            "svg\\triangle_plane_unweighted_group.svg",
            &|_, (i, _)| format!("color={}", COLS[p[i.index()] as usize % COLS.len()]),
            GraphSVGRender::Undirected {
                positions: false,
                edge_label: common::graph::Label::None,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_weighted_edges() {
        let mut graph = generate_triangle_plane::<6, 4>();

        // Demonstrates that we can use weighting (that works, weights array breaks the contiguous graphs) by adding extra edges connecting nodes
        // We use this principle in generating the group graph

        graph.add_edge(
            petgraph::graph::node_index(0),
            petgraph::graph::node_index(1),
            (),
        );

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config.partition_from_graph(3, &graph).unwrap();

        petgraph_to_svg(
            &graph,
            "svg\\triangle_plane_weighted_group.svg",
            &|_, (i, _)| format!("color={}", COLS[p[i.index()] as usize % COLS.len()]),
            GraphSVGRender::Undirected {
                positions: false,
                edge_label: common::graph::Label::None,
            },
        )
        .unwrap();
    }

    #[test]
    fn test_2_node_graph() {
        let mut g = petgraph::Graph::new_undirected();

        let a = g.add_node(());

        let b = g.add_node(());

        g.add_edge(a, b, ());

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config.partition_from_graph(2, &g).unwrap();

        assert_ne!(p[0], p[1]);
    }
}
