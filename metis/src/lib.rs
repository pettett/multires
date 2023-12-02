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
        let graph = generate_triangle_plane::<12, 6, _>(|_, _| ());

        petgraph_to_svg(
            &graph,
            "svg\\triangle_plane.svg",
            &|_, _| String::new(),
            GraphSVGRender::Undirected { positions: false },
        )
        .unwrap();
    }

    #[test]
    fn test_weighted_edges() {
        // We kinda assume each row will be grouped separately
        let graph = generate_triangle_plane::<12, 6, _>(
            |(x1, y1), (x2, y2)| {
                if y1 == 4 && x1 == 4 {
                    10
                } else {
                    0
                }
            },
        );

        let test_config = &PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let p = test_config
            .partition_from_edge_weighted_graph(6, &graph)
            .unwrap();

        petgraph_to_svg(
            &graph,
            "svg\\triangle_plane_row_divided.svg",
            &|_, (i, _)| format!("color={}", COLS[p[i.index()] as usize % COLS.len()]),
            GraphSVGRender::Undirected { positions: false },
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
