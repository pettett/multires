use anyhow::Context;
use core::fmt;
use petgraph::dot;
use std::{fs, ops::Range, path, process, vec};

/// Generate a graph corresponding to the dual mesh of a mesh generated from triangulating a grid
///
/// For example, the triangle grid:
/// / -------------------
/// / |A/|C/|E/|G/|I/|K/|
/// / |/B|/D|/F|/H|/J|/L|
/// / -------------------
/// / |M/|O/|Q/|S/|U/|W/|
/// / |/N|/P|/R|/T|/V|/X|
/// / -------------------
///
/// Will have a graph with node Q connecting to R, P, and F
///
/// On an n by m grid, each row has n triangles, and triangle i, j will connect to
/// - (i-1, j).
/// - (i+1,j).
/// - if i even to (i+1,j+1).
/// - else if i odd to (i-1, j-1).
///
pub fn generate_triangle_plane_weighted<const N: usize, const M: usize, E>(
    f: impl Fn(usize, usize) -> E,
) -> petgraph::Graph<(), E, petgraph::Undirected> {
    let mut graph = petgraph::Graph::with_capacity(N * M, N * M * 3);
    // Add nodes
    let mut nodes = [[petgraph::graph::node_index(0); N]; M];
    for m in 0..M {
        for n in 0..N {
            nodes[m][n] = graph.add_node(());
        }
    }

    // Add edges
    for m in 0..M {
        for n in 0..N {
            let a = nodes[m][n];

            if n < N - 1 {
                graph.update_edge(a, nodes[m][n + 1], f(a.index(), nodes[m][n + 1].index()));

                if m < M - 1 && n % 2 == 0 {
                    graph.update_edge(
                        a,
                        nodes[m + 1][n + 1],
                        f(a.index(), nodes[m + 1][n + 1].index()),
                    );
                }
            }

            if n > 0 {
                graph.update_edge(a, nodes[m][n - 1], f(a.index(), nodes[m][n - 1].index()));

                if m > 0 && n % 2 == 1 {
                    graph.update_edge(
                        a,
                        nodes[m - 1][n - 1],
                        f(a.index(), nodes[m - 1][n - 1].index()),
                    );
                }
            }
        }
    }

    graph
}

pub fn generate_triangle_plane<const N: usize, const M: usize>(
) -> petgraph::Graph<(), (), petgraph::Undirected> {
    generate_triangle_plane_weighted::<N, M, _>(|_, _| ())
}

#[derive(Default)]
pub enum Label {
    Index,
    #[default]
    Weight,
    None,
}
pub enum GraphSVGRender {
    Directed { node_label: Label },
    Undirected { edge_label: Label, positions: bool },
}
pub const COLS: [&str; 10] = [
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
pub fn petgraph_to_svg<
    G: petgraph::visit::IntoNodeReferences
        + petgraph::visit::IntoEdgeReferences
        + petgraph::visit::NodeIndexable
        + petgraph::visit::GraphProp,
>(
    graph: G,
    out: impl AsRef<path::Path>,
    get_node_attrs: &dyn Fn(&G, G::NodeRef) -> String,
    render: GraphSVGRender,
) -> anyhow::Result<()>
where
    <G as petgraph::visit::Data>::EdgeWeight: fmt::Debug,
    <G as petgraph::visit::Data>::NodeWeight: fmt::Debug,
{
    let mut root = std::env::current_dir().unwrap();

    if cfg!(test) {
        root = root.parent().unwrap().to_owned();
    }

    let dot_out_path = root.join("dot.gv");
    let out_path = root.join(out);

    fs::File::create(&out_path)
        .context(out_path.to_str().unwrap().to_string())
        .context("Invalid out path")?;

    match &render {
        GraphSVGRender::Directed { node_label } => fs::write(
            &dot_out_path,
            format!(
                "{:?}",
                dot::Dot::with_attr_getters(
                    &graph,
                    match node_label {
                        Label::Index => &[dot::Config::NodeIndexLabel, dot::Config::EdgeNoLabel],
                        Label::Weight => &[dot::Config::EdgeNoLabel],
                        Label::None => &[dot::Config::NodeNoLabel, dot::Config::EdgeNoLabel],
                    },
                    &|_, _| "arrowhead=none".to_owned(),
                    get_node_attrs
                )
            ),
        )
        .context("Failed to write directed DOT output")?,

        GraphSVGRender::Undirected { edge_label, .. } => fs::write(
            &dot_out_path,
            format!(
                "graph {{ \n layout=\"neato\"\n  {:?} }}",
                dot::Dot::with_attr_getters(
                    &graph,
                    match edge_label {
                        Label::Index => &[
                            dot::Config::GraphContentOnly,
                            dot::Config::NodeNoLabel,
                            dot::Config::EdgeIndexLabel
                        ],
                        Label::Weight => &[dot::Config::GraphContentOnly, dot::Config::NodeNoLabel],
                        Label::None => &[
                            dot::Config::GraphContentOnly,
                            dot::Config::NodeNoLabel,
                            dot::Config::EdgeNoLabel
                        ],
                    },
                    &|_, _| "arrowhead=none".to_owned(),
                    get_node_attrs
                )
            ),
        )
        .context("Failed to write undirected DOT output")?,
    };

    println!("Rendering SVG...");
    let dot_out = match render {
        GraphSVGRender::Directed { .. } => {
            let mut c = process::Command::new("dot");
            c.arg(dot_out_path);
            c
        }
        GraphSVGRender::Undirected { positions, .. } => {
            let mut c = process::Command::new("neato");
            if positions {
                c.arg("-n");
            }

            c.arg(&dot_out_path);
            c
        }
    }
    .arg("-Tsvg")
    .output()
    .context("Failed to execute graphviz process")?;

    //fs::remove_file(DOT_OUT)?;

    println!(
        "{}",
        std::str::from_utf8(&dot_out.stderr).context("STDERR from graphviz is not unicode")?
    );

    assert!(dot_out.status.success());

    println!("Writing svg output to {out_path:?}");

    fs::write(&out_path, dot_out.stdout).context("Failed to write output SVG")?;

    open::that(out_path).context("Failed to trigger auto-open of SVG")?;

    Ok(())
}

pub fn filter_nodes_by_weight<E: Copy, Ty: petgraph::EdgeType>(
    graph: &petgraph::Graph<u32, E, Ty>,
    weights: Range<u32>,
) -> Vec<petgraph::Graph<usize, E, Ty>> {
    weights
        .into_iter()
        .map(|i| {
            graph.filter_map(
                |n, &w| if w == i { Some(n.index()) } else { None },
                |_, &w| Some(w),
            )
        })
        .collect()
}

pub fn graph_contiguous<V, E, Ty: petgraph::EdgeType>(graph: &petgraph::Graph<V, E, Ty>) -> bool {
    if graph.node_count() == 0 {
        return true;
    }

    let mut search = vec![0];
    let mut seen = vec![false; graph.node_count()];

    while let Some(next) = search.pop() {
        seen[next] = true;

        for n in graph.neighbors(petgraph::graph::node_index(next)) {
            if !seen[n.index()] {
                search.push(n.index())
            }
        }
    }

    seen.iter().all(|&x| x)
}

pub fn assert_graph_contiguous<V: std::fmt::Debug, E: std::fmt::Debug, Ty: petgraph::EdgeType>(
    graph: &petgraph::Graph<V, E, Ty>,
) {
    let dir = if cfg!(test) {
        "..\\svg\\non_contig_graph.svg"
    } else {
        "svg\\non_contig_graph.svg"
    };

    if !graph_contiguous(graph) {
        println!("Graph is not contiguous, outputting error...");

        petgraph_to_svg(
            graph,
            dir,
            &|_, _| String::new(),
            GraphSVGRender::Directed {
                node_label: Label::None,
            },
        )
        .unwrap();

        assert!(
            false,
            "Graph is not contiguous. Outputted error graph to {}",
            dir
        );
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_contiguous_graph() {
        let mut g = petgraph::Graph::new();

        let a = g.add_node(());
        let b = g.add_node(());

        g.add_edge(a, b, ());

        assert!(graph_contiguous(&g));
    }

    #[test]
    fn test_contiguous_empty_graph() {
        let g: petgraph::prelude::Graph<(), ()> = petgraph::Graph::new();

        assert!(graph_contiguous(&g));
    }

    #[test]
    fn test_non_contiguous_graph() {
        let mut g = petgraph::Graph::new();

        let a = g.add_node(());
        let b = g.add_node(());
        g.add_node(());

        g.add_edge(a, b, ());

        assert!(!graph_contiguous(&g));
    }
}
