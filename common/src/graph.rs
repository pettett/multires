use anyhow::Context;
use core::fmt;
use petgraph::dot;
use std::{fs, path, process};

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
pub fn generate_triangle_plane<const N: usize, const M: usize>(
) -> petgraph::Graph<(), (), petgraph::Undirected> {
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

    graph
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
    let root = std::env::current_dir()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();

    let dot_out_path = root.join("svg\\dot.gv");
    let out_path = root.join(out);

    println!("Writing svg output to {out_path:?}");

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

    fs::write(&out_path, dot_out.stdout).context("Failed to write output SVG")?;

    open::with(out_path, "firefox").context("Failed to trigger auto-open of SVG")?;

    Ok(())
}
