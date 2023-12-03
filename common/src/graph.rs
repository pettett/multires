use core::fmt;
use petgraph::dot;
use std::{error, fs, path, process};
const OUT: &str = "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\multires";

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
pub fn generate_triangle_plane<const N: usize, const M: usize, E>(
    weight: impl Fn((usize, usize), (usize, usize)) -> E,
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
                graph.update_edge(a, nodes[m][n + 1], weight((n, m), (n + 1, m)));

                if m < M - 1 && n % 2 == 0 {
                    graph.update_edge(a, nodes[m + 1][n + 1], weight((n, m), (n + 1, m + 1)));
                }
            }

            if n > 0 {
                graph.update_edge(a, nodes[m][n - 1], weight((n, m), (n - 1, m)));

                if m > 0 && n % 2 == 1 {
                    graph.update_edge(a, nodes[m - 1][n - 1], weight((n, m), (n - 1, m - 1)));
                }
            }
        }
    }

    graph
}

pub enum Label {
    Index,
    Weight,
    None,
}
pub enum GraphSVGRender {
    Directed { node_label: Label },
    Undirected { positions: bool },
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
) -> Result<(), Box<dyn error::Error>>
where
    <G as petgraph::visit::Data>::EdgeWeight: fmt::Debug,
    <G as petgraph::visit::Data>::NodeWeight: fmt::Debug,
{
    let dot_out_path = path::Path::new(OUT).join("svg\\dot.gv");
    let out_path = path::Path::new(OUT).join(out);

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
        )?,
        GraphSVGRender::Undirected { .. } => fs::write(
            &dot_out_path,
            format!(
                "graph {{ \n layout=\"neato\"\n  {:?} }}",
                dot::Dot::with_attr_getters(
                    &graph,
                    &[dot::Config::GraphContentOnly, dot::Config::NodeNoLabel,],
                    &|_, _| "arrowhead=none".to_owned(),
                    get_node_attrs
                )
            ),
        )?,
    };

    let dot_out = match render {
        GraphSVGRender::Directed { .. } => {
            let mut c = process::Command::new("dot");
            c.arg(dot_out_path);
            c
        }
        GraphSVGRender::Undirected { positions } => {
            let mut c = process::Command::new("neato");
            if positions {
                c.arg("-n");
            }

            c.arg(&dot_out_path);
            c
        }
    }
    .arg("-Tsvg")
    .output()?;

    //fs::remove_file(DOT_OUT)?;

    println!("{}", std::str::from_utf8(&dot_out.stderr)?);

    assert!(dot_out.status.success());

    fs::write(&out_path, dot_out.stdout)?;

    open::that(out_path).unwrap();

    Ok(())
}
