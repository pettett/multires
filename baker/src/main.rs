extern crate gltf;

pub mod winged_mesh;
use metis::{Graph, GraphEdge, GraphVertex, PartitioningConfig};
use std::time;

fn main() -> gltf::Result<()> {
    let mesh = winged_mesh::WingedMesh::from_gltf("../assets/torus_low.glb")?;

    for f in mesh.faces() {
        let e = f.edge.unwrap();

        for e in mesh.iter_edge(e) {
            if let Some(te) = mesh[e].twin {
                println!(
                    "Face {:?} is connected to {:?}",
                    mesh[e].face, mesh[te].face
                )
            }
        }
    }
    let mut graph = Graph {
        vertices: vec![
            GraphVertex {
                edges: vec![],
                original_index: u32::MAX,
                color: u32::MAX
            };
            mesh.faces().len()
        ],
    };

    for f in mesh.faces() {
        let e = f.edge.unwrap();

        for e in mesh.iter_edge(e) {
            if let Some(te) = mesh[e].twin {
                let t1: usize = mesh[e].face.into();
                let t2: usize = mesh[te].face.into();

                graph.vertices[t1].edges.push(GraphEdge {
                    dst: t2 as u32,
                    weight: 1,
                });
            }
        }
    }

    let t1 = time::Instant::now();
    graph
        .partition(
            &PartitioningConfig {
                force_contiguous_partitions: Some(true),
                ..Default::default()
            },
            1 + mesh.faces().len() as u32 / 128,
        )
        .unwrap();
    println!("time: {}ms", t1.elapsed().as_millis());
    println!("edge cut {}", graph.calculate_edge_cut());
    let infos = graph.partition_info();
    println!(
        "partitions: {}, with sizes from {} to {} with an average of {})",
        graph.count_partitions(),
        infos.0,
        infos.1,
        infos.2
    );
    let sizes = graph.partition_sizes();
    for c in 0..=graph.max_partition_color() {
        let s = sizes[c as usize];
        println!(
            "partition {c} with size {s} has {} parts",
            graph.count_partition_parts(c, s)
        );
    }
    Ok(())
}
