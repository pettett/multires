extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, MultiResMesh};
use metis::PartitioningConfig;
use petgraph::data::Build;
use std::{
    collections::{HashMap, HashSet},
    time,
};

fn main() -> gltf::Result<()> {
    let mesh_name = "../assets/cube.glb";

    println!("Loading from gltf!");
    let (mesh, verts, indices) = winged_mesh::WingedMesh::from_gltf(mesh_name)?;

    println!("Loaded winged edge mesh from gltf!");

    println!("Partitioning Graph!");
    let t1 = time::Instant::now();
    let config = PartitioningConfig {
        //force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };
    // let clusters = mesh
    //     .partition(&config, 1 + mesh.faces().len() as u32 / 128)
    //     .unwrap();

    // println!("time: {}ms", t1.elapsed().as_millis());

    // println!("Generating the partition connections Graph!");
    // let mut graph = petgraph::Graph::<i32, i32>::new();

    // let parts: HashSet<_> = clusters.iter().collect();
    // let nodes: HashMap<_, _> = parts
    //     .iter()
    //     .map(|i| {
    //         let n = graph.add_node(1);
    //         (n.index() as i32, n)
    //     })
    //     .collect();

    // for (i, face) in mesh.faces().iter().enumerate() {
    //     for e in mesh.iter_edge(face.edge.unwrap()) {
    //         if let Some(twin) = mesh[e].twin {
    //             let idx: usize = mesh[twin].face.into();

    //             graph.update_edge(nodes[&clusters[i]], nodes[&clusters[idx]], 1);
    //         }
    //     }
    // }

    // println!("Partitioning the partition!");
    // let clusters2 = config.partition_from_graph(5, &graph).unwrap();

    let clusters = vec![];
    let clusters2 = vec![];

    MultiResMesh {
        name: mesh_name.to_owned(),
        clusters,
        clusters2,
        verts: verts.iter().map(|x| [x[0], x[1], x[2], 1.0]).collect(),
        indices,
    }
    .save()
    .unwrap();

    Ok(())
}
