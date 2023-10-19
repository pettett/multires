extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, Meshlet, MultiResMesh};
use metis::PartitioningConfig;
use petgraph::data::Build;
use std::{
    collections::{HashMap, HashSet},
    time,
};

fn main() -> gltf::Result<()> {
    let mesh_name = "../assets/torus.glb";

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
    let clusters = mesh
        .partition(&config, 1 + mesh.faces().len() as u32 / 41)
        .unwrap();

    println!("time: {}ms", t1.elapsed().as_millis());

    println!("Generating the partition connections Graph!");
    let mut graph = petgraph::Graph::<i32, i32>::new();

    let parts: HashSet<_> = clusters.iter().collect();

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets: Vec<_> = parts.iter().map(|_| (Meshlet::default())).collect();

    for i in 0..clusters.len() {
        let verts: Vec<usize> = mesh
            .iter_edge(mesh.faces()[i].edge.unwrap())
            .map(|e| mesh[e].vert_origin.into())
            .collect();
        let m = meshlets.get_mut(clusters[i] as usize).unwrap();

        for v in 0..3 {
            let vert = verts[v] as u32;
            m.indices[m.index_count as usize + v] = vert;

            // If unique, add to list
            let exists = (0..m.vertex_count as usize)
                .find(|j| m.vertices[*j] == vert)
                .is_some();

            if !exists {
                m.vertices[m.vertex_count as usize] = vert;
                m.vertex_count += 1;
            }
        }
        m.index_count += 3;
    }

    let total_indices: u32 = meshlets.iter().map(|m| m.index_count).sum();
    let avg_indices = total_indices / meshlets.len() as u32;
    let max_indices = meshlets.iter().map(|m| m.index_count).max().unwrap();
    let total_verts: u32 = meshlets.iter().map(|m| m.vertex_count).sum();
    let avg_verts = total_verts / meshlets.len() as u32;
    let max_verts = meshlets.iter().map(|m| m.vertex_count).max().unwrap();

    println!("avg_indices: {avg_indices}/126 max_indices: {max_indices}/126 avg_verts: {avg_verts}/64 max_verts: {max_verts}/64");

    let nodes: HashMap<_, _> = parts
        .iter()
        .map(|i| {
            let n = graph.add_node(1);
            (n.index() as i32, n)
        })
        .collect();

    for (i, face) in mesh.faces().iter().enumerate() {
        for e in mesh.iter_edge(face.edge.unwrap()) {
            if let Some(twin) = mesh[e].twin {
                let idx: usize = mesh[twin].face.into();

                graph.update_edge(nodes[&clusters[i]], nodes[&clusters[idx]], 1);
            }
        }
    }

    println!("Partitioning the partition!");
    let clusters2 = config.partition_from_graph(5, &graph).unwrap();

    MultiResMesh {
        name: mesh_name.to_owned(),
        clusters,
        clusters2,
        verts: verts.iter().map(|x| [x[0], x[1], x[2], 1.0]).collect(),
        indices,
        meshlets,
    }
    .save()
    .unwrap();

    Ok(())
}
