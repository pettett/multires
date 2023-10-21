extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, Meshlet, MultiResMesh};
use metis::PartitioningConfig;
use std::{
    collections::{HashMap, HashSet},
    time,
};
use winged_mesh::{EdgeID, VertID};

use crate::winged_mesh::{FaceID, WingedMesh};

fn main() -> gltf::Result<()> {
    let mesh_name = "../assets/plane.glb";

    println!("Loading from gltf!");
    let (mesh, verts, indices) = winged_mesh::WingedMesh::from_gltf(mesh_name)?;

    println!("Loaded winged edge mesh from gltf!");

    mesh.assert_valid();

    println!("Partitioning Graph!");
    let t1 = time::Instant::now();
    let config = PartitioningConfig {
        //force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };
    let clusters = mesh
        .partition(&config, 1 + mesh.faces().len() as u32 / 70)
        .unwrap();

    println!("time: {}ms", t1.elapsed().as_millis());

    println!("Generating the partition connections Graph!");
    let mut graph = petgraph::Graph::<i32, i32>::new();

    let parts: HashSet<_> = clusters.iter().collect();

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets: Vec<_> = parts.iter().map(|_| (Meshlet::default())).collect();

    for i in 0..clusters.len() {
        let Some(verts) = mesh.triangle_from_face(FaceID(i)) else {
            continue;
        };

        let m = meshlets.get_mut(clusters[i] as usize).unwrap();

        for v in 0..3 {
            let vert = verts[v] as u32;

            // If unique, add to list
            let idx = (0..m.vertex_count as usize).find(|j| m.vertices[*j] == vert);

            let idx = if let Some(idx) = idx {
                idx as u32
            } else {
                m.vertex_count += 1;
                m.vertices[m.vertex_count as usize - 1] = vert;
                m.vertex_count - 1
            };

            m.indices[m.index_count as usize + v] = idx;
        }
        m.index_count += 3;
    }

    // halve number of triangles in each meshlet

    println!("Face count L0: {}", mesh.face_count());

    let layer_1_mesh = reduce_mesh(&meshlets, &clusters, mesh.clone());
    let mut layer_1_indices = Vec::new();
    for i in 0..clusters.len() {
        let Some(verts) = layer_1_mesh.triangle_from_face(FaceID(i)) else {
            continue;
        };
        layer_1_indices.extend(verts);
    }

    for i in &layer_1_indices {
        verts[*i as usize];
    }

    println!("Face count L0: {}", layer_1_mesh.face_count());

    let total_indices: u32 = meshlets.iter().map(|m| m.index_count).sum();
    let avg_indices = total_indices / meshlets.len() as u32;
    let max_indices = meshlets.iter().map(|m| m.index_count).max().unwrap();
    let total_verts: u32 = meshlets.iter().map(|m| m.vertex_count).sum();
    let avg_verts = total_verts / meshlets.len() as u32;
    let max_verts = meshlets.iter().map(|m| m.vertex_count).max().unwrap();

    println!("avg_indices: {avg_indices}/378 max_indices: {max_indices}/378 avg_verts: {avg_verts}/64 max_verts: {max_verts}/64");

    let nodes: HashMap<_, _> = parts
        .iter()
        .map(|i| {
            let n = graph.add_node(1);
            (n.index() as i32, n)
        })
        .collect();

    for (i, face) in mesh.faces().iter().enumerate() {
        for e in mesh.iter_edge_loop(face.as_ref().unwrap().edge.unwrap()) {
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
        layer_1_indices,
        meshlets,
    }
    .save()
    .unwrap();

    Ok(())
}

fn reduce_mesh(meshlets: &[Meshlet], partitions: &[i32], mut mesh: WingedMesh) -> WingedMesh {
    //TODO: Bad code
    for i in 0..mesh.edge_count() / 20 {
        //for i in 0..p.vertex_count as usize {
        //let e = mesh[].clone();

        //    if let Some(e) = t.edge {
        //let f = partitions[Into::<usize>::into(mesh[e].face)];

        //let mut valid_face = true;

        //for e in mesh.iter_edge_loop(e) {
        //    if let Some(twin) = mesh[e].twin {
        //        if partitions[Into::<usize>::into(mesh[twin].face)] != f {
        //            valid_face = false;
        //            break;
        //        }
        //    }
        //}

        //if valid_face {
        // all faces are within the partition, we can safely collapse one of the edges

        mesh.collapse_edge(EdgeID(i * 20));

        // println!("Collapsed edge {e:?}");

        //break;
        //}
        //    }
        //}
    }

    mesh
}
