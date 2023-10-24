extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, Meshlet, MultiResMesh};
use metis::PartitioningConfig;
use rand::Rng;
use std::{
    collections::{HashMap, HashSet},
    time,
};
use winged_mesh::VertID;

use crate::winged_mesh::{FaceID, WingedMesh};

fn main() -> gltf::Result<()> {
    let mesh_name = "../assets/torus.glb";

    println!("Loading from gltf!");
    let (mut mesh, verts, indices) = winged_mesh::WingedMesh::from_gltf(mesh_name)?;

    println!("Loaded winged edge mesh from gltf!");

    println!("Partitioning Graph!");
    let t1 = time::Instant::now();
    let config = PartitioningConfig {
        //force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };
    mesh.apply_partition(&config, 1 + mesh.faces().len() as u32 / 70)
        .unwrap();

    let clusters = mesh.get_partition();

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

    let layer_1_mesh = reduce_mesh(&meshlets, mesh.clone());
    let mut layer_1_indices = Vec::new();
    for f in layer_1_mesh.faces().keys() {
        let Some(verts) = layer_1_mesh.triangle_from_face(*f) else {
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

    for (i, face) in mesh.faces().iter() {
        for e in mesh.iter_edge_loop(face.edge) {
            if let Some(twin) = mesh[e].twin {
                let idx: usize = mesh[twin].face.into();

                graph.update_edge(nodes[&clusters[i.0]], nodes[&clusters[idx]], 1);
            }
        }
    }

    println!("Partitioning the partition!");
    let clusters2 = config.partition_from_graph(5, &graph).unwrap();

    let clusters = layer_1_mesh.get_partition();

    assert_eq!(clusters.len() * 3, layer_1_indices.len());

    MultiResMesh {
        name: mesh_name.to_owned(),
        clusters,
        clusters2,
        verts: verts.iter().map(|x| [x[0], x[1], x[2], 1.0]).collect(),
        // layer_1_indices: indices.clone(),
        layer_1_indices,
        indices,
        meshlets,
    }
    .save()
    .unwrap();

    Ok(())
}

fn reduce_mesh(meshlets: &[Meshlet], mut mesh: WingedMesh) -> WingedMesh {
    let mut rng = rand::thread_rng();

    for m in meshlets {
        // reduce triangle count in meshlet by half

        let mut tris = m.index_count / 3;
        let target = tris * 2 / 3;

        let mut vi = 0;

        while tris > target && vi < m.vertex_count as usize {
            // Pick a random edge in the mesh for now

            let v = VertID(m.vertices[vi] as usize);
            vi += rng.gen_range(0..m.vertex_count as usize);
            vi %= m.vertex_count as usize;

            //println!("{tris}/ {target}, {v:?}");

            let Some(i) = mesh[v].edge else {
                continue;
            };

            let mut valid_face = mesh.vertex_has_complete_fan(v);

            if valid_face {
                let f = mesh.faces()[mesh[i].face].part;
                for e in mesh.outgoing_edges(v) {
                    if f != mesh.faces()[mesh[*e].face].part {
                        valid_face = false;
                        break;
                    }
                }
            }
            if valid_face {
                // all faces are within the partition, we can safely collapse one of the edges

                mesh.collapse_edge(i);

                tris -= 2;

                // println!("Collapsed edge {e:?}");

                //break;
            }
        }
    }

    mesh
}
