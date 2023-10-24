extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, MeshLayer, Meshlet, MultiResMesh};
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
    mesh.apply_partition(&config, 1 + mesh.faces().len() as u32 / 60)
        .unwrap();

    let partitions0 = mesh.get_partition();

    println!("Partitioning the partition!");
    let groups = mesh.group(&config).unwrap();

    println!("time: {}ms", t1.elapsed().as_millis());

    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets: Vec<_> = (0..=*partitions0.iter().max().unwrap())
        .map(|_| (Meshlet::default()))
        .collect();

    for i in 0..partitions0.len() {
        let Some(verts) = mesh.triangle_from_face(FaceID(i)) else {
            continue;
        };

        let m = meshlets.get_mut(partitions0[i] as usize).unwrap();

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

    println!("Face count L1: {}", layer_1_mesh.face_count());

    let total_indices: u32 = meshlets.iter().map(|m| m.index_count).sum();
    let avg_indices = total_indices / meshlets.len() as u32;
    let max_indices = meshlets.iter().map(|m| m.index_count).max().unwrap();
    let total_verts: u32 = meshlets.iter().map(|m| m.vertex_count).sum();
    let avg_verts = total_verts / meshlets.len() as u32;
    let max_verts = meshlets.iter().map(|m| m.vertex_count).max().unwrap();

    println!("avg_indices: {avg_indices}/378 max_indices: {max_indices}/378 avg_verts: {avg_verts}/64 max_verts: {max_verts}/64");

    let partitions1 = layer_1_mesh.get_partition();

    assert_eq!(partitions1.len() * 3, layer_1_indices.len());

    MultiResMesh {
        name: mesh_name.to_owned(),
        verts: verts.iter().map(|x| [x[0], x[1], x[2], 1.0]).collect(),
        // layer_1_indices: indices.clone(),
        layers: vec![
            MeshLayer {
                partitions: partitions0,
                groups,
                indices,
                meshlets,
            },
            MeshLayer {
                partitions: partitions1,
                groups: vec![],
                indices: layer_1_indices,
                meshlets: vec![],
            },
        ],
    }
    .save()
    .unwrap();

    Ok(())
}

fn reduce_mesh(meshlets: &[Meshlet], mut mesh: WingedMesh) -> WingedMesh {
    let mut rng = rand::thread_rng();

    for (i, m) in meshlets.iter().enumerate() {
        println!("Reducing meshlet {i}/{}", meshlets.len());

        // reduce triangle count in meshlet by half

        let mut tris = m.index_count / 3;
        let target = tris * 3 / 4;

        let mut todo: Vec<_> = (0..m.vertex_count as usize).collect();

        while tris > target && todo.len() > 0 {
            // Pick a random edge in the mesh for now
            let i = rng.gen_range(0..todo.len());
            let v = VertID(m.vertices[todo[i]] as usize);
            todo.swap_remove(i);

            //println!("{tris}/ {target}, {v:?}");

            let Some(i) = mesh[v].edge else {
                continue;
            };

            let mut valid_edge = mesh.vertex_has_complete_fan(v);

            if valid_edge {
                let f = mesh.faces()[mesh[i].face].group;
                for e in mesh.outgoing_edges(v) {
                    if f != mesh.faces()[mesh[*e].face].group {
                        valid_edge = false;
                        break;
                    }
                }
            }
            if valid_edge {
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
