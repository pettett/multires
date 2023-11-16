pub mod mesh;

use common::{asset::Asset, MeshLevel, Meshlet, MultiResMesh, SubMesh};

use glam::Vec4;
use mesh::winged_mesh::WingedMesh;
// use meshopt::VertexDataAdapter;

pub fn to_mesh_layer(mesh: &WingedMesh, verts: &[Vec4]) -> MeshLevel {
    MeshLevel {
        partition_indices: mesh.get_partition(),
        group_indices: mesh.get_group(),
        indices: grab_indicies(&mesh),
        meshlets: vec![], //generate_meshlets(&mesh),
        submeshes: generate_submeshes(mesh, verts),
        partitions: mesh.partitions.clone(),
        groups: mesh.groups.clone(),
    }
}

pub fn group_and_partition_full_res(mut working_mesh: WingedMesh, verts: &[Vec4], name: String) {
    let config = metis::PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelKWay,
        force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };

    // Apply primary partition, that will define the lowest level clusterings
    working_mesh.partition_within_groups(&config, None).unwrap();

    working_mesh.group(&config, &verts).unwrap();

    let mut layers = Vec::new();

    layers.push(to_mesh_layer(&working_mesh, &verts));

    // Generate 2 more meshes
    for i in 0..10 {
        // i = index of previous mesh layer
        //working_mesh = reduce_mesh(working_mesh);

        println!("Face count L{}: {}", i + 1, working_mesh.face_count());

        match working_mesh.partition_within_groups(&config, Some(2)) {
            Ok(partition_count) => {
                // View a snapshot of the mesh without any re-groupings applied
                //layers.push(to_mesh_layer(&next_mesh));

                println!("{partition_count} Partitions from groups");

                match working_mesh.group(&config, &verts) {
                    Ok(group_count) => {
                        // view a snapshot of the mesh ready to create the next layer
                        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

                        println!("{group_count} Groups from partitions");

                        layers.push(to_mesh_layer(&working_mesh, &verts));

                        if group_count == 1 {
                            println!("Finished with single group");
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("{}", e);
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("{}", e);
                break;
            }
        }
    }

    println!("Done with partitioning");

    //assert_eq!(partitions1.len() * 3, layer_1_indices.len());

    MultiResMesh {
        name,
        verts: verts.iter().map(|v| [v.x, v.y, v.z, 1.0]).collect(),
        // layer_1_indices: indices.clone(),
        lods: layers,
    }
    .save()
    .unwrap();
}

pub fn apply_simplification(mut working_mesh: WingedMesh, verts: &[Vec4], name: String) {
    let config = metis::PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelKWay,
        force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };

    // Apply primary partition, that will define the lowest level clusterings
    //working_mesh.partition_within_groups(&config, None).unwrap();

    working_mesh.group(&config, &verts).unwrap();

    let mut layers = Vec::new();

    layers.push(to_mesh_layer(&working_mesh, &verts));

    let mut quads = working_mesh.create_quadrics(verts);

    // Generate 2 more meshes
    for i in 0..8 {
        // i = index of previous mesh layer
        println!("Face count L{}: {}", i + 1, working_mesh.face_count());

        let Ok(e) = working_mesh.reduce(verts, &mut quads) else {
            println!("Experience error with reducing, exiting early with what we have");
            break;
        };
        // View a snapshot of the mesh without any re-groupings applied

        layers.push(to_mesh_layer(&working_mesh, &verts));
    }

    println!("Done with partitioning");

    //assert_eq!(partitions1.len() * 3, layer_1_indices.len());

    MultiResMesh {
        name,
        verts: verts.iter().map(|v| [v.x, v.y, v.z, 1.0]).collect(),
        // layer_1_indices: indices.clone(),
        lods: layers,
    }
    .save()
    .unwrap();
}

pub fn grab_indicies(mesh: &WingedMesh) -> Vec<u32> {
    let mut indices = Vec::with_capacity(mesh.face_count() * 3);

    for f in mesh.faces().values() {
        let [a, b, c] = mesh.triangle_from_face(f);
        indices.push(a as _);
        indices.push(b as _);
        indices.push(c as _);
    }

    // Validation
    //for i in &indices {
    //    verts[*i as usize];
    //}

    indices
}

pub fn generate_meshlets(mesh: &WingedMesh) -> Vec<Meshlet> {
    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets: Vec<_> = (0..=mesh.partition_count())
        .map(|_| (Meshlet::default()))
        .collect();

    for face in mesh.faces().values() {
        let verts = mesh.triangle_from_face(face);

        let m = meshlets.get_mut(face.part as usize).unwrap();

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

    let total_indices: u32 = meshlets.iter().map(|m| m.index_count).sum();
    let avg_indices = total_indices / meshlets.len() as u32;
    let max_indices = meshlets.iter().map(|m| m.index_count).max().unwrap();
    let total_verts: u32 = meshlets.iter().map(|m| m.vertex_count).sum();
    let avg_verts = total_verts / meshlets.len() as u32;
    let max_verts = meshlets.iter().map(|m| m.vertex_count).max().unwrap();

    println!("avg_indices: {avg_indices}/378 max_indices: {max_indices}/378 avg_verts: {avg_verts}/64 max_verts: {max_verts}/64");

    meshlets
}

/// Debug code to generate meshlets with no max size. Used for testing partition trees with no remeshing
pub fn generate_submeshes(mesh: &WingedMesh, verts: &[Vec4]) -> Vec<SubMesh> {
    println!("Generating meshlets!");

    let inds = 5.0 / mesh.face_count() as f32;

    // Precondition: partition indexes completely span in some range 0..N
    let mut submeshes: Vec<_> = (0..mesh.partition_count())
        .map(|part| {
            let gi = mesh.partitions[part].group_index;
            let g = &mesh.groups[gi];
            SubMesh::new(
                0.0,
                g.monotonic_bound.center(),
                g.monotonic_bound.radius(),
                mesh.partitions[part].tight_bound.radius(),
                gi,
            )
        })
        .collect();

    for face in mesh.faces().values() {
        let verts = mesh.triangle_from_face(face);

        let m = submeshes.get_mut(face.part as usize).unwrap();

        for v in 0..3 {
            let vert = verts[v] as u32;
            m.indices.push(vert);
        }

        m.error += inds;
    }

    // let data = VertexDataAdapter::new(bytemuck::cast_slice(&verts), std::mem::size_of::<Vec4>(), 0)
    //     .unwrap();

    // for s in &mut submeshes {
    //     s.indices = meshopt::simplify(&s.indices, &data, s.indices.len() / 2, 0.01);
    // }

    submeshes
}

// pub fn reduce_mesh(mut mesh: WingedMesh) -> WingedMesh {
//     let mut rng = rand::thread_rng();

//     for (i, m) in mesh.groups.iter().enumerate() {
//         //println!("Reducing meshlet {i}/{}", meshlets.len());

//         // reduce triangle count in meshlet by half

//         // let mut tris = m.tris;
//         // let target = tris * 3 / 4;

//         // let mut todo: Vec<_> = (0..m.vertex_count as usize).collect();

//         // while tris > target && todo.len() > 0 {
//         //     // Pick a random edge in the mesh for now
//         //     let i = rng.gen_range(0..todo.len());
//         //     let v = VertID(m.vertices[todo[i]] as usize);
//         //     todo.swap_remove(i);

//         //     // println!("{i} {tris}/ {target}, {v:?}");

//         //     let Some(i) = mesh[v].edge else {
//         //         continue;
//         //     };

//         //     let valid_edge = v.is_local_manifold(&mesh, true);

//         //     if valid_edge {
//         //         // all faces are within the partition, we can safely collapse one of the edges

//         //         mesh.collapse_edge(i);

//         //         tris -= 2;

//         //         // println!("Collapsed edge {e:?}");

//         //         //break;
//         //     }
//         // }
//     }

//     mesh
// }
