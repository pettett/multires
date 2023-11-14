use baker::{mesh::winged_mesh::WingedMesh, to_mesh_layer};
use common::{asset::Asset, MultiResMesh};
use metis::PartitioningConfig;

fn main() {
    let mesh_name = "../assets/sphere_low.glb";

    let config = PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelKWay,
        force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };

    let within_group_config = PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelKWay,
        force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };

    println!("Loading from gltf!");
    let (mut working_mesh, verts) = WingedMesh::from_gltf(mesh_name);

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

        match working_mesh.partition_within_groups(&within_group_config, Some(2)) {
            Ok(partition_count) => {
                // View a snapshot of the mesh without any re-groupings applied
                //layers.push(to_mesh_layer(&next_mesh));

                println!("{partition_count} Partitions from groups");

                match working_mesh.group(&within_group_config, &verts) {
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
        name: mesh_name.to_owned(),
        verts: verts.iter().map(|v| [v.x, v.y, v.z, 1.0]).collect(),
        // layer_1_indices: indices.clone(),
        lods: layers,
    }
    .save()
    .unwrap();
}
