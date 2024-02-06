#![cfg(test)]
use baker::mesh::winged_mesh::WingedMesh;

fn group_and_partition_and_simplify(mesh: &mut WingedMesh, verts: &[glam::Vec4]) {
    let config = &metis::PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelKWay,
        force_contiguous_partitions: Some(true),
        minimize_subgraph_degree: Some(true),
        ..Default::default()
    };

    let mut quadrics = mesh.create_quadrics(verts);

    // Apply primary partition, that will define the lowest level clusterings
    mesh.partition_full_mesh(config, mesh.face_count().div_ceil(200) as _)
        .unwrap();

    mesh.group(config, &verts).unwrap();

    // Generate 2 more meshes
    for i in 0..10 {
        // i = index of previous mesh layer
        //working_mesh = reduce_mesh(working_mesh);

        println!("Face count L{}: {}", i, mesh.face_count());

        // We must regenerate the queue each time, as boundaries change.

        // Each group requires half it's triangles removed
        let collapse_requirements: Vec<usize> = mesh.groups.iter().map(|g| g.tris / 4).collect();

        mesh.age();

        // Make sure groups are contiguous before reduction

        println!("Reducing within {} groups:", collapse_requirements.len());

        let e = match mesh.reduce_within_groups(verts, &mut quadrics, &collapse_requirements) {
            Ok(e) => e,
            Err(e) => {
                println!(
                    "Experience error {} with reducing, exiting early with what we have",
                    e
                );
                break;
            }
        };
        println!(
            "Introduced error of {e}. Max edge age: {}, Mean: {}",
            mesh.max_edge_age(),
            mesh.avg_edge_age(),
        );

        //layers.push(to_mesh_layer(&working_mesh, &verts));

        let partition_count = match mesh.partition_within_groups(&config, Some(2), None) {
            Ok(partition_count) => partition_count,
            Err(e) => {
                println!("{}", e);
                break;
            }
        };
        // View a snapshot of the mesh without any re-groupings applied
        //layers.push(to_mesh_layer(&next_mesh));

        println!("{partition_count} Partitions from groups");

        let group_count = mesh.group(&config, &verts).unwrap();

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("{group_count} Groups from partitions");

        if group_count == 1 {
            println!("Finished with single group");
            break;
        }
    }

    println!("Done with partitioning");
}

#[test]
fn test_determinacy() {
    // let mesh_name = "../assets/rock.glb";
    let mesh_name = "../../assets/rock.glb";

    println!("Loading from gltf!");
    let (mut mesh1, tri_mesh) = WingedMesh::from_gltf(mesh_name);

    let mut mesh0 = mesh1.clone();

    let num_contiguous = mesh1.partition_contiguous();

    println!("We have {:?} Contiguous segments", num_contiguous);

    assert_eq!(num_contiguous.len(), 1);
    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
    group_and_partition_and_simplify(&mut mesh0, &tri_mesh.verts);
    group_and_partition_and_simplify(&mut mesh1, &tri_mesh.verts);

    assert_eq!(mesh0, mesh1)
}
