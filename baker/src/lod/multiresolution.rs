use common::{MeshVert, MultiResMesh, TriMesh};

use crate::{
    lod::{generate_clusters, stat_readout},
    mesh::{half_edge_mesh::HalfEdgeMesh, PartitionCount},
    CLUSTERS_PER_SIMPLIFIED_GROUP, STARTING_CLUSTER_SIZE,
};

pub fn group_and_partition_and_simplify(
    mut mesh: HalfEdgeMesh,
    tri_mesh: TriMesh,
    name: String,
    starting_cluster_size: usize,
) -> anyhow::Result<MultiResMesh> {
    let triangle_clustering_config = &metis::MultilevelKWayPartitioningConfig {
        //u_factor: Some(10),
        //minimize_subgraph_degree: Some(true), // this will sometimes break contiguous partitions
        ..Default::default()
    }
    .into();

    let group_clustering_config = &metis::MultilevelKWayPartitioningConfig {
        //u_factor: Some(1),
        //objective_type: Some(metis::ObjectiveType::Volume),
        minimize_subgraph_degree: Some(true), // this will sometimes break contiguous partitions
        ..Default::default()
    }
    .into();

    let _non_contig_even_clustering_config = &metis::PartitioningConfig {
        method: metis::PartitioningMethod::MultilevelRecursiveBisection,
        //force_contiguous_partitions: true,
        //u_factor: Some(1),
        //objective_type: Some(metis::ObjectiveType::Volume),
        minimize_subgraph_degree: Some(true), // this will sometimes break contiguous partitions
        ..Default::default()
    };

    let grouping_config = &metis::MultilevelKWayPartitioningConfig {
        //objective_type: Some(metis::ObjectiveType::Volume),
        force_contiguous_partitions: Some(true),
        u_factor: Some(100), // Allow large 'inequality' to get similar sized groups
        //partitioning_attempts: Some(3),
        //separator_attempts: Some(3),
        //two_hop_matching: Some(true),
        //initial_partitioning: Some(metis::InitialPartitioningAlgorithm::RandomRefined),
        //refinement: Some(metis::RefinementAlgorithm::TwoSidedFm),
        //refinement_iterations: Some(30),
        //coarsening: Some(metis::CoarseningScheme::SortedHeavyEdgeMatching),
        //minimize_subgraph_degree: Some(true), // this will sometimes break contiguous partitions
        ..Default::default()
    }
    .into();

    let mut quadrics = mesh.create_quadrics(&tri_mesh.verts);

    // Apply primary partition, that will define the lowest level clusterings
    mesh.cluster_full_mesh(
        triangle_clustering_config,
        mesh.face_count().div_ceil(starting_cluster_size) as _,
        &tri_mesh.verts,
    )?;

    println!("Beginning first grouping");

    mesh.group(grouping_config).unwrap();

    println!("Finished first grouping");

    let mut multi_res = MultiResMesh {
        name,
        full_indices: tri_mesh.indices,
        verts: tri_mesh
            .verts
            .iter()
            .zip(tri_mesh.normals.iter())
            .map(|(v, n)| MeshVert {
                pos: [v.x, v.y, v.z, 1.0],
                normal: [n.x, n.y, n.z, 1.0],
            })
            .collect(),
        // layer_1_indices: indices.clone(),
        //lods: Vec::new(),
        clusters: Vec::new(),
        group_count: 0,
    };

    let mut lower_group_range = 0;
    multi_res.group_count += &mesh.groups.len();
    let mut upper_group_range = multi_res.group_count;

    //multi_res.lods.push(to_mesh_layer(&mesh));

    multi_res.clusters.extend_from_slice(&generate_clusters(
        &mesh,
        0,
        &multi_res.verts,
        0,
        0,
        Some(16),
    ));

    // Generate layers, with a fallback exit
    for i in 1..20 {
        // i = index of previous mesh layer
        //working_mesh = reduce_mesh(working_mesh);

        println!("Face count L{}: {}", i, mesh.face_count());

        // We must regenerate the queue each time, as boundaries change.

        // Each group requires half it's triangles removed

        let collapse_requirements: Vec<usize> = mesh
            .groups
            .iter()
            .map(|g| {
                g.num_tris(&mesh) / 4
                //let halved_tris = g.tris / 4;
                //let tris_to_remove_for_cluster_max = g
                //    .tris
                //    .saturating_sub(MAX_TRIS_PER_CLUSTER * CLUSTERS_PER_SIMPLIFIED_GROUP - 25);
                //
                // Each operation removes 2 triangles.
                // Do whichever we need to bring ourselves down to the limit. Error function will make up for variations in density
                //halved_tris.max(halved_tris).div_ceil(2)
            })
            .collect();

        mesh.age();

        // Make sure groups are contiguous before reduction
        #[cfg(debug)]
        {
            println!("Ensuring groups are contiguous... ");
            let graphs = mesh.generate_group_keyed_graphs();
            for graph in graphs {
                graph::assert_graph_contiguous(&graph);
            }
        }

        println!("Reducing within {} groups:", collapse_requirements.len());

        let e =
            match mesh.reduce_within_groups(&tri_mesh.verts, &mut quadrics, &collapse_requirements)
            {
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

        let partition_count = match mesh.cluster_within_groups(
            group_clustering_config,
            &tri_mesh.verts,
            PartitionCount::Partitions(CLUSTERS_PER_SIMPLIFIED_GROUP as _),
        ) {
            Ok(partition_count) => partition_count,
            Err(e) => {
                println!("{}", e);
                break;
            }
        };
        // View a snapshot of the mesh without any re-groupings applied
        //layers.push(to_mesh_layer(&next_mesh));

        println!("{partition_count} Partitions from groups");

        let group_count = mesh.group(grouping_config)?;

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("{group_count} Groups from partitions");

        //multi_res.lods.push(to_mesh_layer(&mesh));

        multi_res.clusters.extend_from_slice(&generate_clusters(
            &mesh,
            i,
            &multi_res.verts,
            lower_group_range,
            upper_group_range,
            Some(16),
        ));

        multi_res.group_count += mesh.groups.len();

        lower_group_range = upper_group_range;
        upper_group_range = multi_res.group_count;

        // Patch cluster indexes within groups
        // for gi in lower_group_range..upper_group_range {
        //     for c in &mut multi_res.groups[gi].clusters {
        //         *c += cluster_count;
        //     }
        // }

        if partition_count == 2 {
            println!("Finished with two partitions");
            break;
        }
    }

    stat_readout(&mut multi_res);

    Ok(multi_res)
}

#[cfg(test)]
mod test {

    use std::collections::HashMap;

    use crate::{
        lod::multiresolution::group_and_partition_and_simplify,
        mesh::half_edge_mesh::{test::TEST_MESH_LOW, HalfEdgeMesh},
        STARTING_CLUSTER_SIZE,
    };

    use common::{graph, Asset, BoundingSphere};

    #[test]
    fn test_group_and_partition_and_simplify() {
        println!("Loading from gltf!");
        let (mesh, tri_mesh) = HalfEdgeMesh::from_gltf(TEST_MESH_LOW);

        // mesh.assert_valid().expect("Invalid Mesh");

        //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
        group_and_partition_and_simplify(mesh, tri_mesh, "".to_owned(), 150)
            .expect("Failed to generate")
            .save("../assets/torus_diag.glb.bin")
            .expect("Failed to save");
    }
    #[test]
    fn test_bounds_saturated() {
        let mesh_name = "../../assets/sphere.glb";

        println!("Loading from gltf!");
        let (mesh, tri_mesh) = HalfEdgeMesh::from_gltf(mesh_name);

        //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
        let mesh =
            group_and_partition_and_simplify(mesh, tri_mesh, "".to_owned(), STARTING_CLUSTER_SIZE)
                .unwrap();

        let mut groups = HashMap::<usize, Vec<BoundingSphere>>::new();

        for cluster in &mesh.clusters {
            groups
                .entry(cluster.group_index())
                .or_default()
                .push(cluster.saturated_bound);
        }

        println!("Checking {} groups have shared bounds", groups.len());

        for bounds in groups.values() {
            println!("Checking {} bounds", bounds.len());

            let (eq, _) = bounds.iter().fold((true, None), |(eq, prev), current| {
                let eq = match prev {
                    None => eq,
                    Some(prev) => eq && (prev == current),
                };
                (eq, Some(current))
            });
            assert!(eq, "All saturated bounds should be equal in a group")
        }

        println!("Checking {} groups have monotonic bounds", groups.len());

        for cluster in &mesh.clusters {
            let Some(child_group) = cluster.child_group_index() else {
                continue;
            };

            groups[&cluster.group_index()][0].assert_contains_sphere(&groups[&child_group][0])
        }
    }
}
