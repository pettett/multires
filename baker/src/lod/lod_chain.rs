use common::{MeshVert, MultiResMesh, TriMesh};

use crate::mesh::winged_mesh::WingedMesh;

use super::{generate_clusters, stat_readout};

pub fn simplify_lod_chain(
    mut mesh: WingedMesh,
    tri_mesh: TriMesh,
    name: String,
) -> anyhow::Result<MultiResMesh> {
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
    mesh.cluster_unity();

    mesh.group(grouping_config).unwrap();

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
        None,
    ));

    // Generate more meshes
    for i in 1..10 {
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

        // Make a new cluster for the new LOD level
        mesh.cluster_unity();

        println!("1 Partitions from groups");

        mesh.group(grouping_config)?;

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("1 Groups from partitions");

        //multi_res.lods.push(to_mesh_layer(&mesh));

        multi_res.clusters.extend_from_slice(&generate_clusters(
            &mesh,
            i,
            &multi_res.verts,
            lower_group_range,
            upper_group_range,
            None,
        ));

        multi_res.group_count += mesh.groups.len();

        lower_group_range = upper_group_range;
        upper_group_range = multi_res.group_count;
    }
    // Patch cluster indexes within groups
    // for gi in lower_group_range..upper_group_range {
    //     for c in &mut multi_res.groups[gi].clusters {
    //         *c += cluster_count;
    //     }
    // }

    stat_readout(&mut multi_res);

    Ok(multi_res)
}
