pub mod mesh;
pub mod pidge;

use std::{
    collections::{BTreeSet, HashSet},
    ops::Range,
    vec,
};

use common::{asset::Asset, graph, MeshLevel, MeshVert, MultiResMesh, SubMesh};

use glam::Vec4;
use mesh::winged_mesh::WingedMesh;
use petgraph::visit::IntoNeighbors;

// use meshopt::VertexDataAdapter;

const CLUSTERS_PER_SIMPLIFIED_GROUP: usize = 2;
const STARTING_CLUSTER_SIZE: usize = 280;

const COLOUR_CLUSTER_SIZE: usize = 55;
//TODO: Curb random sized groups and the like to bring this number to more reasonable amounts
//const MAX_TRIS_PER_CLUSTER: usize = 126 * 3;

pub fn to_mesh_layer(mesh: &WingedMesh, verts: &[Vec4]) -> MeshLevel {
    MeshLevel {
        partition_indices: mesh.get_partition(),
        group_indices: mesh.get_group(),
        indices: grab_indicies(&mesh),
        submeshes: generate_submeshes(mesh, verts),
        clusters: mesh.clusters.clone(),
        groups: mesh.groups.clone(),
    }
}

pub fn group_and_partition_and_simplify(
    mut mesh: WingedMesh,
    verts: &[Vec4],
    normals: &[Vec4],
) -> Vec<MeshLevel> {
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

    let non_contig_even_clustering_config = &metis::PartitioningConfig {
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

    let mut quadrics = mesh.create_quadrics(verts);

    // Apply primary partition, that will define the lowest level clusterings
    mesh.partition_full_mesh(
        triangle_clustering_config,
        mesh.face_count().div_ceil(STARTING_CLUSTER_SIZE) as _,
    )
    .unwrap();

    mesh.group(grouping_config, &verts).unwrap();

    let mut layers = Vec::new();

    mesh.colour_within_clusters(&non_contig_even_clustering_config, COLOUR_CLUSTER_SIZE)
        .unwrap();

    layers.push(to_mesh_layer(&mesh, &verts));

    // Generate 2 more meshes
    for i in 0..10 {
        // i = index of previous mesh layer
        //working_mesh = reduce_mesh(working_mesh);

        println!("Face count L{}: {}", i, mesh.face_count());

        // We must regenerate the queue each time, as boundaries change.

        // Each group requires half it's triangles removed

        let collapse_requirements: Vec<usize> = mesh
            .groups
            .iter()
            .map(|g| {
                g.tris / 4
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
        #[cfg(test)]
        {
            println!("Ensuring groups are contiguous... ");
            let graphs = mesh.generate_group_keyed_graphs();
            for graph in graphs {
                graph::assert_graph_contiguous(&graph);
            }
        }

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

        let partition_count = match mesh.partition_within_groups(
            &group_clustering_config,
            Some(CLUSTERS_PER_SIMPLIFIED_GROUP as _),
            None,
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

        let group_count = mesh.group(&grouping_config, &verts).unwrap();

        mesh.colour_within_clusters(&non_contig_even_clustering_config, COLOUR_CLUSTER_SIZE)
            .unwrap();

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("{group_count} Groups from partitions");

        layers.push(to_mesh_layer(&mesh, &verts));

        if group_count == 1 {
            println!("Finished with single group");
            break;
        }
    }

    let mut min_tris = 10000;
    let mut max_tris = 0;
    for l in &layers {
        for m in &l.submeshes {
            min_tris = min_tris.min(m.index_count() / 3);
            max_tris = max_tris.max(m.index_count() / 3);
        }
    }

    println!("Done with partitioning. Min tris: {min_tris}, Max tris: {max_tris}");

    //assert_eq!(partitions1.len() * 3, layer_1_indices.len());

    layers
}

pub fn apply_simplification(mut mesh: WingedMesh, verts: &[Vec4], name: String) -> WingedMesh {
    // Apply primary partition, that will define the lowest level clusterings

    mesh.groups = vec![
        common::GroupInfo {
            tris: 0,
            monotonic_bound: Default::default(),
            partitions: vec![0],
            group_neighbours: BTreeSet::new()
        };
        1
    ];

    let mut layers = Vec::new();

    layers.push(to_mesh_layer(&mesh, &verts));

    let mut quadrics = mesh.create_quadrics(verts);
    // Generate 2 more meshes
    for i in 0..8 {
        // i = index of previous mesh layer
        println!(
            "Face count LOD{}: {}, beginning generating LOD{}",
            i,
            mesh.face_count(),
            i + 1
        );

        let _e = match mesh.reduce_within_groups(verts, &mut quadrics, &[mesh.face_count() / 4]) {
            Ok(e) => e,
            Err(e) => {
                println!(
                    "Experience error {} with reducing, exiting early with what we have",
                    e
                );
                break;
            }
        };

        // View a snapshot of the mesh without any re-groupings applied

        layers.push(to_mesh_layer(&mesh, &verts));

        if mesh.face_count() < 10 {
            println!("Reduced to low enough amount of faces, ending");
            break;
        }
    }

    println!("Done with partitioning");

    //assert_eq!(partitions1.len() * 3, layer_1_indices.len());

    MultiResMesh {
        name,
        verts: verts
            .iter()
            .map(|v| MeshVert {
                pos: [v.x, v.y, v.z, 1.0],
                normal: [0.0; 4],
            })
            .collect(),
        // layer_1_indices: indices.clone(),
        lods: layers,
    }
    .save("simplified_asset.bin")
    .unwrap();

    mesh
}

pub fn grab_indicies(mesh: &WingedMesh) -> Vec<u32> {
    let mut indices = Vec::with_capacity(mesh.face_count() * 3);

    for (_fid, f) in mesh.iter_faces() {
        let [a, b, c] = mesh.triangle_from_face(&f);
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

/// Debug code to generate meshlets with no max size. Used for testing partition trees with no remeshing
pub fn generate_submeshes(mesh: &WingedMesh, _verts: &[Vec4]) -> Vec<SubMesh> {
    println!("Generating meshlets!");

    let inds = 5.0 / mesh.face_count() as f32;

    // Precondition: partition indexes completely span in some range 0..N
    let mut submeshes: Vec<_> = mesh
        .clusters
        .iter()
        .map(|cluster| {
            let gi = cluster.group_index;
            let g = &mesh.groups[gi];

            SubMesh::new(
                cluster.num_colours,
                //TODO: Connect to quadric error or something
                1.0,
                g.monotonic_bound.center(),
                g.monotonic_bound.radius(),
                cluster.tight_bound.radius(),
                gi,
            )
        })
        .collect();

    for (_fid, face) in mesh.iter_faces() {
        let verts = mesh.triangle_from_face(&face);

        let m = submeshes.get_mut(face.cluster_idx as usize).unwrap();

        m.push_tri(face.colour, verts);

        m.error += inds;
    }

    for s in &submeshes {
        for i in 0..s.colour_count() {
            assert!(s.colour_vert_count(i) <= 64)
        }
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

#[cfg(test)]
mod test {
    use crate::mesh::winged_mesh::test::TEST_MESH_CONE;

    use super::*;
    use common::graph;

    #[test]
    fn test_contiguous_meshes() {
        println!("Loading from gltf!");
        let (mesh, _verts, norms) = WingedMesh::from_gltf("../../assets/dragon_1m.glb");

        let mesh_dual = mesh.generate_face_graph();

        println!("Testing Contiguous!");
        graph::assert_graph_contiguous(&mesh_dual);
    }

    #[test]
    fn test_group_and_partition_and_simplify() {
        let mesh_name = "../../assets/sphere.glb";

        println!("Loading from gltf!");
        let (mesh, verts, norms) = WingedMesh::from_gltf(mesh_name);

        //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
        group_and_partition_and_simplify(mesh, &verts, &norms);
    }

    #[test]
    fn test_apply_simplification() {
        let (mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_CONE);

        // WE know the circle is contiguous
        //assert_contiguous_graph(&working_mesh.generate_face_graph());

        // group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        let mesh = apply_simplification(mesh, &verts, TEST_MESH_CONE.to_owned());

        println!("Asserting face graph is contiguous");
        // It should still be contiguous
        graph::assert_graph_contiguous(&mesh.generate_face_graph());
    }
}
