pub mod evaluation;
pub mod mesh;
pub mod pidge;

use std::mem;

use common::{graph, tri_mesh::TriMesh, MeshCluster, MeshLevel, MeshVert, Meshlet, MultiResMesh};

use glam::Vec4;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use mesh::winged_mesh::WingedMesh;
use meshopt::SimplifyOptions;
use mimalloc::MiMalloc;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
// use meshopt::VertexDataAdapter;

const CLUSTERS_PER_SIMPLIFIED_GROUP: usize = 2;
const STARTING_CLUSTER_SIZE: usize = 280;

const COLOUR_CLUSTER_SIZE: usize = 55;
//TODO: Curb random sized groups and the like to bring this number to more reasonable amounts
//const MAX_TRIS_PER_CLUSTER: usize = 126 * 3;

pub fn to_mesh_layer(mesh: &WingedMesh) -> MeshLevel {
    MeshLevel {
        partition_indices: mesh.get_partition(),
        group_indices: mesh.get_group(),
        indices: grab_indices(mesh),
    }
}

pub fn group_and_partition_and_simplify(
    mut mesh: WingedMesh,
    tri_mesh: TriMesh,
    name: String,
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

    let mut quadrics = mesh.create_quadrics(&tri_mesh.verts);

    // Apply primary partition, that will define the lowest level clusterings
    mesh.cluster_full_mesh(
        triangle_clustering_config,
        mesh.face_count().div_ceil(STARTING_CLUSTER_SIZE) as _,
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
        lods: Vec::new(),
        clusters: Vec::new(),
        group_count: 0,
    };

    let mut lower_group_range = 0;
    multi_res.group_count += &mesh.groups.len();
    let mut upper_group_range = multi_res.group_count;

    multi_res.lods.push(to_mesh_layer(&mesh));

    multi_res
        .clusters
        .extend_from_slice(&generate_clusters(&mesh, 0, &multi_res.verts, 0, 0));

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

        let partition_count = match mesh.partition_within_groups(
            group_clustering_config,
            &tri_mesh.verts,
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

        let group_count = mesh.group(grouping_config)?;

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("{group_count} Groups from partitions");

        multi_res.lods.push(to_mesh_layer(&mesh));

        multi_res.clusters.extend_from_slice(&generate_clusters(
            &mesh,
            i,
            &multi_res.verts,
            lower_group_range,
            upper_group_range,
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

        if group_count == 1 {
            println!("Finished with single group");
            break;
        }
    }

    stat_readout(&mut multi_res);

    Ok(multi_res)
}

pub fn simplify_lod_chain(
    mut mesh: WingedMesh,
    tri_mesh: TriMesh,
    name: String,
) -> anyhow::Result<MultiResMesh> {
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

    let mut quadrics = mesh.create_quadrics(&tri_mesh.verts);

    // Apply primary partition, that will define the lowest level clusterings
    mesh.cluster_unity(None);

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
        lods: Vec::new(),
        clusters: Vec::new(),
        group_count: 0,
    };

    let mut lower_group_range = 0;
    multi_res.group_count += &mesh.groups.len();
    let mut upper_group_range = multi_res.group_count;

    multi_res.lods.push(to_mesh_layer(&mesh));

    multi_res
        .clusters
        .extend_from_slice(&generate_clusters(&mesh, 0, &multi_res.verts, 0, 0));

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
        mesh.cluster_unity(Some(0));

        println!("1 Partitions from groups");

        mesh.group(grouping_config)?;

        // view a snapshot of the mesh ready to create the next layer
        // let error = (1.0 + i as f32) / 10.0 + rng.gen_range(-0.05..0.05);

        println!("1 Groups from partitions");

        multi_res.lods.push(to_mesh_layer(&mesh));

        multi_res.clusters.extend_from_slice(&generate_clusters(
            &mesh,
            i,
            &multi_res.verts,
            lower_group_range,
            upper_group_range,
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

pub fn meshopt_simplify_lod_chain(tri_mesh: TriMesh, name: String) -> anyhow::Result<MultiResMesh> {
    let verts = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&tri_mesh.verts),
        mem::size_of::<glam::Vec4>(),
        0,
    )
    .unwrap();

    let mut indices = tri_mesh.indices.to_vec();

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
        lods: Vec::new(),
        clusters: Vec::new(),
        group_count: 0,
    };

    for i in 0..9 {
        let prev_indices = indices.clone();

        indices =
            meshopt::simplify_sloppy(&prev_indices, &verts, prev_indices.len() / 2, 1.0, None);

        println!("Simplified to {} indices", indices.len());

        todo!("Fix giant clusters")

        // multi_res
        //     .clusters
        //     .push(MeshCluster::new_raw_temp(prev_indices, i));
    }

    //opt_multires(&mut multi_res);

    Ok(multi_res)
}

fn stat_readout(multi_res: &mut MultiResMesh) {
    let mut min_tris = 10000;
    let mut max_tris = 0;
    let mut max_colours = 0;

    let mut total_colours = 0;
    let mut total_indices = 0;
    let mut total_stripped_indices = 0;

    for m in &multi_res.clusters {
        min_tris = min_tris.min(m.index_count() / 3);
        max_tris = max_tris.max(m.index_count() / 3);
        max_colours = max_colours.max(m.colour_count());

        total_colours += m.colour_count();
        total_indices += m.index_count();
        total_stripped_indices += m.stripped_index_count();
    }

    println!("Done with partitioning. Min tris: {min_tris}, Max tris: {max_tris}, Max colours: {max_colours}, Indices per colours: {}. Total indices: {total_indices}, stripped indices: {total_stripped_indices}", total_indices/total_colours);

    //assert_eq!(partitions1.len() * 3, layer_1_indices.len());
}

pub fn grab_indices(mesh: &WingedMesh) -> Vec<u32> {
    let mut indices = Vec::with_capacity(mesh.face_count() * 3);

    for (_fid, f) in mesh.iter_faces() {
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

/// Generate clusters, splitting too large meshlets by colour
/// Also fix group indexing to be global scope, both in our own data and the global group array
pub fn generate_clusters(
    mesh: &WingedMesh,
    lod: usize,
    verts: &[MeshVert],
    child_group_offset: usize,
    group_offset: usize,
) -> Vec<MeshCluster> {
    println!("Generating meshlets!");

    let inds = 5.0 / mesh.face_count() as f32;

    // Precondition: partition indexes completely span in some range 0..N
    let mut clusters: Vec<_> = mesh
        .clusters
        .par_iter()
        .enumerate()
        .map(|(_i, cluster)| {
            let gi = cluster.group_index;
            let group = &mesh.groups[gi];

            MeshCluster::new(
                1,
                //TODO: Connect to quadric error or something
                1.0,
                cluster.tight_bound,
                cluster.tight_cone,
                group.saturated_bound,
                lod,
                cluster.group_index + group_offset,
                cluster.child_group_index.map(|c| c + child_group_offset),
            )
        })
        .collect();

    // Split into large blocks to save allocating massive amounts of memory
    let step = 1000;
    let mut cluster_indices =
        vec![Vec::with_capacity(STARTING_CLUSTER_SIZE / 4); (mesh.clusters.len()).min(step)];

    for i in (0..mesh.clusters.len()).step_by(step).progress() {
        let cluster_count = (mesh.clusters.len() - i).min(step);
        let cluster_range = i..i + cluster_count;

        for (_fid, face) in mesh.iter_faces() {
            if cluster_range.contains(&face.cluster_idx) {
                let verts = mesh.triangle_from_face(face);

                cluster_indices[face.cluster_idx - i].extend_from_slice(&verts);
            }

            // let m = clusters.get_mut(face.cluster_idx).unwrap();
            // m.error += inds;
        }

        clusters[cluster_range]
            .par_iter_mut()
            .zip(cluster_indices.par_iter_mut())
            .for_each(|(cluster, cluster_indices)| {
                //condense cluster indices down to local lookup

                let mut local_verts = Vec::new();
                let mut local_vert_positions = Vec::new();
                let mut local_indices = Vec::new();

                for &mesh_vert in cluster_indices.iter() {
                    let i = (match local_verts.iter().position(|&x| x == mesh_vert) {
                        Some(idx) => idx,
                        None => {
                            local_vert_positions.push(verts[mesh_vert as usize].pos);
                            local_verts.push(mesh_vert);

                            local_verts.len() - 1
                        }
                    }) as u32;

                    local_indices.push(i)
                }

                let local_vert_adapter = meshopt::VertexDataAdapter::new(
                    bytemuck::cast_slice(&local_vert_positions),
                    mem::size_of_val(&local_vert_positions[0]),
                    0,
                )
                .unwrap();

                let indices = meshopt::optimize_vertex_cache(&local_indices, local_verts.len());
                cluster_indices.clear();

                let meshlets = meshopt::build_meshlets(&indices, &local_vert_adapter, 64, 124, 0.1);

                cluster.error += inds * indices.len() as f32;

                for m in meshlets.iter() {
                    //remap the meshlet-local to mesh local verts

                    let mut verts = Vec::with_capacity(m.vertices.len());
                    for &v in m.vertices {
                        verts.push(local_verts[v as usize]);
                    }

                    cluster.add_meshlet(Meshlet::from_local_indices(m.triangles.to_vec(), verts));
                }

                // large upper bound
                assert!(cluster.meshlets().len() < 16);

                // optimise_clusters
                // for c in 0..cluster.colour_count() {
                //     let meshlet = cluster.meshlet_for_colour_mut(c);
                //     let vertex_count = meshlet.vert_count();
                //     let indices = meshlet.local_indices_mut();

                //     //*indices = meshopt::optimize_vertex_cache(indices, vertex_count);

                //     // Compressed a previously optimised cluster array to triangle strips

                //     //*meshlet.strip_indices_mut() =
                //     //    meshopt::stripify(meshlet.local_indices(), vertex_count, 0).unwrap();
                // }
            });
    }
    // for cluster in &mut clusters {
    //     for i in 0..cluster.colour_count() {
    //         assert!(cluster.meshlet_for_colour(i).vert_count() <= 64);
    //     }
    // }

    // let data = VertexDataAdapter::new(bytemuck::cast_slice(&verts), std::mem::size_of::<Vec4>(), 0)
    //     .unwrap();

    // for s in &mut submeshes {
    //     s.indices = meshopt::simplify(&s.indices, &data, s.indices.len() / 2, 0.01);
    // }

    clusters
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

    use std::collections::HashMap;

    use super::*;
    use common::{graph, BoundingSphere};

    #[test]
    fn test_contiguous_meshes() {
        println!("Loading from gltf!");
        let (mesh, tri_mesh) = WingedMesh::from_gltf("../../assets/dragon_1m.glb");

        let mesh_dual = mesh.generate_face_graph();

        println!("Testing Contiguous!");
        graph::assert_graph_contiguous(&mesh_dual);
    }

    #[test]
    fn test_group_and_partition_and_simplify() {
        let mesh_name = "../../assets/sphere.glb";

        println!("Loading from gltf!");
        let (mesh, tri_mesh) = WingedMesh::from_gltf(mesh_name);

        //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
        group_and_partition_and_simplify(mesh, tri_mesh, "".to_owned());
    }

    #[test]
    fn test_bounds_saturated() {
        let mesh_name = "../../assets/sphere.glb";

        println!("Loading from gltf!");
        let (mut mesh, tri_mesh) = WingedMesh::from_gltf(mesh_name);

        //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
        let mesh = group_and_partition_and_simplify(mesh, tri_mesh, "".to_owned()).unwrap();

        let mut groups = HashMap::<usize, Vec<BoundingSphere>>::new();

        for cluster in &mesh.clusters {
            groups
                .entry(cluster.group_index)
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
            let Some(child_group) = cluster.child_group_index else {
                continue;
            };

            groups[&cluster.group_index][0].assert_contains_sphere(&groups[&child_group][0])
        }
    }

    // #[test]
    // fn test_apply_simplification() {
    //     let (mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_CONE);

    //     // WE know the circle is contiguous
    //     //assert_contiguous_graph(&working_mesh.generate_face_graph());

    //     // group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //     let mesh = apply_simplification(mesh, &verts, TEST_MESH_CONE.to_owned());

    //     println!("Asserting face graph is contiguous");
    //     // It should still be contiguous
    //     graph::assert_graph_contiguous(&mesh.generate_face_graph());
    // }
}
