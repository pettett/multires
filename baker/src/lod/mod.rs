use std::mem;

use common::{MeshCluster, MeshVert, Meshlet, MultiResMesh};
use indicatif::ProgressIterator;
use rayon::prelude::*;

use crate::{mesh::half_edge_mesh::HalfEdgeMesh, STARTING_CLUSTER_SIZE};

pub mod lod_chain;
pub mod meshopt_chain;
pub mod meshopt_multiresolution;
pub mod multiresolution;

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

pub fn grab_indices(mesh: &HalfEdgeMesh) -> Vec<u32> {
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
    mesh: &HalfEdgeMesh,
    lod: usize,
    verts: &[MeshVert],
    child_group_offset: usize,
    group_offset: usize,
    max_meshlets_per_group: Option<usize>,
) -> Vec<MeshCluster> {
    println!("Generating meshlets!");

    //let inds = 5.0 / mesh.face_count() as f32;

    // Precondition: partition indexes completely span in some range 0..N
    let mut clusters: Vec<_> = mesh
        .clusters
        .par_iter()
        .enumerate()
        .map(|(_i, cluster)| {
            let gi = cluster.group_index();
            let group = &mesh.groups[gi];

            MeshCluster::new(
                group.saturated_error,
                cluster.tight_bound,
                cluster.tight_cone,
                group.saturated_bound,
                lod,
                gi + group_offset,
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

                //cluster.error += inds * indices.len() as f32;

                for m in meshlets.iter() {
                    //remap the meshlet-local to mesh local verts

                    let mut verts = Vec::with_capacity(m.vertices.len());
                    for &v in m.vertices {
                        verts.push(local_verts[v as usize]);
                    }

                    cluster.add_meshlet(Meshlet::from_local_indices(m.triangles.to_vec(), verts));
                }

                // large upper bound
                if let Some(max_meshlets) = max_meshlets_per_group {
                    assert!(
                        cluster.meshlets().len() < max_meshlets,
                        "Too many meshlets in this group"
                    );
                }
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
