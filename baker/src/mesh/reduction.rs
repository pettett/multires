use std::{cmp, collections::HashSet, mem};

//#[cfg(feature = "progress")]
use indicatif::ProgressStyle;

use meshopt::SimplifyOptions;
use obj::{Group, IndexTuple, ObjData, Object, SimplePolygon};
use rayon::prelude::*;

use crate::{
    mesh::{graph::colour_graph, triangle::Triangle},
    pidge::Pidge,
};

use super::{
    cluster_info::ClusterInfo, edge::EdgeID, face::FaceID, quadric::Quadric,
    quadric_error::QuadricError, vertex::VertID, winged_mesh::WingedMesh,
};
use anyhow::Result;

/// Similar to `Quadric`, this data is for internal use only.
#[derive(Clone)]
pub struct CollapseQueue {
    queue: priority_queue::PriorityQueue<EdgeID, QuadricError>,
}

// pub fn tri_area(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> f32 {
//     let ab = b - a;
//     let ac = c - a;

//     glam::Vec3A::cross(ab, ac).length() / 2.0
// }

impl WingedMesh {
    pub fn create_quadrics(&self, verts: &[glam::Vec3A]) -> Vec<Quadric> {
        let mut quadrics = Vec::with_capacity(verts.len());
        //The combination of this and the next step is completely safe, as we initialise everything.
        unsafe {
            quadrics.set_len(verts.len());
        }

        quadrics
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, q)| *q = VertID::new(i).generate_error_matrix(&self, verts));

        quadrics
    }

    fn group_faces(&self) -> Vec<Vec<FaceID>> {
        let mut groups = vec![Vec::new(); self.groups.len()];

        for (fid, face) in self.iter_faces() {
            groups[self.clusters[face.cluster_idx].group_index()].push(fid);
        }
        groups
    }

    fn initialise_collapse_queue(
        &self,
        verts: &[glam::Vec3A],
        quadrics: &[Quadric],
        faces: &[FaceID],
    ) -> CollapseQueue {
        let mut queue = priority_queue::PriorityQueue::with_capacity(faces.len() * 3);

        for &fid in faces {
            let face = self.get_face(fid);
            for eid in [
                face.edge,
                self.get_edge(face.edge).edge_back_cw,
                self.get_edge(face.edge).edge_next_ccw,
            ] {
                let error = eid.edge_collapse_error(&self, verts, &quadrics).unwrap();

                queue.push(eid, error);
            }
        }

        CollapseQueue { queue }
    }

    /// Returns estimate of error introduced by halving the number of triangles
    pub fn reduce_within_groups(
        &mut self,
        verts: &[glam::Vec3A],
        quadrics: &mut [Quadric],
        collapse_requirements: &[usize],
    ) -> Result<f64> {
        assert_eq!(
            self.groups.len(),
            collapse_requirements.len(),
            "must have one required collapse count per group"
        );

        let grouped_faces = self.group_faces();
        let group_effects = self.generate_group_effect_graph();
        let colours = colour_graph(&group_effects);

        //let mut collapse_queues =
        //    self.initialise_collapse_queues(verts, quadrics, collapse_requirements.len());

        //assert_eq!(collapse_queues.len(), collapse_requirements.len());
        assert_eq!(verts.len(), quadrics.len());

        // Priority queue with every vertex and their errors.

        let mut new_error = 0.0;
        println!("Beginning Collapse...");
        // let tris = self.face_count();
        // // Need to remove half the triangles - each reduction removes 2
        // let required_reductions = (tris / 4);
        ////#[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(grouped_faces.len() as _);
        ////#[cfg(feature = "progress")]
        bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len} ({per_sec})").unwrap());

        let mut collapses = Vec::with_capacity(colours[0].len());

        for colour_group in colours {
            // Calculate collapsing queues in parallel for this batch

            //bar.println("Generating queues");

            colour_group
                .par_iter()
                .map(|&group_index| {
                    self.initialise_collapse_queue(verts, quadrics, &grouped_faces[group_index])
                })
                .collect_into_vec(&mut collapses);

            //bar.println("Finished queues");

            for (collapses, group_index) in collapses.iter_mut().zip(colour_group.into_iter()) {
                let requirement = collapse_requirements[group_index];

                //#[cfg(test)]
                //{
                //    //self.assert_valid();
                //    for (&q, _e) in collapse_queues[qi].queue.iter() {
                //        assert!(
                //            self.try_get_edge(q).is_ok(),
                //            "Invalid edge in collapse queue {}",
                //            qi
                //        );
                //        self.assert_edge_valid(q)
                //            .context("Invalid edge in collapse queue: ")
                //            .unwrap();
                //    }
                //}

                ////#[cfg(feature = "progress")]
                //bar.set_message(format!("{err:.3e}"));

                //let mut collapses =
                //    self.initialise_collapse_queue(verts, quadrics, &grouped_faces[group_index]);

                'outer: for i in 0..requirement {
                    let (orig, dest, eid, err) = loop {
                        let (_, QuadricError(cmp::Reverse(err), eid)) = match collapses.queue.pop()
                        {
                            Some(err) => err,
                            None => {
                                // FIXME: how to handle early exits
                                ////#[cfg(feature = "progress")]
                                bar.println(format!(
									"Out of valid edges - Exiting early from de-meshing with {} to go",
									requirement - i - 1
								));
                                break 'outer;
                            }
                        };

                        let Ok((orig, dest)) = eid.src_dst(&self) else {
                            // Invalid edges are allowed to show up in the search
                            continue;
                        };

                        // TODO: We check these here, as many operations can influence if an edge can be collapsed due to these factors,
                        //		 but really an edge collapse should update errors of more edges in the region around it.
                        // 		 - However is is slightly faster to manage this here

                        if !eid.can_collapse_edge(self, verts).unwrap() {
                            continue;
                        }

                        break (orig, dest, eid, err);
                    };

                    new_error += err;

                    // Collapse edge, and update quadrics (update before collapsing, as vertex becomes invalid)

                    //#[cfg(test)]
                    //{
                    //    assert!(orig.is_group_embedded(&self));
                    //}

                    quadrics[dest.id() as usize].0 += quadrics[orig.id() as usize].0;

                    #[allow(unreachable_patterns)]
                    match self.collapse_edge(eid) {
                        Ok(()) => (),
                        #[cfg(test)]
                        e => e.unwrap(),
                        Err(e) => {
                            println!("{e} - Exiting early from de-meshing");
                            break 'outer;
                        }
                    }

                    // All the edges from src are now in dest, so we only need to check those
                    let effected_edges = self
                        .get_vert(dest)
                        .outgoing_edges()
                        .iter()
                        .chain(self.get_vert(dest).incoming_edges());

                    // Update priority queue with new errors
                    for &eid in effected_edges {
                        if let Ok(edge) = self.try_get_edge(eid) {
                            if self.clusters[self.get_face(edge.face).cluster_idx].group_index()
                                == group_index
                            {
                                let error =
                                    eid.edge_collapse_error(&self, verts, &quadrics).unwrap();

                                collapses.queue.push(eid, error);
                            }
                        }
                    }
                }

                //#[cfg(feature = "progress")]
                bar.inc(1);
            }
        }

        //#[cfg(feature = "progress")]
        bar.finish();
        // Get mut to save a single lock
        Ok(new_error)
    }

    /// Returns estimate of error introduced by halving the number of triangles
    pub fn meshopt_reduce_within_groups(&mut self, verts: &[glam::Vec3A]) -> Result<f64> {
        let mut group_tris = vec![Vec::new(); self.groups.len()];

        for (i, f) in self.iter_faces() {
            let gi = self.clusters[f.cluster_idx].group_index();
            group_tris[self.clusters[f.cluster_idx].group_index()]
                .extend_from_slice(&self.triangle_from_face(f));
        }

        // let mut border_edges = HashSet::new();

        // for (i, f) in self.iter_faces() {
        //     let gi = self.clusters[f.cluster_idx].group_index();
        //     group_tris[self.clusters[f.cluster_idx].group_index()]
        //         .extend_from_slice(&self.triangle_from_face(f));

        //     for edge in self.iter_edge_loop(f.edge) {
        //         if let Some(twin) = self.get_edge(edge).twin {
        //             let gi2 = self.clusters[self.get_face(self.get_edge(twin).face).cluster_idx]
        //                 .group_index();
        //             if gi != gi2 {
        //                 // println!("{gi} {gi2}");
        //                 let (s, d) = edge.src_dst(self).unwrap();

        //                 border_edges.insert([s.id().min(d.id()), s.id().max(d.id())]);
        //             }
        //         }
        //     }
        // }

        *self.verts_mut() = Pidge::with_capacity(verts.len());
        *self.edges_mut() = Pidge::with_capacity(self.edge_count());
        *self.faces_mut() = Pidge::with_capacity(self.face_count());

        let len_ratio = mem::size_of_val(&verts[0]) / mem::size_of::<u8>();
        let bytes = unsafe {
            core::slice::from_raw_parts(verts.as_ptr() as *const u8, verts.len() * len_ratio)
        };

        let verts_adapter =
            meshopt::VertexDataAdapter::new(bytes, mem::size_of::<glam::Vec3A>(), 0).unwrap();

        assert_eq!(verts_adapter.vertex_count, verts.len());

        println!("Simplifying groups via MESHOPT");

        // Must lock the border of groups.
        let new_groups = group_tris
            .into_par_iter()
            .map(|group| {
                let new_group = meshopt::simplify(
                    &group,
                    &verts_adapter,
                    group.len() / 2,
                    1.0,
                    SimplifyOptions::LockBorder,
                    None,
                );

                // Double check this is my fault
                for ind in &new_group {
                    assert!(group.contains(ind), "Meshopt has made up new indices!")
                }

                new_group
            })
            .collect_vec_list();

        // Allocate new face IDs

        let mut id = FaceID(0);

        // Forget old clusterings; we have just remade the mesh
        self.clusters = vec![ClusterInfo::default(); self.groups.len()];

        println!("Pushing geometry back to half-edge");

        let mut pushed_edges = 0;
        let mut pushed_faces = 0;

        // println!("Done");

        for (group_idx, new_group) in new_groups.iter().flatten().enumerate() {
            // self.assert_valid().unwrap();

            self.clusters[group_idx].set_group_index_once(group_idx);
            for tri in new_group.chunks_exact(3) {
                assert_eq!(tri.len(), 3);

                let [a, b, c] = tri else { unreachable!() };

                if a == b && b == c {
                    // Degenerate triangle - meshopt will sometimes turn a line of 3 points into a triangle on a border

                    continue;
                }

                match self.add_tri(id, (*a).into(), (*b).into(), (*c).into()) {
                    Ok(_) => {
                        self.get_face_mut(id).cluster_idx = group_idx;

                        pushed_edges += 3;
                        pushed_faces += 1;
                        id.0 += 1;
                    }
                    Err(me) => match me {
                        crate::mesh::winged_mesh::MeshError::EdgeExists(e) => {
                            let e = self.get_edge(e);

                            // println!("{:?}", e.twin);

                            println!(
                                "Duplicate face from group {} (we are group {})",
                                self.get_face(e.face).cluster_idx,
                                group_idx
                            );
                        }
                        _ => unreachable!(),
                    },
                }
            }
            assert_eq!(self.face_count(), pushed_faces);
            assert_eq!(self.edge_count(), pushed_edges);

            // println!("{pushed_faces} {}", new_group.len() / 3)
        }

        // if failed {
        let mut obj = ObjData::default();

        obj.position = verts.iter().map(|v| [v[0], v[1], v[2]]).collect();

        // obj.normal = multires
        //     .verts
        //     .into_iter()
        //     .map(|v| [v.normal[0], v.normal[1], v.normal[2]])
        //     .collect();

        for (i, cluster) in new_groups.iter().flatten().enumerate() {
            let name = format!("Cluster  I{}", i);

            let mut group = Group::new("0".to_string());

            // Indices to polygons
            for t in cluster.chunks_exact(3) {
                group.polys.push(SimplePolygon(
                    [
                        IndexTuple(t[0] as _, None, None),
                        IndexTuple(t[1] as _, None, None),
                        IndexTuple(t[2] as _, None, None),
                    ]
                    .to_vec(),
                ))
            }

            let mut object = Object::new(name);

            object.groups.push(group);

            obj.objects.push(object)
        }

        obj.save(format!("shit.obj")).unwrap();
        // }

        // assert!(!failed, "Failed");

        println!("Simplification Complete!");

        // Get mut to save a single lock
        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::mesh::{plane::Plane, winged_mesh::test::TEST_MESH_MONK};

    use super::super::winged_mesh::{test::TEST_MESH_HIGH, WingedMesh};

    fn plane_distance(plane: &Plane, point: glam::Vec3A) -> f32 {
        point.dot(plane.0.into()) + plane.0.w
    }

    // Test that each face generates a valid plane
    #[test]
    pub fn test_planes() -> Result<(), Box<dyn Error>> {
        let (mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value
        let e = 0.000001;

        for (i, (fid, f)) in mesh.iter_faces().enumerate() {
            let plane = fid.plane(&mesh, &tri_mesh.verts);
            let n = plane.normal();
            assert!(
                ((n.x * n.x + n.y * n.y + n.z * n.z) - 1.0).abs() < e,
                "Plane {i} is not normalised"
            );

            for v in mesh.triangle_from_face(&f) {
                let v_dist = plane_distance(&plane, tri_mesh.verts[v as usize].into()).abs();
                assert!(
                    v_dist < e,
                    "Plane invalid at index {v}, tri {i}, value {v_dist}",
                );
            }
        }

        Ok(())
    }

    // Test that each face/plane generates an equivalent quadric matrix
    #[test]
    pub fn test_plane_quadrics() -> Result<(), Box<dyn Error>> {
        let (mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value, only valid for errors within around 50 units.
        let e = 0.001;

        let random_points = [glam::Vec3A::X, glam::Vec3A::Y * 50.0];

        for (i, (fid, f)) in mesh.iter_faces().enumerate() {
            let plane = fid.plane(&mesh, &tri_mesh.verts);

            let mat = plane.fundamental_error_quadric();

            let cols = mat.0.to_cols_array_2d();

            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            for v in mesh.triangle_from_face(&f) {
                let v = tri_mesh.verts[v as usize];
                // v^t * K_p * v
                let q_error = mat.quadric_error(v.into());
                assert!(
                    q_error < e,
                    "Plane invalid at index {v}, tri {i}, value {q_error}",
                );
            }

            for &v in &random_points {
                let q_error = mat.quadric_error(v);
                let p_dist = plane_distance(&plane, v);
                let sqr_p_dist = (p_dist * p_dist) as f64;

                assert!(
                    q_error - sqr_p_dist < e,
                    "Plane different to quadric at point {v}. P: {sqr_p_dist} Q: {q_error}",
                );
            }
        }

        Ok(())
    }

    // Test that each vertex generates a valid quadric matrix that returns 0 at itself.
    #[test]
    pub fn test_vert_quadrics() -> Result<(), Box<dyn Error>> {
        let (mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_MONK);

        let e = 0.0000000001;

        for (vid, v) in mesh.iter_verts() {
            let q = v.generate_error_matrix(&mesh, &tri_mesh.verts);

            let cols = q.0.to_cols_array_2d();
            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            let v = tri_mesh.verts[vid.id() as usize];
            // v^t * K_p * v
            let q_error = q.quadric_error(v.into());
            assert!(
                q_error < e,
                "Plane invalid at index {v}, tri {vid:?}, value {q_error}",
            );
        }

        Ok(())
    }

    #[test]
    pub fn test_reduction() -> Result<(), Box<dyn Error>> {
        let (mut mesh, tri_mesh) = WingedMesh::from_gltf(TEST_MESH_MONK);
        let mut quadrics = mesh.create_quadrics(&tri_mesh.verts);

        mesh.group_unity();
        for _i in 0..4 {
            mesh.assert_valid().unwrap();
            mesh.reduce_within_groups(&tri_mesh.verts, &mut quadrics, &[mesh.face_count() / 4])
                .unwrap();
        }

        mesh.assert_valid().unwrap();

        Ok(())
    }
}
