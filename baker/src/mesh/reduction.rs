use std::{cmp, ops};

use glam::vec4;
#[cfg(feature = "progress")]
use indicatif::ProgressStyle;

use rayon::prelude::*;

use super::{
    edge::EdgeID,
    quadric::Quadric,
    quadric_error::QuadricError,
    vertex::VertID,
    winged_mesh::{MeshError, WingedMesh},
};
use anyhow::{Context, Result};

/// Similar to `Quadric`, this data is for internal use only.
#[derive(Clone)]
pub struct CollapseQueue {
    queue: priority_queue::PriorityQueue<EdgeID, QuadricError>,
}

pub fn tri_area(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> f32 {
    let ab = b - a;
    let ac = c - a;

    glam::Vec3A::cross(ab, ac).length() / 2.0
}

impl WingedMesh {
    pub fn create_quadrics(&self, verts: &[glam::Vec4]) -> Vec<Quadric> {
        let mut quadrics = Vec::with_capacity(verts.len());
        //The combination of this and the next step is completely safe, as we initialise everything.
        unsafe {
            quadrics.set_len(verts.len());
        }

        quadrics
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, q)| *q = VertID(i).generate_error_matrix(&self, verts));

        quadrics
    }

    /// Generate a vector of priority queues for the mesh, based on errors calculated in `edge_collapse_error`.
    ///
    /// `queue_lookup` -  Function to index a face into the queue it should have its error placed into. Must output `0..=queue_count`.
    pub fn initialise_collapse_queues(
        &self,
        verts: &[glam::Vec4],
        quadrics: &[Quadric],
        queue_count: usize,
    ) -> Vec<CollapseQueue> {
        println!("Generating Collapse Queues...");
        let mut pq = Vec::with_capacity(queue_count);

        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.edge_count() as _);

        let queue_t = priority_queue::PriorityQueue::with_capacity(self.edge_count() / queue_count);
        for _ in 0..queue_count {
            pq.push(CollapseQueue {
                queue: queue_t.clone(),
            });

            #[cfg(feature = "progress")]
            bar.inc(1);
        }
        #[cfg(feature = "progress")]
        bar.finish();
        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.edge_count() as _);

        for (eid, edge) in self.iter_edges() {
            let error = eid.edge_collapse_error(&self, verts, &quadrics).unwrap();

            pq[self.clusters[self.get_face(edge.face).cluster_idx].group_index]
                .queue
                .push(eid, error);

            #[cfg(feature = "progress")]
            bar.inc(1);
        }
        #[cfg(feature = "progress")]
        bar.finish();

        pq
    }

    /// Returns estimate of error introduced by halving the number of triangles
    pub fn reduce_within_groups(
        &mut self,
        verts: &[glam::Vec4],
        quadrics: &mut [Quadric],
        collapse_requirements: &[usize],
    ) -> Result<f64> {
        let mut collapse_queues =
            self.initialise_collapse_queues(verts, quadrics, collapse_requirements.len());

        assert_eq!(collapse_queues.len(), collapse_requirements.len());
        assert_eq!(verts.len(), quadrics.len());

        // Priority queue with every vertex and their errors.

        let mut new_error = 0.0;
        println!("Beginning Collapse...");
        // let tris = self.face_count();
        // // Need to remove half the triangles - each reduction removes 2
        // let required_reductions = (tris / 4);
        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(collapse_queues.len() as _);
        #[cfg(feature = "progress")]
        bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len} ({per_sec})").unwrap());

        for (qi, &requirement) in collapse_requirements.iter().enumerate() {
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

            //#[cfg(feature = "progress")]
            //bar.set_message(format!("{err:.3e}"));

            'outer: for i in 0..requirement {
                let (orig, dest, eid, err) = loop {
                    let (_, QuadricError(cmp::Reverse(err), eid)) =
                        match collapse_queues[qi].queue.pop() {
                            Some(err) => err,
                            None => {
                                // FIXME: how to handle early exits
                                #[cfg(feature = "progress")]
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

                quadrics[dest.0].0 += quadrics[orig.0].0;

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
                        let edge_queue =
                            self.clusters[self.get_face(edge.face).cluster_idx].group_index;

                        let error = eid.edge_collapse_error(&self, verts, &quadrics).unwrap();

                        collapse_queues[edge_queue].queue.push(eid, error);
                    }
                }
            }

            #[cfg(feature = "progress")]
            bar.inc(1);
        }

        #[cfg(feature = "progress")]
        bar.finish();
        // Get mut to save a single lock
        Ok(new_error)
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
        let (mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value
        let e = 0.000001;

        for (i, (fid, f)) in mesh.iter_faces().enumerate() {
            let plane = fid.plane(&mesh, &verts);
            let n = plane.normal();
            assert!(
                ((n.x * n.x + n.y * n.y + n.z * n.z) - 1.0).abs() < e,
                "Plane {i} is not normalised"
            );

            for v in mesh.triangle_from_face(&f) {
                let v_dist = plane_distance(&plane, verts[v].into()).abs();
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
        let (mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value, only valid for errors within around 50 units.
        let e = 0.001;

        let random_points = [glam::Vec3A::X, glam::Vec3A::Y * 50.0];

        for (i, (fid, f)) in mesh.iter_faces().enumerate() {
            let plane = fid.plane(&mesh, &verts);

            let mat = plane.fundamental_error_quadric();

            let cols = mat.0.to_cols_array_2d();

            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            for v in mesh.triangle_from_face(&f) {
                let v = verts[v];
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
        let (mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_MONK);

        let e = 0.0000000001;

        for vid in mesh.iter_verts() {
            let q = vid.0.generate_error_matrix(&mesh, &verts);

            let cols = q.0.to_cols_array_2d();
            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            let v = verts[vid.0 .0];
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
        let (mut mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_MONK);
        let mut quadrics = mesh.create_quadrics(&verts);

        for _i in 0..4 {
            mesh.assert_valid().unwrap();
            mesh.reduce_within_groups(&verts, &mut quadrics, &[mesh.face_count() / 4])
                .unwrap();
        }

        mesh.assert_valid().unwrap();

        Ok(())
    }
}
