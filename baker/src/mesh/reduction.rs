use std::cmp;

use glam::{Vec4, Vec4Swizzles};
#[cfg(feature = "progress")]
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use super::{
    vertex::{VertID, Vertex},
    winged_mesh::{EdgeID, Face, FaceID, MeshError, WingedMesh},
};
use anyhow::{Context, Result};

#[derive(Clone, Copy)]
pub struct Plane(glam::Vec4);
/// Quadric type. Internally a DMat4, kept private to ensure only valid reduction operations can effect it.
pub struct Quadric(glam::DMat4);

#[derive(PartialOrd, PartialEq, Clone, Copy)]
struct OrdF64(cmp::Reverse<f64>);
/// Similar to `Quadric`, this data is for internal use only.
#[derive(Clone)]
pub struct CollapseQueue {
    queue: priority_queue::PriorityQueue<EdgeID, OrdF64>,
}

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if let Some(ordering) = self.partial_cmp(other) {
            ordering
        } else {
            // Choose what to do with NaNs, for example:
            panic!("Cannot order invalid floats")
        }
    }
}

pub fn tri_area(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> f32 {
    let ab = b - a;
    let ac = c - a;

    glam::Vec3A::cross(ab, ac).length() / 2.0
}
impl Plane {
    pub fn from_three_points(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> Self {
        let ab = b - a;
        let ac = c - a;

        let normal = glam::Vec3A::cross(ab, ac).normalize();

        Self::from_normal_and_point(normal, a)
    }

    pub fn from_normal_and_point(norm: glam::Vec3A, p: glam::Vec3A) -> Self {
        let d = -p.dot(norm);

        let plane: glam::Vec4 = (norm, d).into();

        Plane(plane)
    }

    /// The fundamental error quadric `K_p`, such that `v^T K_p v` = `sqr distance v <-> p`
    /// Properties: Additive, Symmetric.
    pub fn fundamental_error_quadric(self) -> Quadric {
        let p: glam::DVec4 = self.0.into();
        let (a, b, c, d) = p.into();

        // Do `p p^T`
        Quadric(glam::DMat4::from_cols(a * p, b * p, c * p, d * p))
    }

    pub fn normal(&self) -> glam::Vec3A {
        self.0.into()
    }
}

impl Quadric {
    /// Calculate error from Q and vertex, `v^T K_p v`
    pub fn quadric_error(&self, v: glam::Vec3A) -> f64 {
        let v: glam::Vec4 = (v, 1.0).into();
        let v: glam::DVec4 = v.into();
        v.dot(self.0 * v)
    }
}
impl FaceID {
    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> Plane {
        let [a, b, c] = mesh.triangle_from_face(&mesh.faces[self]);

        Plane::from_three_points(verts[a].into(), verts[b].into(), verts[c].into())
    }
}

impl VertID {
    /// Generate error matrix Q, the sum of Kp for all planes p around this vertex.
    /// TODO: Eventually we can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    #[allow(non_snake_case)]
    pub fn generate_error_matrix(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> Quadric {
        let mut Q = Quadric(glam::DMat4::ZERO);

        for &e in mesh.verts[self].outgoing_edges() {
            let f = mesh.edges[e].face;

            let plane = f.plane(mesh, verts);

            let Kp = plane.fundamental_error_quadric();

            Q.0 += Kp.0;

            if mesh.edges[e].twin.is_none() {
                // Boundary edge, add a plane that stops us moving away from the boundary

                // Edge Plane should have normal of Edge X plane
                let v = e.edge_vec(mesh, verts).unwrap();

                let boundary_norm = v.cross(plane.normal());

                let boundary_plane = Plane::from_normal_and_point(
                    boundary_norm,
                    verts[mesh.edges[e].vert_origin.0].into(),
                );
                // Multiply squared error by large factor, as changing the boundary adds a lot of visible error
                Q.0 += boundary_plane.fundamental_error_quadric().0 * 3000.0;
            }
        }
        Q
    }
}

impl EdgeID {
    pub fn orig_dest(&self, mesh: &WingedMesh) -> Result<(VertID, VertID)> {
        let e = mesh
            .edges
            .get(self)
            .ok_or(MeshError::InvalidEdge)
            .context("Failed to lookup origin/destination")?;
        let orig = e.vert_origin;
        let dest = mesh
            .edges
            .get(e.edge_left_cw)
            .ok_or(MeshError::InvalidCwEdge)
            .context("Failed to lookup origin/destination")?
            .vert_origin;
        Ok((orig, dest))
    }

    pub fn edge_vec(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> Result<glam::Vec3A> {
        let (src, dst) = self.orig_dest(mesh)?;
        let vd: glam::Vec3A = verts[dst.0].into();
        let vs: glam::Vec3A = verts[src.0].into();
        Ok(vd - vs)
    }

    /// Estimate the error introduced by collapsing this edge. Does not take into account penalties from flipping triangles
    pub fn edge_collapse_error(
        self,
        mesh: &WingedMesh,
        verts: &[glam::Vec4],
        quadric_errors: &[Quadric],
    ) -> Result<Option<f64>> {
        let (orig, dest) = self.orig_dest(mesh)?;

        if !orig.is_local_manifold(mesh) && !dest.is_group_embedded(mesh) {
            // If we change the boundary shape, we must move into an embedded position, or we risk separating the partition into two chunks
            // which will break partitioning
            return Ok(None);
        }

        if !orig.is_group_embedded(mesh) {
            // Cannot move a vertex unless it is in the center of a partition
            return Ok(None);
        }

        let q = Quadric(quadric_errors[orig.0].0 + quadric_errors[dest.0].0);
        // Collapsing this edge would move the origin to the destination, so we find the error of the origin at the merged point.

        // Test normals of triangles before and after the swap
        for &e in mesh
            .verts
            .get(orig)
            .ok_or(MeshError::InvalidVertex)
            .context("Failed to lookup vertex for finding plane normals")?
            .outgoing_edges()
        {
            if e == self {
                continue;
            }

            let f = mesh.edges[e].face;
            let [a, b, c] = mesh.triangle_from_face(&mesh.faces[f]);

            let (v_a, v_b, v_c, new_corner) = (
                verts[a].into(),
                verts[b].into(),
                verts[c].into(),
                verts[dest.0].into(),
            );
            let starting_plane = Plane::from_three_points(v_a, v_b, v_c);

            let end_plane = match orig.0 {
                i if i == a => Plane::from_three_points(new_corner, v_b, v_c),
                i if i == b => Plane::from_three_points(v_a, new_corner, v_c),
                i if i == c => Plane::from_three_points(v_a, v_b, new_corner),
                _ => unreachable!(),
            };

            if starting_plane.normal().dot(end_plane.normal()) < 0.0 {
                // Flipped triangle, give this an invalid weight - discard
                return Ok(None);
            }
        }

        Ok(Some(q.quadric_error(verts[orig.0].into())))
    }
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

    pub fn initialise_collapse_queues(
        &self,
        verts: &[glam::Vec4],
        quadrics: &[Quadric],
        queue_count: usize,
        queue_lookup: impl Fn(FaceID, &WingedMesh) -> usize,
    ) -> Vec<CollapseQueue> {
        println!("Generating Collapse Queue...");
        let mut pq = vec![
            CollapseQueue {
                queue: priority_queue::PriorityQueue::with_capacity(self.edges.len() / queue_count),
            };
            queue_count
        ];
		#[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.edges.len() as _);

        for (&eid, edge) in self.edges.iter() {
			#[cfg(feature = "progress")]
            bar.inc(1);
            if let Some(error) = eid.edge_collapse_error(&self, verts, &quadrics).unwrap() {
                pq[queue_lookup(edge.face, self)]
                    .queue
                    .push(eid, OrdF64(cmp::Reverse(error)));
            }
        }
		#[cfg(feature = "progress")]
        bar.finish_and_clear();

        pq
    }

    /// Returns estimate of error introduced by halving the number of triangles
    pub fn reduce(
        &mut self,
        verts: &[glam::Vec4],
        quadrics: &mut [Quadric],
        collapse_reqs: &[usize],
        queue_lookup: impl Fn(FaceID, &WingedMesh) -> usize,
    ) -> Result<f64> {
        let mut collapse_queue =
            self.initialise_collapse_queues(verts, quadrics, collapse_reqs.len(), &queue_lookup);

        assert_eq!(collapse_queue.len(), collapse_reqs.len());
        assert_eq!(verts.len(), quadrics.len());

        // Priority queue with every vertex and their errors.

        let mut new_error = 0.0;
        // let tris = self.face_count();
        // // Need to remove half the triangles - each reduction removes 2
        // let required_reductions = (tris / 4);
		#[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(collapse_reqs.iter().sum::<usize>() as _);
        #[cfg(feature = "progress")]
		bar.set_style(
            ProgressStyle::with_template("(Error:{msg}) {wide_bar} {pos}/{len} ({per_sec})")
                .unwrap(),
        );

        for qi in 0..collapse_queue.len() {
            #[cfg(test)]
            {
                self.assert_valid();
                for (&q, _e) in collapse_queue[qi].queue.iter() {
                    assert!(
                        self.edges.contains_key(q),
                        "Invalid edge in collapse queue {}",
                        qi
                    );
                    self.assert_edge_valid(q, "Invalid edge in collapse queue ");
                }
            }

            for _ in 0..collapse_reqs[qi] {
                let (eid, OrdF64(cmp::Reverse(err))) = match collapse_queue[qi].queue.pop() {
                    Some(err) => err,
                    None => {
                        // FIXME: how to handle early exits
						#[cfg(feature = "progress")]
                        bar.println("Exiting early from demeshing");
                        break;
                    }
                };

				#[cfg(feature = "progress")]
                bar.inc(1);
				#[cfg(feature = "progress")]
                bar.set_message(format!("{err:.3e}"));

                new_error += err;

                // Collapse edge, and update quadrics (update before collapsing, as vertex becomes invalid)

                let (orig, dest) = eid.orig_dest(&self)?;

                #[cfg(not(test))]
                {
                    if !self.verts.contains_key(orig) || !self.verts.contains_key(dest) {
						#[cfg(feature = "progress")]
                        bar.println("Warning - drawn 'valid' edge with no orig or dest");
                        continue;
                    }
                }

                #[cfg(test)]
                {
                    assert!(orig.is_group_embedded(&self));
                }

                quadrics[dest.0].0 += quadrics[orig.0].0;

                //TODO: When we collapse an edge, recalculate any effected edges.

                let mut effected_edges: Vec<_> = self.verts[orig].outgoing_edges().to_vec();
                effected_edges.extend(self.verts[dest].outgoing_edges());
                effected_edges.extend(self.verts[orig].incoming_edges());
                effected_edges.extend(self.verts[dest].incoming_edges());

                self.collapse_edge(eid);

                for eid in effected_edges {
                    if self.edges.contains_key(eid) {
                        let edge_queue = queue_lookup(self.edges[eid].face, &self);

                        if let Some(error) = eid.edge_collapse_error(&self, verts, &quadrics)? {
                            collapse_queue[edge_queue]
                                .queue
                                .push(eid, OrdF64(cmp::Reverse(error)));
                            continue;
                        }
                    }

                    // First check in our own queue, then check in all the others if we cannot find it
                    if collapse_queue[qi].queue.remove(&eid).is_none() {
                        for q in &mut collapse_queue {
                            if q.queue.remove(&eid).is_some() {
                                break;
                            }
                        }
                    }
                }
            }
        }
		#[cfg(feature = "progress")]
        bar.finish();
        Ok(new_error)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::mesh::winged_mesh::test::{TEST_MESH_MONK, TEST_MESH_PLANE, TEST_MESH_PLANE_LOW};

    use super::{
        super::winged_mesh::{test::TEST_MESH_HIGH, WingedMesh},
        Plane,
    };

    fn plane_distance(plane: &Plane, point: glam::Vec3A) -> f32 {
        point.dot(plane.0.into()) + plane.0.w
    }
    // Test that each face generates a valid plane
    #[test]
    pub fn test_planes() -> Result<(), Box<dyn Error>> {
        let (mesh, verts) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value
        let e = 0.000001;

        for (i, (&fid, f)) in mesh.faces.iter().enumerate() {
            let plane = fid.plane(&mesh, &verts);
            let n = plane.normal();
            assert!(
                ((n.x * n.x + n.y * n.y + n.z * n.z) - 1.0).abs() < e,
                "Plane {i} is not normalised"
            );

            for v in mesh.triangle_from_face(f) {
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
        let (mesh, verts) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value, only valid for errors within around 50 units.
        let e = 0.001;

        let random_points = [glam::Vec3A::X, glam::Vec3A::Y * 50.0];

        for (i, (&fid, f)) in mesh.faces.iter().enumerate() {
            let plane = fid.plane(&mesh, &verts);

            let mat = plane.fundamental_error_quadric();

            let cols = mat.0.to_cols_array_2d();

            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            for v in mesh.triangle_from_face(f) {
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
        let (mesh, verts) = WingedMesh::from_gltf(TEST_MESH_MONK);

        let e = 0.0000000001;

        for (vid, v) in mesh.verts.iter() {
            let q = vid.generate_error_matrix(&mesh, &verts);

            let cols = q.0.to_cols_array_2d();
            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            let v = verts[vid.0];
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
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_MONK);
        let mut quadrics = mesh.create_quadrics(&verts);

        for i in 0..4 {
            mesh.assert_valid();
            mesh.reduce(&verts, &mut quadrics, &[mesh.face_count() / 4], |_, _| 0)
                .unwrap();
        }

        mesh.assert_valid();

        Ok(())
    }
}
