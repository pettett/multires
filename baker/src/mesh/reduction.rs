use std::cmp;

use glam::Vec4;

use super::{
    vertex::{VertID, Vertex},
    winged_mesh::{EdgeID, Face, FaceID, WingedMesh},
};

/// The fundamental error quadric `K_p`, such that `v^T K_p v` = `sqr distance v <-> p`
/// Properties: Additive, Symmetric.
pub fn fundamental_error_quadric(p: glam::DVec4) -> glam::DMat4 {
    let (a, b, c, d) = p.into();

    // Do `p p^T`
    glam::DMat4::from_cols(a * p, b * p, c * p, d * p)
}

/// Calculate error from Q and vertex, `v^T K_p v`
pub fn quadric_error(q: glam::DMat4, v: glam::Vec3A) -> f64 {
    let v: glam::Vec4 = (v, 1.0).into();
    let v: glam::DVec4 = v.into();
    v.dot(q * v)
}

impl FaceID {
    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> glam::Vec4 {
        let [a, b, c] = mesh.triangle_from_face(&mesh.faces[self]);

        let a: glam::Vec3A = verts[a].into();
        let b: glam::Vec3A = verts[b].into();
        let c: glam::Vec3A = verts[c].into();

        let ab = b - a;
        let ac = c - a;

        let normal = glam::Vec3A::cross(ab, ac).normalize();

        let d = -a.dot(normal);

        let plane: glam::Vec4 = (normal, d).into();

        plane
    }
}

impl VertID {
    /// Generate error matrix Q, the sum of Kp for all planes p around this vertex.
    /// TODO: Eventually we can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    #[allow(non_snake_case)]
    pub fn generate_error_matrix(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> glam::DMat4 {
        let mut Q = glam::DMat4::ZERO;

        for &e in self.outgoing_edges(mesh) {
            let f = mesh[e].face;

            let plane = f.plane(mesh, verts);

            let Kp = fundamental_error_quadric(plane.into());
            Q += Kp;
        }
        Q
    }
}

impl EdgeID {
    pub fn orig_dest(self, mesh: &WingedMesh) -> (VertID, VertID) {
        let orig = mesh[self].vert_origin;
        let dest = mesh[mesh[self].edge_left_cw].vert_origin;
        (orig, dest)
    }

    /// Estimate the error introduced by collapsing this edge. Does not take into account penalties from flipping triangles
    pub fn edge_collapse_error(
        self,
        mesh: &WingedMesh,
        verts: &[glam::Vec4],
        quadric_errors: &[glam::DMat4],
    ) -> f64 {
        let (orig, dest) = self.orig_dest(mesh);
        let q = quadric_errors[orig.0] + quadric_errors[dest.0];
        // Collapsing this edge would move the origin to the destination, so we find the error of the origin at the merged point.
        quadric_error(q, verts[orig.0].into())
    }
}
#[derive(PartialOrd, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if let Some(ordering) = self.partial_cmp(other) {
            ordering
        } else {
            // Choose what to do with NaNs, for example:
            panic!("Cannot order")
        }
    }
}

impl WingedMesh {
    pub fn create_quadrics(&self, verts: &[glam::Vec4]) -> Vec<glam::DMat4> {
        let mut quadrics = Vec::with_capacity(verts.len());

        for (i, v) in self.verts.iter().enumerate() {
            quadrics.push(VertID(i).generate_error_matrix(&self, verts))
        }

        quadrics
    }
    /// Returns estimate of error introduced by halving the number of triangles
    pub fn reduce(&mut self, verts: &[glam::Vec4], quadrics: &mut [glam::DMat4]) -> f64 {
        // Priority queue with every vertex and their errors.

        let mut pq = priority_queue::PriorityQueue::with_capacity(self.edges.len());
        let mut new_error = 0.0;
        for (i, e) in self.edges.iter().enumerate() {
            if e.valid {
                let eid = EdgeID(i);
                pq.push(
                    eid,
                    cmp::Reverse(OrdF64(eid.edge_collapse_error(&self, verts, &quadrics))),
                );
            }
        }

        let tris = self.face_count();
        // Need to remove half the triangles - each reduction removes 2
        let required_reductions = tris / 4;

        for i in 0..required_reductions {
            let (eid, cmp::Reverse(OrdF64(err))) = pq.pop().unwrap();

            println!("Selected edge with error: {err}");

            new_error += err;

            // Collapse edge, and update quadrics (update before collapsing, as vertex becomes invalid)

            let (orig, dest) = eid.orig_dest(&self);
            quadrics[dest.0] += quadrics[orig.0];

            //TODO: When we collapse an edge, recalculate any effected edges.

            let mut effected_edges: Vec<_> = orig.outgoing_edges(self).to_vec();
            effected_edges.extend(orig.incoming_edges(self));
            effected_edges.extend(dest.outgoing_edges(self));
            effected_edges.extend(dest.incoming_edges(self));

            self.collapse_edge(eid);

            for e in effected_edges {
                if self.edges[e.0].valid {
                    pq.push(
                        e,
                        cmp::Reverse(OrdF64(e.edge_collapse_error(&self, verts, &quadrics))),
                    );
                } else {
                    pq.remove(&e);
                }
            }
        }

        new_error
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::super::{
        reduction::quadric_error,
        vertex::VertID,
        winged_mesh::{test::TEST_MESH_HIGH, WingedMesh},
    };

    use super::fundamental_error_quadric;

    fn plane_distance(plane: glam::Vec4, point: glam::Vec3A) -> f32 {
        point.dot(plane.into()) + plane.w
    }
    // Test that each face generates a valid plane
    #[test]
    pub fn test_planes() -> Result<(), Box<dyn Error>> {
        let (mesh, verts) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        // These operations are not especially accurate, large epsilon value
        let e = 0.000001;

        for (i, (&fid, f)) in mesh.faces.iter().enumerate() {
            let plane = fid.plane(&mesh, &verts);

            assert!(
                ((plane.x * plane.x + plane.y * plane.y + plane.z * plane.z) - 1.0).abs() < e,
                "Plane {i} is not normalised"
            );

            for v in mesh.triangle_from_face(f) {
                let v_dist = plane_distance(plane, verts[v].into()).abs();
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

            let mat = fundamental_error_quadric(plane.into());

            let cols = mat.to_cols_array_2d();

            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            for v in mesh.triangle_from_face(f) {
                let v = verts[v];
                // v^t * K_p * v
                let q_error = quadric_error(mat, v.into());
                assert!(
                    q_error < e,
                    "Plane invalid at index {v}, tri {i}, value {q_error}",
                );
            }

            for &v in &random_points {
                let q_error = quadric_error(mat, v);
                let p_dist = plane_distance(plane, v);
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
        let (mesh, verts) = WingedMesh::from_gltf(TEST_MESH_HIGH);

        let e = 0.0000000001;

        for (i, v) in mesh.verts.iter().enumerate() {
            let vid = VertID(i);
            let Q = vid.generate_error_matrix(&mesh, &verts);

            let cols = Q.to_cols_array_2d();
            // Assert symmetry
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(cols[i][j], cols[j][i]);
                }
            }

            let v = verts[i];
            // v^t * K_p * v
            let q_error = quadric_error(Q, v.into());
            assert!(
                q_error < e,
                "Plane invalid at index {v}, tri {i}, value {q_error}",
            );
        }

        Ok(())
    }
}
