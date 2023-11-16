use std::cmp;

use glam::{Vec4, Vec4Swizzles};

use super::{
    vertex::{VertID, Vertex},
    winged_mesh::{EdgeID, Face, FaceID, WingedMesh},
};

/// Quadric type. Internally a DMat4, kept private to ensure only valid reduction operations can effect it.
pub struct Quadric(glam::DMat4);

pub fn tri_area(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> f32 {
    let ab = b - a;
    let ac = c - a;

    glam::Vec3A::cross(ab, ac).length() / 2.0
}

pub fn plane_from_three_points(a: glam::Vec3A, b: glam::Vec3A, c: glam::Vec3A) -> glam::Vec4 {
    let ab = b - a;
    let ac = c - a;

    let normal = glam::Vec3A::cross(ab, ac).normalize();

    let d = -a.dot(normal);

    let plane: glam::Vec4 = (normal, d).into();

    plane
}

/// The fundamental error quadric `K_p`, such that `v^T K_p v` = `sqr distance v <-> p`
/// Properties: Additive, Symmetric.
pub fn fundamental_error_quadric(p: glam::DVec4) -> Quadric {
    let (a, b, c, d) = p.into();

    // Do `p p^T`
    Quadric(glam::DMat4::from_cols(a * p, b * p, c * p, d * p))
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
    pub fn plane(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> glam::Vec4 {
        let [a, b, c] = mesh.triangle_from_face(&mesh.faces[self]);

        plane_from_three_points(verts[a].into(), verts[b].into(), verts[c].into())
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

            let Kp = fundamental_error_quadric(plane.into());
            Q.0 += Kp.0;
        }
        Q
    }
}

impl EdgeID {
    pub fn orig_dest(&self, mesh: &WingedMesh) -> Result<(VertID, VertID), ()> {
        let e = mesh.edges.get(self).ok_or(())?;
        let orig = e.vert_origin;
        let dest = mesh.edges.get(e.edge_left_cw).ok_or(())?.vert_origin;
        Ok((orig, dest))
    }

    /// Estimate the error introduced by collapsing this edge. Does not take into account penalties from flipping triangles
    pub fn edge_collapse_error(
        self,
        mesh: &WingedMesh,
        verts: &[glam::Vec4],
        quadric_errors: &[Quadric],
    ) -> Result<Option<f64>, ()> {
        let (orig, dest) = self.orig_dest(mesh)?;
        let q = Quadric(quadric_errors[orig.0].0 + quadric_errors[dest.0].0);
        // Collapsing this edge would move the origin to the destination, so we find the error of the origin at the merged point.

        if !orig.is_local_manifold(mesh, true) {
            // Need this to be in the center of the mesh, cannot reduce and change the boundary shape
            return Ok(None);
        }

        // Test normals of triangles before and after the swap
        for &e in mesh.verts.get(orig).ok_or(())?.outgoing_edges() {
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
            let starting_plane = plane_from_three_points(v_a, v_b, v_c);

            let end_plane = match orig.0 {
                i if i == a => plane_from_three_points(new_corner, v_b, v_c),
                i if i == b => plane_from_three_points(v_a, new_corner, v_c),
                i if i == c => plane_from_three_points(v_a, v_b, new_corner),
                _ => unreachable!(),
            };

            if starting_plane.xyz().dot(end_plane.xyz()) < 0.1 {
                // Flipped triangle, give this a very high weight
                return Ok(None);
            }
        }

        Ok(Some(q.quadric_error(verts[orig.0].into())))
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
            panic!("Cannot order invalid floats")
        }
    }
}

impl WingedMesh {
    pub fn create_quadrics(&self, verts: &[glam::Vec4]) -> Vec<Quadric> {
        let mut quadrics = Vec::with_capacity(verts.len());

        for (i, v) in self.verts.iter() {
            quadrics.push(i.generate_error_matrix(&self, verts))
        }

        quadrics
    }
    /// Returns estimate of error introduced by halving the number of triangles
    pub fn reduce(&mut self, verts: &[glam::Vec4], quadrics: &mut [Quadric]) -> Result<f64, ()> {
        // Priority queue with every vertex and their errors.

        let mut pq = priority_queue::PriorityQueue::with_capacity(self.edges.len());
        let mut new_error = 0.0;
        for &eid in self.edges.keys() {
            if let Some(error) = eid.edge_collapse_error(&self, verts, &quadrics)? {
                pq.push(eid, cmp::Reverse(OrdF64(error)));
                continue;
            }

            pq.remove(&eid);
        }

        let tris = self.face_count();
        // Need to remove half the triangles - each reduction removes 2
        let required_reductions = tris / 4;

        for i in 0..required_reductions {
            let (eid, cmp::Reverse(OrdF64(err))) = pq.pop().ok_or(())?;

            if i % 200 == 0 {
                println!("Selected edge {i} with error: {err}");
            }

            new_error += err;

            // Collapse edge, and update quadrics (update before collapsing, as vertex becomes invalid)

            let (orig, dest) = eid.orig_dest(&self)?;

            #[cfg(not(test))]
            {
                if !self.verts.contains_key(orig) || !self.verts.contains_key(dest) {
                    println!("Warning - drawn 'valid' edge with no orig or dest");
                    continue;
                }
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
                    if let Some(error) = eid.edge_collapse_error(&self, verts, &quadrics)? {
                        pq.push(eid, cmp::Reverse(OrdF64(error)));
                        continue;
                    }
                }

                pq.remove(&eid);
            }
        }

        Ok(new_error)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use crate::mesh::winged_mesh::test::{TEST_MESH_PLANE, TEST_MESH_PLANE_LOW};

    use super::super::{
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
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_PLANE);
        println!("Total {} tris", mesh.face_count());
        let mut quadrics = mesh.create_quadrics(&verts);
        mesh.assert_valid();
        mesh.reduce(&verts, &mut quadrics).unwrap();
        mesh.assert_valid();
        mesh.reduce(&verts, &mut quadrics).unwrap();
        mesh.assert_valid();
        mesh.reduce(&verts, &mut quadrics).unwrap();
        mesh.assert_valid();
        mesh.reduce(&verts, &mut quadrics).unwrap();
        mesh.assert_valid();

        Ok(())
    }
}
