use super::{
    vertex::{VertID, Vertex},
    winged_mesh::{Face, FaceID, WingedMesh},
};

/// The fundamental error quadric `K_p`, such that `v^T K_p v` = `sqr distance v <-> p`
/// Properties: Additive, Symmetric.
pub fn fundamental_error_quadric(p: glam::Vec4) -> glam::Mat4 {
    let (a, b, c, d) = p.into();

    // Do `p p^T`
    glam::Mat4::from_cols(a * p, b * p, c * p, d * p)
}

/// Calculate error from Q and vertex, `v^T K_p v`
pub fn quadric_error(mat: glam::Mat4, v: glam::Vec3A) -> f32 {
    let v: glam::Vec4 = (v, 1.0).into();
    v.dot(mat * v)
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
    #[allow(non_snake_case)]
    pub fn generate_error_matrix(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> glam::Mat4 {
        let mut Q = glam::Mat4::ZERO;
        for &e in self.outgoing_edges(mesh) {
            let f = mesh[e].face;

            let plane = f.plane(mesh, verts);

            let Kp = fundamental_error_quadric(plane);
            Q += Kp;
        }
        Q
    }
}

impl WingedMesh {
    pub fn reduce(&mut self) {}
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

            let mat = fundamental_error_quadric(plane);

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
                let sqr_p_dist = p_dist * p_dist;

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

        // These operations are not especially accurate, large epsilon value, only valid for errors within around 50 units.
        let e = 0.0001;

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
