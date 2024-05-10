use super::{edge::EdgeID, half_edge_mesh::HalfEdgeMesh, plane::Plane};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub cluster_idx: usize,
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FaceID(pub u32);

impl From<usize> for FaceID {
    fn from(value: usize) -> Self {
        Self(value as _)
    }
}
impl From<FaceID> for usize {
    fn from(value: FaceID) -> Self {
        value.0 as _
    }
}

impl Face {
    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(&self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Plane {
        let [a, b, c] = mesh.triangle_from_face(self);

        Plane::from_three_points(verts[a as usize], verts[b as usize], verts[c as usize])
    }
}

impl FaceID {
    pub fn center(self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> glam::Vec3A {
        let [a, b, c] = mesh.triangle_from_face(self.face(mesh));

        (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0
    }

    pub fn face(self, mesh: &HalfEdgeMesh) -> &Face {
        mesh.faces().get(self)
    }
    pub fn face_mut(self, mesh: &mut HalfEdgeMesh) -> &mut Face {
        mesh.faces_mut().get_mut(self)
    }
    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Plane {
        self.face(mesh).plane(mesh, verts)
    }
}
