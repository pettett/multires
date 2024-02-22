use idmap::IntegerId;

use super::{edge::EdgeID, plane::Plane, winged_mesh::WingedMesh};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub cluster_idx: usize,
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FaceID(pub usize);

impl Into<usize> for FaceID {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for FaceID {
    fn from(value: usize) -> Self {
        FaceID(value)
    }
}

impl IntegerId for FaceID {
    fn from_id(id: u64) -> Self {
        FaceID(id as usize)
    }

    fn id(&self) -> u64 {
        self.0 as u64
    }

    fn id32(&self) -> u32 {
        self.0 as u32
    }
}

impl Face {
    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(&self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> Plane {
        let [a, b, c] = mesh.triangle_from_face(self);

        Plane::from_three_points(verts[a].into(), verts[b].into(), verts[c].into())
    }
}

impl FaceID {
    pub fn center(self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> glam::Vec4 {
        let [a, b, c] = mesh.triangle_from_face(&mesh.get_face(self));

        (verts[a] + verts[b] + verts[c]) / 3.0
    }

    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(self, mesh: &WingedMesh, verts: &[glam::Vec4]) -> Plane {
        mesh.get_face(self).plane(mesh, verts)
    }
}
