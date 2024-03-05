use idmap::IntegerId;

use super::{edge::EdgeID, plane::Plane, winged_mesh::WingedMesh};

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub cluster_idx: usize,
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FaceID(pub u32);

impl Into<usize> for FaceID {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for FaceID {
    fn from(value: usize) -> Self {
        FaceID(value as _)
    }
}

impl IntegerId for FaceID {
    fn from_id(id: u64) -> Self {
        FaceID(id as _)
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
    pub fn plane(&self, mesh: &WingedMesh, verts: &[glam::Vec3A]) -> Plane {
        let [a, b, c] = mesh.triangle_from_face(self);

        Plane::from_three_points(verts[a as usize], verts[b as usize], verts[c as usize])
    }
}

impl FaceID {
    pub fn center(self, mesh: &WingedMesh, verts: &[glam::Vec3A]) -> glam::Vec3A {
        let [a, b, c] = mesh.triangle_from_face(&mesh.get_face(self));

        (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0
    }

    /// Generate plane from the 3 points a,b,c on this face.
    pub fn plane(self, mesh: &WingedMesh, verts: &[glam::Vec3A]) -> Plane {
        mesh.get_face(self).plane(mesh, verts)
    }
}
