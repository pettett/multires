use glam::Vec3A;
use gltf::mesh::util::ReadIndices;

pub struct TriMesh {
    pub verts: Box<[Vec3A]>,
    pub normals: Box<[Vec3A]>,
    pub indices: Box<[u32]>,
}
impl TriMesh {
    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> gltf::Result<Self> {
        let (doc, buffers, _) = gltf::import(path)?;

        let mesh = doc
            .meshes()
            .next()
            .expect("GLTF must contain at least one mesh");
        let p = mesh
            .primitives()
            .next()
            .expect("Mesh must contain at least one primitive");

        let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));
        let vert_iter = reader
            .read_positions()
            .expect("Mesh must contain positions");
        let norm_iter = reader.read_normals().expect("Mesh must contain normals");

        Ok(TriMesh {
            verts: vert_iter.map(|[x, y, z]| Vec3A::new(x, y, z)).collect(),
            normals: norm_iter.map(|[x, y, z]| Vec3A::new(x, y, z)).collect(),
            indices: match reader.read_indices() {
                Some(ReadIndices::U16(iter)) => iter.map(|i| i as _).collect(),
                Some(ReadIndices::U32(iter)) => iter.collect(),
                _ => panic!("Unsupported index size"),
            },
        })
    }
}
