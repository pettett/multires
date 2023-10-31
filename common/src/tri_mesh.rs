use glam::Vec3;
use gltf::mesh::util::ReadIndices;

pub struct TriMesh {
    pub verts: Box<[Vec3]>,
    pub indices: Box<[u32]>,
}
impl TriMesh {
    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> gltf::Result<Self> {
        let (doc, buffers, _) = gltf::import(path)?;

        let mesh = doc.meshes().next().unwrap();
        let p = mesh.primitives().next().unwrap();
        let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));

        let iter = reader.read_positions().unwrap();

        Ok(TriMesh {
            verts: iter.map(|v| Vec3::from_array(v)).collect(),
            indices: match reader.read_indices() {
                Some(ReadIndices::U16(iter)) => iter.map(|i| i as _).collect(),
                Some(ReadIndices::U32(iter)) => iter.collect(),
                _ => panic!("Unsupported index size"),
            },
        })
    }
}
