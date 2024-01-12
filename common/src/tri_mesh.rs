use glam::Vec4;
use gltf::mesh::util::ReadIndices;

pub struct TriMesh {
    pub verts: Box<[Vec4]>,
    pub normals: Box<[Vec4]>,
    pub indices: Box<[u32]>,
}
impl TriMesh {
    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> gltf::Result<Self> {
        let (doc, buffers, _) = gltf::import(path)?;

        let mesh = doc.meshes().next().unwrap();
        let p = mesh.primitives().next().unwrap();

        let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));
        let vert_iter = reader.read_positions().unwrap();
        let norm_iter = reader.read_normals().unwrap();

        Ok(TriMesh {
            verts: vert_iter.map(|[x, y, z]| Vec4::new(x, y, z, 1.0)).collect(),
            normals: norm_iter.map(|[x, y, z]| Vec4::new(x, y, z, 1.0)).collect(),
            indices: match reader.read_indices() {
                Some(ReadIndices::U16(iter)) => iter.map(|i| i as _).collect(),
                Some(ReadIndices::U32(iter)) => iter.collect(),
                _ => panic!("Unsupported index size"),
            },
        })
    }
}
