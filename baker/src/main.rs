extern crate gltf;

use gltf::{mesh::util::ReadIndices, Gltf};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct VertID(usize);
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct FaceID(usize);
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct EdgeID(usize);

#[derive(Default, Debug, Clone)]
struct HalfEdge {
    vert_origin: VertID,
    vert_destination: VertID,
    face: FaceID,
    /// Edge leading on from the dest vert
    edge_left_cw: EdgeID,
    /// Edge connecting into the origin vert
    edge_left_ccw: EdgeID,

    twin: Option<EdgeID>,
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    edge: Option<EdgeID>,
}

#[derive(Default, Debug, Clone)]
struct Face {
    edge: Option<EdgeID>,
}

struct EdgeIter<'a> {
    mesh: &'a WingedMesh,
    start: EdgeID,
    current: Option<EdgeID>,
}
impl<'a> Iterator for EdgeIter<'a> {
    type Item = EdgeID;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;

        if let Some(curr) = current {
            self.current = Some(self.mesh.edges[curr.0].edge_left_cw);
            if self.current == Some(self.start) {
                self.current = None;
            }
        }

        current
    }
}

#[derive(Debug)]
struct WingedMesh {
    verts: Vec<Vertex>,
    faces: Vec<Face>,
    edges: Vec<HalfEdge>,
}

impl std::ops::Index<VertID> for WingedMesh {
    type Output = Vertex;

    fn index(&self, index: VertID) -> &Self::Output {
        &self.verts[index.0]
    }
}
impl std::ops::IndexMut<VertID> for WingedMesh {
    fn index_mut(&mut self, index: VertID) -> &mut Self::Output {
        &mut self.verts[index.0]
    }
}

impl std::ops::Index<EdgeID> for WingedMesh {
    type Output = HalfEdge;

    fn index(&self, index: EdgeID) -> &Self::Output {
        &self.edges[index.0]
    }
}
impl std::ops::IndexMut<EdgeID> for WingedMesh {
    fn index_mut(&mut self, index: EdgeID) -> &mut Self::Output {
        &mut self.edges[index.0]
    }
}
impl std::ops::Index<FaceID> for WingedMesh {
    type Output = Face;

    fn index(&self, index: FaceID) -> &Self::Output {
        &self.faces[index.0]
    }
}
impl std::ops::IndexMut<FaceID> for WingedMesh {
    fn index_mut(&mut self, index: FaceID) -> &mut Self::Output {
        &mut self.faces[index.0]
    }
}

impl WingedMesh {
    pub fn new(faces: usize, verts: usize) -> Self {
        Self {
            verts: vec![Default::default(); verts],
            faces: vec![Default::default(); faces],
            edges: Default::default(),
        }
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        for (i, e) in self.edges.iter().enumerate() {
            if e.vert_origin == a && e.vert_destination == b {
                return Some(EdgeID(i));
            }
        }
        None
    }

    pub fn iter_edge(&self, e: EdgeID) -> EdgeIter {
        EdgeIter {
            mesh: self,
            start: e,
            current: Some(e),
        }
    }

    fn add_half_edge(&mut self, orig: VertID, dest: VertID, face: FaceID, cw: EdgeID, ccw: EdgeID) {
        let twin = self.find_edge(dest, orig);
        let e = HalfEdge {
            vert_origin: orig,
            vert_destination: dest,
            face,
            edge_left_cw: cw,
            edge_left_ccw: ccw,
            twin,
        };

        if let Some(te) = twin {
            self.edges[te.0].twin = Some(EdgeID(self.edges.len()));
        }

        self.edges.push(e);
    }

    pub fn add_tri(&mut self, f: FaceID, a: VertID, b: VertID, c: VertID) {
        let iea = EdgeID(self.edges.len());
        let ieb = EdgeID(self.edges.len() + 1);
        let iec = EdgeID(self.edges.len() + 2);

        self.add_half_edge(a, b, f, ieb, iec);
        self.add_half_edge(b, c, f, iec, iea);
        self.add_half_edge(c, a, f, iea, ieb);

        if self[a].edge == None {
            self[a].edge = Some(iea);
        }

        if self[b].edge == None {
            self[b].edge = Some(ieb);
        }

        if self[c].edge == None {
            self[c].edge = Some(iec);
        }

        self[f].edge = Some(iea);
    }
}

fn main() -> gltf::Result<()> {
    let (doc, buffers, _) = gltf::import("../assets/cube.glb")?;

    for mesh in doc.meshes() {
        for p in mesh.primitives() {
            let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));

            let iter = reader.read_positions().unwrap();
            let verts: Vec<[f32; 3]> = iter.collect();

            let indices: Vec<u16> = match reader.read_indices() {
                Some(ReadIndices::U16(iter)) => iter.collect(),
                _ => panic!("Unsupported index size"),
            };
            let mut mesh = WingedMesh::new(indices.len() / 3, verts.len());

            for i in 0..mesh.faces.len() {
                let a = indices[i * 3] as usize;
                let b = indices[i * 3 + 1] as usize;
                let c = indices[i * 3 + 2] as usize;

                println!("Face {i}: {a} {b} {c}");

                mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
            }

            for f in &mesh.faces {
                let e = f.edge.unwrap();

                for e in mesh.iter_edge(e) {
                    if let Some(te) = mesh[e].twin {
                        println!(
                            "Face {:?} is connected to {:?}",
                            mesh[e].face.0, mesh[te].face.0
                        )
                    }
                }
            }
        }
    }

    Ok(())
}
