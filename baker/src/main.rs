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

#[derive(Debug)]
struct WingedMesh {
    verts: Vec<Vertex>,
    faces: Vec<Face>,
    edges: Vec<HalfEdge>,
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

    pub fn vert(&self, v: VertID) -> &Vertex {
        return &self.verts[v.0];
    }
    pub fn vert_mut(&mut self, v: VertID) -> &mut Vertex {
        return &mut self.verts[v.0];
    }
    pub fn face(&self, f: FaceID) -> &Face {
        return &self.faces[f.0];
    }
    pub fn face_mut(&mut self, f: FaceID) -> &mut Face {
        return &mut self.faces[f.0];
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

        if self.vert(a).edge == None {
            self.vert_mut(a).edge = Some(iea);
        }

        if self.vert(b).edge == None {
            self.vert_mut(b).edge = Some(ieb);
        }

        if self.vert(c).edge == None {
            self.vert_mut(c).edge = Some(iec);
        }

        self.face_mut(f).edge = Some(iea);
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

                println!("{a} {b} {c}");

                mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
            }

            for f in &mesh.faces {
                let e = f.edge.unwrap();
                if let Some(te) = mesh.edges[e.0].twin {
                    println!(
                        "Face {:?} is connected to {:?}",
                        mesh.edges[e.0].face, mesh.edges[te.0].face
                    )
                }

                let e = mesh.edges[e.0].edge_left_cw;
                if let Some(te) = mesh.edges[e.0].twin {
                    println!(
                        "Face {:?} is connected to {:?}",
                        mesh.edges[e.0].face, mesh.edges[te.0].face
                    )
                }

                let e = mesh.edges[e.0].edge_left_cw;
                if let Some(te) = mesh.edges[e.0].twin {
                    println!(
                        "Face {:?} is connected to {:?}",
                        mesh.edges[e.0].face, mesh.edges[te.0].face
                    )
                }
            }
        }
    }

    Ok(())
}
