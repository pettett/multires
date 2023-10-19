extern crate gltf;

use glam::Vec3;
use metis::{
    idx_t, real_t, rstatus_et_METIS_ERROR_INPUT, rstatus_et_METIS_ERROR_MEMORY,
    rstatus_et_METIS_OK, METIS_PartGraphKway, METIS_PartGraphRecursive, METIS_SetDefaultOptions,
    PartitioningConfig, PartitioningError, PartitioningMethod, METIS_NOPTIONS,
};
use std::{collections::HashMap, ptr::null_mut, time};

use gltf::mesh::util::ReadIndices;

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertID(usize);
impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0
    }
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeID(usize);
impl Into<usize> for EdgeID {
    fn into(self) -> usize {
        self.0
    }
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceID(usize);

impl Into<usize> for FaceID {
    fn into(self) -> usize {
        self.0
    }
}

#[derive(Default, Debug, Clone)]
pub struct HalfEdge {
    pub vert_origin: VertID,
    pub vert_destination: VertID,
    pub face: FaceID,
    /// Edge leading on from the dest vert
    pub edge_left_cw: EdgeID,
    /// Edge connecting into the origin vert
    pub edge_left_ccw: EdgeID,

    pub twin: Option<EdgeID>,
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub edge: Option<EdgeID>,
}

#[derive(Default, Debug, Clone)]
pub struct Face {
    pub edge: Option<EdgeID>,
}

pub struct EdgeIter<'a> {
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
pub struct WingedMesh {
    verts: Vec<Vertex>,
    faces: Vec<Face>,
    partitions: Vec<i32>,
    edges: Vec<HalfEdge>,
    edge_map: HashMap<(VertID, VertID), EdgeID>,
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
            partitions: vec![Default::default(); faces],
            edges: Default::default(),
            edge_map: Default::default(),
        }
    }

    pub fn from_gltf(
        path: impl AsRef<std::path::Path>,
    ) -> gltf::Result<(Self, Vec<[f32; 3]>, Vec<u32>)> {
        let (doc, buffers, _) = gltf::import(path)?;

        let mesh = doc.meshes().next().unwrap();
        let p = mesh.primitives().next().unwrap();
        let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));

        let iter = reader.read_positions().unwrap();
        let verts: Vec<[f32; 3]> = iter.collect();
        //.map(|v| Vec3::from_array(v))
        let indices: Vec<u32> = match reader.read_indices() {
            Some(ReadIndices::U16(iter)) => iter.map(|i| i as _).collect(),
            Some(ReadIndices::U32(iter)) => iter.collect(),
            _ => panic!("Unsupported index size"),
        };
        let mut mesh = WingedMesh::new(indices.len() / 3, verts.len());

        for i in 0..mesh.faces.len() {
            let a = indices[i * 3] as usize;
            let b = indices[i * 3 + 1] as usize;
            let c = indices[i * 3 + 2] as usize;

            mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
        }

        Ok((mesh, verts, indices))
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        self.edge_map.get(&(a, b)).copied()
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

        self.edge_map.insert((orig, dest), EdgeID(self.edges.len()));
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

    pub fn faces(&self) -> &[Face] {
        self.faces.as_ref()
    }

    pub fn partition(
        &self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<Vec<idx_t>, PartitioningError> {
        let mut adjacency = Vec::new(); // adjncy
        let mut adjacency_idx = Vec::new(); // xadj

        for v in self.faces.iter() {
            adjacency_idx.push(adjacency.len() as idx_t);
            for e in self.iter_edge(v.edge.unwrap()) {
                if let Some(twin) = self[e].twin {
                    adjacency.push(self[twin].face.0 as i32);
                }
            }
        }

        let adjacency_weight = vec![1; adjacency.len()]; // adjcwgt

        adjacency_idx.push(adjacency.len() as idx_t);

        let weights = Vec::new();
        // if let Some(cw) = &config.weights {
        //     if cw.len() != partitions as usize {
        //         return Err(PartitioningError::WeightsMismatch);
        //     }
        //     weights.reserve(partitions as usize);
        //     for &w in cw.iter() {
        //         weights.push(w as real_t);
        //     }
        // }

        config.partition_from_adj(
            partitions,
            self.faces.len(),
            weights,
            adjacency,
            adjacency_idx,
            adjacency_weight,
        )
    }
}
