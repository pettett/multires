extern crate gltf;

use idmap::IntegerId;
use metis::{idx_t, PartitioningConfig, PartitioningError};
use petgraph::data::{Build, FromElements};
use std::collections::{HashMap, HashSet};

use gltf::mesh::util::ReadIndices;

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertID(pub usize);
impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0
    }
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeID(pub usize);
impl Into<usize> for EdgeID {
    fn into(self) -> usize {
        self.0
    }
}
#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceID(pub usize);
impl Into<usize> for FaceID {
    fn into(self) -> usize {
        self.0 as usize
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

#[derive(Default, Debug, Clone)]
pub struct HalfEdge {
    pub vert_origin: VertID,
    // This is not actually needed, as the destination is the origin of the cw edge
    //pub vert_destination: VertID,
    pub face: FaceID,
    /// Edge leading on from the dest vert
    pub edge_left_cw: EdgeID,
    /// Edge connecting into the origin vert
    pub edge_left_ccw: EdgeID,

    pub twin: Option<EdgeID>,
    pub valid: bool,
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    /// Edge with vert_source = this id
    pub edge: Option<EdgeID>,
}

#[derive(Default, Debug, Clone)]
pub struct Face {
    pub edge: EdgeID,
    pub part: idx_t,
    pub group: idx_t,
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

pub struct AllEdgeIter {
    edges: Vec<HalfEdge>,
    current: Option<usize>,
}
impl Iterator for AllEdgeIter {
    type Item = EdgeID;

    fn next(&mut self) -> Option<Self::Item> {
        let last = self.current;

        if let Some(current) = &mut self.current {
            loop {
                *current += 1;

                if *current >= self.edges.len() {
                    self.current = None;
                    break;
                }

                if !self.edges[*current].valid {
                    continue;
                } else {
                    break;
                }
            }
        }

        last.map(|l| EdgeID(l))
    }
}

#[derive(Debug, Clone)]
pub struct WingedMesh {
    partition_count: usize,
    verts: Vec<Vertex>,
    faces: idmap::DirectIdMap<FaceID, Face>,
    // partitions: Vec<i32>,
    edges: Vec<HalfEdge>,
    edge_map: HashMap<VertID, Vec<EdgeID>>,
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
        assert!(
            self.edges[index.0].valid,
            "Attempted to index invalid edge {:?}",
            self.edges[index.0]
        );
        &self.edges[index.0]
    }
}

impl std::ops::IndexMut<EdgeID> for WingedMesh {
    fn index_mut(&mut self, index: EdgeID) -> &mut Self::Output {
        assert!(
            self.edges[index.0].valid,
            "Attempted to index invalid edge {:?}",
            self.edges[index.0]
        );
        &mut self.edges[index.0]
    }
}

impl WingedMesh {
    pub fn new(faces: usize, verts: usize) -> Self {
        Self {
            partition_count: 0,
            verts: vec![Default::default(); verts],
            //faces: vec![Default::default(); faces],
            faces: idmap::DirectIdMap::with_capacity_direct(faces),
            // partitions: vec![Default::default(); faces],
            edges: Default::default(),
            edge_map: Default::default(),
        }
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn edge_count(&self) -> usize {
        let mut c = 0;

        for f in &self.edges {
            if f.valid {
                c += 1
            }
        }

        c
    }
    pub fn iter_edges(&self) -> AllEdgeIter {
        AllEdgeIter {
            edges: self.edges.clone(),
            current: Some(0),
        }
    }
    pub fn incoming_edges(&self, vid: VertID) -> Vec<EdgeID> {
        self.edges
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if e.valid && self[e.edge_left_cw].vert_origin == vid {
                    Some(EdgeID(i))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn outgoing_edges(&self, vid: VertID) -> &[EdgeID] {
        const EMPTY: &[EdgeID] = &[];
        self.edge_map.get(&vid).map(|v| &v[..]).unwrap_or(EMPTY)
    }

    pub fn assert_valid(&self) {
        for (i, f) in self.faces.iter() {
            let edges: Vec<HalfEdge> = self
                .iter_edge_loop(f.edge)
                .map(|e| self[e].clone())
                .collect();

            assert_eq!(edges.len(), 3);
            //assert_eq!(edges[0].vert_destination, edges[1].vert_origin);
            //assert_eq!(edges[1].vert_destination, edges[2].vert_origin);
            //assert_eq!(edges[2].vert_destination, edges[0].vert_origin);
        }

        for i in 0..self.verts.len() {
            self.assert_vertex_valid(i);
        }

        for (k, v) in &self.edge_map {
            for e in v {
                assert!(
                    self.edges[e.0].valid,
                    "Edge valid error - Edge map is invalidated"
                );
                assert!(
                    self.edges[e.0].vert_origin == *k,
                    "Source error - Edge map is invalidated"
                );
            }
        }
    }
    pub fn assert_vertex_valid(&self, i: usize) {
        let v = &self.verts[i];
        if let Some(e) = v.edge {
            assert!(
                self.edges[e.0].valid,
                "Invalid vertex edge reference- Mesh made invalid on V{i}: {v:?}"
            );
            assert!(
                self.edges[e.0].vert_origin.0 == i,
                "Invalid vertex edge source loop - Mesh made invalid on V{i}: {v:?}"
            );
        }
    }

    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn vertex_has_complete_fan(&self, vid: VertID) -> bool {
        let Some(eid_first) = self[vid].edge else {
            return false;
        };

        let mut eid = eid_first;
        loop {
            // attempt to move around the fan, by moving to our twin edge and going clockwise
            let Some(twin) = self[eid].twin else {
                return false;
            };

            eid = self[twin].edge_left_cw;

            if eid == eid_first {
                return true;
            }
        }
    }

    /// Collapse a triangle, removing it from the graph, and pulling the two triangles on non-eid edges together
    /// 	A
    ///  	| 1
    ///  	0  D
    ///   	| 2
    /// 	B
    fn collapse_tri(&mut self, eid: EdgeID) {
        let tri = self.iter_edge_loop(eid).collect::<Vec<_>>();
        //println!("Collapsing triangle {tri:?}");
        let e = self[eid].clone();

        self.faces.remove(e.face);

        assert_eq!(tri.len(), 3);
        assert_eq!(tri[0], eid);

        // we are pinching edge to nothing, so make the other two edges twins
        if let Some(t) = self[tri[1]].twin {
            self[t].twin = self[tri[2]].twin;
        }
        if let Some(t) = self[tri[2]].twin {
            //TODO: helper function for moving origin

            self.edge_map.entry(self[t].vert_origin).and_modify(|xs| {
                let index = xs.iter().position(|x| *x == t).unwrap();
                xs.remove(index);
            });

            self[t].vert_origin = self[tri[1]].vert_origin;

            self.edge_map.entry(self[t].vert_origin).and_modify(|xs| {
                xs.push(t);
            });

            self[t].twin = self[tri[1]].twin;
        }

        for e in &tri {
            self.edge_map.entry(self[*e].vert_origin).and_modify(|xs| {
                let index = xs.iter().position(|x| *x == *e).unwrap();
                xs.remove(index);
            });

            self[*e].valid = false;
        }

        // TODO: inefficient
        for i in 0..3 {
            // Make sure vertexes are not referencing this triangle
            let v = self.edges[tri[i].0].vert_origin;

            // TODO: smarter selection of verts that require updating
            if self[v].edge.is_some() {
                self[v].edge = self.outgoing_edges(v).get(0).copied();

                //println!("Updating {v:?}");

                self.assert_vertex_valid(v.0);
            }
        }
        //self.assert_valid();
    }

    /// Collapse an edge so it no longer exists, the source vertex is no longer referenced,
    /// 	A
    ///   /	^ \
    ///  C	|  D
    ///   \	| /
    /// 	B
    pub fn collapse_edge(&mut self, eid: EdgeID) {
        //    println!("Collapsing edge {eid:?}");

        //self.assert_valid();

        assert!(self[eid].valid);
        let edge = self[eid].clone();

        let vb = edge.vert_origin;
        let va = self.edges[edge.edge_left_cw.0].vert_origin;

        self.collapse_tri(eid);

        if let Some(e0t) = edge.twin {
            self.collapse_tri(e0t);
        };

        self.verts[vb.0].edge = None;

        // Remove `vert_origin`
        self.edge_map.remove(&vb);

        let va_outgoing = self.edge_map.entry(va).or_default();

        // Main issue is a situation where triangles do not fan around the cake in both directions
        // This will collapse an edge to have dest and source in same position

        for i in 0..self.edges.len() {
            // Dont fix invalid edges
            // TODO: These shouldn't appear in the iterator
            if !self.edges[i].valid {
                continue;
            }

            if self.edges[i].vert_origin == vb {
                // TODO: leaves dictionary incorrect
                //self.edge_map
                //    .remove(&(other_edge.vert_origin, other_edge.vert_destination));

                self.edges[i].vert_origin = va;

                va_outgoing.push(EdgeID(i));
                //self.edge_map.insert(
                //    (other_edge.vert_origin, other_edge.vert_destination),
                //    EdgeID(i),
                //);
            }
        }

        //self.assert_valid();
    }

    pub fn triangle_from_face(&self, face: &Face) -> [u32; 3] {
        let verts: Vec<usize> = self
            .iter_edge_loop(face.edge)
            .map(|e| self[e].vert_origin.into())
            .collect();

        [verts[0] as _, verts[1] as _, verts[2] as _]
    }

    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> gltf::Result<(Self, Vec<[f32; 3]>)> {
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
        let face_count = indices.len() / 3;
        let mut mesh = WingedMesh::new(face_count, verts.len());

        for i in 0..face_count {
            let a = indices[i * 3] as usize;
            let b = indices[i * 3 + 1] as usize;
            let c = indices[i * 3 + 2] as usize;

            mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
        }

        Ok((mesh, verts))
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        self.edge_map
            .get(&a)
            .map(|v| {
                v.iter()
                    .filter(|p| self[self[**p].edge_left_cw].vert_origin == b)
                    .next()
                    .copied()
            })
            .flatten()
    }

    pub fn iter_edge_loop(&self, e: EdgeID) -> EdgeIter {
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
            //vert_destination: dest,
            face,
            edge_left_cw: cw,
            edge_left_ccw: ccw,
            twin,
            valid: true,
        };

        if let Some(te) = twin {
            self.edges[te.0].twin = Some(EdgeID(self.edges.len()));
        }

        self.edge_map
            .entry(orig)
            .or_default()
            .push(EdgeID(self.edges.len()));

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

        self.faces.insert(
            f,
            Face {
                edge: iea,
                part: -1,
                group: -1,
            },
        );
    }

    pub fn faces(&self) -> &idmap::DirectIdMap<FaceID, Face> {
        &self.faces
    }

    pub fn apply_partition(
        &mut self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<(), PartitioningError> {
        let part = self.partition(config, partitions)?;

        for (i, f) in self.faces.values_mut().enumerate() {
            // Some faces will have already been removed
            f.part = part[i];
        }

        self.partition_count = *part.iter().max().unwrap() as usize + 1;

        Ok(())
    }

    pub fn get_partition(&self) -> Vec<idx_t> {
        self.faces().values().map(|f| f.part).collect()
    }
    pub fn get_group(&self) -> Vec<idx_t> {
        self.faces().values().map(|f| f.group).collect()
    }

    pub fn partition(
        &self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<Vec<idx_t>, PartitioningError> {
        println!("Partitioning into {partitions} partitions");

        let mut adjacency = Vec::new(); // adjncy
        let mut adjacency_idx = Vec::new(); // xadj

        for f in self.faces.values() {
            // Some faces will have already been removed

            adjacency_idx.push(adjacency.len() as idx_t);
            for e in self.iter_edge_loop(f.edge) {
                if let Some(twin) = self[e].twin {
                    adjacency.push(self[twin].face.0 as i32);
                }
            }
        }

        let adjacency_weight = vec![1; adjacency.len()]; // adjcwgt

        adjacency_idx.push(adjacency.len() as idx_t);

        let weights = Vec::new();

        config.partition_from_adj(
            partitions,
            self.faces.len(),
            weights,
            adjacency,
            adjacency_idx,
            adjacency_weight,
        )
    }

    pub fn group(&mut self, config: &PartitioningConfig) -> Result<(), PartitioningError> {
        let group_count = self.partition_count / 4;
        println!("Partitioning into {group_count} groups");

        let mut graph = petgraph::Graph::<i32, i32>::new();

        let parts: HashSet<_> = (0..self.partition_count).collect();

        let nodes: HashMap<_, _> = parts
            .iter()
            .map(|i| {
                let n = graph.add_node(1);
                (n.index() as i32, n)
            })
            .collect();

        for (i, face) in self.faces().iter() {
            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self[e].twin {
                    let other_face = &self.faces[self[twin].face];

                    graph.update_edge(nodes[&face.part], nodes[&other_face.part], 1);
                }
            }
        }

        println!("Partitioning the partition!");
        let groups = config.partition_from_graph(group_count as u32, &graph)?;

        for (i, f) in self.faces.values_mut().enumerate() {
            // Some faces will have already been removed
            f.group = groups[f.part as usize];
        }

        Ok(())
    }

    /// Within each group, split triangles into two completely new partitions, so as not to preserve any old seams between ancient partitions
    /// Ensures the data structure is seamless with changing seams! Yippee!
    pub fn partition_within_groups(
        &mut self,
        config: &PartitioningConfig,
    ) -> Result<(), PartitioningError> {
        let group_count = self.partition_count / 4;
        println!("Partitioning {group_count} groups into sub-partitions");

        let mut graphs: Vec<_> = (0..group_count)
            .map(|i| petgraph::Graph::<i32, i32>::new())
            .collect();

        // Stores vecs of face IDs, which should be associated with data in graphs
        let mut ids: Vec<_> = (0..group_count).map(|i| Vec::new()).collect();
        let mut faces: Vec<_> = (0..group_count).map(|i| HashMap::new()).collect();

        // Give every triangle a node
        for (i, face) in self.faces().iter() {
            let g = face.group as usize;
            let n = graphs[g].add_node(1);

            // indexes should correspond
            assert_eq!(n.index(), ids[g].len());

            ids[g].push(*i);
            faces[g].insert(*i, n);
        }
        // Apply links between nodes
        for (i, face) in self.faces().iter() {
            let g = face.group as usize;

            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self[e].twin {
                    let other_face = self[twin].face;
                    let o_g = self.faces[other_face].group as usize;

                    if g == o_g {
                        let faces = &faces[g];

                        graphs[g].update_edge(faces[i], faces[&other_face], 1);
                    }
                }
            }
        }

        println!("Partitioning the groups into further smaller partitions!");
        println!("(what is even going on anymore)");

        let mut i = 0;
        const PARTS: u32 = 2;
        for (graph, ids) in graphs.iter().zip(ids) {
            let part = config.partition_from_graph(PARTS, graph).unwrap();

            for x in 0..part.len() {
                self.faces[ids[x]].part = i + part[x];
            }

            i += PARTS as i32;
        }

        println!("{i} Partitions from groups");

        // for (i, f) in self.faces.values_mut().enumerate() {
        //     // Some faces will have already been removed
        //     f.group = groups[f.part as usize];
        // }

        Ok(())
        //Ok(groups)
    }

    pub fn partition_count(&self) -> usize {
        self.partition_count
    }
}
