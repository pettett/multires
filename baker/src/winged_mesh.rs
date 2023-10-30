extern crate gltf;

use glam::Vec3;
use idmap::IntegerId;
use metis::{idx_t, PartitioningConfig, PartitioningError};
use petgraph::data::{Build, FromElements};
use std::collections::{HashMap, HashSet};

use gltf::mesh::util::ReadIndices;

//Definition 6: A cut in the DAG is a subset of the tree such that for every node Ci all ancestors
//of Ci are in the cut as well. The front of the cut is the set of arcs that connect a node in the cut
//to a node outside.

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
    pub part: usize,
}

#[derive(Debug, Clone)]
pub struct GroupInfo {
    child_partitions: Vec<usize>,
    tris: usize,
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Debug, Clone)]
pub struct PartitionInfo {
    // Group in the previous layer we have been attached to
    pub child_group_index: usize,
    // Part -> Group in this layer
    pub group_index: usize,
}

#[derive(Debug, Clone)]
pub struct WingedMesh {
    verts: Vec<Vertex>,
    faces: idmap::DirectIdMap<FaceID, Face>,
    // partitions: Vec<i32>,
    edges: Vec<HalfEdge>,
    edge_map: HashMap<VertID, Vec<EdgeID>>,
    pub partitions: Vec<PartitionInfo>,
    // Centres for all groups. TODO: Cleanup
    pub groups: Vec<GroupInfo>,
}

pub struct EdgeIter<'a> {
    mesh: &'a WingedMesh,
    start: EdgeID,
    current: Option<EdgeID>,
}

pub struct AllEdgeIter {
    edges: Vec<HalfEdge>,
    current: Option<usize>,
}
impl VertID {
    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn is_local_manifold(&self, mesh: &WingedMesh, is_group_manifold: bool) -> bool {
        let Some(eid_first) = mesh.verts[self.0].edge else {
            return false;
        };

        let mut eid = eid_first;

        let mut last_e_part = None;

        loop {
            // attempt to move around the fan, by moving to our twin edge and going clockwise
            let Some(twin) = mesh[eid].twin else {
                return false;
            };

            let e = &mesh[twin];

            // Compare against last face's partition
            if is_group_manifold {
                let e_part = mesh.partitions[mesh.faces[e.face].part].group_index;

                if let Some(lep) = last_e_part {
                    if e_part != lep {
                        return false;
                    }
                }

                last_e_part = Some(e_part);
            }

            eid = e.edge_left_cw;

            if eid == eid_first {
                return true;
            }
        }
    }
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
            verts: vec![Default::default(); verts],
            //faces: vec![Default::default(); faces],
            faces: idmap::DirectIdMap::with_capacity_direct(faces),
            // partitions: vec![Default::default(); faces],
            edges: Default::default(),
            edge_map: Default::default(),
            groups: Default::default(),
            partitions: Default::default(),
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

    pub fn edge_sqr_length(&self, edge: EdgeID, verts: &[Vec3]) -> f32 {
        let e = &self.edges[edge.0];
        return verts[e.vert_origin.0]
            .distance_squared(verts[self.edges[e.edge_left_cw.0].vert_origin.0]);
    }

    pub fn triangle_from_face(&self, face: &Face) -> [u32; 3] {
        let verts: Vec<usize> = self
            .iter_edge_loop(face.edge)
            .map(|e| self[e].vert_origin.into())
            .collect();

        [verts[0] as _, verts[1] as _, verts[2] as _]
    }

    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> gltf::Result<(Self, Vec<Vec3>)> {
        let (doc, buffers, _) = gltf::import(path)?;

        let mesh = doc.meshes().next().unwrap();
        let p = mesh.primitives().next().unwrap();
        let reader = p.reader(|buffer| Some(&buffers[buffer.index()]));

        let iter = reader.read_positions().unwrap();
        let verts: Vec<_> = iter.map(|v| Vec3::from_array(v)).collect();
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
                part: usize::MAX,
            },
        );
    }

    pub fn faces(&self) -> &idmap::DirectIdMap<FaceID, Face> {
        &self.faces
    }

    pub fn get_partition(&self) -> Vec<usize> {
        self.faces().values().map(|f| f.part).collect()
    }

    pub fn get_group(&self) -> Vec<usize> {
        self.faces()
            .values()
            .map(|f| self.partitions[f.part].group_index)
            .collect()
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

    pub fn apply_partition(
        &mut self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<(), PartitioningError> {
        let part = self.partition(config, partitions)?;

        let mut max_part = 0;
        for (i, f) in self.faces.values_mut().enumerate() {
            // Some faces will have already been removed
            f.part = part[i] as usize;
            max_part = max_part.max(f.part)
        }

        self.partitions = vec![
            PartitionInfo {
                child_group_index: 0,
                group_index: 0,
            };
            max_part + 1
        ];

        Ok(())
    }

    pub fn group(
        &mut self,
        config: &PartitioningConfig,
        verts: &[Vec3],
    ) -> Result<usize, PartitioningError> {
        //TODO: Give lower weight to grouping partitions that have not been recently grouped, to ensure we are
        // constantly overwriting old borders with remeshes

        let group_count = self.partitions.len().div_ceil(4);
        println!(
            "Partitioning into {group_count} groups from {} partitions",
            self.partitions.len()
        );

        let mut graph = petgraph::Graph::<i32, i32>::new();

        let nodes: Vec<_> = (0..self.partitions.len())
            .map(|i| graph.add_node(1))
            .collect();

        for (i, face) in self.faces().iter() {
            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self[e].twin {
                    let other_face = &self.faces[self[twin].face];

                    graph.update_edge(nodes[face.part], nodes[other_face.part], 1);
                }
            }
        }

        let mut groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            groups.push(GroupInfo {
                child_partitions: Vec::new(),
                tris: 0,
                center: Vec3::ZERO,
                radius: 0.0,
            })
        }
        // Partition -> Group
        if group_count != 1 {
            for (part, group) in config
                .partition_from_graph(group_count as u32, &graph)?
                .iter()
                .enumerate()
            {
                self.partitions[part].group_index = *group as usize;
            }
        } else {
            for p in &mut self.partitions {
                p.group_index = 0;
            }
        };

        for (part, info) in self.partitions.iter().enumerate() {
            groups[info.group_index].child_partitions.push(part);
        }

        for f in self.faces.values_mut() {
            // Some faces will have already been removed
            let f_part_info = &self.partitions[nodes[f.part].index()];
            //f.group = group;
            groups[f_part_info.group_index].tris += 1;
            groups[f_part_info.group_index].center += verts[self.edges[f.edge.0].vert_origin.0];
        }
        // Take averages
        for g in &mut groups {
            g.center /= g.tris as f32;
        }

        self.groups = groups;

        Ok(group_count)
    }

    /// Within each group, split triangles into two completely new partitions, so as not to preserve any old seams between ancient partitions
    /// Ensures the data structure is seamless with changing seams! Yippee!
    pub fn partition_within_groups(
        &mut self,
        config: &PartitioningConfig,
    ) -> Result<usize, PartitioningError> {
        let group_count = self.partitions.len().div_ceil(4);
        println!("Partitioning {group_count} groups into sub-partitions");

        let mut graphs = vec![petgraph::Graph::<i32, i32>::new(); group_count];

        // Stores vecs of face IDs, which should be associated with data in graphs
        let mut ids = vec![Vec::new(); group_count];
        let mut faces = vec![HashMap::new(); group_count];

        // Give every triangle a node
        for (i, face) in self.faces().iter() {
            let g = self.partitions[face.part].group_index;
            let n = graphs[g].add_node(1);

            // indexes should correspond
            assert_eq!(n.index(), ids[g].len());

            ids[g].push(*i);
            faces[g].insert(*i, n);
        }
        // Apply links between nodes
        for (i, face) in self.faces().iter() {
            let g = self.partitions[face.part].group_index;

            for e in self.iter_edge_loop(face.edge) {
                if let Some(twin) = self[e].twin {
                    let other_face = self[twin].face;
                    let o_g = self.partitions[self.faces[other_face].part].group_index;

                    if g == o_g {
                        let faces = &faces[g];

                        graphs[g].update_edge(faces[i], faces[&other_face], 1);
                    }
                }
            }
        }

        //println!("Partitioning the groups into further smaller partitions!");
        //println!("(what is even going on anymore)");

        // Ungrouped partitions but with a dependence on an old group
        let mut new_partitions = Vec::new();

        for (graph, ids) in graphs.iter().zip(ids) {
            // TODO: fine tune so we get 64/126 meshlets
            let parts = 2; // (graph.node_count() as u32).div_ceil(60);

            let part = config.partition_from_graph(parts, graph)?;

            // Update partitions of the actual triangles
            for x in 0..part.len() {
                self.faces[ids[x]].part = new_partitions.len() + part[x] as usize;
            }
            // Each new part needs to register its dependence on the group we were a part of before
            let old_group = self.partitions[self.faces[ids[0]].part].group_index;

            for p in 0..parts {
                new_partitions.push(PartitionInfo {
                    child_group_index: old_group,
                    group_index: usize::MAX,
                })
            }
        }
        self.partitions = new_partitions;
        // for (i, f) in self.faces.values_mut().enumerate() {
        //     // Some faces will have already been removed
        //     f.group = groups[f.part as usize];
        // }

        Ok(self.partitions.len())
        //Ok(groups)
    }

    pub fn partition_groups(&self) -> Vec<Vec<usize>> {
        let mut partition_groups: Vec<Vec<usize>> = vec![Vec::new(); self.groups.len()];
        for face in self.faces.values() {
            let v = &mut partition_groups[self.partitions[face.part].group_index];
            if !v.contains(&face.part) {
                v.push(face.part);
            }
        }
        partition_groups
    }

    pub fn partition_dependence(&self) -> HashMap<usize, usize> {
        self.partitions
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.child_group_index))
            .collect()
    }

    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }
}
