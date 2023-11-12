use common::{tri_mesh::TriMesh, GroupInfo, PartitionInfo};
use glam::{Vec3, Vec4, Vec4Swizzles};
use idmap::IntegerId;
use metis::{idx_t, PartitioningConfig, PartitioningError};
use std::collections::HashMap;

use crate::{
    mesh_iter::{AllEdgeIter, EdgeIter},
    vertex::{VertID, Vertex},
};

//Definition 6: A cut in the DAG is a subset of the tree such that for every node Ci all ancestors
//of Ci are in the cut as well. The front of the cut is the set of arcs that connect a node in the cut
//to a node outside.

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

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub part: usize,
}

#[derive(Debug, Clone)]
pub struct WingedMesh {
    verts: Vec<Vertex>,
    faces: idmap::DirectIdMap<FaceID, Face>,
    edges: Vec<HalfEdge>,
    edge_map: HashMap<VertID, Vec<EdgeID>>,
    pub partitions: Vec<PartitionInfo>,
    pub groups: Vec<GroupInfo>,
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

    pub fn iter_edges(&self) -> AllEdgeIter {
        AllEdgeIter::new(self.edges.clone(), Some(0))
    }

    pub fn iter_edge_loop(&self, e: EdgeID) -> EdgeIter {
        // emit 3 edges and a none
        EdgeIter::new(self, e, Some(e), 3)
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
                self[v].edge = v.outgoing_edges(self).get(0).copied();
            }
        }
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

    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> (Self, Box<[Vec4]>) {
        let tri_mesh = TriMesh::from_gltf(path).unwrap();

        let face_count = tri_mesh.indices.len() / 3;
        let mut mesh = WingedMesh::new(face_count, tri_mesh.verts.len());

        for i in 0..face_count {
            let a = tri_mesh.indices[i * 3] as usize;
            let b = tri_mesh.indices[i * 3 + 1] as usize;
            let c = tri_mesh.indices[i * 3 + 2] as usize;

            mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
        }

        (mesh, tri_mesh.verts)
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

    fn generate_partition_array(
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

    pub fn partition(
        &mut self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<(), PartitioningError> {
        let part = self.generate_partition_array(config, partitions)?;

        let mut max_part = 0;
        for (i, f) in self.faces.values_mut().enumerate() {
            // Some faces will have already been removed
            f.part = part[i] as usize;
            max_part = max_part.max(f.part)
        }

        self.partitions = vec![
            PartitionInfo {
                child_group_index: None,
                group_index: usize::MAX,
                tight_bound: Default::default()
            };
            max_part + 1
        ];

        Ok(())
    }

    pub fn group(
        &mut self,
        config: &PartitioningConfig,
        verts: &[Vec4],
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

        // create new array of groups, and remember the old groups
        let mut old_groups = vec![
            GroupInfo {
                tris: 0,
                monotonic_bound: Default::default(),
                partitions: Vec::new()
            };
            group_count
        ];

        std::mem::swap(&mut self.groups, &mut old_groups);

        // Partition -> Group
        if group_count != 1 {
            for (part, &group) in config
                .partition_from_graph(group_count as u32, &graph)?
                .iter()
                .enumerate()
            {
                self.partitions[part].group_index = group as usize;
            }
        } else {
            for p in &mut self.partitions {
                p.group_index = 0;
            }
        };

        for (part, info) in self.partitions.iter().enumerate() {
            self.groups[info.group_index].partitions.push(part);
        }

        for f in self.faces.values_mut() {
            let f_group_info = &mut self.groups[self.partitions[nodes[f.part].index()].group_index];
            f_group_info.tris += 1;
            f_group_info
                .monotonic_bound
                .translate(verts[self.edges[f.edge.0].vert_origin.0].xyz());
        }
        // Take averages
        for g in &mut self.groups {
            g.monotonic_bound.normalise(g.tris);
        }

        // Find radii of groups
        for f in self.faces.values_mut() {
            let f_group_info = &mut self.groups[self.partitions[nodes[f.part].index()].group_index];

            f_group_info
                .monotonic_bound
                .include_point(verts[self.edges[f.edge.0].vert_origin.0].xyz());
        }

        println!(
            "Including child bounds with {} old groups",
            old_groups.len()
        );

        for g in &mut self.groups {
            // SQRT each group
            //    g.monotonic_bound.radius = g.monotonic_bound.radius.sqrt();

            // Each group also must envelop all the groups it is descended from,
            // as our partitions must do the same, as we base them off group info

            for p in &g.partitions {
                if let Some(child_group_index) = self.partitions[*p].child_group_index {
                    let child_group = &old_groups[child_group_index];
                    // combine groups radius
                    g.monotonic_bound
                        .include_sphere(&child_group.monotonic_bound);
                }
            }
        }

        Ok(group_count)
    }

    /// Within each group, split triangles into two completely new partitions, so as not to preserve any old seams between ancient partitions
    /// Ensures the data structure is seamless with changing seams! Yippee!
    /// Will update the partitions list, but groups list will still refer to old partitions. To find out what group these should be in, before regrouping,
    /// look at `child_group_index`
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

        // Ungrouped partitions but with a dependence on an old group
        let mut new_partitions = Vec::new();

        for (i_group, (graph, ids)) in graphs.iter().zip(ids).enumerate() {
            // TODO: fine tune so we get 64/126 meshlets
            let parts = 2; // (graph.node_count() as u32).div_ceil(60);

            let part = config.partition_from_graph(parts, graph)?;

            // Each new part needs to register its dependence on the group we were a part of before
            let child_group = self.partitions[self.faces[ids[0]].part].group_index;

            assert_eq!(i_group, child_group);

            // Update partitions of the actual triangles
            for x in 0..part.len() {
                self.faces[ids[x]].part = new_partitions.len() + part[x] as usize;
            }

            for p in 0..parts {
                //    self.groups[group].partitions.push(new_partitions.len());

                new_partitions.push(PartitionInfo {
                    child_group_index: Some(i_group),
                    group_index: usize::MAX,
                    tight_bound: Default::default(), //TODO:
                })
            }
        }
        self.partitions = new_partitions;

        Ok(self.partitions.len())
        //Ok(groups)
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
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    pub fn edge_map(&self) -> &HashMap<VertID, Vec<EdgeID>> {
        &self.edge_map
    }

    pub fn edges(&self) -> &[HalfEdge] {
        self.edges.as_ref()
    }
}
#[cfg(test)]
pub mod test {
    use super::*;
    use std::{collections::HashSet, error::Error, process};

    use metis::PartitioningConfig;
    use petgraph::dot;
    use std::fs;

    use crate::winged_mesh::WingedMesh;
    pub const TEST_MESH_HIGH: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere.glb";

    pub const TEST_MESH_MID: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere_low.glb";

    pub const TEST_MESH_LOW: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\torus.glb";

    pub const DOT_OUT: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\multires\\baker\\graph.dot";

    pub const SVG_OUT: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\multires\\baker\\graph.svg";

    /// Extra assertion methods for test environment
    impl WingedMesh {
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
        /// Find the inner boundary of a set of faces. Quite simple - record all edges we have seen, and any twins for those.
        /// Inner boundary as will not include edges from an edge of the mesh, as these have no twins.
        pub fn face_boundary(&self, faces: &Vec<&Face>) -> HashSet<EdgeID> {
            let mut unseen_edges = HashSet::<EdgeID>::with_capacity(faces.len() * 3);

            for face in faces {
                for edge in self.iter_edge_loop(face.edge) {
                    if let Some(t) = self[edge].twin {
                        unseen_edges.insert(t);
                    }
                }
            }

            let all_unseen_edges = unseen_edges.clone();
            unseen_edges.retain(|&edge| !all_unseen_edges.contains(&self[edge].twin.unwrap()));

            unseen_edges
        }
    }

    #[test]
    pub fn test_continued_validity() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        println!(
            "Loading from gltf {:?}!",
            std::fs::canonicalize(TEST_MESH_MID)?
        );
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_MID);

        mesh.assert_valid();

        mesh.partition(test_config, (mesh.faces().len() as u32).div_ceil(60))?;

        mesh.assert_valid();

        mesh.group(test_config, &verts)?;

        mesh.assert_valid();

        mesh.partition_within_groups(test_config)?;

        mesh.assert_valid();

        Ok(())
    }

    #[test]
    pub fn test_faces_boundary() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let mesh = TEST_MESH_MID;
        println!("Loading from gltf {:?}!", std::fs::canonicalize(mesh)?);
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition(test_config, (mesh.faces().len() as u32).div_ceil(60))?;

        let mut boundary_face_ratio = 0.0;
        for pi in 0..mesh.partitions.len() {
            // Assert that new parts and the parts in group have the same boundary

            let faces = mesh.faces.values().filter(|&f| pi == f.part).collect();

            let boundary = mesh.face_boundary(&faces);

            for &e in &boundary {
                assert!(
                    !faces.contains(&&mesh.faces[mesh[e].face]),
                    "'Boundary' contains edge inside itself"
                )
            }

            boundary_face_ratio += faces.len() as f32 / boundary.len() as f32;
        }

        boundary_face_ratio /= mesh.partitions.len() as f32;

        println!("Average partition face count / boundary length: {boundary_face_ratio}");

        Ok(())
    }

    #[test]
    pub fn test_group_repartitioning() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let mesh = TEST_MESH_MID;
        println!("Loading from gltf {:?}!", std::fs::canonicalize(mesh)?);
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition(test_config, (mesh.faces().len() as u32).div_ceil(60))?;

        mesh.group(test_config, &verts)?;

        let old_faces = mesh.faces.clone();

        // Old parts must have no child - there is no LOD-1
        for p in &mesh.partitions {
            assert!(p.child_group_index.is_none());
        }

        mesh.partition_within_groups(test_config)?;

        // assert that the group boundaries are the same

        // Same group indices, new set of partitions per group
        let mut new_part_groups = vec![Vec::new(); mesh.groups.len()];

        // new parts must have no group - there is no grouping assigned yet
        for (i, p) in mesh.partitions.iter().enumerate() {
            assert_eq!(p.group_index, usize::MAX);
            assert!(p.child_group_index.is_some());

            let g_i = p.child_group_index.unwrap();

            new_part_groups[g_i].push(i);
        }

        let avg_size = (new_part_groups.iter().map(|l| l.len()).sum::<usize>() as f32)
            / (new_part_groups.len() as f32);

        println!(
            "Test has grouped partitions into {} groups, with {} partitions average",
            new_part_groups.len(),
            avg_size
        );

        let mut boundary_face_ratio = 0.0;

        for (gi, group) in mesh.groups.iter().enumerate() {
            let new_parts = &new_part_groups[gi];
            let old_parts = &group.partitions;

            // Assert that new parts and the parts in group have the same boundary

            let old_faces = old_faces
                .values()
                .filter(|&f| old_parts.contains(&f.part))
                .collect();
            let new_faces = mesh
                .faces
                .values()
                .filter(|&f| new_parts.contains(&f.part))
                .collect();

            let old_boundary = mesh.face_boundary(&old_faces);
            let new_boundary = mesh.face_boundary(&new_faces);

            boundary_face_ratio += old_faces.len() as f32 / old_boundary.len() as f32;

            assert_eq!(
                old_faces.len(),
                new_faces.len(),
                "Repartitioning of group without remesh changed face count"
            );
            assert_eq!(
                old_boundary.len(),
                new_boundary.len(),
                "Repartitioning of group changed group boundary (len)"
            );

            assert!(
                old_boundary.iter().all(|e| new_boundary.contains(e)),
                "Repartitioning of group changed group boundary (edgeid)"
            );
        }

        boundary_face_ratio /= mesh.groups.len() as f32;

        println!("Average group face count / boundary length: {boundary_face_ratio}");

        Ok(())
    }

    #[test]
    pub fn generate_partition_hierarchy_graph() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelRecursiveBisection,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let mesh = TEST_MESH_HIGH;
        println!("Loading from gltf {:?}!", std::fs::canonicalize(mesh)?);
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        println!("Faces: {}, Verts: {}", mesh.face_count(), mesh.verts.len());

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition(test_config, (mesh.face_count() as u32).div_ceil(60))?;

        mesh.group(test_config, &verts)?;
        let mut graph: petgraph::Graph<(), ()> = petgraph::Graph::new();

        let mut old_part_nodes: Vec<_> =
            mesh.partitions.iter().map(|o| graph.add_node(())).collect();

        // Record a big colour map for node indexes, to show grouping
        let mut colouring = HashMap::new();

        let mut seen_groups = 0;

        loop {
            for (i, &n) in old_part_nodes.iter().enumerate() {
                colouring.insert(n, mesh.partitions[i].group_index + seen_groups);
            }

            seen_groups += mesh.groups.len();

            mesh.partition_within_groups(test_config)?;

            let new_part_nodes: Vec<_> =
                mesh.partitions.iter().map(|o| graph.add_node(())).collect();

            for (new_p_i, new_p) in mesh.partitions.iter().enumerate() {
                let g_i = new_p.child_group_index.unwrap();

                for &old_p_i in &mesh.groups[g_i].partitions {
                    graph.add_edge(old_part_nodes[old_p_i], new_part_nodes[new_p_i], ());
                }
            }

            old_part_nodes = new_part_nodes;

            mesh.group(test_config, &verts)?;

            if mesh.partitions.len() <= 2 {
                break;
            }
        }

        for (i, &n) in old_part_nodes.iter().enumerate() {
            colouring.insert(n, mesh.partitions[i].group_index + seen_groups);
        }

        let cols = [
            "red",
            "blue",
            "green",
            "aqua",
            "cornflowerblue",
            "darkgoldenrod1",
            "deeppink",
            "indigo",
            "orchid",
            "peru",
        ];

        fs::write(
            DOT_OUT,
            format!(
                "{:?}",
                dot::Dot::with_attr_getters(
                    &graph,
                    &[dot::Config::NodeNoLabel, dot::Config::EdgeNoLabel],
                    &|_, _| "arrowhead=none".to_owned(),
                    &|_, (n, _)| format!("shape=point, color={}", cols[colouring[&n] % cols.len()])
                )
            ),
        )?;

        let dot_out = process::Command::new("dot")
            .arg("-Tsvg")
            .arg(DOT_OUT)
            .output()?;

        fs::remove_file(DOT_OUT)?;

        println!("{}", std::str::from_utf8(&dot_out.stderr)?);

        fs::write(SVG_OUT, dot_out.stdout)?;

        Ok(())
    }
}
