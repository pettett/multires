use anyhow::Context;
use common::{tri_mesh::TriMesh, BoundingSphere, ClusterInfo, GroupInfo};
use glam::{Vec3, Vec4};
use idmap::IntegerId;
use std::{collections::HashSet, fs};

use crate::pidge::Pidge;

use super::{
    iter::EdgeIter,
    vertex::{VertID, Vertex},
};

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("Invalid triangle {0:?}")]
    InvalidTri(FaceID),
    #[error("Invalid edge {0:?}")]
    InvalidEdge(EdgeID),
    #[error("Invalid vertex {0:?}")]
    InvalidVertex(VertID),
    #[error("{0:?} Has two edges with same property {1:?}, {2:?} [{3:?}, {4:?}]")]
    DuplicateEdges(VertID, EdgeID, EdgeID, HalfEdge, HalfEdge),
    #[error("Cw piece of edge {0:?} is invalid")]
    InvalidCwEdge(EdgeID),
    #[error("Twin of edge {0:?} is invalid")]
    InvalidTwin(EdgeID),
    #[error("CCw piece of edge {0:?} is invalid")]
    InvalidCCwEdge(EdgeID),
    #[error("Edge {0:?} bridges a single vertex {1:?}")]
    SingletonEdge(EdgeID, VertID),
    #[error("Failed edge collapse on {0:?}, {1:?} -> {2:?}")]
    EdgeCollapse(EdgeID, VertID, VertID),
    #[error("Group {0:?} is invalid")]
    InvalidGroup(usize),
    #[error("{0:?} and {1:?} should be neighbours but are not")]
    InvalidNeighbours(usize, usize),
    #[error("Ran out of valid edges while reducing")]
    OutOfEdges,
}

//Definition 6: A cut in the DAG is a subset of the tree such that for every node Ci all ancestors
//of Ci are in the cut as well. The front of the cut is the set of arcs that connect a node in the cut
//to a node outside.

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub struct EdgeID(pub usize);
impl Into<usize> for EdgeID {
    fn into(self) -> usize {
        self.0
    }
}
impl From<usize> for EdgeID {
    fn from(value: usize) -> Self {
        EdgeID(value)
    }
}

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceID(pub usize);
impl Into<usize> for FaceID {
    fn into(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for FaceID {
    fn from(value: usize) -> Self {
        FaceID(value)
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

impl IntegerId for EdgeID {
    fn from_id(id: u64) -> Self {
        EdgeID(id as usize)
    }

    fn id(&self) -> u64 {
        self.0 as u64
    }

    fn id32(&self) -> u32 {
        self.0 as u32
    }
}
impl FaceID {
    pub fn center(self, mesh: &WingedMesh, verts: &[Vec4]) -> Vec4 {
        let mut c = Vec4::ZERO;

        for e in mesh.iter_edge_loop(mesh.get_face(self).edge) {
            c += verts[mesh.get_edge(e).vert_origin.0];
        }

        c / 3.0
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct HalfEdge {
    pub vert_origin: VertID,
    // This is not actually needed, as the destination is the origin of the cw edge
    //pub vert_destination: VertID,
    pub face: FaceID,
    /// Edge leading on from the dest vert
    pub edge_left_cw: EdgeID,
    /// Edge connecting into the origin vert
    pub edge_left_ccw: EdgeID,

    pub age: u32,

    pub twin: Option<EdgeID>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub cluster_idx: usize,
    pub colour: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WingedMesh {
    faces: Pidge<FaceID, Face>,
    edges: Pidge<EdgeID, HalfEdge>,
    verts: Pidge<VertID, Vertex>,
    pub clusters: Vec<ClusterInfo>,
    pub groups: Vec<GroupInfo>,
}

impl WingedMesh {
    pub fn new(faces: usize, verts: usize) -> Self {
        Self {
            faces: Pidge::with_capacity(faces),
            edges: Pidge::with_capacity(faces * 3),
            verts: Pidge::with_capacity(verts),
            groups: vec![],
            clusters: vec![ClusterInfo::default()], 
        }
    }
    pub fn get_face(&self, face: FaceID) -> &Face {
        self.faces.get(face)
    }
    pub fn try_get_face_mut(&mut self, fid: FaceID) -> Option<&mut Face> {
        self.faces.try_get_mut(fid)
    }
    pub fn get_face_mut(&mut self, fid: FaceID) -> &mut Face {
        self.faces.get_mut(fid)
    }

    pub fn try_get_edge(&self, eid: EdgeID) -> Result<&HalfEdge, MeshError> {
        self.edges.try_get(eid).ok_or(MeshError::InvalidEdge(eid))
    }

    pub fn get_edge(&self, eid: EdgeID) -> &HalfEdge {
        self.edges.get(eid)
    }
    pub fn try_get_edge_mut(&mut self, eid: EdgeID) -> Option<&mut HalfEdge> {
        self.edges.try_get_mut(eid)
    }
    pub fn get_edge_mut(&mut self, eid: EdgeID) -> &mut HalfEdge {
        self.edges.get_mut(eid)
    }

    pub fn insert_edge(&mut self, eid: EdgeID, half_edge: HalfEdge) {
        self.edges.insert(eid, half_edge)
    }
    pub fn insert_face(&mut self, fid: FaceID, face: Face) {
        self.faces.insert(fid, face)
    }

    pub fn try_get_vert(&self, vid: VertID) -> Result<&Vertex, MeshError> {
        self.verts.try_get(vid).ok_or(MeshError::InvalidVertex(vid))
    }

    pub fn get_vert(&self, vert: VertID) -> &Vertex {
        self.try_get_vert(vert).as_ref().unwrap()
    }
    pub fn get_vert_mut(&mut self, vert: VertID) -> &mut Vertex {
        self.verts.get_mut(vert)
    }
    pub fn get_vert_or_default(&mut self, vid: VertID) -> &mut Vertex {
        if !self.verts.slot_full(vid) {
            self.verts.insert(vid, Vertex::default());
        }
        self.verts.get_mut(vid)
    }

    pub fn wipe_face(&mut self, face: FaceID) {
        #[cfg(test)]
        {
            self.assert_face_valid(face).unwrap();
        }

        self.faces.wipe(face)
    }
    pub fn wipe_edge(&mut self, edge: EdgeID) {
        self.edges.wipe(edge)
    }

    pub fn wipe_vert(&mut self, vert: VertID) {
        self.verts.wipe(vert)
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn vert_count(&self) -> usize {
        self.verts.len()
    }

    pub fn iter_verts(&self) -> impl Iterator<Item = (VertID, &Vertex)> {
        self.verts
            .iter_with_empty()
            .enumerate()
            .filter_map(|(i, x)| {
                if x.is_some() {
                    Some((VertID(i), x.unwrap()))
                } else {
                    None
                }
            })
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = (EdgeID, &HalfEdge)> {
        self.edges
            .iter_with_empty()
            .enumerate()
            .filter_map(|(i, x)| {
                if x.is_some() {
                    Some((EdgeID(i), x.unwrap()))
                } else {
                    None
                }
            })
    }

    pub fn iter_faces(&self) -> impl Iterator<Item = (FaceID, &Face)> + '_ {
        self.faces
            .iter_with_empty()
            .enumerate()
            .filter_map(|(i, x)| {
                if x.is_some() {
                    Some((FaceID(i), x.unwrap()))
                } else {
                    None
                }
            })
    }

    pub fn iter_faces_mut(&mut self) -> impl Iterator<Item = (FaceID, &mut Face)> {
        self.faces
            .iter_mut_with_empty()
            .enumerate()
            .filter_map(|(i, x)| match x {
                Some(x) => Some((FaceID(i), x)),
                None => None,
            })
    }

    pub fn iter_edges_mut(&mut self) -> impl Iterator<Item = (EdgeID, &mut HalfEdge)> {
        self.edges
            .iter_mut_with_empty()
            .enumerate()
            .filter_map(|(i, x)| match x {
                Some(x) => Some((EdgeID(i), x)),
                None => None,
            })
    }
    pub fn iter_verts_mut(&mut self) -> impl Iterator<Item = (VertID, &mut Vertex)> {
        self.verts
            .iter_mut_with_empty()
            .enumerate()
            .filter_map(|(i, x)| match x {
                Some(x) => Some((VertID(i), x)),
                None => None,
            })
    }

    pub fn edge_sqr_length(&self, edge: EdgeID, verts: &[Vec3]) -> f32 {
        let e = &self.get_edge(edge);
        return verts[e.vert_origin.0]
            .distance_squared(verts[self.get_edge(e.edge_left_cw).vert_origin.0]);
    }

    pub fn triangle_from_face(&self, face: &Face) -> [usize; 3] {
        let e1 = self.get_edge(face.edge);
        let e0 = self.get_edge(e1.edge_left_ccw);
        let e2 = self.get_edge(e1.edge_left_cw);

        [e0.vert_origin.0, e1.vert_origin.0, e2.vert_origin.0]
    }

    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> (Self, Box<[Vec4]>, Box<[Vec4]>) {
        let tri_mesh = TriMesh::from_gltf(&path).unwrap();

        let face_count = tri_mesh.indices.len() / 3;
        let mut mesh = WingedMesh::new(face_count, tri_mesh.verts.len());

        println!(
            "Loading GLTF {:?} with {} faces:",
            fs::canonicalize(path).unwrap(),
            face_count
        );
        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(face_count as u64);

        for i in 0..face_count {
            let a = tri_mesh.indices[i * 3] as usize;
            let b = tri_mesh.indices[i * 3 + 1] as usize;
            let c = tri_mesh.indices[i * 3 + 2] as usize;

            mesh.add_tri(FaceID(i), VertID(a), VertID(b), VertID(c));
            #[cfg(feature = "progress")]
            bar.inc(1);
        }
        #[cfg(feature = "progress")]
        bar.finish();

        (mesh, tri_mesh.verts, tri_mesh.normals)
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        self.try_get_vert(a)
            .as_ref()
            .map(|v| {
                v.outgoing_edges()
                    .iter()
                    .filter(|&&p| self.get_edge(self.get_edge(p).edge_left_cw).vert_origin == b)
                    .next()
                    .copied()
            })
            .ok()
            .flatten()
    }

    fn add_half_edge(
        &mut self,
        orig: VertID,
        dest: VertID,
        face: FaceID,
        eid: EdgeID,
        edge_left_cw: EdgeID,
        edge_left_ccw: EdgeID,
    ) {
        let twin = self.find_edge(dest, orig);
        let e = HalfEdge {
            vert_origin: orig,
            face,
            edge_left_cw,
            edge_left_ccw,
            twin,
            age: 0,
        };

        if let Some(twin_eid) = twin {
            self.get_edge_mut(twin_eid).twin = Some(eid);
        }

        self.get_vert_or_default(orig).add_outgoing(eid);

        self.get_vert_or_default(dest).add_incoming(eid);

        self.insert_edge(eid, e);
    }

    pub fn add_tri(&mut self, f: FaceID, a: VertID, b: VertID, c: VertID) {
        let iea = EdgeID(f.0 * 3 + 0);
        let ieb = EdgeID(f.0 * 3 + 1);
        let iec = EdgeID(f.0 * 3 + 2);

        self.add_half_edge(a, b, f, iea, ieb, iec);
        self.add_half_edge(b, c, f, ieb, iec, iea);
        self.add_half_edge(c, a, f, iec, iea, ieb);

        self.insert_face(
            f,
            Face {
                edge: iea,
                cluster_idx: 0,
                colour: 0,
            },
        );
    }

    /// Collapse a triangle, removing it from the graph, and pulling the two triangles on non-eid edges together
    /// 	A
    ///  	| 1
    ///  	0  D
    ///   	| 2
    /// 	B
    /// Preconditions: A valid triangle on valid edge `eid`
    /// Postconditions:
    /// - `eid` no longer valid, triangle removed,
    /// - The twins of the two non-eid edges are linked, despite not actually being opposites (invalid)
    /// - No edges have been moved
    ///
    /// This function should only be called as part of an edge collapse, as leaves mesh partially invalid.
    fn collapse_tri(&mut self, eid: EdgeID) -> anyhow::Result<()> {
        let edge = self.try_get_edge(eid)?;

        let fid = edge.face;

        let tri = [eid, edge.edge_left_ccw, edge.edge_left_cw];

        assert_eq!(tri[0], eid);
        // we are pinching edge to nothing, so make the other two edges twins
        let t0 = edge.twin;
        let t1 = self.get_edge(tri[1]).twin;
        let t2 = self.get_edge(tri[2]).twin;

        self.wipe_face(fid);

        if let Some(t) = t1 {
            self.get_edge_mut(t).twin = t2;
        }
        if let Some(t) = t2 {
            self.get_edge_mut(t).twin = t1;
        }

        // Unlikely this will ever matter, but preserve consistency of mesh, as this edge is to be removed.
        if let Some(t) = t0 {
            self.get_edge_mut(t).twin = None;
        }

        // Remove any last references to this triangle
        for &e in &tri {
            let v_o = self.get_edge(e).vert_origin;
            let ccw = self.get_edge(e).edge_left_ccw;

            self.get_vert_mut(v_o).remove_outgoing(e);
            self.get_vert_mut(v_o).remove_incoming(ccw);

            self.wipe_edge(e);
        }
        Ok(())
    }

    fn neighbour_vertices(&self, v: &Vertex) -> HashSet<VertID> {
        // This is the exact number of neighbours, assuming this is a manifold vertex, otherwise it will still be pretty close.
        let mut neighbours = HashSet::with_capacity(v.incoming_edges().len());

        neighbours.extend(
            v.incoming_edges()
                .iter()
                .map(|&e| self.get_edge(e).vert_origin),
        );

        neighbours.extend(v.outgoing_edges().iter().map(|&e| {
            let (_, dest) = e.src_dst(self).unwrap();
            dest
        }));

        neighbours
    }

    /// Does this edge have the right amount of shared neighbours for edge collapse
    ///
    /// Logic sourced from https://stackoverflow.com/a/27049418
    pub fn max_one_joint_neighbour_vertices_per_side(&self, eid: EdgeID) -> bool {
        let (src, dst) = eid.src_dst(&self).unwrap();

        let v_src = self.get_vert(src);
        let has_twin = self.get_edge(eid).twin.is_some();

        let mut joint_shared_count = 0;

        for &incoming in v_src.incoming_edges() {
            // Count every vertex from the incoming first, as these are slightly quicker to find
            let neighbour = self.get_edge(incoming).vert_origin;

            if self.find_edge(neighbour, dst).is_some() || self.find_edge(dst, neighbour).is_some()
            {
                joint_shared_count += 1;
            }
        }

        for &outgoing in v_src.outgoing_edges() {
            // Count every none-incoming-accessible vertex afterwards
            if self.get_edge(outgoing).twin.is_some() {
                // Do not double count vertices that can be reached in two ways
                continue;
            }

            let (_, neighbour) = outgoing.src_dst(self).unwrap();

            assert!(self.find_edge(neighbour, src).is_none());

            if self.find_edge(neighbour, dst).is_some() || self.find_edge(dst, neighbour).is_some()
            {
                joint_shared_count += 1;
            }
        }

        let new = has_twin && joint_shared_count == 2 || !has_twin && joint_shared_count == 1;

        #[cfg(test)]
        {
            // Regression test / much easier logic to read

            let (src, dst) = eid.src_dst(&self).unwrap();
            let v0 = self.get_vert(src);
            let v1 = self.get_vert(dst);

            let n0 = self.neighbour_vertices(v0);
            let n1 = self.neighbour_vertices(v1);

            let joint_shared = n0.intersection(&n1).count();
            let has_twin = self.get_edge(eid).twin.is_some();

            let old = has_twin && joint_shared == 2 || !has_twin && joint_shared == 1;

            assert_eq!(old, new);
        }

        return new;
    }

    /// Collapse an edge so it no longer exists, the source vertex is no longer referenced,
    /// 	A
    ///   /	^ \
    ///  C	|  D
    ///   \	| /
    /// 	B
    ///
    /// Preconditions:
    ///  - `eid` is a valid edge
    ///  - `eid.face` is a valid face
    ///  - A and B have exactly two joint neighbours
    ///
    /// Postconditions:
    ///  - `eid` is an invalid edge.
    ///  - `eid.face` and `eid.twin?.face` are invalid faces
    pub fn collapse_edge(&mut self, eid: EdgeID) -> anyhow::Result<()> {
        let edge = self
            .try_get_edge(eid)
            .context("Attempted to collapse invalid edge")?
            .clone();

        let (vid_orig, vid_dest) = eid.src_dst(self)?;

        #[cfg(test)]
        {
            if !self.max_one_joint_neighbour_vertices_per_side(eid) {
                Err(MeshError::EdgeCollapse(eid, vid_orig, vid_dest)).context(
                    "Attempting to collapse edge with too many joint neighbours on one side",
                )?;
            }
        }

        self.collapse_tri(eid)
            .context("Failed to collapse main triangle")
            .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))?;

        // Assert the vertexes are still valid, which they should be despite some invalid triangles around us
        #[cfg(test)]
        {
            self.assert_vertex_valid(vid_dest)
                .context("Invalid dest vertex after single collapse")
                .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))?;
            self.assert_vertex_valid(vid_orig)
                .context("Invalid orig vertex after single collapse")
                .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))?;
        }

        if let Some(e0t) = edge.twin {
            self.collapse_tri(e0t)
                .context("Failed to collapse twin triangle")
                .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))?;
        }

        // Remove `vert_origin`

        // Main issue is a situation where triangles do not fan around the cake in both directions
        // This will collapse an edge to have dest and source in same position.
        // Because of this, we need to store all ingoing and outgoings per vertex, which isn't the worst in the world
        // Although it would save a bit of memory to just store every fan
        let b_outgoings = self.get_vert(vid_orig).outgoing_edges().to_vec();

        for b_outgoing in b_outgoings {
            // Don't fix invalid edges
            #[cfg(test)]
            {
                let (orig, dest) = b_outgoing.src_dst(self).unwrap();
                assert_eq!(orig, vid_orig);
                assert_ne!(dest, vid_dest);
            }

            let b_outgoing_ccw = self.get_edge(b_outgoing).edge_left_ccw;

            self.get_edge_mut(b_outgoing).vert_origin = vid_dest;

            // Moving this origin moves both the start of this edge and the dest of the previous edge
            self.get_vert_mut(vid_dest).add_outgoing(b_outgoing);
            self.get_vert_mut(vid_dest).add_incoming(b_outgoing_ccw);

            //TODO: add some tests here to make sure we don't break a triangle

            // Reset their ages, as moved
            self.get_edge_mut(b_outgoing).age = 0;
            self.get_edge_mut(b_outgoing_ccw).age = 0;
        }

        self.wipe_vert(vid_orig);

        #[cfg(test)]
        {
            self.assert_vertex_valid(vid_dest)
                .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))
                .unwrap();

            self.try_get_edge(eid)
                .err()
                .ok_or(MeshError::InvalidEdge(eid))
                .context("Edge not destroyed by edge collapse")
                .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))
                .unwrap();

            // Make sure all the faces around va are valid, which will be everything we have touched
            //FIXME: Fails
            for &out_eid in self.get_vert(vid_dest).outgoing_edges() {
                self.assert_face_valid(self.get_edge(out_eid).face)
                    .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))
                    .unwrap();
            }

            //Two faces removed, three linking between faces removed
            //(removed triangles to each other, and 1/2 of the links of the neighbours that are pinched)
            // PASSES: Do not need to test again.
            //let new_graph = self.generate_face_graph();
            //assert_eq!(new_graph.node_count(), previous_graph.node_count() - 2);
            //assert_eq!(new_graph.edge_count(), previous_graph.edge_count() - 3);

            //for g in self.generate_group_graphs() {
            //    test::assert_contiguous_graph(&g);
            //}

            //self.assert_valid();
        }
        //self.assert_valid();

        Ok(())
    }

    pub fn get_partition(&self) -> Vec<usize> {
        self.faces
            .iter_with_empty()
            .filter_map(|f| f.as_ref().map(|f| f.cluster_idx))
            .collect()
    }

    pub fn get_group(&self) -> Vec<usize> {
        self.faces
            .iter_with_empty()
            .filter_map(|f| f.as_ref().map(|f| self.clusters[f.cluster_idx].group_index))
            .collect()
    }

    pub fn iter_edge_loop(&self, e: EdgeID) -> EdgeIter {
        // emit 3 edges and a none
        EdgeIter::new(self, e, Some(e), 3)
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
    pub fn age(&mut self) {
        self.edges.iter_mut_with_empty().for_each(|e| {
            if let Some(e) = e {
                e.age += 1;
            }
        });
    }
    pub fn max_edge_age(&self) -> u32 {
        self.edges.iter().map(|e| e.age).max().unwrap()
    }
    pub fn avg_edge_age(&self) -> f32 {
        self.edges.iter().map(|e| e.age).sum::<u32>() as f32 / self.edge_count() as f32
    }
}
#[cfg(test)]
pub mod test {
    use super::*;
    use std::{
        collections::{HashMap, HashSet},
        error::Error,
    };

    use common::graph;
    use metis::PartitioningConfig;

    use crate::mesh::winged_mesh::WingedMesh;
    pub const TEST_MESH_HIGH: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere.glb";

    pub const TEST_MESH_MONK: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\monk.glb";
    pub const TEST_MESH_MID: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere_low.glb";

    pub const TEST_MESH_LOW: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\torus.glb";

    pub const TEST_MESH_PLANE: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\plane_high.glb";
    pub const TEST_MESH_CIRC: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\circle.glb";
    pub const TEST_MESH_CIRC_LOW: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\circle_low.glb";
    pub const TEST_MESH_CONE: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\cone.glb";
    pub const TEST_MESH_PLANE_LOW: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\plane.glb";

    /// Extra assertion methods for test environment
    impl WingedMesh {
        pub fn assert_valid(&self) -> anyhow::Result<()> {
            for (i, _) in self.iter_faces() {
                self.assert_face_valid(i).context("Invalid Mesh")?;
            }
            for (eid, _) in self.iter_edges() {
                self.assert_edge_valid(eid).context("Invalid Mesh")?;
            }
            for (vid, _) in self.iter_verts() {
                self.assert_vertex_valid(vid).context("Invalid Mesh")?;
            }
            Ok(())
        }

        pub fn assert_face_valid(&self, fid: FaceID) -> anyhow::Result<()> {
            let f = &self.get_face(fid);
            let tri: Vec<_> = self.iter_edge_loop(f.edge).collect();

            if tri.len() > 3 {
                Err(MeshError::InvalidTri(fid)).context("Tri has >3 edges")?;
            }
            if tri.len() < 3 {
                Err(MeshError::InvalidTri(fid)).context("Tri has <3 edges")?;
            }

            for &e in &tri {
                self.assert_edge_valid(e)
                    .context(MeshError::InvalidTri(fid))?;

                if let Some(t) = self.get_edge(e).twin {
                    if tri.contains(&t) {
                        Err(MeshError::InvalidTri(fid)).context("Tri neighbours itself")?
                    }
                }
            }
            Ok(())
        }

        pub fn assert_edge_valid(&self, eid: EdgeID) -> anyhow::Result<()> {
            let edge = self.try_get_edge(eid)?;

            if let Some(t) = edge.twin {
                self.try_get_edge(t).context(MeshError::InvalidTwin(eid))?;
            }

            let v = self
                .try_get_vert(edge.vert_origin)
                .context(MeshError::InvalidEdge(eid))?;

            if !v.outgoing_edges().contains(&eid) {
                Err(MeshError::InvalidEdge(eid))
                    .context(MeshError::InvalidVertex(edge.vert_origin))
                    .context(
                        "Vertex does not contain reference to edge that is it's source in outgoing",
                    )?;
            }

            self.try_get_edge(edge.edge_left_ccw)?;

            self.try_get_edge(edge.edge_left_cw)?;

            let (src, dest) = eid.src_dst(self)?;

            if src == dest {
                Err(MeshError::SingletonEdge(eid, src))?;
            }

            self.assert_vertex_valid(edge.vert_origin)?;
            self.assert_vertex_valid(self.get_edge(edge.edge_left_ccw).vert_origin)?;
            self.assert_vertex_valid(self.get_edge(edge.edge_left_cw).vert_origin)?;

            Ok(())
        }

        pub fn assert_vertex_valid(&self, vid: VertID) -> anyhow::Result<()> {
            self.try_get_vert(vid)?;

            let mut dests = HashMap::new();
            let mut origs = HashMap::new();

            for &eid in self.get_vert(vid).outgoing_edges() {
                let (orig, dest) = eid.src_dst(self).context(MeshError::InvalidVertex(vid))?;

                if orig != vid {
                    return Err(MeshError::InvalidVertex(vid))
                        .context("Invalid vertex edge source loop");
                }

                if let Some(other) = dests.insert(dest, eid) {
                    // Go back and find the other edge with this destination

                    return Err(MeshError::InvalidVertex(vid))
                        .context(MeshError::DuplicateEdges(
                            vid,
                            eid,
                            other,
                            self.get_edge(eid).clone(),
                            self.get_edge(other).clone(),
                        ))
                        .with_context(|| {
                            format!("Vert has outgoing edges with duplicate destinations. ",)
                        });
                }
            }
            for &eid in self.get_vert(vid).incoming_edges() {
                let (orig, dest) = eid.src_dst(self).context(MeshError::InvalidVertex(vid))?;

                if dest != vid {
                    return Err(MeshError::InvalidVertex(vid))
                        .context("Invalid vertex edge dest loop");
                }

                if let Some(_other) = origs.insert(orig, eid) {
                    return Err(MeshError::InvalidVertex(vid))
                        .context("Vert has incoming edges with duplicate sources");
                }
            }
            Ok(())
        }
        /// Find the inner boundary of a set of faces. Quite simple - record all edges we have seen, and any twins for those.
        /// Inner boundary as will not include edges from an edge of the mesh, as these have no twins.
        pub fn face_boundary(&self, faces: &Vec<&Face>) -> HashSet<EdgeID> {
            let mut unseen_edges = HashSet::<EdgeID>::with_capacity(faces.len() * 3);

            for face in faces {
                for edge in self.iter_edge_loop(face.edge) {
                    if let Some(t) = self.get_edge(edge).twin {
                        unseen_edges.insert(t);
                    }
                }
            }

            let all_unseen_edges = unseen_edges.clone();
            unseen_edges
                .retain(|&edge| !all_unseen_edges.contains(&self.get_edge(edge).twin.unwrap()));

            unseen_edges
        }
    }

    #[test]
    pub fn test_continued_validity() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let (mut mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_MID);

        mesh.assert_valid().unwrap();

        mesh.partition_within_groups(test_config, None, Some(60))?;

        mesh.assert_valid().unwrap();

        mesh.group(test_config, &verts)?;

        mesh.assert_valid().unwrap();

        mesh.partition_within_groups(test_config, Some(2), None)?;

        println!("{} {}", mesh.cluster_count(), mesh.groups.len());

        assert!(mesh.cluster_count() <= mesh.groups.len() * 2);

        mesh.assert_valid().unwrap();

        Ok(())
    }

    #[test]
    pub fn test_faces_boundary() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let mesh = TEST_MESH_MID;
        let (mut mesh, _verts, norms) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_within_groups(test_config, None, Some(60))?;

        let mut boundary_face_ratio = 0.0;
        for pi in 0..mesh.clusters.len() {
            // Assert that new parts and the parts in group have the same boundary

            let faces = mesh.faces.iter().filter(|f| pi == f.cluster_idx).collect();

            let boundary = mesh.face_boundary(&faces);

            for &e in &boundary {
                assert!(
                    !faces.contains(&&mesh.get_face(mesh.get_edge(e).face)),
                    "'Boundary' contains edge inside itself"
                )
            }

            boundary_face_ratio += faces.len() as f32 / boundary.len() as f32;
        }

        boundary_face_ratio /= mesh.clusters.len() as f32;

        println!("Average partition face count / boundary length: {boundary_face_ratio}");

        // For the sphere, correct ratio around this
        assert!(boundary_face_ratio > 3.3 && boundary_face_ratio < 3.4);

        Ok(())
    }

    // #[test]
    // pub fn test_group_repartitioning() -> Result<(), Box<dyn Error>> {
    //     let test_config = &PartitioningConfig {
    //         method: metis::PartitioningMethod::MultilevelKWay,
    //         force_contiguous_partitions: true,
    //         minimize_subgraph_degree: Some(true),
    //         ..Default::default()
    //     };

    //     let mesh = TEST_MESH_MID;
    //     let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

    //     // Apply primary partition, that will define the lowest level clusterings
    //     mesh.partition_within_groups(test_config, None)?;

    //     mesh.group(test_config, &verts)?;

    //     let old_faces = mesh.faces.clone();

    //     // Old parts must have no child - there is no LOD-1
    //     for p in &mesh.partitions {
    //         assert!(p.child_group_index.is_none());
    //     }

    //     mesh.partition_within_groups(test_config, Some(2))?;

    //     // assert that the group boundaries are the same

    //     // Same group indices, new set of partitions per group
    //     let mut new_part_groups = vec![Vec::new(); mesh.groups.len()];

    //     // new parts must have no group - there is no grouping assigned yet
    //     for (i, p) in mesh.partitions.iter().enumerate() {
    //         assert_eq!(p.group_index, usize::MAX);
    //         assert!(p.child_group_index.is_some());

    //         let g_i = p.child_group_index.unwrap();

    //         new_part_groups[g_i].push(i);
    //     }

    //     let avg_size = (new_part_groups.iter().map(|l| l.len()).sum::<usize>() as f32)
    //         / (new_part_groups.len() as f32);

    //     println!(
    //         "Test has grouped partitions into {} groups, with {} partitions average",
    //         new_part_groups.len(),
    //         avg_size
    //     );

    //     let mut boundary_face_ratio = 0.0;

    //     for (gi, group) in mesh.groups.iter().enumerate() {
    //         let new_parts = &new_part_groups[gi];
    //         let old_parts = &group.partitions;

    //         // Assert that new parts and the parts in group have the same boundary

    //         let old_faces = old_faces
    //             .iter()
    //             .filter_map(|e| e.as_ref())
    //             .filter(|&f| old_parts.contains(&f.part))
    //             .collect();
    //         let new_faces = mesh
    //             .faces
    //             .iter()
    //             .filter_map(|e| e.as_ref())
    //             .filter(|&f| new_parts.contains(&f.part))
    //             .collect();

    //         let old_boundary = mesh.face_boundary(&old_faces);
    //         let new_boundary = mesh.face_boundary(&new_faces);

    //         boundary_face_ratio += old_faces.len() as f32 / old_boundary.len() as f32;

    //         assert_eq!(
    //             old_faces.len(),
    //             new_faces.len(),
    //             "Repartitioning of group without remesh changed face count"
    //         );
    //         assert_eq!(
    //             old_boundary.len(),
    //             new_boundary.len(),
    //             "Repartitioning of group changed group boundary (len)"
    //         );

    //         assert!(
    //             old_boundary.iter().all(|e| new_boundary.contains(e)),
    //             "Repartitioning of group changed group boundary (edgeid)"
    //         );
    //     }

    //     boundary_face_ratio /= mesh.groups.len() as f32;

    //     println!("Average group face count / boundary length: {boundary_face_ratio}");

    //     Ok(())
    // }

    #[test]
    fn mesh_stats_readout() {
        let (mut mesh, _verts, norms) = WingedMesh::from_gltf(TEST_MESH_CONE);

        let mut avg_outgoing = 0.0;
        let mut avg_incoming = 0.0;

        for (outc, inc) in mesh
            .verts
            .iter()
            .map(|v| (v.outgoing_edges().len(), v.incoming_edges().len()))
        {
            avg_outgoing += outc as f32;
            avg_incoming += inc as f32;
        }
        avg_outgoing /= mesh.verts.len() as f32;
        avg_incoming /= mesh.verts.len() as f32;

        println!("Average Outgoing: {avg_outgoing}, Average Incoming: {avg_incoming}");
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        mesh.partition_full_mesh(test_config, mesh.verts.len().div_ceil(60) as u32)
            .unwrap();

        let mut embed_prop = 0.0;

        for vid in 0..mesh.verts.len() {
            embed_prop +=
                if mesh.try_get_vert(VertID(vid)).is_ok() && VertID(vid).is_group_embedded(&mesh) {
                    1.0
                } else {
                    0.0
                };
        }
        embed_prop /= mesh.verts.len() as f32;
        println!("Embedded Proportion: {embed_prop}");
    }

    #[test]
    fn test_meshes_valid() {
        let mesh_names = ["../../assets/rock.glb", "../../assets/dragon_1m.glb"];

        for mesh_name in &mesh_names {
            println!("Loading from gltf!");
            let (mesh, _, norms) = WingedMesh::from_gltf(mesh_name);

            mesh.assert_valid().unwrap();
        }
    }

    #[test]
    fn test_reduce_contiguous() {
        let (mut mesh, verts, norms) = WingedMesh::from_gltf(TEST_MESH_CONE);

        println!("Asserting contiguous");
        // WE know the circle is contiguous
        //assert_contiguous_graph(&working_mesh.generate_face_graph());

        let mut quadrics = mesh.create_quadrics(&verts);

        let _e = match mesh.reduce_within_groups(&verts, &mut quadrics, &[mesh.face_count() / 4]) {
            Ok(e) => e,
            Err(e) => {
                panic!(
                    "Experience error {} with reducing, exiting early with what we have",
                    e
                );
            }
        };
        println!("Asserting contiguous");
        // It should still be contiguous
        graph::assert_graph_contiguous(&mesh.generate_face_graph());
    }

    #[test]
    pub fn test_partition_contiguity() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: Some(true),
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let (mut mesh, _verts, norms) = WingedMesh::from_gltf(TEST_MESH_LOW);

        // Apply primary partition, that will define the lowest level clusterings
        for i in 9..50 {
            println!("{i}");

            mesh.partition_full_mesh(test_config, i)?;

            println!("Partitioned");
            let mut graph = mesh.generate_face_graph();

            graph.retain_edges(|g, e| {
                let (v1, v2) = g.edge_endpoints(e).unwrap();

                let p1 = mesh.get_face(*g.node_weight(v1).unwrap()).cluster_idx;
                let p2 = mesh.get_face(*g.node_weight(v2).unwrap()).cluster_idx;

                p1 == p2
            });

            let mut work = petgraph::algo::DfsSpace::default();

            println!("Testing continuity");

            for i in graph.node_indices() {
                for j in graph.node_indices() {
                    if graph.node_weight(j) == graph.node_weight(i) && i > j {
                        assert!(petgraph::algo::has_path_connecting(
                            &graph,
                            i,
                            j,
                            Some(&mut work)
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
