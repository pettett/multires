use anyhow::Context;
use common::TriMesh;
use glam::Vec3;

use std::{
    collections::{HashMap, HashSet},
    fs,
};

use super::{
    cluster_info::ClusterInfo,
    edge::{EdgeID, EdgeIter, HalfEdge},
    face::{Face, FaceID},
    group_info::GroupInfo,
    pidge::Pidge,
    vertex::{VertID, Vertex},
};

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("Invalid triangle {0:?}")]
    InvalidFace(FaceID),
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

    #[error("Triangle Already Exists")]
    EdgeExists(EdgeID),
}

//Definition 6: A cut in the DAG is a subset of the tree such that for every node Ci all ancestors
//of Ci are in the cut as well. The front of the cut is the set of arcs that connect a node in the cut
//to a node outside.

#[derive(Debug, Clone)]
pub struct HalfEdgeMesh {
    faces: Pidge<FaceID, Face>,
    edges: Pidge<EdgeID, HalfEdge>,
    verts: Pidge<VertID, Vertex>,
    pub clusters: Vec<ClusterInfo>,
    pub groups: Vec<GroupInfo>,
}

struct TwinLoop<'a> {
    mesh: &'a HalfEdgeMesh,
    first: EdgeID,
    current: EdgeID,
}

impl<'a> Iterator for TwinLoop<'a> {
    type Item = EdgeID;

    fn next(&mut self) -> Option<Self::Item> {
        let twin = self.current.edge(self.mesh).twin;

        if let Some(next) = twin {
            if next == self.first {
                None
            } else {
                self.current = next;
                Some(next)
            }
        } else {
            None
        }
    }
}

impl HalfEdgeMesh {
    pub fn new(faces: usize, verts: usize) -> Self {
        Self {
            faces: Pidge::with_capacity(faces),
            edges: Pidge::with_capacity(faces * 3),
            verts: Pidge::with_capacity(verts),
            groups: vec![],
            clusters: vec![ClusterInfo::default()],
        }
    }

    pub fn try_get_face_mut(&mut self, fid: FaceID) -> Result<&mut Face, MeshError> {
        self.faces
            .try_get_mut(fid)
            .ok_or(MeshError::InvalidFace(fid))
    }

    pub fn try_get_edge(&self, eid: EdgeID) -> Result<&HalfEdge, MeshError> {
        self.edges.try_get(eid).ok_or(MeshError::InvalidEdge(eid))
    }

    pub fn try_get_edge_mut(&mut self, eid: EdgeID) -> Result<&mut HalfEdge, MeshError> {
        self.edges
            .try_get_mut(eid)
            .ok_or(MeshError::InvalidEdge(eid))
    }

    pub fn try_get_vert(&self, vid: VertID) -> Result<&Vertex, MeshError> {
        self.verts.try_get(vid).ok_or(MeshError::InvalidVertex(vid))
    }

    pub fn get_vert_or_default(&mut self, vid: VertID) -> &mut Vertex {
        if !self.verts.slot_full(vid) {
            self.verts.insert(vid, Vertex::default());
        }
        vid.vert_mut(self)
    }

    pub fn iter_edge_loop(&self, e: EdgeID) -> impl Iterator<Item = EdgeID> + '_ {
        // emit 3 edges and a none
        EdgeIter::new(self, e, Some(e), 3)
    }

    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
    pub fn age(&mut self) {
        self.edges.iter_mut().for_each(|e| {
            e.age += 1;
        });
    }
    pub fn max_edge_age(&self) -> u32 {
        self.edges.iter().map(|e| e.age).max().unwrap_or_default()
    }
    pub fn avg_edge_age(&self) -> f32 {
        self.edges.iter().map(|e| e.age).sum::<u32>() as f32 / self.edges().len() as f32
    }

    pub fn edge_sqr_length(&self, edge: EdgeID, verts: &[Vec3]) -> f32 {
        let e = edge.edge(self);
        return verts[e.vert_origin.id() as usize]
            .distance_squared(verts[e.edge_back_cw.edge(self).vert_origin.id() as usize]);
    }

    pub fn triangle_from_face(&self, face: &Face) -> [u32; 3] {
        let first = face.edge.edge(self);
        let next = first.edge_next_ccw.edge(self);
        let back: &HalfEdge = first.edge_back_cw.edge(self);

        [
            back.vert_origin.into(),
            first.vert_origin.into(),
            next.vert_origin.into(),
        ]
    }

    /// Wrapper around [TriMesh::from_gltf] and [WingedMesh::from_tris]
    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> (Self, TriMesh) {
        Self::from_gltf_enable_loops(path, false)
    }

    /// Wrapper around [TriMesh::from_gltf] and [WingedMesh::from_tris]
    pub fn from_gltf_twin_loops(path: impl AsRef<std::path::Path>) -> (Self, TriMesh) {
        Self::from_gltf_enable_loops(path, true)
    }

    pub fn from_gltf_enable_loops(
        path: impl AsRef<std::path::Path>,
        allow_twin_loop: bool,
    ) -> (Self, TriMesh) {
        let tri_mesh = TriMesh::from_gltf(&path).unwrap();
        println!(
            "Loading GLTF {:?} with {} faces:",
            fs::canonicalize(path).unwrap(),
            tri_mesh.indices.len() / 3
        );
        (Self::from_tris(&tri_mesh, allow_twin_loop), tri_mesh)
    }

    pub fn from_tris(tri_mesh: &TriMesh, allow_twin_loop: bool) -> Self {
        let face_count = tri_mesh.indices.len() / 3;
        let mut mesh = HalfEdgeMesh::new(face_count, tri_mesh.verts.len());

        //#[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(face_count as u64);

        let mut current = FaceID(0);

        for i in 0..face_count {
            let a = tri_mesh.indices[i * 3 + 0] as usize;
            let b = tri_mesh.indices[i * 3 + 1] as usize;
            let c = tri_mesh.indices[i * 3 + 2] as usize;

            if a == b || b == c || a == c {
                println!("Discarding 0 area triangle");
            } else {
                match mesh.add_tri(current, a.into(), b.into(), c.into(), allow_twin_loop) {
                    Ok(_) => {
                        current.0 += 1;
                    }
                    Err(_) => println!("Attempted to add potentially duplicated triangle"),
                };
            }

            //#[cfg(feature = "progress")]
            bar.inc(1);
        }
        //#[cfg(feature = "progress")]
        bar.finish();

        mesh
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        self.try_get_vert(a)
            .as_ref()
            .map(|v| {
                v.outgoing_edges()
                    .iter()
                    .filter(|&&p| {
                        let e = p.edge(self);

                        assert_eq!(e.vert_origin, a);

                        e.dst(self).unwrap() == b
                    })
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
        edge_back_cw: EdgeID,
        edge_next_ccw: EdgeID,
        allow_twin_loop: bool,
    ) -> Result<(), MeshError> {
        let twin = if !allow_twin_loop {
            // assert!(
            //     self.find_edge(orig, dest).is_none(),
            //     "Adding duplicate edge to Mesh"
            // );

            self.find_edge(dest, orig)
        } else {
            // Set the "twin" to an existing mesh going this way, if it exists. Try to keep things contiguous
            self.find_edge(dest, orig)
                .or_else(|| self.find_edge(orig, dest))
        };

        let e = HalfEdge {
            vert_origin: orig,
            face,
            edge_back_cw,
            edge_next_ccw,
            twin,
            age: 0,
        };

        // Warning: cursed code
        if let Some(twin_eid) = twin {
            if allow_twin_loop {
                if twin_eid.edge(self).twin.is_none() {
                    // We are safe from the madness
                    twin_eid.edge_mut(self).twin = Some(eid)
                } else {
                    let first_twin = twin_eid;
                    let mut current = twin_eid;
                    // Find the end of the twin loop
                    loop {
                        let current_twin = current.edge(self).twin.unwrap();

                        if current_twin == first_twin {
                            current.edge_mut(self).twin = Some(eid);
                            break;
                        } else {
                            current = current_twin
                        }
                    }
                }
            } else {
                twin_eid.edge_mut(self).twin = Some(eid)
            }
        }

        self.get_vert_or_default(orig).add_outgoing(eid);

        self.get_vert_or_default(dest).add_incoming(eid);

        self.edges_mut().insert(eid, e);

        return Ok(());
    }

    pub fn twin_loop(&self, eid: EdgeID) -> impl Iterator<Item = EdgeID> + '_ {
        TwinLoop {
            mesh: self,
            first: eid,
            current: eid,
        }
    }

    pub fn assert_twin_loop(&self, eid: EdgeID) {
        let mut seen = Vec::new();

        for current in self.twin_loop(eid) {
            if seen.contains(&current) {
                panic!("Twin loop invalid");
            }

            seen.push(current);
        }

        assert!(!seen.contains(&eid));

        let (src, dst) = eid.src_dst(self).unwrap();

        for &o in dst.vert(self).incoming_edges() {
            if o != eid && o.edge(self).vert_origin == src {
                assert!(seen.contains(&o))
            }
        }

        for &o in src.vert(self).incoming_edges() {
            if o.edge(self).vert_origin == dst {
                assert!(seen.contains(&o))
            }
        }
    }

    pub fn add_tri(
        &mut self,
        f: FaceID,
        a: VertID,
        b: VertID,
        c: VertID,
        allow_twin_loop: bool,
    ) -> Result<(), MeshError> {
        if !allow_twin_loop {
            if self.find_edge(a, b).is_some()
                || self.find_edge(b, c).is_some()
                || self.find_edge(c, a).is_some()
            {
                if let Some(e) = self.find_edge(a, b) {
                    return Err(MeshError::EdgeExists(e));
                }
                if let Some(e) = self.find_edge(b, c) {
                    return Err(MeshError::EdgeExists(e));
                }
                if let Some(e) = self.find_edge(c, a) {
                    return Err(MeshError::EdgeExists(e));
                }
            }
        }

        let edge_center = (f.0 * 3 + 0).into();
        let edge_back_cw = (f.0 * 3 + 1).into();
        let edge_next_ccw = (f.0 * 3 + 2).into();

        //     b   e    a
        //     next  prev
        //         c

        self.add_half_edge(
            a,
            b,
            f,
            edge_center,
            edge_back_cw,
            edge_next_ccw,
            allow_twin_loop,
        )?;
        self.add_half_edge(
            b,
            c,
            f,
            edge_next_ccw,
            edge_center,
            edge_back_cw,
            allow_twin_loop,
        )?;
        self.add_half_edge(
            c,
            a,
            f,
            edge_back_cw,
            edge_next_ccw,
            edge_center,
            allow_twin_loop,
        )?;

        self.faces_mut().insert(
            f,
            Face {
                edge: edge_center,
                cluster_idx: 0,
            },
        );

        assert!(self.faces_mut().slot_full(f));

        if allow_twin_loop {
            // assert twin loop is valid
            self.assert_twin_loop(edge_center);
            self.assert_twin_loop(edge_back_cw);
            self.assert_twin_loop(edge_next_ccw);
        }
        Ok(())
    }

    /// Remove every triangle with a `cluster_idx` different to `cluster_idx`
    pub fn filter_tris_by_cluster(&mut self, cluster_idx: usize) -> anyhow::Result<()> {
        let mut remove = Vec::new();

        for (i, f) in self.faces().iter_items() {
            if f.cluster_idx != cluster_idx {
                remove.push(i);
            }
        }

        for i in remove {
            self.wipe_face(i)?;
        }

        Ok(())
    }

    fn neighbour_vertices(&self, v: &Vertex) -> HashSet<VertID> {
        // This is the exact number of neighbours, assuming this is a manifold vertex, otherwise it will still be pretty close.
        let mut neighbours = HashSet::with_capacity(v.incoming_edges().len());

        neighbours.extend(v.incoming_edges().iter().map(|&e| e.edge(self).vert_origin));

        neighbours.extend(
            v.outgoing_edges()
                .iter()
                .map(|&e| e.edge(self).dst(self).unwrap()),
        );

        neighbours
    }

    /// Does this edge have the right amount of shared neighbours for edge collapse
    ///
    /// Logic sourced from https://stackoverflow.com/a/27049418
    pub fn max_one_joint_neighbour_vertices_per_side(&self, eid: EdgeID) -> bool {
        let (src, dst) = eid.src_dst(&self).unwrap();

        let v_src = src.vert(self);
        let has_twin = eid.edge(self).twin.is_some();

        let mut joint_shared_count = 0;

        for &incoming in v_src.incoming_edges() {
            // Count every vertex from the incoming first, as these are slightly quicker to find
            let neighbour = incoming.edge(self).vert_origin;

            if self.find_edge(neighbour, dst).is_some() || self.find_edge(dst, neighbour).is_some()
            {
                joint_shared_count += 1;
            }
        }

        for &outgoing in v_src.outgoing_edges() {
            // Count every none-incoming-accessible vertex afterwards
            if outgoing.edge(self).twin.is_some() {
                // Do not double count vertices that can be reached in two ways
                continue;
            }

            let neighbour = outgoing.edge(self).dst(self).unwrap();

            if self.find_edge(neighbour, src).is_some() {
                println!("ERROR: inconsistency in outgoing/incoming edges - edge exists back but no twin assigned")
            }

            if self.find_edge(neighbour, dst).is_some() || self.find_edge(dst, neighbour).is_some()
            {
                joint_shared_count += 1;
            }
        }

        let new = has_twin && joint_shared_count == 2 || !has_twin && joint_shared_count == 1;

        #[cfg(debug)]
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
        let (vid_orig, vid_dest) = eid.src_dst(self)?;

        #[cfg(debug)]
        {
            if !self.max_one_joint_neighbour_vertices_per_side(eid) {
                Err(MeshError::EdgeCollapse(eid, vid_orig, vid_dest)).context(
                    "Attempting to collapse edge with too many joint neighbours on one side",
                )?;
            }
        }

        let edge = self
            .collapse_tri(eid)
            .context("Failed to collapse main triangle")
            .context(MeshError::EdgeCollapse(eid, vid_orig, vid_dest))?;

        // Assert the vertexes are still valid, which they should be despite some invalid triangles around us
        #[cfg(debug)]
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
        self.wipe_vert(vid_orig, vid_dest);

        #[cfg(debug)]
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
            //assert_eq!(new_graph.edges().len(), previous_graph.edges().len() - 3);

            //for g in self.generate_group_graphs() {
            //    test::assert_contiguous_graph(&g);
            //}
        }

        Ok(())
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
    /// Returns:
    /// 	The halfedge eid collapsed into nothing and removed
    fn collapse_tri(&mut self, eid: EdgeID) -> anyhow::Result<HalfEdge> {
        let edge = self.try_get_edge(eid)?;

        let fid = edge.face;

        let (_f, eids, [e0, e1, e2]) = self.wipe_face(fid)?;

        // we are pinching edge `eid` to nothing, so make the other two edges twins
        let (o1, o0, t) = match eid {
            eid if eid == eids[0] => (e1, e2, e0),
            eid if eid == eids[1] => (e0, e2, e1),
            eid if eid == eids[2] => (e0, e1, e2),
            _ => unreachable!(),
        };

        if let Some(t) = o1.twin {
            t.edge_mut(self).twin = o0.twin;
        }
        if let Some(t) = o0.twin {
            t.edge_mut(self).twin = o1.twin;
        }

        Ok(t)
    }

    pub fn wipe_face(
        &mut self,
        face: FaceID,
    ) -> Result<(Face, [EdgeID; 3], [HalfEdge; 3]), MeshError> {
        #[cfg(debug)]
        {
            self.assert_face_valid(face).unwrap();
        }

        let f = self.faces.wipe(face);
        let edge = self.wipe_edge(f.edge);

        let tri = [f.edge, edge.edge_next_ccw, edge.edge_back_cw];

        let edges = [edge, self.wipe_edge(tri[1]), self.wipe_edge(tri[2])];

        // Remove any references to this triangle
        for i in 0..3 {
            let v_o = edges[i].vert_origin;

            v_o.vert_mut(self).remove_outgoing(tri[i]);
            v_o.vert_mut(self).remove_incoming(edges[i].edge_back_cw)?;
        }

        Ok((f, tri, edges))
    }

    /// Remove an edge from the mesh.
    /// For this to be valid, the cw and ccw edges must also be wiped,
    /// so this function is non public
    fn wipe_edge(&mut self, edge: EdgeID) -> HalfEdge {
        let e = self.edges.wipe(edge);
        // Consistency - neighbours must not reference this

        if let Some(t) = e.twin {
            t.edge_mut(self).twin = None;
        }

        e
    }

    /// Remove the vert `vid`, and shift any edges that reference it to `replacement`.
    pub fn wipe_vert(&mut self, vid: VertID, replacement: VertID) {
        // Main issue is a situation where triangles do not fan around the cake in both directions
        // This will collapse an edge to have dest and source in same position.
        // Because of this, we need to store all ingoing and outgoings per vertex, which isn't the worst in the world
        // Although it would save a bit of memory to just store every fan

        let (_incomings, outgoings) = self.verts.wipe(vid).unpack();

        for outgoing in outgoings {
            let outgoing_edge = outgoing.edge(self);

            // Don't fix invalid edges
            #[cfg(debug)]
            {
                let (orig, dest) = outgoing_edge.src_dst(self).unwrap();
                assert_eq!(orig, vid);
                assert_ne!(dest, replacement);
            }

            let outgoing_prev = outgoing_edge.edge_back_cw;

            outgoing.edge_mut(self).vert_origin = replacement;

            // Moving this origin moves both the start of this edge and the dest of the previous edge
            replacement.vert_mut(self).add_outgoing(outgoing);
            replacement.vert_mut(self).add_incoming(outgoing_prev);

            //TODO: add some tests here to make sure we don't break a triangle

            // Reset their ages, as moved
            outgoing.edge_mut(self).age = 0;
            outgoing_prev.edge_mut(self).age = 0;
        }
    }

    pub fn assert_valid(&self) -> anyhow::Result<()> {
        println!("Validating mesh!");
        // for i in self.faces().iter_keys() {
        //     self.assert_face_valid(i).context("Invalid Mesh")?;
        // }
        for eid in self.edges().iter_keys() {
            self.assert_edge_valid(eid).context("Invalid Mesh")?;
        }
        // for vid in self.verts().iter_keys() {
        //     self.assert_vertex_valid(vid).context("Invalid Mesh")?;
        // }
        Ok(())
    }

    pub fn assert_face_valid(&self, fid: FaceID) -> anyhow::Result<()> {
        let f = fid.face(self);
        let tri: Vec<_> = self.iter_edge_loop(f.edge).collect();

        if tri.len() > 3 {
            Err(MeshError::InvalidFace(fid)).context("Tri has >3 edges")?;
        }
        if tri.len() < 3 {
            Err(MeshError::InvalidFace(fid)).context("Tri has <3 edges")?;
        }

        for &e in &tri {
            self.assert_edge_valid(e)
                .context(MeshError::InvalidFace(fid))?;

            if let Some(t) = e.edge(self).twin {
                if tri.contains(&t) {
                    Err(MeshError::InvalidFace(fid)).context("Tri neighbours itself")?
                }
            }
        }
        Ok(())
    }

    pub fn assert_edge_valid(&self, eid: EdgeID) -> anyhow::Result<()> {
        let edge = self.try_get_edge(eid)?;

        edge.face.face(self);

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

        self.try_get_edge(edge.edge_next_ccw)?;

        self.try_get_edge(edge.edge_back_cw)?;

        let (src, dest) = eid.src_dst(self)?;

        if src == dest {
            Err(MeshError::SingletonEdge(eid, src))?;
        }

        // self.assert_vertex_valid(edge.vert_origin)?;
        // self.assert_vertex_valid(edge.edge_next_ccw.edge(self).vert_origin)?;
        // self.assert_vertex_valid(edge.edge_back_cw.edge(self).vert_origin)?;

        Ok(())
    }

    pub fn assert_vertex_valid(&self, vid: VertID) -> anyhow::Result<()> {
        self.try_get_vert(vid)?;

        let mut dests = HashMap::new();
        let mut origs = HashMap::new();

        for &eid in vid.vert(self).outgoing_edges() {
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
                        eid.edge(self).clone(),
                        other.edge(self).clone(),
                    ))
                    .with_context(|| {
                        format!("Vert has outgoing edges with duplicate destinations. ",)
                    });
            }
        }
        for &eid in vid.vert(self).incoming_edges() {
            let (orig, dest) = eid.src_dst(self).context(MeshError::InvalidVertex(vid))?;

            if dest != vid {
                return Err(MeshError::InvalidVertex(vid)).context("Invalid vertex edge dest loop");
            }

            if let Some(_other) = origs.insert(orig, eid) {
                return Err(MeshError::InvalidVertex(vid))
                    .context("Vert has incoming edges with duplicate sources");
            }
        }
        Ok(())
    }

    pub fn faces_mut(&mut self) -> &mut Pidge<FaceID, Face> {
        &mut self.faces
    }

    pub fn edges_mut(&mut self) -> &mut Pidge<EdgeID, HalfEdge> {
        &mut self.edges
    }

    pub fn verts_mut(&mut self) -> &mut Pidge<VertID, Vertex> {
        &mut self.verts
    }

    pub fn faces(&self) -> &Pidge<FaceID, Face> {
        &self.faces
    }

    pub fn edges(&self) -> &Pidge<EdgeID, HalfEdge> {
        &self.edges
    }

    pub fn verts(&self) -> &Pidge<VertID, Vertex> {
        &self.verts
    }
}
#[cfg(test)]
pub mod test {
    use super::*;
    use std::{collections::HashSet, error::Error};

    use common::graph;
    use metis::PartitioningConfig;

    use crate::mesh::{half_edge_mesh::HalfEdgeMesh, partition::PartitionCount};
    pub const TEST_MESH_HIGH: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere.glb";
    pub const TEST_MESH_DRAGON: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\dragon_high.glb";
    pub const TEST_MESH_MONK: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\monk.glb";
    pub const TEST_MESH_MID: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\sphere_low.glb";

    pub const TEST_MESH_LOW: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\torus.glb";

    pub const TEST_MESH_LOWER: &str =
        "C:\\Users\\maxwe\\OneDriveC\\sync\\projects\\assets\\torus_low.glb";
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
    impl HalfEdgeMesh {
        /// Find the inner boundary of a set of faces. Quite simple - record all edges we have seen, and any twins for those.
        /// Inner boundary as will not include edges from an edge of the mesh, as these have no twins.
        pub fn face_boundary(&self, faces: &Vec<&Face>) -> HashSet<EdgeID> {
            let mut unseen_edges = HashSet::<EdgeID>::with_capacity(faces.len() * 3);

            for face in faces {
                for edge in self.iter_edge_loop(face.edge) {
                    if let Some(t) = edge.edge(self).twin {
                        unseen_edges.insert(t);
                    }
                }
            }

            let all_unseen_edges = unseen_edges.clone();
            unseen_edges.retain(|&edge| !all_unseen_edges.contains(&edge.edge(self).twin.unwrap()));

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

        let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(TEST_MESH_MID);

        mesh.assert_valid().unwrap();

        mesh.group_unity();

        mesh.cluster_within_groups(
            test_config,
            &tri_mesh.verts,
            PartitionCount::MembersPerPartition(60),
        )?;

        mesh.assert_valid().unwrap();

        mesh.group(test_config)?;

        mesh.assert_valid().unwrap();

        mesh.cluster_within_groups(test_config, &tri_mesh.verts, PartitionCount::Partitions(2))?;

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
        let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(mesh);

        mesh.group_unity();
        // Apply primary partition, that will define the lowest level clusterings
        mesh.cluster_within_groups(
            test_config,
            &tri_mesh.verts,
            PartitionCount::MembersPerPartition(60),
        )?;

        let mut boundary_face_ratio = 0.0;
        for pi in 0..mesh.clusters.len() {
            // Assert that new parts and the parts in group have the same boundary

            let faces = mesh.faces.iter().filter(|f| pi == f.cluster_idx).collect();

            let boundary = mesh.face_boundary(&faces);

            for &e in &boundary {
                assert!(
                    !faces.contains(&e.edge(&mesh).face.face(&mesh)),
                    "'Boundary' contains edge inside itself"
                )
            }

            boundary_face_ratio += faces.len() as f32 / boundary.len() as f32;
        }

        boundary_face_ratio /= mesh.clusters.len() as f32;

        println!("Average partition face count / boundary length: {boundary_face_ratio}");

        // For the sphere, correct ratio around this
        assert!(boundary_face_ratio > 3.3 && boundary_face_ratio < 3.6);

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
        let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(TEST_MESH_CONE);

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

        mesh.cluster_full_mesh(
            test_config,
            mesh.verts.len().div_ceil(60) as u32,
            &tri_mesh.verts,
        )
        .unwrap();

        mesh.group_unity();

        let mut embed_prop = 0.0;

        for vid in 0..mesh.verts.len() {
            embed_prop += if mesh.try_get_vert(VertID::new(vid as _)).is_ok()
                && VertID::new(vid as _).is_group_embedded(&mesh)
            {
                1.0
            } else {
                0.0
            };
        }
        embed_prop /= mesh.verts.len() as f32;
        println!("Embedded Proportion: {embed_prop}");
    }

    #[test]
    fn test_reduce_contiguous() {
        let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(TEST_MESH_CONE);

        println!("Asserting contiguous");
        // WE know the circle is contiguous
        //assert_contiguous_graph(&working_mesh.generate_face_graph());

        let mut quadrics = mesh.create_quadrics(&tri_mesh.verts);

        mesh.group_unity();

        let _e = match mesh.reduce_within_groups(
            &tri_mesh.verts,
            &mut quadrics,
            &[mesh.faces().len() / 4],
        ) {
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
        let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(TEST_MESH_LOW);

        // Apply primary partition, that will define the lowest level clusterings
        for i in 9..50 {
            println!("{i}");

            mesh.cluster_full_mesh(test_config, i, &tri_mesh.verts)?;

            println!("Partitioned");
            let mut graph = mesh.generate_face_graph();

            graph.retain_edges(|g, e| {
                let (v1, v2) = g.edge_endpoints(e).unwrap();

                let p1 = g.node_weight(v1).unwrap().face(&mesh).cluster_idx;
                let p2 = g.node_weight(v2).unwrap().face(&mesh).cluster_idx;

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
