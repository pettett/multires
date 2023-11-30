use common::{tri_mesh::TriMesh, BoundingSphere, GroupInfo, PartitionInfo};
use glam::{Vec3, Vec4, Vec4Swizzles};
use idmap::IntegerId;
use metis::{idx_t, PartitioningConfig, PartitioningError};
use petgraph::Graph;
use std::{collections::HashMap, fs};

use super::{
    iter::EdgeIter,
    vertex::{VertID, Vertex},
};

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error("Invalid edge")]
    InvalidEdge,
    #[error("Invalid edge")]
    InvalidVertex,
    #[error("Cw piece of edge is invalid")]
    InvalidCwEdge,
    #[error("CCw piece of edge is invalid")]
    InvalidCCwEdge,
    #[error("Ran out of valid edges while reducing")]
    OutOfEdges,
}

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
    pub fn center(&self, mesh: &WingedMesh, verts: &[Vec4]) -> Vec4 {
        let mut c = Vec4::ZERO;

        for e in mesh.iter_edge_loop(mesh.faces[self].edge) {
            c += verts[mesh.edges[e].vert_origin.0];
        }

        c / 3.0
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

    pub age: u32,

    pub twin: Option<EdgeID>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Face {
    pub edge: EdgeID,
    pub part: usize,
}

#[derive(Debug, Clone)]
pub struct WingedMesh {
    pub faces: idmap::DirectIdMap<FaceID, Face>,
    pub edges: idmap::DirectIdMap<EdgeID, HalfEdge>,
    pub verts: idmap::DirectIdMap<VertID, Vertex>,
    pub partitions: Vec<PartitionInfo>,
    pub groups: Vec<GroupInfo>,
}

impl WingedMesh {
    pub fn new(faces: usize, verts: usize) -> Self {
        Self {
            faces: idmap::DirectIdMap::with_capacity_direct(faces),
            edges: idmap::DirectIdMap::with_capacity_direct(faces * 3),
            verts: idmap::DirectIdMap::with_capacity_direct(verts),
            groups: vec![],
            partitions: vec![PartitionInfo {
                child_group_index: None,
                group_index: 0,
                tight_bound: BoundingSphere::default(),
            }],
        }
    }

    pub fn edge_sqr_length(&self, edge: EdgeID, verts: &[Vec3]) -> f32 {
        let e = &self.edges[edge];
        return verts[e.vert_origin.0]
            .distance_squared(verts[self.edges[e.edge_left_cw].vert_origin.0]);
    }

    pub fn triangle_from_face(&self, face: &Face) -> [usize; 3] {
        let verts: Vec<usize> = self
            .iter_edge_loop(face.edge)
            .map(|e| self.edges[e].vert_origin.into())
            .collect();

        assert_eq!(verts.len(), 3);
        [verts[0], verts[1], verts[2]]
    }

    pub fn from_gltf(path: impl AsRef<std::path::Path>) -> (Self, Box<[Vec4]>) {
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

        (mesh, tri_mesh.verts)
    }

    fn find_edge(&self, a: VertID, b: VertID) -> Option<EdgeID> {
        self.verts
            .get(&a)
            .map(|v| {
                v.outgoing_edges()
                    .iter()
                    .filter(|&&p| self.edges[self.edges[p].edge_left_cw].vert_origin == b)
                    .next()
                    .copied()
            })
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
            self.edges[twin_eid].twin = Some(eid);
        }

        self.verts
            .entry(orig)
            .or_insert(Vertex::default())
            .add_outgoing(eid);

        self.verts
            .entry(dest)
            .or_insert(Vertex::default())
            .add_incoming(eid);

        assert!(self.edges.insert(eid, e).is_none());
    }

    pub fn add_tri(&mut self, f: FaceID, a: VertID, b: VertID, c: VertID) {
        let iea = EdgeID(f.0 * 3 + 0);
        let ieb = EdgeID(f.0 * 3 + 1);
        let iec = EdgeID(f.0 * 3 + 2);

        self.add_half_edge(a, b, f, iea, ieb, iec);
        self.add_half_edge(b, c, f, ieb, iec, iea);
        self.add_half_edge(c, a, f, iec, iea, ieb);

        assert!(self.faces.insert(f, Face { edge: iea, part: 0 }).is_none());
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
    fn collapse_tri(&mut self, eid: EdgeID) {
        assert!(self.edges.contains_key(eid));

        #[cfg(test)]
        {
            self.assert_face_valid(self.edges[eid].face);
        }

        let tri = self.iter_edge_loop(eid).collect::<Vec<_>>();
        self.faces.remove(self.edges[eid].face);

        assert_eq!(tri[0], eid);
        // we are pinching edge to nothing, so make the other two edges twins
        if let Some(t) = self.edges[tri[1]].twin {
            self.edges[t].twin = self.edges[tri[2]].twin;
        }
        if let Some(t) = self.edges[tri[2]].twin {
            self.edges[t].twin = self.edges[tri[1]].twin;
        }

        // Unlikely this will ever matter, but preserve consistency of mesh, as this edge is to be removed.
        if let Some(t) = self.edges[eid].twin {
            self.edges[t].twin = None;
        }

        // Remove any last references to this triangle
        for &e in &tri {
            self.verts[self.edges[e].vert_origin].remove_outgoing(e);
            self.verts[self.edges[e].vert_origin].remove_incoming(self.edges[e].edge_left_ccw);
            self.edges.remove(e);
        }
    }

    /// Collapse an edge so it no longer exists, the source vertex is no longer referenced,
    /// 	A
    ///   /	^ \
    ///  C	|  D
    ///   \	| /
    /// 	B
    ///
    pub fn collapse_edge(&mut self, eid: EdgeID) {
        //    println!("Collapsing edge {eid:?}");

        //self.assert_valid();

        #[cfg(test)]
        {
            assert!(self.edges.contains_key(eid));
        }
        //#[cfg(test)]
        //let previous_graph = self.generate_face_graph();

        let edge = self.edges[eid].clone();

        let vb = edge.vert_origin;
        let va = self.edges[edge.edge_left_cw].vert_origin;

        self.collapse_tri(eid);

        #[cfg(test)]
        {
            self.assert_vertex_valid(va);
            self.assert_vertex_valid(vb);
        }

        if let Some(e0t) = edge.twin {
            self.collapse_tri(e0t);
        }

        // Remove `vert_origin`

        // Main issue is a situation where triangles do not fan around the cake in both directions
        // This will collapse an edge to have dest and source in same position.
        // Because of this, we need to store all ingoings and outgoings per vertex, which isn't the worst in the world
        // Although it would save a bit of memory to just store every fan
        let b_outgoings = self.verts[vb].outgoing_edges().to_vec();

        for b_outgoing in b_outgoings {
            // Dont fix invalid edges
            assert_eq!(self.edges[b_outgoing].vert_origin, vb);

            let b_outgoing_ccw = self.edges[b_outgoing].edge_left_ccw;

            self.edges[b_outgoing].vert_origin = va;

            // Moving this origin moves both the start of this edge and the dest of the previous edge
            self.verts[va].add_outgoing(b_outgoing);
            self.verts[va].add_incoming(b_outgoing_ccw);

            // Reset their ages, as moved
            self.edges[b_outgoing].age = 0;
            self.edges[b_outgoing_ccw].age = 0;
        }

        self.verts.remove(&vb);

        #[cfg(test)]
        {
            self.assert_vertex_valid(va);

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
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
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

    pub fn iter_edge_loop(&self, e: EdgeID) -> EdgeIter {
        // emit 3 edges and a none
        EdgeIter::new(self, e, Some(e), 3)
    }

    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }
    pub fn age(&mut self) {
        self.edges.values_mut().for_each(|e| e.age += 1)
    }
    pub fn max_edge_age(&self) -> u32 {
        self.edges.values().map(|e| e.age).max().unwrap()
    }
    pub fn avg_edge_age(&self) -> f32 {
        self.edges.values().map(|e| e.age).sum::<u32>() as f32 / self.edges.len() as f32
    }
}
#[cfg(test)]
pub mod test {
    use super::*;
    use std::{collections::HashSet, error::Error};

    use metis::PartitioningConfig;

    use crate::{
        apply_simplification, group_and_partition_and_simplify, group_and_partition_full_res,
        mesh::{graph::test::assert_contiguous_graph, winged_mesh::WingedMesh},
    };
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
        pub fn assert_valid(&self) {
            for (&i, f) in self.faces.iter() {
                self.assert_face_valid(i);
            }
            for (&eid, edge) in &self.edges {
                self.assert_edge_valid(eid, "Invalid edge in array");
            }
            for (&vid, vert) in &self.verts {
                self.assert_vertex_valid(vid);
            }
        }

        pub fn assert_face_valid(&self, fid: FaceID) {
            let f = &self.faces[fid];
            let tri: Vec<_> = self.iter_edge_loop(f.edge).collect();

            assert_eq!(tri.len(), 3);

            for &e in &tri {
                assert!(self.edges.contains_key(e));

                self.assert_edge_valid(e, "Invalid Edge in Face");

                if let Some(t) = self.edges[e].twin {
                    assert!(self.edges.contains_key(t));
                    assert!(
                        !tri.contains(&t),
                        "Tri neighbours itself, total tri count: {}",
                        self.faces.len()
                    );
                }
            }
        }

        pub fn assert_edge_valid(&self, eid: EdgeID, msg: &'static str) {
            let edge = &self.edges[eid];
            if let Some(t) = edge.twin {
                assert!(self.edges.contains_key(t));
            }
            assert!(self.verts.contains_key(edge.vert_origin), "{}", msg);
            assert!(
                self.verts[edge.vert_origin].outgoing_edges().contains(&eid),
                "{}",
                msg
            );
            assert!(self.edges.contains_key(edge.edge_left_ccw), "{}", msg);
            assert!(self.edges.contains_key(edge.edge_left_cw), "{}", msg);

            self.assert_vertex_valid(edge.vert_origin);
            self.assert_vertex_valid(self.edges[edge.edge_left_ccw].vert_origin);
            self.assert_vertex_valid(self.edges[edge.edge_left_cw].vert_origin);
        }

        pub fn assert_vertex_valid(&self, vid: VertID) {
            assert!(self.verts.contains_key(vid));

            for e in self.verts[vid].outgoing_edges() {
                assert!(
                    self.edges[e].vert_origin == vid,
                    "Invalid vertex edge source loop - Mesh made invalid on V{vid:?} "
                );
            }
            for e in self.verts[vid].incoming_edges() {
                assert!(
                    self.edges[self.edges[e].edge_left_cw].vert_origin == vid,
                    "Invalid vertex edge dest loop - Mesh made invalid on V{vid:?} "
                );
            }
        }
        /// Find the inner boundary of a set of faces. Quite simple - record all edges we have seen, and any twins for those.
        /// Inner boundary as will not include edges from an edge of the mesh, as these have no twins.
        pub fn face_boundary(&self, faces: &Vec<&Face>) -> HashSet<EdgeID> {
            let mut unseen_edges = HashSet::<EdgeID>::with_capacity(faces.len() * 3);

            for face in faces {
                for edge in self.iter_edge_loop(face.edge) {
                    if let Some(t) = self.edges[edge].twin {
                        unseen_edges.insert(t);
                    }
                }
            }

            let all_unseen_edges = unseen_edges.clone();
            unseen_edges
                .retain(|&edge| !all_unseen_edges.contains(&self.edges[edge].twin.unwrap()));

            unseen_edges
        }
    }

    #[test]
    pub fn test_continued_validity() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_MID);

        mesh.assert_valid();

        mesh.partition_within_groups(test_config, None)?;

        mesh.assert_valid();

        mesh.group(test_config, &verts)?;

        mesh.assert_valid();

        mesh.partition_within_groups(test_config, Some(2))?;

        println!("{} {}", mesh.partition_count(), mesh.groups.len());

        assert!(mesh.partition_count() <= mesh.groups.len() * 2);

        mesh.assert_valid();

        Ok(())
    }

    #[test]
    pub fn test_faces_boundary() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let mesh = TEST_MESH_MID;
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_within_groups(test_config, None)?;

        let mut boundary_face_ratio = 0.0;
        for pi in 0..mesh.partitions.len() {
            // Assert that new parts and the parts in group have the same boundary

            let faces = mesh.faces.values().filter(|&f| pi == f.part).collect();

            let boundary = mesh.face_boundary(&faces);

            for &e in &boundary {
                assert!(
                    !faces.contains(&&mesh.faces[mesh.edges[e].face]),
                    "'Boundary' contains edge inside itself"
                )
            }

            boundary_face_ratio += faces.len() as f32 / boundary.len() as f32;
        }

        boundary_face_ratio /= mesh.partitions.len() as f32;

        println!("Average partition face count / boundary length: {boundary_face_ratio}");

        // For the sphere, correct ratio around this
        assert!(boundary_face_ratio > 3.3 && boundary_face_ratio < 3.4);

        Ok(())
    }

    #[test]
    pub fn test_group_repartitioning() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        let mesh = TEST_MESH_MID;
        let (mut mesh, verts) = WingedMesh::from_gltf(mesh);

        // Apply primary partition, that will define the lowest level clusterings
        mesh.partition_within_groups(test_config, None)?;

        mesh.group(test_config, &verts)?;

        let old_faces = mesh.faces.clone();

        // Old parts must have no child - there is no LOD-1
        for p in &mesh.partitions {
            assert!(p.child_group_index.is_none());
        }

        mesh.partition_within_groups(test_config, Some(2))?;

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
    fn mesh_stats_redout() {
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_CONE);

        let mut avg_outgoing = 0.0;
        let mut avg_incoming = 0.0;

        for (vid, v) in &mesh.verts {
            avg_outgoing += v.outgoing_edges().len() as f32;
            avg_incoming += v.incoming_edges().len() as f32;
        }
        avg_outgoing /= mesh.verts.len() as f32;
        avg_incoming /= mesh.verts.len() as f32;

        println!("Average Outgoing: {avg_outgoing}, Average Incoming: {avg_incoming}");
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };

        mesh.partition_full_mesh(test_config, mesh.verts.len().div_ceil(60) as u32)
            .unwrap();

        let mut embed_prop = 0.0;

        for (vid, v) in &mesh.verts {
            embed_prop += if vid.is_group_embedded(&mesh) {
                1.0
            } else {
                0.0
            };
        }
        embed_prop /= mesh.verts.len() as f32;
        println!("Embedded Proportion: {embed_prop}");
    }

    #[test]
    fn test_group_and_partition_and_simplify() {
        let (working_mesh, verts) = WingedMesh::from_gltf(TEST_MESH_CONE);

        // group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        group_and_partition_and_simplify(working_mesh, &verts, TEST_MESH_CONE.to_owned());
    }

    #[test]
    fn test_group_and_partition() {
        let (working_mesh, verts) = WingedMesh::from_gltf(TEST_MESH_CONE);

        // group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        group_and_partition_full_res(working_mesh, &verts, TEST_MESH_CONE.to_owned());
    }
    #[test]
    fn test_reduce_contiguous() {
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_CONE);

        println!("Asserting contiguous");
        // WE know the circle is contiguous
        //assert_contiguous_graph(&working_mesh.generate_face_graph());

        let mut quadrics = mesh.create_quadrics(&verts);

        let e = match mesh.reduce(&verts, &mut quadrics, &[mesh.face_count() / 4], |_, _| 0) {
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
        assert_contiguous_graph(&mesh.generate_face_graph());
    }

    #[test]
    fn test_apply_simplification() {
        let (working_mesh, verts) = WingedMesh::from_gltf(TEST_MESH_CONE);

        // WE know the circle is contiguous
        //assert_contiguous_graph(&working_mesh.generate_face_graph());

        // group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
        let working_mesh = apply_simplification(working_mesh, &verts, TEST_MESH_CONE.to_owned());

        println!("Asserting face graph is contiguous");
        // It should still be contiguous
        assert_contiguous_graph(&working_mesh.generate_face_graph());
    }

    #[test]
    pub fn test_partition_contiguity() -> Result<(), Box<dyn Error>> {
        let test_config = &PartitioningConfig {
            method: metis::PartitioningMethod::MultilevelKWay,
            force_contiguous_partitions: true,
            minimize_subgraph_degree: Some(true),
            ..Default::default()
        };
        let (mut mesh, verts) = WingedMesh::from_gltf(TEST_MESH_LOW);

        // Apply primary partition, that will define the lowest level clusterings
        for i in 9..50 {
            println!("{i}");

            mesh.partition_full_mesh(test_config, i)?;

            println!("Partitioned");
            let mut graph = mesh.generate_face_graph();

            graph.retain_edges(|g, e| {
                let (v1, v2) = g.edge_endpoints(e).unwrap();

                let p1 = mesh.faces[g.node_weight(v1).unwrap()].part;
                let p2 = mesh.faces[g.node_weight(v2).unwrap()].part;

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
