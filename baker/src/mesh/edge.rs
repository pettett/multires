use super::{
    face::FaceID,
    half_edge_mesh::{HalfEdgeMesh, MeshError},
    plane::Plane,
    quadric::Quadric,
    quadric_error::QuadricError,
    vertex::VertID,
};
use anyhow::{Context, Result};
use idmap::IntegerId;

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
#[repr(transparent)]
pub struct EdgeID(u32);
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct HalfEdge {
    pub vert_origin: VertID,
    // This is not actually needed, as the destination is the origin of the cw edge
    //pub vert_destination: VertID,
    pub face: FaceID,
    /// Edge connecting into the origin vert
    pub edge_back_cw: EdgeID,
    /// Edge leading on from the dest vert
    pub edge_next_ccw: EdgeID,

    pub age: u32,

    pub twin: Option<EdgeID>,
}

impl IntegerId for EdgeID {
    fn from_id(id: u64) -> Self {
        EdgeID(id as _)
    }

    fn id(&self) -> u64 {
        self.0 as u64
    }

    fn id32(&self) -> u32 {
        self.0 as u32
    }
}

impl From<usize> for EdgeID {
    fn from(value: usize) -> Self {
        Self(value as _)
    }
}
impl From<EdgeID> for usize {
    fn from(value: EdgeID) -> Self {
        value.0 as _
    }
}
impl Into<EdgeID> for u32 {
    fn into(self) -> EdgeID {
        EdgeID(self)
    }
}

pub struct EdgeIter<'a> {
    mesh: &'a HalfEdgeMesh,
    start: EdgeID,
    current: Option<EdgeID>,
    max_iter: usize,
}

impl<'a> EdgeIter<'a> {
    pub fn new(
        mesh: &'a HalfEdgeMesh,
        start: EdgeID,
        current: Option<EdgeID>,
        max_iter: usize,
    ) -> Self {
        Self {
            mesh,
            start,
            current,
            max_iter,
        }
    }
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = EdgeID;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;

        if let Some(current) = current {
            self.current = Some(current.edge(&self.mesh).edge_back_cw);
            if self.current == Some(self.start) {
                self.current = None;
            }

            if self.max_iter == 0 {
                panic!("Iterated too many edges for polygon mesh")
            } else {
                self.max_iter -= 1;
            }
        }

        current
    }
}

impl HalfEdge {
    /// Grab the source and destination vertex IDs from this edge.
    /// Source vertex is just `HalfEdge.vert_origin`, destination vertex is `HalfEdge.edge_left_cw`'s vert_origin
    pub fn src_dst(&self, mesh: &HalfEdgeMesh) -> Result<(VertID, VertID)> {
        Ok((self.vert_origin, self.dst(mesh)?))
    }

    pub fn dst(&self, mesh: &HalfEdgeMesh) -> Result<VertID> {
        Ok(mesh
            .try_get_edge(self.edge_next_ccw)
            .context(MeshError::InvalidCCwEdge(self.edge_next_ccw))?
            .vert_origin)
    }
}

impl EdgeID {
    /// Grab the source and destination vertex IDs from this edge.
    /// Source vertex is just `HalfEdge.vert_origin`, destination vertex is `HalfEdge.edge_left_cw`'s vert_origin
    pub fn src_dst(self, mesh: &HalfEdgeMesh) -> Result<(VertID, VertID)> {
        let edge = mesh.try_get_edge(self)?;

        edge.src_dst(mesh)
    }

    /// Get a vector A-B for the source and destinations from [`EdgeID::src_dst`]
    pub fn edge_vec(&self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Result<glam::Vec3A> {
        let (src, dst) = self.src_dst(mesh)?;
        let vd = verts[dst.id() as usize];
        let vs = verts[src.id() as usize];
        Ok(vd - vs)
    }

    pub fn edge(self, mesh: &HalfEdgeMesh) -> &HalfEdge {
        mesh.edges().get(self)
    }
    pub fn edge_mut(self, mesh: &mut HalfEdgeMesh) -> &mut HalfEdge {
        mesh.edges_mut().get_mut(self)
    }

    /// Evaluate if any restrictions on edge collapse apply to this edge.
    ///
    /// Current restrictions:
    /// - Cannot change group boundaries.
    /// - Cannot split group into non-contiguous segments.
    /// - Cannot split with connected triangles which would cause an overlap.
    /// - Cannot flip normals of any triangles when collapsing.
    pub fn can_collapse_edge(self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Result<bool> {
        let (orig, dest) = self.src_dst(mesh)?;

        if !orig.is_group_embedded(mesh) {
            // Cannot move a vertex unless it is in the center of a partition
            return Ok(false);
        }

        //if !orig.is_group_embedded(mesh) {
        //    // Cannot change group boundary
        //    // Cannot move a vertex unless it is in the center of a partition
        //    return Ok(false);
        //}

        if self.edge(mesh).twin.is_some() {
            // This edge has a twin, so a bad collapse risks splitting the group into 2 pieces.

            // If we change the boundary shape, we must move into a manifold embedded position,
            // or we risk separating the partition into two chunks, which will break partitioning

            if !orig.is_local_manifold(mesh) {
                // We can split the group into two chunks by collapsing onto another edge (non manifold dest),
                //  or collapsing onto another group
                if !dest.is_group_embedded(mesh) || !dest.is_local_manifold(mesh) {
                    return Ok(false);
                }
            }
        }

        // Cannot collapse edges that would result in combining triangles over each other
        if !mesh.max_one_joint_neighbour_vertices_per_side(self) {
            return Ok(false);
        }

        // Test normals of triangles before and after the swap
        for &e in orig.vert(mesh).outgoing_edges() {
            if e == self {
                continue;
            }

            let f = e.edge(mesh).face;
            let [a, b, c] = mesh.triangle_from_face(f.face(mesh));

            let (v_a, v_b, v_c, new_corner) = (
                verts[a as usize].into(),
                verts[b as usize].into(),
                verts[c as usize].into(),
                verts[dest.id() as usize].into(),
            );
            let starting_plane = Plane::from_three_points(v_a, v_b, v_c);

            let end_plane = match orig.id() {
                i if i == a => Plane::from_three_points(new_corner, v_b, v_c),
                i if i == b => Plane::from_three_points(v_a, new_corner, v_c),
                i if i == c => Plane::from_three_points(v_a, v_b, new_corner),
                _ => unreachable!(),
            };

            if starting_plane.normal().dot(end_plane.normal()) < 0.0 {
                // Flipped triangle, give this an invalid weight - discard
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Estimate the error introduced by collapsing this edge. Does not take into account penalties from flipping triangles
    pub fn edge_collapse_error(
        self,
        mesh: &HalfEdgeMesh,
        verts: &[glam::Vec3A],
        quadric_errors: &[Quadric],
    ) -> Result<QuadricError> {
        let (orig, dest) = self.src_dst(mesh)?;

        // Collapsing this edge would move the origin to the destination, so we find the error of the origin at the merged point.
        let q = (&quadric_errors[orig.id() as usize]) + (&quadric_errors[dest.id() as usize]);

        let error = q.quadric_error(verts[orig.id() as usize].into());

        assert!(
            !error.is_nan(),
            "Edge between {orig:?} and {dest:?} is NaN.",
        );

        Ok(QuadricError::new(error, self))
    }
}
