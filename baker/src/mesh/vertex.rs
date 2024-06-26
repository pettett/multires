use std::fmt::Display;

use super::{
    edge::EdgeID,
    half_edge_mesh::{HalfEdgeMesh, MeshError},
    plane::Plane,
    quadric::Quadric,
};

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct VertID(u32);

impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0 as _
    }
}
impl Into<u32> for VertID {
    fn into(self) -> u32 {
        self.0
    }
}
impl From<usize> for VertID {
    fn from(value: usize) -> Self {
        VertID(value as _)
    }
}

impl From<u32> for VertID {
    fn from(value: u32) -> Self {
        VertID(value)
    }
}

impl Display for VertID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Vertex {
    // Edge with vert_source = this id
    outgoing_edges: Vec<EdgeID>,
    incoming_edges: Vec<EdgeID>,

    vert_i: usize,
}

impl VertID {
    pub fn new(id: usize) -> Self {
        Self(id as _)
    }
    pub fn id(&self) -> u32 {
        self.0
    }

    pub fn vert(self, mesh: &HalfEdgeMesh) -> &Vertex {
        mesh.verts().get(self)
    }
    pub fn vert_mut(self, mesh: &mut HalfEdgeMesh) -> &mut Vertex {
        mesh.verts_mut().get_mut(self)
    }
    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn is_local_manifold(self, mesh: &HalfEdgeMesh) -> bool {
        let v = mesh.try_get_vert(self);
        let Ok(vert) = v else {
            return false;
        };

        let eid_first = vert.outgoing_edges()[0];

        let mut eid = eid_first;

        //let mut last_e_part = None;

        for _i in 0..(vert.outgoing_edges().len() * 2) {
            // attempt to move around the fan, by moving to our twin edge and going clockwise
            let e = eid.edge(mesh);

            if e.vert_origin != self {
                println!("Iteration around vertex escaped vertex");
                return false;
            }

            assert_eq!(
                e.vert_origin, self,
                "Iteration around vertex escaped vertex"
            );

            let Some(twin) = e.twin else {
                return false;
            };

            let e = twin.edge(mesh);

            // Compare against last face's partition
            // if is_group_manifold {
            //     let e_part = mesh.faces()[e.face].part;

            //     if let Some(lep) = last_e_part {
            //         if e_part != lep {
            //             return false;
            //         }
            //     }

            //     last_e_part = Some(e_part);
            // }

            eid = e.edge_next_ccw;

            if eid == eid_first {
                return true;
            }
        }

        println!(
            "ERROR: manifold vertex looped too many times, but only {} outgoing",
            vert.outgoing_edges.len()
        );
        return false;
    }

    pub fn is_group_embedded(self, mesh: &HalfEdgeMesh) -> bool {
        let Ok(vert) = mesh.try_get_vert(self) else {
            return false;
        };

        let outgoings = &vert.outgoing_edges;
        let group_index =
            mesh.clusters[outgoings[0].edge(mesh).face.face(mesh).cluster_idx].group_index();

        for &eid in &outgoings[1..] {
            if group_index
                != mesh.clusters[eid.edge(mesh).face.face(mesh).cluster_idx].group_index()
            {
                return false;
            }
        }

        #[cfg(test)]
        for &eid in &vert.incoming_edges {
            assert_eq!(
                group_index,
                mesh.clusters[eid.edge(mesh).face.face(mesh).cluster_idx].group_index()
            );
        }

        return true;
    }

    /// Generate error matrix Q, the sum of Kp for all planes p around this vertex.
    /// TODO: Eventually we can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    pub fn generate_error_matrix(self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Quadric {
        self.vert(mesh).generate_error_matrix(mesh, verts)
    }
}

impl Vertex {
    pub fn remove_outgoing(&mut self, e: EdgeID) {
        #[cfg(test)]
        {
            assert!(
                self.outgoing_edges.contains(&e),
                "Attempted to remove invalid outgoing edge"
            );
        }

        let index = self
            .outgoing_edges
            .iter()
            .position(|&x| x == e)
            .expect("Outgoing edges should contain edge id we are removing");
        self.outgoing_edges.swap_remove(index);

        #[cfg(test)]
        {
            assert!(!self.outgoing_edges.contains(&e));
        }
    }
    pub fn remove_incoming(&mut self, e: EdgeID) -> Result<(), MeshError> {
        // #[cfg(test)]
        // {
        //     assert!(
        //         self.incoming_edges.contains(&e),
        //         "Attempted to remove invalid incoming edge"
        //     );
        // }

        let index = self
            .incoming_edges
            .iter()
            .position(|&x| x == e)
            .ok_or(MeshError::InvalidEdge(e))?;

        self.incoming_edges.swap_remove(index);

        #[cfg(test)]
        {
            assert!(!self.incoming_edges.contains(&e));
        }
        return Ok(());
    }
    pub fn add_outgoing(&mut self, e: EdgeID) {
        #[cfg(test)]
        {
            assert!(!self.outgoing_edges.contains(&e));
        }
        assert!(self.outgoing_edges.len() < 1000);

        self.outgoing_edges.push(e);
    }
    pub fn add_incoming(&mut self, e: EdgeID) {
        #[cfg(test)]
        {
            assert!(!self.incoming_edges.contains(&e));
        }
        assert!(self.incoming_edges.len() < 1000);

        self.incoming_edges.push(e);
    }

    pub fn outgoing_edges(&self) -> &[EdgeID] {
        self.outgoing_edges.as_ref()
    }

    pub fn incoming_edges(&self) -> &[EdgeID] {
        self.incoming_edges.as_ref()
    }

    pub fn unpack(self) -> (Vec<EdgeID>, Vec<EdgeID>) {
        (self.incoming_edges, self.outgoing_edges)
    }

    /// Generate error matrix Q, the sum of Kp for all planes p around this vertex.
    /// We can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    pub fn generate_error_matrix(&self, mesh: &HalfEdgeMesh, verts: &[glam::Vec3A]) -> Quadric {
        let mut q = Quadric::default();

        for &eid in self.outgoing_edges() {
            let e = eid.edge(mesh);
            let f = e.face;

            let plane = f.plane(mesh, verts);

            // If these two vertices are in the exact same place, we will get a NaN plane.
            // In this case, there is 'no' error from collapsing between them (as we do not include normals).
            let k_p = if plane.0.is_nan() {
                Quadric::default()
            } else {
                plane.fundamental_error_quadric()
            };

            q += k_p;

            if e.twin.is_none() {
                // Boundary edge, add a plane that stops us moving away from the boundary

                // Edge Plane should have normal of Edge X plane
                let v = eid.edge_vec(mesh, verts).unwrap();

                let boundary_norm = v.cross(plane.normal());

                let boundary_plane = Plane::from_normal_and_point(
                    boundary_norm,
                    verts[e.vert_origin.0 as usize].into(),
                );
                // Multiply squared error by large factor, as changing the boundary adds a lot of visible error
                q += boundary_plane.fundamental_error_quadric() * 3000.0;
            }
        }

        q
    }
}
