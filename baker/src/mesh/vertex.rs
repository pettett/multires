

use super::{edge::EdgeID, plane::Plane, quadric::Quadric, winged_mesh::WingedMesh};

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

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Vertex {
    // Edge with vert_source = this id
    outgoing_edges: Vec<EdgeID>,
    incoming_edges: Vec<EdgeID>,
}

impl VertID {
    pub fn new(id: usize) -> Self {
        Self(id as _)
    }
    pub fn id(&self) -> u32 {
        self.0
    }

    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn is_local_manifold(self, mesh: &WingedMesh) -> bool {
        let v = mesh.try_get_vert(self);
        let Ok(vert) = v else {
            return false;
        };

        let eid_first = vert.outgoing_edges()[0];

        let mut eid = eid_first;

        //let mut last_e_part = None;

        for _i in 0..(vert.outgoing_edges().len() * 2) {
            // attempt to move around the fan, by moving to our twin edge and going clockwise
            let e = mesh.get_edge(eid);

            assert_eq!(
                e.vert_origin, self,
                "Iteration around vertex escaped vertex"
            );

            let Some(twin) = e.twin else {
                return false;
            };

            let e = &mesh.get_edge(twin);

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

            eid = e.edge_left_cw;

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

    pub fn is_group_embedded(self, mesh: &WingedMesh) -> bool {
        let Ok(vert) = mesh.try_get_vert(self) else {
            return false;
        };

        let outgoings = &vert.outgoing_edges;
        let group_index = mesh.clusters
            [mesh.get_face(mesh.get_edge(outgoings[0]).face).cluster_idx]
            .group_index();

        for &eid in &outgoings[1..] {
            if group_index
                != mesh.clusters[mesh.get_face(mesh.get_edge(eid).face).cluster_idx].group_index()
            {
                return false;
            }
        }

        #[cfg(test)]
        for &eid in &vert.incoming_edges {
            assert_eq!(
                group_index,
                mesh.clusters[mesh.get_face(mesh.get_edge(eid).face).cluster_idx].group_index()
            );
        }

        return true;
    }

    /// Generate error matrix Q, the sum of Kp for all planes p around this vertex.
    /// TODO: Eventually we can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    pub fn generate_error_matrix(self, mesh: &WingedMesh, verts: &[glam::Vec3A]) -> Quadric {
        mesh.get_vert(self).generate_error_matrix(mesh, verts)
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
    pub fn remove_incoming(&mut self, e: EdgeID) {
        #[cfg(test)]
        {
            assert!(
                self.incoming_edges.contains(&e),
                "Attempted to remove invalid incoming edge"
            );
        }

        let index = self
            .incoming_edges
            .iter()
            .position(|&x| x == e)
            .expect("Incoming edges should contain edge id we are removing");
        self.incoming_edges.swap_remove(index);

        #[cfg(test)]
        {
            assert!(!self.incoming_edges.contains(&e));
        }
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
    /// TODO: Eventually we can also add a high penality plane if this is a vertex on a boundary, but manually checking may be easier
    pub fn generate_error_matrix(&self, mesh: &WingedMesh, verts: &[glam::Vec3A]) -> Quadric {
        let mut q = Quadric::default();

        for &eid in self.outgoing_edges() {
            let e = mesh.get_edge(eid);
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
