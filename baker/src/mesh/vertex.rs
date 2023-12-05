use idmap::IntegerId;

use super::winged_mesh::{EdgeID, WingedMesh};

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertID(pub usize);
impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0
    }
}
impl From<usize> for VertID {
    fn from(value: usize) -> Self {
        VertID(value)
    }
}
impl IntegerId for VertID {
    fn from_id(id: u64) -> Self {
        VertID(id as usize)
    }

    fn id(&self) -> u64 {
        self.0 as _
    }

    fn id32(&self) -> u32 {
        self.0 as _
    }
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    // Edge with vert_source = this id
    outgoing_edges: Vec<EdgeID>,
    incoming_edges: Vec<EdgeID>,
}

impl Vertex {
    pub fn remove_outgoing(&mut self, e: EdgeID) {
        let index = self
            .outgoing_edges
            .iter()
            .position(|&x| x == e)
            .expect("Outgoing edges should contain edge id we are removing");
        self.outgoing_edges.swap_remove(index);
    }
    pub fn remove_incoming(&mut self, e: EdgeID) {
        let index = self
            .incoming_edges
            .iter()
            .position(|&x| x == e)
            .expect("Incoming edges should contain edge id we are removing");
        self.incoming_edges.swap_remove(index);
    }
    pub fn add_outgoing(&mut self, e: EdgeID) {
        self.outgoing_edges.push(e);
    }
    pub fn add_incoming(&mut self, e: EdgeID) {
        self.incoming_edges.push(e);
    }

    pub fn outgoing_edges(&self) -> &[EdgeID] {
        self.outgoing_edges.as_ref()
    }

    pub fn incoming_edges(&self) -> &[EdgeID] {
        self.incoming_edges.as_ref()
    }
}

impl VertID {
    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn is_local_manifold(self, mesh: &WingedMesh) -> bool {
        let v = mesh.try_get_vert(self);
        let Some(vert) = v else {
            return false;
        };

        let eid_first = vert.outgoing_edges()[0];

        let mut eid = eid_first;

        //let mut last_e_part = None;

        loop {
            // attempt to move around the fan, by moving to our twin edge and going clockwise
            let Some(twin) = mesh.get_edge(eid).twin else {
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
    }

    pub fn is_group_embedded(self, mesh: &WingedMesh) -> bool {
        let Some(vert) = mesh.try_get_vert(self) else {
            return false;
        };

        let outgoings = &vert.outgoing_edges;
        let part =
            mesh.partitions[mesh.get_face(mesh.get_edge(outgoings[0]).face).part].group_index;

        for &eid in &outgoings[1..] {
            if part != mesh.partitions[mesh.get_face(mesh.get_edge(eid).face).part].group_index {
                return false;
            }
        }

        #[cfg(test)]
        for &eid in &vert.incoming_edges {
            if part != mesh.partitions[mesh.get_face(mesh.get_edge(eid).face).part].group_index {
                unreachable!();
            }
        }

        return true;
    }
}
