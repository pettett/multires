use idmap::IntegerId;

use super::winged_mesh::{EdgeID, WingedMesh};

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertID(pub usize);
impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0
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
        let index = self.outgoing_edges.iter().position(|&x| x == e).unwrap();
        self.outgoing_edges.swap_remove(index);
    }
    pub fn remove_incoming(&mut self, e: EdgeID) {
        let index = self.incoming_edges.iter().position(|&x| x == e).unwrap();
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
    pub fn is_local_manifold(&self, mesh: &WingedMesh, is_group_manifold: bool) -> bool {
        let Some(&eid_first) = self.outgoing_edges(mesh).get(0) else {
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
                let e_part = mesh.partitions[mesh.faces()[e.face].part].group_index;

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

    pub fn incoming_edges(self, mesh: &WingedMesh) -> Vec<EdgeID> {
        mesh.edges()
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if e.valid && mesh[e.edge_left_cw].vert_origin == self {
                    Some(EdgeID(i))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn outgoing_edges(self, mesh: &WingedMesh) -> &[EdgeID] {
        const EMPTY: &[EdgeID] = &[];
        mesh.verts
            .get(&self)
            .map(|v| &v.outgoing_edges[..])
            .unwrap_or(EMPTY)
    }
}
