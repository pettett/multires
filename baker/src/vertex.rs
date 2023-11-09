use crate::winged_mesh::{EdgeID, WingedMesh};

#[derive(Default, Hash, Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertID(pub usize);
impl Into<usize> for VertID {
    fn into(self) -> usize {
        self.0
    }
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    /// Edge with vert_source = this id
    pub edge: Option<EdgeID>,
}

impl VertID {
    /// Does this vertex have a complete fan of triangles surrounding it?
    pub fn is_local_manifold(&self, mesh: &WingedMesh, is_group_manifold: bool) -> bool {
        let Some(eid_first) = mesh[*self].edge else {
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
        mesh.edge_map().get(&self).map(|v| &v[..]).unwrap_or(EMPTY)
    }
}
