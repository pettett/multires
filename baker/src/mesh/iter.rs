use super::winged_mesh::{EdgeID, WingedMesh};

pub struct EdgeIter<'a> {
    mesh: &'a WingedMesh,
    start: EdgeID,
    current: Option<EdgeID>,
    max_iter: usize,
}

impl<'a> EdgeIter<'a> {
    pub fn new(
        mesh: &'a WingedMesh,
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
            self.current = Some(self.mesh.get_edge(current).edge_left_cw);
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
