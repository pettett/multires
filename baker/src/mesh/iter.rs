use super::winged_mesh::{EdgeID, HalfEdge, WingedMesh};

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

        if let Some(curr) = current {
            self.current = Some(self.mesh[curr].edge_left_cw);
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
pub struct AllEdgeIter {
    edges: Vec<HalfEdge>,
    current: Option<usize>,
}

impl AllEdgeIter {
    pub fn new(edges: Vec<HalfEdge>, current: Option<usize>) -> Self {
        Self { edges, current }
    }
}
impl Iterator for AllEdgeIter {
    type Item = EdgeID;

    fn next(&mut self) -> Option<Self::Item> {
        let last = self.current;

        if let Some(current) = &mut self.current {
            loop {
                *current += 1;

                if *current >= self.edges.len() {
                    self.current = None;
                    break;
                }

                if !self.edges[*current].valid {
                    continue;
                } else {
                    break;
                }
            }
        }

        last.map(|l| EdgeID(l))
    }
}
