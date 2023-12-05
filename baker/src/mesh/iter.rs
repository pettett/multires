use std::marker::PhantomData;

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

pub struct IDVecIter<'a, IDT: From<usize>, T> {
    vec: &'a Vec<Option<T>>,
    current: usize,
    _p: PhantomData<IDT>,
}

impl<'a, IDT: From<usize>, T> IDVecIter<'a, IDT, T> {
    pub fn new(vec: &'a Vec<Option<T>>) -> Self {
        Self {
            vec,
            current: 0,
            _p: PhantomData,
        }
    }
}

impl<'a, IDT: From<usize>, T> Iterator for IDVecIter<'a, IDT, T> {
    type Item = (IDT, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        // Find and return the first non-none
        while self.current < self.vec.len() {
            if let Some(x) = &self.vec[self.current] {
                return Some((self.current.into(), x));
            } else {
                self.current += 1;
            }
        }
        // If there isn't one, return none
        return None;
    }
}
