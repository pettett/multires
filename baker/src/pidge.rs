use std::marker::PhantomData;

pub trait Key: Into<usize> {}
pub trait Value: Clone + PartialEq {}

impl<T> Key for T where T: Into<usize> {}
impl<T> Value for T where T: Clone + PartialEq {}

/// A `Pidge` is a form of slotmap without key generations
/// Very useful for 'Object Soups', where we want a low cost layer of indirection for object references
/// This structure is tailored for a series of inserts, followed by removals, and as such does not support interleaving these operations.
///
/// We store only the data in full slots, and in empty spans we store information required to skip that span when iterating
/// Only the cells at the start and end of a span are required to contain valid data, as these will be the only ones ever read.
///
/// The start of the span contains a index for its endpoint, and the end stores an index for its start.
/// This allows us to merge spans in O(1) time, and completely skip all their elements when iterating, creating a decent speedup
#[derive(Debug, Clone, PartialEq)]
pub struct Pidge<K: Key, V> {
    data: Vec<PidgeHole<V>>,
    len: usize,
    _k: PhantomData<K>,
}
#[derive(Debug, Clone, PartialEq)]
enum PidgeHole<V> {
    Filled(V),
    Empty { span_start: usize, span_end: usize },
}
struct Iter<'a, V> {
    idx: usize,
    data: &'a Vec<PidgeHole<V>>,
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.data.len() {
            return None;
        }

        match &self.data[self.idx] {
            PidgeHole::Filled(v) => {
                self.idx += 1;
                return Some(v);
            }
            PidgeHole::Empty { span_end, .. } => {
                if span_end + 1 >= self.data.len() {
                    // Break the iterator, no data after this span
                    return None;
                } else {
                    self.idx = span_end + 1;
                }
            }
        }

        match &self.data[self.idx] {
            PidgeHole::Filled(v) => {
                self.idx += 1;
                Some(v)
            }
            _ => panic!("Span end should be followed by a valid filled pidge hole"),
        }
    }
}

impl<V> Into<Option<V>> for PidgeHole<V> {
    fn into(self) -> Option<V> {
        match self {
            PidgeHole::Filled(v) => Some(v),
            PidgeHole::Empty { .. } => None,
        }
    }
}

impl<V> PidgeHole<V> {
    fn as_ref<'a>(&'a self) -> Option<&'a V> {
        match self {
            PidgeHole::Filled(v) => Some(v),
            PidgeHole::Empty { .. } => None,
        }
    }

    fn as_mut<'a>(&'a mut self) -> Option<&'a mut V> {
        match self {
            PidgeHole::Filled(v) => Some(v),
            PidgeHole::Empty { .. } => None,
        }
    }
}

impl<K: Key, V> Pidge<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut p = Pidge {
            data: Vec::with_capacity(capacity),
            len: 0,
            _k: PhantomData::default(),
        };

        // We assume that every value in the pidge will be written over
        for _ in 0..capacity {
            p.data.push(PidgeHole::Empty {
                span_start: 0,
                span_end: capacity - 1,
            })
        }

        p
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert(&mut self, key: K, val: V) {
        // Don't worry about breaking gaps, as we assume we only iterate after entire pidge is populated
        self.data[key.into()] = PidgeHole::Filled(val);
        self.len += 1;
    }

    pub fn wipe(&mut self, key: K) {
        let id = key.into();

        let mut new_span_start = None;
        let mut new_span_end = None;

        //TODO: test if this improves performance

        if id > 1 {
            // Merge with span on the left
            match &self.data[id - 1] {
                PidgeHole::Empty { span_start, .. } => new_span_start = Some(*span_start),
                _ => (),
            }
        }

        if id < self.data.len() - 1 {
            // Merge with span on the left
            match &self.data[id + 1] {
                PidgeHole::Empty { span_end, .. } => new_span_end = Some(*span_end),
                _ => (),
            }
        }

        // Write in the new span. Only write over data if it has changed
        if let Some(new_span_start) = new_span_start {
            match &mut self.data[new_span_start] {
                PidgeHole::Empty {
                    span_start,
                    span_end,
                } => {
                    *span_start = new_span_start;
                    *span_end = new_span_end.unwrap_or(id);
                }
                _ => panic!("Invalid Span Start"),
            }
        }

        if let Some(new_span_end) = new_span_end {
            match &mut self.data[new_span_end] {
                PidgeHole::Empty {
                    span_start,
                    span_end,
                } => {
                    *span_start = new_span_start.unwrap_or(id);
                    *span_end = new_span_end;
                }
                _ => panic!("Invalid Span End"),
            }
        }

        // And write in ourself to have correct values, in the case that either of the above span points are still None.
        self.data[id] = PidgeHole::Empty {
            span_start: new_span_start.unwrap_or(id),
            span_end: new_span_end.unwrap_or(id),
        };

        self.len -= 1;
    }
    pub fn get(&self, key: K) -> &V {
        self.data[key.into()].as_ref().unwrap()
    }

    pub fn get_mut(&mut self, key: K) -> &mut V {
        self.data[key.into()].as_mut().unwrap()
    }

    pub fn get_mut_or_default(&mut self, key: K) -> &mut V {
        self.data[key.into()].as_mut().unwrap()
    }

    pub fn try_get(&self, key: K) -> Option<&V> {
        self.data[key.into()].as_ref()
    }

    pub fn try_get_mut(&mut self, key: K) -> Option<&mut V> {
        self.data[key.into()].as_mut()
    }

    pub fn slot_full(&mut self, key: K) -> bool {
        self.data[key.into()].as_ref().is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &V> + '_ {
        Iter {
            idx: 0,
            data: &self.data,
        }
    }
    pub fn iter_with_empty(&self) -> impl Iterator<Item = Option<&V>> + '_ {
        self.data.iter().map(|x| x.as_ref())
    }
    pub fn iter_mut_with_empty(&mut self) -> impl Iterator<Item = Option<&mut V>> + '_ {
        self.data.iter_mut().map(|x| x.as_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::Pidge;

    #[test]
    fn test_general() {
        let mut pidge = Pidge::with_capacity(10);

        for i in 0..10usize {
            pidge.insert(i, i);
        }

        println!("{pidge:?}");

        pidge.wipe(1);
        pidge.wipe(2);
        pidge.wipe(3);
        pidge.wipe(9);

        println!("{pidge:?}");

        let mut t = 0;

        for p in pidge.iter() {
            t += 1;
            println!("{p}")
        }
        assert_eq!(t, pidge.len());
    }
}
