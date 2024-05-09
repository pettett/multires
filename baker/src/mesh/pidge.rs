use std::{marker::PhantomData, mem};

pub trait Key: Into<usize> + From<usize> {}
pub trait Value: Clone + PartialEq {}

impl<T> Key for T where T: Into<usize> + From<usize> {}
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

impl<V> Into<Option<V>> for PidgeHole<V> {
    fn into(self) -> Option<V> {
        match self {
            PidgeHole::Filled(v) => Some(v),
            PidgeHole::Empty { .. } => None,
        }
    }
}

impl<V> PidgeHole<V> {
    fn as_ref(&self) -> Option<&V> {
        match self {
            PidgeHole::Filled(v) => Some(v),
            PidgeHole::Empty { .. } => None,
        }
    }

    fn as_mut(&mut self) -> Option<&mut V> {
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

    pub fn wipe(&mut self, key: K) -> V {
        let id = key.into();

        let new_span_start = None;
        let new_span_end = None;

        // And write in ourself to have correct values, in the case that either of the above span points are still None.
        let mut data = PidgeHole::Empty {
            span_start: new_span_start.unwrap_or(id),
            span_end: new_span_end.unwrap_or(id),
        };
        mem::swap(&mut data, &mut self.data[id]);

        self.len -= 1;

        match data {
            PidgeHole::Filled(d) => d,
            PidgeHole::Empty { .. } => unreachable!(),
        }
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
        self.data.iter().filter_map(|p| p.as_ref())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut V> + '_ {
        self.data.iter_mut().filter_map(|p| p.as_mut())
    }

    pub fn iter_items(&self) -> impl Iterator<Item = (K, &V)> + '_ {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().map(|v| (K::from(i), v)))
    }

    pub fn iter_items_mut(&mut self) -> impl Iterator<Item = (K, &mut V)> + '_ {
        self.data
            .iter_mut()
            .enumerate()
            .filter_map(|(i, p)| p.as_mut().map(|v| (K::from(i), v)))
    }

    pub fn iter_keys(&self) -> impl Iterator<Item = K> + '_ {
        self.data
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().and(Some(K::from(i))))
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
