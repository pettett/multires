use std::cmp;

use super::edge::EdgeID;

/// Reverses the ordering of a float value, such that we take min values from a priority queue.
#[derive(Clone, Copy)]
pub struct QuadricError(pub cmp::Reverse<f64>, pub EdgeID);

impl PartialEq for QuadricError {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for QuadricError {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Eq for QuadricError {}

impl Ord for QuadricError {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if let Some(ordering) = self.partial_cmp(other) {
            ordering
        } else {
            // Choose what to do with NaNs, for example:
            panic!("Cannot order invalid floats")
        }
    }
}

impl QuadricError {
    pub fn new(error: f64, e: EdgeID) -> Self {
        QuadricError(cmp::Reverse(error), e)
    }
}
