use std::collections::BTreeSet;

use common::BoundingSphere;

/// Information for a group on layer n
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupInfo {
    /// Partitions in LOD`n-1` that we were created from. Will be empty in LOD0
    //pub child_partitions: Vec<usize>,
    // Partitions that we created by subdividing ourselves
    pub clusters: Vec<usize>,

    /// Indexes of all groups that touch this one and could be effected by an edge collapse in this group
    // pub group_neighbours: BTreeSet<usize>,
    pub tris: usize,
    /// Monotonic bounds for error function of partitions. Includes bounds of all other partitions in the group,
    /// and all partitions we are children to
    pub monotonic_bound: BoundingSphere,
}
