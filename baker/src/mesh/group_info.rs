use common::BoundingSphere;

use super::half_edge_mesh::HalfEdgeMesh;

/// Information for a group on layer n
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupInfo {
    /// Partitions in LOD`n-1` that we were created from. Will be empty in LOD0
    //pub child_partitions: Vec<usize>,
    // Partitions that we created by subdividing ourselves
    pub clusters: Vec<usize>,

    /// Monotonic bounds for error function of partitions. Includes bounds of all other partitions in the group,
    /// and all partitions we are children to
    pub saturated_bound: BoundingSphere,

    // TODO: Evaluate edge length error vs quadric error
    pub saturated_error: f32,
}

impl GroupInfo {
    pub fn num_tris(&self, mesh: &HalfEdgeMesh) -> usize {
        self.clusters
            .iter()
            .map(|&i| mesh.clusters[i].num_tris)
            .sum()
    }
}
