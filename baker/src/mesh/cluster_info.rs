use common::{BoundingSphere, OriginCone};



/// Information for a partition on layer n
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClusterInfo {
    /// Group in the previous LOD layer (LOD`n-1`) we have been attached to. LOD0 will have none
    pub child_group_index: Option<usize>,
    /// Group in this layer. will be usize::MAX if not yet grouped, but always valid in a loaded asset
    pub group_index: usize,
    /// For culling purposes - smallest bounding sphere for the partition
    pub tight_bound: BoundingSphere,
    pub tight_cone: OriginCone,

    /// Number of triangles
    pub num_tris: usize,
}
