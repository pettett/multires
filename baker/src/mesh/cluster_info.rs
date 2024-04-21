use common::{BoundingSphere, OriginCone};

/// Information for a partition on layer n
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterInfo {
    /// Group in the previous LOD layer (LOD`n-1`) we have been attached to. LOD0 will have none
    pub child_group_index: Option<usize>,
    /// Group in this layer. will be usize::MAX if not yet grouped, but always valid in a loaded asset
    group: Option<usize>,
    /// For culling purposes - smallest bounding sphere for the partition
    pub tight_bound: BoundingSphere,
    pub tight_cone: OriginCone,

    pub average_edge_length: f32,
    /// Number of triangles
    pub num_tris: usize,
}

impl ClusterInfo {
    pub fn new(group: Option<usize>, num_tris: usize) -> Self {
        Self {
            group,
            num_tris,
            ..Default::default()
        }
    }

    pub fn inherit(child_group_index: usize) -> Self {
        Self {
            child_group_index: Some(child_group_index),
            ..Default::default()
        }
    }

    pub fn group_index(&self) -> usize {
        self.group
            .expect("Attempted to get group index of orphan cluster")
    }

    pub fn try_get_group_index(&self) -> Option<usize> {
        self.group
    }

    /// Set the group index. Will panic if we attempt to set multiple times, as this is incorrect usage - a cluster cannot change group
    pub fn set_group_index_once(&mut self, group_index: usize) {
        assert_eq!(
            self.group, None,
            "Attempted to set group index of cluster multiple times"
        );
        self.group = Some(group_index);
    }

    pub fn group(&self) -> Option<usize> {
        self.group
    }
}

impl Default for ClusterInfo {
    fn default() -> Self {
        Self {
            child_group_index: None,
            group: None,
            tight_bound: Default::default(),
            tight_cone: Default::default(),
            average_edge_length: 0.0,
            num_tris: 0,
        }
    }
}
