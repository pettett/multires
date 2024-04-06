use crate::{bounding_sphere::BoundingSphere, meshlet::Meshlet, origin_cone::OriginCone};

#[derive(Debug, Clone, bincode::Decode, bincode::Encode)]
pub struct MeshCluster {
    meshlets: Vec<Meshlet>,
    // Bounding sphere for the submesh
    // Will only be used for culling, so uncomment later
    pub tight_bound: BoundingSphere,
    pub tight_cone: OriginCone,

    // Similarly, the bounding sphere must be enlarged to enclose the bounding spheres of all its children in the DAG,
    // in order to ensure a monotonic view-dependent error function.
    pub saturated_bound: BoundingSphere,
    pub lod: usize, //We should not need this - group indexes are consistent across LOD
    error: f32,
    group_index: usize,
    child_group_index: Option<usize>,
}

impl MeshCluster {
    pub fn new(
        error: f32,
        tight_bound: BoundingSphere,
        tight_cone: OriginCone,
        saturated_bound: BoundingSphere,
        lod: usize,
        group_index: usize,
        child_group_index: Option<usize>,
    ) -> Self {
        Self {
            meshlets: Vec::new(),
            tight_bound,
            tight_cone,

            saturated_bound,
            error,
            lod,
            group_index,
            child_group_index,
        }
    }

    // pub fn new_raw_temp(indices: Vec<u32>, lod: usize) -> Self {
    //     Self {
    //         meshlets: Vec::new(),
    //         tight_bound: Default::default(),
    //         tight_cone: Default::default(),
    //         saturated_bound: Default::default(),
    //         error: 0.0,
    //         lod,
    //         group_index: 0,
    //         child_group_index: None,
    //     }
    // }

    // pub fn reset_meshlets(&mut self) {
    //     self.meshlets.clear()
    // }

    pub fn add_meshlet(&mut self, m: Meshlet) {
        self.meshlets.push(m)
    }

    pub fn meshlet_for_colour(&self, colour: usize) -> &Meshlet {
        &self.meshlets[colour]
    }

    pub fn meshlet_for_colour_mut(&mut self, colour: usize) -> &mut Meshlet {
        &mut self.meshlets[colour]
    }

    pub fn meshlets(&self) -> &[Meshlet] {
        &self.meshlets
    }

    pub fn colour_count(&self) -> usize {
        self.meshlets.len()
    }

    pub fn index_count(&self) -> usize {
        self.meshlets.iter().map(|x| x.local_indices().len()).sum()
    }

    pub fn stripped_index_count(&self) -> usize {
        self.meshlets
            .iter()
            .map(|x| x.local_strip_indices().len())
            .sum()
    }

    pub fn error(&self) -> f32 {
        self.error
		// self.index_count() as f32 * 0.0001
    }

    pub fn group_index(&self) -> usize {
        self.group_index
    }

    pub fn child_group_index(&self) -> Option<usize> {
        self.child_group_index
    }
}
