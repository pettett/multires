#[derive(Debug, Clone, bincode::Decode, bincode::Encode, Default)]
pub struct Meshlet {
    //indices: Vec<u32>,
    local_indices: Vec<u8>,
    local_strip_indices: Vec<u8>,
    verts: Vec<u32>,
}

impl Meshlet {
    pub fn from_local_indices(local_indices: Vec<u8>, verts: Vec<u32>) -> Self {
        Self {
            local_strip_indices: Vec::new(),
            //indices: Vec::new(),
            local_indices,
            verts,
        }
    }

    pub fn vert_count(&self) -> usize {
        self.verts.len()
    }

    pub fn local_to_global_vert_index(&self, local_vert: u32) -> u32 {
        self.verts[local_vert as usize]
    }

    pub fn local_indices(&self) -> &[u8] {
        self.local_indices.as_ref()
    }

    pub fn calc_indices_to_vec(&self, indices: &mut Vec<u32>) {
        for &l in &self.local_indices {
            indices.push(self.verts[l as usize]);
        }
    }

    pub fn calc_indices(&self) -> Vec<u32> {
        let mut indices = Vec::with_capacity(self.local_indices.len());
        self.calc_indices_to_vec(&mut indices);
        indices
    }

    pub fn verts(&self) -> &[u32] {
        self.verts.as_ref()
    }

    pub fn local_strip_indices(&self) -> &[u8] {
        &self.local_strip_indices
    }
}
