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

#[cfg(test)]
mod tests {

    use bincode::{
        config::{self, BigEndian, Configuration},
        Encode,
    };

    use crate::Meshlet;

    #[test]
    fn test_local_indices() {
        let verts = vec![100, 200, 300];
        let m = Meshlet::from_local_indices(vec![0, 1, 2], verts.clone());

        assert_eq!(m.local_to_global_vert_index(1), 200);
        assert_eq!(m.calc_indices(), verts);
        let mut indices = vec![0];

        m.calc_indices_to_vec(&mut indices);

        assert_eq!(indices, vec![0, 100, 200, 300]);

        assert_eq!(m.verts(), &verts);
    }

    #[test]
    fn test_verts() {
        let verts = vec![100, 200, 300];
        let m = Meshlet::from_local_indices(vec![0, 1, 2], verts.clone());

        assert_eq!(m.vert_count(), 3);
        assert_eq!(m.local_to_global_vert_index(0), 100);
        assert_eq!(m.verts(), verts);
        assert_eq!(m.calc_indices(), verts);
        let mut verts2 = Vec::new();
        m.calc_indices_to_vec(&mut verts2);
        assert_eq!(verts2, verts);
    }

    #[test]
    fn test_bincode() {
        let config = config::standard();
        let verts = vec![100, 200, 300];
        let m = Meshlet::from_local_indices(vec![0, 1, 2], verts.clone());

        let e = bincode::encode_to_vec(&m, config).unwrap();

        let m2: Meshlet = bincode::decode_from_slice(&e, config).unwrap().0;

        assert_eq!(m.calc_indices(), m2.calc_indices())
    }
}
