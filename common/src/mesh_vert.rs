#[repr(C)]
#[derive(Clone, bincode::Decode, bincode::Encode, bytemuck::Pod, bytemuck::Zeroable, Copy)]
pub struct MeshVert {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}

impl meshopt::DecodePosition for MeshVert {
    fn decode_position(&self) -> [f32; 3] {
        [self.pos[0], self.pos[1], self.pos[2]]
    }
}
