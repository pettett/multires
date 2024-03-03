#[repr(C)]
#[derive(Clone, bincode::Decode, bincode::Encode, bytemuck::Pod, bytemuck::Zeroable, Copy)]
pub struct MeshVert {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}
