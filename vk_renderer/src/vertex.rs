use vulkano::buffer::BufferContents;

#[derive(BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
#[repr(C)]
pub struct PosVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}
