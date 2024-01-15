use bytemuck::Zeroable;

use crate::{utility::buffer::Buffer, ModelUniformBufferObject};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuMeshlet {
    pub vertices: [u32; 64],
    pub indices: [u32; 378], // 126 triangles => 378 indices
    pub vertex_count: u32,
    pub index_count: u32,
}

unsafe impl bytemuck::Zeroable for GpuMeshlet {}
unsafe impl bytemuck::Pod for GpuMeshlet {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuSubmesh {
    pub meshlet_start: u32,
    pub meshlet_count: u32,
}

unsafe impl bytemuck::Zeroable for GpuSubmesh {}
unsafe impl bytemuck::Pod for GpuSubmesh {}

pub struct Model {
    model: ModelUniformBufferObject,
    model_buffers: Vec<Buffer>,
    dirty: bool,
}

pub fn generate_meshlets(mesh: &common::MultiResMesh) -> (Vec<GpuSubmesh>, Vec<GpuMeshlet>) {
    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets = Vec::new();
    let mut submeshes = Vec::new();

    for lod in mesh.lods.iter().rev() {
        for submesh in &lod.submeshes {
            let mut s = GpuSubmesh::zeroed();

            s.meshlet_start = meshlets.len() as _;

            for c in 0..submesh.colour_count() {
                let mut m = GpuMeshlet::zeroed();

                assert!(submesh.indices_for_colour(c).len() <= 126 * 3);

                for &vert in submesh.indices_for_colour(c) {
                    // If unique, add to list
                    let idx = (0..m.vertex_count as usize).find(|&j| m.vertices[j] == vert);

                    let idx = if let Some(idx) = idx {
                        idx as u32
                    } else {
                        assert!((m.vertex_count as usize) < 64);

                        m.vertex_count += 1;

                        m.vertices[m.vertex_count as usize - 1] = vert;
                        m.vertex_count - 1
                    };

                    m.indices[m.index_count as usize] = idx;

                    m.index_count += 1;
                }

                meshlets.push(m);
            }

            s.meshlet_count = meshlets.len() as u32 - s.meshlet_start;
            submeshes.push(s);
        }
    }

    (submeshes, meshlets)
}
