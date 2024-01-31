use bytemuck::Zeroable;
use common::MeshCluster;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PackedTri {
    pub t: [u8; 3],
    pub _0: u8,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuMeshlet {
    pub vertices: [u32; 64],
    pub triangles: [PackedTri; 126], // 126 triangles => 378 indices
    pub vertex_count: u32,
    pub tri_count: u32,
}

unsafe impl bytemuck::Zeroable for GpuMeshlet {}
unsafe impl bytemuck::Pod for GpuMeshlet {}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GpuCluster {
    pub meshlet_start: u32,
    pub meshlet_count: u32,
}

unsafe impl bytemuck::Zeroable for GpuCluster {}
unsafe impl bytemuck::Pod for GpuCluster {}

// pub struct Model {
//     model: ModelUniformBufferObject,
//     model_buffers: Vec<Buffer>,
//     dirty: bool,
// }

pub fn generate_meshlets(cluster_order: &[&MeshCluster]) -> (Vec<GpuCluster>, Vec<GpuMeshlet>) {
    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets = Vec::new();
    let mut clusters = Vec::new();

    for cluster in cluster_order {
        let mut s = GpuCluster::zeroed();

        s.meshlet_start = meshlets.len() as _;

        for c in 0..cluster.colour_count() {
            let mut m = GpuMeshlet::zeroed();

            assert!(cluster.indices_for_colour(c).len() <= m.triangles.len() * 3);

            for &vert in cluster.indices_for_colour(c) {
                // If unique, add to list
                let idx = (0..m.vertex_count as u8).find(|&j| m.vertices[j as usize] == vert);

                // Use byte addressing within the vertex array
                let idx = if let Some(idx) = idx {
                    idx
                } else {
                    assert!((m.vertex_count as usize) < m.vertices.len());

                    m.vertex_count += 1;

                    m.vertices[m.vertex_count as usize - 1] = vert;
                    (m.vertex_count - 1) as u8
                };

                // Pack triangles into a single uint
                let tri_idx = (m.tri_count as usize) / 3;
                let offset = (m.tri_count as usize) % 3;

                m.triangles[tri_idx].t[offset] = idx;

                m.tri_count += 1;
            }

            assert_eq!(m.tri_count % 3, 0);
            m.tri_count /= 3;

            meshlets.push(m);
        }

        s.meshlet_count = meshlets.len() as u32 - s.meshlet_start;
        clusters.push(s);
    }

    (clusters, meshlets)
}
