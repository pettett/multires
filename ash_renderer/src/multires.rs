use bytemuck::Zeroable;
use common::MeshCluster;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PackedTri {
    pub t: [u8; 4],
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

pub fn generate_meshlets(
    cluster_order: &[&MeshCluster],
) -> (Vec<GpuCluster>, Vec<GpuMeshlet>, Vec<GpuMeshlet>) {
    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets = Vec::with_capacity(cluster_order.len());
    let mut clusters = Vec::with_capacity(cluster_order.len());
    let mut strip_meshlets = Vec::with_capacity(cluster_order.len());

    for cluster in cluster_order {
        let mut s = GpuCluster::zeroed();

        s.meshlet_start = meshlets.len() as _;

        for meshlet in cluster.meshlets() {
            //TRIANGLE LIST
            {
                let mut m = GpuMeshlet::zeroed();

                assert!(meshlet.local_indices().len() <= m.triangles.len() * 3);
                assert!(meshlet.verts().len() <= m.vertices.len());

                for (tri_idx, tri) in meshlet.local_indices().chunks(3).enumerate() {
                    assert_eq!(tri.len(), 3);
                    // Pack triangles into a single uint
                    for i in 0..3 {
                        assert!(tri[i] < meshlet.vert_count() as _);

                        m.triangles[tri_idx].t[i] = tri[i] as _;
                    }
                }
                for (i, &vert) in meshlet.verts().iter().enumerate() {
                    m.vertices[i] = vert;
                }
                m.tri_count = (meshlet.local_indices().len() / 3) as _;
                m.vertex_count = meshlet.verts().len() as _;

                meshlets.push(m);
            }

            //TRIANGLE STRIP
            {
                let mut stripped_m = GpuMeshlet::zeroed();

                for (tri_idx, indices) in meshlet.local_strip_indices().chunks(4).enumerate() {
                    for i in 0..indices.len() {
                        assert!(indices[i] < meshlet.vert_count() as _);

                        stripped_m.triangles[tri_idx].t[i] = indices[i] as _;
                    }
                }

                for (i, &vert) in meshlet.verts().iter().enumerate() {
                    stripped_m.vertices[i] = vert;
                }

                stripped_m.tri_count = (meshlet.local_strip_indices().len().div_ceil(4)) as _;
                stripped_m.vertex_count = meshlet.verts().len() as _;
                strip_meshlets.push(stripped_m);
            }
        }

        s.meshlet_count = meshlets.len() as u32 - s.meshlet_start;
        clusters.push(s);
    }

    (clusters, meshlets, strip_meshlets)
}
