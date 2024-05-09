use std::mem;

use common::{MeshCluster, MeshVert, Meshlet, MultiResMesh, TriMesh};

pub fn meshopt_simplify_lod_chain(tri_mesh: TriMesh, name: String) -> anyhow::Result<MultiResMesh> {
    let verts = tri_mesh
        .verts
        .iter()
        .zip(tri_mesh.normals.iter())
        .map(|(v, n)| MeshVert {
            pos: [v.x, v.y, v.z, 1.0],
            normal: [n.x, n.y, n.z, 1.0],
        })
        .collect();

    let mut indices = tri_mesh.indices.to_vec();

    let mut multi_res = MultiResMesh {
        name,
        full_indices: tri_mesh.indices,
        verts,
        clusters: Vec::new(),
        group_count: 0,
    };

    let verts_adapter = meshopt::VertexDataAdapter::new(
        bytemuck::cast_slice(&multi_res.verts),
        mem::size_of::<MeshVert>(),
        0,
    )
    .unwrap();

    for i in 0..10 {
        let meshlets = meshopt::build_meshlets(&indices, &verts_adapter, 64, 124, 0.1);

        //cluster.error += inds * indices.len() as f32;
        let mut cluster = MeshCluster::default();

        for m in meshlets.iter() {
            //remap the meshlet-local to mesh local verts

            let mut verts = Vec::with_capacity(m.vertices.len());
            for &v in m.vertices {
                verts.push(v as u32);
            }

            cluster.add_meshlet(Meshlet::from_local_indices(m.triangles.to_vec(), verts));
        }

        cluster.lod = i;

        multi_res.clusters.push(cluster);

        indices = meshopt::simplify(
            &indices,
            &verts_adapter,
            indices.len() / 2,
            1.0,
            meshopt::SimplifyOptions::None,
            None,
        );

        println!("Simplified to {} indices", indices.len());

        if indices.len() == 0 {
            break;
        }
    }

    //opt_multires(&mut multi_res);

    Ok(multi_res)
}
