pub mod mesh_iter;
pub mod vertex;
pub mod winged_mesh;

use common::{MeshLevel, Meshlet, SubMesh};

use rand::Rng;
use vertex::VertID;

use crate::winged_mesh::WingedMesh;

pub fn to_mesh_layer(mesh: &WingedMesh) -> MeshLevel {
    MeshLevel {
        partition_indices: mesh.get_partition(),
        group_indices: mesh.get_group(),
        indices: grab_indicies(&mesh),
        meshlets: vec![], //generate_meshlets(&mesh),
        submeshes: generate_submeshes(mesh),
        partitions: mesh.partitions.clone(),
        groups: mesh.groups.clone(),
    }
}

pub fn grab_indicies(mesh: &WingedMesh) -> Vec<u32> {
    let mut indices = Vec::with_capacity(mesh.face_count() * 3);

    for f in mesh.faces().values() {
        indices.extend(mesh.triangle_from_face(f));
    }

    // Validation
    //for i in &indices {
    //    verts[*i as usize];
    //}

    indices
}

pub fn generate_meshlets(mesh: &WingedMesh) -> Vec<Meshlet> {
    println!("Generating meshlets!");

    // Precondition: partition indexes completely span in some range 0..N
    let mut meshlets: Vec<_> = (0..=mesh.partition_count())
        .map(|_| (Meshlet::default()))
        .collect();

    for face in mesh.faces().values() {
        let verts = mesh.triangle_from_face(face);

        let m = meshlets.get_mut(face.part as usize).unwrap();

        for v in 0..3 {
            let vert = verts[v] as u32;

            // If unique, add to list
            let idx = (0..m.vertex_count as usize).find(|j| m.vertices[*j] == vert);

            let idx = if let Some(idx) = idx {
                idx as u32
            } else {
                m.vertex_count += 1;
                m.vertices[m.vertex_count as usize - 1] = vert;
                m.vertex_count - 1
            };

            m.indices[m.index_count as usize + v] = idx;
        }
        m.index_count += 3;
    }

    let total_indices: u32 = meshlets.iter().map(|m| m.index_count).sum();
    let avg_indices = total_indices / meshlets.len() as u32;
    let max_indices = meshlets.iter().map(|m| m.index_count).max().unwrap();
    let total_verts: u32 = meshlets.iter().map(|m| m.vertex_count).sum();
    let avg_verts = total_verts / meshlets.len() as u32;
    let max_verts = meshlets.iter().map(|m| m.vertex_count).max().unwrap();

    println!("avg_indices: {avg_indices}/378 max_indices: {max_indices}/378 avg_verts: {avg_verts}/64 max_verts: {max_verts}/64");

    meshlets
}

/// Debug code to generate meshlets with no max size. Used for testing partition trees with no remeshing
pub fn generate_submeshes(mesh: &WingedMesh) -> Vec<SubMesh> {
    println!("Generating meshlets!");

    let inds = 5.0 / mesh.face_count() as f32;

    // Precondition: partition indexes completely span in some range 0..N
    let mut submeshes: Vec<_> = (0..mesh.partition_count())
        .map(|part| {
            let gi = mesh.partitions[part].group_index;
            let g = &mesh.groups[gi];
            SubMesh::new(
                0.0,
                g.monotonic_bound.center(),
                g.monotonic_bound.radius(),
                mesh.partitions[part].tight_bound.radius(),
                gi,
            )
        })
        .collect();

    for face in mesh.faces().values() {
        let verts = mesh.triangle_from_face(face);

        let m = submeshes.get_mut(face.part as usize).unwrap();

        for v in 0..3 {
            let vert = verts[v] as u32;
            m.indices.push(vert);
        }

        m.error += inds;
    }

    submeshes
}

pub fn reduce_mesh(meshlets: &[Meshlet], mut mesh: WingedMesh) -> WingedMesh {
    let mut rng = rand::thread_rng();

    for (i, m) in meshlets.iter().enumerate() {
        //println!("Reducing meshlet {i}/{}", meshlets.len());

        // reduce triangle count in meshlet by half

        let mut tris = m.index_count / 3;
        let target = tris * 3 / 4;

        let mut todo: Vec<_> = (0..m.vertex_count as usize).collect();

        while tris > target && todo.len() > 0 {
            // Pick a random edge in the mesh for now
            let i = rng.gen_range(0..todo.len());
            let v = VertID(m.vertices[todo[i]] as usize);
            todo.swap_remove(i);

            // println!("{i} {tris}/ {target}, {v:?}");

            let Some(i) = mesh[v].edge else {
                continue;
            };

            let valid_edge = v.is_local_manifold(&mesh, true);

            if valid_edge {
                // all faces are within the partition, we can safely collapse one of the edges

                mesh.collapse_edge(i);

                tris -= 2;

                // println!("Collapsed edge {e:?}");

                //break;
            }
        }
    }

    mesh
}
