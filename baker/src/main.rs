use baker::{group_and_partition_and_simplify, mesh::winged_mesh::WingedMesh};

fn main() {
    let mesh_name = "../assets/torrin_main.glb";
    //let mesh_name = "../assets/rock.glb";
    //let mesh_name = "../assets/sphere.glb";

    println!("Loading from gltf!");
    let (mut mesh, verts) = WingedMesh::from_gltf(mesh_name);

    let num_contiguous = mesh.partition_contiguous();

    println!("We have {} Contiguous segments", num_contiguous);

    assert_eq!(num_contiguous, 1);
    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
    group_and_partition_and_simplify(mesh, &verts, mesh_name.to_owned());
}
