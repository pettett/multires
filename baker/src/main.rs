use baker::{
    apply_simplification, group_and_partition_and_simplify, group_and_partition_full_res,
    mesh::winged_mesh::WingedMesh,
};

fn main() {
    let mesh_name = "../assets/monk_60k.glb";

    println!("Loading from gltf!");
    let (working_mesh, verts) = WingedMesh::from_gltf(mesh_name);

    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
    group_and_partition_and_simplify(working_mesh, &verts, mesh_name.to_owned());
}
