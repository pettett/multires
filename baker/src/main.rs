extern crate gltf;

pub mod winged_mesh;
use common::{asset::Asset, MultiResMesh};
use metis::PartitioningConfig; 
use std::time;

fn main() -> gltf::Result<()> {
    let mesh_name = "../assets/dragon_high.glb";

    println!("Loading from gltf!");
    let mesh = winged_mesh::WingedMesh::from_gltf(mesh_name)?;

    println!("Loaded winged edge mesh from gltf!");

    println!("Partitioning Graph!");
    let t1 = time::Instant::now();

    let clusters = mesh
        .partition(
            &PartitioningConfig {
                //force_contiguous_partitions: Some(true),
                ..Default::default()
            },
            1 + mesh.faces().len() as u32 / 128,
        )
        .unwrap();

    println!("time: {}ms", t1.elapsed().as_millis());

    let mut graph = petgraph::Graph::<i32, i32>::new();

    MultiResMesh {
        name: mesh_name.to_owned(),
        clusters,
    }
    .save()
    .unwrap();

    Ok(())
}
