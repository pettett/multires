use std::{thread, time};

use baker::{
    lod::{
        lod_chain::simplify_lod_chain, meshopt_chain::meshopt_simplify_lod_chain,
        meshopt_multiresolution::meshopt_multiresolution,
        multiresolution::group_and_partition_and_simplify,
    },
    mesh::half_edge_mesh::HalfEdgeMesh,
    Args, Mode, Simplifier, STARTING_CLUSTER_SIZE,
};
use clap::Parser;
use common::Asset;

// Castle
// https://sketchfab.com/3d-models/upnor-castle-a08280d12911401aa6022c1a58f2b49a

fn bake_mesh(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    mode: Mode,
    simplifier: Simplifier,
) {
    println!("Loading from gltf!");
    let start_time = time::Instant::now();
    let (mut mesh, tri_mesh) = HalfEdgeMesh::from_gltf(&input);

    // mesh.assert_valid().unwrap();``

    let num_contiguous = mesh.partition_contiguous();

    println!("We have {:?} Contiguous segments", num_contiguous);

    if num_contiguous.len() > 1 {
        mesh.filter_tris_by_cluster(
            num_contiguous
                .iter()
                .enumerate()
                .max_by_key(|(_, &x)| x)
                .unwrap()
                .0
                + 1,
        )
        .unwrap();
    }

    let num_contiguous = mesh.partition_contiguous();

    assert_eq!(num_contiguous.len(), 1);

    let name = input.to_str().unwrap().to_owned();

    let multi_res = match (mode, simplifier) {
        (Mode::DAG, Simplifier::Quadrics) => {
            group_and_partition_and_simplify(mesh, tri_mesh, name, STARTING_CLUSTER_SIZE).unwrap()
        }
        (Mode::DAG, Simplifier::Meshopt) => {
            meshopt_multiresolution(mesh, tri_mesh, name, STARTING_CLUSTER_SIZE).unwrap()
        }
        (Mode::Chain, Simplifier::Meshopt) => {
            meshopt_simplify_lod_chain(tri_mesh, name).expect("Failed to extract meshopt stuff")
        }
        (Mode::Chain, Simplifier::Quadrics) => simplify_lod_chain(mesh, tri_mesh, name).unwrap(),
    };

    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());

    multi_res.save(output).unwrap();

    let finish_time = time::Instant::now();

    println!("Finished in {}ms", (finish_time - start_time).as_millis())
}

fn main() {
    let args = Args::parse();

    let mut threads: Vec<thread::JoinHandle<()>> = Vec::new();

    for entry in glob::glob(&args.input()).expect("Failed to read glob") {
        match entry {
            Ok(path) => {
                let mut name = path.file_name().unwrap().to_str().unwrap().to_owned();
                name.push_str(".bin");
                let output = args.output().join(name);
                println!("{:?}", path.display());
                println!("{:?}", output.display());

                let mode = args.mode();
                let simp = args.simplifier();
                threads.push(thread::spawn(move || bake_mesh(path, output, mode, simp)))
            }
            Err(e) => println!("{:?}", e),
        }
    }

    for t in threads {
        t.join().unwrap();
    }

    //let mesh_name = "../assets/torrin_main.glb";
    // let mesh_name = "../assets/rock.glb";
    //let mesh_name = "../assets/dragon_high.glb";
    // let mesh_name = "../assets/pole.glb";
    // let mesh_name = "../assets/monk_250k.glb";
    //let mesh_name = "../assets/sphere.glb";
}
