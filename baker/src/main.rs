use std::thread;

use baker::{
    lod::{
        lod_chain::simplify_lod_chain, meshopt_chain::meshopt_simplify_lod_chain,
        multiresolution::group_and_partition_and_simplify,
    },
    mesh::winged_mesh::WingedMesh,
    Args, Mode,
};
use clap::Parser;
use common::Asset;

// Castle
// https://sketchfab.com/3d-models/upnor-castle-a08280d12911401aa6022c1a58f2b49a

fn bake_mesh(input: std::path::PathBuf, output: std::path::PathBuf, mode: Mode) {
    println!("Loading from gltf!");
    let (mut mesh, tri_mesh) = WingedMesh::from_gltf(&input);

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

    let multi_res = match mode {
        Mode::DAG => group_and_partition_and_simplify(mesh, tri_mesh, name).unwrap(),
        Mode::MeshoptLOD => {
            meshopt_simplify_lod_chain(tri_mesh, name).expect("Failed to extract meshopt stuff")
        }
        Mode::BakerLOD => simplify_lod_chain(mesh, tri_mesh, name).unwrap(),
    };

    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());

    multi_res.save(output).unwrap();
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
                threads.push(thread::spawn(move || bake_mesh(path, output, mode)))
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
