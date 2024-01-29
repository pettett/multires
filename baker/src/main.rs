use baker::{group_and_partition_and_simplify, mesh::winged_mesh::WingedMesh};
use clap::Parser;
use common::asset::Asset;

// Castle
// https://sketchfab.com/3d-models/upnor-castle-a08280d12911401aa6022c1a58f2b49a

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser)]
struct Args {
    /// The path to the mesh to read
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    output: std::path::PathBuf,
}

fn main() {
    let args = Args::parse();

    for entry in glob::glob(&args.input).expect("Failed to read glob") {
        match entry {
            Ok(path) => println!("{:?}", path.display()),
            Err(e) => println!("{:?}", e),
        }
    }

    //let mesh_name = "../assets/torrin_main.glb";
    // let mesh_name = "../assets/rock.glb";
    //let mesh_name = "../assets/dragon_high.glb";
    // let mesh_name = "../assets/pole.glb";
    // let mesh_name = "../assets/monk_250k.glb";
    //let mesh_name = "../assets/sphere.glb";

    println!("Loading from gltf!");
    let (mut mesh, verts, normals) = WingedMesh::from_gltf(&args.input);

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

    //group_and_partition_full_res(working_mesh, &verts, mesh_name.to_owned());
    //apply_simplification(working_mesh, &verts, mesh_name.to_owned());
    let lods = group_and_partition_and_simplify(mesh, &verts, &normals);

    common::MultiResMesh {
        name: args.input,
        verts: verts
            .iter()
            .zip(normals.iter())
            .map(|(v, n)| common::MeshVert {
                pos: [v.x, v.y, v.z, 1.0],
                normal: [n.x, n.y, n.z, 1.0],
            })
            .collect(),
        // layer_1_indices: indices.clone(),
        lods,
    }
    .save(args.output)
    .unwrap();
}
