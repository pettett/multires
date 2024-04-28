use evaluation::evaluation_mesh_error::sample_multires_error;

fn main() {
    for entry in glob::glob("assets/*.glb.bin").expect("Failed to read glob") {
        match entry {
            Ok(path) => {
                println!("{:?}", path.display());

                sample_multires_error(&path)
            }
            Err(e) => println!("{:?}", e),
        }
    }
}
