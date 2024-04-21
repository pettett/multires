use crate::evaluation_mesh_error::sample_multires_error;

mod evaluation_mesh_error;
mod line;
mod triangle;

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
