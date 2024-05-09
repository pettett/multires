use std::env;

use evaluation::evaluation_mesh_error::sample_multires_error;

fn main() {
    let args: Vec<String> = env::args().collect();

    sample_multires_error(&args[1])
}
