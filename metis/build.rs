extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let dst = cmake::Config::new("../../vendor").build();
    println!("cargo:rustc-link-search={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=metis");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("couldn't write bindings!");

    println!("cargo:rerun-if-changed=wrapper.h");
}
