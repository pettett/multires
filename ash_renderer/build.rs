use std::fs;

fn main() {
    // Tell Cargo that if any shaders change, to rerun this build script.
    println!("cargo:rerun-if-changed=shaders/src");

    let entries = fs::read_dir("shaders/src").unwrap();

    // Compile all shaders to spirv::1.6

    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.add_macro_definition("EP", Some("main"));

    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        println!("Compiling: {}", path.display());

        let ext = match path.extension() {
            Some(str) => str.to_str(),
            _ => None,
        };

        let kind = match ext {
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("frag") => shaderc::ShaderKind::Fragment,
            Some("mesh") => shaderc::ShaderKind::Mesh,
            x => {
                println!("Cannot read shader {:?}", x);
                continue;
            }
        };

        let name = entry.file_name().to_str().unwrap().to_owned();

        let shader = compiler
            .compile_into_spirv(
                &fs::read_to_string(&path).unwrap(),
                kind,
                &name,
                "main",
                Some(&options),
            )
            .unwrap();

        let mut out_path = path.to_owned();
        out_path.pop();
        out_path.pop();

        out_path.push("spv");
        out_path.push(name);

        println!("{out_path:?}");

        fs::write(out_path, shader.as_binary_u8()).unwrap();
    }
}