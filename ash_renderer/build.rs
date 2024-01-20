// #![cfg(not(feature = "shaderc"))]

use std::fs;

#[cfg(not(feature = "shaderc"))]
fn main() {
    //TODO: better way to skip build script
}

#[cfg(feature = "shaderc")]
fn main() {
    // Tell Cargo that if any shaders change, to rerun this build script.

    use std::{path::PathBuf, str::FromStr};

    println!("cargo:rerun-if-changed=shaders/src");

    // Compile all shaders to spirv::1.6

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_spirv(shaderc::SpirvVersion::V1_6);
    options.add_macro_definition("EP", Some("main"));
    options.set_include_callback(
        |requested_source, type_, requesting_source, include_depth| {
            if include_depth > 8 {
                panic!("Include depth too high, likely loop detected")
            }

            let content = match type_ {
                shaderc::IncludeType::Relative => {
                    let mut path = PathBuf::from_str(requesting_source).unwrap();
                    path.pop(); // Remove file name
                    path.push(requested_source); // Push requested name;
                    fs::read_to_string(path).unwrap()
                }
                shaderc::IncludeType::Standard => fs::read_to_string(requested_source).unwrap(),
            };

            Ok(shaderc::ResolvedInclude {
                resolved_name: requested_source.to_owned(),
                content,
            })
        },
    );

    let entries = fs::read_dir("shaders/src").unwrap();
    let mut success = true;
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
            Some("task") => shaderc::ShaderKind::Task,
            x => {
                println!("Cannot read shader {:?}", x);
                continue;
            }
        };

        let local_name = path.to_str().unwrap().to_owned();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();

        let shader = compiler.compile_into_spirv(
            &fs::read_to_string(&path).unwrap(),
            kind,
            &local_name,
            "main",
            Some(&options),
        );

        match shader {
            Ok(shader) => {
                let mut out_path = path.to_owned();
                out_path.pop();
                out_path.pop();

                out_path.push("spv");
                out_path.push(name);

                println!("{out_path:?} : {:#?}", shader.get_warning_messages());

                fs::write(out_path, shader.as_binary_u8()).unwrap();
            }
            Err(e) => {
                println!("ERROR: {e:#?}");
                success = false
            }
        }
    }

    if !success {
        panic!("Failed to compile all shaders, see logs above.");
    }
}
