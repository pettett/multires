use std::fs;

use ash_renderer::{
    app::{app::App, window::ProgramProc},
    Config,
};

fn main() {
    let filename = "renderer.toml";
    let contents = match fs::read_to_string(filename) {
        Ok(c) => c,
        Err(_) => {
            panic!("Could not read file `{}`", filename);
        }
    };

    let config: Config = toml::from_str(&contents).unwrap();

    let program_proc = ProgramProc::new();
    let vulkan_app = App::new(&program_proc.event_loop, &config);

    program_proc.main_loop(vulkan_app);
}
// -------------------------------------------------------------------------------------------
