use ash_renderer::app::{app::App, window::ProgramProc};

fn main() {
    let program_proc = ProgramProc::new();
    let vulkan_app = App::new(&program_proc.event_loop);

    program_proc.main_loop(vulkan_app);
}
// -------------------------------------------------------------------------------------------
