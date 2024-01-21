use std::sync::{Arc, Mutex};

use gpu_allocator::vulkan::{Allocator, AllocatorVisualizer};

use super::window::GuiWindow;

pub struct AllocatorVisualiserWindow {
    allocator: Arc<Mutex<Allocator>>,
    visualiser: AllocatorVisualizer,
    visualiser_open: bool,
}

impl AllocatorVisualiserWindow {
    pub fn new(allocator: Arc<Mutex<Allocator>>) -> Self {
        Self {
            allocator,
            visualiser: AllocatorVisualizer::new(),
            visualiser_open: false,
        }
    }
}

impl GuiWindow for AllocatorVisualiserWindow {
    fn draw(&mut self, ctx: &egui::Context) {
        self.visualiser.render_breakdown_window(
            ctx,
            &self.allocator.lock().unwrap(),
            &mut self.visualiser_open,
        );
    }

    fn state(&mut self) -> (&mut bool, &str) {
        (&mut self.visualiser_open, "Alloc Vis")
    }
}
