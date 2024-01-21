pub trait GuiWindow {
    fn draw(&mut self, ctx: &egui::Context);

    fn state(&mut self) -> (&mut bool, &str);
}
