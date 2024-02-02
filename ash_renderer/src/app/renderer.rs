use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::components::camera::Camera;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    draw_pipelines::DrawPipeline,
    gui::{gui::Gui, window::GuiWindow},
    screen::Screen,
    utility::{
        pooled::descriptor_pool::DescriptorPool, render_pass::RenderPass, sync::SyncObjects,
    },
};

use super::{
    fps_limiter::FPSMeasure,
    scene::{Scene, SceneEvent},
};
#[derive(Debug, Clone, Copy, Event)]
pub enum MeshDrawingPipelineType {
    IndirectTasks,
    DrawIndirect,
    ComputeCulledMesh,
    ComputeCulledIndices,
    None,
}
pub struct Renderer {
    pub allocator: Arc<Mutex<Allocator>>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub draw_pipeline: Box<dyn DrawPipeline>,
    pub descriptor_pool: Arc<DescriptorPool>,
    pub gui: Gui,
    pub windows: Vec<Box<dyn GuiWindow>>,
    pub current_pipeline: MeshDrawingPipelineType,
    pub sync_objects: SyncObjects,
    pub query: bool,
    pub current_frame: usize,
    pub is_framebuffer_resized: bool,
    pub cluster_count: u32,
    pub app_info_open: bool,
    // Make sure to drop the core last
    pub screen: Screen,
    pub core: Arc<Core>,
}

impl Renderer {
    pub fn recreate_swapchain(&mut self, scene: &Scene, cam: &mut Camera) {
        self.core.device.wait_device_idle();

        let size = self.window_ref().inner_size();
        {
            cam.on_resize(&size);
        }

        //self.cleanup_swapchain();

        self.screen.remake_swapchain(
            self.graphics_queue,
            &self.render_pass,
            self.allocator.clone(),
        );

        self.draw_pipeline.init_swapchain(
            &self.core,
            &self.screen,
            self.cluster_count,
            scene.uniform_transform_buffer.item_len() as _,
            &self.render_pass,
        );

        // Egui Integration
        self.gui.update_swapchain(self.screen.swapchain());
    }

    pub fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    pub fn window_ref(&self) -> &winit::window::Window {
        &self.core.window
    }
    pub fn get_allocator(&self) -> MutexGuard<Allocator> {
        self.allocator.lock().unwrap()
    }
    pub fn draw_gui(
        &mut self,
        scene: &mut ResMut<Scene>,
        image_index: usize,
        mut scene_events: EventWriter<SceneEvent>,
        mut draw_events: EventWriter<MeshDrawingPipelineType>,
        fps: &FPSMeasure,
    ) -> vk::CommandBuffer {
        let mut add_more = false;

        let cmd = self.gui.draw(image_index, |ctx| {
            for w in &mut self.windows {
                w.draw(ctx);
            }

            egui::Window::new("App Config")
                .open(&mut self.app_info_open)
                .show(ctx, |ui| {
                    {
                        ui.checkbox(&mut scene.freeze_pos, "Freeze");
                        ui.add(
                            egui::Slider::new(&mut scene.target_error, 0.0..=1.0)
                                .text("Target Error"),
                        );
                    }

                    ui.add(fps);
                    if self.query {
                        self.draw_pipeline.stats_gui(ui, image_index);
                    }
                    ui.label(format!("Current Pipeline: {:?}", self.current_pipeline));

                    ui.add_enabled_ui(self.core.device.features.mesh_shader, |ui| {
                        if ui.button("Indirect Tasks").clicked() {
                            draw_events.send(MeshDrawingPipelineType::IndirectTasks)
                        }
                    });

                    ui.add_enabled_ui(false, |ui| {
                        if ui.button("Compute Culled Mesh").clicked() {
                            draw_events.send(MeshDrawingPipelineType::ComputeCulledMesh)
                        }
                    });

                    if ui.button("Compute Culled Indices").clicked() {
                        draw_events.send(MeshDrawingPipelineType::ComputeCulledIndices)
                    }

                    if ui.button("Draw Full Res").clicked() {
                        draw_events.send(MeshDrawingPipelineType::DrawIndirect)
                    }

                    if ui.button("Add More Instances").clicked() {
                        add_more = true;
                    }

                    if ui.checkbox(&mut self.query, "Enable Queries").clicked() {
                        draw_events.send(self.current_pipeline)
                    }
                });

            egui::TopBottomPanel::top("test").show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    for w in &mut self.windows {
                        let (open, name) = w.state();
                        ui.checkbox(open, name);
                    }
                    ui.checkbox(&mut self.app_info_open, "App Info");
                });
            });
        });

        if add_more {
            scene_events.send(SceneEvent::AddInstances(50));
        }

        cmd
    }
}
