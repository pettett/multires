use std::{collections::HashMap, sync::Arc, time::Instant};

use super::partition_graph::{
    evaluate_node, AllMyNodeTemplates, MyEditorState, MyGraphState, MyResponse,
};
use crate::{
    components::{camera_uniform::CameraUniform, mesh::Mesh},
    core::Renderer,
};
use bevy_ecs::{
    component::Component,
    system::{Commands, Query, Resource},
};
use common_renderer::components::camera::Camera;
use egui::{FontDefinitions, Id, LayerId, Pos2, Rect, TextStyle, Ui, Vec2};
use egui_node_graph::NodeResponse;
use egui_wgpu::renderer::ScreenDescriptor;
use egui_winit::screen_size_in_pixels;
use winit::{event::Event, window::Window};

#[derive(Resource)]
pub struct Gui {
    last_frame: Instant,
    renderer: egui_wgpu::renderer::Renderer,
    //puffin_ui : puffin_imgui::ProfilerUi,
    // The `GraphEditorState` is the top-level object. You "register" all your
    // custom types by specifying it as its generic parameters.
    state: MyEditorState,

    user_state: MyGraphState,
}

impl Gui {
    pub fn render_pass(
        &mut self,
        renderer: &Renderer,
        state: &mut egui_winit::State,
        ctx: &egui::Context,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        mut meshes: Query<&mut Mesh>,
        mut camera: Query<&mut Camera>,
        commands: &mut Commands,
    ) {
        let _delta_s = self.last_frame.elapsed();
        let now = Instant::now();
        //self.imgui.io_mut().update_delta_time(now - self.last_frame);
        self.last_frame = now;

        let full_output = ctx.run(state.take_egui_input(&renderer.window()), |ctx| {
            // egui::TopBottomPanel::top("menu").show(&ctx, |ui| {
            //     egui::menu::bar(ui, |ui| {
            //         for mut window in windows.iter_mut() {
            //             ui.checkbox(&mut window.opened, "Open");
            //         }

            //         //ui.menu_button("File", |ui| if ui.button("Open").clicked() {});
            //     });
            // });

            egui::Window::new("Mesh View")
                .min_height(100.0)
                .min_width(100.0)
                .show(&ctx, |ui| {
                    for mut camera in camera.iter_mut() {
                        ui.add(
                            egui::widgets::DragValue::new(&mut camera.part_highlight)
                                .prefix("highlight partition: "),
                        );
                        for mut mesh in meshes.iter_mut() {
                            let max = mesh.remeshes() - 1;
                            ui.add(
                                egui::widgets::DragValue::new(&mut mesh.remesh)
                                    .prefix("layer: ")
                                    .clamp_range(0..=max),
                            );
                        }
                    }

                    ui.add_space(10.0);

                    let graph_response = egui::ScrollArea::both()
                        .show(ui, |ui| {
                            // …
                            self.state.draw_graph_editor(
                                ui,
                                AllMyNodeTemplates,
                                &mut self.user_state,
                                Vec::default(),
                            )
                        })
                        .inner;

                    for node_response in graph_response.node_responses {
                        // Here, we ignore all other graph events. But you may find
                        // some use for them. For example, by playing a sound when a new
                        // connection is created
                        if let NodeResponse::User(user_event) = node_response {
                            match user_event {
                                MyResponse::SetActiveNode(node) => {
                                    self.user_state.active_node = Some(node)
                                }
                                MyResponse::ClearActiveNode => self.user_state.active_node = None,
                            }
                        }
                    }

                    if let Some(node) = self.user_state.active_node {
                        if self.state.graph.nodes.contains_key(node) {
                            let text =
                                match evaluate_node(&self.state.graph, node, &mut HashMap::new()) {
                                    Ok(value) => format!("The result is: {:?}", value),
                                    Err(err) => format!("Execution error: {}", err),
                                };
                            ctx.debug_painter().text(
                                egui::pos2(10.0, 35.0),
                                egui::Align2::LEFT_TOP,
                                text,
                                TextStyle::Button.resolve(&ctx.style()),
                                egui::Color32::WHITE,
                            );
                        } else {
                            self.user_state.active_node = None;
                        }
                    }
                });

            // for mut window in windows.iter_mut() {
            //     window.draw_window(ctx, renderer, &mut self.renderer, commands);
            // }
        });
        //self.platform
        //    .prepare_frame(self.imgui.io_mut(), renderer.window())
        //    .expect("Failed to prepare frame");
        //
        //let ui = self.imgui.frame();

        //if let Some(_menu_bar) = ui.begin_main_menu_bar() {
        //
        //
        //
        //}

        // if self.last_cursor != ui.mouse_cursor() {
        //     self.last_cursor = ui.mouse_cursor();
        //     self.platform.prepare_render(ui, renderer.window());
        // }

        {
            let paint_jobs = ctx.tessellate(full_output.shapes);

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: renderer.window().inner_size().into(),
                pixels_per_point: state.pixels_per_point(),
            };

            for (id, image_delta) in &full_output.textures_delta.set {
                self.renderer
                    .update_texture(renderer.device(), renderer.queue(), *id, image_delta);
            }
            self.renderer.update_buffers(
                renderer.device(),
                renderer.queue(),
                encoder,
                &paint_jobs,
                &screen_descriptor,
            );

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.renderer
                .render(&mut rpass, &paint_jobs, &screen_descriptor);

            // self.renderer
            //     .render(
            //         self.imgui.render(),
            //         renderer.queue(),
            //         renderer.device(),
            //         &mut rpass,
            //     )
            //     .expect("Rendering failed");
        }
    }

    pub fn init(renderer: &Renderer) -> Self {
        // Set up dear imgui

        let renderer =
            egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1);

        Self {
            last_frame: Instant::now(),
            renderer,
            state: MyEditorState::default(),
            user_state: MyGraphState::default(),
        }
    }
    pub fn handle_event<T>(&mut self, window: &Window, event: &Event<T>) -> bool {
        //self.platform
        //    .handle_event(self.imgui.io_mut(), window, event);
        //
        //self.imgui.io().want_capture_mouse

        false
    }
}
