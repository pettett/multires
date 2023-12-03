use std::{collections::HashMap, thread, time::Instant};

use crate::{
    components::multi_res_mesh::{ErrorMode, MultiResMeshComponent, SubMeshComponent},
    core::Renderer,
};
use bevy_ecs::{
    entity::Entity,
    query::QueryState,
    system::{Commands, Query, Resource},
    world::World,
};
use common::graph::petgraph_to_svg;
use common_renderer::components::{camera::Camera, transform::Transform};
use egui::{Pos2, Vec2};
use egui_wgpu::renderer::ScreenDescriptor;
use glam::Vec3;
use winit::{event::Event, window::Window};

#[derive(Resource)]
pub struct Gui {
    last_frame: Instant,
    renderer: egui_wgpu::renderer::Renderer,
}

impl Gui {
    pub fn render_pass(
        &mut self,
        renderer: &Renderer,
        state: &mut egui_winit::State,
        ctx: &egui::Context,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        mut meshes: Query<&mut MultiResMeshComponent>,
        submeshes: &Query<(Entity, &SubMeshComponent)>,
        mut camera: Query<(&mut Camera, &Transform)>,
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
                    for (mut camera, transform) in camera.iter_mut() {
                        ui.add(
                            egui::widgets::DragValue::new(&mut camera.part_highlight)
                                .prefix("highlight partition: "),
                        );

                        for mut mesh in meshes.iter_mut() {
                            ui.add(
                                egui::widgets::DragValue::new(&mut mesh.focus_part)
                                    .prefix("focus partition: "),
                            );

                            ui.add(egui::widgets::Checkbox::new(&mut mesh.freeze, "Freeze"));

                            egui::ComboBox::from_label("Mode ")
                                .selected_text(format!("{:?}", &mut mesh.error_calc))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut mesh.error_calc,
                                        ErrorMode::MaxError { error: 0.1 },
                                        "Max Error",
                                    );
                                    ui.selectable_value(
                                        &mut mesh.error_calc,
                                        ErrorMode::PointDistance {
                                            camera_point: Vec3::ZERO,
                                            target_error: 2.0,
                                            cam: Camera::new(1.0),
                                        },
                                        "Point distance",
                                    );
                                    ui.selectable_value(
                                        &mut mesh.error_calc,
                                        ErrorMode::ExactLayer { layer: 0 },
                                        "Exact Layer",
                                    );
                                });

                            let freeze = mesh.freeze;

                            match &mut mesh.error_calc {
                                ErrorMode::PointDistance {
                                    camera_point,
                                    target_error,
                                    cam,
                                } => {
                                    if !freeze {
                                        *camera_point = (*transform.get_pos()).into();
                                        *cam = camera.clone();
                                    }
                                    // ui.add(
                                    //     egui::widgets::Slider::new(&mut camera_point.x, -2.5..=2.5)
                                    //         .prefix("X: "),
                                    // );
                                    // ui.add(
                                    //     egui::widgets::Slider::new(&mut camera_point.y, -2.5..=2.5)
                                    //         .prefix("Y: "),
                                    // );
                                    // ui.add(
                                    //     egui::widgets::Slider::new(&mut camera_point.z, -2.5..=2.5)
                                    //         .prefix("Z: "),
                                    // );
                                    ui.add_space(10.0);
                                    ui.add(
                                        egui::widgets::Slider::new(target_error, 0.1..=10.0)
                                            .prefix("Target Error: "),
                                    );
                                }
                                ErrorMode::MaxError { error } => {
                                    ui.add(
                                        egui::widgets::Slider::new(error, 0.0..=10.0)
                                            .prefix("MAX: "),
                                    );
                                }
                                ErrorMode::ExactLayer { layer } => {
                                    ui.add(
                                        egui::widgets::Slider::new(layer, 0..=10).prefix("Layer: "),
                                    );
                                }
                            }

                            if ui.button("Snapshot Error Graph").clicked() {
                                let g = mesh.submesh_error_graph(submeshes);
                                let e_calc = mesh.error_calc.clone();

                                thread::spawn(move || {
                                    petgraph_to_svg(
                                        &g,
                                        "svg/error.svg",
                                        &|_, (_, &e)| match e_calc {
                                            ErrorMode::PointDistance { target_error, .. } => {
                                                if e < target_error {
                                                    "color=green".to_owned()
                                                } else {
                                                    "color=red".to_owned()
                                                }
                                            }
                                            _ => String::new(),
                                        },
                                        common::graph::GraphSVGRender::Directed {
                                            node_label: common::graph::Label::Weight,
                                        },
                                    )
                                    .unwrap();
                                });
                            }
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

    pub fn init(
        renderer: &Renderer,
        mut mesh: QueryState<(&MultiResMeshComponent)>,
        mut submeshes: QueryState<(Entity, &SubMeshComponent)>,
        world: &World,
    ) -> Self {
        let renderer =
            egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1);

        let mesh = mesh.get_single(world).unwrap();

        Self {
            last_frame: Instant::now(),
            renderer,
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
