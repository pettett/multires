use std::{thread, time::Instant};

use crate::{
    components::gpu_multi_res_mesh::{
        ClusterComponent, ErrorMode, MultiResMeshComponent, MultiResMeshRenderer,
    },
    core::Renderer,
};
use bevy_ecs::{
    entity::Entity,
    query::QueryState,
    system::{Commands, Query, ResMut, Resource},
    world::World,
};
use common::graph::petgraph_to_svg;
use common_renderer::components::{
    camera::Camera, camera_controller::CameraController, transform::Transform,
};

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
        meshes: &mut Query<(&mut MultiResMeshComponent, &Transform)>,
        mut mesh_renderer: ResMut<MultiResMeshRenderer>,
        submeshes: &Query<(Entity, &ClusterComponent)>,
        mut camera: Query<(&mut Camera, &mut CameraController, &Transform)>,
        _commands: &mut Commands,
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

            for (mut camera, mut camera_controller, transform) in camera.iter_mut() {
                egui::Window::new("Camera Settings")
                    .min_height(100.0)
                    .min_width(100.0)
                    .show(&ctx, |ui| {
                        ui.label(format!("FPS: {}", 1.0 / _delta_s.as_secs_f32()));

                        ui.add(
                            egui::widgets::DragValue::new(&mut camera.part_highlight)
                                .prefix("highlight partition: "),
                        );

                        ui.add(
                            egui::widgets::Slider::new(camera_controller.speed_mut(), 0.03..=3.0)
                                .logarithmic(true)
                                .prefix("Camera Speed: "),
                        );
                    });

                //FIXME: This is really nothing to do with gui.
                let freeze = mesh_renderer.freeze;

                match &mut mesh_renderer.error_calc {
                    ErrorMode::PointDistance { camera_point, cam } => {
                        if !freeze {
                            *camera_point = (*transform.get_pos()).into();
                            *cam = camera.clone();
                        }
                    }
                    _ => (),
                }
            }

            egui::Window::new("Mesh Renderer")
                .min_height(100.0)
                .min_width(100.0)
                .show(&ctx, |ui| {
                    ui.add(
                        egui::widgets::DragValue::new(&mut mesh_renderer.focus_part)
                            .prefix("focus partition: "),
                    );

                    ui.add(egui::widgets::Checkbox::new(
                        &mut mesh_renderer.freeze,
                        "Freeze",
                    ));
                    ui.add(egui::widgets::Checkbox::new(
                        &mut mesh_renderer.show_solid,
                        "Show Solid",
                    ));
                    ui.add(egui::widgets::Checkbox::new(
                        &mut mesh_renderer.show_wire,
                        "Show Wire",
                    ));
                    ui.add(egui::widgets::Checkbox::new(
                        &mut mesh_renderer.show_bounds,
                        "Show Bounds",
                    ));

                    egui::ComboBox::from_label("Mode ")
                        .selected_text(format!("{:?}", &mut mesh_renderer.error_calc))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut mesh_renderer.error_calc,
                                ErrorMode::MaxError,
                                "Max Error",
                            );
                            ui.selectable_value(
                                &mut mesh_renderer.error_calc,
                                ErrorMode::PointDistance {
                                    camera_point: Vec3::ZERO,
                                    cam: Camera::new(1.0),
                                },
                                "Point distance",
                            );
                            ui.selectable_value(
                                &mut mesh_renderer.error_calc,
                                ErrorMode::ExactLayer,
                                "Exact Layer",
                            );
                        });

                    ui.add(
                        egui::widgets::Slider::new(&mut mesh_renderer.error_target, 0.1..=10.0)
                            .prefix("Target Error: "),
                    );
                    for (mesh, _trans) in meshes {
                        ui.label(mesh.name());

                        if ui.button("Snapshot Error Graph").clicked() {
                            let g = mesh.submesh_error_graph(submeshes, &mesh_renderer);
                            let error_target = mesh_renderer.error_target;

                            thread::spawn(move || {
                                petgraph_to_svg(
                                    &g,
                                    "svg/error.svg",
                                    &|_, (_, &e)| {
                                        if e < error_target {
                                            "color=green".to_owned()
                                        } else {
                                            "color=red".to_owned()
                                        }
                                    },
                                    common::graph::GraphSVGRender::Directed {
                                        node_label: common::graph::Label::Weight,
                                    },
                                )
                                .unwrap();
                            });
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
        _mesh: QueryState<&MultiResMeshComponent>,
        _submeshes: QueryState<(Entity, &ClusterComponent)>,
        _world: &World,
    ) -> Self {
        let renderer =
            egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1);

        Self {
            last_frame: Instant::now(),
            renderer,
        }
    }
    pub fn handle_event<T>(&mut self, _window: &Window, _event: &Event<T>) -> bool {
        //self.platform
        //    .handle_event(self.imgui.io_mut(), window, event);
        //
        //self.imgui.io().want_capture_mouse

        false
    }
}
