use std::{collections::HashMap, time::Instant};

use super::partition_graph::{
    AllMyNodeTemplates, MyDataType, MyEditorState, MyGraphState, MyNodeTemplate, MyResponse,
    MyValueType,
};
use crate::{
    components::mesh::{Mesh, SubMeshComponent},
    core::Renderer,
};
use bevy_ecs::{
    entity::Entity,
    query::QueryState,
    system::{Commands, Query, Resource},
    world::World,
};
use common_renderer::components::camera::Camera;
use egui::{Pos2, Vec2};
use egui_node_graph::{InputParamKind, NodeResponse, NodeTemplateTrait};
use egui_wgpu::renderer::ScreenDescriptor;
use glam::Vec3;
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
        submeshes: &Query<&SubMeshComponent>,
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
                            egui::ComboBox::from_label("Mode ")
                                .selected_text(format!("{:?}", &mut mesh.error_calc))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut mesh.error_calc,
                                        crate::components::mesh::ErrorMode::MaxError { error: 0.1 },
                                        "Max Error",
                                    );
                                    ui.selectable_value(
                                        &mut mesh.error_calc,
                                        crate::components::mesh::ErrorMode::PointDistance {
                                            camera_point: Vec3::ZERO,
                                            error_falloff: 2.0,
                                        },
                                        "Point distnace",
                                    );
                                });

                            match &mut mesh.error_calc {
                                crate::components::mesh::ErrorMode::PointDistance {
                                    camera_point,
                                    error_falloff,
                                } => {
                                    ui.add(
                                        egui::widgets::Slider::new(&mut camera_point.x, -2.5..=2.5)
                                            .prefix("X: "),
                                    );
                                    ui.add(
                                        egui::widgets::Slider::new(&mut camera_point.y, -2.5..=2.5)
                                            .prefix("Y: "),
                                    );
                                    ui.add(
                                        egui::widgets::Slider::new(&mut camera_point.z, -2.5..=2.5)
                                            .prefix("Z: "),
                                    );
                                    ui.add_space(10.0);
                                    ui.add(
                                        egui::widgets::Slider::new(error_falloff, 0.1..=10.0)
                                            .prefix("falloff: "),
                                    );
                                }
                                crate::components::mesh::ErrorMode::MaxError { error } => {
                                    ui.add(
                                        egui::widgets::Slider::new(error, 0.0..=1.0).prefix("X: "),
                                    );
                                }
                            }

                            //    println!("{:?}", self.state.node_positions);

                            // let graph_response = egui::ScrollArea::both()
                            //     .show(ui, |ui| {
                            //         ui.allocate_space(Vec2 {
                            //             x: 300.0,
                            //             y: 15000.0,
                            //         });
                            //         // …
                            //         self.state.draw_graph_editor(
                            //             ui,
                            //             AllMyNodeTemplates,
                            //             &mut self.user_state,
                            //             Vec::default(),
                            //         )
                            //     })
                            //     .inner;

                            // for node_response in graph_response.node_responses {
                            //     // Here, we ignore all other graph events. But you may find
                            //     // some use for them. For example, by playing a sound when a new
                            //     // connection is created
                            //     if let NodeResponse::User(user_event) = node_response {
                            //         match user_event {
                            //             MyResponse::SetActiveNode(node) => {
                            //                 camera.part_highlight =
                            //                     self.state.graph.nodes[node].user_data.part;
                            //                 //mesh.remesh =
                            //                 //    self.state.graph.nodes[node].user_data.layer;

                            //                 self.user_state.active_nodes.insert(node);
                            //                 mesh.add_submesh(
                            //                     self.state.graph[node].user_data.entity,
                            //                     submeshes,
                            //                 );
                            //             }
                            //         }
                            //     }
                            // }
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
        mut mesh: QueryState<(&Mesh)>,
        mut submeshes: QueryState<(Entity, &SubMeshComponent)>,
        world: &World,
    ) -> Self {
        let focused_layer = 1;
        let focused_part = 1;
        let (f_e, f_s) = {
            let mut focused = Entity::PLACEHOLDER;
            let mut fs = None;
            for (e, s) in submeshes.iter(world) {
                if s.part == focused_part && s.layer == focused_layer {
                    focused = e;
                    fs = Some(s);
                    break;
                }
            }
            (focused, fs.unwrap())
        };

        let renderer =
            egui_wgpu::Renderer::new(renderer.device(), renderer.surface_format(), None, 1);

        let mut state = MyEditorState::default();
        let mut user_state = MyGraphState::default();

        let mesh = mesh.get_single(world).unwrap();

        //let mut outputs = HashMap::new();

        // for (e, s) in submeshes.iter(world) {
        //     if e == f_e || s.dependants.contains(&f_e) || f_s.dependants.contains(&e) {
        //         let mut data = MyNodeTemplate::Partition.user_data(&mut user_state);
        //         data.part = s.part;
        //         data.layer = s.layer;
        //         data.entity = e;

        //         let id = state
        //             .graph
        //             .add_node("".to_string(), data, |graph, node_id| {
        //                 let output =
        //                     graph.add_output_param(node_id, "".to_string(), MyDataType::Hierarchy);
        //                 outputs.insert(e, (node_id, output));
        //             });
        //         state.node_positions.insert(
        //             id,
        //             Pos2 {
        //                 x: (s.layer as f32) * 230.0,
        //                 y: ((s.part % 10) as f32) * 130.0,
        //             },
        //         );
        //         state.node_order.push(id);

        //         if mesh.submeshes.contains(&e) {
        //             user_state.active_nodes.insert(id);
        //         }
        //     }
        // }

        // Do Connections

        //for (e, s) in submeshes.iter(world) {
        //    for dependent in &s.dependences {
        //        let Some((_, output)) = outputs.get(&dependent) else {
        //            continue;
        //        };
        //        let Some((node_id, _)) = outputs.get(&e) else {
        //            continue;
        //        };
        //
        //        // For partition that we depend on, add an input param
        //        let input = state.graph.add_input_param(
        //            *node_id,
        //            "".to_string(),
        //            MyDataType::Hierarchy,
        //            MyValueType::Hierarchy,
        //            InputParamKind::ConnectionOnly,
        //            true,
        //        );
        //        //y += *parent_part as f32;
        //        state.graph.add_connection(*output, input)
        //    }
        //}

        Self {
            last_frame: Instant::now(),
            renderer,
            state,
            user_state,
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
