use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::{
    components::{camera::Camera, transform::Transform},
    resources::time::Time,
};

use crate::{
    app::mesh_data::MeshData,
    draw_pipelines::{
        compute_culled_indices::ComputeCulledIndices,
        compute_culled_mesh::ComputeCulledMesh,
        draw_indirect::DrawIndirect,
        expanding_compute_culled_mesh::ExpandingComputeCulledMesh,
        indirect_tasks::{IndirectTasks, MeshShaderMode},
        stub::Stub,
        DrawPipeline,
    },
    gui::gui::Gui,
    utility::constants::MAX_FRAMES_IN_FLIGHT,
};

use super::{
    benchmarker::Benchmarker,
    fps_limiter::FPSMeasure,
    renderer::{Fragment, MeshDrawingPipelineType, Renderer},
    scene::{Scene, SceneEvent},
};

pub fn update_pipeline(
    mut renderer: ResMut<Renderer>,
    scene: Res<Scene>,
    mut events: EventReader<MeshDrawingPipelineType>,
    mesh_data: Res<MeshData>,
    transforms: Query<&Transform>,
) {
    let s = events.read().next();

    let s = match s {
        Some(MeshDrawingPipelineType::None) => return,
        Some(s) => {
            renderer.core.device.wait_device_idle();
            s
        }
        _ => return,
    };

    renderer.draw_pipeline = Box::new(Stub);

    let mut draw_pipeline: Box<dyn DrawPipeline + Send + Sync> = match s {
        MeshDrawingPipelineType::IndirectTasks => Box::new(IndirectTasks::new(
            renderer.core.clone(),
            &renderer,
            &renderer.screen,
            transforms,
            &mesh_data,
            scene.uniform_transform_buffer.clone(),
            &scene.uniform_camera_buffers,
        )),
        MeshDrawingPipelineType::ComputeCulledMesh => Box::new(ComputeCulledMesh::new(
            renderer.core.clone(),
            &renderer,
            &renderer.screen,
            &mesh_data,
            renderer.allocator.clone(),
            &renderer.render_pass,
            renderer.graphics_queue,
            renderer.descriptor_pool.clone(),
            scene.uniform_transform_buffer.clone(),
            &scene.uniform_camera_buffers,
            mesh_data.cluster_count,
        )),
        MeshDrawingPipelineType::ExpandingComputeCulledMesh => {
            Box::new(ExpandingComputeCulledMesh::new(
                renderer.core.clone(),
                &renderer,
                &renderer.screen,
                &scene,
                &mesh_data,
                renderer.allocator.clone(),
                &renderer.render_pass,
                renderer.graphics_queue,
                renderer.descriptor_pool.clone(),
            ))
        }
        MeshDrawingPipelineType::ComputeCulledIndices => Box::new(ComputeCulledIndices::new(
            renderer.core.clone(),
            &renderer,
            &renderer.screen,
            &mesh_data,
            renderer.allocator.clone(),
            &renderer.render_pass,
            renderer.graphics_queue,
            renderer.descriptor_pool.clone(),
            scene.uniform_transform_buffer.clone(),
            &scene.uniform_camera_buffers,
            mesh_data.cluster_count,
        )),
        MeshDrawingPipelineType::DrawIndirect => {
            Box::new(DrawIndirect::new(&renderer, &mesh_data, &scene))
        }
        MeshDrawingPipelineType::None => unreachable!(),
    };

    draw_pipeline.init_swapchain(&renderer.core, &renderer.screen, &renderer.render_pass);

    renderer.draw_pipeline = draw_pipeline;
    renderer.current_pipeline = *s;
}

pub fn tick_clocks(mut time: ResMut<Time>, mut fps: ResMut<FPSMeasure>) {
    time.tick(fps.delta_time());

    fps.tick_frame();
}

pub fn start_gui(mut gui: NonSendMut<Gui>, renderer: Res<Renderer>) {
    if renderer.render_gui {
        gui.start_draw();
    }
}

pub fn draw_gui(
    gui: NonSendMut<Gui>,
    mut renderer: ResMut<Renderer>,
    mut scene: ResMut<Scene>,
    mut camera: Query<(&Camera, &mut Transform)>,
    mut scene_events: EventWriter<SceneEvent>,
    mut draw_events: EventWriter<MeshDrawingPipelineType>,
    fps: Res<FPSMeasure>,
    mut commands: Commands,
) {
    if !renderer.render_gui {
        return;
    }

    let mut app_info_open = renderer.app_info_open;

    let ctx = &gui.draw();

    for w in &mut renderer.windows {
        w.draw(ctx);
    }

    let bench_end_pos = glam::Vec3A::Z * 500.0;

    egui::Window::new("App Config")
        .open(&mut app_info_open)
        .show(ctx, |ui| {
            {
                ui.checkbox(&mut scene.freeze_pos, "Freeze");
                ui.add(egui::Slider::new(&mut scene.target_error, 0.0..=10.0).text("Target Error"));
                ui.add(egui::Slider::new(&mut scene.dist_pow, 0.001..=3.0).text("Distance Power"));
            }

            ui.add(fps.as_ref());

            ui.label(format!("Current Pipeline: {:?}", renderer.current_pipeline));

            ui.add_enabled_ui(renderer.core.device.features.mesh_shader, |ui| {
                if ui.button("Indirect Tasks").clicked() {
                    draw_events.send(MeshDrawingPipelineType::IndirectTasks)
                }
            });

            // if ui.button("Compute Culled Mesh").clicked() {
            //     draw_events.send(MeshDrawingPipelineType::ComputeCulledMesh)
            // }

            if ui.button("Expanding Compute Culled Mesh").clicked() {
                draw_events.send(MeshDrawingPipelineType::ExpandingComputeCulledMesh)
            }

            if ui.button("Compute Culled Indices").clicked() {
                draw_events.send(MeshDrawingPipelineType::ComputeCulledIndices)
            }

            if ui.button("Draw Full Res").clicked() {
                draw_events.send(MeshDrawingPipelineType::DrawIndirect)
            }
            if ui.button("Add 1 More Instance").clicked() {
                scene_events.send(SceneEvent::AddInstances(1));
            }
            if ui.button("Add 50 More Instances").clicked() {
                scene_events.send(SceneEvent::AddInstances(50));
            }
            if ui.button("Add 200 More Instances").clicked() {
                scene_events.send(SceneEvent::AddInstances(200));
            }
            if ui.button("Reset Scene").clicked() {
                scene_events.send(SceneEvent::ResetScene);
            }
            if ui.button("Camera to benchmark end").clicked() {
                let (cam, mut trans) = camera.single_mut();

                trans.set_pos(bench_end_pos);
                trans.look_at(glam::Vec3A::ZERO);
            }

            egui::ComboBox::from_label("Cluster Tri Encoding")
                .selected_text(format!("{:?}", renderer.mesh_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut renderer.mesh_mode,
                        MeshShaderMode::TriangleList,
                        "Triangle List",
                    );
                    ui.selectable_value(
                        &mut renderer.mesh_mode,
                        MeshShaderMode::TriangleStrip,
                        "Triangle Strip",
                    );
                });

            egui::ComboBox::from_label("Fragment")
                .selected_text(format!("{:?}", renderer.fragment))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut renderer.fragment, Fragment::Lit, "Lit");
                    ui.selectable_value(
                        &mut renderer.fragment,
                        Fragment::VertexColour,
                        "Vertex Colour",
                    );
                });

            if ui.checkbox(&mut renderer.query, "Enable Queries").clicked() {
                draw_events.send(renderer.current_pipeline);
            }

            if ui.button("Refresh").clicked() {
                draw_events.send(renderer.current_pipeline);
            }

            if ui.button("Begin Benchmarking").clicked() {
                // Refresh render pipeline
                commands.insert_resource(Benchmarker::new(
                    glam::Vec3A::Z * 10.0,
                    bench_end_pos,
                    5.0,
                ))
            }

            if renderer.query {
                renderer.draw_pipeline.stats_gui(ui, 0);
            }
        });

    egui::TopBottomPanel::top("test").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            for w in &mut renderer.windows {
                let (open, name) = w.state();
                ui.checkbox(open, name);
            }
            ui.checkbox(&mut app_info_open, "App Info");
        });
    });

    renderer.app_info_open = app_info_open;
}

pub fn draw_frame(
    mut renderer: ResMut<Renderer>,
    mut scene: ResMut<Scene>,
    mut camera: Query<(&mut Camera, &Transform)>,
    mut gui: NonSendMut<Gui>,
    mesh_data: Res<MeshData>,
) {
    let wait_fences = [renderer.sync_objects.in_flight_fences[renderer.current_frame]];

    let (mut cam, cam_trans) = camera.single_mut();

    unsafe {
        renderer
            .core
            .device
            .wait_for_fences(&wait_fences, true, std::u64::MAX)
            .expect("Failed to wait for Fence!");
    }

    let (image_index, is_sub_optimal) = unsafe {
        let result = renderer.core.device.fn_swapchain.acquire_next_image(
            renderer.screen.swapchain().handle,
            std::u64::MAX,
            renderer.sync_objects.image_available_semaphores[renderer.current_frame],
            vk::Fence::null(),
        );

        match result {
            Ok(image_index) => image_index,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR => {
                    renderer.recreate_swapchain(&scene, &mesh_data, &mut gui, &mut cam);
                    return;
                }
                _ => panic!("Failed to acquire Swap Chain Image!"),
            },
        }
    };

    // if self.transform_dirty[image_index as usize] {
    //     self.update_model_uniform_buffer(image_index as usize, delta_time);
    //     self.transform_dirty[image_index as usize] = false;
    // }

    scene.update_camera_uniform_buffer(&cam, cam_trans, image_index as usize);

    let wait_semaphores =
        [renderer.sync_objects.image_available_semaphores[renderer.current_frame]];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let signal_semaphores =
        [renderer.sync_objects.render_finished_semaphores[renderer.current_frame]];

    let draw_cmd = renderer.draw_pipeline.draw(image_index as usize);

    let temp1;
    let temp2;

    let command_buffers = if renderer.render_gui {
        let ui_cmd = gui.finish_draw(image_index as usize);
        temp1 = [draw_cmd, ui_cmd];
        &temp1[..]
    } else {
        temp2 = [draw_cmd];
        &temp2
    };

    let submit_infos = [vk::SubmitInfo::builder()
        .command_buffers(command_buffers)
        .wait_dst_stage_mask(&wait_stages)
        .wait_semaphores(&wait_semaphores)
        .signal_semaphores(&signal_semaphores)
        .build()];

    unsafe {
        renderer
            .core
            .device
            .reset_fences(&wait_fences)
            .expect("Failed to reset Fence!");

        renderer
            .core
            .device
            .queue_submit(
                renderer.graphics_queue,
                &submit_infos,
                renderer.sync_objects.in_flight_fences[renderer.current_frame],
            )
            .expect("Failed to execute queue submit.");
    }

    let swapchains = [renderer.screen.swapchain().handle];
    let image_indices = [image_index];

    let present_info = vk::PresentInfoKHR::builder()
        .swapchains(&swapchains)
        .wait_semaphores(&signal_semaphores)
        .image_indices(&image_indices);

    let result = unsafe {
        renderer
            .core
            .device
            .fn_swapchain
            .queue_present(renderer.present_queue, &present_info)
    };

    let is_resized = match result {
        Ok(_) => renderer.is_framebuffer_resized,
        Err(vk_result) => match vk_result {
            vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
            _ => panic!("Failed to execute queue present."),
        },
    };
    if is_resized || is_sub_optimal {
        renderer.is_framebuffer_resized = false;
        renderer.recreate_swapchain(&scene, &mesh_data, &mut gui, &mut cam);
    }

    renderer.current_frame = (renderer.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}
