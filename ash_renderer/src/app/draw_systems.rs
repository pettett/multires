use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::{
    components::{camera::Camera, transform::Transform},
    resources::time::Time,
};

use crate::{
    app::mesh_data::MeshDataBuffers,
    draw_pipelines::{
        compute_culled_indices::ComputeCulledIndices, compute_culled_mesh::ComputeCulledMesh,
        draw_indirect::DrawIndirect, expanding_compute_culled_mesh::ExpandingComputeCulledMesh,
        indirect_tasks::IndirectTasks, stub::Stub, DrawPipeline,
    },
    utility::constants::MAX_FRAMES_IN_FLIGHT,
};

use super::{
    fps_limiter::FPSMeasure,
    renderer::{MeshDrawingPipelineType, Renderer},
    scene::{Scene, SceneEvent},
};

pub fn update_pipeline(
    mut renderer: NonSendMut<Renderer>,
    scene: Res<Scene>,
    mut events: EventReader<MeshDrawingPipelineType>,
    mesh_data: Res<MeshDataBuffers>,
    transforms: Query<&Transform>,
    mut commands: Commands,
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

    let mut draw_pipeline: Box<dyn DrawPipeline> = match s {
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

pub fn draw_frame(
    mut renderer: NonSendMut<Renderer>,
    mut scene: ResMut<Scene>,
    mut camera: Query<(&mut Camera, &Transform)>,
    scene_events: EventWriter<SceneEvent>,
    draw_events: EventWriter<MeshDrawingPipelineType>,
    fps: Res<FPSMeasure>,
    mesh_data: Res<MeshDataBuffers>,
    mut commands: Commands,
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
                    renderer.recreate_swapchain(&scene, &mesh_data, &mut cam);
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

    let ui_cmd = renderer.draw_gui(
        &mut scene,
        image_index as usize,
        scene_events,
        draw_events,
        &fps,
        commands,
    );

    let submit_infos = [vk::SubmitInfo::builder()
        .command_buffers(&[renderer.draw_pipeline.draw(image_index as usize), ui_cmd])
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
        renderer.recreate_swapchain(&scene, &mesh_data, &mut cam);
    }

    renderer.current_frame = (renderer.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}
