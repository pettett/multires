use std::sync::Arc;

use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::components::{camera::Camera, transform::Transform};

use crate::{spiral::Spiral, utility::buffer::TBuffer};

use super::renderer::{MeshDrawingPipelineType, Renderer};
#[derive(Debug, Clone, Copy, Event)]
pub enum SceneEvent {
    AddInstances(usize),
    UpdateInstanceBuffers,
}

#[repr(C)]
#[derive(Clone, Debug, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniformBufferObject {
    pub model: glam::Mat4,
    pub inv_model: glam::Mat4,
}

#[repr(C)]
#[derive(Clone, Debug, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniformBufferObject {
    view_proj: glam::Mat4,
    // Freezable view-proj matrix, for culling calculations
    culling_view_proj: glam::Mat4,
    pub cam_pos: glam::Vec3,
    pub target_error: f32,
}

impl CameraUniformBufferObject {
    pub fn new(view_proj: glam::Mat4, cam_pos: glam::Vec3, target_error: f32) -> Self {
        Self {
            view_proj,
            culling_view_proj: view_proj,
            cam_pos,
            target_error,
        }
    }
    pub fn update_view_proj(&mut self, view_proj: glam::Mat4, frozen: bool) {
        self.view_proj = view_proj;
        if !frozen {
            self.culling_view_proj = view_proj
        }
    }
}

#[derive(Resource)]
pub struct Scene {
    //texture_image: Image,
    pub uniform_transform_buffer: Arc<TBuffer<ModelUniformBufferObject>>,
    pub uniform_camera: CameraUniformBufferObject,
    pub target_error: f32,
    pub uniform_camera_buffers: Vec<Arc<TBuffer<CameraUniformBufferObject>>>,
    pub freeze_pos: bool,
    pub instances: usize,
}

impl Scene {
    pub fn update_camera_uniform_buffer(
        &mut self,
        camera: &Camera,
        camera_transform: &Transform,
        current_image: usize,
    ) {
        self.uniform_camera.update_view_proj(
            camera.build_view_projection_matrix(camera_transform),
            self.freeze_pos,
        );

        if !self.freeze_pos {
            self.uniform_camera.cam_pos = (*camera_transform.get_pos()).into();
        }

        self.uniform_camera.target_error = self.target_error;

        self.uniform_camera_buffers[current_image].update_uniform_buffer(self.uniform_camera);
    }
}

pub fn process_scene_events(
    mut scene: ResMut<Scene>,
    renderer: NonSend<Renderer>,
    mut commands: Commands,
    mut event_read: EventReader<SceneEvent>,
    mut draw_write: EventWriter<MeshDrawingPipelineType>,
    transforms: Query<&Transform>,
) {
    for e in event_read.read() {
        match e {
            SceneEvent::AddInstances(count) => {
                for (i, j) in Spiral::default().skip(scene.instances).take(*count) {
                    let p = glam::Vec3A::X * i as f32 * 20.0 + glam::Vec3A::Z * j as f32 * 40.0;

                    let transform = Transform::new_pos(p);

                    println!("{i} {j}");

                    commands.spawn(transform);
                }
                scene.instances += count;
                // Update buffers after commands have been completed next frame
                // We are not allowed to read and write events in the same system
                commands.add(|w: &mut World| {
                    w.send_event(SceneEvent::UpdateInstanceBuffers);
                });
            }
            SceneEvent::UpdateInstanceBuffers => {
                let mut uniform_transforms = Vec::with_capacity(scene.instances);

                for transform in transforms.iter() {
                    uniform_transforms.push(ModelUniformBufferObject {
                        model: transform.get_local_to_world(),
                        inv_model: transform.get_local_to_world().inverse(),
                    });
                }
                println!("{}", uniform_transforms.len());
                scene.uniform_transform_buffer = TBuffer::new_filled(
                    &renderer.core,
                    renderer.allocator.clone(),
                    renderer.graphics_queue,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    &uniform_transforms,
                    "Transform Buffer",
                );

                // Refresh pipelines
                // TODO: Some way to just replace the descriptor groups instead of the entire pipeline

                draw_write.send(renderer.current_pipeline);
            }
        }
    }
}
