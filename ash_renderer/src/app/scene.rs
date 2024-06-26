use std::sync::Arc;

use ash::vk;
use bevy_ecs::prelude::*;
use bytemuck::Zeroable;
use common_renderer::components::{camera::Camera, transform::Transform};

use crate::{spiral::Spiral, utility::buffer::TBuffer};

use super::{
    app::AssetLib,
    mesh_data::MeshData,
    renderer::{MeshDrawingPipelineType, Renderer},
};
#[derive(Debug, Clone, Copy, Event)]
pub enum SceneEvent {
    AddInstances(usize),
    ResetScene,
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
    pub view_proj: glam::Mat4,
    // Freezable view-proj matrix, for culling calculations
    pub culling_view_proj: glam::Mat4,
    pub cam_pos: glam::Vec3,
    pub target_error: f32,
    pub dist_pow: f32,
    pub _0: u32,
    pub _1: u32,
    pub _2: u32,
}

#[derive(Resource)]
pub struct Scene {
    //texture_image: Image,
    pub uniform_transform_buffer: Arc<TBuffer<ModelUniformBufferObject>>,
    pub uniform_transforms: Vec<ModelUniformBufferObject>,
    pub uniform_camera: CameraUniformBufferObject,
    pub target_error: f32,
    pub dist_pow: f32,
    pub uniform_camera_buffers: Vec<Arc<TBuffer<CameraUniformBufferObject>>>,
    pub freeze_error: bool,
    pub freeze_culling: bool,
    pub instances: usize,
}
#[derive(Component)]
pub struct Mesh {
    pub id: usize,
}

impl CameraUniformBufferObject {
    pub fn new(view_proj: glam::Mat4, cam_pos: glam::Vec3, target_error: f32) -> Self {
        Self {
            view_proj,
            culling_view_proj: view_proj,
            cam_pos,
            target_error,
            dist_pow: 1.0,
            _0: 0,
            _1: 0,
            _2: 0,
        }
    }
    pub fn update_view_proj(&mut self, view_proj: glam::Mat4, frozen: bool) {
        self.view_proj = view_proj;
        if !frozen {
            self.culling_view_proj = view_proj
        }
    }
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
            self.freeze_culling,
        );

        if !self.freeze_error {
            self.uniform_camera.cam_pos = (*camera_transform.get_pos()).into();
        }

        self.uniform_camera.target_error = self.target_error;
        self.uniform_camera.dist_pow = self.dist_pow;

        self.uniform_camera_buffers[current_image].update_uniform_buffer(self.uniform_camera);
    }
}

pub fn process_scene_events(
    mut scene: ResMut<Scene>,
    renderer: Res<Renderer>,
    mesh_data: Res<AssetLib<MeshData>>,
    mut commands: Commands,
    mut event_read: EventReader<SceneEvent>,
    mut draw_write: EventWriter<MeshDrawingPipelineType>,
    transforms: Query<(Entity, &Transform, &Mesh)>,
) {
    for e in event_read.read() {
        match e {
            SceneEvent::AddInstances(count) => {
                for (i, j) in Spiral::default().skip(scene.instances).take(*count) {
                    let p = glam::Vec3A::X * i as f32 * 20.0 + glam::Vec3A::Y * j as f32 * 20.0;

                    let mut transform = Transform::new_pos(p);

                    let scale = 30.0 / mesh_data.get(&renderer.mesh).size;

                    println!("{scale}");

                    *transform.scale_mut() = glam::Vec3A::ONE * scale;

                    commands.spawn((
                        transform,
                        Mesh {
                            id: scene.instances,
                        },
                    ));
                    scene.instances += 1;
                }
                // Update buffers after commands have been completed next frame
                // We are not allowed to read and write events in the same system
                commands.add(|w: &mut World| {
                    w.send_event(SceneEvent::UpdateInstanceBuffers);
                });
            }
            // Reset to 50 instances
            SceneEvent::ResetScene => {
                for (e, _t, _m) in transforms.iter() {
                    commands.entity(e).despawn()
                }
                scene.instances = 0;
            }
            SceneEvent::UpdateInstanceBuffers => {
                let mut uniform_transforms = vec![Zeroable::zeroed(); scene.instances];

                for (_, transform, mesh) in transforms.iter() {
                    uniform_transforms[mesh.id] = ModelUniformBufferObject {
                        model: transform.get_local_to_world(),
                        inv_model: transform.get_local_to_world().inverse(),
                    };
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
                scene.uniform_transforms = uniform_transforms;
                // Refresh pipelines
                // TODO: Some way to just replace the descriptor groups instead of the entire pipeline

                draw_write.send(renderer.current_pipeline);
            }
        }
    }
}
