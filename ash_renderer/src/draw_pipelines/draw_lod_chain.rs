use std::sync::Arc;

use ash::vk::{self};

use bevy_ecs::prelude::*;
use common::MeshVert;
use common_renderer::components::{camera::Camera, transform::Transform};
use glam::{Mat4, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    app::{
        mesh_data::MeshData,
        renderer::{MeshDrawingPipelineType, Renderer},
        scene::{Mesh, Scene},
    },
    core::Core,
    screen::Screen,
    utility::{
        buffer::TBuffer,
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::{CommandBuffer, CommandPool},
            descriptor_pool::{DescriptorPool, DescriptorSet},
        },
        render_pass::RenderPass,
        GraphicsPipeline,
    },
    VkHandle, CLEAR_VALUES,
};

use super::{
    render_multires_indices::{
        create_traditional_graphics_descriptor_set_layout,
        create_traditional_graphics_descriptor_sets, create_traditional_graphics_pipeline,
    },
    DrawPipeline,
};

#[derive(Resource)]
pub struct DrawLODChainData {
    graphics_pipeline: GraphicsPipeline,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_pool: Arc<DescriptorPool>,
    core: Arc<Core>,
    vertex_buffer: Arc<TBuffer<MeshVert>>,
    query: bool,
    command_buffers: Vec<CommandBuffer>,
}

// Calculates the frustum and planes from a projection matrix.
fn planes_from_mat(mat: Mat4) -> [Vec4; 6] {
    let mut planes = [Vec4::ZERO; 6];

    for i in 0..4 {
        planes[0][i] = mat.col(i)[3] + mat.col(i)[0];
    }
    for i in 0..4 {
        planes[1][i] = mat.col(i)[3] - mat.col(i)[0];
    }

    for i in 0..4 {
        planes[2][i] = mat.col(i)[3] + mat.col(i)[1];
    }
    for i in 0..4 {
        planes[3][i] = mat.col(i)[3] - mat.col(i)[1];
    }

    // Vulkan places it's near plane at w = 0, same as DX11, so use that part of the paper
    for i in 0..4 {
        planes[4][i] = mat.col(i)[3];
    }
    for i in 0..4 {
        planes[5][i] = mat.col(i)[3] - mat.col(i)[2];
    }

    // Normalise planes
    for i in 0..6 {
        planes[i] *= (planes[i].xyz()).length_recip();
    }

    return planes;
}
// Calculate the distance from a plane to a point
fn dist_to_plane(plane: Vec4, point: glam::Vec3) -> f32 {
    return plane.xyz().dot(point) + plane.w;
}
// Is a sphere within a set of six planes Based on the distance to each plane.
fn sphere_inside_planes(planes: &[Vec4; 6], sphere: Vec4) -> bool {
    for i in 0..6 {
        if -sphere.w > dist_to_plane(planes[i], sphere.xyz()) {
            return false;
        }
    }
    return true;
}

pub fn create_lod_command_buffer(
    mut renderer: ResMut<Renderer>,
    scene: Res<Scene>,
    camera: Query<(&Camera, &Transform)>,
    meshes: Query<(&Transform, &Mesh)>,
    draw: Option<ResMut<DrawLODChainData>>,
    mesh_data: Res<MeshData>,
) {
    if let Some(mut draw) = draw {
        let device = draw.core.device.clone();

        while renderer.image_index >= draw.command_buffers.len() {
            let cmd = draw.core.command_pool.begin_one_shot_command();
            draw.command_buffers.push(cmd);
        }

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(renderer.render_pass.handle())
            .framebuffer(renderer.screen.swapchain_framebuffers[renderer.image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: renderer.screen.swapchain().extent,
            })
            .clear_values(&CLEAR_VALUES);

        {
            let mut command_buffer_writer =
                draw.command_buffers[renderer.image_index].reset_and_write();

            unsafe {
                device.cmd_begin_render_pass(
                    *command_buffer_writer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                command_buffer_writer.set_dynamic_screen(&renderer.screen);

                device.cmd_bind_pipeline(
                    *command_buffer_writer,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.graphics_pipeline.handle(),
                );

                let descriptor_sets_to_bind = [draw.descriptor_sets[renderer.image_index].handle()];
                device.cmd_bind_descriptor_sets(
                    *command_buffer_writer,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw.graphics_pipeline.layout(),
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.cmd_bind_vertex_buffers(
                    *command_buffer_writer,
                    0,
                    &[draw.vertex_buffer.handle()],
                    &[0],
                );

                let mut prev_level = usize::MAX;

                let (cam, cam_trans) = camera.single();

                let draw = scene
                    .uniform_transforms
                    .par_iter()
                    .map(|u| {
                        let planes =
                            planes_from_mat(scene.uniform_camera.culling_view_proj * u.model);
                        // Calculate screen space error
                        //let center = *transform.get_pos();
                        // vec3 cam =  ubo.camera_pos;
                        // vec3 center = (models[idy].model * vec4(clusters[idx].center, 1.0)).xyz ;
                        let radius = mesh_data.size;

                        let sphere = glam::Vec3A::ZERO.extend(radius);

                        if !sphere_inside_planes(&planes, sphere) {
                            return None;
                        }

                        // float radius = length((models[idy].model * vec4(normalize(vec3(1)) * clusters[idx].radius, 0.0)).xyz);

                        let mut level = 0;
                        let mut current_error = 0.0;

                        let local_cam_pos = u.inv_model.transform_point3a(*cam_trans.get_pos());

                        while current_error <= scene.target_error
                            && level < mesh_data.lod_chain.len()
                        {
                            // center is zero - model space
                            let inv_distance = local_cam_pos.length_recip();

                            let err_radius = mesh_data.lod_chain[level].error * radius;

                            current_error = err_radius * inv_distance;

                            level += 1;
                        }
                        // rust doesn't have do-while
                        level -= 1;

                        Some(level)
                    })
                    .collect::<Vec<_>>();

                for (transform, mesh) in meshes.iter() {
                    if mesh.id >= scene.uniform_transforms.len() {
                        continue;
                    }

                    let Some(level) = draw[mesh.id] else {
                        continue;
                    };

                    // Draw

                    if level != prev_level {
                        device.cmd_bind_index_buffer(
                            *command_buffer_writer,
                            mesh_data.lod_chain[level].index_buffer.handle(),
                            0,
                            vk::IndexType::UINT32,
                        );
                        prev_level = level;
                    }

                    // Each instance has their own indirect drawing buffer, tracing out their position in the result buffer
                    device.cmd_draw_indexed(
                        *command_buffer_writer,
                        mesh_data.lod_chain[level].index_buffer.len() as _,
                        1,
                        0,
                        0,
                        mesh.id as _,
                    );
                }

                device.cmd_end_render_pass(*command_buffer_writer);
            }
        }
        renderer.hacky_command_buffer_passthrough =
            Some(draw.command_buffers[renderer.image_index].handle());
    }
}

impl DrawLODChainData {
    pub fn new(renderer: &Renderer, mesh_data: &MeshData, scene: &Scene) -> Self {
        let ubo_layout = create_traditional_graphics_descriptor_set_layout(&renderer.core);

        let graphics_pipeline = create_traditional_graphics_pipeline(
            &renderer.core,
            &renderer.render_pass,
            renderer.screen.swapchain().extent,
            ubo_layout.clone(),
        );

        let descriptor_sets = create_traditional_graphics_descriptor_sets(
            &renderer.core.device,
            &renderer.descriptor_pool,
            &ubo_layout,
            &scene.uniform_transform_buffer,
            &scene.uniform_camera_buffers,
            renderer.screen.swapchain().images.len(),
        );

        Self {
            graphics_pipeline,
            descriptor_sets,
            core: renderer.core.clone(),
            descriptor_pool: renderer.descriptor_pool.clone(),
            query: renderer.query,
            vertex_buffer: mesh_data.vertex_buffer.clone(),
            command_buffers: Vec::new(),
        }
    }
}

pub struct DrawLODChain;

impl DrawLODChain {
    pub fn new(
        renderer: &Renderer,
        mesh_data: &MeshData,
        scene: &Scene,
        commands: &mut Commands,
    ) -> Self {
        commands.insert_resource(DrawLODChainData::new(renderer, mesh_data, scene));
        DrawLODChain
    }
}

impl DrawPipeline for DrawLODChain {
    fn draw(
        &self,
        frame_index: usize,
        screen: &Screen,
        render_pass: &RenderPass,
    ) -> &CommandBuffer {
        panic!("Should not call draw - has set hacky workaround that needs to be replaced")
    }
    fn init_swapchain(&mut self, core: &Core, screen: &Screen, render_pass: &RenderPass) {
        // only ever top-up our commands - we dont need to reset on resize
    }

    fn stats_gui(&mut self, _ui: &mut egui::Ui, _image_index: usize) {}

    fn cleanup(&mut self, commands: &mut Commands) {
        commands.remove_resource::<DrawLODChainData>()
    }
}
