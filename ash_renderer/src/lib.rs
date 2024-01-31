pub mod core;
pub mod draw_pipelines;
pub mod gui;
pub mod multires;
pub mod screen;
pub mod utility;
pub mod vertex;

pub mod app;
pub mod spiral;

use crate::{
    app::{
        draw_systems::{draw_frame, tick_clocks, update_pipeline},
        mesh_data::MeshDataBuffers,
        renderer::{MeshDrawingPipelineType, Renderer},
        scene::{process_scene_events, SceneEvent},
    },
    core::Core,
    gui::allocator_visualiser_window::AllocatorVisualiserWindow,
    screen::find_depth_format,
    spiral::Spiral,
    utility::{
        // the mod define some fixed functions that have been learned before.
        constants::*,
        pooled::descriptor_pool::DescriptorPool,
        render_pass::RenderPass,
        structures::*,
        swapchain::SwapChainSupportDetail,
        sync::SyncObjects,
    },
};

use app::{fps_limiter::FPSMeasure, scene::Scene};
use ash::vk;

use bevy_ecs::prelude::*;
use common::{asset::Asset, MeshVert, MultiResMesh};
use common_renderer::{
    components::{
        camera::Camera,
        camera_controller::{
            camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
        },
        gpu_mesh_util::MultiResData,
        transform::Transform,
    },
    resources::time::Time,
};
use draw_pipelines::{
    compute_culled_indices::ComputeCulledIndices, compute_culled_mesh::ComputeCulledMesh,
    draw_indirect::DrawIndirect, indirect_tasks::IndirectTasks, stub::Stub, DrawPipeline,
};
use glam::{vec3a, Quat, Vec3A};
use gpu_allocator::{vulkan::*, AllocationSizes, AllocatorDebugSettings};
use gui::{gui::Gui, window::GuiWindow};
use rand::{Rng, SeedableRng};
use screen::Screen;
use utility::buffer::{AsBuffer, Buffer, TBuffer};
use winit::event::WindowEvent;

use std::{f32::consts::PI, sync::Arc};
use std::{
    mem::take,
    sync::{Mutex, MutexGuard},
};

// Constants
const WINDOW_TITLE: &'static str = "Multires Mesh Renderer";
const TASK_GROUP_SIZE: u32 = 4;

pub trait VkHandle {
    type VkItem;

    fn handle(&self) -> Self::VkItem;
}
pub trait VkDeviceOwned: VkHandle<VkItem = vk::Device> {}

pub struct App {
    world: bevy_ecs::world::World,
    schedule: bevy_ecs::schedule::Schedule,
    draw_schedule: bevy_ecs::schedule::Schedule,
    camera: Entity,
}

/// https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
fn random_point_on_sphere(rng: &mut impl rand::Rng) -> glam::Vec3A {
    let u = rng.gen_range(0.0..=1.0);
    let v = rng.gen_range(0.0..=1.0);
    let c = rng.gen_range(0.0..=1.0);
    let theta = u * 2.0 * PI;
    let phi = f32::acos(2.0 * v - 1.0);
    let r = f32::cbrt(c);

    let (sin_theta, cos_theta) = f32::sin_cos(theta);
    let (sin_phi, cos_phi) = f32::sin_cos(phi);

    let x = r * sin_phi * cos_theta;
    let y = r * sin_phi * sin_theta;
    let z = r * cos_phi;
    vec3a(x, y, z)
}

impl App {
    pub fn renderer(&self) -> &Renderer {
        self.world.get_non_send_resource().as_ref().unwrap()
    }
    pub fn scene(&self) -> &Scene {
        self.world.get_resource().as_ref().unwrap()
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        if self
            .world
            .get_non_send_resource_mut::<Renderer>()
            .as_mut()
            .unwrap()
            .gui
            .handle_event(event)
        {
            return true;
        }

        match event {
            WindowEvent::MouseInput { state, button, .. } => self
                .world
                .send_event(MouseIn(state.clone(), button.clone())),
            WindowEvent::KeyboardInput { input, .. } => self.world.send_event(KeyIn(input.clone())),
            WindowEvent::CursorMoved { position, .. } => {
                self.world.send_event(MouseMv(position.clone()));
            }
            _ => (),
        }

        false
    }

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> App {
        let core = Core::new(event_loop);

        let device = &core.device;
        let physical_device = &core.physical_device;
        let instance = &core.instance;
        let window = &core.window;
        let queue_family = &core.queue_family;
        let surface = &core.surface;
        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.handle.clone(),
                device: device.handle.clone(),
                physical_device: physical_device.handle(),
                debug_settings: AllocatorDebugSettings {
                    log_memory_information: false,
                    log_leaks_on_shutdown: true,
                    store_stack_traces: false,
                    log_allocations: false,
                    log_frees: false,
                    log_stack_traces: false,
                },
                buffer_device_address: true, // Ideally, check the BufferDeviceAddressFeatures struct.
                allocation_sizes: AllocationSizes::new(1 << 24, 1 << 24), // 16 MB for both
            })
            .unwrap(),
        ));

        println!("Loading queues");
        let graphics_queue = unsafe {
            device
                .handle
                .get_device_queue(queue_family.graphics_family.unwrap(), 0)
        };
        let present_queue = unsafe {
            device
                .handle
                .get_device_queue(queue_family.present_family.unwrap(), 0)
        };

        println!("Loading swapchain");

        let mut screen = Screen::new(core.clone());

        let swapchain_support = SwapChainSupportDetail::query(physical_device.handle(), &surface);

        let surface_format = swapchain_support.choose_swapchain_format().format;
        let depth_format = find_depth_format(&instance.handle, physical_device);

        let render_pass = RenderPass::new(device.clone(), surface_format, depth_format);

        screen.remake_swapchain(graphics_queue, &render_pass, allocator.clone());

        // println!("Loading texture");
        // let texture_image = Image::create_texture_image(
        //     device.clone(),
        //     allocator.clone(),
        //     &command_pool,
        //     graphics_queue,
        //     &physical_device_memory_properties,
        //     &Path::new(TEXTURE_PATH),
        // )
        // .create_texture_sampler()
        // .create_texture_image_view(1);

        let mut uniform_transforms = Vec::new();

        let mut world: World = World::new();

        let mut r = rand::rngs::StdRng::seed_from_u64(42);

        let mut mid = Vec3A::ZERO;

        let instances = 100;

        for (i, j) in Spiral::default().take(instances) {
            let p = glam::Vec3A::X * i as f32 * 20.0 + glam::Vec3A::Z * j as f32 * 40.0;
            mid += p;
            let mut transform = Transform::new_pos(p);

            //if i == 10 && j == 10 {
            //*transform.scale_mut() *= 10.0;
            //transform.scale_mut().z *= 10.0;
            //};

            uniform_transforms.push(ModelUniformBufferObject {
                model: transform.get_local_to_world(),
                inv_model: transform.get_local_to_world().inverse(),
            });

            world.spawn(transform);
        }

        mid /= (instances) as f32;

        let uniform_transform_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &uniform_transforms,
            "Transform Buffer",
        );

        let uniform_camera_buffers = TBuffer::<CameraUniformBufferObject>::new_per_swapchain(
            &core,
            allocator.clone(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            screen.swapchain().images.len(),
            "Camera Buffer",
        );

        println!("Loading descriptors");
        let descriptor_pool =
            DescriptorPool::new(device.clone(), screen.swapchain().images.len() as u32);

        println!("Loading command buffers");

        let mesh_draw = Stub;

        let sync_objects: SyncObjects = SyncObjects::new(device.clone(), MAX_FRAMES_IN_FLIGHT);

        println!("Generated App");

        let cam = Camera::new(1.0);
        let transform = Transform::new(mid + Vec3A::Y * 200.0, Quat::IDENTITY);

        let uniform_camera = CameraUniformBufferObject {
            view_proj: cam.build_view_projection_matrix(&transform),
            cam_pos: (*transform.get_pos()).into(),
            target_error: 0.5,
        };

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());
        world.insert_resource(Events::<SceneEvent>::default());
        world.insert_resource(Events::<MeshDrawingPipelineType>::default());
        world.insert_resource(Time::default());

        let camera = world
            .spawn((CameraController::new(50.0), cam, transform))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((
            camera_handle_input,
            update_camera,
            process_scene_events,
            update_pipeline,
        ));
        let mut draw_schedule = Schedule::default();
        draw_schedule.add_systems((tick_clocks, draw_frame));

        let gui = Gui::new(
            device.clone(),
            window.clone(),
            event_loop,
            allocator.clone(),
            &core.command_pool,
            &queue_family,
            graphics_queue,
            &screen.swapchain(),
        );
        let mesh = MeshDataBuffers::new(&core, &allocator, graphics_queue);

        world.insert_resource(Scene {
            uniform_transform_buffer,
            uniform_camera,
            uniform_camera_buffers,
            target_error: 0.5,
            freeze_pos: false,
            instances,
        });

        world.insert_non_send_resource(Renderer {
            graphics_queue,
            present_queue,
            render_pass,
            draw_pipeline: Box::new(mesh_draw),
            descriptor_pool,
            gui,
            windows: vec![Box::new(AllocatorVisualiserWindow::new(allocator.clone()))],

            allocator,
            current_pipeline: MeshDrawingPipelineType::None,
            sync_objects,
            query: true,
            current_frame: 0,
            is_framebuffer_resized: false,
            app_info_open: true,
            //FIXME: This should be used directly
            cluster_count: mesh.cluster_buffer.item_len() as _,
            core,
            screen,
        });
        world.insert_resource(mesh);
        world.insert_resource(FPSMeasure::new());

        world.send_event(MeshDrawingPipelineType::IndirectTasks);

        // cleanup(); the 'drop' function will take care of it.
        App {
            world,
            schedule,
            draw_schedule,
            camera,
        }
    }
}

// Fix content -------------------------------------------------------------------------------

// fn update_model_uniform_buffer(&mut self, current_image: usize, _delta_time: f32) {
//     unsafe {
//         self.uniform_transform_buffer
//             .update_uniform_buffer(self.uniform_transform);
//     }
// }

#[cfg(test)]
mod tests {
    use crate::Spiral;

    #[test]
    fn test_spiral() {
        for (x, y) in Spiral::default().skip(10).take(50) {
            print!("({} {})", x, y)
        }
    }
}
