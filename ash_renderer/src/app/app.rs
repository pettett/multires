use crate::app::eval::random_benchmarker::random_benchmark;
use crate::app::eval::scene_complexity_benchmarker::scene_complexity_benchmark;
use crate::app::{fps_limiter::FPSMeasure, scene::Scene};
use crate::gui::gui::Gui;
use crate::utility::buffer::TBuffer;
use crate::VkHandle;
use crate::{
    app::{
        draw_systems::{
            acquire_swapchain, draw_frame, draw_gui, gather_queries, start_gui, tick_clocks,
            update_pipeline,
        },
        eval::recorder::record,
        eval::zoom_benchmarker::benchmark,
        mesh_data::MeshData,
        renderer::{MeshDrawingPipelineType, Renderer},
        scene::{
            process_scene_events, CameraUniformBufferObject, ModelUniformBufferObject, SceneEvent,
        },
    },
    core::Core,
    draw_pipelines::draw_lod_chain::create_lod_command_buffer,
    utility::{
        constants::*,
        pooled::descriptor_pool::DescriptorPool,
        render_pass::RenderPass,
        screen::{find_depth_format, Screen},
        sync::SyncObjects,
    },
    Config,
};
use ash::vk;
use bevy_ecs::prelude::*;
use bytemuck::Zeroable;
use common_renderer::{
    components::{
        camera::Camera,
        camera_controller::{
            camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
        },
        transform::Transform,
    },
    resources::time::Time,
};
use egui::ahash::HashMap;
use glam::{Quat, Vec3A};
use gpu_allocator::{vulkan::*, AllocationSizes, AllocatorDebugSettings};
use std::sync::Arc;
use std::sync::Mutex;
use winit::event::WindowEvent;

pub struct App {
    pub world: bevy_ecs::world::World,
    pub schedule: bevy_ecs::schedule::Schedule,
    pub draw_schedule: bevy_ecs::schedule::Schedule,
    pub camera: Entity,
}

#[derive(Resource)]
pub struct AssetLib<T> {
    assets: HashMap<String, T>,
}

impl<T> Default for AssetLib<T> {
    fn default() -> Self {
        Self {
            assets: Default::default(),
        }
    }
}
impl<T> AssetLib<T> {
    pub fn get(&self, name: &str) -> &T {
        &self.assets[name]
    }
}

#[derive(Event, Debug)]
pub enum QueryEvent {
    ClippedPrimitives(u32),
    GPUMilliseconds(f64),
}

impl App {
    pub fn renderer(&self) -> &Renderer {
        self.world.get_resource().as_ref().unwrap()
    }
    pub fn scene(&self) -> &Scene {
        self.world.get_resource().as_ref().unwrap()
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        let core = self.renderer().core.clone();

        if self
            .world
            .get_non_send_resource_mut::<Gui>()
            .as_mut()
            .unwrap()
            .handle_event(&core.window, event)
        {
            return true;
        }

        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                self.world.send_event(MouseIn(*state, *button))
            }
            WindowEvent::KeyboardInput { event, .. } => self.world.send_event(KeyIn(event.clone())),
            WindowEvent::CursorMoved { position, .. } => {
                self.world.send_event(MouseMv(*position));
            }
            _ => (),
        }

        false
    }

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>, config: Config) -> App {
        assert!(config.mesh_names.len() >= 1);

        let core = Core::new(event_loop, &config);

        let device = &core.device;
        let physical_device = &core.physical_device;
        let instance = &core.instance;
        let window = &core.window;
        let queue_family = &core.queue_family;
        let surface = &core.surface;
        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: (**instance).clone(),
                device: (**device).clone(),
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
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };

        println!("Loading swapchain");

        let mut screen = Screen::new(core.clone());

        let swapchain_support = surface.swapchain_support(physical_device.handle());

        let surface_format = swapchain_support.choose_swapchain_format().format;
        let depth_format = find_depth_format(&instance, physical_device);

        let render_pass = RenderPass::new_color_depth(
            device.clone(),
            surface_format,
            depth_format,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

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

        let uniform_transforms = vec![ModelUniformBufferObject::zeroed()];

        let mut world = World::new();

        let mid = Vec3A::ZERO;

        world.send_event(SceneEvent::ResetScene);

        let uniform_transform_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
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

        let sync_objects: SyncObjects = SyncObjects::new(device.clone(), MAX_FRAMES_IN_FLIGHT);

        println!("Generated App");

        let cam = Camera::new(1.0);
        let mut transform = Transform::new(mid + Vec3A::Z * 200.0, Quat::IDENTITY);

        transform.look_at(glam::Vec3A::ZERO);

        let uniform_camera = CameraUniformBufferObject::new(
            cam.build_view_projection_matrix(&transform),
            (*transform.get_pos()).into(),
            0.5,
        );

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());
        world.insert_resource(Events::<QueryEvent>::default());
        world.insert_resource(Events::<SceneEvent>::default());
        world.insert_resource(Events::<MeshDrawingPipelineType>::default());
        world.insert_resource(Time::default());

        if core.device.features.mesh_shader {
            world.send_event(MeshDrawingPipelineType::LocalSelectMesh);
        } else {
            world.send_event(MeshDrawingPipelineType::DrawIndirect);
        }

        let camera = world
            .spawn((CameraController::new(50.0), cam, transform))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((
            camera_handle_input,
            benchmark,
            scene_complexity_benchmark,
            random_benchmark,
            update_camera,
            process_scene_events,
            update_pipeline.after(process_scene_events),
        ));
        let mut draw_schedule = Schedule::default();
        draw_schedule.add_systems((
            tick_clocks,
            start_gui,
            acquire_swapchain,
            (
                record,
                (
                    gather_queries,
                    draw_gui.after(start_gui).after(gather_queries),
                    create_lod_command_buffer,
                )
                    .before(draw_frame),
                draw_frame,
            )
                .after(acquire_swapchain),
        ));

        let gui = Gui::new(
            core.clone(),
            window.clone(),
            event_loop,
            allocator.clone(),
            &core.command_pool,
            queue_family,
            graphics_queue,
            screen.swapchain(),
        );

        let mut meshes = AssetLib::default();

        for n in &config.mesh_names {
            meshes.assets.insert(
                n.clone(),
                MeshData::new(&core, &allocator, graphics_queue, n),
            );
        }

        world.insert_resource(Scene {
            uniform_transform_buffer,
            uniform_camera,
            uniform_camera_buffers,
            target_error: config.starting_error,
            dist_pow: 0.5,
            freeze_culling: false,
            freeze_error: false,
            instances: 0,
            uniform_transforms,
        });

        world.insert_resource(Renderer::new(
            &config,
            graphics_queue,
            present_queue,
            screen,
            core,
            descriptor_pool,
            sync_objects,
            allocator,
            render_pass,
        ));

        world.insert_non_send_resource(gui);
        world.insert_resource(meshes);
        world.insert_resource(FPSMeasure::new());
        world.insert_resource(config);
        // cleanup(); the 'drop' function will take care of it.
        App {
            world,
            schedule,
            draw_schedule,
            camera,
        }
    }
}
