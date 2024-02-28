use crate::{
    app::{
        benchmarker::benchmark,
        draw_systems::{draw_frame, draw_gui, start_gui, tick_clocks, update_pipeline},
        mesh_data::MeshDataBuffers,
        renderer::{Fragment, MeshDrawingPipelineType, Renderer},
        scene::{
            process_scene_events, CameraUniformBufferObject, ModelUniformBufferObject, SceneEvent,
        },
    },
    core::Core,
    draw_pipelines::indirect_tasks::MeshShaderMode,
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
        ShaderModule,
    },
};

use crate::app::{fps_limiter::FPSMeasure, scene::Scene};
use ash::vk;

use bevy_ecs::prelude::*;
use bytemuck::Zeroable;

use crate::draw_pipelines::stub::Stub;
use crate::gui::gui::Gui;
use crate::screen::Screen;
use crate::utility::buffer::TBuffer;
use crate::VkHandle;
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
use glam::{vec3a, Quat, Vec3A};
use gpu_allocator::{vulkan::*, AllocationSizes, AllocatorDebugSettings};
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use std::{f32::consts::PI, sync::Arc};
use winit::event::WindowEvent;

pub struct App {
    pub world: bevy_ecs::world::World,
    pub schedule: bevy_ecs::schedule::Schedule,
    pub draw_schedule: bevy_ecs::schedule::Schedule,
    pub camera: Entity,
}

impl App {
    pub fn renderer(&self) -> &Renderer {
        self.world.get_resource().as_ref().unwrap()
    }
    pub fn scene(&self) -> &Scene {
        self.world.get_resource().as_ref().unwrap()
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        if self
            .world
            .get_non_send_resource_mut::<Gui>()
            .as_mut()
            .unwrap()
            .handle_event(event)
        {
            return true;
        }

        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                self.world.send_event(MouseIn(*state, *button))
            }
            WindowEvent::KeyboardInput { input, .. } => self.world.send_event(KeyIn(*input)),
            WindowEvent::CursorMoved { position, .. } => {
                self.world.send_event(MouseMv(*position));
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

        let swapchain_support = SwapChainSupportDetail::query(physical_device.handle(), surface);

        let surface_format = swapchain_support.choose_swapchain_format().format;
        let depth_format = find_depth_format(&instance, physical_device);

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

        let mesh_draw = Stub;

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
        world.insert_resource(Events::<SceneEvent>::default());
        world.insert_resource(Events::<MeshDrawingPipelineType>::default());
        world.insert_resource(Time::default());

        if core.device.features.mesh_shader {
            world.send_event(MeshDrawingPipelineType::IndirectTasks);
        } else {
            world.send_event(MeshDrawingPipelineType::ComputeCulledIndices);
        }

        let camera = world
            .spawn((CameraController::new(50.0), cam, transform))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((
            camera_handle_input,
            benchmark,
            update_camera,
            process_scene_events,
            update_pipeline,
        ));
        let mut draw_schedule = Schedule::default();
        draw_schedule.add_systems((
            tick_clocks,
            start_gui,
            draw_gui.after(start_gui).before(draw_frame),
            draw_frame,
        ));

        let gui = Gui::new(
            device.clone(),
            window.clone(),
            event_loop,
            allocator.clone(),
            &core.command_pool,
            queue_family,
            graphics_queue,
            screen.swapchain(),
        );
        let mesh = MeshDataBuffers::new(&core, &allocator, graphics_queue);

        world.insert_resource(Scene {
            uniform_transform_buffer,
            uniform_camera,
            uniform_camera_buffers,
            target_error: 0.1,
            freeze_pos: false,
            instances: 0,
        });

        world.insert_resource(Renderer {
            fragment_colour: ShaderModule::new(
                device.clone(),
                include_bytes!("../../shaders/spv/frag_colour.frag"),
            ),
            fragment_lit: ShaderModule::new(
                device.clone(),
                include_bytes!("../../shaders/spv/frag_pbr.frag"),
            ),

            graphics_queue,
            present_queue,
            render_pass,
            draw_pipeline: Box::new(mesh_draw),
            descriptor_pool,
            windows: vec![Box::new(AllocatorVisualiserWindow::new(allocator.clone()))],
            mesh_mode: MeshShaderMode::TriangleList,
            allocator,
            current_pipeline: MeshDrawingPipelineType::None,
            sync_objects,
            query: false,
            current_frame: 0,
            is_framebuffer_resized: false,
            app_info_open: true,
            render_gui: true, // disable GUI during benchmarks
            core,
            screen,
            fragment: Fragment::Lit,
        });
        world.insert_non_send_resource(gui);
        world.insert_resource(mesh);
        world.insert_resource(FPSMeasure::new());

        // cleanup(); the 'drop' function will take care of it.
        App {
            world,
            schedule,
            draw_schedule,
            camera,
        }
    }
}
