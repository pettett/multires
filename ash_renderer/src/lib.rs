pub mod core;
pub mod draw_pipelines;
pub mod gui;
pub mod multires;
pub mod screen;
pub mod utility;

use crate::{
    core::Core,
    gui::allocator_visualiser_window::AllocatorVisualiserWindow,
    screen::find_depth_format,
    utility::{
        // the mod define some fixed functions that have been learned before.
        constants::*,
        descriptor_pool::DescriptorPool,
        render_pass::RenderPass,
        structures::*,
        swapchain::{SwapChainSupportDetail, Swapchain},
        sync::SyncObjects,
    },
};

use ash::vk;
use bevy_ecs::{entity::Entity, event::Events, schedule::Schedule, world::World};
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
    indirect_tasks::IndirectTasks, stub::Stub, DrawPipeline,
};
use glam::{Mat4, Quat, Vec3A};
use gpu_allocator::{vulkan::*, AllocationSizes, AllocatorDebugSettings};
use gui::{gui::Gui, window::GuiWindow};
use screen::Screen;
use utility::{
    buffer::{AsBuffer, Buffer, TBuffer},
    descriptor_pool::DescriptorSet,
};
use winit::event::WindowEvent;

use std::{any::Any, sync::Arc};
use std::{
    mem,
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

#[derive(Debug, Clone, Copy)]
pub enum MeshDrawingPipelineType {
    IndirectTasks,
    ComputeCulledMesh,
    ComputeCulledIndices,
    None,
}

pub trait Vertex {
    fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription>;
    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

impl Vertex for MeshVert {
    fn get_binding_descriptions() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<MeshVert>() as _,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: mem::size_of::<[f32; 4]>() as _,
            },
        ]
    }
}

pub struct App {
    world: bevy_ecs::world::World,
    schedule: bevy_ecs::schedule::Schedule,
    camera: Entity,

    pub allocator: Arc<Mutex<Allocator>>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    //texture_image: Image,
    uniform_transforms: Vec<ModelUniformBufferObject>,
    uniform_transform_buffer: Arc<Buffer>,
    uniform_camera: CameraUniformBufferObject,
    uniform_camera_buffers: Vec<Arc<TBuffer<CameraUniformBufferObject>>>,

    render_pass: RenderPass,

    draw: Box<dyn DrawPipeline>,

    descriptor_pool: Arc<DescriptorPool>,

    gui: Gui,
    windows: Vec<Box<dyn GuiWindow>>,
    switch_pipeline: MeshDrawingPipelineType,
    current_pipeline: MeshDrawingPipelineType,

    sync_objects: SyncObjects,

    current_frame: usize,

    vertex_buffer: Arc<TBuffer<MeshVert>>,
    meshlet_buffer: Arc<Buffer>,
    submesh_buffer: Arc<Buffer>,
    indices_buffer: Arc<TBuffer<u32>>,

    is_framebuffer_resized: bool,

    submesh_count: u32,
    app_info_open: bool,

    // Make sure to drop the core last
    screen: Screen,
    core: Arc<Core>,
}

impl App {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        if self.gui.handle_event(event) {
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

        println!("Loading verts");

        let data = MultiResMesh::load().unwrap();

        let (clusters, meshlets) = multires::generate_meshlets(&data);

        let mut cluster_data = data.generate_cluster_data();
        let (indices, partitions, groups) = data.indices_partitions_groups();

        for (i, submesh) in clusters.into_iter().enumerate() {
            cluster_data[i].meshlet_start = submesh.meshlet_start;
            cluster_data[i].meshlet_count = submesh.meshlet_count;
        }

        println!("V: {:?} M: {:?}", data.verts.len(), meshlets.len());

        let vertex_buffer = TBuffer::new_filled(
            device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            &data.verts,
        );

        let meshlet_buffer = Buffer::new_storage_filled(
            device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            &meshlets,
        );

        let submesh_buffer = Buffer::new_storage_filled(
            device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            &cluster_data,
        );

        let indices_buffer = TBuffer::new_filled(
            core.device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            // Allow index use for testing
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );

        let mut uniform_transforms = Vec::new();

        let mut world: World = World::new();
        for i in 0..1 {
            for j in 0..1 {
                let mut transform = Transform::new_pos(
                    glam::Vec3A::X * i as f32 * 40.0 + glam::Vec3A::Y * j as f32 * 40.0,
                );

                if i == 10 && j == 10 {
                    *transform.scale_mut() *= 20.0
                };

                uniform_transforms.push(ModelUniformBufferObject {
                    model: transform.get_local_to_world(),
                });

                world.spawn(transform);
            }
        }

        let uniform_transform_buffer = Buffer::new_storage_filled(
            device.clone(),
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            &uniform_transforms,
        );

        let uniform_camera_buffers = TBuffer::<CameraUniformBufferObject>::new_per_swapchain(
            device.clone(),
            allocator.clone(),
            gpu_allocator::MemoryLocation::CpuToGpu,
            screen.swapchain().images.len(),
        );

        println!("Loading descriptors");
        let descriptor_pool =
            DescriptorPool::new(device.clone(), screen.swapchain().images.len() as u32);

        println!("Loading command buffers");

        let mesh_draw = Stub;

        let sync_objects: SyncObjects = SyncObjects::new(device.clone(), MAX_FRAMES_IN_FLIGHT);

        println!("Generated App");

        let cam = Camera::new(1.0);
        let transform = Transform::new(Vec3A::ZERO, Quat::IDENTITY);

        let uniform_camera = CameraUniformBufferObject {
            view_proj: cam.build_view_projection_matrix(&transform),
            cam_pos: (*transform.get_pos()).into(),
            target_error: 0.3,
        };

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());
        world.insert_resource(Time::default());

        let camera = world
            .spawn((CameraController::new(50.0), cam, transform))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((camera_handle_input, update_camera));

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

        // cleanup(); the 'drop' function will take care of it.
        App {
            gui,
            windows: vec![Box::new(AllocatorVisualiserWindow::new(allocator.clone()))],

            core,
            screen,

            world,
            schedule,
            camera,
            allocator,

            // vulkan stuff
            graphics_queue,
            present_queue,
            uniform_camera,

            vertex_buffer,
            submesh_buffer,
            meshlet_buffer,
            indices_buffer,

            render_pass,

            draw: Box::new(mesh_draw),
            switch_pipeline: MeshDrawingPipelineType::ComputeCulledIndices,
            current_pipeline: MeshDrawingPipelineType::None,

            uniform_transforms,
            uniform_transform_buffer,
            uniform_camera_buffers,

            //texture_image,
            descriptor_pool,

            sync_objects,
            current_frame: 0,

            is_framebuffer_resized: false,

            submesh_count: cluster_data.len() as u32,
            app_info_open: true,
        }
    }
}

// Fix content -------------------------------------------------------------------------------
impl App {
    fn update_camera_uniform_buffer(&mut self, current_image: usize) {
        let cam = self.world.entity(self.camera);

        self.uniform_camera.view_proj = cam
            .get::<Camera>()
            .unwrap()
            .build_view_projection_matrix(cam.get().unwrap());

        self.uniform_camera.cam_pos = (*cam.get::<Transform>().unwrap().get_pos()).into();

        self.uniform_camera_buffers[current_image].update_uniform_buffer(self.uniform_camera);
    }

    // fn update_model_uniform_buffer(&mut self, current_image: usize, _delta_time: f32) {
    //     unsafe {
    //         self.uniform_transform_buffer
    //             .update_uniform_buffer(self.uniform_transform);
    //     }
    // }

    fn draw_frame(&mut self) {
        let wait_fences = [self.sync_objects.in_flight_fences[self.current_frame]];

        unsafe {
            self.core
                .device
                .handle
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.core.device.fn_swapchain.acquire_next_image(
                self.screen.swapchain().handle,
                std::u64::MAX,
                self.sync_objects.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );

            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
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

        self.update_camera_uniform_buffer(image_index as usize);

        let wait_semaphores = [self.sync_objects.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.sync_objects.render_finished_semaphores[self.current_frame]];

        let ui_cmd = self.draw_gui(image_index as usize);

        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[self.draw.draw(image_index as usize), ui_cmd])
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .wait_semaphores(&wait_semaphores)
            .build()];

        unsafe {
            self.core
                .device
                .handle
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.core
                .device
                .handle
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.sync_objects.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.screen.swapchain().handle];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .wait_semaphores(&signal_semaphores)
            .image_indices(&image_indices);

        let result = unsafe {
            self.core
                .device
                .fn_swapchain
                .queue_present(self.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn update_pipeline(&mut self) {
        match self.switch_pipeline {
            MeshDrawingPipelineType::None => return,
            _ => self.core.device.wait_device_idle(),
        }

        self.draw = Box::new(Stub);

        match self.switch_pipeline {
            MeshDrawingPipelineType::IndirectTasks => {
                self.draw = Box::new(IndirectTasks::new(
                    self.core.clone(),
                    &self.screen,
                    &mut self.world,
                    self.allocator.clone(),
                    &self.render_pass,
                    self.graphics_queue,
                    self.descriptor_pool.clone(),
                    self.uniform_transform_buffer.clone(),
                    &self.uniform_camera_buffers,
                    self.vertex_buffer.buffer(),
                    self.meshlet_buffer.clone(),
                    self.submesh_buffer.clone(),
                    self.submesh_count,
                ));
            }
            MeshDrawingPipelineType::ComputeCulledMesh => {
                self.draw = Box::new(ComputeCulledMesh::new(
                    self.core.clone(),
                    &self.screen,
                    &mut self.world,
                    self.allocator.clone(),
                    &self.render_pass,
                    self.graphics_queue,
                    self.descriptor_pool.clone(),
                    self.uniform_transform_buffer.clone(),
                    &self.uniform_camera_buffers,
                    self.vertex_buffer.buffer(),
                    self.meshlet_buffer.clone(),
                    self.submesh_buffer.clone(),
                    self.submesh_count,
                ));
            }
            MeshDrawingPipelineType::ComputeCulledIndices => {
                self.draw = Box::new(ComputeCulledIndices::new(
                    self.core.clone(),
                    &self.screen,
                    &mut self.world,
                    self.allocator.clone(),
                    &self.render_pass,
                    self.graphics_queue,
                    self.descriptor_pool.clone(),
                    self.uniform_transform_buffer.clone(),
                    &self.uniform_camera_buffers,
                    self.vertex_buffer.clone(),
                    self.meshlet_buffer.clone(),
                    self.submesh_buffer.clone(),
                    self.indices_buffer.clone(),
                    self.submesh_count,
                ));
            }
            MeshDrawingPipelineType::None => (),
        }

        self.draw.init_swapchain(
            &self.core,
            &self.screen,
            self.submesh_count,
            self.uniform_transforms.len() as _,
            &self.render_pass,
        );

        self.current_pipeline = self.switch_pipeline;
        self.switch_pipeline = MeshDrawingPipelineType::None
    }

    fn recreate_swapchain(&mut self) {
        self.core.device.wait_device_idle();

        //self.cleanup_swapchain();

        self.screen.remake_swapchain(
            self.graphics_queue,
            &self.render_pass,
            self.allocator.clone(),
        );

        self.draw.init_swapchain(
            &self.core,
            &self.screen,
            self.submesh_count,
            self.uniform_transforms.len() as _,
            &self.render_pass,
        );

        // Egui Integration
        self.gui.update_swapchain(&self.screen.swapchain());
    }

    fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn window_ref(&self) -> &winit::window::Window {
        &self.core.window
    }
    pub fn get_allocator(&self) -> MutexGuard<Allocator> {
        self.allocator.lock().unwrap()
    }
    fn draw_gui(&mut self, image_index: usize) -> vk::CommandBuffer {
        self.gui.draw(image_index, |ctx| {
            for w in &mut self.windows {
                w.draw(ctx);
            }
            egui::Window::new("App Config")
                .open(&mut self.app_info_open)
                .show(ctx, |ui| {
                    ui.label(format!("Current Pipeline: {:?}", self.current_pipeline));

                    if ui.button("Indirect Tasks").clicked() {
                        self.switch_pipeline = MeshDrawingPipelineType::IndirectTasks;
                    }
                    if ui.button("Compute Culled Mesh").clicked() {
                        self.switch_pipeline = MeshDrawingPipelineType::ComputeCulledMesh;
                    }
                    if ui.button("Compute Culled Indices").clicked() {
                        self.switch_pipeline = MeshDrawingPipelineType::ComputeCulledIndices;
                    }
                });

            egui::TopBottomPanel::top("test").show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    for w in &mut self.windows {
                        let (open, name) = w.state();
                        ui.checkbox(open, name);
                    }
                    ui.checkbox(&mut self.app_info_open, "App Info");
                });
            });
        })
    }
}
