pub mod gui;
pub mod multires;
mod utility;

use crate::utility::{
    command_pool::CommandPool,
    // the mod define some fixed functions that have been learned before.
    constants::*,
    debug::*,
    descriptor_pool::{create_descriptor_set_layout, DescriptorPool},
    image::Image,
    instance::Instance,
    pipeline::create_graphics_pipeline,
    render_pass::create_render_pass,
    structures::*,
    sync::SyncObjects,
    window::ProgramProc,
};

use ash::{extensions::ext::MeshShader, vk};
use bevy_ecs::{entity::Entity, event::Events, schedule::Schedule, world::World};
use bytemuck::Zeroable;
use common::{asset::Asset, MultiResMesh};
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
use glam::{Mat4, Quat, Vec3A};
use gpu_allocator::vulkan::*;
use gui::gpu_allocator::GpuAllocator;
use utility::{
    buffer::{AsBuffer, Buffer, TypedBuffer},
    descriptor_pool::DescriptorSet,
    device::Device,
    physical_device::PhysicalDevice,
    pipeline::Pipeline,
    surface::Surface,
    swapchain::Swapchain,
};
use winit::event::WindowEvent;

use std::{
    mem,
    ptr::{self, null_mut},
    sync::Mutex,
};
use std::{path::Path, sync::Arc};

// Constants
const WINDOW_TITLE: &'static str = "26.Depth Buffering";
const TEXTURE_PATH: &'static str = "../../../assets/vulkan.jpg";
const TASK_GROUP_SIZE: u32 = 16;

pub trait VkHandle {
    type VkItem;

    fn handle(&self) -> Self::VkItem;
}
pub trait VkDeviceOwned: VkHandle<VkItem = vk::Device> {}

pub struct App {
    window: winit::window::Window,
    world: bevy_ecs::world::World,
    schedule: bevy_ecs::schedule::Schedule,
    camera: Entity,
    // vulkan stuff
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,

    allocator: Arc<Mutex<Allocator>>,
    integration: egui_winit_ash_integration::Integration<GpuAllocator>,

    physical_device: Arc<PhysicalDevice>,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Arc<Device>,

    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain: Swapchain,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    graphics_pipeline: Pipeline,

    depth_image: Image,
    texture_image: Image,

    indirect_task_buffer: Arc<TypedBuffer<vk::DrawMeshTasksIndirectCommandEXT>>,

    uniform_transforms: Vec<ModelUniformBufferObject>,
    uniform_transform_buffer: Arc<Buffer>,
    uniform_camera: CameraUniformBufferObject,
    uniform_camera_buffers: Vec<TypedBuffer<CameraUniformBufferObject>>,

    descriptor_pool: Arc<DescriptorPool>,
    command_pool: Arc<CommandPool>,

    descriptor_sets: Vec<DescriptorSet>,

    mesh_command_buffers: Vec<vk::CommandBuffer>,

    ui_command_buffers: Vec<vk::CommandBuffer>,
    visualizer: AllocatorVisualizer,
    visualizer_open: bool,

    sync_objects: SyncObjects,

    current_frame: usize,

    is_framebuffer_resized: bool,

    submesh_count: u32,
}

impl App {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        if self.integration.handle_event(event).consumed {
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
        println!("initing window");
        let window =
            utility::window::init_window(&event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        // init vulkan stuff
        println!("initing vulkan");
        let entry = ash::Entry::linked();
        let instance = Instance::new(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers.to_vec(),
        );

        println!("initing surface");
        let surface = Surface::new(
            &entry,
            instance.clone(),
            &window,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        );
        let (debug_utils_loader, debug_messenger) =
            setup_debug_utils(VALIDATION.is_enable, &entry, &instance.handle);
        let physical_device = instance.pick_physical_device(&surface, &DEVICE_EXTENSIONS);

        let physical_device_memory_properties = physical_device.get_memory_properties();

        let physical_device_subgroup_properties = physical_device.get_subgroup_properties();

        // Features required for subgroupMax to work in task shader
        assert!(TASK_GROUP_SIZE <= physical_device_subgroup_properties.subgroup_size);
        assert!(physical_device_subgroup_properties
            .supported_stages
            .contains(vk::ShaderStageFlags::TASK_EXT));
        assert!(physical_device_subgroup_properties
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::ARITHMETIC));

        let (device, queue_family) = Device::create_logical_device(
            instance.clone(),
            physical_device.clone(),
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface,
        );

        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.handle.clone(),
                device: device.handle.clone(),
                physical_device: physical_device.handle(),
                debug_settings: Default::default(),
                buffer_device_address: true, // Ideally, check the BufferDeviceAddressFeatures struct.
                allocation_sizes: Default::default(),
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
        let swapchain = Swapchain::new(
            device.clone(),
            &physical_device,
            &window,
            surface.clone(),
            &queue_family,
            None, // This is the first swapchain
        );
        let swapchain_imageviews =
            Image::create_image_views(&device, swapchain.surface_format.format, &swapchain.images);
        let render_pass = create_render_pass(
            &instance.handle,
            &device.handle,
            &physical_device,
            swapchain.surface_format.format,
        );
        let ubo_layout = create_descriptor_set_layout(&device.handle);
        let graphics_pipeline =
            create_graphics_pipeline(device.clone(), render_pass, swapchain.extent, ubo_layout);

        let command_pool = CommandPool::new(device.clone(), queue_family.graphics_family.unwrap());

        let depth_image = App::create_depth_resources(
            &instance.handle,
            device.clone(),
            &physical_device,
            command_pool.handle,
            graphics_queue,
            swapchain.extent,
            &physical_device_memory_properties,
        );
        let swapchain_framebuffers = App::create_framebuffers(
            &device.handle,
            render_pass,
            &swapchain_imageviews,
            depth_image.image_view(),
            swapchain.extent,
        );

        println!("Loading texture");
        let texture_image = Image::create_texture_image(
            device.clone(),
            &command_pool,
            graphics_queue,
            &physical_device_memory_properties,
            &Path::new(TEXTURE_PATH),
        )
        .create_texture_sampler()
        .create_texture_image_view(1);

        println!("Loading verts");

        let data = MultiResMesh::load().unwrap();

        let (submeshs, meshlets) = multires::generate_meshlets(&data);

        let mut cluster_data = data.generate_cluster_data();

        for (i, submesh) in submeshs.into_iter().enumerate() {
            cluster_data[i].meshlet_start = submesh.meshlet_start;
            cluster_data[i].meshlet_count = submesh.meshlet_count;
        }

        println!("V: {:?} M: {:?}", data.verts.len(), meshlets.len());

        let vertex_buffer = Buffer::new_storage_filled(
            device.clone(),
            &physical_device_memory_properties,
            &command_pool,
            graphics_queue,
            &data.verts,
        );

        let meshlet_buffer = Buffer::new_storage_filled(
            device.clone(),
            &physical_device_memory_properties,
            &command_pool,
            graphics_queue,
            &meshlets,
        );

        let submesh_buffer = Buffer::new_storage_filled(
            device.clone(),
            &physical_device_memory_properties,
            &command_pool,
            graphics_queue,
            &cluster_data,
        );

        let mut uniform_transforms = Vec::new();
        let mut task_indirect_data = Vec::new();

        for i in 0..100 {
            for j in 0..50 {
                let mut model = Mat4::from_translation(
                    glam::Vec3::X * i as f32 * 40.0 + glam::Vec3::Y * j as f32 * 40.0,
                );

                if i == 10 && j == 10 {
                    model *= Mat4::from_scale(glam::Vec3::ONE * 20.0)
                };

                uniform_transforms.push(ModelUniformBufferObject { model });

                task_indirect_data.push(vk::DrawMeshTasksIndirectCommandEXT {
                    group_count_x: (cluster_data.len() as u32).div_ceil(TASK_GROUP_SIZE) / 4,
                    group_count_y: 1,
                    group_count_z: 1,
                });
            }
        }

        let uniform_transform_buffer = Buffer::new_storage_filled(
            device.clone(),
            &physical_device_memory_properties,
            &command_pool,
            graphics_queue,
            &uniform_transforms,
        );

        let indirect_task_buffer = TypedBuffer::new_filled(
            device.clone(),
            &physical_device_memory_properties,
            &command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            &task_indirect_data,
        );

        let uniform_camera_buffers = TypedBuffer::<CameraUniformBufferObject>::new_per_swapchain(
            device.clone(),
            &physical_device_memory_properties,
            swapchain.images.len(),
        );

        println!("Loading descriptors");
        let descriptor_pool = DescriptorPool::new(device.clone(), swapchain.images.len() as u32);

        let descriptor_sets = DescriptorSet::create_descriptor_sets(
            &device,
            &descriptor_pool,
            ubo_layout,
            &uniform_transform_buffer,
            &uniform_camera_buffers,
            &vertex_buffer,
            &meshlet_buffer,
            &submesh_buffer,
            &indirect_task_buffer,
            &texture_image,
            swapchain.images.len(),
        );

        println!("Loading command buffers");

        let mesh_command_buffers = App::create_command_buffers(
            cluster_data.len() as _,
            uniform_transforms.len() as _,
            &device,
            &command_pool,
            graphics_pipeline.pipeline(),
            &swapchain_framebuffers,
            render_pass,
            swapchain.extent,
            graphics_pipeline.layout(),
            &descriptor_sets,
            &indirect_task_buffer,
        );

        let ui_command_buffers = command_pool.allocate_group(swapchain_framebuffers.len() as u32);

        let sync_objects: SyncObjects = SyncObjects::new(device.clone(), MAX_FRAMES_IN_FLIGHT);

        println!("Generated App");

        let cam = Camera::new(1.0);
        let transform = Transform::new(Vec3A::ZERO, Quat::IDENTITY);

        let uniform_camera = CameraUniformBufferObject {
            view_proj: cam.build_view_projection_matrix(&transform),
            cam_pos: (*transform.get_pos()).into(),
            target_error: 0.3,
        };

        let mut world: World = World::new();

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());
        world.insert_resource(Time::default());

        let camera = world
            .spawn((CameraController::new(50.0), cam, transform))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((camera_handle_input, update_camera));

        let integration = egui_winit_ash_integration::Integration::<GpuAllocator>::new(
            &event_loop,
            swapchain.extent.width,
            swapchain.extent.height,
            1.0,
            egui::FontDefinitions::default(),
            egui::Style::default(),
            device.handle.clone(),
            GpuAllocator(allocator.clone()),
            queue_family.graphics_family.unwrap(),
            graphics_queue,
            device.fn_swapchain.clone(),
            swapchain.handle,
            swapchain.surface_format,
        );

        // cleanup(); the 'drop' function will take care of it.
        App {
            // winit stuff
            window,
            world,
            schedule,
            camera,

            // vulkan stuff
            instance,
            surface,
            debug_utils_loader,
            debug_messenger,

            allocator,
            integration,

            physical_device,
            memory_properties: physical_device_memory_properties,
            device,

            queue_family,
            graphics_queue,
            present_queue,
            uniform_camera,

            indirect_task_buffer,

            uniform_transforms,
            uniform_transform_buffer,
            uniform_camera_buffers,

            swapchain,
            swapchain_imageviews,
            swapchain_framebuffers,

            ubo_layout,
            render_pass,
            graphics_pipeline,

            depth_image,

            texture_image,

            descriptor_pool,
            descriptor_sets,

            command_pool,
            mesh_command_buffers,
            ui_command_buffers,
            visualizer: AllocatorVisualizer::new(),
            visualizer_open: true,

            sync_objects,
            current_frame: 0,

            is_framebuffer_resized: false,

            submesh_count: cluster_data.len() as u32,
        }
    }

    fn create_depth_resources(
        instance: &ash::Instance,
        device: Arc<Device>,
        physical_device: &PhysicalDevice,
        _command_pool: vk::CommandPool,
        _submit_queue: vk::Queue,
        swapchain_extent: vk::Extent2D,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Image {
        let depth_format = App::find_depth_format(instance, physical_device);
        Image::create_image(
            device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            device_memory_properties,
        )
        .create_image_view(depth_format, vk::ImageAspectFlags::DEPTH, 1)
    }

    fn find_depth_format(instance: &ash::Instance, physical_device: &PhysicalDevice) -> vk::Format {
        App::find_supported_format(
            instance,
            physical_device,
            &[
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    fn find_supported_format(
        instance: &ash::Instance,
        physical_device: &PhysicalDevice,
        candidate_formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties = unsafe {
                instance.get_physical_device_format_properties(physical_device.handle(), format)
            };
            if tiling == vk::ImageTiling::LINEAR
                && format_properties.linear_tiling_features.contains(features)
            {
                return format.clone();
            } else if tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features)
            {
                return format.clone();
            }
        }

        panic!("Failed to find supported format!")
    }

    #[allow(dead_code)]
    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &Vec<vk::ImageView>,
        depth_image_view: vk::ImageView,
        swapchain_extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = vec![];

        for &image_view in image_views.iter() {
            let attachments = [image_view, depth_image_view];

            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .expect("Failed to create Framebuffer!")
            };

            framebuffers.push(framebuffer);
        }

        framebuffers
    }
}

// Fix content -------------------------------------------------------------------------------
impl App {
    fn create_command_buffers(
        submesh_count: u32,
        instance_count: u32,
        device: &Device,
        command_pool: &Arc<CommandPool>,
        graphics_pipeline: vk::Pipeline,
        framebuffers: &Vec<vk::Framebuffer>,
        render_pass: vk::RenderPass,
        surface_extent: vk::Extent2D,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &Vec<DescriptorSet>,
        indirect_task_buffer: &TypedBuffer<vk::DrawMeshTasksIndirectCommandEXT>,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffers = command_pool.allocate_group(framebuffers.len() as u32);

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);

            unsafe {
                device
                    .handle
                    .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                    .expect("Failed to begin recording Command Buffer at beginning!");
            }

            let clear_values = [
                vk::ClearValue {
                    // clear value for color buffer
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    // clear value for depth buffer
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_extent,
                })
                .clear_values(&clear_values);

            unsafe {
                device.handle.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.handle.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                //let vertex_buffers = [vertex_buffer];
                //let offsets = [0_u64];
                let descriptor_sets_to_bind = [descriptor_sets[i].handle()];

                //device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                //device.cmd_bind_index_buffer(
                //    command_buffer,
                //    index_buffer,
                //    0,
                //    vk::IndexType::UINT32,
                //);
                device.handle.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                device.fn_mesh_shader.cmd_draw_mesh_tasks_indirect(
                    command_buffer,
                    indirect_task_buffer.buffer(),
                    0,
                    instance_count,
                    indirect_task_buffer.stride() as _,
                );
                // device.cmd_draw_indexed(
                //     command_buffer,
                //     RECT_TEX_COORD_INDICES_DATA.len() as u32,
                //     1,
                //     0,
                //     0,
                //     0,
                // );

                device.handle.cmd_end_render_pass(command_buffer);

                device
                    .handle
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }

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
            self.device
                .handle
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.device.fn_swapchain.acquire_next_image(
                self.swapchain.handle,
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

        let cmd = self.ui_command_buffers[image_index as usize];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device
                .handle
                .begin_command_buffer(cmd, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        // //FIXME: this can be offloaded to a different thread
        self.integration.begin_frame(&self.window);

        self.visualizer.render_breakdown_window(
            &self.integration.context(),
            &self.allocator.lock().unwrap(),
            &mut self.visualizer_open,
        );

        let output = self.integration.end_frame(&self.window);

        let clipped_meshes = self.integration.context().tessellate(output.shapes);
        self.integration.paint(
            cmd,
            image_index as usize,
            clipped_meshes,
            output.textures_delta,
        );

        unsafe {
            self.device
                .handle
                .end_command_buffer(cmd)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&[self.mesh_command_buffers[image_index as usize], cmd])
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .wait_semaphores(&wait_semaphores)
            .build()];

        unsafe {
            self.device
                .handle
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .handle
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.sync_objects.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain.handle];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(&swapchains)
            .wait_semaphores(&signal_semaphores)
            .image_indices(&image_indices);

        let result = unsafe {
            self.device
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

    fn recreate_swapchain(&mut self) {
        self.device.wait_device_idle();

        self.cleanup_swapchain();

        self.swapchain = Swapchain::new(
            self.device.clone(),
            &self.physical_device,
            &self.window,
            self.surface.clone(),
            &self.queue_family,
            Some(&self.swapchain),
        );

        self.swapchain_imageviews = Image::create_image_views(
            &self.device,
            self.swapchain.surface_format.format,
            &self.swapchain.images,
        );
        self.render_pass = create_render_pass(
            &self.instance.handle,
            &self.device.handle,
            &self.physical_device,
            self.swapchain.surface_format.format,
        );
        let graphics_pipeline = create_graphics_pipeline(
            self.device.clone(),
            self.render_pass,
            self.swapchain.extent,
            self.ubo_layout,
        );
        self.graphics_pipeline = graphics_pipeline;

        self.depth_image = App::create_depth_resources(
            &self.instance.handle,
            self.device.clone(),
            &self.physical_device,
            self.command_pool.handle,
            self.graphics_queue,
            self.swapchain.extent,
            &self.memory_properties,
        );

        self.swapchain_framebuffers = App::create_framebuffers(
            &self.device.handle,
            self.render_pass,
            &self.swapchain_imageviews,
            self.depth_image.image_view(),
            self.swapchain.extent,
        );
        self.mesh_command_buffers = App::create_command_buffers(
            self.submesh_count,
            self.uniform_transforms.len() as _,
            &self.device,
            &self.command_pool,
            self.graphics_pipeline.pipeline(),
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain.extent,
            self.graphics_pipeline.layout(),
            &self.descriptor_sets,
            &self.indirect_task_buffer,
        );

        // Egui Integration
        self.integration.update_swapchain(
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.swapchain.handle,
            self.swapchain.surface_format,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device
                .handle
                .free_command_buffers(self.command_pool.handle, &self.mesh_command_buffers);

            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.handle.destroy_framebuffer(framebuffer, None);
            }

            self.device
                .handle
                .destroy_render_pass(self.render_pass, None);

            for &image_view in self.swapchain_imageviews.iter() {
                self.device.handle.destroy_image_view(image_view, None);
            }
        }
    }

    fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn window_ref(&self) -> &winit::window::Window {
        &self.window
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.cleanup_swapchain();

            self.device
                .handle
                .destroy_descriptor_set_layout(self.ubo_layout, None);

            self.integration.destroy();

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}

fn main() {
    let program_proc = ProgramProc::new();
    let vulkan_app = App::new(&program_proc.event_loop);

    program_proc.main_loop(vulkan_app);
}
// -------------------------------------------------------------------------------------------
