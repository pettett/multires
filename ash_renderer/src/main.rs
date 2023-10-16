mod utility;

use crate::utility::{
    buffer::{create_storage_buffer, create_uniform_buffers},
    // the mod define some fixed functions that have been learned before.
    constants::*,
    debug::*,
    image::Image,
    pipeline::create_graphics_pipeline,
    render_pass::create_render_pass,
    share,
    structures::*,
    window::{ProgramProc, VulkanApp},
};

use ash::{extensions::ext::MeshShader, vk};
use bevy_ecs::{entity::Entity, event::Events, schedule::Schedule, world::World};
use common::{asset::Asset, MultiResMesh};
use common_renderer::components::{
    camera::Camera,
    camera_controller::{
        camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
    },
    transform::Transform,
};
use glam::{Mat4, Quat, Vec3, Vec3A};
use memoffset::offset_of;
use shaderc::CompilationArtifact;
use utility::pipeline::Pipeline;
use winit::event::WindowEvent;

use std::ffi::CString;
use std::path::Path;
use std::ptr;

// Constants
const WINDOW_TITLE: &'static str = "26.Depth Buffering";
const TEXTURE_PATH: &'static str = "../../../assets/vulkan.jpg";

pub struct VulkanApp26 {
    window: winit::window::Window,
    world: bevy_ecs::world::World,
    schedule: bevy_ecs::schedule::Schedule,
    camera: Entity,
    // vulkan stuff
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_merssager: vk::DebugUtilsMessengerEXT,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,

    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    graphics_pipeline: Pipeline,

    depth_image: Image,
    texture_image: Image,

    vertex_buffer: utility::buffer::Buffer,
    index_buffer: utility::buffer::Buffer,

    uniform_transform: UniformBufferObject,
    uniform_buffers: Vec<utility::buffer::Buffer>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,

    ms: MeshShader,
}

impl VulkanApp26 {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
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

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> VulkanApp26 {
        println!("initing window");
        let window =
            utility::window::init_window(&event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        // init vulkan stuff
        println!("initing vulkan");
        let entry = ash::Entry::linked();
        let instance = share::create_instance(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers.to_vec(),
        );

        println!("initing surface");
        let surface_stuff =
            share::create_surface(&entry, &instance, &window, WINDOW_WIDTH, WINDOW_HEIGHT);
        let (debug_utils_loader, debug_merssager) =
            setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let physical_device =
            share::pick_physical_device(&instance, &surface_stuff, &DEVICE_EXTENSIONS);

        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = share::create_logical_device(
            &instance,
            physical_device,
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface_stuff,
        );
        let ms = MeshShader::new(&instance, &device);

        println!("Loading queues");
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };

        println!("Loading swapchain");
        let swapchain_stuff = share::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            &surface_stuff,
            &queue_family,
        );
        let swapchain_imageviews = Image::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let render_pass = create_render_pass(
            &instance,
            &device,
            physical_device,
            swapchain_stuff.swapchain_format,
        );
        let ubo_layout = share::v2::create_descriptor_set_layout(&device);
        let graphics_pipeline = create_graphics_pipeline(
            device.clone(),
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
        );
        let command_pool = share::v1::create_command_pool(&device, &queue_family);

        let depth_image = VulkanApp26::create_depth_resources(
            &instance,
            device.clone(),
            physical_device,
            command_pool,
            graphics_queue,
            swapchain_stuff.swapchain_extent,
            &physical_device_memory_properties,
        );
        let swapchain_framebuffers = VulkanApp26::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            depth_image.image_view(),
            swapchain_stuff.swapchain_extent,
        );

        println!("Loading texture");
        let texture_image = Image::create_texture_image(
            device.clone(),
            command_pool,
            graphics_queue,
            &physical_device_memory_properties,
            &Path::new(TEXTURE_PATH),
        )
        .create_texture_sampler()
        .create_texture_image_view(1);

        println!("Loading verts");

        let data = MultiResMesh::load().unwrap();

        println!("V: {:?} I: {:?}", data.verts, data.indices);

        let vertex_buffer = create_storage_buffer(
            &device,
            &physical_device_memory_properties,
            command_pool,
            graphics_queue,
            &data.verts,
        );

        let index_buffer = create_storage_buffer(
            &device,
            &physical_device_memory_properties,
            command_pool,
            graphics_queue,
            &data.indices,
        );
        let uniform_buffers = create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_stuff.swapchain_images.len(),
        );

        println!("Loading descriptors");
        let descriptor_pool =
            share::v2::create_descriptor_pool(&device, swapchain_stuff.swapchain_images.len());
        let descriptor_sets = share::v2::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &uniform_buffers,
            &vertex_buffer,
            &index_buffer,
            &texture_image,
            swapchain_stuff.swapchain_images.len(),
        );

        println!("Loading command buffers");

        let command_buffers = VulkanApp26::create_command_buffers(
            &ms,
            &device,
            command_pool,
            graphics_pipeline.pipeline(),
            &swapchain_framebuffers,
            render_pass,
            swapchain_stuff.swapchain_extent,
            &vertex_buffer,
            &index_buffer,
            graphics_pipeline.layout(),
            &descriptor_sets,
        );
        let sync_ojbects = share::v1::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT);

        println!("Generated App");

        let cam = Camera::new(1.0);
        let transform = Transform::new(Vec3A::new(0.0, -0.2, 0.0), Quat::IDENTITY);

        let mut world = World::new();

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());

        let camera = world
            .spawn((
                CameraController::new(0.005),
                Camera::new(1.0),
                Transform::new(Vec3A::ZERO, Quat::IDENTITY),
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems((camera_handle_input, update_camera));

        // cleanup(); the 'drop' function will take care of it.
        VulkanApp26 {
            // winit stuff
            window,
            world,
            schedule,
            camera,

            // vulkan stuff
            instance,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            debug_utils_loader,
            debug_merssager,

            physical_device,
            memory_properties: physical_device_memory_properties,
            device,

            queue_family,
            graphics_queue,
            present_queue,

            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            swapchain_framebuffers,

            ubo_layout,
            render_pass,
            graphics_pipeline,

            depth_image,

            texture_image,

            vertex_buffer,
            index_buffer,

            uniform_transform: UniformBufferObject {
                model: Mat4::from_rotation_z(1.5),
                view_proj: cam.build_view_projection_matrix(&transform),
            },
            uniform_buffers,

            descriptor_pool,
            descriptor_sets,

            command_pool,
            command_buffers,

            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,

            ms,
        }
    }

    fn create_depth_resources(
        instance: &ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        _command_pool: vk::CommandPool,
        _submit_queue: vk::Queue,
        swapchain_extent: vk::Extent2D,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Image {
        let depth_format = VulkanApp26::find_depth_format(instance, physical_device);
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

    fn find_depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> vk::Format {
        VulkanApp26::find_supported_format(
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
        physical_device: vk::PhysicalDevice,
        candidate_formats: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        for &format in candidate_formats.iter() {
            let format_properties =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
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

            let framebuffer_create_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
            };

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
impl VulkanApp26 {
    fn create_command_buffers(
        ms: &MeshShader,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        graphics_pipeline: vk::Pipeline,
        framebuffers: &Vec<vk::Framebuffer>,
        render_pass: vk::RenderPass,
        surface_extent: vk::Extent2D,
        vertex_buffer: &utility::buffer::Buffer,
        index_buffer: &utility::buffer::Buffer,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &Vec<vk::DescriptorSet>,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: framebuffers.len() as u32,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        };

        for (i, &command_buffer) in command_buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                p_next: ptr::null(),
                p_inheritance_info: ptr::null(),
                flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
            };

            unsafe {
                device
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

            let render_pass_begin_info = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                p_next: ptr::null(),
                render_pass,
                framebuffer: framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };

            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );

                let vertex_buffers = [vertex_buffer];
                let offsets = [0_u64];
                let descriptor_sets_to_bind = [descriptor_sets[i]];

                //device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                //device.cmd_bind_index_buffer(
                //    command_buffer,
                //    index_buffer,
                //    0,
                //    vk::IndexType::UINT32,
                //);
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets_to_bind,
                    &[],
                );

                ms.cmd_draw_mesh_tasks(command_buffer, 1, 1, 1);
                // device.cmd_draw_indexed(
                //     command_buffer,
                //     RECT_TEX_COORD_INDICES_DATA.len() as u32,
                //     1,
                //     0,
                //     0,
                //     0,
                // );

                device.cmd_end_render_pass(command_buffer);

                device
                    .end_command_buffer(command_buffer)
                    .expect("Failed to record Command Buffer at Ending!");
            }
        }

        command_buffers
    }

    fn update_uniform_buffer(&mut self, current_image: usize, _delta_time: f32) {
        let cam = self.world.entity(self.camera);

        self.uniform_transform.view_proj = cam
            .get::<Camera>()
            .unwrap()
            .build_view_projection_matrix(cam.get().unwrap());

        let ubos = [self.uniform_transform.clone()];

        let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

        unsafe {
            let data_ptr =
                self.device
                    .map_memory(
                        self.uniform_buffers[current_image].memory(),
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to Map Memory") as *mut UniformBufferObject;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.device
                .unmap_memory(self.uniform_buffers[current_image].memory());
        }
    }
}

impl Drop for VulkanApp26 {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.cleanup_swapchain();

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_merssager, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl VulkanApp for VulkanApp26 {
    fn draw_frame(&mut self, delta_time: f32) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
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

        self.update_uniform_buffer(image_index as usize, delta_time);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        let result = unsafe {
            self.swapchain_loader
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
        // parameters -------------
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
            screen_width: WINDOW_WIDTH,
            screen_height: WINDOW_HEIGHT,
        };
        // ------------------------

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        let swapchain_stuff = share::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.window,
            &surface_suff,
            &self.queue_family,
        );
        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.swapchain_images;
        self.swapchain_format = swapchain_stuff.swapchain_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        self.swapchain_imageviews =
            Image::create_image_views(&self.device, self.swapchain_format, &self.swapchain_images);
        self.render_pass = create_render_pass(
            &self.instance,
            &self.device,
            self.physical_device,
            self.swapchain_format,
        );
        let graphics_pipeline = create_graphics_pipeline(
            self.device.clone(),
            self.render_pass,
            swapchain_stuff.swapchain_extent,
            self.ubo_layout,
        );
        self.graphics_pipeline = graphics_pipeline;

        self.depth_image = VulkanApp26::create_depth_resources(
            &self.instance,
            self.device.clone(),
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            self.swapchain_extent,
            &self.memory_properties,
        );

        self.swapchain_framebuffers = VulkanApp26::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.depth_image.image_view(),
            self.swapchain_extent,
        );
        self.command_buffers = VulkanApp26::create_command_buffers(
            &self.ms,
            &self.device,
            self.command_pool,
            self.graphics_pipeline.pipeline(),
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_extent,
            &self.vertex_buffer,
            &self.index_buffer,
            self.graphics_pipeline.layout(),
            &self.descriptor_sets,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            self.device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }

    fn wait_device_idle(&self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn window_ref(&self) -> &winit::window::Window {
        &self.window
    }
}

fn main() {
    let program_proc = ProgramProc::new();
    let vulkan_app = VulkanApp26::new(&program_proc.event_loop);

    program_proc.main_loop(vulkan_app);
}
// -------------------------------------------------------------------------------------------
