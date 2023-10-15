use std::sync::Arc;

use crate::components::camera_uniform::CameraUniform;
use crate::components::mesh::Mesh;
use crate::components::{camera::Camera, camera_controller::CameraController};
use crate::vertex::{MyVertex, Vertex};
use bevy_ecs::event::EventReader;
use bevy_ecs::system::{Query, ResMut, Resource};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::InstanceCreateInfo;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, VulkanLibrary};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalSize;
use winit::window::{Window, WindowBuilder};

use super::app::ScreenEvent;
use super::buffer::BindGroupLayout;
use super::{BufferGroup, Instance, Texture};
#[derive(Resource)]
pub struct Renderer {
    window: Arc<Window>,
    instance: Arc<Instance>,
    //config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    //camera_buffer: BufferGroup<1>,
    render_pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>, // size = 24 (0x18), align = 0x8

    command_buffer_allocator: StandardCommandBufferAllocator,
    //depth_texture: Texture,
}

fn select_physical_device(
    instance: &Arc<vulkano::instance::Instance>,
    surface: &Surface,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // Find the first first queue family that is suitable.
                // If none is found, `None` is returned to `filter_map`,
                // which disqualifies this physical device.
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,

            // Note that there exists `PhysicalDeviceType::Other`, however,
            // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
            // match wildcard `_` to catch all unknown device types.
            _ => 4,
        })
        .expect("no device available")
}

impl Renderer {
    /// Creating some of the wgpu types requires async code
    /// https://sotrh.github.io/learn-wgpu/beginner/tutorial2-surface/#state-new
    pub async fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = vulkano_win::required_extensions(&library);

        let instance = vulkano::instance::Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let size = window.inner_size();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions);

        for family in physical_device.queue_family_properties() {
            println!(
                "Found a queue family with {:?} queue(s)",
                family.queue_count
            );
        }

        let (device, mut queues) = vulkano::device::Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![vulkano::device::QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device.clone());

        let surface_caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = window.inner_size();
        let composite_alpha = surface_caps
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let (mut swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count + 1, // How many buffers to use in the swapchain
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        // let surface_caps = surface.get_capabilities(&adapter);
        // // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // // one will result all the colors coming out darker. If you want to support non
        // // sRGB surfaces, you'll need to account for that when drawing to the frame.
        // let surface_format = surface_caps
        //     .formats
        //     .iter()
        //     .copied()
        //     .find(|f| f.is_srgb())
        //     .unwrap_or(surface_caps.formats[0]);
        // let config = wgpu::SurfaceConfiguration {
        //     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        //     format: surface_format,
        //     width: size.width,
        //     height: size.height,
        //     present_mode: surface_caps.present_modes[0],
        //     alpha_mode: surface_caps.alpha_modes[0],
        //     view_formats: vec![],
        // };
        // surface.configure(&device, &config);

        // let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        // let camera_bind_group_layout = BindGroupLayout::create(
        //     &device,
        //     &[wgpu::BindGroupLayoutEntry {
        //         binding: 0,
        //         visibility: wgpu::ShaderStages::VERTEX,
        //         ty: wgpu::BindingType::Buffer {
        //             ty: wgpu::BufferBindingType::Uniform,
        //             has_dynamic_offset: false,
        //             min_binding_size: None,
        //         },
        //         count: None,
        //     }],
        //     Some("camera_bind_group_layout"),
        // );

        // let camera_buffer = BufferGroup::create_single(
        //     &[CameraUniform::new()],
        //     wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     &device,
        //     &camera_bind_group_layout,
        //     Some("Camera Buffer"),
        // );

        // let partition_bind_group_layout = BindGroupLayout::create(
        //     &device,
        //     &[
        //         wgpu::BindGroupLayoutEntry {
        //             binding: 0,
        //             visibility: wgpu::ShaderStages::FRAGMENT,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Storage { read_only: true },
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         },
        //         wgpu::BindGroupLayoutEntry {
        //             binding: 1,
        //             visibility: wgpu::ShaderStages::FRAGMENT,
        //             ty: wgpu::BindingType::Buffer {
        //                 ty: wgpu::BufferBindingType::Storage { read_only: true },
        //                 has_dynamic_offset: false,
        //                 min_binding_size: None,
        //             },
        //             count: None,
        //         },
        //     ],
        //     Some("partition_bind_group_layout"),
        // );

        // let render_pipeline_layout =
        //     device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Render Pipeline Layout"),
        //         bind_group_layouts: &[
        //             (&camera_bind_group_layout).into(),
        //             (&partition_bind_group_layout).into(),
        //         ],
        //         push_constant_ranges: &[],
        //     });

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();

        // let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        //     label: Some("Render Pipeline"),
        //     layout: Some(&render_pipeline_layout),
        //     vertex: wgpu::VertexState {
        //         module: &shader,
        //         entry_point: "vs_main", // 1.
        //         buffers: &[<[f32; 3]>::desc()],
        //     },
        //     fragment: Some(wgpu::FragmentState {
        //         // 3.
        //         module: &shader,
        //         entry_point: "fs_main",
        //         targets: &[Some(wgpu::ColorTargetState {
        //             // 4.
        //             format: config.format,
        //             blend: Some(wgpu::BlendState::REPLACE),
        //             write_mask: wgpu::ColorWrites::ALL,
        //         })],
        //     }),
        //     primitive: wgpu::PrimitiveState {
        //         topology: wgpu::PrimitiveTopology::TriangleList,
        //         strip_index_format: None,
        //         front_face: wgpu::FrontFace::Ccw,
        //         cull_mode: Some(wgpu::Face::Back),
        //         // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
        //         polygon_mode: wgpu::PolygonMode::Fill,
        //         // Requires Features::DEPTH_CLIP_CONTROL
        //         unclipped_depth: false,
        //         // Requires Features::CONSERVATIVE_RASTERIZATION
        //         conservative: false,
        //     },
        //     depth_stencil: Some(wgpu::DepthStencilState {
        //         format: Texture::DEPTH_FORMAT,
        //         depth_write_enabled: true,
        //         depth_compare: wgpu::CompareFunction::Less,
        //         stencil: wgpu::StencilState::default(),
        //         bias: wgpu::DepthBiasState::default(),
        //     }),
        //     multisample: wgpu::MultisampleState {
        //         count: 1,
        //         mask: !0,
        //         alpha_to_coverage_enabled: false,
        //     },
        //     multiview: None,
        // });

        // More on this latter
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        };

        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let render_pipeline = GraphicsPipeline::start()
            // Describes the layout of the vertex input and how should it behave
            .vertex_input_state(
                <MyVertex as vulkano::pipeline::graphics::vertex_input::Vertex>::per_vertex(),
            )
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // Indicate the type of the primitives (the default is a list of triangles)
            .input_assembly_state(InputAssemblyState::new())
            // Set the fixed viewport
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
            // Same as the vertex input, but this for the fragment input
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            // This graphics pipeline object concerns the first pass of the render pass.
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            // Now that everything is specified, we call `build`.
            .build(device.clone())
            .unwrap();

        //let depth_texture = Texture::create_depth_texture(&device, &config, "Depth Texture");
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );
        Self {
            window,
            instance: Arc::new(Instance::new(
                surface,
                device,
                queue,
                memory_allocator,
                // camera_bind_group_layout,
                // partition_bind_group_layout,
            )),
            //config,
            size,
            //    depth_texture,
            command_buffer_allocator,
            render_pipeline,
            render_pass,
            swapchain,
            images, //camera_buffer,
        }
    }
    pub fn on_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            //self.config.width = new_size.width;
            //self.config.height = new_size.height;
            //self.depth_texture =
            //    Texture::create_depth_texture(self.device(), &self.config, "Resized Depth Texture");
            //self.instance
            //    .surface()
            //    .configure(&self.instance.device(), &self.config);
        }
    }
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
    pub fn surface(&self) -> &Surface {
        self.instance.surface()
    }
    pub fn device(&self) -> Arc<Device> {
        self.instance.device()
    }
    pub fn queue(&self) -> Arc<Queue> {
        self.instance.queue()
    }
    // pub fn config(&self) -> &wgpu::SurfaceConfiguration {
    //     &self.config
    // }
    pub fn render_pipeline(&self) -> &GraphicsPipeline {
        &self.render_pipeline
    }
    // pub fn camera_buffer(&self) -> &BufferGroup<1> {
    //     &self.camera_buffer
    // }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

pub fn render(mut renderer: ResMut<Renderer>, meshes: Query<&Mesh>, camera: Query<&CameraUniform>) {
    // let output = renderer.instance.surface().get_current_texture().unwrap();
    // let view = output
    //     .texture
    //     .create_view(&wgpu::TextureViewDescriptor::default());

    // let mut encoder =
    //     renderer
    //         .instance
    //         .device()
    //         .create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //             label: Some("Render Encoder"),
    //         });

    // for camera in camera.iter() {
    //     renderer.instance.queue().write_buffer(
    //         renderer.camera_buffer.buffer(),
    //         0,
    //         bytemuck::cast_slice(&[*camera]),
    //     );
    // }

    let (image_i, suboptimal, acquire_future) =
        match vulkano::swapchain::acquire_next_image(renderer.swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                //recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("failed to acquire next image: {e}"),
        };

    let view = ImageView::new_default(renderer.images[image_i as usize].clone()).unwrap();
    let framebuffer = Framebuffer::new(
        renderer.render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        &renderer.command_buffer_allocator,
        renderer.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(renderer.render_pipeline.clone());

    // let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    //     label: Some("Mesh Render Pass"),
    //     color_attachments: &[
    //         // This is what @location(0) in the fragment shader targets
    //         Some(wgpu::RenderPassColorAttachment {
    //             view: &view,
    //             resolve_target: None,
    //             ops: wgpu::Operations {
    //                 load: wgpu::LoadOp::Clear(wgpu::Color {
    //                     r: 0.1,
    //                     g: 0.2,
    //                     b: 0.3,
    //                     a: 1.0,
    //                 }),
    //                 store: true,
    //             },
    //         }),
    //     ],
    //     depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
    //         view: &renderer.depth_texture.view(),
    //         depth_ops: Some(wgpu::Operations {
    //             load: wgpu::LoadOp::Clear(1.0),
    //             store: true,
    //         }),
    //         stencil_ops: None,
    //     }),
    // });

    for mesh in meshes.iter() {
        mesh.render_pass(&renderer, &mut builder);
    }

    // submit will accept anything that implements IntoIter
    // renderer
    //     .instance
    //     .queue()
    //     .submit(std::iter::once(encoder.finish()));
    // output.present();

    builder.end_render_pass().unwrap();

    let execution = sync::now(renderer.device())
        .join(acquire_future)
        .then_execute(renderer.queue(), builder.build().unwrap())
        .unwrap()
        .then_swapchain_present(
            renderer.queue(),
            SwapchainPresentInfo::swapchain_image_index(renderer.swapchain.clone(), image_i),
        )
        .then_signal_fence_and_flush();
}

pub fn handle_screen_events(
    mut events: EventReader<ScreenEvent>,
    mut renderer: ResMut<Renderer>,
    mut cameras: Query<&mut Camera>,
) {
    for event in events.iter() {
        match event {
            ScreenEvent::Resize(new_size) => {
                renderer.on_resize(*new_size);
                for mut camera in cameras.iter_mut() {
                    camera.on_resize(new_size);
                }
            }
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
