use std::sync::Arc;

use crate::components::camera_uniform::CameraUniform;
use crate::components::mesh::Mesh;
use crate::components::{camera::Camera, camera_controller::CameraController};
use crate::vertex::Vertex;
use bevy_ecs::event::EventReader;
use bevy_ecs::system::{Query, ResMut, Resource};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use super::app::ScreenEvent;
use super::{Instance, Texture};
#[derive(Resource)]
pub struct Renderer {
    instance: Arc<Instance>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: Texture,
}

impl Renderer {
    /// Creating some of the wgpu types requires async code
    /// https://sotrh.github.io/learn-wgpu/beginner/tutorial2-surface/#state-new
    pub async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::SHADER_PRIMITIVE_INDEX,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let partition_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &partition_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main", // 1.
                buffers: &[<[f32; 3]>::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "Depth Texture");

        Self {
            window,
            instance: Arc::new(Instance::new(
                surface,
                device,
                queue,
                camera_bind_group_layout,
                partition_bind_group_layout,
            )),
            config,
            size,
            depth_texture,
            render_pipeline,
            camera_buffer,
            camera_bind_group,
        }
    }
    pub fn on_resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.depth_texture =
                Texture::create_depth_texture(self.device(), &self.config, "Resized Depth Texture");
            self.instance
                .surface()
                .configure(&self.instance.device(), &self.config);
        }
    }
    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
    pub fn surface(&self) -> &wgpu::Surface {
        &self.instance.surface()
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.instance.device()
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.instance.queue()
    }
    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }
    pub fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.render_pipeline
    }
    pub fn camera_bind_group(&self) -> &wgpu::BindGroup {
        &self.camera_bind_group
    }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }
}

pub fn render(mut renderer: ResMut<Renderer>, meshes: Query<&Mesh>, camera: Query<&CameraUniform>) {
    let output = renderer.instance.surface().get_current_texture().unwrap();
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        renderer
            .instance
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

    for camera in camera.iter() {
        renderer.instance.queue().write_buffer(
            &renderer.camera_buffer,
            0,
            bytemuck::cast_slice(&[*camera]),
        );
    }
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mesh Render Pass"),
            color_attachments: &[
                // This is what @location(0) in the fragment shader targets
                Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &renderer.depth_texture.view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });

        for mesh in meshes.iter() {
            mesh.render_pass(&renderer, &mut render_pass);
        }
    }

    // submit will accept anything that implements IntoIter
    renderer
        .instance
        .queue()
        .submit(std::iter::once(encoder.finish()));
    output.present();
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
