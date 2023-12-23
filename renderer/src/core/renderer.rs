use std::sync::Arc;

use crate::components::camera_uniform::CameraUniform;
use crate::components::debug_mesh::DebugMesh;
use crate::components::gpu_multi_res_mesh::{
    ClusterComponent, MultiResMeshComponent, MultiResMeshRenderer,
};
use crate::gui::gui::Gui;
use crate::vertex::Vertex;
use bevy_ecs::entity::Entity;
use bevy_ecs::event::EventReader;
use bevy_ecs::system::{Commands, NonSend, NonSendMut, Query, Res, ResMut, Resource};
use common::tri_mesh::TriMesh;
use common_renderer::components::camera::Camera;
use common_renderer::components::camera_controller::CameraController;
use common_renderer::components::transform::Transform;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::window::Window;

use super::app::ScreenEvent;
use super::buffer::BindGroupLayout;
use super::{BufferGroup, Instance, Texture};
#[derive(Resource)]
pub struct Renderer {
    instance: Arc<Instance>,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    camera_buffer: BufferGroup<1>,
    render_pipeline: wgpu::RenderPipeline,
    render_pipeline_wire: wgpu::RenderPipeline,
    pub compute_pipeline: wgpu::ComputePipeline,
    depth_texture: Texture,
    surface_format: wgpu::TextureFormat,
    pub sphere_gizmo: DebugMesh,
    pub mesh_index: usize,
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
                    features: wgpu::Features::SHADER_PRIMITIVE_INDEX
                        | wgpu::Features::POLYGON_MODE_LINE,
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: wgpu::Limits::default(),
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
        let shader_compute =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/should_draw.wgsl"));
        let shader_wire =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/shader_wire.wgsl"));

        let camera_bind_group_layout = BindGroupLayout::create(
            &device,
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("camera_bind_group_layout"),
        );

        let camera_buffer = BufferGroup::create_single(
            &[CameraUniform::new()],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &device,
            &camera_bind_group_layout,
            Some("Camera Buffer"),
        );

        let model_bind_group_layout = BindGroupLayout::create(
            &device,
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("model_bind_group_layout"),
        );

        let write_compute_bind_group_layout: BindGroupLayout<1> = BindGroupLayout::create(
            &device,
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("writeable_compute_buffer_bind_group"),
        );
        let read_compute_buffer_bind_group: BindGroupLayout<1> = BindGroupLayout::create(
            &device,
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            Some("readable_compute_buffer_bind_group"),
        );

        let partition_bind_group_layout = BindGroupLayout::create(
            &device,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            Some("partition_bind_group_layout"),
        );

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    (&camera_bind_group_layout).into(),
                    (&partition_bind_group_layout).into(),
                    (&model_bind_group_layout).into(),
                ],
                push_constant_ranges: &[],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    // Reminder: Max of 4 for these, don't add any more
                    (&write_compute_bind_group_layout).into(),
                    (&read_compute_buffer_bind_group).into(),
                    (&read_compute_buffer_bind_group).into(),
                    (&read_compute_buffer_bind_group).into(),
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = make_render_pipeline(
            &device,
            &render_pipeline_layout,
            &shader,
            config.format,
            wgpu::PolygonMode::Fill,
            Some(wgpu::Face::Back),
        );
        let render_pipeline_wire = make_render_pipeline(
            &device,
            &render_pipeline_layout,
            &shader_wire,
            config.format,
            wgpu::PolygonMode::Line,
            None,
        );

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Culling compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_compute,
            entry_point: "main",
        });

        let depth_texture = Texture::create_depth_texture(&device, &config, "Depth Texture");

        let sphere_gizmo = TriMesh::from_gltf("../assets/sphere_low.glb").unwrap();

        let instance = Arc::new(Instance::new(
            surface,
            device,
            queue,
            camera_bind_group_layout,
            model_bind_group_layout,
            partition_bind_group_layout,
            write_compute_bind_group_layout,
            read_compute_buffer_bind_group,
        ));

        Self {
            window,
            sphere_gizmo: DebugMesh::from_tris(instance.clone(), &sphere_gizmo),
            instance,
            config,
            size,
            compute_pipeline,
            surface_format,
            depth_texture,
            render_pipeline,
            render_pipeline_wire,
            camera_buffer,
            mesh_index: 0,
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
    pub fn render_pipeline_wire(&self) -> &wgpu::RenderPipeline {
        &self.render_pipeline_wire
    }
    pub fn camera_buffer(&self) -> &BufferGroup<1> {
        &self.camera_buffer
    }

    pub fn instance(&self) -> Arc<Instance> {
        self.instance.clone()
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }
}

fn make_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
    polygon_mode: wgpu::PolygonMode,
    cull_mode: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: "vs_main", // 1.
            buffers: &[<[f32; 4]>::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            // 3.
            module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                // 4.
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode,
            polygon_mode,
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
    })
}

pub fn render(
    renderer: Res<Renderer>,
    mut gui: ResMut<Gui>,
    ctx: NonSend<egui::Context>,
    mut state: NonSendMut<egui_winit::State>,
    mut meshes: Query<(&mut MultiResMeshComponent, &Transform)>,
    mesh_renderer: ResMut<MultiResMeshRenderer>,
    submeshes: Query<(Entity, &ClusterComponent)>,
    camera: Query<(&CameraUniform, &Transform)>,
    cameras: Query<(&mut Camera, &mut CameraController, &Transform)>,
    mut commands: Commands,
) {
    let output = renderer.instance.surface().get_current_texture().unwrap();
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut compute_encoder =
        renderer
            .instance
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

    let (camera, camera_trans) = camera.single();

    renderer.instance.queue().write_buffer(
        renderer.camera_buffer.buffer(),
        0,
        bytemuck::cast_slice(&[*camera]),
    );

    {
        let mut compute_pass = compute_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
        });

        for (mesh, trans) in meshes.iter() {
            mesh.compute_pass(
                trans,
                &renderer,
                &mesh_renderer,
                camera_trans,
                &submeshes,
                &mut compute_pass,
            );
        }
    }

    // {
    //     for mesh in meshes.iter() {
    //         // Sets adds copy operation to command encoder.
    //         // Will copy data from storage buffer on GPU to staging buffer on CPU.
    //         encoder.copy_buffer_to_buffer(
    //             &mesh.asset().cluster_data_real_error_buffer.buffer(),
    //             0,
    //             &mesh.debug_staging_buffer,
    //             0,
    //             mesh.staging_buffer_size as _,
    //         );
    //     }
    // }

    renderer
        .instance
        .queue()
        .submit(std::iter::once(compute_encoder.finish()));

    // {
    //     for mesh in meshes.iter() {
    //         // Note that we're not calling `.await` here.
    //         let buffer_slice = mesh.debug_staging_buffer.slice(..);
    //         // Gets the future representing when `staging_buffer` can be read from
    //         buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    //     }
    // }

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    //renderer.device().poll(wgpu::Maintain::Wait);

    let mut render_encoder =
        renderer
            .instance
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

    {
        let mut render_pass = render_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

        for (mesh, _trans) in meshes.iter() {
            mesh.render_pass(&renderer, &submeshes, &mut render_pass, &mesh_renderer);
        }
    }

    gui.render_pass(
        &renderer,
        &mut state,
        &ctx,
        &mut render_encoder,
        &view,
        &mut meshes,
        mesh_renderer,
        &submeshes,
        cameras,
        &mut commands,
    );

    // submit will accept anything that implements IntoIter
    renderer
        .instance
        .queue()
        .submit(std::iter::once(render_encoder.finish()));
    output.present();

    // {
    //     for mesh in meshes.iter() {
    //         mesh.debug_staging_buffer.unmap();
    //     }
    // }
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
            ScreenEvent::Key(k) => {
                if k.state == ElementState::Pressed {
                    match k.virtual_keycode {
                        Some(winit::event::VirtualKeyCode::F) => {
                            renderer.mesh_index += 1;
                        }
                        Some(winit::event::VirtualKeyCode::R) if renderer.mesh_index > 0 => {
                            renderer.mesh_index -= 1;
                        }
                        _ => (),
                    }
                }
            }
        }
    }
}
