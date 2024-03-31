#![warn(missing_docs)]

// use self::allocator::{AllocatorTrait, MemoryLocation};
// use crate::gui::allocator::AllocationCreateInfoTrait;
// use crate::gui::allocator::AllocationTrait;
use crate::{
    utility::pooled::descriptor_pool::{
        DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorWriteData,
    },
    VkHandle,
};
use ::gpu_allocator::vulkan::Allocator;
use ash::{extensions::khr::Swapchain, vk, Device};
use bytemuck::bytes_of;
use egui::{
    epaint::{ahash::AHashMap, ImageDelta},
    Context, TextureId, TexturesDelta,
};
use egui_winit::{winit::window::Window, EventResponse};
use gpu_allocator::{vulkan::*, MemoryLocation};
use raw_window_handle::HasRawDisplayHandle;
use std::{
    ffi::CString,
    sync::{Arc, Mutex},
};

use crate::{core::Core, utility::pooled::descriptor_pool::DescriptorPool};

use super::{utils::insert_image_memory_barrier, *};

/// egui integration with winit and ash.
pub struct Integration {
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
    context: Context,
    egui_winit: egui_winit::State,
    core: Arc<Core>,
    allocator: Arc<Mutex<Allocator>>,
    qfi: u32,
    queue: vk::Queue,
    swapchain_loader: Swapchain,
    descriptor_pool: Arc<DescriptorPool>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    framebuffer_color_image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_allocations: Vec<Allocation>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_allocations: Vec<Allocation>,
    texture_desc_sets: AHashMap<TextureId, DescriptorSet>,
    texture_images: AHashMap<TextureId, vk::Image>,
    texture_image_infos: AHashMap<TextureId, vk::ImageCreateInfo>,
    texture_allocations: AHashMap<TextureId, Allocation>,
    texture_image_views: AHashMap<TextureId, vk::ImageView>,

    user_textures: Vec<Option<vk::DescriptorSet>>,
}
impl Integration {
    /// Create an instance of the integration.
    pub fn new<H: HasRawDisplayHandle>(
        display_target: &H,
        physical_width: u32,
        physical_height: u32,
        scale_factor: f64,
        font_definitions: egui::FontDefinitions,
        style: egui::Style,
        core: Arc<Core>,
        allocator: Arc<Mutex<Allocator>>,
        qfi: u32,
        queue: vk::Queue,
        swapchain_loader: Swapchain,
        swapchain: vk::SwapchainKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) -> Self {
        // Create context
        let context = Context::default();
        context.set_fonts(font_definitions);
        context.set_style(style);

        let mut egui_winit = egui_winit::State::new(display_target);
        egui_winit.set_pixels_per_point(scale_factor as f32);

        // Get swap_images to get len of swapchain images and to create framebuffers
        let swap_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images.")
        };

        // Create DescriptorPool
        let descriptor_pool = DescriptorPool::new_sized(
            core.device.clone(),
            &[*vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1024)],
            1024,
        );

        // Create DescriptorSetLayouts
        // let descriptor_set_layouts = {
        //     let mut sets = vec![];
        //     for _ in 0..swap_images.len() {
        //         sets.push(
        //             unsafe {
        //                 core.device.create_descriptor_set_layout(
        //                     &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
        //                         vk::DescriptorSetLayoutBinding::builder()
        //                             .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        //                             .descriptor_count(1)
        //                             .binding(0)
        //                             .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        //                             .build(),
        //                     ]),
        //                     None,
        //                 )
        //             }
        //             .expect("Failed to create descriptor set layout."),
        //         );
        //     }
        //     sets
        // };

        let descriptor_set_layout = Arc::new(DescriptorSetLayout::new(
            &core,
            vec![DescriptorSetLayoutBinding::Sampler {
                vis: vk::ShaderStageFlags::FRAGMENT,
            }],
            "EGUI Texture layout",
        ));

        // Create RenderPass
        let render_pass = unsafe {
            core.device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[vk::AttachmentDescription::builder()
                        .format(surface_format.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .build()])
                    .subpasses(&[vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::builder()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .build()])
                        .build()])
                    .dependencies(&[vk::SubpassDependency::builder()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create render pass.");

        // Create PipelineLayout
        let pipeline_layout = unsafe {
            core.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layout.handle()])
                    .push_constant_ranges(&[vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(std::mem::size_of::<f32>() as u32 * 2) // screen size
                        .build()]),
                None,
            )
        }
        .expect("Failed to create pipeline layout.");

        // Create Pipeline
        let pipeline = {
            let bindings = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(
                    4 * std::mem::size_of::<f32>() as u32 + 4 * std::mem::size_of::<u8>() as u32,
                )
                .build()];

            let attributes = [
                // position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // uv
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // color
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .build(),
            ];

            let vertex_shader_module = {
                let bytes_code = include_bytes!("../../shaders/spv/egui_vert.vert");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    core.device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create vertex shader module.")
            };
            let fragment_shader_module = {
                let bytes_code = include_bytes!("../../shaders/spv/egui_frag.frag");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    core.device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create fragment shader module.")
            };
            let main_function_name = CString::new("main").unwrap();
            let pipeline_shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name)
                    .build(),
            ];

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(false)
                .depth_write_enable(false)
                .depth_compare_op(vk::CompareOp::ALWAYS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .build()];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&pipeline_shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_info)
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
                .dynamic_state(&dynamic_state_info)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0)
                .build()];

            let pipeline = unsafe {
                core.device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_info,
                    None,
                )
            }
            .expect("Failed to create graphics pipeline.")[0];
            unsafe {
                core.device
                    .destroy_shader_module(vertex_shader_module, None);
                core.device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            pipeline
        };

        // Create Sampler
        let sampler = unsafe {
            core.device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .anisotropy_enable(false)
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .min_lod(0.0)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }
        .expect("Failed to create sampler.");

        // Create Framebuffers
        let framebuffer_color_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                core.device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(swapchain_image.clone())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();
        let framebuffers = framebuffer_color_image_views
            .iter()
            .map(|&image_views| unsafe {
                let attachments = &[image_views];
                core.device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(render_pass)
                            .attachments(attachments)
                            .width(physical_width)
                            .height(physical_height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer.")
            })
            .collect::<Vec<_>>();

        // Create vertex buffer and index buffer
        let mut vertex_buffers = vec![];
        let mut vertex_buffer_allocations = vec![];
        let mut index_buffers = vec![];
        let mut index_buffer_allocations = vec![];
        for _ in 0..framebuffers.len() {
            let vertex_buffer = unsafe {
                core.device
                    .create_buffer(
                        &vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::vertex_buffer_size()),
                        None,
                    )
                    .expect("Failed to create vertex buffer.")
            };
            let vertex_buffer_requirements =
                unsafe { core.device.get_buffer_memory_requirements(vertex_buffer) };
            let vertex_buffer_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    requirements: vertex_buffer_requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    name: "EGUI Vert Buffer",
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to create vertex buffer.");
            unsafe {
                core.device
                    .bind_buffer_memory(
                        vertex_buffer,
                        vertex_buffer_allocation.memory(),
                        vertex_buffer_allocation.offset(),
                    )
                    .expect("Failed to create vertex buffer.")
            }

            let index_buffer = unsafe {
                core.device
                    .create_buffer(
                        &vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::INDEX_BUFFER)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .size(Self::index_buffer_size()),
                        None,
                    )
                    .expect("Failed to create index buffer.")
            };
            let index_buffer_requirements =
                unsafe { core.device.get_buffer_memory_requirements(index_buffer) };
            let index_buffer_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    requirements: index_buffer_requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    name: "EGUI Index Buffer",
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .expect("Failed to create index buffer.");
            unsafe {
                core.device
                    .bind_buffer_memory(
                        index_buffer,
                        index_buffer_allocation.memory(),
                        index_buffer_allocation.offset(),
                    )
                    .expect("Failed to create index buffer.")
            }

            vertex_buffers.push(vertex_buffer);
            vertex_buffer_allocations.push(vertex_buffer_allocation);
            index_buffers.push(index_buffer);
            index_buffer_allocations.push(index_buffer_allocation);
        }

        // User Textures

        let user_textures = vec![];

        Self {
            physical_width,
            physical_height,
            scale_factor,
            context,
            egui_winit,

            core,
            allocator,
            qfi,
            queue,
            swapchain_loader,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            render_pass,
            framebuffer_color_image_views,
            framebuffers,
            vertex_buffers,
            vertex_buffer_allocations,
            index_buffers,
            index_buffer_allocations,
            texture_desc_sets: AHashMap::new(),
            texture_images: AHashMap::new(),
            texture_image_infos: AHashMap::new(),
            texture_allocations: AHashMap::new(),
            texture_image_views: AHashMap::new(),

            user_textures,
        }
    }

    // vertex buffer size
    fn vertex_buffer_size() -> u64 {
        1024 * 1024 * 4
    }

    // index buffer size
    fn index_buffer_size() -> u64 {
        1024 * 1024 * 2
    }

    /// handling winit event.
    pub fn handle_event(
        &mut self,
        winit_event: &egui_winit::winit::event::WindowEvent,
    ) -> EventResponse {
        self.egui_winit.on_event(&self.context, winit_event)
    }

    /// begin frame.
    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.context.begin_frame(raw_input);
    }

    /// end frame.
    pub fn end_frame(&mut self, window: &Window) -> egui::FullOutput {
        let output = self.context.end_frame();

        self.egui_winit.handle_platform_output(
            window,
            &self.context,
            output.platform_output.clone(),
        );

        output
    }

    /// Get [`egui::Context`].
    pub fn context(&self) -> Context {
        self.context.clone()
    }

    /// Record paint commands.
    pub fn paint(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain_image_index: usize,
        clipped_meshes: Vec<egui::ClippedPrimitive>,
        textures_delta: TexturesDelta,
    ) {
        let index = swapchain_image_index;

        for (id, image_delta) in textures_delta.set {
            self.update_texture(id, image_delta);
        }

        let mut vertex_buffer_ptr = self.vertex_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8;
        // let mut vertex_buffer_ptr = unsafe {
        //     self.core.device
        //         .map_memory(
        //             self.vertex_buffer_allocations[index].memory(),
        //             self.vertex_buffer_allocations[index].offset(),
        //             self.vertex_buffer_allocations[index].size(),
        //             vk::MemoryMapFlags::empty(),
        //         )
        //         .expect("Failed to map buffers.") as *mut u8
        // };
        let vertex_buffer_ptr_end =
            unsafe { vertex_buffer_ptr.add(Self::vertex_buffer_size() as usize) };
        // let mut index_buffer_ptr = unsafe {
        //     self.core.device
        //         .map_memory(
        //             self.index_buffer_allocations[index].memory(),
        //             self.index_buffer_allocations[index].offset(),
        //             self.index_buffer_allocations[index].size(),
        //             vk::MemoryMapFlags::empty(),
        //         )
        //         .expect("Failed to map buffers.") as *mut u8
        // };
        let mut index_buffer_ptr = self.index_buffer_allocations[index]
            .mapped_ptr()
            .unwrap()
            .as_ptr() as *mut u8;
        let index_buffer_ptr_end =
            unsafe { index_buffer_ptr.add(Self::index_buffer_size() as usize) };

        // begin render pass
        unsafe {
            self.core.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[index])
                    .clear_values(&[])
                    .render_area(
                        *vk::Rect2D::builder().extent(
                            vk::Extent2D::builder()
                                .width(self.physical_width)
                                .height(self.physical_height)
                                .build(),
                        ),
                    ),
                vk::SubpassContents::INLINE,
            );
        }

        // bind resources
        unsafe {
            self.core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.core.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffers[index]],
                &[0],
            );
            self.core.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffers[index],
                0,
                vk::IndexType::UINT32,
            );
            self.core.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(self.physical_width as f32)
                    .height(self.physical_height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)
                    .build()],
            );
            let width_points = self.physical_width as f32 / self.scale_factor as f32;
            let height_points = self.physical_height as f32 / self.scale_factor as f32;
            self.core.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes_of(&width_points),
            );
            self.core.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                std::mem::size_of_val(&width_points) as u32,
                bytes_of(&height_points),
            );
        }

        // render meshes
        let mut vertex_base = 0;
        let mut index_base = 0;
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            let mesh = match primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh,
                egui::epaint::Primitive::Callback(_) => todo!(),
            };
            if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                continue;
            }

            unsafe {
                if let egui::TextureId::User(id) = mesh.texture_id {
                    if let Some(descriptor_set) = self.user_textures[id as usize] {
                        self.core.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                    } else {
                        eprintln!(
                            "This UserTexture has already been unregistered: {:?}",
                            mesh.texture_id
                        );
                        continue;
                    }
                } else {
                    self.core.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self
                            .texture_desc_sets
                            .get(&mesh.texture_id)
                            .unwrap()
                            .handle()],
                        &[],
                    );
                }
            }
            let v_slice = &mesh.vertices;
            let v_size = std::mem::size_of_val(&v_slice[0]);
            let v_copy_size = v_slice.len() * v_size;

            let i_slice = &mesh.indices;
            let i_size = std::mem::size_of_val(&i_slice[0]);
            let i_copy_size = i_slice.len() * i_size;

            let vertex_buffer_ptr_next = unsafe { vertex_buffer_ptr.add(v_copy_size) };
            let index_buffer_ptr_next = unsafe { index_buffer_ptr.add(i_copy_size) };

            if vertex_buffer_ptr_next >= vertex_buffer_ptr_end
                || index_buffer_ptr_next >= index_buffer_ptr_end
            {
                panic!("egui paint out of memory");
            }

            // map memory
            unsafe { vertex_buffer_ptr.copy_from(v_slice.as_ptr() as *const u8, v_copy_size) };
            unsafe { index_buffer_ptr.copy_from(i_slice.as_ptr() as *const u8, i_copy_size) };

            vertex_buffer_ptr = vertex_buffer_ptr_next;
            index_buffer_ptr = index_buffer_ptr_next;

            // record draw commands
            unsafe {
                let min = clip_rect.min;
                let min = egui::Pos2 {
                    x: min.x * self.scale_factor as f32,
                    y: min.y * self.scale_factor as f32,
                };
                let min = egui::Pos2 {
                    x: f32::clamp(min.x, 0.0, self.physical_width as f32),
                    y: f32::clamp(min.y, 0.0, self.physical_height as f32),
                };
                let max = clip_rect.max;
                let max = egui::Pos2 {
                    x: max.x * self.scale_factor as f32,
                    y: max.y * self.scale_factor as f32,
                };
                let max = egui::Pos2 {
                    x: f32::clamp(max.x, min.x, self.physical_width as f32),
                    y: f32::clamp(max.y, min.y, self.physical_height as f32),
                };
                self.core.device.cmd_set_scissor(
                    command_buffer,
                    0,
                    &[vk::Rect2D::builder()
                        .offset(
                            vk::Offset2D::builder()
                                .x(min.x.round() as i32)
                                .y(min.y.round() as i32)
                                .build(),
                        )
                        .extent(
                            vk::Extent2D::builder()
                                .width((max.x.round() - min.x) as u32)
                                .height((max.y.round() - min.y) as u32)
                                .build(),
                        )
                        .build()],
                );
                self.core.device.cmd_draw_indexed(
                    command_buffer,
                    mesh.indices.len() as u32,
                    1,
                    index_base,
                    vertex_base,
                    0,
                );
            }

            vertex_base += mesh.vertices.len() as i32;
            index_base += mesh.indices.len() as u32;
        }

        // end render pass
        unsafe {
            self.core.device.cmd_end_render_pass(command_buffer);
        }

        for &id in &textures_delta.free {
            self.texture_desc_sets.remove_entry(&id); // dsc_set is destroyed with dsc_pool
            self.texture_image_infos.remove_entry(&id);
            if let Some((_, image)) = self.texture_images.remove_entry(&id) {
                unsafe {
                    self.core.device.destroy_image(image, None);
                }
            }
            if let Some((_, image_view)) = self.texture_image_views.remove_entry(&id) {
                unsafe {
                    self.core.device.destroy_image_view(image_view, None);
                }
            }
            if let Some((_, allocation)) = self.texture_allocations.remove_entry(&id) {
                self.allocator.lock().unwrap().free(allocation).unwrap();
            }
        }
    }

    fn update_texture(&mut self, texture_id: TextureId, delta: ImageDelta) {
        // Extract pixel data from egui
        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image
                    .pixels
                    .iter()
                    .flat_map(|color| color.to_array())
                    .collect()
            }
            egui::ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };
        let cmd_pool = {
            let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(self.qfi)
                .build();
            unsafe {
                self.core
                    .device
                    .create_command_pool(&cmd_pool_info, None)
                    .unwrap()
            }
        };
        let cmd_buff = {
            let cmd_buff_alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1u32)
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .build();
            unsafe {
                self.core
                    .device
                    .allocate_command_buffers(&cmd_buff_alloc_info)
                    .unwrap()[0]
            }
        };
        let fence_info = vk::FenceCreateInfo::builder().build();
        let cmd_buff_fence = unsafe { self.core.device.create_fence(&fence_info, None).unwrap() };

        let (staging_buffer, staging_allocation) = {
            let buffer_size = data.len() as vk::DeviceSize;
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC);
            let texture_buffer =
                unsafe { self.core.device.create_buffer(&buffer_info, None) }.unwrap();
            let requirements = unsafe {
                self.core
                    .device
                    .get_buffer_memory_requirements(texture_buffer)
            };
            let allocation = self
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    requirements,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    name: "EGUI staging texture",
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap();
            unsafe {
                self.core
                    .device
                    .bind_buffer_memory(texture_buffer, allocation.memory(), allocation.offset())
                    .unwrap()
            };
            (texture_buffer, allocation)
        };
        let ptr = staging_allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
        unsafe {
            ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        let (texture_image, info, texture_allocation) = {
            let extent = vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            };
            let create_info = vk::ImageCreateInfo::builder()
                .array_layers(1)
                .extent(extent)
                .flags(vk::ImageCreateFlags::empty())
                .format(vk::Format::R8G8B8A8_UNORM)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .mip_levels(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .build();
            let handle = unsafe { self.core.device.create_image(&create_info, None) }.unwrap();
            let requirements = unsafe { self.core.device.get_image_memory_requirements(handle) };
            let allocation = self
                .allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    name: "EGUI texture",
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .unwrap();
            unsafe {
                self.core
                    .device
                    .bind_image_memory(handle, allocation.memory(), allocation.offset())
                    .unwrap()
            };
            (handle, create_info, allocation)
        };
        self.texture_image_infos.insert(texture_id, info);
        let texture_image_view = {
            let create_info = vk::ImageViewCreateInfo::builder()
                .components(vk::ComponentMapping::default())
                .flags(vk::ImageViewCreateFlags::empty())
                .format(vk::Format::R8G8B8A8_UNORM)
                .image(texture_image)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .base_mip_level(0)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                )
                .view_type(vk::ImageViewType::TYPE_2D)
                .build();
            unsafe {
                self.core
                    .device
                    .create_image_view(&create_info, None)
                    .unwrap()
            }
        };
        // begin cmd buff
        unsafe {
            let cmd_buff_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .build();
            self.core
                .device
                .begin_command_buffer(cmd_buff, &cmd_buff_begin_info)
                .unwrap();
        }
        // Transition texture image for transfer dst
        insert_image_memory_barrier(
            &self.core.device,
            &cmd_buff,
            &texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::NONE_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0u32)
                .layer_count(1u32)
                .base_mip_level(0u32)
                .level_count(1u32)
                .build(),
        );
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(delta.image.width() as u32)
            .buffer_image_height(delta.image.height() as u32)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .layer_count(1)
                    .mip_level(0)
                    .build(),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            })
            .build();
        unsafe {
            self.core.device.cmd_copy_buffer_to_image(
                cmd_buff,
                staging_buffer,
                texture_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        insert_image_memory_barrier(
            &self.core.device,
            &cmd_buff,
            &texture_image,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::VERTEX_SHADER,
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_array_layer(0u32)
                .layer_count(1u32)
                .base_mip_level(0u32)
                .level_count(1u32)
                .build(),
        );

        unsafe {
            self.core.device.end_command_buffer(cmd_buff).unwrap();
        }
        let cmd_buffs = [cmd_buff];
        let submit_infos = [vk::SubmitInfo::builder()
            .command_buffers(&cmd_buffs)
            .build()];
        unsafe {
            self.core
                .device
                .queue_submit(self.queue, &submit_infos, cmd_buff_fence)
                .unwrap();
            self.core
                .device
                .wait_for_fences(&[cmd_buff_fence], true, u64::MAX)
                .unwrap();
        }

        // texture is now in GPU memory, now we need to decide whether we should register it as new or update existing

        if let Some(pos) = delta.pos {
            // Blit texture data to existing texture if delta pos exists (e.g. font changed)
            let existing_texture = self.texture_images.get(&texture_id);
            if let Some(existing_texture) = existing_texture {
                let info = self.texture_image_infos.get(&texture_id).unwrap();
                unsafe {
                    self.core
                        .device
                        .reset_command_pool(cmd_pool, vk::CommandPoolResetFlags::empty())
                        .unwrap();
                    self.core.device.reset_fences(&[cmd_buff_fence]).unwrap();
                    // begin cmd buff
                    let cmd_buff_begin_info = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build();
                    self.core
                        .device
                        .begin_command_buffer(cmd_buff, &cmd_buff_begin_info)
                        .unwrap();

                    // Transition existing image for transfer dst
                    insert_image_memory_barrier(
                        &self.core.device,
                        &cmd_buff,
                        &existing_texture,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0u32)
                            .layer_count(1u32)
                            .base_mip_level(0u32)
                            .level_count(1u32)
                            .build(),
                    );
                    // Transition new image for transfer src
                    insert_image_memory_barrier(
                        &self.core.device,
                        &cmd_buff,
                        &texture_image,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::SHADER_READ,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0u32)
                            .layer_count(1u32)
                            .base_mip_level(0u32)
                            .level_count(1u32)
                            .build(),
                    );
                    let top_left = vk::Offset3D {
                        x: pos[0] as i32,
                        y: pos[1] as i32,
                        z: 0,
                    };
                    let bottom_right = vk::Offset3D {
                        x: pos[0] as i32 + delta.image.width() as i32,
                        y: pos[1] as i32 + delta.image.height() as i32,
                        z: 1,
                    };

                    let region = vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: info.extent.width as i32,
                                y: info.extent.height as i32,
                                z: info.extent.depth as i32,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offsets: [top_left, bottom_right],
                    };
                    self.core.device.cmd_blit_image(
                        cmd_buff,
                        texture_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        *existing_texture,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                        vk::Filter::NEAREST,
                    );

                    // Transition existing image for shader read
                    insert_image_memory_barrier(
                        &self.core.device,
                        &cmd_buff,
                        &existing_texture,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0u32)
                            .layer_count(1u32)
                            .base_mip_level(0u32)
                            .level_count(1u32)
                            .build(),
                    );
                    self.core.device.end_command_buffer(cmd_buff).unwrap();
                    let cmd_buffs = [cmd_buff];
                    let submit_infos = [vk::SubmitInfo::builder()
                        .command_buffers(&cmd_buffs)
                        .build()];
                    self.core
                        .device
                        .queue_submit(self.queue, &submit_infos, cmd_buff_fence)
                        .unwrap();
                    self.core
                        .device
                        .wait_for_fences(&[cmd_buff_fence], true, u64::MAX)
                        .unwrap();

                    // destroy texture_image and view
                    self.core.device.destroy_image(texture_image, None);
                    self.core
                        .device
                        .destroy_image_view(texture_image_view, None);
                    self.allocator
                        .lock()
                        .unwrap()
                        .free(texture_allocation)
                        .unwrap();
                }
            } else {
                return;
            }
        } else {
            // Otherwise save the newly created texture

            // update dsc set
            let dsc_set = self
                .descriptor_pool
                .alloc(&self.descriptor_set_layout, 1, |_| {
                    vec![DescriptorWriteData::Image {
                        view: texture_image_view,
                        sampler: self.sampler,
                    }]
                })
                .into_iter()
                .next()
                .expect("There must be exactly one result from this                    Some(buf.write_descriptor_sets(
					handle,
					layout.binding_type(),
					descriptor_info[i].as_ref().unwrap(),
					i as _,
				))");

            // register new texture
            self.texture_images.insert(texture_id, texture_image);
            self.texture_allocations
                .insert(texture_id, texture_allocation);
            self.texture_image_views
                .insert(texture_id, texture_image_view);
            self.texture_desc_sets.insert(texture_id, dsc_set);
        }
        // cleanup
        unsafe {
            self.core.device.destroy_buffer(staging_buffer, None);
            self.allocator
                .lock()
                .unwrap()
                .free(staging_allocation)
                .unwrap();
            self.core.device.destroy_command_pool(cmd_pool, None);
            self.core.device.destroy_fence(cmd_buff_fence, None);
        }
    }

    /// Update swapchain.
    pub fn update_swapchain(
        &mut self,
        physical_width: u32,
        physical_height: u32,
        swapchain: vk::SwapchainKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) {
        self.physical_width = physical_width;
        self.physical_height = physical_height;

        // release vk objects to be regenerated.
        unsafe {
            self.core.device.destroy_render_pass(self.render_pass, None);
            self.core.device.destroy_pipeline(self.pipeline, None);
            for &image_view in self.framebuffer_color_image_views.iter() {
                self.core.device.destroy_image_view(image_view, None);
            }
            for &framebuffer in self.framebuffers.iter() {
                self.core.device.destroy_framebuffer(framebuffer, None);
            }
        }

        // swap images
        let swap_images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images.");

        // Recreate render pass for update surface format
        self.render_pass = unsafe {
            self.core.device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&[vk::AttachmentDescription::builder()
                        .format(surface_format.format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        .build()])
                    .subpasses(&[vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::builder()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .build()])
                        .build()])
                    .dependencies(&[vk::SubpassDependency::builder()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .build()]),
                None,
            )
        }
        .expect("Failed to create render pass.");

        // Recreate pipeline for update render pass
        self.pipeline = {
            let bindings = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .input_rate(vk::VertexInputRate::VERTEX)
                .stride(5 * std::mem::size_of::<f32>() as u32)
                .build()];
            let attributes = [
                // position
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(0)
                    .location(0)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // uv
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(8)
                    .location(1)
                    .format(vk::Format::R32G32_SFLOAT)
                    .build(),
                // color
                vk::VertexInputAttributeDescription::builder()
                    .binding(0)
                    .offset(16)
                    .location(2)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .build(),
            ];

            let vertex_shader_module = {
                let bytes_code = include_bytes!("../../shaders/spv/egui_vert.vert");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.core
                        .device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create vertex shader module.")
            };
            let fragment_shader_module = {
                let bytes_code = include_bytes!("../../shaders/spv/egui_frag.frag");
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: bytes_code.len(),
                    p_code: bytes_code.as_ptr() as *const u32,
                    ..Default::default()
                };
                unsafe {
                    self.core
                        .device
                        .create_shader_module(&shader_module_create_info, None)
                }
                .expect("Failed to create fragment shader module.")
            };
            let main_function_name = CString::new("main").unwrap();
            let pipeline_shader_stages = [
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module)
                    .name(&main_function_name)
                    .build(),
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module)
                    .name(&main_function_name)
                    .build(),
            ];

            let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0);
            let stencil_op = vk::StencilOpState::builder()
                .fail_op(vk::StencilOp::KEEP)
                .pass_op(vk::StencilOp::KEEP)
                .compare_op(vk::CompareOp::ALWAYS)
                .build();
            let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .front(stencil_op)
                .back(stencil_op);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .build()];
            let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(&color_blend_attachments);
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state_info =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_attribute_descriptions(&attributes)
                .vertex_binding_descriptions(&bindings);
            let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let pipeline_create_info = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&pipeline_shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_info)
                .viewport_state(&viewport_info)
                .rasterization_state(&rasterization_info)
                .multisample_state(&multisample_info)
                .depth_stencil_state(&depth_stencil_info)
                .color_blend_state(&color_blend_info)
                .dynamic_state(&dynamic_state_info)
                .layout(self.pipeline_layout)
                .render_pass(self.render_pass)
                .subpass(0)
                .build()];

            let pipeline = unsafe {
                self.core.device.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_info,
                    None,
                )
            }
            .expect("Failed to create graphics pipeline")[0];
            unsafe {
                self.core
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
                self.core
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
            }
            pipeline
        };

        // Recreate color image views for new framebuffers
        self.framebuffer_color_image_views = swap_images
            .iter()
            .map(|swapchain_image| unsafe {
                self.core
                    .device
                    .create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(swapchain_image.clone())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(surface_format.format)
                            .subresource_range(
                                vk::ImageSubresourceRange::builder()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(0)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1)
                                    .build(),
                            ),
                        None,
                    )
                    .expect("Failed to create image view.")
            })
            .collect::<Vec<_>>();
        // Recreate framebuffers for new swapchain
        self.framebuffers = self
            .framebuffer_color_image_views
            .iter()
            .map(|&image_views| unsafe {
                let attachments = &[image_views];
                self.core
                    .device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::builder()
                            .render_pass(self.render_pass)
                            .attachments(attachments)
                            .width(physical_width)
                            .height(physical_height)
                            .layers(1),
                        None,
                    )
                    .expect("Failed to create framebuffer.")
            })
            .collect::<Vec<_>>();
    }

    /// Registering user texture.
    ///
    /// Pass the Vulkan ImageView and Sampler.
    /// `image_view`'s image layout must be `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// UserTexture needs to be unregistered when it is no longer needed.
    ///
    /// # Example
    /// ```sh
    /// cargo run --example user_texture
    /// ```
    /// [The example for user texture is in examples directory](https://github.com/MatchaChoco010/egui_winit_ash_vk_mem/tree/main/examples/user_texture)
    pub fn register_user_texture(
        &mut self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> egui::TextureId {
        // get texture id
        let mut id = None;
        for (i, user_texture) in self.user_textures.iter().enumerate() {
            if user_texture.is_none() {
                id = Some(i as u64);
                break;
            }
        }
        let id = if let Some(i) = id {
            i
        } else {
            self.user_textures.len() as u64
        };

        // allocate and update descriptor set
        let layouts = [self.descriptor_set_layout.handle()];
        let descriptor_set = unsafe {
            self.core.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(self.descriptor_pool.handle())
                    .set_layouts(&layouts),
            )
        }
        .expect("Failed to create descriptor sets.")[0];
        unsafe {
            self.core.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::builder()
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .image_info(&[vk::DescriptorImageInfo::builder()
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(sampler)
                        .build()])
                    .dst_binding(0)
                    .build()],
                &[],
            );
        }

        if id == self.user_textures.len() as u64 {
            self.user_textures.push(Some(descriptor_set));
        } else {
            self.user_textures[id as usize] = Some(descriptor_set);
        }

        egui::TextureId::User(id)
    }

    /// Unregister user texture.
    ///
    /// The internal texture (egui::TextureId::Egui) cannot be unregistered.
    pub fn unregister_user_texture(&mut self, texture_id: egui::TextureId) {
        if let egui::TextureId::User(id) = texture_id {
            if let Some(descriptor_set) = self.user_textures[id as usize] {
                unsafe {
                    self.core
                        .device
                        .free_descriptor_sets(self.descriptor_pool.handle(), &[descriptor_set])
                        .expect("Failed to free descriptor sets.");
                }
                self.user_textures[id as usize] = None;
            }
        } else {
            eprintln!("The internal texture cannot be unregistered; please pass the texture ID of UserTexture.");
            return;
        }
    }

    /// destroy vk objects.
    ///
    /// # Unsafe
    /// This method release vk objects memory that is not managed by Rust.
    pub unsafe fn destroy(&mut self) {
        for (buffer, allocation) in self
            .index_buffers
            .drain(0..)
            .zip(self.index_buffer_allocations.drain(0..))
        {
            self.core.device.destroy_buffer(buffer, None);
            self.allocator
                .lock()
                .unwrap()
                .free(allocation)
                .expect("Failed to free allocation");
        }
        for (buffer, allocation) in self
            .vertex_buffers
            .drain(0..)
            .zip(self.vertex_buffer_allocations.drain(0..))
        {
            self.core.device.destroy_buffer(buffer, None);
            self.allocator
                .lock()
                .unwrap()
                .free(allocation)
                .expect("Failed to free allocation");
        }
        for &image_view in self.framebuffer_color_image_views.iter() {
            self.core.device.destroy_image_view(image_view, None);
        }
        for &framebuffer in self.framebuffers.iter() {
            self.core.device.destroy_framebuffer(framebuffer, None);
        }
        self.core.device.destroy_render_pass(self.render_pass, None);
        self.core.device.destroy_sampler(self.sampler, None);
        self.core.device.destroy_pipeline(self.pipeline, None);
        self.core
            .device
            .destroy_pipeline_layout(self.pipeline_layout, None);

        for (_texture_id, texture_image) in self.texture_images.drain() {
            self.core.device.destroy_image(texture_image, None);
        }
        for (_texture_id, texture_image_view) in self.texture_image_views.drain() {
            self.core
                .device
                .destroy_image_view(texture_image_view, None);
        }
        for (_texture_id, texture_allocation) in self.texture_allocations.drain() {
            self.allocator
                .lock()
                .unwrap()
                .free(texture_allocation)
                .unwrap();
        }
    }
}
