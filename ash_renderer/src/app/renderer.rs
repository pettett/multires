use crate::{
    draw_pipelines::stub::Stub,
    gui::allocator_visualiser_window::AllocatorVisualiserWindow,
    utility::{device::Device, image::ImageTrackedLayout, pooled::query_pool::TimestampResults},
    Config, VkHandle,
};
use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::components::camera::Camera;
use gpu_allocator::vulkan::Allocator;
use image::{ImageBuffer, Rgb, RgbImage};
use std::{
    env, fs, path,
    sync::{Arc, Mutex, MutexGuard},
    thread, time,
};

use crate::{
    core::Core,
    draw_pipelines::{indirect_tasks::MeshShaderMode, DrawPipeline},
    gui::{gui::Gui, window::GuiWindow},
    utility::{
        image::Image,
        pooled::{
            command_pool::CommandBuffer,
            descriptor_pool::DescriptorPool,
            query_pool::{PrimitivesQueryResults, QueryPool, TypelessQueryPool},
        },
        render_pass::RenderPass,
        screen::Screen,
        sync::SyncObjects,
        ShaderModule,
    },
};

use super::{
    frame_measure::RollingMeasure,
    material::{Fragment, Material},
    mesh_data::MeshData,
    scene::Scene,
};
#[derive(Debug, Clone, Copy, Event, PartialEq, Eq)]
pub enum MeshDrawingPipelineType {
    DrawIndirect,
    DrawLOD,
    LocalSelectMesh,
    LocalSelectIndices,
    ExpandingComputeCulledMesh,
    ExpandingComputeCulledIndices,
    None,
}

#[derive(Resource)]
pub struct Renderer {
    pub allocator: Arc<Mutex<Allocator>>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub draw_pipeline: Box<dyn DrawPipeline + Sync + Send>,
    pub descriptor_pool: Arc<DescriptorPool>,
    pub windows: Vec<Box<dyn GuiWindow + Sync + Send>>,
    pub current_pipeline: MeshDrawingPipelineType,
    pub mesh_mode: MeshShaderMode,
    pub sync_objects: SyncObjects,

    // Evaluation data
    pub last_sample: time::Instant,
    pub primitives: RollingMeasure<u32, 60>,
    pub gpu_time: RollingMeasure<f64, 60>,
    pub query_primitives: QueryPool<PrimitivesQueryResults>,
    pub query_timestamp: QueryPool<TimestampResults>,
    pub query: bool,
    pub query_time: bool,

    pub render_gui: bool,
    pub current_frame: usize,
    pub image_index: usize,
    pub is_suboptimal: bool,
    pub is_framebuffer_resized: bool,
    pub app_info_open: bool,
    // Make sure to drop the core last
    pub screen: Screen,
    pub core: Arc<Core>,

    pub fragment_colour: Material,
    pub fragment_lit: Material,
    pub fragment_edge: Material,

    pub mesh: String,

    pub fragment: Fragment,
    pub hacky_command_buffer_passthrough: Option<vk::CommandBuffer>,
}

impl Renderer {
    pub fn new(
        config: &Config,
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        screen: Screen,
        core: Arc<Core>,
        descriptor_pool: Arc<DescriptorPool>,
        sync_objects: SyncObjects,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: RenderPass,
    ) -> Self {
        let pbr = Arc::new(ShaderModule::new(
            core.device.clone(),
            include_bytes!("../../shaders/spv/frag_pbr.frag"),
        ));

        let col = Arc::new(ShaderModule::new(
            core.device.clone(),
            include_bytes!("../../shaders/spv/frag_colour.frag"),
        ));

        Self {
            fragment_colour: Material::new(col.clone(), vk::PolygonMode::FILL),
            fragment_lit: Material::new(pbr, vk::PolygonMode::FILL),
            fragment_edge: Material::new(col, vk::PolygonMode::LINE),

            graphics_queue,
            present_queue,
            render_pass,
            draw_pipeline: Box::new(Stub),
            descriptor_pool,
            windows: vec![Box::new(AllocatorVisualiserWindow::new(allocator.clone()))],
            mesh_mode: MeshShaderMode::TriangleList,
            allocator,
            current_pipeline: MeshDrawingPipelineType::None,
            sync_objects,
            query: false,
            query_time: true,
            current_frame: 0,
            is_framebuffer_resized: false,
            app_info_open: true,
            render_gui: true, // disable GUI during benchmarks
            query_primitives: QueryPool::new(core.device.clone(), 1),
            query_timestamp: QueryPool::new(core.device.clone(), 2),
            core,
            screen,
            mesh: config.mesh_names[0].clone(),
            fragment: Fragment::Lit,
            hacky_command_buffer_passthrough: None,
            image_index: 0,
            is_suboptimal: false,
            last_sample: time::Instant::now(),
            primitives: Default::default(),
            gpu_time: Default::default(),
        }
    }

    pub fn recreate_swapchain(&mut self, gui: &mut Gui, cam: &mut Camera) {
        self.core.device.wait_device_idle();

        let size = self.window().inner_size();

        cam.on_resize(&size);

        //self.cleanup_swapchain();

        self.screen.remake_swapchain(
            self.graphics_queue,
            &self.render_pass,
            self.allocator.clone(),
        );

        self.draw_pipeline
            .init_swapchain(&self.core, &self.screen, &self.render_pass);

        // Egui Integration
        gui.update_swapchain(self.screen.swapchain());
    }

    pub fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    pub fn fragment(&self) -> &Material {
        match self.fragment {
            Fragment::VertexColour => &self.fragment_colour,
            Fragment::Lit => &self.fragment_lit,
            Fragment::Edges => &self.fragment_edge,
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.core.window
    }
    pub fn get_allocator(&self) -> MutexGuard<Allocator> {
        self.allocator.lock().unwrap()
    }

    pub fn get_query(&self) -> Option<Arc<TypelessQueryPool>> {
        self.query.then(|| self.query_primitives.typeless())
    }
    pub fn get_timestamp_query(&self) -> Option<Arc<TypelessQueryPool>> {
        self.query_time.then(|| self.query_timestamp.typeless())
    }

    pub fn get_query_ref(&self) -> Option<&TypelessQueryPool> {
        self.query.then(|| self.query_primitives.typeless_ref())
    }

    pub fn screenshot(&self, subpath: String, filename: String) {
        // https://github.com/SaschaWillems/Vulkan/blob/master/examples/screenshot/screenshot.cpp

        // // Check blit support for source and destination
        let supports_blit_src = self
            .core
            .physical_device
            .get_format_properties(self.screen.swapchain().surface_format.format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_SRC);

        if !supports_blit_src {
            println!("Device does not support blitting from optimal tiled images, using copy instead of blit!")
        } else {
            println!("Supports Blit Src")
        }

        // // Check if the device supports blitting to linear images
        let dst_format = vk::Format::R8G8B8A8_SRGB;

        let supports_blit_dst = self
            .core
            .physical_device
            .get_format_properties(dst_format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::BLIT_DST);

        if !supports_blit_dst {
            println!("Device does not support blitting to linear tiled images, using copy instead of blit!")
        } else {
            println!("Supports Blit Dst")
        }

        let supports_blit = supports_blit_dst && supports_blit_src;

        // // Source for the copy is the last rendered swapchain image

        let mut src_image = self.screen.swapchain().image(self.current_frame);

        // // Create the linear tiled destination image to copy to and to read the memory from

        let mut dst_image = Image::create_image(
            &self.core,
            self.screen.width(),
            self.screen.height(),
            1,
            vk::SampleCountFlags::TYPE_1,
            dst_format,
            vk::ImageTiling::LINEAR,
            vk::ImageUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
            self.allocator.clone(),
            "Screenshot",
        );
        let color_subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        {
            // Do the actual blit from the swapchain image to our host visible destination image

            let cmd = self
                .core
                .command_pool
                .begin_instant_command(self.graphics_queue);

            // Transition destination image to transfer destination layout

            dst_image.insert_image_memory_barrier(
                &self.core.device,
                &cmd.handle,
                vk::AccessFlags2::NONE,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::TRANSFER,
                color_subresource,
            );

            // Transition swapchain image from present to transfer source layout

            src_image.insert_image_memory_barrier(
                &self.core.device,
                &cmd.handle,
                vk::AccessFlags2::MEMORY_READ,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::TRANSFER,
                color_subresource,
            );

            //  If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
            if supports_blit {
                //  Define the region to blit (we will blit the whole swapchain image)

                let blit_zero = vk::Offset3D::default();

                let blit_size = vk::Offset3D::default()
                    .x(dst_image.width() as _)
                    .y(dst_image.height() as _)
                    .z(1);

                let offsets = [blit_zero, blit_size];

                let color_subresource = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);

                let regions = [vk::ImageBlit::default()
                    .src_subresource(color_subresource)
                    .dst_subresource(color_subresource)
                    .src_offsets(offsets)
                    .dst_offsets(offsets)];

                // Issue the blit command

                unsafe {
                    self.core.device.cmd_blit_image(
                        cmd.handle,
                        src_image.handle(),
                        src_image.current_layout(),
                        dst_image.handle(),
                        dst_image.current_layout(),
                        &regions,
                        vk::Filter::NEAREST,
                    );
                };
            } else {
                //  Otherwise use image copy (requires us to manually flip components)
                // 	VkImageCopy imageCopyRegion{};
                // 	imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                // 	imageCopyRegion.srcSubresource.layerCount = 1;
                // 	imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                // 	imageCopyRegion.dstSubresource.layerCount = 1;
                // 	imageCopyRegion.extent.width = width;
                // 	imageCopyRegion.extent.height = height;
                // 	imageCopyRegion.extent.depth = 1;

                //  Issue the copy command
                // 	vkCmdCopyImage(
                // 		copyCmd,
                // 		srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                // 		dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                // 		1,
                // 		&imageCopyRegion);
                // }
                todo!("Implement standard copy");
            }

            // Transition destination image to general layout, which is the required layout for mapping the image memory later on

            dst_image.insert_image_memory_barrier(
                &self.core.device,
                &cmd.handle,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::MEMORY_READ,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::TRANSFER,
                color_subresource,
            );

            // Transition back the swap chain image after the blit is done

            src_image.insert_image_memory_barrier(
                &self.core.device,
                &cmd.handle,
                vk::AccessFlags2::TRANSFER_READ,
                vk::AccessFlags2::MEMORY_READ,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::TRANSFER,
                color_subresource,
            );
        }
        let core = self.core.clone();

        thread::spawn(move || {
            let data = dst_image.image_memory().mapped_slice().unwrap();

            // Get layout of the image (including row pitch)

            let layout = unsafe {
                core.device.get_image_subresource_layout(
                    dst_image.handle(),
                    vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
                )
            };

            let data = &data[layout.offset as usize..];

            let mut img = RgbImage::new(dst_image.width(), dst_image.height());

            for y in 0..dst_image.height() {
                let row = &data[(y * layout.row_pitch as u32) as usize..];

                for x in 0..dst_image.width() {
                    img.put_pixel(
                        x,
                        y,
                        Rgb([
                            row[(x * 4) as usize],
                            row[(x * 4 + 1) as usize],
                            row[(x * 4 + 2) as usize],
                        ]),
                    )
                }
            }
            let out_dir = env::current_dir()
                .unwrap()
                .join("screenshots")
                .join(subpath);
            if !out_dir.is_dir() {
                fs::create_dir_all(&out_dir).expect("Failed to create screenshots folder");
            }
            let out = out_dir.join(filename);
            img.save(&out).expect("Failed to save screenshot");

            println!("Saved screenshot to {:?}", out);
        });
    }
}
