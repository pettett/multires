use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    utility::{
        device::Device, image::Image, physical_device::PhysicalDevice,
        render_pass::create_render_pass, swapchain::Swapchain,
    },
    VkHandle,
};

/// data structures that depend on the resolution of the current screen
pub struct Screen {
    core: Arc<Core>,
    swapchain: Option<Swapchain>,
    depth: Option<Arc<Image>>,
    pub render_pass: vk::RenderPass,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl Screen {
    /// Get the swapchain. It can only be null during internal operations
    pub fn swapchain(&self) -> &Swapchain {
        self.swapchain.as_ref().unwrap()
    }

    pub fn new(core: Arc<Core>) -> Self {
        Screen {
            core,
            swapchain: None,
            depth: None,
            render_pass: vk::RenderPass::null(),
            swapchain_image_views: Vec::new(),
            swapchain_framebuffers: Vec::new(),
        }
    }

    pub fn remake_swapchain(
        &mut self,
        graphics_queue: vk::Queue,
        allocator: Arc<Mutex<Allocator>>,
    ) {
        // Cleanup old swapchain

        println!("Remaking swapchain");
        self.swapchain = None;
        self.cleanup();

        self.swapchain = Some(Swapchain::new(
            self.core.device.clone(),
            &self.core.physical_device,
            &self.core.window,
            self.core.surface.clone(),
            &self.core.queue_family,
            None, // This is the first swapchain
        ));

        self.depth = Some(create_depth_resources(
            &self.core.instance.handle,
            self.core.device.clone(),
            &self.core.physical_device,
            self.core.command_pool.handle(),
            graphics_queue,
            self.swapchain().extent,
            allocator.clone(),
        ));

        self.render_pass = create_render_pass(
            &self.core.instance.handle,
            &self.core.device.handle,
            &self.core.physical_device,
            self.swapchain().surface_format.format,
            self.depth.as_ref().unwrap().format(),
        );

        self.swapchain_image_views = Image::create_image_views(
            &self.core.device,
            self.swapchain().surface_format.format,
            &self.swapchain().images,
        );

        self.swapchain_framebuffers = create_framebuffers(
            &self.core.device.handle,
            self.render_pass,
            &self.swapchain_image_views,
            self.depth.as_ref().unwrap().image_view(),
            self.swapchain().extent,
        );
    }

    fn cleanup(&mut self) {
        unsafe {
            println!("Cleaning up old screen data");
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.core
                    .device
                    .handle
                    .destroy_framebuffer(framebuffer, None);
            }

            for &image_view in self.swapchain_image_views.iter() {
                self.core.device.handle.destroy_image_view(image_view, None);
            }

            self.core
                .device
                .handle
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        self.cleanup()
    }
}

fn create_depth_resources(
    instance: &ash::Instance,
    device: Arc<Device>,
    physical_device: &PhysicalDevice,
    _command_pool: vk::CommandPool,
    _submit_queue: vk::Queue,
    swapchain_extent: vk::Extent2D,
    allocator: Arc<Mutex<Allocator>>,
) -> Arc<Image> {
    let depth_format = find_depth_format(instance, physical_device);
    Arc::new(
        Image::create_image(
            device,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            allocator,
        )
        .create_image_view(depth_format, vk::ImageAspectFlags::DEPTH, 1),
    )
}

fn find_depth_format(instance: &ash::Instance, physical_device: &PhysicalDevice) -> vk::Format {
    find_supported_format(
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
