use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    utility::{
        device::Device,
        image::{Image, ImageView},
        physical_device::PhysicalDevice,
        render_pass::RenderPass,
        swapchain::Swapchain,
    },
    VkHandle,
};

use super::macros::vk_device_owned_wrapper;

/// data structures that depend on the resolution of the current screen
pub struct Screen {
    core: Arc<Core>,
    swapchain: Option<Swapchain>,
    depth: Option<Arc<Image>>,
    pub swapchain_framebuffers: Vec<Framebuffer>,
    pub swapchain_image_views: Vec<ImageView>,
}

vk_device_owned_wrapper!(Framebuffer, destroy_framebuffer);

// TODO: Technically this should contain a reference to Arc<Renderpass>
impl Framebuffer {
    pub fn new(
        device: &Arc<Device>,
        render_pass: &RenderPass,
        view: &ImageView,
        physical_width: u32,
        physical_height: u32,
    ) -> Self {
        let attachments = &[view.handle()];

        let handle = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass.handle())
                        .attachments(attachments)
                        .width(physical_width)
                        .height(physical_height)
                        .layers(1),
                    None,
                )
                .expect("Failed to create framebuffer.")
        };

        Framebuffer {
            device: device.clone(),
            handle,
        }
    }
}

impl Screen {
    /// Get the swapchain. It can only be null during internal operations
    pub fn swapchain(&self) -> &Swapchain {
        self.swapchain.as_ref().unwrap()
    }

    pub fn depth(&self) -> &Image {
        self.depth.as_ref().unwrap()
    }

    pub fn new(core: Arc<Core>) -> Self {
        Screen {
            core,
            swapchain: None,
            depth: None,
            swapchain_image_views: Vec::new(),
            swapchain_framebuffers: Vec::new(),
        }
    }

    pub fn remake_swapchain(
        &mut self,
        _graphics_queue: vk::Queue,
        render_pass: &RenderPass,
        allocator: Arc<Mutex<Allocator>>,
    ) {
        // Cleanup old swapchain

        self.swapchain = None;

        self.swapchain = Some(Swapchain::new(
            self.core.device.clone(),
            &self.core.physical_device,
            &self.core.window,
            self.core.surface.clone(),
            &self.core.queue_family,
            None, // This is the first swapchain
        ));

        self.depth = Some(create_depth_resources(
            &self.core,
            self.swapchain().extent,
            allocator.clone(),
        ));

        self.swapchain_image_views = ImageView::create_image_views(
            &self.core.device,
            self.swapchain().surface_format.format,
            &self.swapchain().images,
        );

        self.swapchain_framebuffers = create_framebuffers(
            &self.core.device,
            render_pass,
            &self.swapchain_image_views,
            self.depth().image_view(),
            self.swapchain().extent,
        );
    }
}

fn create_depth_resources(
    core: &Core,
    swapchain_extent: vk::Extent2D,
    allocator: Arc<Mutex<Allocator>>,
) -> Arc<Image> {
    let depth_format = find_depth_format(&core.instance, &core.physical_device);
    Arc::new(
        Image::create_image(
            core,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            vk::SampleCountFlags::TYPE_1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            gpu_allocator::MemoryLocation::GpuOnly,
            allocator,
            "Depth Texture",
        )
        .create_image_view(depth_format, vk::ImageAspectFlags::DEPTH, 1),
    )
}

pub fn find_depth_format(instance: &ash::Instance, physical_device: &PhysicalDevice) -> vk::Format {
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
            || tiling == vk::ImageTiling::OPTIMAL
                && format_properties.optimal_tiling_features.contains(features)
        {
            return format;
        }
    }

    panic!("Failed to find supported format!")
}

fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}

fn create_framebuffers(
    device: &Arc<Device>,
    render_pass: &RenderPass,
    image_views: &[ImageView],
    depth_image_view: &ImageView,
    swapchain_extent: vk::Extent2D,
) -> Vec<Framebuffer> {
    let mut framebuffers = vec![];

    for image_view in image_views.iter() {
        let attachments = [image_view.handle(), depth_image_view.handle()];

        let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.handle())
            .attachments(&attachments)
            .width(swapchain_extent.width)
            .height(swapchain_extent.height)
            .layers(1);

        let handle = unsafe {
            device
                .create_framebuffer(&framebuffer_create_info, None)
                .expect("Failed to create Framebuffer!")
        };

        framebuffers.push(Framebuffer {
            device: device.clone(),
            handle,
        });
    }

    framebuffers
}
