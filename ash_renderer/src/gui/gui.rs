use std::sync::{Arc, Mutex};

use ash::vk;

use gpu_allocator::vulkan::Allocator;
use raw_window_handle::HasRawDisplayHandle;

use crate::{
    core::Core,
    utility::{
        device::Device,
        pooled::{
            command_buffer_group::CommandBufferGroup,
            command_pool::{CommandBuffer, CommandPool},
        },
        queue_family_indices::QueueFamilyIndices,
        swapchain::Swapchain,
    },
};

pub struct Gui {
    device: Arc<Device>,
    window: Arc<winit::window::Window>,
    integration: super::integration::Integration,
    ui_command_buffers: Arc<CommandBufferGroup>,
    in_frame: bool,
}

impl Gui {
    pub fn new(
        core: Arc<Core>,
        window: Arc<winit::window::Window>,
        target: &impl HasRawDisplayHandle,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        queue_family: &QueueFamilyIndices,
        graphics_queue: vk::Queue,
        swapchain: &Swapchain,
    ) -> Self {
        let ui_command_buffers =
            CommandBufferGroup::new(command_pool.clone(), swapchain.images.len() as _);

        let integration = super::integration::Integration::new(
            target,
            swapchain.extent.width,
            swapchain.extent.height,
            1.0,
            egui::FontDefinitions::default(),
            egui::Style::default(),
            core.clone(),
            allocator.clone(),
            queue_family.graphics_family.unwrap(),
            graphics_queue,
            core.device.fn_swapchain.clone(),
            swapchain.handle,
            swapchain.surface_format,
        );
        Self {
            device: core.device.clone(),
            integration,
            window,
            ui_command_buffers: Arc::new(ui_command_buffers),
            in_frame: false,
        }
    }

    pub fn start_draw(&mut self) {
        self.integration.begin_frame(&self.window);
        self.in_frame = true;
    }

    pub fn draw(&self) -> egui::Context {
        assert!(self.in_frame);
        self.integration.context()
    }

    pub fn finish_draw(&mut self, image_index: usize) -> &CommandBuffer {
        let output = self.integration.end_frame(&self.window);
        self.in_frame = false;

        let clipped_meshes = self.integration.context().tessellate(output.shapes);

        let cmd = self.ui_command_buffers.get(image_index);

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(**cmd, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        self.integration
            .paint(**cmd, image_index, clipped_meshes, output.textures_delta);

        unsafe {
            self.device
                .end_command_buffer(**cmd)
                .expect("Failed to record Command Buffer at Ending!");
        }

        cmd
    }

    pub fn handle_event(&mut self, winit_event: &winit::event::WindowEvent<'_>) -> bool {
        self.integration.handle_event(winit_event).consumed
    }
    pub fn update_swapchain(&mut self, swapchain: &Swapchain) {
        self.integration.update_swapchain(
            swapchain.extent.width,
            swapchain.extent.height,
            swapchain.handle,
            swapchain.surface_format,
        );
    }
}
