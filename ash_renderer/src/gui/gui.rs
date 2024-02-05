use std::sync::{Arc, Mutex};

use ash::vk;

use gpu_allocator::vulkan::Allocator;
use raw_window_handle::HasRawDisplayHandle;

use crate::{
    utility::{
        device::Device,
        pooled::{command_buffer_group::CommandBufferGroup, command_pool::CommandPool},
        swapchain::Swapchain,
    },
    QueueFamilyIndices,
};

use super::gpu_allocator::GpuAllocator;

pub struct Gui {
    device: Arc<Device>,
    window: Arc<winit::window::Window>,
    integration: egui_winit_ash_integration::Integration<GpuAllocator>,
    ui_command_buffers: CommandBufferGroup,
}

impl Gui {
    pub fn new(
        device: Arc<Device>,
        window: Arc<winit::window::Window>,
        target: &impl HasRawDisplayHandle,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        queue_family: &QueueFamilyIndices,
        graphics_queue: vk::Queue,
        swapchain: &Swapchain,
    ) -> Self {
        let ui_command_buffers = CommandBufferGroup::new(command_pool.clone(), swapchain.images.len() as _);

        let integration = egui_winit_ash_integration::Integration::<GpuAllocator>::new(
            target,
            swapchain.extent.width,
            swapchain.extent.height,
            1.0,
            egui::FontDefinitions::default(),
            egui::Style::default(),
            (**device).clone(),
            GpuAllocator(allocator.clone()),
            queue_family.graphics_family.unwrap(),
            graphics_queue,
            device.fn_swapchain.clone(),
            swapchain.handle,
            swapchain.surface_format,
        );
        Self {
            device,
            integration,
            window,
            ui_command_buffers,
        }
    }

    pub fn draw(
        &mut self,
        image_index: usize,
        draw_windows: impl FnOnce(&egui::Context),
    ) -> vk::CommandBuffer {
        let cmd = self.ui_command_buffers[image_index];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder();

        unsafe {
            self.device
                .begin_command_buffer(cmd, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        // //FIXME: this can be offloaded to a different thread
        self.integration.begin_frame(&self.window);

        draw_windows(&self.integration.context());

        let output = self.integration.end_frame(&self.window);

        let clipped_meshes = self.integration.context().tessellate(output.shapes);
        self.integration
            .paint(cmd, image_index, clipped_meshes, output.textures_delta);

        unsafe {
            self.device
                .end_command_buffer(cmd)
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

impl Drop for Gui {
    fn drop(&mut self) {
        unsafe {
            self.integration.destroy();
        }
    }
}
