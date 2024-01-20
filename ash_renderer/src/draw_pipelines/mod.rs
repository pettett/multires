use std::sync::Arc;

use ash::vk;
use bevy_ecs::world::World;
use egui::mutex::Mutex;
use gpu_allocator::vulkan::Allocation;

use crate::{
    core::Core,
    screen::Screen,
    utility::{buffer::TypedBuffer, render_pass::RenderPass},
};

pub mod indirect_tasks;

pub trait DrawPipeline {
    fn draw(&self, frame_index: usize) -> vk::CommandBuffer;

    fn init_swapchain(
        &mut self,
        core: &Core,
        screen: &Screen,
        submesh_count: u32,
        instance_count: u32,
        render_pass: &RenderPass,
    );
}
