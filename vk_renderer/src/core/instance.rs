use std::sync::Arc;

use vulkano::memory::allocator::{FreeListAllocator, GenericMemoryAllocator};

/// Data that will be read only for the course of the program
pub struct Instance {
    surface: Arc<vulkano::swapchain::Surface>,
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    //camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
    //partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
}

impl Instance {
    pub fn new(
        surface: Arc<vulkano::swapchain::Surface>,
        device: Arc<vulkano::device::Device>,
        queue: Arc<vulkano::device::Queue>,
        memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
        //camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
        //partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            memory_allocator,
            //camera_bind_group_layout,
            //partition_bind_group_layout,
        }
    }

    pub fn surface(&self) -> &vulkano::swapchain::Surface {
        &self.surface
    }
    pub fn device(&self) -> Arc<vulkano::device::Device> {
        self.device.clone()
    }
    pub fn queue(&self) -> Arc<vulkano::device::Queue> {
        self.queue.clone()
    }

    //pub fn camera_bind_group_layout(&self) -> &super::buffer::BindGroupLayout<1> {
    //    &self.camera_bind_group_layout
    //}
    //
    //pub fn partition_bind_group_layout(&self) -> &super::buffer::BindGroupLayout<2> {
    //    &self.partition_bind_group_layout
    //}

    pub fn memory_allocator(&self) -> &GenericMemoryAllocator<Arc<FreeListAllocator>> {
        &self.memory_allocator
    }
}
