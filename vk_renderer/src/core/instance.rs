use std::sync::Arc;

use vulkano::{
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
        PersistentDescriptorSet,
    },
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator},
};

/// Data that will be read only for the course of the program
pub struct Instance {
    surface: Arc<vulkano::swapchain::Surface>,
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
    descriptor_set_memory_allocator: StandardDescriptorSetAllocator,
    //camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
    partitions_descriptor_set_layout: Arc<DescriptorSetLayout>,
}

impl Instance {
    pub fn new(
        surface: Arc<vulkano::swapchain::Surface>,
        device: Arc<vulkano::device::Device>,
        queue: Arc<vulkano::device::Queue>,
        memory_allocator: GenericMemoryAllocator<Arc<FreeListAllocator>>,
        //camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
        descriptor_set_memory_allocator: StandardDescriptorSetAllocator,
        partitions_descriptor_set_layout: Arc<DescriptorSetLayout>,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            memory_allocator,
            //camera_bind_group_layout,
            descriptor_set_memory_allocator,
            partitions_descriptor_set_layout,
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

    pub fn partitions_descriptor_set_layout(&self) -> Arc<DescriptorSetLayout> {
        self.partitions_descriptor_set_layout.clone()
    }

    pub fn descriptor_set_memory_allocator(&self) -> &StandardDescriptorSetAllocator {
        &self.descriptor_set_memory_allocator
    }
}
