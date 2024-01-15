use std::{marker::PhantomData, ptr, sync::Arc};

use ash::vk::{self, BufferUsageFlags};

use crate::VkHandle;

use super::{device::Device, image::find_memory_type, pools::CommandPool};

pub trait AsBuffer {
    fn buffer(&self) -> vk::Buffer;

    fn memory(&self) -> vk::DeviceMemory;

    fn size(&self) -> vk::DeviceSize;

    fn full_range_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer(),
            offset: 0,
            range: self.size(),
        }
    }
}

pub struct UniformBuffer<T> {
    pub buffer: Buffer,
    _p: PhantomData<T>,
}

impl<T> UniformBuffer<T> {
    pub fn new(buffer: Buffer) -> Self {
        Self {
            buffer,
            _p: PhantomData,
        }
    }
}
impl<T> AsBuffer for UniformBuffer<T> {
    fn buffer(&self) -> vk::Buffer {
        self.buffer.handle
    }

    fn memory(&self) -> vk::DeviceMemory {
        self.buffer.memory
    }

    fn size(&self) -> vk::DeviceSize {
        self.buffer.size
    }
}
impl<T> UniformBuffer<T> {
    pub fn new_per_swapchain(
        device: Arc<Device>,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        swapchain_image_count: usize,
    ) -> Vec<Self> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for _ in 0..swapchain_image_count {
            let uniform_buffer = Buffer::new(
                device.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                device_memory_properties,
            );
            uniform_buffers.push(UniformBuffer::new(uniform_buffer));
        }

        uniform_buffers
    }

    pub fn new_with_data(
        device: Arc<Device>,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        data: Vec<T>,
    ) -> Vec<Self> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for datum in data {
            let uniform_buffer = Buffer::new(
                device.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                device_memory_properties,
            );
            let buf = UniformBuffer::new(uniform_buffer);

            buf.update_uniform_buffer(datum);

            uniform_buffers.push(buf);
        }

        uniform_buffers
    }

    pub fn update_uniform_buffer(&self, data: T) {
        unsafe {
            let ubos = [data];

            let buffer_size = (std::mem::size_of::<T>() * ubos.len()) as u64;

            let data_ptr = self
                .buffer
                .device
                .handle
                .map_memory(
                    self.buffer.memory(),
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut T;

            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

            self.buffer.device.handle.unmap_memory(self.buffer.memory());
        }
    }
}
pub struct Buffer {
    // exists to allow drop
    device: Arc<Device>,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

impl AsBuffer for Buffer {
    fn buffer(&self) -> vk::Buffer {
        self.handle
    }

    fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_buffer(self.handle, None);
            self.device.handle.free_memory(self.memory, None);
        }
    }
}

impl VkHandle for Buffer {
    type VkItem = vk::Buffer;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl Buffer {
    pub fn new(
        device: Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory_properties: vk::MemoryPropertyFlags,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo {
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };

        let buffer = unsafe {
            device
                .handle
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Vertex Buffer")
        };

        let mem_requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_properties,
            device_memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo {
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
        };

        let memory = unsafe {
            device
                .handle
                .allocate_memory(&allocate_info, None)
                .expect("Failed to allocate vertex buffer memory!")
        };
        unsafe {
            device
                .handle
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind Buffer");
        }

        Buffer {
            device,
            handle: buffer,
            size,
            memory,
            usage,
        }
    }

    pub fn new_storage<T: bytemuck::Pod>(
        device: Arc<Device>,
        device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
        command_pool: Arc<CommandPool>,
        submit_queue: vk::Queue,
        data: &[T],
    ) -> Arc<Buffer> {
        let buffer_size = ::std::mem::size_of_val(data) as vk::DeviceSize;

        let staging_buffer = Self::new(
            device.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &device_memory_properties,
        );

        unsafe {
            let data_ptr = device
                .handle
                .map_memory(
                    staging_buffer.memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to Map Memory") as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

            device.handle.unmap_memory(staging_buffer.memory);
        }
        //THIS is not actually a vertex buffer, but a storage buffer that can be accessed from the mesh shader
        let storage_buffer = Self::new(
            device.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            &device_memory_properties,
        );

        staging_buffer.copy_to_other(submit_queue, command_pool, &storage_buffer, buffer_size);

        println!("Created buffer, size: {}", buffer_size);

        Arc::new(storage_buffer)
    }

    pub fn copy_to_other(
        &self,
        submit_queue: vk::Queue,
        command_pool: Arc<CommandPool>,
        other: &Buffer,
        size: vk::DeviceSize,
    ) {
        assert!(self.usage.contains(BufferUsageFlags::TRANSFER_SRC));
        assert!(other.usage.contains(BufferUsageFlags::TRANSFER_DST));

        let command_buffer = command_pool.begin_single_time_command(submit_queue);

        let copy_regions = [vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        }];

        unsafe {
            self.device.handle.cmd_copy_buffer(
                command_buffer.cmd,
                self.handle,
                other.handle,
                &copy_regions,
            );
        }
    }
}
