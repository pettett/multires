use std::ptr;

use ash::vk;

use super::{
    share::{begin_single_time_command, end_single_time_command, find_memory_type},
    structures::UniformBufferObject,
};

pub struct Buffer {
    // exists to allow drop
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Buffer {
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
pub fn create_buffer(
    device: ash::Device,
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
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create Vertex Buffer")
    };

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
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
            .allocate_memory(&allocate_info, None)
            .expect("Failed to allocate vertex buffer memory!")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("Failed to bind Buffer");
    }

    Buffer {
        device,
        buffer,
        size,
        memory,
    }
}

pub fn copy_buffer(
    device: &ash::Device,
    submit_queue: vk::Queue,
    command_pool: vk::CommandPool,
    src: &Buffer,
    dst: &Buffer,
    size: vk::DeviceSize,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let copy_regions = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    }];

    unsafe {
        device.cmd_copy_buffer(command_buffer, src.buffer, dst.buffer, &copy_regions);
    }

    end_single_time_command(device, command_pool, submit_queue, command_buffer);
}

pub fn create_storage_buffer<T: bytemuck::Pod>(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    data: &[T],
) -> Buffer {
    let buffer_size = ::std::mem::size_of_val(data) as vk::DeviceSize;

    let staging_buffer = create_buffer(
        device.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device_memory_properties,
    );

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer.memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to Map Memory") as *mut T;

        data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());

        device.unmap_memory(staging_buffer.memory);
    }
    //THIS is not actually a vertex buffer, but a storage buffer that can be accessed from the mesh shader
    let storage_buffer = create_buffer(
        device.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device_memory_properties,
    );

    copy_buffer(
        &device,
        submit_queue,
        command_pool,
        &staging_buffer,
        &storage_buffer,
        buffer_size,
    );

    println!("{}", buffer_size);

    storage_buffer
}

pub fn create_uniform_buffers(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    swapchain_image_count: usize,
) -> Vec<crate::utility::buffer::Buffer> {
    let buffer_size = ::std::mem::size_of::<UniformBufferObject>();

    let mut uniform_buffers = vec![];

    for _ in 0..swapchain_image_count {
        let uniform_buffer = create_buffer(
            device.clone(),
            buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device_memory_properties,
        );
        uniform_buffers.push(uniform_buffer);
    }

    uniform_buffers
}
