use std::{
    marker::PhantomData,
    mem, ptr,
    sync::{Arc, Mutex},
};

use ash::vk::{self, BufferUsageFlags};
use gpu_allocator::{
    vulkan::{self, Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::VkHandle;

use super::{command_pool::CommandPool, device::Device};

pub trait AsBuffer {
    fn buffer(&self) -> vk::Buffer;

    fn allocation(&self) -> &Allocation;

    fn size(&self) -> vk::DeviceSize;

    fn full_range_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer(),
            offset: 0,
            range: self.size(),
        }
    }
}
#[repr(transparent)]
pub struct TypedBuffer<T> {
    pub buffer: Arc<Buffer>,
    _p: PhantomData<T>,
}

impl<T> TypedBuffer<T> {
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self {
            buffer,
            _p: PhantomData,
        }
    }
}
impl<T> AsBuffer for TypedBuffer<T> {
    fn buffer(&self) -> vk::Buffer {
        self.buffer.handle()
    }

    fn allocation(&self) -> &Allocation {
        self.buffer.allocation()
    }

    fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }
}
impl<T> TypedBuffer<T> {
    pub fn new_per_swapchain(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        location: MemoryLocation,
        swapchain_image_count: usize,
    ) -> Vec<Self> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for _ in 0..swapchain_image_count {
            let uniform_buffer = Buffer::new(
                device.clone(),
                allocator.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            );
            uniform_buffers.push(TypedBuffer::new(Arc::new(uniform_buffer)));
        }

        uniform_buffers
    }

    pub fn new_with_data(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        data: Vec<T>,
    ) -> Vec<Self> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for datum in data {
            let uniform_buffer = Buffer::new(
                device.clone(),
                allocator.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            );
            let buf = TypedBuffer::new(Arc::new(uniform_buffer));

            buf.update_uniform_buffer(datum);

            uniform_buffers.push(buf);
        }

        uniform_buffers
    }

    pub fn update_uniform_buffer(&self, data: T) {
        unsafe {
            let ubos = [data];

            let data_ptr = self
                .buffer
                .allocation
                .mapped_ptr()
                .expect("Failed to Map Memory")
                .as_ptr() as *mut T;
            data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());
        }
    }

    pub fn new_filled(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Arc<Self> {
        Arc::new(Self {
            buffer: Buffer::new_filled(
                device,
                allocator.clone(),
                command_pool,
                submit_queue,
                usage,
                data,
            ),
            _p: PhantomData,
        })
    }

    pub fn item_len(&self) -> usize {
        (self.size() as usize) / self.stride()
    }
    pub fn stride(&self) -> usize {
        mem::size_of::<T>()
    }
}
pub struct Buffer {
    // exists to allow drop
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    allocation: Allocation,
    handle: vk::Buffer,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

impl AsBuffer for Buffer {
    fn buffer(&self) -> vk::Buffer {
        self.handle
    }

    fn allocation(&self) -> &Allocation {
        &self.allocation
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_buffer(self.handle, None);

            let mut allocation = Allocation::default();
            mem::swap(&mut self.allocation, &mut allocation);

            self.allocator.lock().unwrap().free(allocation).unwrap();
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
        allocator: Arc<Mutex<Allocator>>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
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

        let requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "Example allocation",
                requirements,
                location,
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            device
                .handle
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("Failed to bind Buffer");
        }

        Buffer {
            device,
            handle: buffer,
            size,
            allocator,
            allocation,
            usage,
        }
    }

    pub fn new_storage_filled<T: bytemuck::Pod>(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        data: &[T],
    ) -> Arc<Buffer> {
        Self::new_filled(
            device,
            allocator,
            command_pool,
            submit_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            data,
        )
    }

    /// Create a new buffer and fill it with initial data copied in from a staging buffer
    pub fn new_filled<T>(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Arc<Buffer> {
        let buffer_size = ::std::mem::size_of_val(data) as vk::DeviceSize;

        let staging_buffer = Self::new(
            device.clone(),
            allocator.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        );

        unsafe {
            let data_ptr = staging_buffer
                .allocation
                .mapped_ptr()
                .expect("Failed to Map Memory")
                .as_ptr() as *mut T;

            data_ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        //THIS is not actually a vertex buffer, but a storage buffer that can be accessed from the mesh shader
        let buffer = Self::new(
            device.clone(),
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryLocation::GpuOnly,
        );

        staging_buffer.copy_to_other(submit_queue, command_pool, &buffer, buffer_size);

        println!("Created buffer, size: {}", buffer_size);

        Arc::new(buffer)
    }

    pub fn copy_to_other(
        &self,
        submit_queue: vk::Queue,
        command_pool: &Arc<CommandPool>,
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
                command_buffer.handle,
                self.handle,
                other.handle,
                &copy_regions,
            );
        }
    }
}
