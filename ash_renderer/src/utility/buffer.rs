use std::{
    marker::PhantomData,
    mem, ptr,
    sync::{Arc, Mutex},
};

use ash::vk::{self, BufferUsageFlags};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::{core::Core, VkHandle};

use super::{device::Device, macros::vk_handle_wrapper, pooled::command_pool::CommandPool};

pub const STAGING_BUFFER: &str = "Staging Buffer";

pub trait AsBuffer: VkHandle<VkItem = vk::Buffer> {
    fn buffer(self: &Arc<Self>) -> Arc<Buffer>;

    fn allocation(&self) -> &Allocation;

    fn size(&self) -> vk::DeviceSize;

    fn usage(&self) -> vk::BufferUsageFlags;

    fn full_range_descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.handle(),
            offset: 0,
            range: self.size(),
        }
    }
    fn write_descriptor_sets<'a>(
        &self,
        dst_set: vk::DescriptorSet,
        descriptor_type: vk::DescriptorType,
        info: &'a [vk::DescriptorBufferInfo; 1],
        dst_binding: u32,
    ) -> vk::WriteDescriptorSetBuilder<'a> {
        vk::WriteDescriptorSet::builder()
            .dst_set(dst_set)
            .dst_binding(dst_binding)
            .dst_array_element(0)
            .descriptor_type(descriptor_type)
            .buffer_info(info)
    }
}
#[repr(transparent)]
pub struct TBuffer<T> {
    buffer: Arc<Buffer>,
    _p: PhantomData<T>,
}

impl<T> TBuffer<T> {
    pub fn new(buffer: Arc<Buffer>) -> Self {
        Self {
            buffer,
            _p: PhantomData,
        }
    }

    pub fn get_buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl<T> VkHandle for TBuffer<T> {
    type VkItem = vk::Buffer;

    fn handle(&self) -> Self::VkItem {
        assert_ne!(self.buffer.handle, vk::Buffer::null());

        self.buffer.handle
    }
}

impl<T> AsBuffer for TBuffer<T> {
    fn buffer(self: &Arc<Self>) -> Arc<Buffer> {
        self.buffer.clone()
    }

    fn allocation(&self) -> &Allocation {
        self.buffer.allocation()
    }

    fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }

    fn usage(&self) -> vk::BufferUsageFlags {
        self.buffer.usage
    }
}
impl<T> TBuffer<T> {
    pub fn new_per_swapchain(
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        _location: MemoryLocation,
        swapchain_image_count: usize,
        name: &str,
    ) -> Vec<Arc<Self>> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for _ in 0..swapchain_image_count {
            let uniform_buffer = Buffer::new(
                core,
                allocator.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                name,
            );
            uniform_buffers.push(Arc::new(TBuffer::new(Arc::new(uniform_buffer))));
        }

        uniform_buffers
    }

    pub fn new_with_data(
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        data: Vec<T>,
        name: &str,
    ) -> Vec<Self> {
        let buffer_size = ::std::mem::size_of::<T>();

        let mut uniform_buffers = vec![];

        for datum in data {
            let uniform_buffer = Buffer::new(
                core,
                allocator.clone(),
                buffer_size as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                name,
            );
            let buf = TBuffer::new(Arc::new(uniform_buffer));

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
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        submit_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: &str,
    ) -> Arc<Self> {
        Arc::new(Self {
            buffer: Buffer::new_filled(core, allocator.clone(), submit_queue, usage, data, name),
            _p: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        (self.size() as usize) / self.stride()
    }
    pub fn stride(&self) -> usize {
        mem::size_of::<T>()
    }
}

impl TBuffer<vk::DrawMeshTasksIndirectCommandEXT> {
    /// Add cmd_draw_mesh_tasks_indirect to this task buffer
    pub unsafe fn draw_tasks_indirect(&self, cmd: vk::CommandBuffer) {
        self.buffer
            .device
            .fn_mesh_shader
            .cmd_draw_mesh_tasks_indirect(
                cmd,
                self.handle(),
                0,
                self.len() as _,
                self.stride() as _,
            );
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
    fn buffer(self: &Arc<Self>) -> Arc<Buffer> {
        self.clone()
    }

    fn allocation(&self) -> &Allocation {
        &self.allocation
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }

    fn usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.handle, None);

            let mut allocation = Allocation::default();
            mem::swap(&mut self.allocation, &mut allocation);

            self.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

vk_handle_wrapper!(Buffer);

impl Buffer {
    /// Create a new exclusive buffer, with a backing allocation from the allocator
    pub fn new(
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Buffer {
        let buffer_create_info = vk::BufferCreateInfo {
            size: size.max(1),
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };

        let buffer = unsafe {
            core.device
                .create_buffer(&buffer_create_info, None)
                .expect("Failed to create Vertex Buffer")
        };

        core.name_object(name, buffer);

        let requirements = unsafe { core.device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            core.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .expect("Failed to bind Buffer");
        }

        Buffer {
            device: core.device.clone(),
            handle: buffer,
            size,
            allocator,
            allocation,
            usage,
        }
    }

    pub fn new_storage_filled<T: bytemuck::Pod>(
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        submit_queue: vk::Queue,
        data: &[T],
        name: &str,
    ) -> Arc<Buffer> {
        Self::new_filled(
            core,
            allocator,
            submit_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            data,
            name,
        )
    }

    /// Create a new buffer and fill it with initial data copied in from a staging buffer
    pub fn new_filled<T>(
        core: &Core,
        allocator: Arc<Mutex<Allocator>>,
        submit_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: &str,
    ) -> Arc<Buffer> {
        let buffer_size = ::std::mem::size_of_val(data) as vk::DeviceSize;

        let staging_buffer = Self::new(
            core,
            allocator.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            STAGING_BUFFER,
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
            core,
            allocator,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryLocation::GpuOnly,
            name,
        );

        staging_buffer.copy_to_other(submit_queue, &core.command_pool, &buffer, buffer_size);

        // println!("Created buffer, size: {}", buffer_size);

        Arc::new(buffer)
    }

    pub unsafe fn fill(&self, cmd: vk::CommandBuffer, data: u32) {
        self.device
            .cmd_fill_buffer(cmd, self.handle(), 0, self.size, data)
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

        let command_buffer = command_pool.begin_instant_command(submit_queue);

        let copy_regions = [vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        }];

        unsafe {
            self.device.cmd_copy_buffer(
                command_buffer.handle,
                self.handle,
                other.handle,
                &copy_regions,
            );
        }
    }
}
