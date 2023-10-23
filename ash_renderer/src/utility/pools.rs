use std::{ptr, sync::Arc};

use ash::vk;

use super::{buffer::Buffer, device::Device, image::Image, structures::UniformBufferObject};

pub struct DescriptorPool {
    device: Arc<Device>,
    pub pool: vk::DescriptorPool,
}
impl DescriptorPool {
    pub fn new(device: Arc<Device>, swapchain_images_size: u32) -> Arc<DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                // transform descriptor pool
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain_images_size,
            },
            vk::DescriptorPoolSize {
                // SSBO pool
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                // sampler descriptor pool
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: swapchain_images_size,
            },
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: swapchain_images_size as u32,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };

        let pool = unsafe {
            device
                .handle
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        };

        Arc::new(Self { pool, device })
    }
}

pub fn create_descriptor_sets(
    device: &ash::Device,
    descriptor_pool: &DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_buffers: &Vec<Buffer>,
    vert_buffer: &Buffer,
    index_buffer: &Buffer,
    texture: &Image,
    swapchain_images_size: usize,
) -> Vec<vk::DescriptorSet> {
    let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
    for _ in 0..swapchain_images_size {
        layouts.push(descriptor_set_layout);
    }

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool: descriptor_pool.pool,
        descriptor_set_count: swapchain_images_size as u32,
        p_set_layouts: layouts.as_ptr(),
    };

    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .expect("Failed to allocate descriptor sets!")
    };

    for (i, &descritptor_set) in descriptor_sets.iter().enumerate() {
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo {
            buffer: uniform_buffers[i].buffer(),
            offset: 0,
            range: ::std::mem::size_of::<UniformBufferObject>() as u64,
        }];

        let vertex_buffer_infos = [vk::DescriptorBufferInfo {
            buffer: vert_buffer.buffer(),
            offset: 0,
            range: vert_buffer.size(),
        }];
        let index_buffer_infos = [vk::DescriptorBufferInfo {
            buffer: index_buffer.buffer(),
            offset: 0,
            range: index_buffer.size(),
        }];

        let descriptor_image_infos = [vk::DescriptorImageInfo {
            sampler: texture.sampler(),
            image_view: texture.image_view(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];

        let descriptor_write_sets = [
            vk::WriteDescriptorSet {
                // transform uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_buffer_infos.as_ptr(),
                p_texel_buffer_view: ptr::null(),
            },
            vk::WriteDescriptorSet {
                // transform uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 4,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: vertex_buffer_infos.as_ptr(),
                p_texel_buffer_view: ptr::null(),
            },
            vk::WriteDescriptorSet {
                // transform uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 3,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: index_buffer_infos.as_ptr(),
                p_texel_buffer_view: ptr::null(),
            },
            vk::WriteDescriptorSet {
                // sampler uniform
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: descritptor_set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: descriptor_image_infos.as_ptr(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
            },
        ];

        unsafe {
            device.update_descriptor_sets(&descriptor_write_sets, &[]);
        }
    }

    descriptor_sets
}

pub fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let ubo_layout_bindings = [
        vk::DescriptorSetLayoutBinding {
            // transform uniform
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // verts buffer
            binding: 3,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // sampler uniform
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: ptr::null(),
        },
    ];

    let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: ubo_layout_bindings.len() as u32,
        p_bindings: ubo_layout_bindings.as_ptr(),
    };

    unsafe {
        device
            .create_descriptor_set_layout(&ubo_layout_create_info, None)
            .expect("Failed to create Descriptor Set Layout!")
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_descriptor_pool(self.pool, None);
        }
    }
}

pub struct CommandPool {
    pub device: Arc<Device>,
    pub pool: vk::CommandPool,
}

pub struct SingleTimeCommandBuffer {
    pool: Arc<CommandPool>,
    pub cmd: vk::CommandBuffer,
    submit_queue: vk::Queue,
}

impl Drop for SingleTimeCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.pool
                .device
                .handle
                .end_command_buffer(self.cmd)
                .expect("Failed to record Command Buffer at Ending!");
        }

        let buffers_to_submit = [self.cmd];

        let sumbit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            self.pool
                .device
                .handle
                .queue_submit(self.submit_queue, &sumbit_infos, vk::Fence::null())
                .expect("Failed to Queue Submit!");
            self.pool
                .device
                .handle
                .queue_wait_idle(self.submit_queue)
                .expect("Failed to wait Queue idle!");
            self.pool
                .device
                .handle
                .free_command_buffers(self.pool.pool, &buffers_to_submit);
        }
    }
}

impl CommandPool {
    pub fn new(device: Arc<Device>, queue_family_index: u32) -> Arc<Self> {
        let command_pool_create_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index,
        };

        let pool = unsafe {
            device
                .handle
                .create_command_pool(&command_pool_create_info, None)
                .expect("Failed to create Command Pool!")
        };

        Arc::new(Self { device, pool })
    }

    pub fn begin_single_time_command(
        self: Arc<Self>,
        submit_queue: vk::Queue,
    ) -> SingleTimeCommandBuffer {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool: self.pool,
            level: vk::CommandBufferLevel::PRIMARY,
        };

        let command_buffer = unsafe {
            self.device
                .handle
                .allocate_command_buffers(&command_buffer_allocate_info)
                .expect("Failed to allocate Command Buffers!")
        }[0];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: ptr::null(),
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        };

        unsafe {
            self.device
                .handle
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!")
        };

        SingleTimeCommandBuffer {
            pool: self,
            cmd: command_buffer,
            submit_queue,
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        //TODO: Command buffers will be freed when the pool is dropped, so any buffers created from this pool must be invalidated
        unsafe {
            self.device.handle.destroy_command_pool(self.pool, None);
        }
    }
}
