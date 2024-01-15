use std::{ptr, sync::Arc};

use ash::vk;

use crate::{CameraUniformBufferObject, VkHandle};

use super::{
    buffer::{AsBuffer, Buffer},
    device::Device,
    image::Image,
    structures::ModelUniformBufferObject,
};

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
                descriptor_count: swapchain_images_size * 2,
            },
            vk::DescriptorPoolSize {
                // SSBO pool
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 3,
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

pub struct DescriptorSet {
    set: vk::DescriptorSet,
    device: Arc<Device>,
    buffers: Vec<Arc<Buffer>>,
}

impl VkHandle for DescriptorSet {
    type VkItem = vk::DescriptorSet;

    fn handle(&self) -> Self::VkItem {
        self.set
    }
}

impl DescriptorSet {
    pub fn create_descriptor_sets(
        device: &Arc<Device>,
        descriptor_pool: &DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_transform_buffer: &Arc<Buffer>,
        uniform_camera_buffers: &[impl AsBuffer],
        vertex_buffer: &Arc<Buffer>,
        meshlet_buffer: &Arc<Buffer>,
        submesh_buffer: &Arc<Buffer>,
        texture: &Image,
        swapchain_images_size: usize,
    ) -> Vec<DescriptorSet> {
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

        let vk_descriptor_sets = unsafe {
            device
                .handle
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        let descriptor_sets: Vec<_> = vk_descriptor_sets
            .into_iter()
            .map(|set| DescriptorSet {
                set,
                device: device.clone(),
                buffers: vec![
                    vertex_buffer.clone(),
                    uniform_transform_buffer.clone(),
                    meshlet_buffer.clone(),
                    submesh_buffer.clone(),
                ],
            })
            .collect();

        for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
            let descriptor_transform_buffer_infos =
                [uniform_transform_buffer.full_range_descriptor()];
            let descriptor_camera_buffer_infos =
                [uniform_camera_buffers[i].full_range_descriptor()];

            let vertex_buffer_infos = [vertex_buffer.full_range_descriptor()];
            let index_buffer_infos = [meshlet_buffer.full_range_descriptor()];
            let submesh_buffer_infos = [submesh_buffer.full_range_descriptor()];

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
                    dst_set: descriptor_set.set,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: descriptor_transform_buffer_infos.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // transform uniform
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set.set,
                    dst_binding: 5,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: descriptor_camera_buffer_infos.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // sampler uniform
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set.set,
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: descriptor_image_infos.as_ptr(),
                    p_buffer_info: ptr::null(),
                    p_texel_buffer_view: ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // submesh info buffer
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set.set,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: submesh_buffer_infos.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // meshlet info buffer
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set.set,
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: index_buffer_infos.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                },
                vk::WriteDescriptorSet {
                    // vertex buffer
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    p_next: ptr::null(),
                    dst_set: descriptor_set.set,
                    dst_binding: 4,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_image_info: ptr::null(),
                    p_buffer_info: vertex_buffer_infos.as_ptr(),
                    p_texel_buffer_view: ptr::null(),
                },
            ];

            unsafe {
                device
                    .handle
                    .update_descriptor_sets(&descriptor_write_sets, &[]);
            }
        }

        descriptor_sets
    }
}
pub fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let ubo_layout_bindings = [
        vk::DescriptorSetLayoutBinding {
            // transform uniform
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
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
        vk::DescriptorSetLayoutBinding {
            // sampler uniform
            binding: 2,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::TASK_EXT,
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
            // verts buffer
            binding: 4,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT,
            p_immutable_samplers: ptr::null(),
        },
        vk::DescriptorSetLayoutBinding {
            // camera uniform
            binding: 5,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
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
