use std::{ptr, sync::Arc};

use ash::vk::{self, DescriptorSetLayoutCreateInfo};

use crate::{CameraUniformBufferObject, VkHandle};

use super::{
    buffer::{AsBuffer, Buffer},
    device::Device,
    image::Image,
    structures::ModelUniformBufferObject,
};

pub struct DescriptorPool {
    device: Arc<Device>,
    pub handle: vk::DescriptorPool,
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

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(swapchain_images_size as u32)
            //.flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
			;

        let pool = unsafe {
            device
                .handle
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        };

        Arc::new(Self {
            handle: pool,
            device,
        })
    }
}

pub struct DescriptorSet {
    handle: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    device: Arc<Device>,
    buffers: Vec<Arc<Buffer>>,
}
impl VkHandle for DescriptorSet {
    type VkItem = vk::DescriptorSet;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}
pub struct DescriptorSetLayout {
    handle: vk::DescriptorSetLayout,
    device: Arc<Device>,
}

impl VkHandle for DescriptorSetLayout {
    type VkItem = vk::DescriptorSetLayout;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl DescriptorSet {
    pub fn create_descriptor_sets(
        device: &Arc<Device>,
        descriptor_pool: &Arc<DescriptorPool>,
        descriptor_set_layout: &Arc<DescriptorSetLayout>,
        uniform_transform_buffer: &Arc<Buffer>,
        uniform_camera_buffers: &[impl AsBuffer],
        vertex_buffer: &Arc<Buffer>,
        meshlet_buffer: &Arc<Buffer>,
        submesh_buffer: &Arc<Buffer>,
        indirect_draw_array_buffer: &Arc<impl AsBuffer>,
        //texture: &Image,
        swapchain_images_size: usize,
    ) -> Vec<DescriptorSet> {
        let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..swapchain_images_size {
            layouts.push(descriptor_set_layout.handle());
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool.handle)
            .set_layouts(&layouts);

        let vk_descriptor_sets = unsafe {
            device
                .handle
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        let descriptor_sets: Vec<_> = vk_descriptor_sets
            .into_iter()
            .map(|set| DescriptorSet {
                handle: set,
                device: device.clone(),
                buffers: vec![
                    vertex_buffer.clone(),
                    uniform_transform_buffer.clone(),
                    meshlet_buffer.clone(),
                    submesh_buffer.clone(),
                ],
                pool: descriptor_pool.clone(),
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
            let indirect_draw_array_buffer_infos =
                [indirect_draw_array_buffer.full_range_descriptor()];

            // let descriptor_image_infos = [vk::DescriptorImageInfo {
            //     sampler: texture.sampler(),
            //     image_view: texture.image_view(),
            //     image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            // }];

            let descriptor_write_sets = [
                vk::WriteDescriptorSet {
                    // transform uniform
                    dst_set: descriptor_set.handle,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: descriptor_transform_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    // transform uniform
                    dst_set: descriptor_set.handle,
                    dst_binding: 5,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,

                    p_buffer_info: descriptor_camera_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                // vk::WriteDescriptorSet {
                //     // sampler uniform
                //     dst_set: descriptor_set.handle,
                //     dst_binding: 1,
                //     dst_array_element: 0,
                //     descriptor_count: 1,
                //     descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                //     p_image_info: descriptor_image_infos.as_ptr(),
                //     ..Default::default()
                // },
                vk::WriteDescriptorSet {
                    // submesh info buffer
                    dst_set: descriptor_set.handle,
                    dst_binding: 2,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: submesh_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    // meshlet info buffer
                    dst_set: descriptor_set.handle,
                    dst_binding: 3,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,

                    p_buffer_info: index_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    // vertex buffer
                    dst_set: descriptor_set.handle,
                    dst_binding: 4,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: vertex_buffer_infos.as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    // vertex buffer
                    dst_set: descriptor_set.handle,
                    dst_binding: 6,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    p_buffer_info: indirect_draw_array_buffer_infos.as_ptr(),
                    ..Default::default()
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

impl DescriptorSetLayout {
    pub fn new(device: Arc<Device>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Self {
        Self {
            handle: unsafe {
                device
                    .handle
                    .create_descriptor_set_layout(
                        &DescriptorSetLayoutCreateInfo::builder().bindings(bindings),
                        None,
                    )
                    .expect("Failed to create Descriptor Set Layout!")
            },
            device,
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_pool(self.handle, None);
        }
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}
//FIXME:
// impl Drop for DescriptorSet {
//     fn drop(&mut self) {
//         unsafe {
//             self.device
//                 .handle
//                 .free_descriptor_sets(self.pool.handle, &[self.handle])
//                 .unwrap();
//         }
//     }
// }
