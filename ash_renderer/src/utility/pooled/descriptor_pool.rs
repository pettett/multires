use std::{ptr, sync::Arc};

use ash::vk::{self, DescriptorSetLayoutCreateInfo};
use egui::ahash::HashMap;

use crate::{VkHandle};

use super::super::{
    buffer::{AsBuffer, Buffer},
    device::Device, 
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
                descriptor_count: swapchain_images_size * 4,
            },
            vk::DescriptorPoolSize {
                // SSBO pool
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 8,
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
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

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
    types: HashMap<usize, vk::DescriptorType>,
}

impl VkHandle for DescriptorSetLayout {
    type VkItem = vk::DescriptorSetLayout;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

pub enum DescriptorWriteData {
    Buffer { buf: Arc<Buffer> },
    Empty,
    //TODO:
    Image,
}

impl DescriptorSet {
    pub fn new(
        handle: vk::DescriptorSet,
        pool: Arc<DescriptorPool>,
        layout: Arc<DescriptorSetLayout>,
        device: Arc<Device>,
        buffers: Vec<DescriptorWriteData>,
    ) -> Self {
        let descriptor_info: Vec<_> = buffers
            .iter()
            .map(|w| match w {
                DescriptorWriteData::Buffer { buf, .. } => Some([buf.full_range_descriptor()]),
                _ => None,
            })
            .collect();

        let descriptor_write_sets: Vec<_> = buffers
            .iter()
            .enumerate()
            .filter_map(|(i, w)| match w {
                DescriptorWriteData::Buffer { buf } => Some(buf.write_descriptor_sets(
                    handle,
                    layout.types[&i],
                    descriptor_info[i].as_ref().unwrap(),
                    i as _,
                )),
                _ => None,
            })
            .collect();

        let buffers: Vec<_> = buffers
            .into_iter()
            .filter_map(|w| match w {
                DescriptorWriteData::Buffer { buf, .. } => Some(buf),
                _ => None,
            })
            .collect();

        unsafe {
            device
                .handle
                .update_descriptor_sets(&descriptor_write_sets, &[]);
        }

        Self {
            handle,
            pool,
            device,
            buffers,
        }
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
            types: bindings
                .iter()
                .map(|b| (b.binding as _, b.descriptor_type))
                .collect(),
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
impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .free_descriptor_sets(self.pool.handle, &[self.handle])
                .unwrap();
        }
    }
}
