use std::sync::Arc;

use ash::vk::{self, DescriptorSetLayoutCreateInfo};

use crate::{core::Core, utility::macros::*, VkHandle};

use super::super::{
    buffer::{AsBuffer, Buffer},
    device::Device,
};

#[derive(Clone, Copy)]
pub enum DescriptorSetLayoutBinding {
    Storage { vis: vk::ShaderStageFlags },
    Uniform { vis: vk::ShaderStageFlags },
    Sampler { vis: vk::ShaderStageFlags },
    None,
}

impl DescriptorSetLayoutBinding {
    pub fn binding_type(&self) -> vk::DescriptorType {
        match self {
            DescriptorSetLayoutBinding::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
            DescriptorSetLayoutBinding::Uniform { .. } => vk::DescriptorType::UNIFORM_BUFFER,
            DescriptorSetLayoutBinding::Sampler { .. } => {
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            }
            DescriptorSetLayoutBinding::None => unreachable!(),
        }
    }
}

vk_device_owned_wrapper!(DescriptorPool, destroy_descriptor_pool);

pub struct DescriptorSet {
    handle: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    device: Arc<Device>,
    buffers: Vec<Arc<Buffer>>,
}
vk_handle_wrapper!(DescriptorSet);

pub struct DescriptorSetLayout {
    handle: vk::DescriptorSetLayout,
    device: Arc<Device>,
    layout: Vec<DescriptorSetLayoutBinding>,
}
vk_handle_wrapper!(DescriptorSetLayout);
vk_device_drop!(DescriptorSetLayout, destroy_descriptor_set_layout);

pub enum DescriptorWriteData {
    Buffer {
        buf: Arc<Buffer>,
    },
    Empty,
    //TODO:
    Image {
        view: vk::ImageView,
        sampler: vk::Sampler,
    },
}

impl DescriptorPool {
    pub fn new(device: Arc<Device>, swapchain_images_size: u32) -> Arc<DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                // transform descriptor pool
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: swapchain_images_size * 8,
            },
            vk::DescriptorPoolSize {
                // SSBO pool
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 16,
            },
            vk::DescriptorPoolSize {
                // sampler descriptor pool
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: swapchain_images_size,
            },
        ];

        Self::new_sized(device, &pool_sizes, swapchain_images_size * 3)
    }

    pub fn new_sized(
        device: Arc<Device>,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> Arc<DescriptorPool> {
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let handle = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("Failed to create Descriptor Pool!")
        };

        Arc::new(Self { handle, device })
    }

    pub fn alloc<Write: Fn(usize) -> Vec<DescriptorWriteData>>(
        self: &Arc<DescriptorPool>,
        descriptor_set_layout: &Arc<DescriptorSetLayout>,
        count: usize,
        write: Write,
    ) -> Vec<DescriptorSet> {
        let layouts = vec![descriptor_set_layout.handle(); count];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.handle())
            .set_layouts(&layouts);

        let vk_descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)
                .expect("Failed to allocate descriptor sets!")
        };

        vk_descriptor_sets
            .into_iter()
            .enumerate()
            .map(|(i, set)| {
                DescriptorSet::new(
                    set,
                    self.clone(),
                    descriptor_set_layout.clone(),
                    self.device.clone(),
                    write(i),
                )
            })
            .collect()
    }
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

        let image_info: Vec<_> = buffers
            .iter()
            .enumerate()
            .filter_map(|(i, w)| match w {
                DescriptorWriteData::Image { view, sampler } => Some(
                    *vk::DescriptorImageInfo::builder()
                        .image_view(*view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .sampler(*sampler),
                ),
                _ => None,
            })
            .collect();

        let mut image_index = 0;

        let descriptor_write_sets: Vec<_> = buffers
            .iter()
            .enumerate()
            .filter_map(|(i, w)| match w {
                DescriptorWriteData::Buffer { buf } => {
                    let layout = &layout.layout[i];

                    match layout {
                        DescriptorSetLayoutBinding::Storage { vis: _ } => {
                            assert!(buf.usage().contains(vk::BufferUsageFlags::STORAGE_BUFFER))
                        }
                        DescriptorSetLayoutBinding::Uniform { vis: _ } => {
                            assert!(buf.usage().contains(vk::BufferUsageFlags::UNIFORM_BUFFER))
                        }
                        DescriptorSetLayoutBinding::None => {
                            panic!("Attempted to write into empty binding slot")
                        }
                        DescriptorSetLayoutBinding::Sampler { vis } => {
                            panic!("Attempted to write buffer into sampler slot")
                        }
                    }

                    Some(buf.write_descriptor_sets(
                        handle,
                        layout.binding_type(),
                        descriptor_info[i].as_ref().unwrap(),
                        i as _,
                    ))
                }
                DescriptorWriteData::Image { .. } => {
                    let dsc_writes = vk::WriteDescriptorSet::builder()
                        .dst_set(handle)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .dst_array_element(0_u32)
                        .dst_binding(i as u32)
                        .image_info(&image_info[image_index..image_index + 1]);

                    image_index += 1;
                    Some(dsc_writes)
                }
                _ => None,
            })
            .collect();

        unsafe {
            let descriptor_writes: Vec<_> = descriptor_write_sets.into_iter().map(|x| *x).collect();
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        let buffers: Vec<_> = buffers
            .into_iter()
            .filter_map(|w| match w {
                DescriptorWriteData::Buffer { buf, .. } => Some(buf),
                _ => None,
            })
            .collect();

        Self {
            handle,
            pool,
            device,
            buffers,
        }
    }
}

fn binding_struct_to_vk(
    binding_struct: &[DescriptorSetLayoutBinding],
) -> Vec<vk::DescriptorSetLayoutBinding> {
    let mut bindings = vec![vk::DescriptorSetLayoutBinding::default(); binding_struct.len()];
    let mut i = 0;
    for (binding, &b) in binding_struct.iter().enumerate() {
        match b {
            DescriptorSetLayoutBinding::None => {
                bindings.swap_remove(i);
                continue;
            }

            DescriptorSetLayoutBinding::Storage { vis } => {
                bindings[i].stage_flags = vis;
                bindings[i].descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
            }
            DescriptorSetLayoutBinding::Uniform { vis } => {
                bindings[i].stage_flags = vis;
                bindings[i].descriptor_type = vk::DescriptorType::UNIFORM_BUFFER;
            }
            DescriptorSetLayoutBinding::Sampler { vis } => {
                bindings[i].stage_flags = vis;
                bindings[i].descriptor_type = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            }
        }

        bindings[i].binding = binding as _;
        bindings[i].descriptor_count = 1;

        i += 1;
    }
    bindings
}

impl DescriptorSetLayout {
    pub fn new(core: &Core, layout: Vec<DescriptorSetLayoutBinding>, name: &str) -> Self {
        let bindings = binding_struct_to_vk(&layout);

        let handle = unsafe {
            core.device
                .create_descriptor_set_layout(
                    &DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
                    None,
                )
                .expect("Failed to create Descriptor Set Layout!")
        };

        core.name_object(name, handle);

        Self {
            handle,
            device: core.device.clone(),
            layout,
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .free_descriptor_sets(self.pool.handle, &[self.handle])
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_binding_struct() {
        let ubo_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                // transform uniform
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
                p_immutable_samplers: ptr::null(),
            },
            // vk::DescriptorSetLayoutBinding {
            //     // sampler uniform
            //     binding: 1,
            //     descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            //     descriptor_count: 1,
            //     stage_flags: vk::ShaderStageFlags::FRAGMENT,
            //     p_immutable_samplers: ptr::null(),
            // },
            vk::DescriptorSetLayoutBinding {
                // sampler uniform
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::TASK_EXT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                // camera uniform
                binding: 3,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
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
                // meshlet buffer
                binding: 5,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::MESH_EXT,
                p_immutable_samplers: ptr::null(),
            },
            vk::DescriptorSetLayoutBinding {
                // indirect draw params buffer array
                binding: 6,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::TASK_EXT,
                p_immutable_samplers: ptr::null(),
            },
        ];

        let bindings = [
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
            },
            DescriptorSetLayoutBinding::None,
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::TASK_EXT,
            },
            DescriptorSetLayoutBinding::Uniform {
                vis: vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT,
            },
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::MESH_EXT,
            },
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::MESH_EXT,
            },
            DescriptorSetLayoutBinding::Storage {
                vis: vk::ShaderStageFlags::TASK_EXT,
            },
        ];

        let b = binding_struct_to_vk(&bindings);

        assert_eq!(b[1].binding, 2);

        println!("{:#?}", b);

        for (&x, &y) in b.iter().zip(ubo_layout_bindings.iter()) {
            assert_eq!(x.binding, y.binding);
            assert_eq!(x.stage_flags, y.stage_flags);
            assert_eq!(x.descriptor_type, y.descriptor_type);
            assert_eq!(x.descriptor_count, y.descriptor_count);
        }
    }
}
