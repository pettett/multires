use std::{
    cmp::max,
    mem,
    path::Path,
    ptr,
    sync::{Arc, Mutex},
};

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::{core::Core, VkHandle};

use super::{
    buffer::{AsBuffer, Buffer, STAGING_BUFFER},
    device::Device,
    instance::Instance,
    macros::vk_handle_wrapper,
    pooled::command_pool::CommandPool,
};

pub struct Image {
    device: Arc<Device>,
    handle: vk::Image,
    image_memory: Allocation,
    format: vk::Format,
    allocator: Arc<Mutex<Allocator>>,
    image_view: Option<vk::ImageView>,
    sampler: Option<vk::Sampler>,
}

vk_handle_wrapper!(Image);

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if let Some(sampler) = self.sampler {
                self.device.handle.destroy_sampler(sampler, None);
            }
            if let Some(image_view) = self.image_view {
                self.device.handle.destroy_image_view(image_view, None);
            }
            self.device.handle.destroy_image(self.handle, None);

            let mut allocation = Default::default();

            mem::swap(&mut self.image_memory, &mut allocation);

            self.allocator.lock().unwrap().free(allocation).unwrap();
        }
    }
}

impl Image {
    pub fn create_image(
        core: &Core,
        width: u32,
        height: u32,
        mip_levels: u32,
        num_samples: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        location: MemoryLocation,
        allocator: Arc<Mutex<Allocator>>,
        name: &str,
    ) -> Image {
        let image_create_info = vk::ImageCreateInfo {
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type: vk::ImageType::TYPE_2D,
            format,
            mip_levels,
            array_layers: 1,
            samples: num_samples,
            tiling,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        let image = unsafe {
            core.device
                .handle
                .create_image(&image_create_info, None)
                .expect("Failed to create Texture Image!")
        };

        core.name_object(name, image);

        let requirements = unsafe { core.device.handle.get_image_memory_requirements(image) };

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
                .handle
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .expect("Failed to bind Image Memmory!");
        }

        Image {
            device: core.device.clone(),
            handle: image,
            image_memory: allocation,
            allocator,
            format,
            image_view: None,
            sampler: None,
        }
    }

    pub fn create_texture_image(
        core: &Core,

        allocator: Arc<Mutex<Allocator>>,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        image_path: &Path,
        name: &str,
    ) -> Self {
        // let mut image_object = image::open(image_path).unwrap(); // this function is slow in debug mode.
        // image_object = image_object.flipv();
        // let (image_width, image_height) = (image_object.width(), image_object.height());
        // let image_data = match &image_object {
        //     image::DynamicImage::ImageBgr8(_)
        //     | image::DynamicImage::ImageLuma8(_)
        //     | image::DynamicImage::ImageRgb8(_) => image_object.to_rgba().into_raw(),
        //     image::DynamicImage::ImageBgra8(_)
        //     | image::DynamicImage::ImageLumaA8(_)
        //     | image::DynamicImage::ImageRgba8(_) => image_object.raw_pixels(),
        // };

        let image_width = 2;
        let image_height = 1;
        let image_data = vec![0, 255, 0, 0, 0, 0, 255, 0];

        let image_size =
            (::std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;

        if image_size <= 0 {
            panic!("Failed to load texture image!")
        }

        let staging_buffer = Buffer::new(
            core,
            allocator.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
            STAGING_BUFFER,
        );

        unsafe {
            let data_ptr = staging_buffer
                .allocation()
                .mapped_ptr()
                .expect("Failed to Map Memory")
                .as_ptr() as *mut u8;

            data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());
        }

        let texture_image = Self::create_image(
            core,
            image_width,
            image_height,
            1,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            MemoryLocation::GpuOnly,
            allocator,
            name,
        );

        Self::transition_image_layout(
            &core.device,
            command_pool,
            submit_queue,
            &texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            1,
        );

        copy_buffer_to_image(
            &core.device,
            command_pool,
            submit_queue,
            staging_buffer.handle(),
            texture_image.image(),
            image_width,
            image_height,
        );

        Self::transition_image_layout(
            &core.device,
            command_pool,
            submit_queue,
            &texture_image,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
        );

        texture_image
    }

    pub fn create_texture_sampler(mut self) -> Self {
        let sampler_create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            min_lod: 0.0,
            max_lod: 0.0,
            mip_lod_bias: 0.0,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            anisotropy_enable: vk::TRUE,
            unnormalized_coordinates: vk::FALSE,
        };

        unsafe {
            self.sampler = Some(
                self.device
                    .handle
                    .create_sampler(&sampler_create_info, None)
                    .expect("Failed to create Sampler!"),
            );
        }
        self
    }

    pub fn create_texture_image_view(self, mip_levels: u32) -> Self {
        self.create_image_view(
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )
    }

    pub fn transition_image_layout(
        device: &Device,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        image: &Image,
        _format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let command_buffer = command_pool.begin_single_time_command(submit_queue);

        let src_access_mask;
        let dst_access_mask;
        let source_stage;
        let destination_stage;

        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dst_access_mask = vk::AccessFlags::SHADER_READ;
            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        {
            src_access_mask = vk::AccessFlags::empty();
            dst_access_mask =
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        } else {
            panic!("Unsupported layout transition!")
        }

        let image_barriers = [vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask,
            dst_access_mask,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: image.handle,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
        }];

        unsafe {
            device.handle.cmd_pipeline_barrier(
                command_buffer.handle,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_barriers,
            );
        }
    }

    pub fn create_depth_resources(
        core: &Core,
        physical_device: vk::PhysicalDevice,
        swapchain_extent: vk::Extent2D,
        allocator: Arc<Mutex<Allocator>>,
        msaa_samples: vk::SampleCountFlags,
        name: &str,
    ) -> Self {
        let depth_format = core.instance.find_depth_format(physical_device);
        Self::create_image(
            core,
            swapchain_extent.width,
            swapchain_extent.height,
            1,
            msaa_samples,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            MemoryLocation::GpuOnly,
            allocator,
            name,
        )
        .create_image_view(depth_format, vk::ImageAspectFlags::DEPTH, 1)
    }

    pub fn generate_mipmaps(
        device: &Device,
        command_pool: &Arc<CommandPool>,
        submit_queue: vk::Queue,
        image: vk::Image,
        tex_width: u32,
        tex_height: u32,
        mip_levels: u32,
    ) {
        let command_buffer = command_pool.begin_single_time_command(submit_queue);

        let mut image_barrier = vk::ImageMemoryBarrier {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::empty(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::UNDEFINED,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };

        let mut mip_width = tex_width as i32;
        let mut mip_height = tex_height as i32;

        for i in 1..mip_levels {
            image_barrier.subresource_range.base_mip_level = i - 1;
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                device.handle.cmd_pipeline_barrier(
                    command_buffer.handle,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            let blits = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: max(mip_width / 2, 1),
                        y: max(mip_height / 2, 1),
                        z: 1,
                    },
                ],
            }];

            unsafe {
                device.handle.cmd_blit_image(
                    command_buffer.handle,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );
            }

            image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                device.handle.cmd_pipeline_barrier(
                    command_buffer.handle,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
            }

            mip_width = max(mip_width / 2, 1);
            mip_height = max(mip_height / 2, 1);
        }

        image_barrier.subresource_range.base_mip_level = mip_levels - 1;
        image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.handle.cmd_pipeline_barrier(
                command_buffer.handle,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[image_barrier.clone()],
            );
        }
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler.unwrap()
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.image_view.unwrap()
    }

    pub fn create_image_views(
        device: &Device,
        surface_format: vk::Format,
        images: &Vec<vk::Image>,
    ) -> Vec<vk::ImageView> {
        let swapchain_imageviews: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                Self::create_raw_image_view(
                    device,
                    image,
                    surface_format,
                    vk::ImageAspectFlags::COLOR,
                    1,
                )
            })
            .collect();

        swapchain_imageviews
    }

    fn create_raw_image_view(
        device: &Device,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let imageview_create_info = vk::ImageViewCreateInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            image,
            ..Default::default()
        };

        unsafe {
            device
                .handle
                .create_image_view(&imageview_create_info, None)
                .expect("Failed to create Image View!")
        }
    }

    pub fn create_image_view(
        mut self,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Self {
        unsafe {
            self.image_view = Some(Image::create_raw_image_view(
                &self.device,
                self.handle,
                format,
                aspect_flags,
                mip_levels,
            ));
        }
        self
    }

    pub fn image(&self) -> vk::Image {
        self.handle
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }
}

pub fn copy_buffer_to_image(
    device: &Device,
    command_pool: &Arc<CommandPool>,
    submit_queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    let command_buffer = command_pool.begin_single_time_command(submit_queue);

    let buffer_image_regions = [vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        buffer_offset: 0,
        buffer_image_height: 0,
        buffer_row_length: 0,
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
    }];

    unsafe {
        device.handle.cmd_copy_buffer_to_image(
            command_buffer.handle,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &buffer_image_regions,
        );
    }
}

pub fn has_stencil_component(format: vk::Format) -> bool {
    format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
}
