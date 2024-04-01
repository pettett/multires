use ash::vk;

use crate::{utility::image::Image, VkHandle};

pub fn insert_image_memory_barrier(
    device: &ash::Device,
    cmd_buff: &ash::vk::CommandBuffer,
    image: &mut Image,
    src_q_family_index: u32,
    dst_q_family_index: u32,
    src_access_mask: ash::vk::AccessFlags2,
    dst_access_mask: ash::vk::AccessFlags2,
    new_image_layout: ash::vk::ImageLayout,
    src_stage_mask: ash::vk::PipelineStageFlags2,
    dst_stage_mask: ash::vk::PipelineStageFlags2,
    subresource_range: ash::vk::ImageSubresourceRange,
) {
    let image_memory_barrier = [vk::ImageMemoryBarrier2::builder()
        .src_queue_family_index(src_q_family_index)
        .dst_queue_family_index(dst_q_family_index)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .src_stage_mask(src_stage_mask)
        .dst_stage_mask(dst_stage_mask)
        .old_layout(image.current_layout())
        .new_layout(new_image_layout)
        .image(image.handle())
        .subresource_range(subresource_range)
        .build()];

    image.set_current_layout(new_image_layout);

    let dep_info = vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barrier);

    unsafe {
        device.cmd_pipeline_barrier2(*cmd_buff, &dep_info);
    }
}