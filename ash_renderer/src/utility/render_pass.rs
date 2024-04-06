use std::{ptr, sync::Arc};

use ash::vk::{self, AttachmentDescription};

use crate::VkHandle;

use super::{device::Device, macros::vk_device_owned_wrapper};

vk_device_owned_wrapper!(RenderPass, destroy_render_pass);

impl RenderPass {
    pub fn new_color(device: Arc<Device>, surface_format: vk::Format) -> RenderPass {
        let handle = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&[vk::AttachmentDescription::default()
                        .format(surface_format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                        ])
                    .subpasses(&[vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[vk::AttachmentReference::default()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            ])
                        ])
                    .dependencies(&[vk::SubpassDependency::default()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        ]),
                None,
            )
        }
        .expect("Failed to create render pass.");

        RenderPass { device, handle }
    }

    pub fn new_color_depth(
        device: Arc<Device>,
        surface_format: vk::Format,
        depth_format: vk::Format,
        final_layout: vk::ImageLayout,
    ) -> Self {
        let color_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            // Pass forward to the ui stage
            final_layout,
        };

        let depth_attachment = vk::AttachmentDescription {
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: depth_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let render_pass_attachments = [color_attachment, depth_attachment];

        let color_attachment_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpasses = [vk::SubpassDescription::default()
            .color_attachments(&color_attachment_ref)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];

        let subpass_dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        }];

        let renderpass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&render_pass_attachments)
            .subpasses(&subpasses)
            .dependencies(&subpass_dependencies);

        Self::new(device, renderpass_create_info)
    }

    fn new(
        device: Arc<Device>,
        render_pass_create_info: vk::RenderPassCreateInfo<'_>,
    ) -> Self {
        let handle = unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .expect("Failed to create render pass!")
        };
        Self { device, handle }
    }
}
