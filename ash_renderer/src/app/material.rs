use std::ffi;
use std::sync::Arc;

use crate::utility::ShaderModule;
use crate::VkHandle;
use ash::vk;
use bevy_ecs::event::Event;

pub const MAIN_FUNCTION_NAME: &ffi::CStr = c"main"; // the beginning function name in shader code.

#[derive(Debug, Clone, Copy, Event, PartialEq)]
pub enum Fragment {
    VertexColour,
    Lit,
    Edges,
}

pub struct Material {
    fragment: Arc<ShaderModule>,
    polygon_mode: vk::PolygonMode,
}

impl Material {
    pub fn new(fragment: Arc<ShaderModule>, polygon_mode: vk::PolygonMode) -> Self {
        Self {
            fragment,
            polygon_mode,
        }
    }

    pub fn shader_stage_create_info(&self) -> vk::PipelineShaderStageCreateInfo<'static> {
        vk::PipelineShaderStageCreateInfo::default()
            .module(self.fragment.handle())
            .name(&MAIN_FUNCTION_NAME)
            .stage(vk::ShaderStageFlags::FRAGMENT)
    }

    pub fn rasterization_statue_create_info(&self) -> vk::PipelineRasterizationStateCreateInfo {
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0)
            .polygon_mode(self.polygon_mode)
            .rasterizer_discard_enable(false)
            .depth_bias_enable(false)
    }
}
