use std::{
    sync::{Arc, Mutex, MutexGuard},
    time,
};

use ash::vk;
use bevy_ecs::prelude::*;
use common_renderer::components::camera::Camera;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    draw_pipelines::{indirect_tasks::MeshShaderMode, DrawPipeline},
    gui::{gui::Gui, window::GuiWindow},
    utility::{
        pooled::{
            command_pool::CommandBuffer,
            descriptor_pool::DescriptorPool,
            query_pool::{PrimitivesQueryResults, QueryPool, TypelessQueryPool},
        },
        render_pass::RenderPass,
        screen::Screen,
        sync::SyncObjects,
        ShaderModule,
    },
};

use super::{frame_measure::RollingMeasure, mesh_data::MeshData, scene::Scene};
#[derive(Debug, Clone, Copy, Event, PartialEq, Eq)]
pub enum MeshDrawingPipelineType {
    DrawIndirect,
    DrawLOD,
    LocalSelectMesh,
    LocalSelectIndices,
    ExpandingComputeCulledMesh,
    ExpandingComputeCulledIndices,
    None,
}

#[derive(Debug, Clone, Copy, Event, PartialEq)]
pub enum Fragment {
    VertexColour,
    Lit,
}

#[derive(Resource)]
pub struct Renderer {
    pub allocator: Arc<Mutex<Allocator>>,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub draw_pipeline: Box<dyn DrawPipeline + Sync + Send>,
    pub descriptor_pool: Arc<DescriptorPool>,
    pub windows: Vec<Box<dyn GuiWindow + Sync + Send>>,
    pub current_pipeline: MeshDrawingPipelineType,
    pub mesh_mode: MeshShaderMode,
    pub sync_objects: SyncObjects,

    // Evaluation data
    pub last_sample: time::Instant,
    pub primitives: RollingMeasure<u32, 60>,
    pub query_primitives: QueryPool<PrimitivesQueryResults>,
    pub query: bool,

    pub render_gui: bool,
    pub current_frame: usize,
    pub image_index: usize,
    pub is_suboptimal: bool,
    pub is_framebuffer_resized: bool,
    pub app_info_open: bool,
    // Make sure to drop the core last
    pub screen: Screen,
    pub core: Arc<Core>,

    pub fragment_colour: ShaderModule,
    pub fragment_lit: ShaderModule,

    pub mesh: String,

    pub fragment: Fragment,
    pub hacky_command_buffer_passthrough: Option<vk::CommandBuffer>,
}

impl Renderer {
    pub fn recreate_swapchain(&mut self, gui: &mut Gui, cam: &mut Camera) {
        self.core.device.wait_device_idle();

        let size = self.window().inner_size();

        cam.on_resize(&size);

        //self.cleanup_swapchain();

        self.screen.remake_swapchain(
            self.graphics_queue,
            &self.render_pass,
            self.allocator.clone(),
        );

        self.draw_pipeline
            .init_swapchain(&self.core, &self.screen, &self.render_pass);

        // Egui Integration
        gui.update_swapchain(self.screen.swapchain());
    }

    pub fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    pub fn fragment(&self) -> &ShaderModule {
        match self.fragment {
            Fragment::VertexColour => &self.fragment_colour,
            Fragment::Lit => &self.fragment_lit,
        }
    }

    pub fn window(&self) -> &winit::window::Window {
        &self.core.window
    }
    pub fn get_allocator(&self) -> MutexGuard<Allocator> {
        self.allocator.lock().unwrap()
    }

    pub fn get_query(&self) -> Option<Arc<TypelessQueryPool>> {
        self.query.then(|| self.query_primitives.typeless())
    }
    pub fn get_query_ref(&self) -> Option<&TypelessQueryPool> {
        self.query.then(|| self.query_primitives.typeless_ref())
    }
}
