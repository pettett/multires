use std::sync::{Arc, Mutex};

use bevy_ecs::{
    event::Events,
    schedule::Schedule,
    system::Query,
    world::{Mut, World},
};
use glam::{Quat, Vec3A};
use winit::event::WindowEvent;

use crate::components::{
    camera::Camera,
    camera_controller::{
        camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
    },
    camera_uniform::{update_view_proj, CameraUniform},
    mesh::Mesh,
    transform::Transform,
};

use super::{renderer::render, Renderer};

pub struct App {
    world: World,
    schedule: Schedule,
    //puffin_ui : puffin_imgui::ProfilerUi,
}

impl App {
    /// Creating some of the wgpu types requires async code
    /// https://sotrh.github.io/learn-wgpu/beginner/tutorial2-surface/#state-new
    pub async fn new(renderer: Renderer) -> Self {
        let mut world = World::new();

        world.insert_resource(Events::<MouseIn>::default());
        world.insert_resource(Events::<MouseMv>::default());
        world.insert_resource(Events::<KeyIn>::default());

        let mesh = Mesh::load_mesh(renderer.instance());

        world.spawn(mesh);
        world.spawn((
            CameraController::new(0.5),
            Camera::new(1.0),
            Transform::new(Vec3A::ZERO, Quat::IDENTITY),
            CameraUniform::new(),
        ));

        let mut schedule = Schedule::default();

        world.insert_resource(renderer);

        // Add our system to the schedule
        schedule.add_systems((camera_handle_input, update_camera, update_view_proj, render));

        Self { world, schedule }
    }

    pub fn render(&mut self) {
        self.schedule.run(&mut self.world);
    }

    pub fn renderer(&self) -> &Renderer {
        &self.world.get_resource().as_ref().unwrap()
    }

    // impl State
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let mut r: Mut<'_, Renderer> = self.world.get_resource_mut().unwrap();
            r.resize(new_size);
        }
    }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.renderer().size()
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::MouseInput { state, button, .. } => self
                .world
                .send_event(MouseIn(state.clone(), button.clone())),
            WindowEvent::KeyboardInput { input, .. } => self.world.send_event(KeyIn(input.clone())),
            WindowEvent::CursorMoved { position, .. } => {
                self.world.send_event(MouseMv(position.clone()))
            }
            _ => (),
        }

        false
    }
}
