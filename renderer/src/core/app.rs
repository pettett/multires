use crate::components::{
    camera_uniform::{update_view_proj, CameraUniform},
    mesh::Mesh,
};
use bevy_ecs::{
    event::{Event, Events},
    schedule::Schedule,
    world::{Mut, World},
};
use common_renderer::components::{
    camera::Camera,
    camera_controller::{
        camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
    },
    transform::Transform,
};
use glam::{Quat, Vec3A};
use winit::event::WindowEvent;

use super::{
    renderer::{handle_screen_events, render},
    Renderer,
};

#[derive(Event)]
pub enum ScreenEvent {
    Resize(winit::dpi::PhysicalSize<u32>),
}

pub struct App {
    world: World,
    window_schedule: Schedule,
    redraw_schedule: Schedule,
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
        world.insert_resource(Events::<ScreenEvent>::default());

        let mesh = Mesh::load_mesh(renderer.instance());

        world.spawn(mesh);
        world.spawn((
            CameraController::new(0.05),
            Camera::new(1.0),
            Transform::new(Vec3A::ZERO, Quat::IDENTITY),
            CameraUniform::new(),
        ));

        world.insert_resource(renderer);

        let mut window_schedule = Schedule::default();
        window_schedule.add_systems((handle_screen_events, camera_handle_input));

        let mut redraw_schedule = Schedule::default();
        redraw_schedule.add_systems((update_camera, update_view_proj, render));

        Self {
            world,
            window_schedule,
            redraw_schedule,
        }
    }

    pub fn redraw(&mut self) {
        self.redraw_schedule.run(&mut self.world);
    }

    pub fn update(&mut self) {
        self.window_schedule.run(&mut self.world);
    }

    pub fn renderer(&self) -> &Renderer {
        &self.world.get_resource().as_ref().unwrap()
    }

    // impl State
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.world.send_event(ScreenEvent::Resize(new_size));
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
