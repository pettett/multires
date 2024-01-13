use std::sync::Arc;

use crate::{
    components::{
        camera_uniform::{update_view_proj, CameraUniform},
        gpu_multi_res_mesh::{
            DrawMode, MultiResMeshAsset, MultiResMeshComponent, MultiResMeshRenderer,
        },
    },
    gui::gui::Gui,
};
use bevy_ecs::{
    event::{Event, Events},
    schedule::Schedule,
    system::{NonSend, NonSendMut, SystemState},
    world::World,
};
use common_renderer::components::{
    camera::Camera,
    camera_controller::{
        camera_handle_input, update_camera, CameraController, KeyIn, MouseIn, MouseMv,
    },
    transform::Transform,
};
use glam::{Quat, Vec3, Vec3A};
use winit::event::{KeyboardInput, WindowEvent};

use super::{
    renderer::{handle_screen_events, render},
    Renderer,
};

#[derive(Event)]
pub enum ScreenEvent {
    Resize(winit::dpi::PhysicalSize<u32>),
    Key(KeyboardInput),
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

        let mesh_renderer = MultiResMeshRenderer::new(&renderer);

        let asset = Arc::new(MultiResMeshAsset::load_mesh(
            renderer.instance(),
            &mesh_renderer,
        ));

        for i in 0..10 {
            for j in 0..10 {
                MultiResMeshComponent::from_asset(
                    "Mesh 0".to_owned(),
                    renderer.instance(),
                    &mesh_renderer,
                    &mut world,
                    asset.clone(),
                    Transform::new_pos(Vec3A::X * 30.0 * i as f32 + Vec3A::Y * 30.0 * j as f32),
                );
            }
        }

        world.insert_resource(mesh_renderer);

        //GUI state
        let q0 = world.query();
        let q1 = world.query();
        world.insert_resource(Gui::init(&renderer, q0, q1, &world));

        world.insert_non_send_resource(egui::Context::default());
        world.insert_non_send_resource(egui_winit::State::new(renderer.window()));

        world.spawn((
            CameraController::new(0.03),
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
        {
            // Pass to GUI
            let mut state =
                SystemState::<(NonSend<egui::Context>, NonSendMut<egui_winit::State>)>::new(
                    &mut self.world,
                );

            let (context, mut state) = state.get_mut(&mut self.world);

            if state.on_event(&context, event).consumed {
                return true;
            }
        }

        match event {
            WindowEvent::MouseInput { state, button, .. } => self
                .world
                .send_event(MouseIn(state.clone(), button.clone())),
            WindowEvent::KeyboardInput { input, .. } => {
                self.world.send_event(KeyIn(input.clone()));
                self.world.send_event(ScreenEvent::Key(input.clone()))
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.world.send_event(MouseMv(position.clone()))
            }
            _ => (),
        }

        false
    }
}
