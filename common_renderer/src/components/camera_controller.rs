use crate::{components::transform::Transform, resources::time::Time};
use bevy_ecs::prelude::*;
use glam::{Vec2};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode},
};
#[derive(Component)]
pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    dragging: bool,
    last_mouse_pos: PhysicalPosition<f64>,
    mouse_delta: Vec2,
}
#[derive(Event)]
pub struct KeyIn(pub KeyboardInput);
#[derive(Event)]
pub struct MouseIn(pub ElementState, pub MouseButton);

#[derive(Event)]
pub struct MouseMv(pub PhysicalPosition<f64>);

pub fn camera_handle_input(
    mut controllers: Query<(&mut CameraController,)>,
    mut ev_keyin: EventReader<KeyIn>,
    mut ev_mousein: EventReader<MouseIn>,
    mut ev_mousemv: EventReader<MouseMv>,
) {
    for KeyIn(input) in ev_keyin.read() {
        let KeyboardInput {
            state,
            virtual_keycode: Some(keycode),
            ..
        } = input
        else {
            continue;
        };

        for (mut c,) in controllers.iter_mut() {
            let is_pressed = *state == ElementState::Pressed;
            match keycode {
                VirtualKeyCode::W | VirtualKeyCode::Up => c.is_forward_pressed = is_pressed,
                VirtualKeyCode::A | VirtualKeyCode::Left => c.is_left_pressed = is_pressed,
                VirtualKeyCode::S | VirtualKeyCode::Down => c.is_backward_pressed = is_pressed,
                VirtualKeyCode::D | VirtualKeyCode::Right => c.is_right_pressed = is_pressed,
                VirtualKeyCode::E => c.is_up_pressed = is_pressed,
                VirtualKeyCode::Q => c.is_down_pressed = is_pressed,
                _ => (),
            }
        }
    }

    for MouseIn(state, button) in ev_mousein.read() {
        for (mut c,) in controllers.iter_mut() {
            match button {
                winit::event::MouseButton::Left => {
                    c.dragging = match state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    }
                }
                _ => (),
            }
        }
    }

    for MouseMv(position) in ev_mousemv.read() {
        for (mut c,) in controllers.iter_mut() {
            c.mouse_delta.x = (c.last_mouse_pos.x - position.x) as f32;
            c.mouse_delta.y = (c.last_mouse_pos.y - position.y) as f32;
            c.last_mouse_pos = *position;
        }
    }
}

pub fn update_camera(mut q: Query<(&mut CameraController, &mut Transform)>, time: Res<Time>) {
    for (mut controller, mut transform) in q.iter_mut() {
        let speed = controller.speed * time.delta_time;

        let forward = transform.forward();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if controller.is_forward_pressed {
            transform.translate(forward * speed);
        }
        if controller.is_backward_pressed {
            transform.translate(-forward * speed);
        }

        let left = transform.left();

        if controller.is_right_pressed {
            transform.translate(-left * speed);
        }
        if controller.is_left_pressed {
            transform.translate(left * speed);
        }

        let up = transform.up();

        if controller.is_down_pressed {
            transform.translate(-up * speed);
        }
        if controller.is_up_pressed {
            transform.translate(up * speed);
        }

        // rotate camera
        if controller.dragging {
            let off = controller.mouse_delta / 100.0;

            let mut e = transform.get_euler();

            e.z = 0.0;

            e.x += off.x;
            e.y += off.y;

            e.y = e.y.clamp(-1.0, 1.0);

            transform.set_euler(e);
        }
        controller.mouse_delta = Vec2::ZERO;
    }
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            dragging: false,
            last_mouse_pos: Default::default(),
            mouse_delta: Default::default(),
        }
    }

    pub fn speed_mut(&mut self) -> &mut f32 {
        &mut self.speed
    }
}
