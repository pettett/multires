pub mod components;
pub mod core;
pub mod vertex;

use core::{App, Renderer};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub fn main() {
    pollster::block_on(run());
}

pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let renderer = Renderer::new(window).await;

    let mut state = App::new(renderer).await;

    event_loop.run(move |event, _, control_flow| {
        //state.handle_event(&event);

        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.renderer().window().id() => match event {
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    // new_inner_size is &&mut so we have to dereference it twice
                    state.resize(**new_inner_size);
                }

                event => {
                    state.input(event);
                }
            },

            Event::RedrawRequested(window_id) if window_id == state.renderer().window().id() => {
                state.redraw()

                // match state.render() {
                //     Ok(_) => {}
                //     // Reconfigure the surface if lost
                //     Err(wgpu::SurfaceError::Lost) => state.resize(state.size()),
                //     // The system is out of memory, we should probably quit
                //     Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                //     // All other errors (Outdated, Timeout) should be resolved by the next frame
                //     Err(e) => eprintln!("{:?}", e),
                // }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.update();

                state.renderer().window().request_redraw();
            }
            _ => {}
        };
    });
}
