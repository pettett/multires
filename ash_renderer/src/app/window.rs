use std::sync::Arc;

use common_renderer::resources::time::Time;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::App;

use super::fps_limiter::FPSLimiter;

const IS_PAINT_FPS_COUNTER: bool = true;

pub fn init_window(
    event_loop: &EventLoop<()>,
    title: &str,
    width: u32,
    height: u32,
) -> Arc<winit::window::Window> {
    Arc::new(
        winit::window::WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(event_loop)
            .expect("Failed to create window."),
    )
}

pub struct ProgramProc {
    pub event_loop: EventLoop<()>,
}

impl ProgramProc {
    pub fn new() -> ProgramProc {
        // init window stuff
        let event_loop = EventLoop::new();

        ProgramProc { event_loop }
    }

    pub fn main_loop(self, mut vulkan_app: App) {
        let mut tick_counter = FPSLimiter::new();

        self.event_loop
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent { event, .. } => {
                    vulkan_app.input(&event);

                    match event {
                        WindowEvent::CloseRequested => {
                            vulkan_app.core.device.wait_device_idle();
                            *control_flow = ControlFlow::Exit
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                virtual_keycode,
                                state,
                                ..
                            } => match (virtual_keycode, state) {
                                (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                    vulkan_app.core.device.wait_device_idle();
                                    *control_flow = ControlFlow::Exit
                                }
                                _ => {}
                            },
                        },
                        WindowEvent::Resized(_new_size) => {
                            vulkan_app.core.device.wait_device_idle();
                            vulkan_app.resize_framebuffer();
                        }
                        _ => {}
                    };
                }
                Event::MainEventsCleared => {
                    vulkan_app.schedule.run(&mut vulkan_app.world);

                    vulkan_app.update_pipeline();

                    vulkan_app.window_ref().request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    vulkan_app
                        .world
                        .resource_mut::<Time>()
                        .tick(tick_counter.delta_time());

                    vulkan_app.draw_frame(&tick_counter);

                    tick_counter.tick_frame();
                }
                Event::LoopDestroyed => {
                    vulkan_app.core.device.wait_device_idle();
                }
                _ => (),
            })
    }
}
