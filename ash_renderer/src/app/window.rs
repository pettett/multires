use std::sync::Arc;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use super::app::App;
use super::benchmarker::Benchmarker;
use super::renderer::{MeshDrawingPipelineType, Renderer};
use super::scene::SceneEvent;

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
impl Default for ProgramProc {
    fn default() -> Self {
        Self::new()
    }
}
impl ProgramProc {
    pub fn new() -> ProgramProc {
        // init window stuff
        let event_loop = EventLoop::new();

        ProgramProc { event_loop }
    }

    pub fn main_loop(self, mut vulkan_app: App) {
        self.event_loop
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent { event, .. } => {
                    vulkan_app.input(&event);

                    match event {
                        WindowEvent::CloseRequested => {
                            vulkan_app.renderer().core.device.wait_device_idle();
                            *control_flow = ControlFlow::Exit
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                virtual_keycode,
                                state,
                                ..
                            } => match (virtual_keycode, state) {
                                (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                    vulkan_app.renderer().core.device.wait_device_idle();
                                    *control_flow = ControlFlow::Exit
                                }
                                (Some(VirtualKeyCode::F1), ElementState::Pressed) => {
                                    vulkan_app.world.send_event(SceneEvent::AddInstances(20));
                                }
                                (Some(VirtualKeyCode::F5), ElementState::Pressed) => {
                                    // FIXME: release mode fucks the UI
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::DrawLOD);
                                }
                                (Some(VirtualKeyCode::F6), ElementState::Pressed) => {
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::DrawIndirect);
                                }
                                (Some(VirtualKeyCode::F7), ElementState::Pressed) => {
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::IndirectTasks);
                                }
                                (Some(VirtualKeyCode::F8), ElementState::Pressed) => {
                                    vulkan_app.world.send_event(
                                        MeshDrawingPipelineType::ExpandingComputeCulledMesh,
                                    );
                                }
                                (Some(VirtualKeyCode::F9), ElementState::Pressed) => {
                                    vulkan_app.world.insert_resource(Benchmarker::default());
                                }
                                _ => {}
                            },
                        },
                        WindowEvent::Resized(_new_size) => {
                            vulkan_app.renderer().core.device.wait_device_idle();
                            vulkan_app
                                .world
                                .get_resource_mut::<Renderer>()
                                .unwrap()
                                .resize_framebuffer();
                        }
                        _ => {}
                    };
                }
                Event::MainEventsCleared => {
                    vulkan_app.schedule.run(&mut vulkan_app.world);

                    //    vulkan_app.renderer().update_pipeline();

                    vulkan_app.renderer().window_ref().request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    vulkan_app.draw_schedule.run(&mut vulkan_app.world);
                }
                Event::LoopDestroyed => {
                    vulkan_app.renderer().core.device.wait_device_idle();
                }
                _ => (),
            })
    }
}
