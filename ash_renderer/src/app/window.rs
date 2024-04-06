use std::sync::Arc;

use winit::event::{ElementState, Event, KeyEvent,  WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};

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
        let event_loop = EventLoop::new().unwrap();

        ProgramProc { event_loop }
    }

    pub fn main_loop(self, mut vulkan_app: App) {
        self.event_loop
            .run(move |event,  control_flow| match event {
                Event::WindowEvent { event, .. } => {
                    vulkan_app.input( &event);

                    match event {
                        WindowEvent::CloseRequested => {
                            vulkan_app.renderer().core.device.wait_device_idle();
                            control_flow.exit();
                        }
                        WindowEvent::KeyboardInput { event, .. } => match event {
                            KeyEvent {
                                physical_key : PhysicalKey::Code(key_code),
                                state,
                                ..
                            } => match (key_code, state) {
                                (KeyCode::Escape, ElementState::Pressed) => {
                                    vulkan_app.renderer().core.device.wait_device_idle();
                                    control_flow.exit();
                                }
                                (KeyCode::F1, ElementState::Pressed) => {
                                    vulkan_app.world.send_event(SceneEvent::AddInstances(20));
                                }
                                (KeyCode::F5, ElementState::Pressed) => {
                                    // FIXME: release mode fucks the UI
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::DrawLOD);
                                }
                                (KeyCode::F6, ElementState::Pressed) => {
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::DrawIndirect);
                                }
                                (KeyCode::F7, ElementState::Pressed) => {
                                    vulkan_app
                                        .world
                                        .send_event(MeshDrawingPipelineType::IndirectTasks);
                                }
                                (KeyCode::F8, ElementState::Pressed) => {
                                    vulkan_app.world.send_event(
                                        MeshDrawingPipelineType::ExpandingComputeCulledMesh,
                                    );
                                }
                                (KeyCode::F9, ElementState::Pressed) => {
                                    vulkan_app.world.insert_resource(Benchmarker::default());
                                }
                                _ => {}
                            },
							_ => {}
                        },
                        WindowEvent::Resized(_new_size) => {
                            vulkan_app.renderer().core.device.wait_device_idle();
                            vulkan_app
                                .world
                                .get_resource_mut::<Renderer>()
                                .unwrap()
                                .resize_framebuffer();
                        },
						WindowEvent::RedrawRequested => {
							vulkan_app.draw_schedule.run(&mut vulkan_app.world);
						}
                        _ => {}
                    };
                }
                Event::AboutToWait => {
                    vulkan_app.schedule.run(&mut vulkan_app.world);

                    //    vulkan_app.renderer().update_pipeline();

                    vulkan_app.renderer().window_ref().request_redraw();
                }
    
                Event::LoopExiting => {
                    vulkan_app.renderer().core.device.wait_device_idle();
                }
                _ => (),
            }).unwrap()
    }
}
