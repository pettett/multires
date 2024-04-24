use std::{
    f32::consts,
    fs,
    io::Write,
    rc::Rc,
    sync::{
        mpsc::{self, Sender},
        Arc,
    },
    thread::{self, sleep},
    time::Duration,
};

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};
use glam::{vec3a, FloatExt, Vec3A};

use crate::{app::scene::SceneEvent, Config};

use super::{fps_limiter::FPSMeasure, renderer::Renderer};

#[derive(Clone)]
enum RecordStage {
    Start,
    Running(usize, Arc<String>),
    EnterStage(usize),
    Finish,
}

/// Entity to hold data about current benchmark, controlling camera and recording values
#[derive(Resource)]
pub struct Recorder {
    start: f32,
    end: f32,
    rotation: f32,

    t: usize,
    p: usize,
    rounds: usize,
    stage: RecordStage,

    total_instances: usize,
    instances_per_round: usize,
}

impl Recorder {
    pub fn default() -> Self {
        Self::new(10.0, 150.0, 2.0 * consts::PI, 300, 1)
    }

    pub fn new(
        start: f32,
        end: f32,
        rotation: f32,
        instances_per_round: usize,
        rounds: usize,
    ) -> Self {
        println!("Beginning benchmark between {} and {}", start, end);

        Self {
            start,
            end,
            rotation,
            t: 0,
            p: 0,
            stage: RecordStage::Start,
            rounds,
            total_instances: 0,
            instances_per_round,
        }
    }

    fn max_t(&self) -> usize {
        200
    }

    fn pos(&self) -> glam::Vec3A {
        let t = self.t as f32 / self.max_t() as f32;
        vec3a(
            f32::sin(self.rotation * t),
            0.0,
            f32::cos(self.rotation * t),
        ) * self.start.lerp(self.end, t)
    }
}

pub fn record(
    mut cameras: Query<(&mut Transform, &Camera)>,
    recorder: Option<ResMut<Recorder>>,
    mut commands: Commands,
    mut scene_events: EventWriter<SceneEvent>,
    mut renderer: ResMut<Renderer>,
    config: Res<Config>,
    time: Res<FPSMeasure>,
) {
    if let Some(mut bench) = recorder {
        let (mut camera, _) = cameras.single_mut();

        match bench.stage.clone() {
            RecordStage::Start => {
                scene_events.send(SceneEvent::ResetScene);
                renderer.render_gui = false;
                bench.stage = RecordStage::EnterStage(0);
            }
            RecordStage::EnterStage(i) => {
                bench.t = 0;
                if i == bench.rounds {
                    bench.stage = RecordStage::Finish;
                } else {
                    scene_events.send(SceneEvent::AddInstances(bench.instances_per_round));
                    bench.total_instances += bench.instances_per_round;

                    // Get next data ready
                    let ti = bench.total_instances;

                    let t = chrono::offset::Local::now().format("%Y-%m-%d(%H-%M)");
                    let p = format!(
                        "benchmark/{:?}{}/{t}",
                        renderer.current_pipeline,
                        renderer.mesh.split('/').skip(1).next().unwrap()
                    );

                    bench.stage = RecordStage::Running(i, Arc::new(p));
                }
            }

            RecordStage::Running(i, p) => {
                // use longer running deltatime for recording

                camera.set_pos(bench.pos());
                camera.look_at(Vec3A::ZERO);

                bench.p += 1;
                // A frame buffer is about 3 frames long. If we dont do this, things sadly get janky
                if bench.p == 6 {
                    renderer.screenshot(p.to_string(), format!("{}.png", bench.t));
                }
                if bench.p == 12 {
                    bench.t += 1;
                    bench.p = 0;
                }

                if bench.t >= bench.max_t() {
                    bench.t = 0;

                    bench.stage = RecordStage::EnterStage(i + 1);
                }
            }
            RecordStage::Finish => {
                // cleanup

                renderer.render_gui = true;
                commands.remove_resource::<Recorder>();
            }
        }
    }
}
