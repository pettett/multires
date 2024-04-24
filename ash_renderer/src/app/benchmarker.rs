use std::{
    fs,
    io::Write,
    sync::mpsc::{self, Sender},
    thread::{self, sleep},
    time::Duration,
};

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};
use enigo::KeyboardControllable;
use glam::Vec3A;

use crate::{app::scene::SceneEvent, Config};

use super::{fps_limiter::FPSMeasure, renderer::Renderer};

enum BenchmarkStage {
    Start,
    Heating(usize),
    Running(usize),
    EnterStage(usize),
    Finish,
}

/// Entity to hold data about current benchmark, controlling camera and recording values
#[derive(Resource)]
pub struct Benchmarker {
    start: glam::Vec3A,
    end: glam::Vec3A,

    t: f32,
    time: f32,
    rounds: usize,
    stage: BenchmarkStage,

    total_instances: usize,

    instances_per_round: usize,

    results: Vec<(usize, Vec<(f32, f32, u32)>)>,
}

impl Benchmarker {
    pub fn default() -> Self {
        Benchmarker::new(glam::Vec3A::Z * 10.0, glam::Vec3A::Z * 1500.0, 15.0, 500, 5)
    }

    pub fn new(
        start: glam::Vec3A,
        end: glam::Vec3A,
        time: f32,
        instances_per_round: usize,
        rounds: usize,
    ) -> Self {
        println!("Beginning benchmark between {} and {}", start, end);

        Self {
            start,
            end,
            t: 0.0,
            time,
            stage: BenchmarkStage::Start,
            rounds,
            results: Vec::new(),
            total_instances: 0,
            instances_per_round,
        }
    }

    fn record(&self, renderer: &Renderer, config: &Config) {
        // Output all results to files

        let t = chrono::offset::Local::now().format("%Y-%m-%d(%H-%M)");
        let p = format!(
            "benchmark/{:?}{}/{t}",
            renderer.current_pipeline,
            renderer.mesh.split('/').skip(1).next().unwrap()
        );

        println!("Making dir  {:?}", p);

        fs::create_dir_all(&p).expect("Failed to create directory");

        println!("Writing to {:?}", fs::canonicalize(&p).unwrap());

        for (i, r) in &self.results {
            let mut f =
                fs::File::create(format!("{p}/{i}.txt")).expect("Failed to create log file");
            for (t, dt, prims) in r {
                writeln!(&mut f, "{t}, {dt}, {prims}").expect("Failed to write line in file");
            }
        }
    }
}

pub fn benchmark(
    mut cameras: Query<(&mut Transform, &Camera)>,
    benchmarker: Option<ResMut<Benchmarker>>,
    mut commands: Commands,
    mut scene_events: EventWriter<SceneEvent>,
    mut renderer: ResMut<Renderer>,
    config: Res<Config>,
    time: Res<FPSMeasure>,
) {
    if let Some(mut bench) = benchmarker {
        let (mut camera, _) = cameras.single_mut();

        match bench.stage {
            BenchmarkStage::Start => {
                scene_events.send(SceneEvent::ResetScene);
                renderer.render_gui = false;
                bench.stage = BenchmarkStage::EnterStage(0);
            }
            BenchmarkStage::EnterStage(i) => {
                bench.t = 0.0;
                if i == bench.rounds {
                    bench.stage = BenchmarkStage::Finish;
                } else {
                    scene_events.send(SceneEvent::AddInstances(bench.instances_per_round));
                    bench.total_instances += bench.instances_per_round;

                    // Get next data ready
                    let ti = bench.total_instances;
                    bench.results.push((ti, Vec::new()));

                    println!("Benchmark Warming up");
                    bench.stage = BenchmarkStage::Heating(i);
                }
            }
            BenchmarkStage::Heating(i) => {
                camera.set_pos(bench.start);
                camera.look_at(Vec3A::ZERO);
                bench.t += time.delta_time();
                if bench.t > 1.0 {
                    println!("Benchmark beginning record");
                    bench.t = 0.0;
                    bench.stage = BenchmarkStage::Running(i);
                }
            }
            BenchmarkStage::Running(i) => {
                bench.t += time.delta_time() / bench.time;
                // use longer running deltatime for recording
                let t = bench.t;
                bench.results[i].1.push((
                    t,
                    time.long_delta_time(),
                    renderer.primitives.last_sample(),
                ));
                camera.set_pos(bench.start.lerp(bench.end, bench.t));
                camera.look_at(Vec3A::ZERO);
                if bench.t > 1.0 {
                    bench.t = 0.0;

                    bench.stage = BenchmarkStage::EnterStage(i + 1);
                }
            }
            BenchmarkStage::Finish => {
                if bench.t < 0.2 {
                    bench.t = 0.5;
                }
                // give a second for the recording to finish

                bench.t += time.delta_time();
                if bench.t > 1.0 {
                    // cleanup
                    bench.record(&renderer, &config);
                    renderer.render_gui = true;
                    commands.remove_resource::<Benchmarker>();
                }
            }
        }
    }
}
