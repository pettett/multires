use std::{fs, io::Write};

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};

use glam::Vec3A;

use crate::app::scene::SceneEvent;

use super::super::{fps_limiter::FPSMeasure, renderer::Renderer, scene::Scene};

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
    samples: usize,
    rounds: usize,
    stage: BenchmarkStage,

    instances: usize,

    results: Vec<(usize, Vec<(f32, f64, u32, f32)>)>,
}

impl Benchmarker {
    pub fn default() -> Self {
        Benchmarker::new(
            glam::Vec3A::Z * 10.0,
            glam::Vec3A::Z * 1500.0,
            2000,
            2500,
            10,
        )
    }

    pub fn new(
        start: glam::Vec3A,
        end: glam::Vec3A,
        samples: usize,
        instances: usize,
        rounds: usize,
    ) -> Self {
        println!("Beginning benchmark between {} and {}", start, end);

        Self {
            start,
            end,
            t: 0.0,
            samples,
            stage: BenchmarkStage::Start,
            rounds,
            results: Vec::new(),
            instances,
        }
    }

    fn record(&self, renderer: &Renderer, scene: &Scene) {
        // Output all results to files

        let t = chrono::offset::Local::now().format("%Y-%m-%d(%H-%M)");
        let p = format!(
            "benchmark/{:?}{}{}{}/{t}",
            renderer.current_pipeline,
            renderer.mesh.split('/').skip(1).next().unwrap(),
            renderer.query,
            scene.target_error
        );

        println!("Making dir  {:?}", p);

        fs::create_dir_all(&p).expect("Failed to create directory");

        println!("Writing to {:?}", fs::canonicalize(&p).unwrap());

        for (i, r) in &self.results {
            let mut f =
                fs::File::create(format!("{p}/{i}.txt")).expect("Failed to create log file");
            for (t, dt, prims, dt2) in r {
                writeln!(&mut f, "{t}, {dt}, {prims}, {dt2}")
                    .expect("Failed to write line in file");
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
    scene: Res<Scene>,
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
                    if i == 0 {
                        scene_events.send(SceneEvent::AddInstances(bench.instances));
                    }

                    // Get next data ready
                    bench.results.push((i, Vec::new()));

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
                // use longer running deltatime for recording
                let t = bench.t;
                bench.results[i].1.push((
                    t,
                    renderer.gpu_time.last_sample(),
                    renderer.primitives.last_sample(),
                    time.delta_time(),
                ));
                bench.t += 1.0 / bench.samples as f32;

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
                    bench.record(&renderer, &scene);
                    renderer.render_gui = true;
                    commands.remove_resource::<Benchmarker>();
                }
            }
        }
    }
}
