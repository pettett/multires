use std::{fs, io::Write, mem};

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};

use glam::Vec3A;

use crate::app::{app::QueryEvent, scene::SceneEvent};

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
pub struct SceneComplexityBenchmarker {
    t: f32,
    samples: usize,
    rounds: usize,
    stage: BenchmarkStage,

    instances: usize,

    last_gpu_time: Vec<f64>,
    last_prims: Option<u32>,

    results: Vec<(usize, Vec<f64>, u32)>,
}

impl SceneComplexityBenchmarker {
    pub fn default() -> Self {
        Self::new(100, 2500, 80)
    }

    pub fn new(samples: usize, instances: usize, rounds: usize) -> Self {
        Self {
            t: 0.0,
            samples,
            stage: BenchmarkStage::Start,
            rounds,
            results: Vec::new(),
            last_gpu_time: Vec::new(),
            last_prims: None,
            instances,
        }
    }

    fn record(&self, renderer: &Renderer, scene: &Scene) {
        // Output all results to files

        let t = chrono::offset::Local::now().format("%Y-%m-%d(%H-%M)");
        let p = format!(
            "benchmark/scene_{:?}{}{}{}/{t}",
            renderer.current_pipeline,
            renderer.mesh.split('/').skip(1).next().unwrap(),
            renderer.query,
            scene.target_error
        );

        println!("Making dir  {:?}", p);

        fs::create_dir_all(&p).expect("Failed to create directory");

        println!("Writing to {:?}", fs::canonicalize(&p).unwrap());

        for (i, r, prims) in &self.results {
            let mut f =
                fs::File::create(format!("{p}/{i}.txt")).expect("Failed to create log file");
            for dt in r {
                writeln!(&mut f, "{i}, {dt}, {prims} ").expect("Failed to write line in file");
            }
        }
    }
}

const CAM_POS: Vec3A = Vec3A::new(0.0, 50.0, 50.0);
const CAM_LOOK_AT: Vec3A = Vec3A::new(50.0, 50.0, 50.0);

pub fn scene_complexity_benchmark(
    mut cameras: Query<(&mut Transform, &Camera)>,
    benchmarker: Option<ResMut<SceneComplexityBenchmarker>>,
    mut commands: Commands,
    mut scene_events: EventWriter<SceneEvent>,
    mut renderer: ResMut<Renderer>,
    scene: Res<Scene>,
    mut query_events: EventReader<QueryEvent>,
    time: Res<FPSMeasure>,
) {
    if let Some(mut bench) = benchmarker {
        let (mut camera, _) = cameras.single_mut();

        match bench.stage {
            BenchmarkStage::Start => {
                scene_events.send(SceneEvent::ResetScene);
                renderer.render_gui = false;
                bench.stage = BenchmarkStage::EnterStage(1);
            }
            BenchmarkStage::EnterStage(i) => {
                bench.t = 0.0;
                if i == bench.rounds + 1 {
                    bench.stage = BenchmarkStage::Finish;
                } else {
                    let new_instances = bench.instances;
                    scene_events.send(SceneEvent::AddInstances(new_instances));

                    // Get next data ready
                    // bench.results.push((i * new_instances, Vec::new()));

                    println!("Benchmark Warming up");
                    bench.stage = BenchmarkStage::Heating(i);
                }
            }
            BenchmarkStage::Heating(i) => {
                camera.set_pos(CAM_POS);
                camera.look_at(CAM_LOOK_AT);
                bench.t += time.delta_time();
                renderer.query = true;
                if bench.t > 1.0 {
                    println!("Benchmark beginning record");
                    bench.t = 0.0;
                    bench.stage = BenchmarkStage::Running(i);
                }
            }
            BenchmarkStage::Running(i) => {
                // use longer running deltatime for recording

                camera.set_pos(CAM_POS);
                camera.look_at(CAM_LOOK_AT);

                for e in query_events.read() {
                    match e {
                        QueryEvent::ClippedPrimitives(p) => {
                            println!("Prims: {p}");
                            if Some(*p) != bench.last_prims {
                                bench.last_prims = Some(*p);
                            } else {
                                // Equalised
                                renderer.query = false;
                            }
                        }
                        QueryEvent::GPUMilliseconds(m) => {
                            println!("GPU: {m}");

                            if bench.last_gpu_time.len() < bench.samples {
                                bench.last_gpu_time.push(*m);
                            } else if bench.last_prims.is_none() {
                                renderer.query = true;
                            } else {
                                let instances = i * bench.instances;
                                let mut last_gpu_time = Vec::new();
                                let mut last_prims = None;

                                mem::swap(&mut last_gpu_time, &mut bench.last_gpu_time);
                                mem::swap(&mut last_prims, &mut bench.last_prims);

                                bench
                                    .results
                                    .push((instances, last_gpu_time, last_prims.unwrap()));

                                renderer.query = true;
                                bench.stage = BenchmarkStage::EnterStage(i + 1);
                            }
                        }
                    }
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
                    commands.remove_resource::<SceneComplexityBenchmarker>();
                }
            }
        }
    }
}
