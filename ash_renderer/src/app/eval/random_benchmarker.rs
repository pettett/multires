use std::{fs, io::Write};

use bevy_ecs::prelude::*;

use common_renderer::components::{camera::Camera, transform::Transform};

use glam::Vec3A;
use rand::SeedableRng;

use crate::{app::scene::SceneEvent, random_point_on_sphere, Config};

use super::super::{app::QueryEvent, fps_limiter::FPSMeasure, renderer::Renderer, scene::Scene};

pub enum BenchmarkAction {
    Screenshot,
    Log,
}
#[derive(Clone, Copy)]
pub enum BenchmarkMovement {
    Spiral {
        radius: f32,
    },
    Zoom {
        start: glam::Vec3A,
        end: glam::Vec3A,
    },
    Random {
        look_at: glam::Vec3A,
        radius: f32,
    },
    None,
}

enum BenchmarkStage {
    Start,
    Heating(usize),
    Running(usize),
    EnterStage(usize),
    Finish,
}

/// Entity to hold data about current benchmark, controlling camera and recording values
#[derive(Resource)]
pub struct RandomBenchmarker {
    samples: usize,
    s: usize,
    rounds: usize,
    stage: BenchmarkStage,

    total_instances: usize,
    rng: rand::rngs::StdRng,

    movement: BenchmarkMovement,

    instances: usize,

    last_low_error_result: Option<u32>,
    last_high_error_result: Option<u32>,

    results: Vec<(usize, Vec<(f64, u32, u32)>)>,
}

impl RandomBenchmarker {
    pub fn default() -> Self {
        Self::new(
            BenchmarkMovement::Random {
                look_at: Vec3A::ZERO,
                radius: 1300.0,
            },
            2500,
            1,
        )
    }

    pub fn default_zoom() -> Self {
        Self::new(
            BenchmarkMovement::Zoom {
                start: glam::Vec3A::Z * 10.0,
                end: glam::Vec3A::Z * 1500.0,
            },
            2500,
            1,
        )
    }

    pub fn new(movement: BenchmarkMovement, instances: usize, rounds: usize) -> Self {
        Self {
            movement,
            samples: 1500,
            s: 0,
            last_low_error_result: None,
            last_high_error_result: None,
            stage: BenchmarkStage::Start,
            rounds,
            results: Vec::new(),
            rng: rand::rngs::StdRng::seed_from_u64(0),
            total_instances: 0,
            instances,
        }
    }

    fn record(&self, renderer: &Renderer, scene: &Scene) {
        // Output all results to files

        let t = chrono::offset::Local::now().format("%Y-%m-%d(%H-%M)");
        let p = format!(
            "benchmark_rand/{:?}{}{}{}/{t}",
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
            for (dt, prims1, prims2) in r {
                writeln!(&mut f, "0.0, {dt}, {prims1}, {prims2}")
                    .expect("Failed to write line in file");
            }
        }
    }
}

/// ECS System
pub fn random_benchmark(
    mut cameras: Query<(&mut Transform, &Camera)>,
    benchmarker: Option<ResMut<RandomBenchmarker>>,
    mut commands: Commands,
    mut scene_events: EventWriter<SceneEvent>,
    mut query_events: EventReader<QueryEvent>,
    mut renderer: ResMut<Renderer>,
    mut scene: ResMut<Scene>,
) {
    let mut prims = None;
    for e in query_events.read() {
        // println!("{e:?}");

        match e {
            QueryEvent::ClippedPrimitives(c) => prims = Some(*c),
            _ => (),
        }
    }

    if let Some(mut bench) = benchmarker {
        let (mut camera, _) = cameras.single_mut();

        match bench.stage {
            BenchmarkStage::Start => {
                scene_events.send(SceneEvent::ResetScene);
                renderer.render_gui = false;
                bench.stage = BenchmarkStage::EnterStage(0);
            }
            BenchmarkStage::EnterStage(i) => {
                bench.s = 0;
                if i == bench.rounds {
                    bench.stage = BenchmarkStage::Finish;
                } else {
                    scene_events.send(SceneEvent::AddInstances(bench.instances));
                    bench.total_instances += bench.instances;

                    // Get next data ready
                    let ti = bench.total_instances;
                    bench.results.push((ti, Vec::new()));

                    println!("Benchmark Warming up");
                    bench.stage = BenchmarkStage::Heating(i);
                }
            }
            BenchmarkStage::Heating(i) => {
                println!("Benchmark beginning record");
                bench.s = 0;
                bench.stage = BenchmarkStage::Running(i);
            }
            BenchmarkStage::Running(i) if prims.is_some() => {
                // use longer running deltatime for recording

                if scene.target_error == 0.1 {
                    if bench.last_low_error_result != prims {
                        bench.last_low_error_result = prims;
                    } else {
                        scene.target_error = 0.2;
                    }
                } else if scene.target_error == 0.2 && prims != bench.last_low_error_result {
                    if bench.last_high_error_result != prims {
                        bench.last_high_error_result = prims;
                    } else {
                        let p1 = bench.last_low_error_result.unwrap();
                        let p2 = bench.last_high_error_result.unwrap();
                        bench.results[i]
                            .1
                            .push((renderer.gpu_time.last_sample(), p1, p2));

                        println!("{p1} / {p2}");

                        bench.s += 1;
                        bench.last_low_error_result = None;
                        bench.last_high_error_result = None;

                        match bench.movement {
                            BenchmarkMovement::Random { look_at, radius } => {
                                camera.set_pos(random_point_on_sphere(&mut bench.rng) * radius);
                                camera.look_at(look_at);
                            }
                            BenchmarkMovement::Zoom { start, end } => {
                                camera.set_pos(
                                    start.lerp(end, bench.s as f32 / bench.samples as f32),
                                );
                                camera.look_at(Vec3A::ZERO);
                            }
                            _ => (),
                        }

                        scene.target_error = 0.1;
                    }
                }

                if bench.s > bench.samples {
                    bench.stage = BenchmarkStage::EnterStage(i + 1);
                }
            }
            BenchmarkStage::Running(_) => (),
            BenchmarkStage::Finish => {
                // give a second for the recording to finish

                // cleanup
                bench.record(&renderer, &scene);
                renderer.render_gui = true;
                commands.remove_resource::<RandomBenchmarker>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use crate::random_point_on_sphere;

    #[test]
    fn test_random_points_deterministic() {
        let mut rng_1 = rand::rngs::StdRng::seed_from_u64(0);
        let mut rng_2 = rand::rngs::StdRng::seed_from_u64(0);

        for i in 0..2500 {
            assert_eq!(
                random_point_on_sphere(&mut rng_1),
                random_point_on_sphere(&mut rng_2)
            );
        }
    }
}
