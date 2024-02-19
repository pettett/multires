use bevy_ecs::prelude::*;
use common_renderer::{
    components::{camera::Camera, transform::Transform},
    resources::time::Time,
};
use glam::Vec3A;

use crate::app::scene::SceneEvent;

enum BenchmarkStage {
    Start,
    Heating(usize),
    Running(usize),
}

/// Entity to hold data about current benchmark, controlling camera and recording values
#[derive(Resource)]
pub struct Benchmarker {
    start: glam::Vec3A,
    end: glam::Vec3A,

    t: f32,
    time: f32,
    max_runs: usize,
    stage: BenchmarkStage,
}

impl Benchmarker {
    pub fn new(start: glam::Vec3A, end: glam::Vec3A, time: f32) -> Self {
        println!("Begining benchmark between {} and {}", start, end);
        Self {
            start,
            end,
            t: 0.0,
            time,
            stage: BenchmarkStage::Start,
            max_runs: 20,
        }
    }
}

pub fn benchmark(
    mut cameras: Query<(&mut Transform, &Camera)>,
    benchmarker: Option<ResMut<Benchmarker>>,
    mut commands: Commands,
    mut scene_events: EventWriter<SceneEvent>,
    time: Res<Time>,
) {
    if let Some(mut bench) = benchmarker {
        let (mut camera, _) = cameras.single_mut();

        match bench.stage {
            BenchmarkStage::Start => {
                scene_events.send(SceneEvent::ResetScene);
                bench.stage = BenchmarkStage::Heating(0);
            }
            BenchmarkStage::Heating(i) => {
                bench.t += time.delta_time;

                camera.set_pos(bench.start);
                camera.look_at(Vec3A::ZERO);

                if bench.t > 1.0 {
                    println!("Benchmark moving to next stage");

                    bench.t = 0.0;
                    scene_events.send(SceneEvent::AddInstances(50));

                    bench.stage = BenchmarkStage::Running(i);
                }
            }
            BenchmarkStage::Running(i) => {
                bench.t += time.delta_time / bench.time;

                camera.set_pos(bench.start.lerp(bench.end, bench.t));
                camera.look_at(Vec3A::ZERO);

                if bench.t > 1.0 {
                    println!("Benchmark moving to next stage");
                    bench.t = 0.0;
                    if i == bench.max_runs {
                        commands.remove_resource::<Benchmarker>();
                    } else {
                        scene_events.send(SceneEvent::AddInstances(50));
                        bench.stage = BenchmarkStage::Heating(i + 1);
                    }
                }
            }
        }
    }
}
