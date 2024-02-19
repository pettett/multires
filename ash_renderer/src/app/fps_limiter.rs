use std::time::Duration;
use std::time::Instant;

use bevy_ecs::system::Resource;

use super::frame_measure::RollingMeasure;

const NANOS_PER_SEC: f32 = NANOS_PER_SEC_64 as _;
const NANOS_PER_SEC_64: f64 = 1000_000_000.0;

#[derive(Resource)]
pub struct FPSMeasure {
    delta_time_counter: Instant,
    fps_measure_counter: Instant,
    delta_time_nanos: RollingMeasure<u32, 15>,
    fps_measure: RollingMeasure<f32, 60>,
}
impl Default for FPSMeasure {
    fn default() -> Self {
        FPSMeasure {
            delta_time_counter: Instant::now(),
            fps_measure_counter: Instant::now(),
            delta_time_nanos: RollingMeasure::default(),
            fps_measure: RollingMeasure::default(),
        }
    }
}
impl FPSMeasure {
    pub fn new() -> FPSMeasure {
        FPSMeasure::default()
    }

    /// Call this function in game loop to update its inner status.
    pub fn tick_frame(&mut self) {
        let time_elapsed = self.delta_time_counter.elapsed();
        self.delta_time_counter = Instant::now();

        self.delta_time_nanos.tick(time_elapsed.as_nanos() as _);

        if self.fps_measure_counter.elapsed() > Duration::from_secs_f32(0.03) {
            self.fps_measure.tick(self.long_fps());
            self.fps_measure_counter = Instant::now();
        }
    }

    /// Calculate the current FPS. Slow to react, long running.
    pub fn long_fps(&self) -> f32 {
        NANOS_PER_SEC / self.delta_time_nanos.mean() as f32
    }

    /// Calculate the current FPS. Quick to react, jumpy
    pub fn short_fps(&self) -> f32 {
        NANOS_PER_SEC / self.delta_time_nanos.last_sample() as f32
    }

    /// Return current delta time in seconds
    /// this function ignore its second part, since the second is mostly zero.
    pub fn delta_time(&self) -> f32 {
        self.delta_time_nanos.last_sample() as f32 / NANOS_PER_SEC // time in second
    }
}

impl egui::Widget for &FPSMeasure {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        self.fps_measure.gui("FPS", ui)
    }
}
