use std::collections::VecDeque;
use std::fmt;
use std::thread;
use std::time::Duration;
use std::time::Instant;

#[derive(Default)]
pub struct RollingMeasure<T, const S: usize>
where
    T: Default,
{
    samples: VecDeque<T>,
}

impl<T, const S: usize> RollingMeasure<T, S>
where
    // T is a numeric type
    T: num::Num
        + num::ToPrimitive
        + num::FromPrimitive
        + Copy
        + std::ops::AddAssign
        + Default
        + fmt::Display,
{
    /// Call this function in game loop to update its inner status.
    pub fn tick(&mut self, sample: T) {
        if self.samples.len() >= S {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Calculate the current rolling mean.
    ///
    /// Returns 0 for a 0 sampled collection
    pub fn mean(&self) -> T {
        let mut sum = T::zero();

        let div = T::from_usize(self.samples.len().max(1)).unwrap();

        self.samples.iter().for_each(|&val| sum += val / div);

        sum
    }

    /// Grab the most recent sample
    pub fn last_sample(&self) -> T {
        self.samples.front().copied().unwrap_or(T::zero())
    }

    pub fn gui(&self, measure: &str, ui: &mut egui::Ui) -> egui::Response {
        let r = ui.label(format!("{}: {}", measure, self.mean()));

        let points = egui_plot::PlotPoints::new(
            self.samples
                .iter()
                .enumerate()
                .map(|(x, &y)| [x as _, y.to_f64().unwrap()])
                .collect(),
        );
        // let sin: egui_plot::PlotPoints = (0..1000)
        //     .map(|i| {
        //         let x = i as f64 * 0.01;
        //         [x, x.sin()]
        //     })
        //     .collect();

        let line = egui_plot::Line::new(points);

        egui_plot::Plot::new(measure)
            .allow_boxed_zoom(false)
            .auto_bounds_y()
            .include_y(self.last_sample().to_f64().unwrap() + 1.0)
            .include_y(0.0)
            .view_aspect(2.0)
            .show(ui, |plot_ui| {
                plot_ui.line(line);
            });

        r
    }
}
