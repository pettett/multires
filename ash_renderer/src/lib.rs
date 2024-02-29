use ash::vk;

pub mod core;
pub mod draw_pipelines;
pub mod gui;
pub mod multires;
pub mod screen;
pub mod utility;
pub mod vertex;

pub mod app;
pub mod spiral;

// Constants
const WINDOW_TITLE: &'static str = "Multires Mesh Renderer";
const TASK_GROUP_SIZE: u32 = 32;
const CLEAR_COL: vk::ClearColorValue = vk::ClearColorValue {
    float32: [1.0, 1.0, 1.0, 1.0],
};

pub trait VkHandle {
    type VkItem;

    fn handle(&self) -> Self::VkItem;
}
pub trait VkDeviceOwned: VkHandle<VkItem = vk::Device> {}

/// https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
fn random_point_on_sphere(rng: &mut impl rand::Rng) -> glam::Vec3A {
    let u = rng.gen_range(0.0..=1.0);
    let v = rng.gen_range(0.0..=1.0);
    let c = rng.gen_range(0.0..=1.0);
    let theta = u * 2.0 * std::f32::consts::PI;
    let phi = f32::acos(2.0 * v - 1.0);
    let r = f32::cbrt(c);

    let (sin_theta, cos_theta) = f32::sin_cos(theta);
    let (sin_phi, cos_phi) = f32::sin_cos(phi);

    let x = r * sin_phi * cos_theta;
    let y = r * sin_phi * sin_theta;
    let z = r * cos_phi;
    glam::vec3a(x, y, z)
}

// Fix content -------------------------------------------------------------------------------

// fn update_model_uniform_buffer(&mut self, current_image: usize, _delta_time: f32) {
//     unsafe {
//         self.uniform_transform_buffer
//             .update_uniform_buffer(self.uniform_transform);
//     }
// }
