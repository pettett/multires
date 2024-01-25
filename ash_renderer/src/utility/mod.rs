//! The utility mod define some fixed function using in this tutorial.
//! Help to simplify the code.
pub mod pooled;
pub mod buffer;
pub mod constants;
pub mod debug;
pub mod device;
pub mod fps_limiter;
pub mod image;
pub mod instance;
pub mod physical_device;
mod pipeline;
pub mod platforms;
pub mod render_pass;
pub mod structures;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod tools;
pub mod window;

pub use pipeline::ComputePipeline;
pub use pipeline::GraphicsPipeline;
pub use pipeline::ShaderModule;
