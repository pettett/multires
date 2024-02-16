//! The utility mod define some fixed function using in this tutorial.
//! Help to simplify the code.
pub mod buffer;
pub mod constants;
pub mod debug;
pub mod device;
pub mod image;
pub mod instance;
mod macros;
pub mod physical_device;
mod pipeline;
pub mod pooled;
pub mod render_pass;
pub mod structures;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod extensions;
pub mod tools;

pub use pipeline::ComputePipeline;
pub use pipeline::GraphicsPipeline;
pub use pipeline::ShaderModule;

// Macros for common wrapper implementations
