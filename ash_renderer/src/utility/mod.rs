//! The utility mod define some fixed function using in this tutorial.
//! Help to simplify the code.
pub mod buffer;
pub mod constants;
pub mod debug;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
mod pipeline;
pub mod platforms;
pub mod pooled;
pub mod render_pass;
pub mod structures;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod tools;

pub use pipeline::ComputePipeline;
pub use pipeline::GraphicsPipeline;
pub use pipeline::ShaderModule;

// Macros for common wrapper implementations
mod macros {

    macro_rules! vk_device_owned_wrapper {
        ($struct_name:ident, $destructor_name:ident) => {
            pub struct $struct_name {
                device: Arc<Device>,
                handle: vk::$struct_name,
            }

            crate::utility::macros::vk_handle_wrapper!($struct_name);
            crate::utility::macros::vk_device_drop!($struct_name, $destructor_name);
        };
    }

    macro_rules! vk_handle_wrapper {
        ($struct_name:ident) => {
            impl crate::VkHandle for $struct_name {
                type VkItem = vk::$struct_name;

                fn handle(&self) -> Self::VkItem {
                    self.handle
                }
            }
        };

        ($struct_name:ident, $vk_name:ident) => {
            impl crate::VkHandle for $struct_name {
                type VkItem = vk::$vk_name;

                fn handle(&self) -> Self::VkItem {
                    self.handle
                }
            }
        };
    }
    macro_rules! vk_handle_wrapper_g {
        ($struct_name:ident) => {
            impl<P> crate::VkHandle for $struct_name<P> {
                type VkItem = vk::$struct_name;

                fn handle(&self) -> Self::VkItem {
                    self.handle
                }
            }
        };
    }

    macro_rules! vk_device_drop {
        ($struct_name:ident, $destructor_name:ident) => {
            impl Drop for $struct_name {
                fn drop(&mut self) {
                    unsafe {
                        self.device.handle.$destructor_name(self.handle, None);
                    }
                }
            }
        };

		($struct_name:ident, $destructor_name:ident, $expect:expr ) => {
            impl Drop for $struct_name {
                fn drop(&mut self) {
                    unsafe {
                        self.device.handle.$destructor_name(self.handle, None).expect($expect:ident);
                    }
                }
            }
        };
    }

    pub(crate) use vk_device_drop;
    pub(crate) use vk_device_owned_wrapper;
    pub(crate) use vk_handle_wrapper;
    pub(crate) use vk_handle_wrapper_g;
}
