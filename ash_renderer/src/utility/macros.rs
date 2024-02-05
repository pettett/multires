macro_rules! vk_device_owned_wrapper {
    ($struct_name:ident, $destructor_name:ident) => {
        pub struct $struct_name {
            device: Arc<Device>,
            handle: vk::$struct_name,
        }

		impl $struct_name{
			pub fn parent(&self) -> &Device{
				&self.device
			}
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

        impl std::ops::Deref for $struct_name {
            type Target = vk::$struct_name;

            fn deref(&self) -> &Self::Target {
                &self.handle
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

        impl<P> std::ops::Deref for $struct_name<P> {
            type Target = vk::$struct_name;

            fn deref(&self) -> &Self::Target {
                &self.handle
            }
        }
    };
}

macro_rules! vk_handle_wrapper_const {
    ($struct_name:ident, $const:ident) => {
        impl<const T: $const> crate::VkHandle for $struct_name<T> {
            type VkItem = vk::$struct_name;

            fn handle(&self) -> Self::VkItem {
                self.handle
            }
        }

        impl<const T: $const> std::ops::Deref for $struct_name<T> {
            type Target = vk::$struct_name;

            fn deref(&self) -> &Self::Target {
                &self.handle
            }
        }
    };
}

macro_rules! vk_device_drop {
	($struct_name:ident, $destructor_name:ident) => {
		impl Drop for $struct_name {
			fn drop(&mut self) {
				unsafe {
					self.device.$destructor_name(self.handle, None);
				}
			}
		}
	};

	($struct_name:ident, $destructor_name:ident, $expect:expr ) => {
		impl Drop for $struct_name {
			fn drop(&mut self) {
				unsafe {
					self.device.$destructor_name(self.handle, None).expect($expect:ident);
				}
			}
		}
	};
}

pub(crate) use vk_device_drop;
pub(crate) use vk_device_owned_wrapper;
pub(crate) use vk_handle_wrapper;
pub(crate) use vk_handle_wrapper_const;
pub(crate) use vk_handle_wrapper_g;
