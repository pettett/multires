use std::{collections::HashMap, ffi};

use ash::vk;

use super::instance::Instance;

/// Stores a sorted list of extensions
#[derive( Debug)]
pub struct Extensions {
    names: Vec<&'static ffi::CStr>,
}

impl PartialEq for Extensions{
    fn eq(&self, other: &Self) -> bool {
        for n in &self.names{
			if other.names.binary_search(n).is_err(){
				return false;
			}
		}
		return true;
    }
}

impl Extensions {
    pub fn new(mut names: Vec<&'static ffi::CStr>) -> Self {
		names.sort();

        Self { names }
    }
    pub fn union(mut self, other: &Extensions) -> Self {
        self.names.extend_from_slice(&other.names);
		
        Extensions::new(self.names)
    }


    pub fn get_extensions_names(&self) -> &[&'static ffi::CStr] {
        &self.names
    }

    pub fn get_extensions_raw_names(&self) -> Vec<*const ffi::c_char> {
        self.names.iter().map(|s| s.as_ptr()).collect()
    }

    pub fn get_device_extension_support(
        &self,
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
    ) -> Extensions {
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .expect("Failed to get device extension properties.")
        };

        use std::collections::HashSet;
        let mut available_extension_names = HashSet::new();

        let mut required_extensions = HashMap::new();
        for extension in self.get_extensions_names() {
            required_extensions.insert(*extension, *extension);
        }

        for extension in available_extensions.iter() {
            let raw_string = unsafe { ffi::CStr::from_ptr(extension.extension_name.as_ptr()) };
            if let Some(ext) = required_extensions.get(&raw_string) {
                available_extension_names.insert(*ext);
            }
        }

        Extensions {
            names: available_extension_names.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests{
    use super::Extensions;

	#[test]
	fn test_equality(){
		let e1 = Extensions::new(vec![
            ash::extensions::khr::BufferDeviceAddress::name(),
            ash::extensions::khr::Swapchain::name(),
        ]);
		let e2 = Extensions::new(vec![
            ash::extensions::khr::Swapchain::name(),
            ash::extensions::khr::BufferDeviceAddress::name(),
        ]);

		assert_eq!(e1, e2)
	}
}