use std::sync::Arc;

use ash::vk;

use crate::utility::device::Device;

pub struct QueryPool {
    device: Arc<Device>,
    handle: vk::QueryPool,
}

impl QueryPool {
    pub fn new(device: Arc<Device>) -> Self {
        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .pipeline_statistics(vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES)
            .query_count(2)
            .build();

        let handle = unsafe {
            device
                .handle
                .create_query_pool(&create_info, None)
                .expect("Failed to create query group")
        };

        Self { device, handle }
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_query_pool(self.handle, None);
        }
    }
}
