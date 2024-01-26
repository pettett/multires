use std::sync::Arc;

use ash::vk;

use crate::{utility::device::Device, VkHandle};

pub struct QueryPool {
    device: Arc<Device>,
    handle: vk::QueryPool,
    query_count: u32,
}

impl VkHandle for QueryPool {
    type VkItem = vk::QueryPool;

    fn handle(&self) -> Self::VkItem {
        self.handle
    }
}

impl QueryPool {
    pub fn new(device: Arc<Device>, query_count: u32) -> Arc<Self> {
        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .pipeline_statistics(
                vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT
                    | vk::QueryPipelineStatisticFlags::TASK_SHADER_INVOCATIONS_EXT, //| vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS,
            )
            .query_count(query_count)
            .build();

        let handle = unsafe {
            device
                .handle
                .create_query_pool(&create_info, None)
                .expect("Failed to create query group")
        };

        Arc::new(Self {
            device,
            handle,
            query_count,
        })
    }
    /// Call `cmd_reset_query_pool` for all queries in this buffer
    pub fn reset(&self, cmd: vk::CommandBuffer, i: u32) {
        unsafe {
            self.device
                .handle
                .cmd_reset_query_pool(cmd, self.handle, i, 1);
        }
    }
    /// Call `get_query_pool_results` for the correct sized
    pub fn get_results(&self, i: u32) -> Option<[i32; 9]> {
        let mut results = [[0; 9]];
        unsafe {
            self.device
                .handle
                .get_query_pool_results(
                    self.handle,
                    i,
                    1,
                    &mut results,
                    vk::QueryResultFlags::WITH_AVAILABILITY,
                )
                .ok()
                .map(|()| results[0])
        }
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_query_pool(self.handle, None);
        }
    }
}
