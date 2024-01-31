use std::{marker::PhantomData, sync::Arc};

use ash::vk;

use crate::{
    utility::{
        device::Device,
        macros::{vk_handle_wrapper, vk_handle_wrapper_g},
    },
    VkHandle,
};

pub struct QueryPool<R> {
    device: Arc<Device>,
    handle: vk::QueryPool,
    query_count: u32,
    _p: PhantomData<R>,
}

vk_handle_wrapper_g!(QueryPool);

impl<R> QueryPool<R>
where
    R: bytemuck::Zeroable + Copy,
{
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
            _p: Default::default(),
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
    pub fn get_results(&self, i: u32) -> Option<R> {
        let mut results = [R::zeroed()];
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

impl<R> Drop for QueryPool<R> {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_query_pool(self.handle, None);
        }
    }
}
