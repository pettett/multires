use std::{
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, OnceLock},
};

use ash::vk;

use crate::{
    utility::{
        device::Device,
        macros::{vk_handle_wrapper, vk_handle_wrapper_g},
    },
    VkHandle,
};

#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct MeshInvocationsQueryResults {
    pub mesh: u32,
    pub avail: u32,
}

impl QueryResult for MeshInvocationsQueryResults {
    fn flags() -> vk::QueryPipelineStatisticFlags {
        vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT
    }
}

#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct PrimitivesQueryResults {
    pub clipping_primitives: u32,
    pub avail: u32,
}

impl QueryResult for PrimitivesQueryResults {
    fn flags() -> vk::QueryPipelineStatisticFlags {
        vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
    }
}

pub struct TypelessQueryPool {
    device: Arc<Device>,
    handle: vk::QueryPool,
    query_count: u32,
}
pub struct QueryPool<R> {
    typeless: Arc<TypelessQueryPool>,
    _p: PhantomData<R>,
}

pub struct Query<'a> {
    pool: &'a TypelessQueryPool,
    cmd: vk::CommandBuffer,
    query: u32,
}

pub trait QueryResult {
    fn flags() -> vk::QueryPipelineStatisticFlags;
}

vk_handle_wrapper!(TypelessQueryPool, QueryPool);

impl TypelessQueryPool {
    /// Call `cmd_reset_query_pool` for a query
    fn reset(&self, cmd: vk::CommandBuffer, i: u32) {
        unsafe {
            self.device.cmd_reset_query_pool(cmd, self.handle, i, 1);
        }
    }

    /// Being a query, controlled by the lifetime of the result. Also reset the query at this index
    pub fn begin_query(&self, cmd: vk::CommandBuffer, query: u32) -> Query {
        self.reset(cmd, query);

        unsafe {
            self.device
                .cmd_begin_query(cmd, self.handle(), query, vk::QueryControlFlags::empty())
        }

        Query {
            pool: self,
            cmd,
            query,
        }
    }
}

impl<R> QueryPool<R>
where
    R: bytemuck::Zeroable + Copy + QueryResult,
{
    pub fn new(device: Arc<Device>, query_count: u32) -> Self {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::PIPELINE_STATISTICS)
            .pipeline_statistics(R::flags())
            .query_count(query_count);

        let handle = unsafe {
            device
                .create_query_pool(&create_info, None)
                .expect("Failed to create query group")
        };

        Self {
            _p: Default::default(),
            typeless: Arc::new(TypelessQueryPool {
                device,
                handle,
                query_count,
            }),
        }
    }
    /// Call `get_query_pool_results` for the correct sized
    pub fn get_results(&self, i: u32) -> Option<R> {
        let mut results = [R::zeroed()];
        unsafe {
            self.device
                .get_query_pool_results(
                    self.handle,
                    i,
                    &mut results,
                    vk::QueryResultFlags::WITH_AVAILABILITY,
                )
                .ok()
                .map(|()| results[0])
        }
    }

    pub fn typeless(&self) -> Arc<TypelessQueryPool> {
        self.typeless.clone()
    }
    pub fn typeless_ref(&self) -> &TypelessQueryPool {
        &self.typeless
    }
}

impl<'a> Query<'a> {
    fn end_query(&self) {
        unsafe {
            self.pool
                .device
                .cmd_end_query(self.cmd, self.pool.handle(), self.query)
        }
    }
}
impl<'a> Drop for Query<'a> {
    fn drop(&mut self) {
        self.end_query()
    }
}

impl<R> Deref for QueryPool<R> {
    type Target = TypelessQueryPool;

    fn deref(&self) -> &Self::Target {
        &self.typeless
    }
}

impl Drop for TypelessQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.handle, None);
        }
    }
}
