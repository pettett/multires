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
    fn flags() -> QueryType {
        QueryType::Pipeline(vk::QueryPipelineStatisticFlags::MESH_SHADER_INVOCATIONS_EXT)
    }
}

#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct PrimitivesQueryResults {
    pub clipping_primitives: u32,
    pub avail: u32,
}

#[derive(bytemuck::Zeroable, Copy, Clone)]
pub struct TimestampResults {
    pub timestamp: u64,
    pub avail: u64,
}

impl QueryResult for PrimitivesQueryResults {
    fn flags() -> QueryType {
        QueryType::Pipeline(vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES)
    }
}

impl QueryResult for TimestampResults {
    fn flags() -> QueryType {
        QueryType::Timestamp
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

pub enum QueryType {
    Pipeline(vk::QueryPipelineStatisticFlags),
    Timestamp,
}

pub trait QueryResult {
    fn flags() -> QueryType;
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

    pub fn write_timestamp_top(&self, cmd: vk::CommandBuffer) {
        self.reset(cmd, 0);

        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.handle(),
                0,
            )
        }
    }

    pub fn write_timestamp_bottom(&self, cmd: vk::CommandBuffer) {
        self.reset(cmd, 1);

        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.handle(),
                1,
            )
        }
    }
}

impl<R> QueryPool<R>
where
    R: bytemuck::Zeroable + Copy + QueryResult,
{
    pub fn new(device: Arc<Device>, query_count: u32) -> Self {
        let create_info = vk::QueryPoolCreateInfo::default().query_count(query_count);

        let create_info = match R::flags() {
            QueryType::Pipeline(flags) => create_info
                .query_type(vk::QueryType::PIPELINE_STATISTICS)
                .pipeline_statistics(flags),
            QueryType::Timestamp => create_info.query_type(vk::QueryType::TIMESTAMP),
        };

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
        self.get_many_results::<1>(i).map(|results| results[0])
    }

    pub fn get_many_results<const N: usize>(&self, offset: u32) -> Option<[R; N]> {
        let mut results = [R::zeroed(); N];

        let flags = match R::flags() {
            QueryType::Pipeline(_) => vk::QueryResultFlags::WITH_AVAILABILITY,
            QueryType::Timestamp => {
                vk::QueryResultFlags::WITH_AVAILABILITY | vk::QueryResultFlags::TYPE_64
            }
        };

        unsafe {
            self.device
                .get_query_pool_results(self.handle, offset, &mut results, flags)
                .ok()
                .map(|()| results)
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
