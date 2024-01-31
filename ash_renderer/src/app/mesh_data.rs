use std::sync::{Arc, Mutex};

use ash::vk;
use bevy_ecs::system::Resource;
use common::{asset::Asset, MeshVert, MultiResMesh};
use common_renderer::components::gpu_mesh_util::{ClusterData, MultiResData};
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    multires::{self, GpuMeshlet},
    utility::buffer::{Buffer, TBuffer},
};

#[derive(Resource)]
pub struct MeshDataBuffers {
    pub vertex_buffer: Arc<TBuffer<MeshVert>>,
    pub meshlet_buffer: Arc<TBuffer<GpuMeshlet>>,
    pub cluster_buffer: Arc<TBuffer<ClusterData>>,
    pub index_buffer: Arc<TBuffer<u32>>,
}

impl MeshDataBuffers {
    pub fn new(core: &Core, allocator: &Arc<Mutex<Allocator>>, graphics_queue: vk::Queue) -> Self {
        let data = MultiResMesh::load("assets/torrin_main.glb.bin").unwrap();

        let (cluster_order, cluster_groups) = data.order_clusters();
        let mut cluster_data = data.generate_cluster_data(&cluster_order, &cluster_groups);

        let (clusters, meshlets) = multires::generate_meshlets(&cluster_order);

        let (indices, partitions, groups) = data.indices_partitions_groups(&cluster_order);

        for (i, submesh) in clusters.into_iter().enumerate() {
            cluster_data[i].meshlet_start = submesh.meshlet_start;
            cluster_data[i].meshlet_count = submesh.meshlet_count;
        }

        let vertex_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            &data.verts,
            "Vertex Buffer",
        );

        let meshlet_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &meshlets,
            "Meshlet Buffer",
        );

        let cluster_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &cluster_data,
            "Submesh Buffer",
        );

        let index_buffer = TBuffer::new_filled(
            &core,
            allocator.clone(),
            &core.command_pool,
            graphics_queue,
            // Allow index use for testing
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
            "Indices Buffer",
        );
        Self {
            vertex_buffer,
            meshlet_buffer,
            cluster_buffer,
            index_buffer,
        }
    }
}
