use std::sync::{Arc, Mutex};

use ash::vk;
use bevy_ecs::system::Resource;
use common::{Asset, MeshVert, MultiResMesh};
use common_renderer::components::gpu_mesh_util::{ClusterData, MultiResData};
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    multires::{self, GpuMeshlet},
    utility::buffer::TBuffer,
    Config,
};

#[derive(Resource)]
pub struct MeshData {
    pub cluster_count: u32,
    pub size: f32,
    pub vertex_buffer: Arc<TBuffer<MeshVert>>,
    pub meshlet_buffer: Arc<TBuffer<GpuMeshlet>>,
    pub stripped_meshlet_buffer: Arc<TBuffer<GpuMeshlet>>,
    pub cluster_buffer: Arc<TBuffer<ClusterData>>,
    pub meshlet_index_buffer: Arc<TBuffer<u32>>,
    pub index_buffer: Arc<TBuffer<u32>>,
}

impl MeshData {
    pub fn new(
        core: &Core,
        allocator: &Arc<Mutex<Allocator>>,
        graphics_queue: vk::Queue,
        config: &Config,
    ) -> Self {
        let data = MultiResMesh::load(&config.mesh_name).unwrap();
        // let data = MultiResMesh::load("assets/lucy.glb.bin").unwrap();

        let (cluster_order, cluster_groups) = data.order_clusters();
        let mut cluster_data = data.generate_cluster_data(&cluster_order, &cluster_groups);

        let size = cluster_data[0].radius;

        let (clusters, meshlets, stripped_meshlets) = multires::generate_meshlets(&cluster_order);

        assert_eq!(clusters.len(), cluster_data.len());

        let (meshlet_indices, _partitions, _groups) =
            data.indices_partitions_groups(&cluster_order);

        let cluster_count = clusters.len() as _;

        for (i, cluster) in clusters.into_iter().enumerate() {
            cluster_data[i].meshlet_start = cluster.meshlet_start;
            cluster_data[i].meshlet_count = cluster.meshlet_count;
        }

        let vertex_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
            &data.verts,
            "Vertex Buffer",
        );

        let meshlet_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &meshlets,
            "Meshlet Buffer",
        );

        let stripped_meshlet_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &stripped_meshlets,
            "Stripped Meshlet Buffer",
        );

        let cluster_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &cluster_data,
            "Submesh Buffer",
        );

        let index_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            // Allow index use for testing
            vk::BufferUsageFlags::INDEX_BUFFER,
            &data.full_indices,
            "Indices Buffer",
        );
        let meshlet_index_buffer = TBuffer::new_filled(
            core,
            allocator.clone(),
            graphics_queue,
            // Allow index use for testing
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            &meshlet_indices,
            "Indices Buffer",
        );
        Self {
            vertex_buffer,
            size,
            meshlet_buffer,
            stripped_meshlet_buffer,
            cluster_buffer,
            meshlet_index_buffer,
            index_buffer,
            cluster_count,
        }
    }
}
