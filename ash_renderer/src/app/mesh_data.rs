use std::sync::{Arc, Mutex};

use ash::vk;
use bevy_ecs::system::Resource;
use common::{Asset, MeshCluster, MeshVert, MultiResMesh};
use common_renderer::components::gpu_mesh_util::{ClusterData, MultiResData};
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;

use crate::{
    core::Core,
    multires::{self, GpuMeshlet},
    utility::buffer::TBuffer,
    Config,
};

pub struct LODChainLevel {
    pub index_buffer: Arc<TBuffer<u32>>,
    pub error: f32,
    pub radius: f32,
}

pub struct MeshData {
    pub cluster_count: u32,
    pub size: f32,
    pub vertex_buffer: Arc<TBuffer<MeshVert>>,
    pub meshlet_buffer: Arc<TBuffer<GpuMeshlet>>,
    pub stripped_meshlet_buffer: Arc<TBuffer<GpuMeshlet>>,
    pub cluster_buffer: Arc<TBuffer<ClusterData>>,
    pub meshlet_index_buffer: Arc<TBuffer<u32>>,
    pub lod_chain: Vec<LODChainLevel>,
}

impl MeshData {
    pub fn new(
        core: &Core,
        allocator: &Arc<Mutex<Allocator>>,
        graphics_queue: vk::Queue,
        mesh_name: &str,
    ) -> Self {
        let data = MultiResMesh::load(mesh_name).unwrap();
        // let data = MultiResMesh::load("assets/lucy.glb.bin").unwrap();

        let (cluster_order, cluster_groups) = data.order_clusters();
        let mut cluster_data = data.generate_cluster_data(&cluster_order, &cluster_groups);

        let levels = cluster_order[0].lod + 1;
        let mut indices = vec![Vec::new(); levels];
        let mut sum_errors = vec![10000.0; levels];
        let mut sum_rads = vec![0.0; levels];
        // let mut sums = vec![0.0; levels];

        for c in &cluster_order {
            for m in c.meshlets() {
                m.calc_indices_to_vec(&mut indices[c.lod]);
            }

            // sum_errors[c.lod] += c.error();
            // sum_rads[c.lod] += c.saturated_bound.radius();
            // sums[c.lod] += 1.0;

            if sum_errors[c.lod] > c.error() {
                sum_errors[c.lod] = c.error();
                sum_rads[c.lod] = c.saturated_bound.radius();
            }
        }

        // for l in 0..levels{
        // 	sum_errors[l] /= sums[l] ;
        // 	sum_rads[l] /= sums[l] ;
        // }

        let mut lod_chain = Vec::new();

        for (l, (error, radius)) in indices
            .into_iter()
            .zip(sum_errors.into_iter().zip(sum_rads.into_iter()))
        {
            let index_buffer = TBuffer::new_filled(
                core,
                allocator.clone(),
                graphics_queue,
                // Allow index use for testing
                vk::BufferUsageFlags::INDEX_BUFFER,
                &l,
                "Indices Buffer",
            );
            // let error = 500025.0 / l.len() as f32;
            // measure edge lengths
            // for t in l.chunks(3) {
            //     let [t1, t2, t3] = t else { unreachable!() };
            //     for (a, b) in [(t1, t2), (t2, t3), (t3, t1)] {
            //         error += Vec3::from_slice(&data.verts[*a as usize].pos)
            //             .distance(Vec3::from_slice(&data.verts[*b as usize].pos));
            //     }
            // }

            // error /= 200000000;

            // println!("Size: {}, Error: {}", l.len(), error);
            lod_chain.push(LODChainLevel {
                index_buffer,
                error,
                radius,
            })
        }

        // assert_eq!(
        //     indices[0].len(),
        //     data.full_indices.len(),
        //     "LOD0 inconsistent"
        // );

        let size = cluster_data[0].radius;

        println!("{}  ---  Size: {}", mesh_name, size);

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
            lod_chain,
            cluster_count,
        }
    }
}
