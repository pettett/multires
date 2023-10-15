// use std::marker::PhantomData;

// use vulkano::{
//     buffer::{Buffer, BufferCreateInfo, BufferUsage},
//     descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet, WriteDescriptorSet},
//     device::Device,
//     memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage},
// };
// use wgpu::util::DeviceExt;

// /// Buffer layout group with typed const length, to held invalidate any bugs for mismatched layouts and groups
// /// Buffer group that can be slotted into any bindings with the originating bind group layout
// pub struct BufferGroup<T: bytemuck::Pod + Send + Sync, const N: usize> {
//     buffers: [Buffer<T>; N],
//     bind_group: PersistentDescriptorSet,
// }

// impl<T: bytemuck::Pod + Send + Sync> BufferGroup<T, 1> {
//     pub fn create_single_storage<T: bytemuck::Pod + Send + Sync>(
//         data: Vec<T>,
//         device: &Device,
//         layout: PersistentDescriptorSet,
//         memory_allocator: &(impl MemoryAllocator + ?Sized),
//         label: Option<&str>,
//     ) -> Self {
//         Self::create_plural_storage(&[data], device, layout, memory_allocator, label)
//     }

//     pub fn create_single<T: bytemuck::Pod + Send + Sync>(
//         data: Vec<T>,
//         usage: BufferUsage,
//         device: &Device,
//         layout: PersistentDescriptorSet,
//         memory_allocator: &(impl MemoryAllocator + ?Sized),
//         label: Option<&str>,
//     ) -> Self {
//         Self::create_plural(&[data], &[usage], device, layout, memory_allocator, label)
//     }

//     pub fn buffer(&self) -> &Buffer {
//         &self.buffers[0]
//     }
// }

// impl<const N: usize> BufferGroup<N> {
//     pub fn create_plural_storage<T: bytemuck::Pod + Send + Sync>(
//         data: &[Vec<T>; N],
//         device: &Device,
//         layout: PersistentDescriptorSet,
//         memory_allocator: &(impl MemoryAllocator + ?Sized),
//         label: Option<&str>,
//     ) -> Self {
//         Self::create_plural(
//             data,
//             &[BufferUsage::STORAGE_BUFFER; N],
//             device,
//             layout,
//             memory_allocator,
//             label,
//         )
//     }

//     pub fn create_plural<T: bytemuck::Pod + Send + Sync>(
//         data: &[Vec<T>; N],
//         usages: &[BufferUsage; N],
//         device: &Device,
//         layout: PersistentDescriptorSet,
//         memory_allocator: &(impl MemoryAllocator + ?Sized),
//         label: Option<&str>,
//     ) -> Self {
//         let buffers: [_; N] = data
//             .iter()
//             .enumerate()
//             .map(|(i, datum)| {
//                 Buffer::from_iter(
//                     memory_allocator,
//                     BufferCreateInfo {
//                         usage: usages[i],
//                         ..Default::default()
//                     },
//                     AllocationCreateInfo {
//                         usage: MemoryUsage::Upload,
//                         ..Default::default()
//                     },
//                     datum.clone(),
//                 )
//                 .expect("failed to create camera buffer")
//             })
//             .collect::<Vec<_>>()
//             .try_into()
//             .unwrap();

//         let entries: [_; N] = buffers
//             .iter()
//             .enumerate()
//             .map(|(i, buffer)| wgpu::BindGroupEntry {
//                 binding: i as u32,
//                 resource: buffer.as_entire_binding(),
//             })
//             .collect::<Vec<_>>()
//             .try_into()
//             .unwrap();

//         let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//             label,
//             layout: layout.into(),
//             entries: &entries[..],
//         });

//         Self {
//             buffers,
//             bind_group,
//         }
//     }

//     pub fn bind_group(&self) -> &wgpu::BindGroup {
//         &self.bind_group
//     }
// }
