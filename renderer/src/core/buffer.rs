use std::marker::PhantomData;

use wgpu::util::DeviceExt;

/// Buffer layout group with typed const length, to held invalidate any bugs for mismatched layouts and groups
pub struct BindGroupLayout<const N: usize> {
    layout: wgpu::BindGroupLayout,
}

impl<'a, const N: usize> Into<&'a wgpu::BindGroupLayout> for &'a BindGroupLayout<N> {
    fn into(self) -> &'a wgpu::BindGroupLayout {
        &self.layout
    }
}

impl<const N: usize> BindGroupLayout<N> {
    pub fn create(
        device: &wgpu::Device,
        entries: &[wgpu::BindGroupLayoutEntry; N],
        label: Option<&'static str>,
    ) -> Self {
        Self {
            layout: device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { entries, label }),
        }
    }
}
/// Buffer group that can be slotted into any bindings with the originating bind group layout
pub struct BufferGroup<const N: usize> {
    buffers: [wgpu::Buffer; N],
    bind_group: wgpu::BindGroup,
}

impl BufferGroup<1> {
    pub fn create_single_storage<T: bytemuck::Pod>(
        data: &[T],
        device: &wgpu::Device,
        layout: &BindGroupLayout<1>,
        label: Option<&str>,
    ) -> Self {
        Self::create_plural_storage(&[data], device, layout, label)
    }

    pub fn create_single<T: bytemuck::Pod>(
        data: &[T],
        usage: wgpu::BufferUsages,
        device: &wgpu::Device,
        layout: &BindGroupLayout<1>,
        label: Option<&str>,
    ) -> Self {
        Self::create_plural(&[data], &[usage], device, layout, label)
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffers[0]
    }
}

impl<const N: usize> BufferGroup<N> {
    pub fn create_plural_storage<T: bytemuck::Pod>(
        data: &[&[T]; N],
        device: &wgpu::Device,
        layout: &BindGroupLayout<N>,
        label: Option<&str>,
    ) -> Self {
        Self::create_plural(
            data,
            &[wgpu::BufferUsages::STORAGE; N],
            device,
            layout,
            label,
        )
    }

    pub fn create_plural<T: bytemuck::Pod>(
        data: &[&[T]; N],
        usages: &[wgpu::BufferUsages; N],
        device: &wgpu::Device,
        layout: &BindGroupLayout<N>,
        label: Option<&str>,
    ) -> Self {
        let buffers: [_; N] = data
            .iter()
            .enumerate()
            .map(|(i, datum)| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label,
                    contents: bytemuck::cast_slice(datum),
                    usage: usages[i],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let entries: [_; N] = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: layout.into(),
            entries: &entries[..],
        });

        Self {
            buffers,
            bind_group,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
