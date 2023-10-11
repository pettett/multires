use wgpu::util::DeviceExt;

pub struct Buffer {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Buffer {
    pub fn create_storage<T: bytemuck::Pod>(
        data: &[T],
        device: &wgpu::Device,
        label: Option<&str>,
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label,
        });

        Self { buffer, bind_group }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
