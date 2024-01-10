/// Data that will be read only for the course of the program
pub struct Instance {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
    pub model_bind_group_layout: super::buffer::BindGroupLayout<1>,
    pub partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
    pub write_compute_bind_group_layout: super::buffer::BindGroupLayout<1>,
    pub cluster_info_buffer_bind_group_layout: super::buffer::BindGroupLayout<2>,
    pub result_indices_buffer_bind_group_layout: super::buffer::BindGroupLayout<2>,
    pub read_compute_bind_group_layout: super::buffer::BindGroupLayout<1>,
}

impl Instance {
    pub fn new(
        surface: wgpu::Surface,
        device: wgpu::Device,
        queue: wgpu::Queue,
        camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
        model_bind_group_layout: super::buffer::BindGroupLayout<1>,
        partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
        write_compute_bind_group_layout: super::buffer::BindGroupLayout<1>,
        cluster_info_buffer_bind_group_layout: super::buffer::BindGroupLayout<2>,
        result_indices_buffer_bind_group_layout: super::buffer::BindGroupLayout<2>,
        read_compute_bind_group_layout: super::buffer::BindGroupLayout<1>,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            camera_bind_group_layout,
            model_bind_group_layout,
            partition_bind_group_layout,
            write_compute_bind_group_layout,
            cluster_info_buffer_bind_group_layout,
            result_indices_buffer_bind_group_layout,
            read_compute_bind_group_layout,
        }
    }

    pub fn surface(&self) -> &wgpu::Surface {
        &self.surface
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
