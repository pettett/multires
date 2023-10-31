/// Data that will be read only for the course of the program
pub struct Instance {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
    model_bind_group_layout: super::buffer::BindGroupLayout<1>,
    partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
}

impl Instance {
    pub fn new(
        surface: wgpu::Surface,
        device: wgpu::Device,
        queue: wgpu::Queue,
        camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
        model_bind_group_layout: super::buffer::BindGroupLayout<1>,
        partition_bind_group_layout: super::buffer::BindGroupLayout<2>,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            camera_bind_group_layout,
            model_bind_group_layout,
            partition_bind_group_layout,
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

    pub fn camera_bind_group_layout(&self) -> &super::buffer::BindGroupLayout<1> {
        &self.camera_bind_group_layout
    }
    pub fn model_bind_group_layout(&self) -> &super::buffer::BindGroupLayout<1> {
        &self.model_bind_group_layout
    }
    pub fn partition_bind_group_layout(&self) -> &super::buffer::BindGroupLayout<2> {
        &self.partition_bind_group_layout
    }
}
