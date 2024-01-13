/// Data that will be read only for the course of the program
pub struct Instance {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
}

impl Instance {
    pub fn new(
        surface: wgpu::Surface,
        device: wgpu::Device,
        queue: wgpu::Queue,
        camera_bind_group_layout: super::buffer::BindGroupLayout<1>,
    ) -> Self {
        Self {
            surface,
            device,
            queue,
            camera_bind_group_layout,
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
