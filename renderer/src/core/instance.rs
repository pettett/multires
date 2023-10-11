/// Data that will be read only for the course of the program
pub struct Instance {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Instance {
    pub fn new(surface: wgpu::Surface, device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            surface,
            device,
            queue,
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
