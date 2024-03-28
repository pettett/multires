use std::{sync::Arc};

use ash::vk;

use crate::utility::device::Device;

pub struct SyncObjects {
    device: Arc<Device>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
}
impl Drop for SyncObjects {
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.image_available_semaphores.len() {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
        }
    }
}
impl SyncObjects {
    pub fn new(device: Arc<Device>, max_frame_in_flight: usize) -> Self {
        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut in_flight_fences = vec![];

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..max_frame_in_flight {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!");
                let inflight_fence = device
                    .create_fence(&fence_create_info, None)
                    .expect("Failed to create Fence Object!");

                image_available_semaphores.push(image_available_semaphore);
                render_finished_semaphores.push(render_finished_semaphore);
                in_flight_fences.push(inflight_fence);
            }
        }

        SyncObjects {
            device,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        }
    }
}
