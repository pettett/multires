// https://github.com/MatchaChoco010/egui-winit-ash-integration/blob/main/src/gpu_allocator.rs
// With bumped version
use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpu_allocator::vulkan::*;

use egui_winit_ash_integration::{AllocationCreateInfoTrait, AllocationTrait, AllocatorTrait};

#[repr(transparent)]
pub struct GpuAllocator(pub Arc<Mutex<Allocator>>);
#[repr(transparent)]
pub struct GpuAllocation(Allocation);

impl Into<Allocation> for GpuAllocation {
    fn into(self) -> Allocation {
        self.0
    }
}

#[repr(transparent)]
pub struct GpuAllocationCreateDesc(AllocationCreateDesc<'static>);

impl Into<AllocationCreateDesc<'static>> for GpuAllocationCreateDesc {
    fn into(self) -> AllocationCreateDesc<'static> {
        self.0
    }
}

impl AllocationTrait for GpuAllocation {
    unsafe fn memory(&self) -> ash::vk::DeviceMemory {
        Allocation::memory(&self.0)
    }

    fn offset(&self) -> u64 {
        Allocation::offset(&self.0)
    }

    fn size(&self) -> u64 {
        Allocation::size(&self.0)
    }

    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        Allocation::mapped_ptr(&self.0)
    }
}

impl AllocationCreateInfoTrait for GpuAllocationCreateDesc {
    fn new(
        requirements: ash::vk::MemoryRequirements,
        location: egui_winit_ash_integration::MemoryLocation,
        linear: bool,
    ) -> Self {
        Self(AllocationCreateDesc {
            name: "egui-winit-ash-integration",
            requirements,
            location: match location {
                egui_winit_ash_integration::MemoryLocation::Unknown => {
                    gpu_allocator::MemoryLocation::Unknown
                }
                egui_winit_ash_integration::MemoryLocation::GpuOnly => {
                    gpu_allocator::MemoryLocation::GpuOnly
                }
                egui_winit_ash_integration::MemoryLocation::CpuToGpu => {
                    gpu_allocator::MemoryLocation::CpuToGpu
                }
                egui_winit_ash_integration::MemoryLocation::GpuToCpu => {
                    gpu_allocator::MemoryLocation::GpuToCpu
                }
            },
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
    }
}

impl AllocatorTrait for GpuAllocator {
    type Allocation = GpuAllocation;
    type AllocationCreateInfo = GpuAllocationCreateDesc;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> Result<Self::Allocation> {
        Ok(GpuAllocation(Allocator::allocate(
            &mut self.0.lock().unwrap(),
            &desc.0,
        )?))
    }

    fn free(&self, allocation: Self::Allocation) -> Result<()> {
        Ok(Allocator::free(&mut self.0.lock().unwrap(), allocation.0)?)
    }
}
