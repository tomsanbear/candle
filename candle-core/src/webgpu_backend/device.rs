use std::sync::{Arc, RwLock};

use crate::{
    backend::{BackendDevice, BackendStorage},
    CpuStorage, Result, WebGPUStorage,
};

use super::Block;
use wgpu::util::DeviceExt;

fn round_up_to_multiple_of_4(n: u64) -> u64 {
    (n + 3) & !3
}

#[derive(Debug, Clone)]
pub struct WebGPUDevice {
    pub(crate) id: u64,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl WebGPUDevice {
    pub fn wait_until_completed(&self) -> Result<()> {
        self.queue.submit(std::iter::empty());
        Ok(())
    }

    pub fn webgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn webgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn allocate_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Result<wgpu::Buffer> {
        let size = round_up_to_multiple_of_4(size) as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }

    pub fn allocate_output_buffer(&self, size: u64, label: &str) -> Result<wgpu::Buffer> {
        self.allocate_buffer(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            label,
        )
    }

    pub fn allocate_buffer_with_data<T>(&self, data: &[T], label: &str) -> Result<Arc<wgpu::Buffer>>
    where
        T: bytemuck::Pod,
    {
        let device = self.device.clone();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Arc::new(buffer))
    }
}

impl BackendDevice for WebGPUDevice {
    type Storage = WebGPUStorage;

    fn new(_: usize) -> Result<Self> {
        let instance = wgpu::Instance::default();

        // TODO: replace wait with event loop
        let adapter = instance
            .request_adapter(&Default::default())
            .wait()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .wait()
            .unwrap();

        Ok(Self {
            id: device.global_id().inner(),
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::WebGPU { gpu_id: self.id }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn ones_impl(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    unsafe fn alloc_uninit(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        let (_, buffer) = match storage {
            CpuStorage::U8(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::U32(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::I64(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::BF16(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::F16(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::F32(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
            CpuStorage::F64(storage) => (
                storage.len(),
                self.allocate_buffer_with_data(&storage, "todo"),
            ),
        };
        Ok(Self::Storage::new(
            buffer?,
            storage.dtype(),
            self.clone(),
            None,
            Arc::new(RwLock::new(None)),
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> crate::Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        _: &crate::Shape,
        _: crate::DType,
        _: f64,
        _: f64,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(
        &self,
        _: &crate::Shape,
        _: crate::DType,
        _: f64,
        _: f64,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_device() {
        WebGPUDevice::new(0).unwrap();
    }
}
