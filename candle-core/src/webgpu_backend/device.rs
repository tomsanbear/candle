use std::sync::Arc;

use crate::{backend::BackendDevice, Result, WebGPUStorage};

use super::Block;

#[derive(Debug, Clone)]
pub struct WebGPUDevice {
    pub(crate) id: u64,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
}

impl WebGPUDevice {
    pub fn wait_until_completed(&self) -> Result<()> {
        todo!()
    }

    pub fn webgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn webgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn allocate_buffer(
        &self,
        size: usize,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Result<Arc<wgpu::Buffer>> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
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

    fn storage_from_cpu_storage(&self, _: &crate::CpuStorage) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage_owned(&self, _: crate::CpuStorage) -> crate::Result<Self::Storage> {
        todo!()
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
