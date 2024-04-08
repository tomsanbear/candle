use crate::{backend::BackendDevice, WebGPUStorage};

#[derive(Debug, Clone)]
pub struct WebGPUDevice {}

impl WebGPUDevice {}

impl BackendDevice for WebGPUDevice {
    type Storage = WebGPUStorage;

    fn new(_: usize) -> crate::Result<Self> {
        todo!()
    }

    fn location(&self) -> crate::DeviceLocation {
        todo!()
    }

    fn same_device(&self, _: &Self) -> bool {
        todo!()
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
