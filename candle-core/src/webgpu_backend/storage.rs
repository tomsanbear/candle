use core::panic;
use std::sync::{Arc, RwLock};

use wgpu::{Buffer, BufferUsages, CommandEncoder};

use crate::{backend::BackendStorage, CpuStorage, DType, Result, WebGPUDevice, WebGPUError};

#[derive(Debug, Clone)]
pub struct WebGPUStorage {
    /// The buffer that holds the output data
    buffer: Arc<Buffer>,

    staging_buffer: Arc<RwLock<Option<Buffer>>>,

    /// The encoder that is used for processing the op
    encoder: Arc<RwLock<Option<CommandEncoder>>>,

    /// The parent storage that this storage is derived from
    /// This is used for computing the operation graph
    parent: Option<Arc<WebGPUStorage>>,

    /// The data type of the storage
    dtype: DType,

    /// The device that the storage is associated with
    device: WebGPUDevice,
}

impl WebGPUStorage {
    pub fn new(
        buffer: Arc<Buffer>,
        dtype: DType,
        device: WebGPUDevice,
        parent: Option<Arc<Self>>,
        encoder: Arc<RwLock<Option<CommandEncoder>>>,
    ) -> Self {
        Self {
            buffer,
            parent,
            encoder,
            dtype,
            device,
            staging_buffer: Arc::new(RwLock::new(None)),
        }
    }

    /// Submits the storage to the underlying device queue
    pub fn submit(&self) -> crate::Result<()> {
        if let Some(parent) = self.parent.as_ref() {
            parent.submit()?;
        }
        if self.encoder.try_read().unwrap().is_some() {
            let mut encoder = self.encoder.try_write().unwrap().take().unwrap();
            if let Some(staging_buffer) = self.staging_buffer.try_read().unwrap().as_ref() {
                encoder.copy_buffer_to_buffer(
                    &self.buffer,
                    0,
                    staging_buffer,
                    0,
                    self.buffer.size(),
                );
            }
            self.device.queue.submit(std::iter::once(encoder.finish()));
        }
        Ok(())
    }

    /// Returns the buffer
    pub fn buffer(&self) -> Arc<Buffer> {
        self.buffer.clone()
    }

    /// Returns the parent storage
    pub fn parent(&self) -> Option<Arc<WebGPUStorage>> {
        self.parent.clone()
    }

    /// Allocates a staging buffer
    pub fn allocate_staging_buffer(&self) -> Result<()> {
        let mut staging_buffer = self.staging_buffer.write().unwrap();
        if staging_buffer.is_none() {
            *staging_buffer = Some(self.device.allocate_buffer(
                self.buffer.size(),
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                "staging",
            )?);
        }
        Ok(())
    }

    pub(crate) fn to_cpu<T: std::fmt::Debug + Clone + bytemuck::Pod>(&self) -> Result<Vec<T>> {
        // setup the staging buffer
        self.allocate_staging_buffer()?;

        // evaluate the graph and queue the encoders
        self.submit()?;

        // Note that we're not calling `.await` here.
        let staging_buffer = self.staging_buffer.try_write().unwrap();
        let staging_buffer = staging_buffer.as_ref().unwrap();
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::unbounded();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device
            .device
            .poll(wgpu::Maintain::wait())
            .panic_on_timeout();

        // receive the data from the buffer until the buffer is empty
        if let Ok(Ok(())) = receiver.recv() {
            let result = {
                let data = buffer_slice.get_mapped_range();
                let result = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
                result
            };
            Ok(result)
        } else {
            panic!("Failed to receive data from the buffer");
        }
    }
}

impl BackendStorage for WebGPUStorage {
    type Device = WebGPUDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> crate::DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    fn affine(&self, _layout: &crate::Layout, _mul: f64, _add: f64) -> crate::Result<Self> {
        todo!()
    }

    fn powf(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn elu(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn reduce_op(
        &self,
        _: crate::op::ReduceOp,
        _: &crate::Layout,
        _: &[usize],
    ) -> crate::Result<Self> {
        todo!()
    }

    fn cmp(
        &self,
        _: crate::op::CmpOp,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &crate::Layout, _: crate::DType) -> crate::Result<Self> {
        todo!()
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> crate::Result<Self> {
        let device = self.device.clone();
        let dtype = self.dtype;

        let shape = layout.shape();
        let el = shape.elem_count();

        let name = B::KERNEL.to_string();

        let output_buffer = Arc::new(device.allocate_output_buffer(el as u64, &name)?);

        let op = candle_webgpu_kernels::ops::unary::UnaryOp {
            dtype: "f32".to_string(),
            input_shape: layout.shape().dims().to_vec(),
            op: name,
        };

        let command_encoder = op
            .run(&device.device, &self.buffer, &output_buffer)
            .map_err(WebGPUError::from)?;

        Ok(Self::new(
            output_buffer.clone(),
            dtype,
            device,
            Some(Arc::new(self.clone())),
            command_encoder,
        ))
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &crate::Layout,
        rhs_layout: &crate::Layout,
    ) -> crate::Result<Self> {
        let device = self.device.clone();
        let dtype = self.dtype;

        let name = B::KERNEL.to_string();

        let output_buffer = Arc::new(device.allocate_output_buffer(self.buffer.size(), &name)?);

        let op = candle_webgpu_kernels::ops::binary::BinaryOp {
            dtype: "f32".to_string(),
            lhs_shape: lhs_layout.shape().dims().to_vec(),
            rhs_shape: rhs_layout.shape().dims().to_vec(),
            op: name,
        };

        let encoder = op
            .run(&device.device, &self.buffer, &rhs.buffer, &output_buffer)
            .map_err(WebGPUError::from)?;

        Ok(Self::new(
            output_buffer.clone(),
            dtype,
            device,
            Some(Arc::new(self.clone())),
            encoder,
        ))
    }

    fn where_cond(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn avg_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn max_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn gather(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn scatter_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn index_select(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> crate::Result<()> {
        todo!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> crate::Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[test]
    fn test_add() {
        let device = Device::new_webgpu(0).unwrap();

        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.to_vec1::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }
}
