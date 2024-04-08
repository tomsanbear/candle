use std::sync::Arc;

use wgpu::{Buffer, BufferUsages, CommandEncoder};

use crate::{backend::BackendStorage, DType, WebGPUDevice, WebGPUError};

#[derive(Debug, Clone)]
pub struct WebGPUStorage {
    /// The buffer that holds the output data
    buffer: Arc<Buffer>,

    /// The encoder that is used for processing the op
    encoder: Arc<Option<CommandEncoder>>,

    /// The parent storage that this storage is derived from
    /// This is used for computing the operation graph
    parent: Arc<Option<WebGPUStorage>>,

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
        parent: Option<Self>,
    ) -> Self {
        Self {
            buffer,
            parent: Arc::new(parent),
            encoder: Arc::new(None),
            dtype,
            device,
        }
    }

    /// Waits for the gpu to finish processing and unlock reading from the staging buffer
    pub fn synchronize(&self) -> crate::Result<()> {
        todo!()
    }

    /// Returns the buffer
    pub fn buffer(&self) -> Arc<Buffer> {
        self.buffer.clone()
    }

    /// Returns the parent storage
    pub fn parent(&self) -> Arc<Option<WebGPUStorage>> {
        self.parent.clone()
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
        todo!()
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> crate::Result<Self> {
        if mul == 1.0 && add == 0.0 {
            return Ok(self.clone());
        }

        let device = self.device.clone();
        let dtype = self.dtype;

        let shape = layout.shape();
        let el = shape.elem_count();

        let output_buffer = device.allocate_buffer(
            el,
            BufferUsages::STORAGE | BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            "affine",
        )?;

        Ok(Self::new(
            output_buffer.clone(),
            dtype,
            device,
            Some(self.clone()),
        ))
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

        let output_buffer = device.allocate_output_buffer(el, &name)?;

        let op = candle_webgpu_kernels::ops::unary::UnaryOp {
            dtype: "f32".to_string(),
            input_shape: layout.shape().dims().to_vec(),
            op: name,
        };

        op.run(
            &device.device,
            &device.queue,
            &self.buffer,
            &output_buffer,
            None,
        )
        .map_err(WebGPUError::from)?;

        Ok(Self::new(
            output_buffer.clone(),
            dtype,
            device,
            Some(self.clone()),
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

        let shape = lhs_layout.shape();
        let el = shape.elem_count();
        let name = B::KERNEL.to_string();

        let output_buffer = device.allocate_output_buffer(el, &name)?;

        let op = candle_webgpu_kernels::ops::binary::BinaryOp {
            dtype: "f32".to_string(),
            lhs_shape: lhs_layout.shape().dims().to_vec(),
            rhs_shape: rhs_layout.shape().dims().to_vec(),
            op: name,
        };

        let encoder = op
            .run(
                &device.device,
                &device.queue,
                &self.buffer,
                &rhs.buffer,
                &output_buffer,
                None,
            )
            .map_err(WebGPUError::from)?;

        Ok(Self::new(
            output_buffer.clone(),
            dtype,
            device,
            Some(self.clone()),
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

        let a = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();
        let b = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();

        let c = a.add(&b).unwrap();
        println!("{:?}", c);
    }
}
