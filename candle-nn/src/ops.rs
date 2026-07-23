//! Tensor ops.
//!

use candle::{CpuStorage, DType, Layout, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.silu()
}

pub fn swiglu(xs: &Tensor) -> Result<Tensor> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

struct Sigmoid;

impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use candle::cuda_backend::SlicePtrOrNull;
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let shape = layout.shape();
                let dims = shape.dims();
                let el_count = shape.elem_count();
                let cfg = LaunchConfig::for_num_elems(el_count as u32);
                let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;
                let src = &src.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(&kernel_name::<T>("usigmoid"), &kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<T>(el_count)? };

                let mut builder = func.builder();
                candle::builder_arg!(builder, el_count, dims.len());
                ds.builder_arg(&mut builder);
                builder.arg(src);
                builder.arg(&out);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;
        let device = storage.device();
        let dtype = storage.dtype();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device
            .new_buffer_builder()
            .with_size_for(el_count, dtype)
            .with_label("sigmoid")
            .build()?;
        let encoder = device.command_encoder()?;
        encoder.set_label("sigmoid");
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };

        if layout.is_contiguous() {
            use candle_metal_kernels::unary::contiguous;
            let kernel_name = match dtype {
                DType::F16 => contiguous::sigmoid::HALF,
                DType::F32 => contiguous::sigmoid::FLOAT,
                DType::BF16 => contiguous::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal contiguous unary sigmoid {dtype:?} not implemented")
                }
            };
            candle_metal_kernels::call_unary_contiguous(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                dtype.size_in_bytes(),
                el_count,
                src,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match dtype {
                DType::F16 => strided::sigmoid::HALF,
                DType::F32 => strided::sigmoid::FLOAT,
                DType::BF16 => strided::sigmoid::BFLOAT,
                dtype => {
                    candle::bail!("Metal strided unary sigmoid {dtype:?} not implemented")
                }
            };
            let dst = candle_metal_kernels::BufferOffset::zero_offset(&buffer);
            candle_metal_kernels::call_unary_strided(
                device.metal_device(),
                &encoder,
                device.kernels(),
                kernel_name,
                layout.dims(),
                src,
                layout.stride(),
                dst,
            )
            .map_err(MetalError::from)?;
        }

        let new_storage = candle::MetalStorage::new(buffer, device.clone(), el_count, dtype);
        Ok((new_storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid(xs: &Tensor) -> Result<Tensor> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    xs * (1.0 + xs.exp()?)?.log()?.tanh()
}

pub fn leaky_relu(xs: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn selu(xs: &Tensor, alpha: f32, gamma: f32) -> Result<Tensor> {
    let is_pos = xs.gt(0f32)?;
    let alpha_t = Tensor::full(alpha, xs.dims(), xs.device())?;
    let neg = xs.exp()?.mul(&alpha_t)?.sub(&alpha_t)?;
    let selu = is_pos.where_cond(xs, &neg)?;
    let gamma_t = Tensor::full(gamma, xs.dims(), xs.device())?;
    selu.broadcast_mul(&gamma_t)
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map1, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("softmax"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                candle::builder_arg!(builder, n_cols as i32);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = storage.device();
        let slice = S.map(&storage.slice, dev, layout)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = storage.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax");
        let kernels = device.kernels();
        let name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle::bail!("softmax-last-dim is not implemented for {dtype:?}"),
        };

        let n = layout.stride().len();
        if !(layout.is_contiguous() && layout.stride()[n - 1] == 1) {
            candle::bail!("Non contiguous softmax-last-dim is not implemented");
        }

        let last_dim = layout.dims()[layout.shape().rank() - 1];
        let elem_count = layout.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, storage.dtype())
            .with_label("softmax")
            .build()?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, storage.dtype());
        Ok((newstorage, layout.shape().clone()))
    }
}

pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let n_rows = el_count / dim_m1;
            let mut dst = vec![T::zero(); el_count];

            fn rms_row<
                T: candle::WithDType
                    + num_traits::Float
                    + num_traits::AsPrimitive<f32>
                    + num_traits::FromPrimitive,
            >(
                src: &[T],
                alpha: &[T],
                n: usize,
                eps: f32,
                dst: &mut [T],
            ) {
                let sum2 = src
                    .iter()
                    .map(|&v| {
                        let v = v.as_();
                        v * v
                    })
                    .sum::<f32>();
                let m = (sum2 / n as f32 + eps).sqrt();
                let m = T::from_f32(m).unwrap_or_else(T::nan);
                for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                    *d = *s / m * *alpha
                }
            }

            if n_rows <= 32 {
                let n = dim_m1;
                for row in 0..n_rows {
                    let src = &src[row * n..(row + 1) * n];
                    let dst = &mut dst[row * n..(row + 1) * n];
                    rms_row(src, alpha, n, eps, dst);
                }
            } else {
                src.par_chunks(dim_m1)
                    .zip(dst.par_chunks_mut(dim_m1))
                    .for_each(|(src, dst)| {
                        let n = src.len();
                        rms_row(src, alpha, n, eps, dst);
                    });
            }
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func = dev.get_or_load_func(&kernel_name::<T>("rmsnorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("rmsnorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle::bail!("rmsnorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous rmsnorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("rmsnorm")
            .build()?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

pub fn rms_norm(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps })
}

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
}

impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map3, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map3 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                beta: &CudaSlice<T>,
                beta_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let beta = match beta_layout.contiguous_offsets() {
                    None => candle::bail!("beta has to be contiguous"),
                    Some((o1, o2)) => beta.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("layernorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                builder.arg(&beta);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, &s3.slice, l3, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("layernorm");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("layernorm")
            .build()?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            Some((s3.buffer(), l3.start_offset() * s3.dtype().size_in_bytes())),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

#[derive(Debug, Clone)]
struct LayerNormNoBias {
    eps: f32,
}

impl candle::CustomOp2 for LayerNormNoBias {
    fn name(&self) -> &'static str {
        "layer-norm-no-bias"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha.iter()) {
                        let d_ = (s.as_() - mean) * inv_std * alpha.as_();
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for layernorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            eps: f32,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                layout: &Layout,
                alpha: &CudaSlice<T>,
                alpha_layout: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let alpha = match alpha_layout.contiguous_offsets() {
                    None => candle::bail!("alpha has to be contiguous"),
                    Some((o1, o2)) => alpha.slice(o1..o2),
                };
                let el = layout.shape().elem_count();
                let dims = layout.shape().dims();
                let dim_m1 = dims[dims.len() - 1];
                let (n_rows, n_cols) = (el / dim_m1, dim_m1);

                let block_size = if n_cols < 1024 { 32 } else { 1024 };
                let cfg = LaunchConfig {
                    grid_dim: (n_rows as u32, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                let func =
                    dev.get_or_load_func(&kernel_name::<T>("layernorm"), &kernels::REDUCE)?;
                // SAFETY: Set later by running the kernel.
                let dst = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&src);
                builder.arg(&dst);
                builder.arg(&alpha);
                // Null beta: the kernel skips the bias add for this combination.
                builder.arg(&0usize);
                candle::builder_arg!(builder, n_cols as i32, block_size as i32, self.eps);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(dst)
            }
        }

        use candle::backend::BackendStorage;
        let dev = s1.device();
        let slice = S { eps: self.eps }.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("layernorm-no-bias");
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2) => candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, s1.dtype())
            .with_label("layernorm-no-bias")
            .build()?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &encoder,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            None,
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

pub fn layer_norm(xs: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}

/// Fused layer-norm without a bias term.
pub fn layer_norm_no_bias(xs: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?}",
            xs.shape(),
            alpha.shape(),
        )
    }
    xs.apply_op2_no_bwd(alpha, &LayerNormNoBias { eps })
}

/// Unary activation fusable into `matmul_bias`'s GEMM epilogue on Metal.
/// Each variant computes the same formula as the matching tensor op:
/// `relu`, `gelu` (the tanh approximation) or `silu`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatmulActivation {
    Relu,
    Gelu,
    Silu,
}

struct MatmulBias {
    #[allow(dead_code)] // only read by metal_fwd
    activation: Option<MatmulActivation>,
}

impl candle::CustomOp3 for MatmulBias {
    fn name(&self) -> &'static str {
        "matmul-bias"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("matmul-bias is fused on Metal only; the public fn composes elsewhere")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        lhs: &candle::MetalStorage,
        lhs_l: &Layout,
        rhs: &candle::MetalStorage,
        rhs_l: &Layout,
        bias: &candle::MetalStorage,
        bias_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;

        let lhs_dims = lhs_l.dims();
        let rhs_dims = rhs_l.dims();
        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            candle::bail!("matmul-bias expects at least 2d lhs/rhs");
        }
        let m = lhs_dims[lhs_dims.len() - 2];
        let k = lhs_dims[lhs_dims.len() - 1];
        let k2 = rhs_dims[rhs_dims.len() - 2];
        let n = rhs_dims[rhs_dims.len() - 1];
        let b: usize = lhs_dims[..lhs_dims.len() - 2].iter().product();
        let rhs_b: usize = rhs_dims[..rhs_dims.len() - 2].iter().product();
        if k != k2 || (rhs_b != 1 && rhs_b != b) {
            candle::bail!(
                "matmul-bias shape mismatch lhs {lhs_dims:?} rhs {rhs_dims:?} (rhs batch must be 1 or match)"
            );
        }
        if bias_l.dims() != [n] || !bias_l.is_contiguous() {
            candle::bail!(
                "matmul-bias bias must be a contiguous ({n},) vector, got {:?}",
                bias_l.dims()
            );
        }
        let dtype = match lhs.dtype() {
            DType::F32 => candle_metal_kernels::GemmDType::F32,
            DType::F16 => candle_metal_kernels::GemmDType::F16,
            DType::BF16 => candle_metal_kernels::GemmDType::BF16,
            dtype => candle::bail!("matmul-bias does not support {dtype:?}"),
        };
        if rhs.dtype() != lhs.dtype() || bias.dtype() != lhs.dtype() {
            candle::bail!("matmul-bias dtypes must match");
        }

        let activation = match self.activation {
            None => candle_metal_kernels::GemmActivation::None,
            Some(MatmulActivation::Relu) => candle_metal_kernels::GemmActivation::Relu,
            Some(MatmulActivation::Gelu) => candle_metal_kernels::GemmActivation::Gelu,
            Some(MatmulActivation::Silu) => candle_metal_kernels::GemmActivation::Silu,
        };

        let device = lhs.device();
        let elem_count = b * m * n;
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, lhs.dtype())
            .with_label("matmul-bias")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_mlx_gemm_with_bias(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            (b, m, n, k),
            lhs_l.stride(),
            lhs_l.start_offset() * lhs.dtype().size_in_bytes(),
            lhs.buffer(),
            rhs_l.stride(),
            rhs_l.start_offset() * rhs.dtype().size_in_bytes(),
            rhs.buffer(),
            Some((
                bias.buffer(),
                bias_l.start_offset() * bias.dtype().size_in_bytes(),
            )),
            activation,
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let mut out_dims = lhs_dims.to_vec();
        out_dims[lhs_dims.len() - 1] = n;
        let out_shape = Shape::from_dims(&out_dims);
        let storage = candle::MetalStorage::new(output, device.clone(), elem_count, lhs.dtype());
        Ok((storage, out_shape))
    }
}

struct GridSample {
    align_corners: bool,
}

/// Map a [-1, 1] grid coordinate to input pixel space, torch-style.
fn grid_unnormalize(coord: f32, size: usize, align_corners: bool) -> f32 {
    if align_corners {
        (coord + 1.0) / 2.0 * (size as f32 - 1.0)
    } else {
        ((coord + 1.0) * size as f32 - 1.0) / 2.0
    }
}

/// Bilinear sample with zeros padding at `(ix, iy)` in pixel space.
fn bilinear_sample<T: Copy>(
    plane: &[T],
    (h, w): (usize, usize),
    (ix, iy): (f32, f32),
    to_f32: fn(T) -> f32,
) -> f32 {
    let x0 = ix.floor();
    let y0 = iy.floor();
    let wx1 = ix - x0;
    let wy1 = iy - y0;
    let mut acc = 0f32;
    for (dy, wy) in [(0f32, 1.0 - wy1), (1.0, wy1)] {
        for (dx, wx) in [(0f32, 1.0 - wx1), (1.0, wx1)] {
            let x = x0 + dx;
            let y = y0 + dy;
            if x >= 0.0 && x < w as f32 && y >= 0.0 && y < h as f32 {
                acc += wx * wy * to_f32(plane[y as usize * w + x as usize]);
            }
        }
    }
    acc
}

fn grid_sample_cpu<T: Copy + Send + Sync>(
    input: &[T],
    grid: &[T],
    (n, c, h, w): (usize, usize, usize, usize),
    (h_out, w_out): (usize, usize),
    align_corners: bool,
    to_f32: fn(T) -> f32,
    from_f32: fn(f32) -> T,
) -> Vec<T> {
    let mut out = vec![from_f32(0.0); n * c * h_out * w_out];
    out.par_chunks_mut(h_out * w_out)
        .enumerate()
        .for_each(|(nc, chunk)| {
            let (ni, ci) = (nc / c, nc % c);
            let plane = &input[(ni * c + ci) * h * w..][..h * w];
            let grid_n = &grid[ni * h_out * w_out * 2..][..h_out * w_out * 2];
            for (i, o) in chunk.iter_mut().enumerate() {
                let ix = grid_unnormalize(to_f32(grid_n[2 * i]), w, align_corners);
                let iy = grid_unnormalize(to_f32(grid_n[2 * i + 1]), h, align_corners);
                *o = from_f32(bilinear_sample(plane, (h, w), (ix, iy), to_f32));
            }
        });
    out
}

impl candle::CustomOp2 for GridSample {
    fn name(&self) -> &'static str {
        "grid-sample"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        grid: &CpuStorage,
        grid_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (dims, out_shape) = grid_sample_check(input_l, grid_l)?;
        let (n, c, h, w, h_out, w_out) = dims;
        let out = match (input, grid) {
            (CpuStorage::F32(input), CpuStorage::F32(grid)) => CpuStorage::F32(grid_sample_cpu(
                input,
                grid,
                (n, c, h, w),
                (h_out, w_out),
                self.align_corners,
                |v| v,
                |v| v,
            )),
            (CpuStorage::F16(input), CpuStorage::F16(grid)) => CpuStorage::F16(grid_sample_cpu(
                input,
                grid,
                (n, c, h, w),
                (h_out, w_out),
                self.align_corners,
                half::f16::to_f32,
                half::f16::from_f32,
            )),
            (CpuStorage::BF16(input), CpuStorage::BF16(grid)) => CpuStorage::BF16(grid_sample_cpu(
                input,
                grid,
                (n, c, h, w),
                (h_out, w_out),
                self.align_corners,
                half::bf16::to_f32,
                half::bf16::from_f32,
            )),
            _ => candle::bail!("grid-sample requires matching f32/f16/bf16 input and grid"),
        };
        Ok((out, out_shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &candle::CudaStorage,
        input_l: &Layout,
        grid: &candle::CudaStorage,
        grid_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use candle::cuda_backend::{kernel_name, kernels, Map2, WrapErr};
        use candle::{CudaDevice, WithDType};

        struct S {
            dims: GridSampleDims,
            align_corners: bool,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                input: &CudaSlice<T>,
                input_l: &Layout,
                grid: &CudaSlice<T>,
                grid_l: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let (n, c, h, w, h_out, w_out) = self.dims;
                let input = match input_l.contiguous_offsets() {
                    None => candle::bail!("grid-sample input has to be contiguous"),
                    Some((o1, o2)) => input.slice(o1..o2),
                };
                let grid = match grid_l.contiguous_offsets() {
                    None => candle::bail!("grid-sample grid has to be contiguous"),
                    Some((o1, o2)) => grid.slice(o1..o2),
                };
                let cfg = LaunchConfig::for_num_elems((n * h_out * w_out) as u32);
                let func = dev.get_or_load_func(&kernel_name::<T>("grid_sample"), &kernels::CONV)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<T>(n * c * h_out * w_out)? };
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                    h_out as u32,
                    w_out as u32,
                    self.align_corners as u32
                );
                builder.arg(&input);
                builder.arg(&grid);
                builder.arg(&out);
                // SAFETY: ffi.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        let (dims, out_shape) = grid_sample_check(input_l, grid_l)?;
        let dev = input.device();
        let slice = S {
            dims,
            align_corners: self.align_corners,
        }
        .map(&input.slice, input_l, &grid.slice, grid_l, dev)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, out_shape))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input: &candle::MetalStorage,
        input_l: &Layout,
        grid: &candle::MetalStorage,
        grid_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;

        let (dims, out_shape) = grid_sample_check(input_l, grid_l)?;
        let (n, c, h, w, h_out, w_out) = dims;
        if grid.dtype() != input.dtype() {
            candle::bail!("grid-sample requires matching input and grid dtypes");
        }
        let name = match input.dtype() {
            DType::F32 => "grid_sample_f32",
            DType::F16 => "grid_sample_f16",
            DType::BF16 => "grid_sample_bf16",
            dtype => candle::bail!("grid-sample does not support {dtype:?} on Metal"),
        };
        let device = input.device();
        let elem_count = out_shape.elem_count();
        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, input.dtype())
            .with_label("grid-sample")
            .build()?;
        let encoder = device.command_encoder()?;
        let input_src = candle_metal_kernels::BufferOffset {
            buffer: input.buffer(),
            offset_in_bytes: input_l.start_offset() * input.dtype().size_in_bytes(),
        };
        let grid_src = candle_metal_kernels::BufferOffset {
            buffer: grid.buffer(),
            offset_in_bytes: grid_l.start_offset() * grid.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_grid_sample(
            device.metal_device(),
            &encoder,
            device.kernels(),
            name,
            (n, c, h, w),
            (h_out, w_out),
            self.align_corners,
            input_src,
            grid_src,
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let storage = candle::MetalStorage::new(output, device.clone(), elem_count, input.dtype());
        Ok((storage, out_shape))
    }
}

type GridSampleDims = (usize, usize, usize, usize, usize, usize);

/// Validate layouts and shapes; returns (n, c, h, w, h_out, w_out) and the
/// output shape.
fn grid_sample_check(input_l: &Layout, grid_l: &Layout) -> Result<(GridSampleDims, Shape)> {
    if !input_l.is_contiguous() || !grid_l.is_contiguous() {
        candle::bail!("grid-sample requires contiguous input and grid")
    }
    let (n, c, h, w) = input_l.shape().dims4()?;
    let (gn, h_out, w_out, two) = grid_l.shape().dims4()?;
    if gn != n || two != 2 {
        candle::bail!(
            "grid-sample shape mismatch: input {:?} grid {:?} (want (n, h_out, w_out, 2))",
            input_l.shape(),
            grid_l.shape()
        )
    }
    Ok((
        (n, c, h, w, h_out, w_out),
        Shape::from_dims(&[n, c, h_out, w_out]),
    ))
}

/// Bilinear `grid_sample` with zeros padding — `torch.nn.functional.
/// grid_sample(input, grid, mode="bilinear", padding_mode="zeros")`
/// semantics. `input` is `(n, c, h, w)`, `grid` is `(n, h_out, w_out, 2)`
/// with x/y coordinates in `[-1, 1]`; the output is `(n, c, h_out, w_out)`.
pub fn grid_sample(input: &Tensor, grid: &Tensor, align_corners: bool) -> Result<Tensor> {
    input
        .contiguous()?
        .apply_op2_no_bwd(&grid.contiguous()?, &GridSample { align_corners })
}

/// Multiscale deformable attention (Deformable-DETR semantics), composed
/// on-device from [`grid_sample`]: per level, bilinear-sample the value
/// feature map at the sampling locations, then reduce with the attention
/// weights.
///
/// * `value`: `(n, len_v, heads, head_dim)` — flattened multi-level features
///   (levels concatenated along `len_v`).
/// * `spatial_shapes`: per-level `(h, w)`; their products must sum to
///   `len_v`.
/// * `sampling_locations`: `(n, len_q, heads, levels, points, 2)` in
///   `[0, 1]` (normalized per level).
/// * `attention_weights`: `(n, len_q, heads, levels, points)`.
///
/// Returns `(n, len_q, heads * head_dim)`.
pub fn ms_deform_attn(
    value: &Tensor,
    spatial_shapes: &[(usize, usize)],
    sampling_locations: &Tensor,
    attention_weights: &Tensor,
) -> Result<Tensor> {
    let (n, len_v, heads, head_dim) = value.dims4()?;
    let (sn, len_q, sh, levels, points, two) = match *sampling_locations.dims() {
        [a, b, c, d, e, f] => (a, b, c, d, e, f),
        _ => candle::bail!(
            "ms-deform-attn sampling_locations must be 6d, got {:?}",
            sampling_locations.shape()
        ),
    };
    if sn != n || sh != heads || two != 2 || levels != spatial_shapes.len() {
        candle::bail!(
            "ms-deform-attn shape mismatch: value {:?} sampling_locations {:?} ({} levels)",
            value.shape(),
            sampling_locations.shape(),
            spatial_shapes.len()
        )
    }
    let total: usize = spatial_shapes.iter().map(|&(h, w)| h * w).sum();
    if total != len_v {
        candle::bail!("ms-deform-attn: spatial shapes cover {total} positions, value has {len_v}")
    }
    if attention_weights.dims() != [n, len_q, heads, levels, points] {
        candle::bail!(
            "ms-deform-attn attention_weights shape {:?} does not match",
            attention_weights.shape()
        )
    }

    let mut level_outputs = Vec::with_capacity(levels);
    let mut offset = 0;
    for (lid, &(h, w)) in spatial_shapes.iter().enumerate() {
        // (n, h*w, heads, d) -> (n*heads, d, h, w)
        let v = value
            .narrow(1, offset, h * w)?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .reshape((n * heads, head_dim, h, w))?;
        // (n, len_q, heads, points, 2) -> (n*heads, len_q, points, 2), [0,1] -> [-1,1]
        let grid = sampling_locations
            .narrow(3, lid, 1)?
            .squeeze(3)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((n * heads, len_q, points, 2))?;
        let grid = ((grid * 2.0)? - 1.0)?;
        // (n*heads, d, len_q, points)
        level_outputs.push(grid_sample(&v, &grid, false)?);
        offset += h * w;
    }
    // (n*heads, d, len_q, levels*points)
    let sampled = Tensor::stack(&level_outputs, 3)?
        .reshape((n * heads, head_dim, len_q, levels * points))?;
    // (n, len_q, heads, levels, points) -> (n*heads, 1, len_q, levels*points)
    let weights = attention_weights
        .transpose(1, 2)?
        .contiguous()?
        .reshape((n * heads, 1, len_q, levels * points))?;
    let out = sampled.broadcast_mul(&weights)?.sum(D::Minus1)?; // (n*heads, d, len_q)
    out.reshape((n, heads, head_dim, len_q))?
        .permute((0, 3, 1, 2))?
        .contiguous()?
        .reshape((n, len_q, heads * head_dim))
}

struct TopK {
    /// Padded k (power of two); written into the last dim of the index tensor.
    k_pad: usize,
}

impl candle::CustomOp1 for TopK {
    fn name(&self) -> &'static str {
        "topk"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("topk CustomOp is GPU-only; the public fn composes on CPU")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
        };
        use candle::cuda_backend::{
            kernel_name, kernels, CudaDevice, CudaStorageSlice as S, Map1Any, WrapErr,
        };
        use candle::WithDType;

        if !layout.is_contiguous() {
            candle::bail!("topk requires a contiguous input")
        }
        let dims = layout.dims();
        let k_pad = self.k_pad;
        if k_pad == 0 || k_pad > 1024 || !k_pad.is_power_of_two() {
            candle::bail!("topk k_pad={k_pad} must be a power of two in 1..=1024");
        }
        match storage.dtype() {
            DType::F32 | DType::F16 | DType::BF16 => {}
            dtype => candle::bail!("topk does not support {dtype:?} on CUDA"),
        }

        // Shared: (TILE + 2k) floats + (TILE + 2k) u32s — O(k), not O(ncols).
        const TILE: usize = 1024;
        let shared_mem_bytes =
            (TILE + 2 * k_pad) * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

        struct TopKLaunch {
            k_pad: usize,
            shared_mem_bytes: usize,
        }
        impl Map1Any for TopKLaunch {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
                _wrap: W,
            ) -> Result<S> {
                let src = match layout.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => src.slice(o1..o2),
                };
                let dims = layout.dims();
                let ncols = *dims.last().unwrap();
                let nrows: usize = dims[..dims.len() - 1].iter().product();
                let elem_count = nrows * self.k_pad;
                let dst = unsafe { dev.alloc::<u32>(elem_count)? };
                let func = dev.get_or_load_func(&kernel_name::<T>("topk"), &kernels::SORT)?;
                let cfg = LaunchConfig {
                    grid_dim: (nrows as u32, 1, 1),
                    block_dim: (1024u32, 1, 1),
                    shared_mem_bytes: self.shared_mem_bytes as u32,
                };
                let stream = dev.cuda_stream();
                let mut builder = stream.launch_builder(&func);
                let ncols_i = ncols as i32;
                let k_i = self.k_pad as i32;
                builder.arg(&src).arg(&dst).arg(&ncols_i).arg(&k_i);
                // SAFETY: topk kernel writes elem_count u32 indices.
                unsafe { builder.launch(cfg) }.w()?;
                Ok(S::U32(dst))
            }
        }

        let dev = storage.device();
        let slice = TopKLaunch {
            k_pad,
            shared_mem_bytes,
        }
        .map(&storage.slice, dev, layout)?;
        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = k_pad;
        Ok((dst, Shape::from_dims(&out_dims)))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;

        if !layout.is_contiguous() {
            candle::bail!("topk requires a contiguous input")
        }
        let dims = layout.dims();
        let ncols = *dims.last().unwrap();
        let nrows: usize = dims[..dims.len() - 1].iter().product();
        let name = match storage.dtype() {
            DType::F32 => "topk_f32",
            DType::F16 => "topk_f16",
            DType::BF16 => "topk_bf16",
            dtype => candle::bail!("topk does not support {dtype:?} on Metal"),
        };
        let device = storage.device();
        let elem_count = nrows * self.k_pad;
        let dst = device
            .new_buffer_builder()
            .with_size_for(elem_count, DType::U32)
            .with_label("topk")
            .build()?;
        let encoder = device.command_encoder()?;
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_topk(
            device.metal_device(),
            &encoder,
            device.kernels(),
            name,
            nrows,
            ncols,
            self.k_pad,
            src,
            &dst,
        )
        .map_err(candle::Error::wrap)?;
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = self.k_pad;
        let storage = candle::MetalStorage::new(dst, device.clone(), elem_count, DType::U32);
        Ok((storage, Shape::from_dims(&out_dims)))
    }
}

/// The `k` largest values along the last dimension and their indices, both
/// descending by value — `torch.topk` semantics (ties break arbitrarily).
///
/// **Metal and CUDA** use a dedicated tile-and-merge kernel that works for
/// arbitrary row widths (shared memory is O(k), not O(ncols)). Full-width
/// `arg_sort_last_dim` on CUDA is limited by dynamic shared memory and must
/// not be used for wide last-dims (e.g. Heron RT-DETR encoder tokens ≈ 8k).
/// **CPU** (and other backends) compose a descending arg-sort with a narrow.
pub fn topk(xs: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let last = xs.dim(D::Minus1)?;
    if k == 0 || k > last {
        candle::bail!("topk k={k} out of range for last dim {last}");
    }
    let xs = xs.contiguous()?;
    let k_pad = k.next_power_of_two();
    let fused = k_pad <= 1024
        && matches!(xs.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && (xs.device().is_metal() || xs.device().is_cuda());
    let indices = if fused {
        xs.apply_op1_no_bwd(&TopK { k_pad })?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?
    } else {
        xs.arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?
    };
    let values = xs.gather(&indices, D::Minus1)?;
    Ok((values, indices))
}

/// `lhs.matmul(rhs) + bias`, with the bias add fused into the GEMM epilogue
/// on Metal. `bias` is an `n`-element vector broadcast over rows and batch;
/// `rhs` may be 2d (broadcast over the batch) or share `lhs`'s batch shape.
/// Other backends compose `broadcast_matmul` + `broadcast_add`.
pub fn matmul_bias(lhs: &Tensor, rhs: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if !lhs.device().is_metal() {
        return lhs.broadcast_matmul(rhs)?.broadcast_add(bias);
    }
    fused_matmul_bias(lhs, rhs, bias, None)
}

/// `matmul_bias` followed by a unary activation, all fused into the GEMM
/// epilogue on Metal. Other backends compose the same chain from separate
/// ops.
pub fn matmul_bias_act(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: &Tensor,
    activation: MatmulActivation,
) -> Result<Tensor> {
    if !lhs.device().is_metal() {
        let out = lhs.broadcast_matmul(rhs)?.broadcast_add(bias)?;
        return match activation {
            MatmulActivation::Relu => out.relu(),
            MatmulActivation::Gelu => out.gelu(),
            MatmulActivation::Silu => out.silu(),
        };
    }
    fused_matmul_bias(lhs, rhs, bias, Some(activation))
}

fn fused_matmul_bias(
    lhs: &Tensor,
    rhs: &Tensor,
    bias: &Tensor,
    activation: Option<MatmulActivation>,
) -> Result<Tensor> {
    // Give a lower-rank rhs explicit zero-stride batch dims so the kernel's
    // per-operand batch strides are exact.
    let rhs = if rhs.rank() < lhs.rank() {
        let batch_dims = &lhs.dims()[..lhs.rank() - 2];
        rhs.broadcast_left(batch_dims)?
    } else {
        rhs.clone()
    };
    lhs.apply_op3_no_bwd(&rhs, bias, &MatmulBias { activation })
}

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
    mask: Option<Tensor>,
    do_causal: bool,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();
        let out_shape = Shape::from_dims(&out_dims);
        let out_layout = Layout::contiguous(out_shape.clone());

        let output = device
            .new_buffer_builder()
            .with_size_for(elem_count, q.dtype())
            .with_label("sdpa_o")
            .build()?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;
        let k_seq = k_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim = q_head == 32
            || q_head == 64
            || q_head == 72
            || q_head == 80
            || q_head == 96
            || q_head == 128
            || q_head == 256
            || q_head == 512;

        let supports_sdpa_full_mask = self.mask.is_none() || q_seq <= k_seq;
        // F32 full attention at head_dim=512 exceeds 32KB Metal threadgroup memory
        let supports_sdpa_full_dtype = !(q_head == 512 && q.dtype() == DType::F32);
        // The vector kernel (`sdpa_vector` / `sdpa_vector_2pass_*` in
        // `scaled_dot_product_attention.metal`) dispatches one threadgroup per
        // `(batch, qhead)` pair and writes a single output position per head —
        // the kernel source has no q-axis loop. For `q_seq > 1` it leaves the
        // extra output positions uninitialised (pooled Metal buffer → garbage).
        // Restrict it to the decode case and route everything else to the full
        // kernel. Regression guard: `candle-nn/tests/sdpa.rs`
        // `sdpa_vector_q_seq_2_to_8_matches_reference`.
        let supports_sdpa_full =
            q_seq > 1 && supported_head_dim && supports_sdpa_full_mask && supports_sdpa_full_dtype;
        let supports_sdpa_vector = q_seq == 1 && supported_head_dim && q_seq <= k_seq;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("unsupported sdpa type {other:?}"),
        };

        let encoder = q.device().command_encoder()?;
        if supports_sdpa_vector {
            // Route to the 2 pass fused attention if the k seqlen is large.
            // https://github.com/ml-explore/mlx/pull/1597
            const TWO_PASS_K_THRESHOLD: usize = 1024;
            if k_seq >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_intermediate")
                    .build()?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_sums")
                    .build()?;
                let maxs = device
                    .new_buffer_builder()
                    .with_size_for(intermediate_shape.iter().product::<usize>(), DType::F32)
                    .with_label("sdpa_2pass_maxs")
                    .build()?;

                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector_2pass(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset() * q.dtype().size_in_bytes(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset() * k.dtype().size_in_bytes(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset() * v.dtype().size_in_bytes(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            } else {
                encoder.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector(
                    q.device().device(),
                    &encoder,
                    q.device().kernels(),
                    q_l.start_offset() * q.dtype().size_in_bytes(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset() * k.dtype().size_in_bytes(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset() * v.dtype().size_in_bytes(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            encoder.set_label("full_attention");
            if self.softcapping != 1. {
                candle::bail!("SDPA full requires softcapping to be disabled (1.0)");
            }

            let mask_s_l = self.mask.as_ref().map(|m| m.storage_and_layout());

            let (mask_type, mask_buffer, mask_strides) = if let Some(mask) = &self.mask {
                let (mask_s, mask_l) = mask_s_l.as_ref().unwrap();

                let mask_buffer = match &**mask_s {
                    candle::Storage::Metal(m) => m.buffer(),
                    _ => candle::bail!("Expected metal device for mask"),
                };

                let mask_type = match mask.dtype() {
                    DType::BF16 => SdpaDType::BF16,
                    DType::F16 => SdpaDType::F16,
                    DType::F32 => SdpaDType::F32,
                    other => candle::bail!("unsupported sdpa type {other:?}"),
                };
                if mask_type != itype {
                    candle::bail!("Mask type {mask_type:?} must match q type {itype:?}");
                }

                if mask_l.dims() != [q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, k_seq] {
                    candle::bail!(
                        "Mask shape must be {:?} (bs, qheads, qseq, kseq), got {:?}",
                        [q_l.dim(0)?, q_head, q_l.dim(2)?, k_seq],
                        mask_l.dims()
                    );
                }

                (
                    Some(mask_type),
                    Some(mask_buffer),
                    Some(mask_l.stride().to_vec()),
                )
            } else {
                (None, None, None)
            };

            candle_metal_kernels::call_sdpa_full(
                q.device().device(),
                &encoder,
                q.device().kernels(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                q_l.dims(),
                q_l.stride(),
                q.buffer(),
                k_l.start_offset() * k.dtype().size_in_bytes(),
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset() * v.dtype().size_in_bytes(),
                v.buffer(),
                v_l.stride(),
                mask_type,
                mask_buffer,
                mask_strides.as_deref(),
                &output,
                out_layout.stride(),
                self.scale,
                self.do_causal,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else {
            candle::bail!("must be vector or full sdpa kernel");
        }

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `mask`: (bs, qhead, seq, kv_seq)
/// - `do_causal`: Apply causal masking. If this is true, the mask does not need to be provided.
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// Note: For Grouped Query Attention and Multi-Query Attention, the k and v inputs should not be pre-tiled to match q.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Masking is supported
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
///     - Softcapping is not supported.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    do_causal: bool,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    q.apply_op3_no_bwd(
        k,
        v,
        &Sdpa {
            scale,
            softcapping,
            mask: mask.cloned(),
            do_causal,
        },
    )
}
