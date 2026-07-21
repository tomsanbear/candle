use crate::utils::EncoderProvider;
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    // `align` must equal the number of dst rows each threadgroup of the
    // kernel covers: the grid is ceil-divided over ne01, so a non-divisible
    // row count leaves a tail threadgroup that must not touch rows past ne01.
    // The unguarded kernels assume no such tail exists; the `_guarded`
    // variants clamp tail-row reads and guard tail stores. The dense f16/f32
    // kernel covers one row per threadgroup so it has no tail. `ny` is the
    // number of src1 rows the kernel covers per threadgroup in grid y.
    let (name, guarded_name, nth0, nth1, align, ny) = match dtype {
        GgmlDType::Q4_0 => ("kernel_mul_mv_q4_0_f32", "kernel_mul_mv_q4_0_f32_guarded", 8, 8, 8, 1),
        GgmlDType::Q4_1 => ("kernel_mul_mv_q4_1_f32", "kernel_mul_mv_q4_1_f32_guarded", 8, 8, 8, 1),
        GgmlDType::Q5_0 => ("kernel_mul_mv_q5_0_f32", "kernel_mul_mv_q5_0_f32_guarded", 8, 8, 8, 1),
        GgmlDType::Q5_1 => ("kernel_mul_mv_q5_1_f32", "kernel_mul_mv_q5_1_f32_guarded", 8, 8, 8, 1),
        GgmlDType::Q8_0 => ("kernel_mul_mv_q8_0_f32", "kernel_mul_mv_q8_0_f32_guarded", 8, 8, 8, 1),
        GgmlDType::Q2K => ("kernel_mul_mv_q2_K_f32", "kernel_mul_mv_q2_K_f32_guarded", 2, 32, 8, 1),
        GgmlDType::Q3K => ("kernel_mul_mv_q3_K_f32", "kernel_mul_mv_q3_K_f32_guarded", 2, 32, 4, 1),
        GgmlDType::Q4K => ("kernel_mul_mv_q4_K_f32", "kernel_mul_mv_q4_K_f32_guarded", 4, 8, 4, 1),
        // q5_K guards per simdgroup in-kernel, so one pipeline serves both.
        GgmlDType::Q5K => ("kernel_mul_mv_q5_K_f32", "kernel_mul_mv_q5_K_f32", 2, 32, 4, 1),
        GgmlDType::Q6K => ("kernel_mul_mv_q6_K_f32", "kernel_mul_mv_q6_K_f32_guarded", 2, 32, 2, 1),
        GgmlDType::F16 => ("kernel_mul_mv_f16_f32", "kernel_mul_mv_f16_f32", 32, 1, 1, 4),
        GgmlDType::F32 => ("kernel_mul_mv_f32_f32", "kernel_mul_mv_f32_f32", 32, 1, 1, 4),
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul_mv"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul_mv"))?,
        GgmlDType::BF16 => Err(MetalKernelError::UnsupportedDTypeForOp("BF16", "qmatmul_mv"))?,
    };
    let name = if n % align == 0 { name } else { guarded_name };

    // The dense f16/f32 kernel addresses src0/src1 through these byte
    // strides; the quantized kernels derive theirs from ne00/ne01 and
    // ignore nb00..nb02.
    let src0_elem_size = match dtype {
        GgmlDType::F32 => 4i64,
        GgmlDType::F16 => 2i64,
        _ => 0i64,
    };
    let nb00 = src0_elem_size;
    let nb01 = src0_elem_size * ne00;
    let nb02 = src0_elem_size * ne00 * ne01;

    let nb10 = 4i64;
    let nb11 = nb10 * ne10;
    let nb12 = nb11 * ne11;

    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: divide(ne11 as usize, ny),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            (dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(lhs, MTLResourceUsage::Read);
    encoder.use_resource(rhs, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            (dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    encoder.use_resource(src0, MTLResourceUsage::Read);
    encoder.use_resource(src1, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
