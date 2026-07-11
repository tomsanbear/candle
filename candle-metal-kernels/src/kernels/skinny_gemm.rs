use crate::metal::Value;
use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, ConstantValues, Device, GemmDType,
    Kernels, MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

const SK_MAX_M: usize = 12;
const SK_SG_PER_TG: usize = 8;

/// Skinny GEMM (2 <= m <= 12) for the nt layout the linear layers produce:
/// A[m,k] row-major x B[n,k] row-major (transposed view). Streams B exactly
/// once at gemv-class bandwidth; the mlx tile kernels run these shapes at
/// roughly half bandwidth. Validates dtype/layout and errs so callers can
/// fall back to the tile gemm.
#[allow(clippy::too_many_arguments)]
pub fn call_skinny_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    if dtype != GemmDType::BF16 || b != 1 || !(2..=SK_MAX_M).contains(&m) {
        return Err(MetalKernelError::LoadLibraryError(
            "skinny_gemm: unsupported dtype/batch/m".to_string(),
        ));
    }
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    // A row-major [m,k]; B presented as the transposed view of a row-major
    // [n,k] weight (strides [1, k]).
    let a_ok = lhs_m1 == 1 && lhs_m2 == k;
    let b_ok = rhs_m2 == 1 && rhs_m1 == k;
    if !a_ok || !b_ok || lhs_offset != 0 && lhs_offset % 2 != 0 {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        });
    }

    // m is a function constant so the accumulator loop unrolls into
    // registers; the pipeline cache keys on (name, constants), so at most
    // SK_MAX_M-1 specializations compile per process.
    let constants = Some(ConstantValues::new(vec![(0, Value::U16(m as u16))]));
    let pipeline = kernels.load_pipeline_with_constants(
        device,
        Source::SkinnyGemm,
        "skinny_gemm_nt_bf16".to_string(),
        constants,
    )?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "skinny_gemm m={m} n={n} k={k}");

    set_params!(
        encoder,
        (
            (lhs_buffer, lhs_offset),
            (rhs_buffer, rhs_offset),
            Output::new(output),
            m as u32,
            n as u32,
            k as u32
        )
    );

    encoder.dispatch_thread_groups(
        MTLSize {
            width: n.div_ceil(SK_SG_PER_TG),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: SK_SG_PER_TG * 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
