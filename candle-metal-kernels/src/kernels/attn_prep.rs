use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// Threadgroup width baked into attn_prep.metal; the norm's reduction order
/// (and bitwise identity with rmsnorm) holds only when
/// `(d / 2).next_power_of_two() == ATTN_PREP_BLOCK`.
pub const ATTN_PREP_BLOCK: usize = 128;
/// Static threadgroup buffer bound in attn_prep.metal.
pub const ATTN_PREP_MAX_D: usize = 256;

fn check_dims(d: usize, rd: usize) -> Result<(), MetalKernelError> {
    if d > ATTN_PREP_MAX_D || (d / 2).next_power_of_two() != ATTN_PREP_BLOCK {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "attn_prep requires (d/2).next_power_of_two() == {ATTN_PREP_BLOCK} and d <= {ATTN_PREP_MAX_D}, got d={d}"
        )));
    }
    if rd == 0 || !rd.is_multiple_of(2) || rd > d {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "attn_prep requires even 0 < rd <= d, got rd={rd} d={d}"
        )));
    }
    Ok(())
}

/// Head-norm + partial rope on q rows read from the packed qkv projection,
/// written in (heads, t_len, d) layout. One threadgroup per (head, token).
#[allow(clippy::too_many_arguments)]
pub fn call_attn_q_prep(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    heads: usize,
    t_len: usize,
    d: usize,
    rd: usize,
    qkv_row_stride: usize,
    head_in_stride: usize,
    q_base: usize,
    pos_base: usize,
    eps: f32,
    qkv: &Buffer,
    qkv_offset: usize,
    alpha: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    check_dims(d, rd)?;
    let pipeline = kernels.load_pipeline(device, Source::AttnPrep, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "attn_q_prep {kernel_name} heads={heads} t_len={t_len} d={d} rd={rd}"
    );

    set_params!(
        encoder,
        (
            t_len as u32,
            d as u32,
            rd as u32,
            qkv_row_stride as u32,
            head_in_stride as u32,
            q_base as u32,
            pos_base as u32,
            eps,
            (qkv, qkv_offset),
            alpha,
            cos,
            sin,
            Output::new(dst)
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize { width: heads * t_len, height: 1, depth: 1 },
        MTLSize { width: ATTN_PREP_BLOCK, height: 1, depth: 1 },
    );
    Ok(())
}

/// Head-norm + partial rope on k plus the raw v copy, written directly into
/// the (kv_heads, cache_cap, d) KV-cache buffers at write_pos. One
/// threadgroup per (kv_head, token).
#[allow(clippy::too_many_arguments)]
pub fn call_attn_kv_prep(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    kv_heads: usize,
    t_len: usize,
    d: usize,
    rd: usize,
    qkv_row_stride: usize,
    k_base: usize,
    v_base: usize,
    cache_cap: usize,
    write_pos: usize,
    pos_base: usize,
    eps: f32,
    qkv: &Buffer,
    qkv_offset: usize,
    alpha: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    cache_k: &Buffer,
    cache_v: &Buffer,
) -> Result<(), MetalKernelError> {
    check_dims(d, rd)?;
    if write_pos + t_len > cache_cap {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "attn_kv_prep write {write_pos}+{t_len} exceeds cache capacity {cache_cap}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::AttnPrep, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "attn_kv_prep {kernel_name} kv_heads={kv_heads} t_len={t_len} d={d} rd={rd} write_pos={write_pos}"
    );

    set_params!(
        encoder,
        (
            t_len as u32,
            d as u32,
            rd as u32,
            qkv_row_stride as u32,
            k_base as u32,
            v_base as u32,
            cache_cap as u32,
            write_pos as u32,
            pos_base as u32,
            eps,
            (qkv, qkv_offset),
            alpha,
            cos,
            sin,
            Output::new(cache_k),
            Output::new(cache_v)
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize { width: kv_heads * t_len, height: 1, depth: 1 },
        MTLSize { width: ATTN_PREP_BLOCK, height: 1, depth: 1 },
    );
    Ok(())
}
