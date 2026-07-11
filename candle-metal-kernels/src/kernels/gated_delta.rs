use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatedDeltaParams {
    pub heads: u32,
    pub dk: u32,
    pub dv: u32,
    pub conv_dim: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub ksz: u32,
    pub l2_eps: f32,
    pub norm_eps: f32,
}

/// Fused GatedDeltaNet single-token decode step; see gated_delta.metal for
/// layouts and semantics. BF16 activations, F32 state, one dispatch of
/// `heads` threadgroups x `dv` threads.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_decode(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    proj: &Buffer,
    conv_in: &Buffer,
    state_in: &Buffer,
    conv_w: &Buffer,
    dt_bias: &Buffer,
    a_log_exp: &Buffer,
    norm_w: &Buffer,
    out: &Buffer,
    conv_out: &Buffer,
    state_out: &Buffer,
) -> Result<(), MetalKernelError> {
    if params.dv % 32 != 0 || params.dv > 256 || params.dk != params.dv {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "gated_delta_decode requires dk == dv, dv % 32 == 0, dv <= 256; got dk={} dv={}",
            params.dk, params.dv
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::GatedDelta, "gated_delta_decode_bf16")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gated_delta_decode heads={}", params.heads);

    set_params!(
        encoder,
        (
            proj,
            conv_in,
            state_in,
            conv_w,
            dt_bias,
            a_log_exp,
            norm_w,
            Output::new(out),
            Output::new(conv_out),
            Output::new(state_out),
            params.heads,
            params.dk,
            params.dv,
            params.conv_dim,
            params.key_dim,
            params.value_dim,
            params.ksz,
            params.l2_eps,
            params.norm_eps
        )
    );

    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.heads as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: params.dv as usize,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
