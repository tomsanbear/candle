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

/// Fused GatedDeltaNet chunk step (l <= 12); see gated_delta_chunk.metal.
/// Emits the rollback-capture intermediates alongside outputs and states.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_chunk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    seq_len: usize,
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
    cap_k: &Buffer,
    cap_delta: &Buffer,
    cap_gcs: &Buffer,
) -> Result<(), MetalKernelError> {
    if params.dk != 128 || params.dv != 128 || seq_len == 0 || seq_len > 12 {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "gated_delta_chunk requires dk == dv == 128 and 1 <= l <= 12; got dk={} dv={} l={seq_len}",
            params.dk, params.dv
        )));
    }
    let pipeline =
        kernels.load_pipeline(device, Source::GatedDeltaChunk, "gated_delta_chunk_bf16")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gated_delta_chunk l={seq_len}");

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
            Output::new(cap_k),
            Output::new(cap_delta),
            Output::new(cap_gcs),
            params.heads,
            params.dk,
            params.dv,
            params.conv_dim,
            params.key_dim,
            params.value_dim,
            params.ksz,
            seq_len as u32,
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

/// Fused GatedDeltaNet single-token decode step; see gated_delta.metal for
/// layouts and semantics. BF16 activations, F32 state, one dispatch of
/// `heads` threadgroups x `dv` threads.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_decode(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    batch: usize,
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
            params.norm_eps,
            batch as u32
        )
    );

    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.heads as usize * batch.max(1),
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

/// Buffers staged between the v2 sub-kernels; allocated by the caller so the
/// capture tensors (kn = cap_k, cap_gcs, cap_delta) come back as ordinary
/// tensors for the host-side rollback math.
#[allow(clippy::too_many_arguments)]
pub struct GatedDeltaV2Stages<'a> {
    /// f32 [b*heads, l, dk] — l2-normed keys; doubles as the cap_k capture.
    pub kn: &'a Buffer,
    /// f32 [b*heads, l, dk] — l2-normed, 1/sqrt(dk)-scaled queries.
    pub qn: &'a Buffer,
    /// f32 [b*heads, l, dv] — conv'd + silu'd values.
    pub vc: &'a Buffer,
    /// f32 [b*heads, l] — per-step log decay.
    pub g_step: &'a Buffer,
    /// f32 [b*heads, l] — sigmoid(b) write gate.
    pub beta: &'a Buffer,
    /// f32 [b*heads, l] — inclusive log-decay cumsum (cap_gcs capture).
    pub cap_gcs: &'a Buffer,
    /// f32 [b*heads, l, dv] — WY pseudo-values (cap_delta capture).
    pub cap_delta: &'a Buffer,
    /// f32 [b, l, value_dim] — pre-norm outputs.
    pub o_pre: &'a Buffer,
}

/// Fused GatedDeltaNet v2: unified decode/chunk (1 <= l <= 12) across three
/// dispatches (prep, delta core, epilogue). State layout is TRANSPOSED vs the
/// v1 kernels: f32 [b, heads, dv, dk]. See gated_delta_v2.metal.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_v2(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    seq_len: usize,
    batch: usize,
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
    stages: GatedDeltaV2Stages<'_>,
) -> Result<(), MetalKernelError> {
    if params.dk != 128 || params.dv != 128 || seq_len == 0 || seq_len > 12 {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "gated_delta_v2 requires dk == dv == 128 and 1 <= l <= 12; got dk={} dv={} l={seq_len}",
            params.dk, params.dv
        )));
    }
    let bh = params.heads as usize * batch.max(1);
    let l = seq_len as u32;

    let prep = kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_prep_bf16")?;
    let core = kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_core")?;
    let epilogue =
        kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_epilogue_bf16")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&prep);
    debug_group!(encoder, "gated_delta_v2_prep l={seq_len}");
    set_params!(
        encoder,
        (
            proj,
            conv_in,
            conv_w,
            dt_bias,
            a_log_exp,
            Output::new(conv_out),
            Output::new(stages.kn),
            Output::new(stages.qn),
            Output::new(stages.vc),
            Output::new(stages.g_step),
            Output::new(stages.beta),
            Output::new(stages.cap_gcs),
            params.heads,
            params.dk,
            params.dv,
            params.conv_dim,
            params.key_dim,
            params.value_dim,
            params.ksz,
            l,
            params.l2_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: bh,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: params.dk as usize,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&core);
    debug_group!(encoder, "gated_delta_v2_core l={seq_len}");
    set_params!(
        encoder,
        (
            state_in,
            Output::new(state_out),
            stages.kn,
            stages.qn,
            stages.vc,
            stages.g_step,
            stages.beta,
            Output::new(stages.cap_delta),
            Output::new(stages.o_pre),
            params.heads,
            params.dk,
            params.dv,
            params.value_dim,
            l
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.dv as usize / 4,
            height: bh,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&epilogue);
    debug_group!(encoder, "gated_delta_v2_epilogue l={seq_len}");
    set_params!(
        encoder,
        (
            stages.o_pre,
            proj,
            norm_w,
            Output::new(out),
            params.heads,
            params.dv,
            params.conv_dim,
            params.value_dim,
            l,
            params.norm_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: bh,
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

/// Fused GatedDeltaNet v2 single-token decode: fused prep+core at l = 1
/// (grid (dv/4, batch*heads) — full-occupancy state streaming, simd-scope
/// reductions only) followed by the epilogue kernel for the per-head
/// RMSNorm + silu(z) gate. Two dispatches on one encoder. State layout is
/// TRANSPOSED (f32 [b, heads, dv, dk]) like the other v2 kernels.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_v2_decode(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    batch: usize,
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
    o_pre: &Buffer,
) -> Result<(), MetalKernelError> {
    if params.dk != 128 || params.dv != 128 {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "gated_delta_v2_decode requires dk == dv == 128; got dk={} dv={}",
            params.dk, params.dv
        )));
    }
    let bh = params.heads as usize * batch.max(1);

    let core =
        kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_decode_bf16")?;
    let epilogue =
        kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_epilogue_bf16")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&core);
    debug_group!(encoder, "gated_delta_v2_decode heads={}", params.heads);
    set_params!(
        encoder,
        (
            proj,
            conv_in,
            state_in,
            conv_w,
            dt_bias,
            a_log_exp,
            Output::new(conv_out),
            Output::new(state_out),
            Output::new(o_pre),
            params.heads,
            params.dk,
            params.dv,
            params.conv_dim,
            params.key_dim,
            params.value_dim,
            params.ksz,
            params.l2_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.dv as usize / 4,
            height: bh,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&epilogue);
    debug_group!(encoder, "gated_delta_v2_decode_epilogue");
    set_params!(
        encoder,
        (
            o_pre,
            proj,
            norm_w,
            Output::new(out),
            params.heads,
            params.dv,
            params.conv_dim,
            params.value_dim,
            1u32,
            params.norm_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: bh,
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

/// Fused GatedDeltaNet v2 tree-verify step: main segment [anchor, a_1..a_w]
/// (rows [0, seg1)) and alternate segment [b_1..b_w] (rows [seg1, seg1+alt))
/// in ONE prep/core/epilogue pass. The alternate restarts inside the core
/// dispatch from the state after main row `branch_after`-1; that branch-point
/// state is also written to `state_mid` (the alternate capture's S0 for the
/// host's closed-form rollback). Staging/capture buffers span all
/// seg1+alt_len rows; cap_gcs restarts its cumsum at the segment boundary so
/// each segment's capture slice matches a separate dispatch. Single stream.
#[allow(clippy::too_many_arguments)]
pub fn call_gated_delta_v2_tree(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    params: GatedDeltaParams,
    seg1: usize,
    alt_len: usize,
    branch_after: usize,
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
    state_mid: &Buffer,
    stages: GatedDeltaV2Stages<'_>,
) -> Result<(), MetalKernelError> {
    let l_total = seg1 + alt_len;
    if params.dk != 128
        || params.dv != 128
        || seg1 == 0
        || alt_len == 0
        || l_total > 12
        || branch_after == 0
        || branch_after > seg1
    {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "gated_delta_v2_tree requires dk == dv == 128, 1 <= branch_after <= seg1 and \
             2 <= seg1+alt <= 12; got dk={} dv={} seg1={seg1} alt={alt_len} branch_after={branch_after}",
            params.dk, params.dv
        )));
    }
    let heads = params.heads as usize;

    let prep =
        kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_prep_tree_bf16")?;
    let core = kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_core_tree")?;
    let epilogue =
        kernels.load_pipeline(device, Source::GatedDeltaV2, "gated_delta_v2_epilogue_bf16")?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&prep);
    debug_group!(encoder, "gated_delta_v2_prep_tree l={l_total}");
    set_params!(
        encoder,
        (
            proj,
            conv_in,
            conv_w,
            dt_bias,
            a_log_exp,
            Output::new(conv_out),
            Output::new(stages.kn),
            Output::new(stages.qn),
            Output::new(stages.vc),
            Output::new(stages.g_step),
            Output::new(stages.beta),
            Output::new(stages.cap_gcs),
            params.heads,
            params.dk,
            params.dv,
            params.conv_dim,
            params.key_dim,
            params.value_dim,
            params.ksz,
            seg1 as u32,
            alt_len as u32,
            branch_after as u32,
            params.l2_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: heads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: params.dk as usize,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&core);
    debug_group!(encoder, "gated_delta_v2_core_tree l={l_total}");
    set_params!(
        encoder,
        (
            state_in,
            Output::new(state_out),
            stages.kn,
            stages.qn,
            stages.vc,
            stages.g_step,
            stages.beta,
            Output::new(stages.cap_delta),
            Output::new(stages.o_pre),
            Output::new(state_mid),
            params.heads,
            params.dk,
            params.dv,
            params.value_dim,
            seg1 as u32,
            alt_len as u32,
            branch_after as u32
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.dv as usize / 4,
            height: heads,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&epilogue);
    debug_group!(encoder, "gated_delta_v2_epilogue l={l_total}");
    set_params!(
        encoder,
        (
            stages.o_pre,
            proj,
            norm_w,
            Output::new(out),
            params.heads,
            params.dv,
            params.conv_dim,
            params.value_dim,
            l_total as u32,
            params.norm_eps
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: heads,
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
