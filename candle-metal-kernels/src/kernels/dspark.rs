use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

#[cfg(feature = "debug-labels")]
use crate::debug_group;

/// Threads per threadgroup for both chain kernels (fixed by the metal side's
/// threadgroup arrays).
pub const MARKOV_TPG: usize = 256;
/// Threadgroups for the partial pass; the reduce pass consumes this many
/// partials in one threadgroup.
pub const MARKOV_NTG: usize = 64;

pub struct MarkovChainArgs<'a> {
    /// Steps to draft (gamma).
    pub gamma: usize,
    /// Draft-vocabulary rows in w2/base (32768 under FR-Spec, else full).
    pub draft_vocab: usize,
    /// Markov rank (row width of w1, input dim of w2). Must be <= 256 and a
    /// multiple of 32.
    pub rank: usize,
    /// w2 is a ggml q8_0 row matrix (else bf16 rows).
    pub w2_q8: bool,
    /// [vocab_full, rank] bf16 embedding table, global-id indexed.
    pub w1: (&'a Buffer, usize),
    /// [draft_vocab, rank] q8_0 blocks or bf16 rows.
    pub w2: (&'a Buffer, usize),
    /// [gamma, draft_vocab] bf16 base logits (lm_head output rows).
    pub base: (&'a Buffer, usize),
    /// [gamma+1] u32; slot 0 pre-seeded with the anchor GLOBAL id. The
    /// kernels thread each step's winner through slots 1..=gamma.
    pub chain: &'a Buffer,
    /// [MARKOV_NTG] (f32, u32) pairs scratch (two-dispatch shape only).
    pub partials: &'a Buffer,
    /// [draft_vocab] u32 draft->global id map; ignored when `remap` is false
    /// (pass any buffer, e.g. `tokens`).
    pub ids: (&'a Buffer, usize),
    pub remap: bool,
    /// Out: [gamma] u32 global token ids.
    pub tokens: &'a Buffer,
    /// Out: [gamma, rank] bf16 — each step's INPUT embedding row (the
    /// confidence-head feature half).
    pub prev_embs: &'a Buffer,
    /// Single-dispatch-per-step scratch: [gamma] interleaved u32 pairs
    /// (packed-max slot, arrival counter), ZEROED at allocation. The kernels
    /// self-clean each step's pair, so the buffer is reusable across calls
    /// with no host clears and may be sized for any gamma >= the call's.
    /// Requires draft_vocab <= 32768 (the packed-key index width); `None`
    /// selects the two-dispatch partial+reduce shape.
    pub fused_scratch: Option<&'a Buffer>,
}

/// Largest draft vocabulary the single-dispatch chain step supports (its
/// packed atomic key carries a 15-bit index).
pub const MARKOV_FUSED_MAX_VD: usize = 1 << 15;

/// Fused Markov-chain proposal. With `fused_scratch`: ONE dispatch per step
/// — cross-threadgroup argmax through a packed 32-bit atomic max with a
/// last-threadgroup epilogue (metal_src/dspark.metal documents the packing,
/// determinism, and memory-model reasoning); requires draft_vocab <= 32768.
/// Without: two dispatches per step (grid partial + single-threadgroup
/// reduce), the shape for larger vocabularies. Both replace the legacy ~5-6
/// tiny serial kernels per step. Tie rule and dtype path documented in
/// metal_src/dspark.metal. Per-row dot products accumulate sequentially in
/// one thread, so a same-order CPU reference reproduces the tokens BITWISE
/// (the bench task's gate); only comparisons against the legacy qmv path
/// carry ulp-tie caveats.
pub fn call_markov_chain(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    args: MarkovChainArgs<'_>,
) -> Result<(), MetalKernelError> {
    if args.rank > MARKOV_TPG || args.rank % 32 != 0 {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "markov rank must be a multiple of 32 and <= 256",
            "markov_chain",
        ));
    }
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    let vd = args.draft_vocab as u32;
    let r = args.rank as u32;
    let use_remap: u32 = args.remap as u32;
    let tg_grid = MTLSize {
        width: MARKOV_NTG,
        height: 1,
        depth: 1,
    };
    let tpg = MTLSize {
        width: MARKOV_TPG,
        height: 1,
        depth: 1,
    };

    if let Some(scratch) = args.fused_scratch {
        if args.draft_vocab > MARKOV_FUSED_MAX_VD {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "single-dispatch markov step requires draft_vocab <= 32768",
                "markov_chain",
            ));
        }
        let name = if args.w2_q8 {
            "markov_step_fused_q8"
        } else {
            "markov_step_fused_bf16"
        };
        let fused = kernels.load_pipeline(device, Source::Dspark, name)?;
        for k in 0..args.gamma as u32 {
            encoder.set_compute_pipeline_state(&fused);
            #[cfg(feature = "debug-labels")]
            debug_group!(encoder, "markov_fused k={k}");
            set_params!(
                encoder,
                (
                    args.w1,
                    args.w2,
                    args.base,
                    args.ids,
                    Output::new(args.chain),
                    Output::new(args.tokens),
                    Output::new(args.prev_embs),
                    Output::new(scratch),
                    k,
                    vd,
                    r,
                    use_remap
                )
            );
            encoder.dispatch_thread_groups(tg_grid, tpg);
        }
        return Ok(());
    }

    let partial_name = if args.w2_q8 {
        "markov_step_partial_q8"
    } else {
        "markov_step_partial_bf16"
    };
    let partial = kernels.load_pipeline(device, Source::Dspark, partial_name)?;
    let reduce = kernels.load_pipeline(device, Source::Dspark, "markov_step_reduce")?;
    let ntg = MARKOV_NTG as u32;
    let tg_one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    for k in 0..args.gamma as u32 {
        encoder.set_compute_pipeline_state(&partial);
        #[cfg(feature = "debug-labels")]
        debug_group!(encoder, "markov_partial k={k}");
        set_params!(
            encoder,
            (
                args.w1,
                args.w2,
                args.base,
                args.chain,
                Output::new(args.partials),
                k,
                vd,
                r
            )
        );
        encoder.dispatch_thread_groups(tg_grid, tpg);

        encoder.set_compute_pipeline_state(&reduce);
        #[cfg(feature = "debug-labels")]
        debug_group!(encoder, "markov_reduce k={k}");
        set_params!(
            encoder,
            (
                args.partials,
                args.ids,
                Output::new(args.chain),
                Output::new(args.tokens),
                args.w1,
                Output::new(args.prev_embs),
                k,
                ntg,
                r,
                use_remap
            )
        );
        encoder.dispatch_thread_groups(tg_one, tpg);
    }
    Ok(())
}
