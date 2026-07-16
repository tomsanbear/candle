use crate::utils::EncoderProvider;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_0,
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

/// True when `dtype` has BF16-activation (src1) mv kernel variants, i.e. the
/// F32 cast round-trip around the matmul can be skipped.
pub fn quantized_matmul_mv_bf16_src1_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q2_0
    )
}

/// True when `dtype` has BF16-dst mv/mc kernel variants (only with BF16
/// src1). The kernels accumulate in F32 either way and convert at the final
/// store, so a BF16 dst is bit-identical to an F32 dst followed by a
/// cast_f32_bf16 dispatch — it just skips that dispatch.
pub fn quantized_matmul_mv_bf16_dst_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K | GgmlDType::Q2_0
    )
}

/// SoA plane-split q4_K mv experiment (bench-only until the micro gate
/// passes): `rhs` holds [n*nb 16B headers | n*nb 128B quant planes] instead
/// of interleaved 144B blocks. Same grid geometry as the AoS q4_K mv path.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16_soa(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    // Q4K mv geometry: 32-thread TGs (one simdgroup), N_DST=4 rows each.
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, 4),
        height: ne11 as usize,
        depth: ne12 as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 4,
        height: 8,
        depth: 1,
    };
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q4_K_bf16_soa")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_q4k_soa M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
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
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src1_bf16: bool,
    dst_bf16: bool,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    if src1_bf16 && !quantized_matmul_mv_bf16_src1_supported(dtype) {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "bf16 src1",
            "qmatmul_mv",
        ));
    }
    if dst_bf16 && !(src1_bf16 && quantized_matmul_mv_bf16_dst_supported(dtype)) {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "bf16 dst",
            "qmatmul_mv",
        ));
    }
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2_0 => {
            // 2 simdgroups x nr=2 rows -> align 4. Each 128-code block is
            // split across tpb=8 threads.
            let nth0 = 8;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            let nth0 = 4;
            let nth1 = 8;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 2;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    let name = match (dtype, src1_bf16) {
        (GgmlDType::Q8_0, true) if dst_bf16 => "kernel_mul_mv_q8_0_bf16_bf16",
        (GgmlDType::Q4K, true) if dst_bf16 => "kernel_mul_mv_q4_K_bf16_bf16",
        (GgmlDType::Q6K, true) if dst_bf16 => "kernel_mul_mv_q6_K_bf16_bf16",
        (GgmlDType::Q2_0, true) if dst_bf16 => "kernel_mul_mv_q2_0_bf16_bf16",
        (GgmlDType::Q8_0, true) => "kernel_mul_mv_q8_0_bf16",
        (GgmlDType::Q4K, true) => "kernel_mul_mv_q4_K_bf16",
        (GgmlDType::Q6K, true) => "kernel_mul_mv_q6_K_bf16",
        (GgmlDType::Q2_0, true) => "kernel_mul_mv_q2_0_bf16",
        (GgmlDType::Q4_0, _) => "kernel_mul_mv_q4_0_f32",
        (GgmlDType::Q4_1, _) => "kernel_mul_mv_q4_1_f32",
        (GgmlDType::Q5_0, _) => "kernel_mul_mv_q5_0_f32",
        (GgmlDType::Q5_1, _) => "kernel_mul_mv_q5_1_f32",
        (GgmlDType::Q8_0, false) => "kernel_mul_mv_q8_0_f32",
        (GgmlDType::Q2_0, false) => "kernel_mul_mv_q2_0_f32",
        (GgmlDType::Q8_1, _) => "kernel_mul_mv_q8_1_f32",
        (GgmlDType::Q2K, _) => "kernel_mul_mv_q2_K_f32",
        (GgmlDType::Q3K, _) => "kernel_mul_mv_q3_K_f32",
        (GgmlDType::Q4K, false) => "kernel_mul_mv_q4_K_f32",
        (GgmlDType::Q5K, _) => "kernel_mul_mv_q5_K_f32",
        (GgmlDType::Q6K, false) => "kernel_mul_mv_q6_K_f32",
        (GgmlDType::Q8K, _) => "kernel_mul_mv_q8_K_f32",
        (GgmlDType::F16, _) => "kernel_mul_mv_f16_f32",
        (GgmlDType::BF16, _) => "kernel_mul_mv_bf16_f32",
        (GgmlDType::F32, _) => "kernel_mul_mv_f32_f32",
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv {name} B={b} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
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

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// V1 experiment (q4k-mv-rewrite-round2): q4_K bf16-in/bf16-out mv with `nsg`
/// simdgroups per threadgroup. Same per-row arithmetic as the nsg=1 kernel
/// (bit-identical results); the threadgroup count shrinks nsg-fold, which
/// targets the measured launch limiter on huge-n shapes (lm_head: 62k
/// single-simdgroup TGs at nsg=1).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16_nsg(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    nsg: usize,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match nsg {
        1 => "kernel_mul_mv_q4_K_bf16_bf16",
        2 => "kernel_mul_mv_q4_K_bf16_bf16_nsg2",
        4 => "kernel_mul_mv_q4_K_bf16_bf16_nsg4",
        8 => "kernel_mul_mv_q4_K_bf16_bf16_nsg8",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "nsg must be 1/2/4/8",
                "qmatmul_mv_nsg",
            ))
        }
    };
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_nsg{nsg} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            ne12,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    // N_DST = 4 rows per simdgroup; nsg simdgroups per TG.
    let thread_groups_count = MTLSize {
        width: divide(n, 4 * nsg),
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 4,
        height: 8,
        depth: nsg,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Round-3 geometry variants: `nsg` simdgroups x `ndst` rows each, plus the
/// f32-activation discriminator arm. Bit-identical per row to the baseline
/// (per-row arithmetic is geometry-independent); only row->simdgroup
/// assignment and threadgroup shape change. `f32_y` selects the f32-y
/// instantiation of the BASELINE geometry (nsg/ndst ignored in that case).
#[allow(clippy::too_many_arguments)]
/// Wide q4_K matvec: VECS activation rows streamed per weight decode (the
/// verify-chunk kernel; see kernel_mul_mv_q4_K_wide_impl). One tile of up to
/// 4 vectors per tgpig.x step; rows follow the mv geometry (8 per
/// threadgroup: 2 simdgroups x 4 rows). bf16 activations and dst only.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16_wide(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, k): (usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    if m < 2 || !k.is_multiple_of(256) {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "wide wants m >= 2 and k % 256 == 0",
            "qmatmul_mv_wide",
        ));
    }
    // Exact-fit tiles for the spec-verify chunk sizes; m > 4 tiles by 4
    // (each extra tile re-reads the weights, like MLX's qmv_wide).
    let (name, vecs) = match m {
        2 => ("kernel_mul_mv_q4_K_bf16_bf16_wide_v2", 2),
        3 => ("kernel_mul_mv_q4_K_bf16_bf16_wide_v3", 3),
        _ => ("kernel_mul_mv_q4_K_bf16_bf16_wide_v4", 4),
    };
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_wide{vecs} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            1i64,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            1i64,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    let thread_groups_count = MTLSize {
        width: divide(m, vecs),
        height: divide(n, 8),
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 32,
        height: 2,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// MSL-4.1 packed_numeric variant (nr2sg2 geometry, unpack inner loop).
/// Compiles only on macOS 27+; callers treat a load error as "route to the
/// default kernel" (see the m3-macos26-eval-suite ticket receipts).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16_unpk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let (nsg, ndst) = (2usize, 2usize);
    let name = "kernel_mul_mv_q4_K_bf16_bf16_unpk";
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    let pipeline = kernels.load_pipeline(device, Source::QuantizedUnpk, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_unpk M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            ne12,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    let thread_groups_count = MTLSize {
        width: divide(n, ndst * nsg),
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 4,
        height: 8,
        depth: nsg,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Tensor-op (matmul2d) q4_K matmul for m in [1,8]: exact q4_K semantics on
/// the repacked plane layout (nibbles [k, n_pad] + fp16 dsc/dmm planes; see
/// mm2d_q4k.metal). One dispatch per linear (the dmin row sums are computed
/// in-kernel). dst is bf16 [m, n]. The M tile is hardware-fixed at 8;
/// m <= 8 rides one tile; k caps at 8192 (threadgroup row-sum budget).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm2d_q4k(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, n_pad, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    nibbles: &Buffer,
    dsc: &Buffer,
    dmm: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    debug_assert!(m <= 8 && k % 32 == 0 && k <= 8192 && n_pad % 64 == 0);
    // Tile geometry: probe12's ISOLATED latencies favoured the 32-wide
    // single-simdgroup tile on wide-N shapes, but IN-LOOP it regressed
    // verify by ~2.6ms/round (2026-07-14 A/B on the head shape) — isolated
    // dispatch latency does not transfer to a dependent layer chain. The
    // 64-wide tile stays the default everywhere; LMBRRR_MM2D_TILE=32 keeps
    // the variant reachable for in-loop study.
    static TILE_OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let override_tile = *TILE_OVERRIDE.get_or_init(|| {
        std::env::var("LMBRRR_MM2D_TILE")
            .ok()
            .and_then(|v| v.parse().ok())
    });
    let tile = override_tile.unwrap_or(64);
    let (name, tg_threads) = if tile == 32 {
        ("kernel_mul_mm2d_q4k_bf16_t32", 32)
    } else {
        ("kernel_mul_mm2d_q4k_bf16", 128)
    };
    let pipeline = kernels.load_pipeline(device, Source::Mm2dQ4k, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm2d_q4k M={m} K={k} N={n} t{tile}");
    let dims: [i32; 4] = [k as i32, n_pad as i32, n as i32, m as i32];
    set_params!(
        encoder,
        (
            (lhs, lhs_offset),
            nibbles,
            dsc,
            dmm,
            Output::with_offset(dst, dst_offset),
            &dims[..]
        )
    );
    let thread_groups_count = MTLSize {
        width: n_pad / tile,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: tg_threads,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// A compile-time `mm2d_q2_0` kernel variant (see the `instantiate_mm2d_q2_0`
/// list in mm2d_q2_0.metal). `tile_n` is the output-N tile (threadgroup count =
/// n_pad / tile_n); `tg_threads` is `NSIMD * 32`.
#[derive(Clone, Copy, Debug)]
pub struct Mm2dQ2Variant {
    pub kernel: &'static str,
    pub tile_n: usize,
    pub tg_threads: usize,
}

impl Mm2dQ2Variant {
    pub const T64_K32: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t64_k32",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const T64_K64: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t64_k64",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const T64_K128: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t64_k128",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const T64_K128_RELAXED: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t64_k128_relaxed",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const T32_K128: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t32_k128",
        tile_n: 32,
        tg_threads: 32,
    };
    pub const T32_K128_RELAXED: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_t32_k128_relaxed",
        tile_n: 32,
        tg_threads: 32,
    };
    /// Diagnostic probe (numerically WRONG): matmul structure without the fold
    /// epilogue — isolates the scalar-fold instruction cost. See mm2d_q2_0.metal.
    pub const PROBE_NOFOLD_T64_K128: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_nofold_t64_k128",
        tile_n: 64,
        tg_threads: 128,
    };
    /// Diagnostic probe (numerically WRONG): a single op.run over the full K
    /// (dynamic_extent), no host K-loop — separates the op's MMA throughput from
    /// the discrete-loop machinery. See mm2d_q2_0.metal.
    pub const PROBE_FULLK_T64: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_fullk_t64",
        tile_n: 64,
        tg_threads: 128,
    };
    /// M-tile sweep probes (does a bigger tile amortize the weight read for free?).
    pub const PROBE_FULLK_T64_M16: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_fullk_t64_m16",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const PROBE_FULLK_T64_M32: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_fullk_t64_m32",
        tile_n: 64,
        tg_threads: 128,
    };
    pub const PROBE_FULLK_T32: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_fullk_t32",
        tile_n: 32,
        tg_threads: 32,
    };
    pub const PROBE_FULLK_T128: Self = Self {
        kernel: "kernel_mul_mm2d_q2_0_probe_fullk_t128",
        tile_n: 128,
        tg_threads: 256,
    };
    /// Every variant, for benchmark sweeps.
    pub const ALL: [Self; 12] = [
        Self::T64_K32,
        Self::T64_K64,
        Self::T64_K128,
        Self::T64_K128_RELAXED,
        Self::T32_K128,
        Self::T32_K128_RELAXED,
        Self::PROBE_NOFOLD_T64_K128,
        Self::PROBE_FULLK_T64,
        Self::PROBE_FULLK_T64_M16,
        Self::PROBE_FULLK_T64_M32,
        Self::PROBE_FULLK_T32,
        Self::PROBE_FULLK_T128,
    ];
    /// Production default (the sweep winner).
    pub const DEFAULT: Self = Self::T64_K128;
}

/// Tensor-op (matmul2d) Q2_0 (ternary) matmul for m in [1,8], the ternary
/// mirror of [`call_quantized_matmul_mm2d_q4k`]: the packed 2-bit codes are the
/// hardware uint2b_format weight operand and a single per-128 fp16 `d` plane
/// folds `d*(P - rowsum)` in the epilogue (vs q4_K's dsc*P - dmm*rowsum). Plane
/// layout: codes [k, n_pad] 2-bit + d [k/128, n_pad] fp16 (candle-core
/// q2_0_mm2d_planes). One dispatch per linear; dst is bf16 [m, n]. Requires the
/// Metal 4.1 toolchain (see mm2d_q2_0.metal); on unsupported toolchains
/// pipeline creation fails and callers fall back by routing.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm2d_q2_0(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, n_pad, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    codes: &Buffer,
    d: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
    variant: Mm2dQ2Variant,
) -> Result<(), MetalKernelError> {
    debug_assert!(m <= 8 && k % 128 == 0 && k <= 8192 && n_pad % 64 == 0);
    let pipeline = kernels.load_pipeline(device, Source::Mm2dQ2_0, variant.kernel)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm2d_q2_0 M={m} K={k} N={n} {}", variant.kernel);
    let dims: [i32; 4] = [k as i32, n_pad as i32, n as i32, m as i32];
    set_params!(
        encoder,
        (
            (lhs, lhs_offset),
            codes,
            d,
            Output::with_offset(dst, dst_offset),
            &dims[..]
        )
    );
    let thread_groups_count = MTLSize {
        width: n_pad / variant.tile_n,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: variant.tg_threads,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Split-K variant of [`call_quantized_matmul_mm2d_q4k`] for small-N shapes:
/// the K/32 slices are partitioned across `n_splits` threadgroup rows
/// writing f32 partials into caller-provided scratch (`n_splits * 8 * n_pad`
/// f32), reduced by a second dispatch into the bf16 dst. Exact — the fold
/// is linear over slices. `n_splits` must not exceed k/32.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm2d_q4k_splitk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, n_pad, k, n_splits): (usize, usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    nibbles: &Buffer,
    dsc: &Buffer,
    dmm: &Buffer,
    partials: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    debug_assert!(m <= 8 && k % 32 == 0 && k <= 8192 && n_pad % 64 == 0);
    debug_assert!(n_splits >= 1 && n_splits <= k / 32);
    let dims: [i32; 4] = [k as i32, n_pad as i32, n as i32, m as i32];
    let ns = n_splits as i32;

    let pipeline =
        kernels.load_pipeline(device, Source::Mm2dQ4k, "kernel_mul_mm2d_q4k_bf16_splitk")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm2d_q4k_splitk M={m} K={k} N={n} S={n_splits}");
    // partials registers as an output so the hazard tracker barriers the
    // reduce's read (the rowsums lesson).
    set_params!(
        encoder,
        (
            (lhs, lhs_offset),
            nibbles,
            dsc,
            dmm,
            Output::with_offset(partials, 0),
            &dims[..],
            ns
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n_pad / 64,
            height: n_splits,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );

    let reduce =
        kernels.load_pipeline(device, Source::Mm2dQ4k, "kernel_mm2d_q4k_splitk_reduce")?;
    encoder.set_compute_pipeline_state(&reduce);
    debug_group!(encoder, "qmm_mm2d_q4k_splitk_reduce N={n} M={m}");
    set_params!(
        encoder,
        (
            partials,
            Output::with_offset(dst, dst_offset),
            &dims[..],
            ns
        )
    );
    encoder.dispatch_threads(
        MTLSize {
            width: n,
            height: m,
            depth: 1,
        },
        MTLSize {
            width: 64.min(n),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Fused m-row head argmax on the mm2d plane layout: per-tile fold +
/// bf16-rounded per-row best (lowest-index ties, matching fast_argmax on
/// the stored tensor) + a per-row reduce. `out` receives m u32 ids; the
/// m x n logits tensor is never materialized. Scratch: pval (n_pad/64 * 8
/// f32), pidx (same, u32).
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm2d_q4k_argmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, n_pad, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    nibbles: &Buffer,
    dsc: &Buffer,
    dmm: &Buffer,
    pval: &Buffer,
    pidx: &Buffer,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    debug_assert!(m <= 8 && k % 32 == 0 && k <= 8192 && n_pad % 64 == 0);
    let dims: [i32; 4] = [k as i32, n_pad as i32, n as i32, m as i32];
    let n_tiles = n_pad / 64;

    let pipeline =
        kernels.load_pipeline(device, Source::Mm2dQ4k, "kernel_mul_mm2d_q4k_argmax")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mm2d_q4k_argmax M={m} K={k} N={n}");
    set_params!(
        encoder,
        (
            (lhs, lhs_offset),
            nibbles,
            dsc,
            dmm,
            Output::with_offset(pval, 0),
            Output::with_offset(pidx, 0),
            &dims[..]
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: n_tiles,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );

    let reduce =
        kernels.load_pipeline(device, Source::Mm2dQ4k, "kernel_mm2d_q4k_argmax_reduce")?;
    encoder.set_compute_pipeline_state(&reduce);
    debug_group!(encoder, "qmm_mm2d_q4k_argmax_reduce M={m}");
    let nt = n_tiles as i32;
    set_params!(encoder, (pval, pidx, Output::with_offset(out, 0), nt));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: m,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn call_quantized_matmul_mv_q4k_bf16_geo(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (nsg, ndst, f32_y): (usize, usize, bool),
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    // nsg == 0 encodes the batch-fastest nr2sg2 variant (swizzled tgpig;
    // grid transposed below so the m rows co-schedule on the same weights).
    let batch_fast = nsg == 0;
    let (nsg, ndst) = if batch_fast { (2, 2) } else { (nsg, ndst) };
    let name = match (batch_fast, nsg, ndst, f32_y) {
        (true, _, _, false) => "kernel_mul_mv_q4_K_bf16_bf16_nr2sg2_bfast",
        (false, 1, 4, true) => "kernel_mul_mv_q4_K_f32y_bf16_base",
        (false, 2, 2, false) => "kernel_mul_mv_q4_K_bf16_bf16_nr2sg2",
        (false, 1, 2, false) => "kernel_mul_mv_q4_K_bf16_bf16_nr2sg1",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "geo wants (2,2,false), (1,2,false), (1,4,true) or (0,_,false)",
                "qmatmul_mv_geo",
            ))
        }
    };
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_geo{nsg}x{ndst} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            ne12,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    let thread_groups_count = if batch_fast {
        MTLSize {
            width: m,
            height: divide(n, ndst * nsg),
            depth: b,
        }
    } else {
        MTLSize {
            width: divide(n, ndst * nsg),
            height: m,
            depth: b,
        }
    };
    let threads_per_threadgroup = MTLSize {
        width: 4,
        height: 8,
        depth: nsg,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Rows per threadgroup in the fused argmax head kernel
/// (ARGMAX_NTILES * ARGMAX_NSG * ARGMAX_NDST in quantized.metal). The
/// partials buffer needs `ceil(n / this)` 8-byte entries.
pub const MV_ARGMAX_ROWS_PER_TG: usize = 32;

/// Fused greedy head: q4_K GEMV whose per-row arithmetic is byte-for-byte
/// the nr2sg2 production kernel, reduced straight to the argmax row index
/// (bf16-rounded comparison, candle's lowest-index tie rule) without ever
/// materializing the logits. m=1 only; `out` receives one u32.
pub fn call_quantized_matmul_mv_q4k_argmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (n, k): (usize, usize),
    lhs: (&Buffer, usize),
    rhs: (&Buffer, usize),
    partials: &Buffer,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let partial = kernels.load_pipeline(
        device,
        Source::Quantized,
        "kernel_mul_mv_q4_K_bf16_argmax_partial",
    )?;
    let reduce = kernels.load_pipeline(device, Source::Quantized, "kernel_mv_argmax_reduce")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    let ne00 = k as i64;
    let ne01 = n as i64;
    let ntg = divide(n, MV_ARGMAX_ROWS_PER_TG);

    encoder.set_compute_pipeline_state(&partial);
    debug_group!(encoder, "qmv_argmax_partial N={n} K={k}");
    set_params!(encoder, (rhs, lhs, Output::new(partials), ne00, ne01));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: ntg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 4,
            height: 8,
            depth: 2,
        },
    );

    encoder.set_compute_pipeline_state(&reduce);
    debug_group!(encoder, "qmv_argmax_reduce ntg={ntg}");
    let ntg_u32 = ntg as u32;
    set_params!(encoder, (partials, Output::new(out), ntg_u32));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// V5 experiment (thread-lifetime hypothesis): q4_K bf16/bf16 mv where each
/// simdgroup processes `ntiles` consecutive 4-row groups sequentially —
/// thread lifetime x ntiles, launch churn / ntiles, bit-identical per row.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q4k_bf16_rowtile(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ntiles: usize,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let name = match ntiles {
        2 => "kernel_mul_mv_q4_K_bf16_bf16_rt2",
        4 => "kernel_mul_mv_q4_K_bf16_bf16_rt4",
        8 => "kernel_mul_mv_q4_K_bf16_bf16_rt8",
        16 => "kernel_mul_mv_q4_K_bf16_bf16_rt16",
        32 => "kernel_mul_mv_q4_K_bf16_bf16_rt32",
        _ => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                "ntiles must be 2/4/8/16/32",
                "qmatmul_mv_rowtile",
            ))
        }
    };
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = 1;
    let r3: u32 = 1;

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_rt{ntiles} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            ne12,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );
    let thread_groups_count = MTLSize {
        width: divide(n, 4 * ntiles),
        height: m,
        depth: b,
    };
    let threads_per_threadgroup = MTLSize {
        width: 4,
        height: 8,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Column count handled per threadgroup by the multi-column mv kernels, or
/// None when the dtype has no `_mc` variant. Must match the NC_MV_* defines in
/// quantized.metal.
/// Coalesced weight-bound Q2_0 verify GEMM (2<=m<=8) over a planar repack
/// (`codes` [k][n_pad] 2-bit, `dscale` [k/128][n_pad]). Activation is f32
/// contiguous [m, k]; dst is f32 [m, n].
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm2d_q2_0_smallm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, k, n_pad): (usize, usize, usize, usize),
    codes: &Buffer,
    dscale: &Buffer,
    lhs: &Buffer,
    lhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne1 = m as i64;
    let npad = n_pad as i64;
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mm2d_q2_0_smallm")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            codes,
            dscale,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne1,
            npad
        )
    );
    encoder.dispatch_thread_groups(
        MTLSize { width: divide(n_pad, 64), height: 1, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
    Ok(())
}

pub fn quantized_matmul_mv_mc_columns(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q8_0 => Some(8),
        GgmlDType::Q4K => Some(8),
        GgmlDType::Q6K => Some(8),
        // Small-m verify (block-size 4-5): the mc kernel (weight shared across
        // columns) beats the tile mm, which under-occupies + pays a bf16->f32
        // cast at m<8. The tile mm still serves large-m prefill (m>=8).
        GgmlDType::Q2_0 => Some(8),
        _ => None,
    }
}

/// Weight-shared small-m quantized matmul: one dispatch covering all m src1
/// rows, reading the quantized weights ceil(m / NC) times instead of m times.
/// See the multi-column section of quantized.metal.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_mc(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src1_bf16: bool,
    dst_bf16: bool,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let nc = quantized_matmul_mv_mc_columns(dtype).ok_or_else(|| {
        MetalKernelError::UnsupportedDTypeForOp("no mc variant", "qmatmul_mv_mc")
    })?;
    if dst_bf16 && !(src1_bf16 && quantized_matmul_mv_bf16_dst_supported(dtype)) {
        return Err(MetalKernelError::UnsupportedDTypeForOp(
            "bf16 dst",
            "qmatmul_mv_mc",
        ));
    }
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

    let (name, nth0, nth1, align) = match (dtype, src1_bf16) {
        (GgmlDType::Q8_0, true) if dst_bf16 => ("kernel_mul_mv_q8_0_bf16_bf16_mc", 8, 8, 8),
        (GgmlDType::Q2_0, true) if dst_bf16 => ("kernel_mul_mv_q2_0_bf16_bf16_mc", 8, 8, 4),
        (GgmlDType::Q4K, true) if dst_bf16 => ("kernel_mul_mv_q4_K_bf16_bf16_mc", 4, 8, 4),
        (GgmlDType::Q6K, true) if dst_bf16 => ("kernel_mul_mv_q6_K_bf16_bf16_mc", 2, 32, 2),
        (GgmlDType::Q8_0, false) => ("kernel_mul_mv_q8_0_f32_mc", 8, 8, 8),
        (GgmlDType::Q8_0, true) => ("kernel_mul_mv_q8_0_bf16_mc", 8, 8, 8),
        (GgmlDType::Q2_0, false) => ("kernel_mul_mv_q2_0_f32_mc", 8, 8, 4),
        (GgmlDType::Q2_0, true) => ("kernel_mul_mv_q2_0_bf16_mc", 8, 8, 4),
        (GgmlDType::Q4K, false) => ("kernel_mul_mv_q4_K_f32_mc", 4, 8, 4),
        (GgmlDType::Q4K, true) => ("kernel_mul_mv_q4_K_bf16_mc", 4, 8, 4),
        (GgmlDType::Q6K, false) => ("kernel_mul_mv_q6_K_f32_mc", 2, 32, 2),
        (GgmlDType::Q6K, true) => ("kernel_mul_mv_q6_K_bf16_mc", 2, 32, 2),
        _ => unreachable!("gated by quantized_matmul_mv_mc_columns"),
    };
    // A/B toggle: LMBRRR_Q2_MC2=1 routes the Q2_0 verify GEMV to the ILP-fixed
    // mc2 kernel (independent column accumulators) instead of the serial mc.
    static MC2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let use_mc2 = *MC2.get_or_init(|| std::env::var("LMBRRR_Q2_MC2").is_ok());
    let name: &str = if use_mc2 && matches!(dtype, GgmlDType::Q2_0) {
        match name {
            "kernel_mul_mv_q2_0_f32_mc" => "kernel_mul_mv_q2_0_f32_mc2",
            "kernel_mul_mv_q2_0_bf16_mc" => "kernel_mul_mv_q2_0_bf16_mc2",
            "kernel_mul_mv_q2_0_bf16_bf16_mc" => "kernel_mul_mv_q2_0_bf16_bf16_mc2",
            other => other,
        }
    } else {
        name
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: divide(m, nc),
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
    debug_group!(encoder, "qmm_mv_mc {name} B={b} M={m} K={k} N={n}");

    set_params!(
        encoder,
        (
            rhs,
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            ne12,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// Transposed-activation Q2_0 verify GEMV (mct): same geometry as the Q2_0 mc,
/// but `act_t` is the activation TRANSPOSED to `[K][M]` (bf16), so the NC column
/// reads are contiguous (the mc's L1-thrash fix). dst is f32 `[m, n]`
/// (`dst[col*n + row]`). Diagnostic wrapper for the bench A/B.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_q2_0_mct(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (m, n, k): (usize, usize, usize),
    weights: &Buffer,
    act_t: &Buffer,
    act_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let (nc, nth0, nth1, align) = (8usize, 8usize, 8usize, 4usize);
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne0 = n as i64;
    let ne1 = m as i64;
    let pipeline =
        kernels.load_pipeline(device, Source::Quantized, "kernel_mul_mv_q2_0_bf16_mct")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qmm_mv_mct M={m} K={k} N={n}");
    set_params!(
        encoder,
        (
            weights,
            (act_t, act_offset),
            dst,
            ne00,
            ne01,
            1i64,
            0i64,
            0i64,
            0i64,
            ne10,
            ne11,
            1i64,
            0i64,
            0i64,
            0i64,
            ne0,
            ne1,
            1u32,
            1u32
        )
    );
    let thread_groups_count = MTLSize {
        width: divide(n, align),
        height: divide(m, nc),
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
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
        GgmlDType::Q2_0 => "kernel_mul_mm_q2_0_f32",
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
    debug_group!(encoder, "qmm_mm {name} M={ne11} K={ne00} N={ne01}");

    set_params!(
        encoder,
        (
            src0,
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
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

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_get_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    hidden_size: usize,
    row_stride: usize,
    ids_len: usize,
    src: &Buffer,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let dst_row_stride = hidden_size * core::mem::size_of::<f32>();
    let name = match dtype {
        GgmlDType::F32 => "kernel_get_rows_f32",
        GgmlDType::F16 => "kernel_get_rows_f16",
        GgmlDType::BF16 => "kernel_get_rows_bf16",
        GgmlDType::Q4_0 => "kernel_get_rows_q4_0",
        GgmlDType::Q4_1 => "kernel_get_rows_q4_1",
        GgmlDType::Q5_0 => "kernel_get_rows_q5_0",
        GgmlDType::Q5_1 => "kernel_get_rows_q5_1",
        GgmlDType::Q8_0 => "kernel_get_rows_q8_0",
        GgmlDType::Q2K => "kernel_get_rows_q2_K",
        GgmlDType::Q3K => "kernel_get_rows_q3_K",
        GgmlDType::Q4K => "kernel_get_rows_q4_K",
        GgmlDType::Q5K => "kernel_get_rows_q5_K",
        GgmlDType::Q6K => "kernel_get_rows_q6_K",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "get_rows"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "get_rows"))?,
        // Packed-embedding lookup lands with the GGUF loader; dequant for now.
        GgmlDType::Q2_0 => Err(MetalKernelError::UnsupportedDTypeForOp("Q2_0", "get_rows"))?,
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qget_rows {name} ids={ids_len} hidden={hidden_size}"
    );

    let thread_groups_count = MTLSize {
        width: ids_len,
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };

    set_params!(
        encoder,
        (
            src,
            (ids, ids_offset),
            Output::new(dst),
            hidden_size as i64,
            row_stride as u64,
            0u64,
            ids_len as i64,
            core::mem::size_of::<u32>() as u64,
            0u64,
            dst_row_stride as u64,
            0u64
        )
    );

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}
