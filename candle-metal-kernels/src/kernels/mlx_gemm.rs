use crate::metal::{Buffer, ComputeCommandEncoder, Device, MetalDeviceType};
use crate::utils::{EncoderProvider, Input};
use crate::{
    debug_group, set_params, ConstantValues, EncoderParam, Kernels, MetalKernelError, Output,
    Source, Value,
};
use objc2_metal::MTLSize;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

/// Unary activation fused into the GEMM/GEMV epilogue, applied after the
/// optional bias. Selected through function constant 120 in mlx_gemm.metal
/// and gemv.metal; the kernel formulas mirror unary.metal's urelu/ugelu/usilu
/// so the fused result matches the composed matmul + unary chain.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum GemmActivation {
    #[default]
    None,
    Relu,
    /// The tanh-approximation gelu (candle's `Tensor::gelu`).
    Gelu,
    Silu,
}

impl GemmActivation {
    fn constant_value(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Relu => 1,
            Self::Gelu => 2,
            Self::Silu => 3,
        }
    }
}

/// Tile configuration for GEMM kernel.
///
/// These parameters control the block sizes and warp tiling for the Metal GEMM kernel.
/// Different configurations are optimal for different matrix sizes and data types.
///
/// Reference: MLX steel_gemm_fused.metal
#[derive(Copy, Clone, Debug)]
struct TileConfig {
    bm: usize, // Block size M
    bn: usize, // Block size N
    bk: usize, // Block size K
    wm: usize, // Warp tiles M
    wn: usize, // Warp tiles N
}

impl TileConfig {
    const fn new(bm: usize, bn: usize, bk: usize, wm: usize, wn: usize) -> Self {
        Self { bm, bn, bk, wm, wn }
    }
}

// Predefined tile configurations matching MLX's steel_gemm_fused.metal
// Note: TILE_32_32_16_2_2 is kept for backward compatibility and as a fallback.
// It's used by MLX for small devices ('g'/'p') but we default to medium device configs.
#[allow(dead_code)]
const TILE_32_32_16_2_2: TileConfig = TileConfig::new(32, 32, 16, 2, 2);
const TILE_64_64_16_2_2: TileConfig = TileConfig::new(64, 64, 16, 2, 2);
const TILE_64_64_16_1_2: TileConfig = TileConfig::new(64, 64, 16, 1, 2);
const TILE_64_32_32_2_2: TileConfig = TileConfig::new(64, 32, 32, 2, 2);
const TILE_32_64_16_1_2: TileConfig = TileConfig::new(32, 64, 16, 1, 2);

/// Select optimal tile configuration based on matrix dimensions, data type, transpose mode,
/// and device type.
///
/// This implements MLX's GEMM_TPARAM_MACRO tile selection logic.
/// Reference: refs/mlx/mlx/backend/metal/matmul.cpp lines 88-170
///
/// The selection is based on:
/// - Device type (phone/base-pro for small, ultra for large, others for medium)
/// - Total output size (batch_size * M * N)
/// - Data type (F32 vs F16/BF16)
/// - Transpose mode (nn, nt, tn, tt)
/// - K dimension relative to M and N
fn select_tile_config(
    dtype: GemmDType,
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    a_trans: bool,
    b_trans: bool,
    device_type: MetalDeviceType,
) -> TileConfig {
    // Special case: For very small M (vector-matrix multiply),
    // use the original 32x32 tile to avoid thread waste.
    // When M is very small (< bm), using larger bm values causes significant
    // thread underutilization because most threads in the M dimension have no work.
    // This is critical for benchmarks like [1, 2048] @ [2048, 2048] (m=1).
    //
    // We use m < 16 as the threshold because:
    // - For m=1 to m=15, even 32x32 tile has some waste but it's the smallest available
    // - For m >= 16, the larger tiles can provide better throughput despite some waste
    if m < 16 {
        return TILE_32_32_16_2_2;
    }

    // MLX uses batch_size * M * N >= 1M as the threshold for "large matmul"
    let total_output = batch_size * m * n;
    let is_large_matmul = total_output >= (1 << 20); // 1M elements

    match device_type {
        // Small devices: phone ('p') and base/pro ('g')
        MetalDeviceType::Phone | MetalDeviceType::BasePro => {
            // MLX: if (devc == 'g' || devc == 'p')
            if !a_trans && b_trans {
                // nt mode
                TILE_64_32_32_2_2
            } else if dtype != GemmDType::F32 {
                // half and bfloat
                TILE_64_64_16_1_2
            } else {
                // float32 default
                TILE_64_64_16_2_2
            }
        }
        // Large device: ultra ('d')
        MetalDeviceType::Ultra => {
            // MLX: if (devc == 'd')
            if is_large_matmul {
                // Large matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if 2 * m.max(n) > k {
                        // Reasonable K
                        TILE_64_64_16_1_2
                    } else if !a_trans && b_trans {
                        // nt with large K
                        TILE_64_32_32_2_2
                    } else {
                        // nn with large K
                        TILE_32_64_16_1_2
                    }
                } else {
                    // float32 takes default
                    TILE_64_64_16_2_2
                }
            } else {
                // Smaller matmul
                if dtype != GemmDType::F32 {
                    // half and bfloat
                    if !a_trans && b_trans {
                        // nt
                        TILE_64_32_32_2_2
                    } else {
                        // nn
                        TILE_64_64_16_1_2
                    }
                } else {
                    // floats
                    if !a_trans && b_trans {
                        // nt
                        TILE_32_64_16_1_2
                    } else {
                        // nn
                        TILE_64_32_32_2_2
                    }
                }
            }
        }
        // Medium devices: max ('s') and unknown
        MetalDeviceType::Max | MetalDeviceType::Medium => {
            // MLX: default medium device config
            // Use the same logic as before but with medium device defaults
            match dtype {
                GemmDType::F32 => {
                    if !is_large_matmul {
                        if !a_trans && b_trans {
                            TILE_32_64_16_1_2
                        } else {
                            TILE_64_32_32_2_2
                        }
                    } else {
                        TILE_64_64_16_2_2
                    }
                }
                GemmDType::F16 | GemmDType::BF16 => {
                    if is_large_matmul {
                        if 2 * m.max(n) > k {
                            TILE_64_64_16_1_2
                        } else if !a_trans && b_trans {
                            TILE_64_32_32_2_2
                        } else {
                            TILE_32_64_16_1_2
                        }
                    } else if !a_trans && b_trans {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_64_64_16_1_2
                    }
                }
            }
        }
    }
}

/// Check if batch can be collapsed into M dimension.
///
/// MLX's batch collapse optimization (from matmul.cpp lines 700-740):
/// When B is broadcasted (2D), we can collapse batch into M dimension:
/// - [batch, M, K] @ [K, N] -> [batch*M, K] @ [K, N]
///
/// Conditions for batch collapse:
/// 1. batch_size > 1
/// 2. !transpose_a (A is not transposed, i.e., row-major for M dimension)
/// 3. A is contiguous in batch dimension (batch_stride_a == M * K)
/// 4. B is broadcasted (batch_stride_b == 0, meaning B is 2D)
///
/// Returns (effective_batch, effective_m, should_collapse)
fn check_batch_collapse(
    b: usize,
    m: usize,
    k: usize,
    a_trans: bool,
    lhs_stride: &[usize],
    rhs_stride: &[usize],
) -> (usize, usize, bool) {
    if b <= 1 {
        return (b, m, false);
    }

    // A must not be transposed for batch collapse
    if a_trans {
        return (b, m, false);
    }

    // Check A's batch stride - must be contiguous (batch_stride_a == M * K)
    let a_batch_stride = if lhs_stride.len() > 2 {
        lhs_stride[lhs_stride.len() - 3]
    } else {
        m * k
    };

    // Check B's batch stride - must be 0 (broadcasted) for collapse
    let b_batch_stride = if rhs_stride.len() > 2 {
        rhs_stride[rhs_stride.len() - 3]
    } else {
        0 // B is 2D, effectively broadcasted
    };

    // For batch collapse:
    // - A must be contiguous: batch_stride_a == M * K
    // - B must be broadcasted: batch_stride_b == 0
    let a_contiguous = a_batch_stride == m * k;
    let b_broadcasted = b_batch_stride == 0;

    if a_contiguous && b_broadcasted {
        // Collapse batch into M: new_m = batch * m, new_batch = 1
        (1, b * m, true)
    } else {
        (b, m, false)
    }
}

/// Check if we can use split-K strategy for better performance.
///
/// MLX uses split-K when:
/// - batch_size == 1
/// - (M/16) * (N/16) <= 32 (small output)
/// - K/16 >= 8 (large K)
///
/// This is useful for tall-skinny matrices where K >> M*N
#[allow(dead_code)]
fn should_use_split_k(b: usize, m: usize, n: usize, k: usize) -> bool {
    if b != 1 {
        return false;
    }
    let tm = m / 16;
    let tn = n / 16;
    let tk = k / 16;
    (tm * tn) <= 32 && tk >= 8
}

/// M=1 -> gemv_t (vec[K] x mat[K,N] -> vec[N])
/// N=1 -> gemv   (mat[M,K] x vec[K] -> vec[M])
///
/// `bias` fuses `out += bias` into the kernel epilogue (the `_axpby1`
/// variants with alpha = beta = 1). The bias vector has `out_vec_size`
/// elements and broadcasts across the batch. `activation` applies a fused
/// unary op after the bias.
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemv(
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
    bias: Option<(&Buffer, usize)>,
    activation: GemmActivation,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    debug_assert!(m == 1 || n == 1, "call_mlx_gemv requires M=1 or N=1");

    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);

    // Determine transpose flags from strides (same logic as call_mlx_gemm)
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    // Figure out if transpose is needed.
    let is_b_matrix = n != 1;
    let transpose_mat = if is_b_matrix { !b_trans } else { a_trans };
    let mat_ld = if is_b_matrix {
        ldb as usize
    } else {
        lda as usize
    };
    let in_vec_size = k;
    let out_vec_size = if is_b_matrix { n } else { m };

    let (mat_buffer, mat_offset, vec_buffer, vec_offset) = if is_b_matrix {
        (rhs_buffer, rhs_offset, lhs_buffer, lhs_offset)
    } else {
        (lhs_buffer, lhs_offset, rhs_buffer, rhs_offset)
    };

    // Batch strides (elements per batch item)
    let vec_batch_stride: i64 = if is_b_matrix {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    } else {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    };
    // Weight matrix is often 2D (shared across batch) -> stride = 0
    let mat_batch_stride: i64 = if is_b_matrix {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            0
        }
    } else {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            0
        }
    };

    // Tile selection
    let (bm, bn, sm, sn, tm, tn) = if transpose_mat {
        // gemv_t: vec[K] x mat[K,N_out] -> out[N_out]
        let (sm, sn) = if in_vec_size >= 8192 && out_vec_size >= 2048 {
            (4usize, 8usize)
        } else {
            (8, 4)
        };
        let bn = if out_vec_size >= 2048 {
            16usize
        } else if out_vec_size >= 512 {
            4
        } else {
            2
        };
        let tn: usize = if out_vec_size < 4 { 1 } else { 4 };
        (1usize, bn, sm, sn, 4usize, tn)
    } else {
        // gemv: mat[M_out,K] x vec[K] -> out[M_out]
        let (bm, bn, sm, sn): (usize, usize, usize, usize) = if in_vec_size <= 64 {
            (1, 1, 8, 4)
        } else if in_vec_size >= 16 * out_vec_size {
            (1, 8, 1, 32)
        } else if out_vec_size >= 4096 {
            (8, 1, 1, 32)
        } else {
            (4, 1, 1, 32)
        };
        let tm: usize = if out_vec_size < 4 { 1 } else { 4 };
        (bm, bn, sm, sn, tm, 4usize)
    };

    let dtype_str = match dtype {
        GemmDType::F32 => "float32",
        GemmDType::F16 => "float16",
        GemmDType::BF16 => "bfloat16",
    };
    let kernel_prefix = if transpose_mat { "gemv_t" } else { "gemv" };
    let axpby = if bias.is_some() { 1 } else { 0 };
    let name = format!(
        "{}_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc0_axpby{}",
        kernel_prefix, dtype_str, bm, bn, sm, sn, tm, tn, axpby
    );

    // The gemv kernels always reference the activation function constant, so
    // it must be set even for the identity case.
    let constants = Some(ConstantValues::new(vec![(
        120,
        Value::U16(activation.constant_value()),
    )]));
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemv, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let batch_shape = [b as i32];
    let vec_batch_strides = [vec_batch_stride];
    let mat_batch_strides = [mat_batch_stride];
    let bias_batch_strides = [0i64];
    // The kernel only reads the bias when compiled with axpby: out is then
    // alpha * result + beta * bias.
    let beta = if bias.is_some() { 1.0f32 } else { 0.0f32 };

    set_params!(
        encoder,
        (
            Input::with_offset(mat_buffer, mat_offset),
            Input::with_offset(vec_buffer, vec_offset),
            (), // bias, bound below when present
            Output::new(output),
            in_vec_size as i32,
            out_vec_size as i32,
            mat_ld as i32,
            1.0f32, // alpha
            beta,
            1i32, // batch_ndim
            &batch_shape[..],
            &vec_batch_strides[..],
            &mat_batch_strides[..],
            &bias_batch_strides[..],
            1i32 // bias_stride
        )
    );
    if let Some((bias, bias_offset)) = bias {
        encoder.set_input_buffer(2, Some(bias), bias_offset);
    }

    let n_out_per_tgp = if transpose_mat {
        bn * sn * tn
    } else {
        bm * sm * tm
    };
    let n_tgp = out_vec_size.div_ceil(n_out_per_tgp);
    let grid_size = MTLSize {
        width: n_tgp,
        height: 1,
        depth: b,
    };
    let group_size = MTLSize {
        width: 32,
        height: bn,
        depth: bm,
    };
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
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
    call_mlx_gemm_with_bias(
        device,
        ep,
        kernels,
        dtype,
        (b, m, n, k),
        lhs_stride,
        lhs_offset,
        lhs_buffer,
        rhs_stride,
        rhs_offset,
        rhs_buffer,
        None,
        GemmActivation::None,
        output,
    )
}

/// `call_mlx_gemm` with an optional fused epilogue: `bias` is an `n`-element
/// vector broadcast over rows and batch, applied through the steel kernel's
/// addmm path (`use_out_source`/`do_axpby` function constants with a zero row
/// stride) or the gemv `_axpby1` variants for M=1/N=1; `activation` then
/// applies a fused unary op (with or without a bias).
#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm_with_bias(
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
    bias: Option<(&Buffer, usize)>,
    activation: GemmActivation,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(MetalKernelError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        }
        .bt())?;
    };

    if m == 1 || n == 1 {
        return call_mlx_gemv(
            device,
            ep,
            kernels,
            dtype,
            (b, m, n, k),
            lhs_stride,
            lhs_offset,
            lhs_buffer,
            rhs_stride,
            rhs_offset,
            rhs_buffer,
            bias,
            activation,
            output,
        );
    }

    // Check for batch collapse optimization (MLX matmul.cpp lines 700-740)
    // When B is broadcasted (2D), collapse batch into M dimension
    let (effective_batch, effective_m, batch_collapsed) =
        check_batch_collapse(b, m, k, a_trans, lhs_stride, rhs_stride);

    // Use effective dimensions after potential batch collapse
    let m = effective_m;
    let b = effective_batch;

    // Dynamic tile selection based on matrix dimensions, dtype, transpose mode, and device type
    // Reference: MLX GEMM_TPARAM_MACRO in matmul.cpp
    let device_type = device.device_type();
    let tile = select_tile_config(dtype, m, n, k, b, a_trans, b_trans, device_type);
    let (bm, bn, bk, wm, wn) = (tile.bm, tile.bn, tile.bk, tile.wm, tile.wn);

    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    // has_batch should be true when b > 1, matching the original candle behavior
    let has_batch = b > 1;

    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(has_batch)),
        (100, Value::Bool(/* use_out_source */ bias.is_some())),
        (110, Value::Bool(/* do_axpby */ bias.is_some())),
        (120, Value::U16(/* gemm_activation */ activation.constant_value())),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile_swizzle = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile_swizzle;
    let tm = tm.div_ceil(tile_swizzle);

    // Calculate batch strides based on whether batch was collapsed
    let (batch_stride_a, batch_stride_b) = if batch_collapsed {
        // After batch collapse, there's no batch dimension
        (0isize, 0isize)
    } else {
        let a_stride = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as isize
        } else {
            (m * k) as isize
        };
        let b_stride = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as isize
        } else {
            (n * k) as isize
        };
        (a_stride, b_stride)
    };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if batch_collapsed { k as i32 } else { lda }, // After collapse, lda = K
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    // Dynamically generate kernel name based on dtype, transpose mode, and tile config
    // Format: gemm_{trans}_{itype}_{otype}_{bm}_{bn}_{bk}_{wm}_{wn}
    let dtype_str = match dtype {
        GemmDType::F32 => "f32",
        GemmDType::F16 => "f16",
        GemmDType::BF16 => "bf16",
    };
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );

    let pipeline =
        kernels.load_pipeline_with_constants(device, Source::Gemm, name.clone(), constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "mlx_gemm {name} B={b} M={m} N={n} K={k}");

    impl EncoderParam for GemmParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    // Mirrors the MSL GEMMAddMMParams layout (mlx_gemm.metal).
    #[derive(Debug)]
    #[repr(C)]
    struct GemmAddMMParams {
        ldc: i32,
        fdc: i32,
        batch_stride_c: usize,
        alpha: f32,
        beta: f32,
    }
    impl EncoderParam for GemmAddMMParams {
        fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
            encoder.set_bytes(position, &data);
        }
    }

    // Buffer 7 holds per-operand batch strides: A, then B, then — only when
    // use_out_source is set — C (batch_ndim entries each). The zero C stride
    // broadcasts the bias across the batch.
    let batch_strides = [batch_stride_a, batch_stride_b, 0isize];
    let batch_strides: &[isize] = if bias.is_some() {
        &batch_strides[..]
    } else {
        &batch_strides[..2]
    };

    set_params!(
        encoder,
        (
            (lhs_buffer, lhs_offset),
            (rhs_buffer, rhs_offset),
            (), // C, bound below when a bias is fused
            Output::new(output),
            gemm_params,
            (), // addmm_params, set below when a bias is fused
            b as i32,
            batch_strides
        )
    );
    if let Some((bias, bias_offset)) = bias {
        encoder.set_input_buffer(2, Some(bias), bias_offset);
        // A zero row stride broadcasts the n-element bias across rows and
        // batch; alpha/beta = 1 makes the epilogue a plain `+ bias`.
        crate::utils::set_param(
            encoder,
            5,
            GemmAddMMParams {
                ldc: 0,
                fdc: 1,
                batch_stride_c: 0,
                alpha: 1.0,
                beta: 1.0,
            },
        );
    }

    let grid_size = MTLSize {
        width: tn,
        height: tm,
        depth: /* batch_size_out */ b,
    };
    let group_size = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}
