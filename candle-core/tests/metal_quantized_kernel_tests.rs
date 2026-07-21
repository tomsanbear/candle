#![cfg(feature = "metal")]
//! Kernel-level regression tests for the quantized Metal matmul dispatch:
//! tail threadgroups (row counts not divisible by the per-threadgroup
//! coverage) must not touch memory past the buffers. Destination buffers are
//! padded with sentinel-filled slack so out-of-bounds writes are observable
//! (they are invisible to public-API value checks), and in-bounds rows are
//! compared against a CPU dequantize + f32 matmul reference. Out-of-bounds
//! *reads* have no observable effect here (tail sums are discarded); run the
//! suite with `MTL_SHADER_VALIDATION=1` to exercise the read guards too.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use candle_metal_kernels::metal::{Buffer, Commands, Device as RawDevice, ResidencySet};
use candle_metal_kernels::{Kernels, MetalKernelError, RESOURCE_OPTIONS};
use std::sync::Arc;

/// Bit-exact canary; any kernel write replaces it with a computed float.
const SENTINEL: f32 = 1.0e30;

fn raw_device() -> RawDevice {
    RawDevice::system_default().unwrap()
}

fn commands(device: &RawDevice) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, &residency_set).unwrap()
}

fn new_buffer<T>(device: &RawDevice, data: &[T]) -> Buffer {
    let ptr = data.as_ptr() as *const core::ffi::c_void;
    let size = std::mem::size_of_val(data);
    device
        .new_buffer_with_data(ptr, size, RESOURCE_OPTIONS)
        .unwrap()
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

/// Deterministic pseudo-random values in [-5, 5) with varied per-block ranges.
fn test_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i.wrapping_mul(7919).wrapping_add(seed * 4297)) % 251) as f32 / 25.1 - 5.0)
        .collect()
}

/// Quantize an (n, k) weight on CPU, returning its raw bytes and the
/// dequantized reference output for an (m, k) input.
fn quantized_weight_and_reference(
    dtype: GgmlDType,
    n: usize,
    k: usize,
    xs: &[f32],
    m: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    let cpu = Device::Cpu;
    let weight = Tensor::from_vec(test_values(n * k, 1), (n, k), &cpu)?;
    let qweight = QTensor::quantize(&weight, dtype)?;
    let data = qweight.data()?.into_owned();
    let deq = qweight.dequantize(&cpu)?;
    let x = Tensor::from_vec(xs.to_vec(), (m, k), &cpu)?;
    let reference = x.matmul(&deq.t()?)?.flatten_all()?.to_vec1::<f32>()?;
    Ok((data, reference))
}

/// The tolerance floor absorbs accumulation noise on near-zero dot products;
/// tail bugs show up as sentinel or garbage values many orders of magnitude
/// out, so a loose bound still catches them.
fn assert_rows_close(
    got: &[f32],
    reference: &[f32],
    (rel_tol, abs_floor): (f32, f32),
    dtype: GgmlDType,
    ctx: &str,
) {
    assert_eq!(got.len(), reference.len(), "{ctx}: length mismatch");
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let tol = rel_tol * r.abs().max(abs_floor);
        assert!(
            (g - r).abs() <= tol,
            "{ctx} {dtype:?}: row {i} got {g} expected {r} (tol {tol})"
        );
    }
}

fn assert_canary_intact(slack: &[f32], dtype: GgmlDType, ctx: &str) {
    for (i, v) in slack.iter().enumerate() {
        assert!(
            v.to_bits() == SENTINEL.to_bits(),
            "{ctx} {dtype:?}: out-of-bounds write at slack index {i}: {v}"
        );
    }
}

/// The (before, rows, slack) regions of the destination buffer.
type DstRegions = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Dispatch the mv kernel for an (n, k) quantized weight and (m, k) input,
/// with `dst_offset_elems` leading floats and `n + 64` trailing floats of
/// sentinel slack. Returns (before, rows, slack) from the destination buffer.
#[allow(clippy::too_many_arguments)]
fn dispatch_mv(
    dtype: GgmlDType,
    weight_data: &[u8],
    xs: &[f32],
    (m, n, k): (usize, usize, usize),
    lhs_offset_elems: usize,
    dst_offset_elems: usize,
) -> std::result::Result<DstRegions, MetalKernelError> {
    let device = raw_device();
    let kernels = Kernels::new();
    let commands = commands(&device);

    let lhs = new_buffer(&device, xs);
    let rhs = new_buffer(&device, weight_data);
    // Slack past the m*n result area catches tail overwrites; sized to catch
    // even an over-launch writing up to n rows past the end.
    let dst_len = dst_offset_elems + m * n + n + 64;
    let dst = new_buffer(&device, &vec![SENTINEL; dst_len]);

    {
        let encoder = commands.command_encoder()?;
        candle_metal_kernels::call_quantized_matmul_mv_t(
            &device,
            &encoder,
            &kernels,
            dtype.into(),
            (1, m, n, k),
            &lhs,
            lhs_offset_elems * std::mem::size_of::<f32>(),
            &rhs,
            dst_offset_elems * std::mem::size_of::<f32>(),
            &dst,
        )?;
    }
    commands.wait_until_completed()?;

    let out = read_to_vec::<f32>(&dst, dst_len);
    let before = out[..dst_offset_elems].to_vec();
    let rows = out[dst_offset_elems..dst_offset_elems + m * n].to_vec();
    let slack = out[dst_offset_elems + m * n..].to_vec();
    Ok((before, rows, slack))
}

/// Row counts around every tail boundary for a kernel covering `tile` rows
/// per threadgroup, plus a large vocabulary-like odd count.
fn tail_matrix(tile: usize) -> Vec<usize> {
    let mut ns = vec![
        1,
        tile - 1,
        tile,
        tile + 1,
        2 * tile - 1,
        2 * tile,
        2 * tile + 1,
        4 * tile + 3,
        4099,
    ];
    ns.retain(|&n| n > 0);
    ns.dedup();
    ns
}

fn run_mv_tails(dtype: GgmlDType, tile: usize) -> Result<()> {
    let k = 512;
    for n in tail_matrix(tile) {
        for m in [1, 3] {
            let xs = test_values(m * k, 2);
            let (weight_data, reference) = quantized_weight_and_reference(dtype, n, k, &xs, m)?;
            let (_, rows, slack) = dispatch_mv(dtype, &weight_data, &xs, (m, n, k), 0, 0)
                .map_err(candle_core::Error::wrap)?;
            let ctx = format!("mv n={n} m={m}");
            assert_rows_close(&rows, &reference, (1e-3, 1.0), dtype, &ctx);
            assert_canary_intact(&slack, dtype, &ctx);
        }
    }
    Ok(())
}

macro_rules! mv_tail_test {
    ($name:ident, $dtype:expr, $tile:expr) => {
        #[test]
        fn $name() -> Result<()> {
            run_mv_tails($dtype, $tile)
        }
    };
}

mv_tail_test!(qmv_tails_q4_0, GgmlDType::Q4_0, 8);
mv_tail_test!(qmv_tails_q4_1, GgmlDType::Q4_1, 8);
mv_tail_test!(qmv_tails_q5_0, GgmlDType::Q5_0, 8);
mv_tail_test!(qmv_tails_q5_1, GgmlDType::Q5_1, 8);
mv_tail_test!(qmv_tails_q8_0, GgmlDType::Q8_0, 8);
mv_tail_test!(qmv_tails_q2k, GgmlDType::Q2K, 8);
mv_tail_test!(qmv_tails_q3k, GgmlDType::Q3K, 4);
mv_tail_test!(qmv_tails_q4k, GgmlDType::Q4K, 4);
mv_tail_test!(qmv_tails_q5k, GgmlDType::Q5K, 4);
mv_tail_test!(qmv_tails_q6k, GgmlDType::Q6K, 2);

/// F16/F32 weights go through the same mv dispatch; the dense kernel computes
/// one output row per threadgroup in grid x.
#[test]
fn qmv_tails_float_weights() -> Result<()> {
    let k = 512;
    for dtype in [GgmlDType::F32, GgmlDType::F16] {
        for n in [1, 7, 8, 9, 63, 64, 65, 4099] {
            for m in [1, 3] {
                let xs = test_values(m * k, 2);
                let (weight_data, reference) = quantized_weight_and_reference(dtype, n, k, &xs, m)?;
                let (_, rows, slack) = dispatch_mv(dtype, &weight_data, &xs, (m, n, k), 0, 0)
                    .map_err(candle_core::Error::wrap)?;
                let ctx = format!("mv n={n} m={m}");
                assert_rows_close(&rows, &reference, (1e-3, 1.0), dtype, &ctx);
                assert_canary_intact(&slack, dtype, &ctx);
            }
        }
    }
    Ok(())
}

/// Nonzero lhs/dst offsets, as used by the batched host loop in
/// `QMetalStorage::fwd_mv`. The region before `dst_offset` must stay intact.
#[test]
fn qmv_tails_with_offsets() -> Result<()> {
    let k = 512;
    for (dtype, tile) in [(GgmlDType::Q8_0, 8), (GgmlDType::Q4K, 4)] {
        let n = tile + 1;
        let xs = test_values(2 * k, 2);
        let (weight_data, reference) = quantized_weight_and_reference(dtype, n, k, &xs, 2)?;
        let (before, rows, slack) = dispatch_mv(dtype, &weight_data, &xs, (1, n, k), k, 16)
            .map_err(candle_core::Error::wrap)?;
        let ctx = "mv offsets";
        assert_canary_intact(&before, dtype, ctx);
        assert_rows_close(&rows, &reference[n..], (1e-3, 1.0), dtype, ctx);
        assert_canary_intact(&slack, dtype, ctx);
    }
    Ok(())
}

/// The mv dispatch table names kernels for Q8_1, Q8K and BF16 that do not
/// exist in the shader; it must report them as unsupported instead of failing
/// pipeline creation.
#[test]
fn qmv_unsupported_dtypes() {
    let device = raw_device();
    let kernels = Kernels::new();
    for dtype in [GgmlDType::Q8_1, GgmlDType::Q8K, GgmlDType::BF16] {
        let commands = commands(&device);
        let lhs = new_buffer(&device, &vec![0f32; 256]);
        let rhs = new_buffer(&device, &vec![0u8; 4096]);
        let dst = new_buffer(&device, &vec![SENTINEL; 64]);
        let res = {
            let encoder = commands.command_encoder().unwrap();
            candle_metal_kernels::call_quantized_matmul_mv_t(
                &device,
                &encoder,
                &kernels,
                dtype.into(),
                (1, 1, 8, 256),
                &lhs,
                0,
                &rhs,
                0,
                &dst,
            )
        };
        assert!(
            matches!(res, Err(MetalKernelError::UnsupportedDTypeForOp(_, _))),
            "{dtype:?}: expected UnsupportedDTypeForOp, got {res:?}"
        );
    }
}

/// Dispatch the mm kernel the same way `QMetalStorage::fwd` does and check
/// values plus destination slack. The mm kernels stage partial 64x32 tiles
/// through threadgroup memory to avoid writing outside the matrix.
fn dispatch_mm(
    dtype: GgmlDType,
    weight_data: &[u8],
    xs: &[f32],
    (m, n, k): (usize, usize, usize),
) -> std::result::Result<(Vec<f32>, Vec<f32>), MetalKernelError> {
    let device = raw_device();
    let kernels = Kernels::new();
    let commands = commands(&device);

    let src1 = new_buffer(&device, xs);
    let src0 = new_buffer(&device, weight_data);
    let dst_len = m * n + n + 64;
    let dst = new_buffer(&device, &vec![SENTINEL; dst_len]);

    let (core_dtype, kernel_dtype): (GgmlDType, candle_metal_kernels::GgmlDType) =
        (dtype, dtype.into());
    let bytes_per_elem = core_dtype.type_size() as f32 / core_dtype.block_size() as f32;
    let src0_shape = [1usize, 1, n, k];
    let src0_stride: Vec<usize> = [n * k, n * k, k, 1]
        .iter()
        .map(|x| (*x as f32 * bytes_per_elem) as usize)
        .collect();
    let src1_shape = [1usize, 1, m, k];
    let src1_stride: Vec<usize> = [m * k, m * k, k, 1]
        .iter()
        .map(|x| x * std::mem::size_of::<f32>())
        .collect();
    let dst_shape = [1usize, 1, m, n];

    {
        let encoder = commands.command_encoder()?;
        candle_metal_kernels::call_quantized_matmul_mm_t(
            &device,
            &encoder,
            &kernels,
            kernel_dtype,
            &src0_shape,
            &src0_stride,
            &src0,
            &src1_shape,
            &src1_stride,
            &src1,
            0,
            &dst_shape,
            0,
            &dst,
        )?;
    }
    commands.wait_until_completed()?;

    let out = read_to_vec::<f32>(&dst, dst_len);
    Ok((out[..m * n].to_vec(), out[m * n..].to_vec()))
}

#[test]
fn qmm_tails() -> Result<()> {
    let k = 512;
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        for n in [63, 64, 65, 129] {
            for m in [2, 31, 32, 33] {
                let xs = test_values(m * k, 2);
                let (weight_data, reference) = quantized_weight_and_reference(dtype, n, k, &xs, m)?;
                let (rows, slack) = dispatch_mm(dtype, &weight_data, &xs, (m, n, k))
                    .map_err(candle_core::Error::wrap)?;
                let ctx = format!("mm n={n} m={m}");
                // The mm kernels dequantize into half-precision tiles, so the
                // comparison against an f32-dequantized reference is looser.
                assert_rows_close(&rows, &reference, (2e-2, 10.0), dtype, &ctx);
                assert_canary_intact(&slack, dtype, &ctx);
            }
        }
    }
    Ok(())
}
