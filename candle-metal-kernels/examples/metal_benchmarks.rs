use anyhow::Result;
use candle_metal_kernels::{
    metal::{Commands, Device, ResidencySet},
    GemmDType, GgmlDType, RESOURCE_OPTIONS,
};
/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
use clap::{Parser, Subcommand};
use half::{bf16, f16};

fn run_gemm(f32: bool, n: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 2;
    const MIN_DUR: f64 = 4.;

    let device = Device::system_default().unwrap();

    let (b, m, n, k) = (1, n, n, n);
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (lhs, rhs) = if f32 {
        let lhs: Vec<f32> = (0..b * m * k).map(|f| f as f32).collect();
        let rhs: Vec<f32> = (0..b * n * k).map(|f| f as f32).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    } else {
        let lhs: Vec<f16> = (0..b * m * k).map(|f| f16::from_f32(f as f32)).collect();
        let rhs: Vec<f16> = (0..b * n * k).map(|f| f16::from_f32(f as f32)).collect();
        let lhs = device
            .new_buffer_with_data(
                lhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&lhs),
                options,
            )
            .unwrap();
        let rhs = device
            .new_buffer_with_data(
                rhs.as_ptr() as *const core::ffi::c_void,
                std::mem::size_of_val(&rhs),
                options,
            )
            .unwrap();
        (lhs, rhs)
    };
    let (dtype, sizeof) = if f32 {
        (GemmDType::F32, core::mem::size_of::<f32>())
    } else {
        (GemmDType::F16, core::mem::size_of::<f16>())
    };
    let output = device.new_buffer(b * m * n * sizeof, options).unwrap();

    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        let start_time = std::time::Instant::now();
        candle_metal_kernels::call_mlx_gemm(
            &device,
            &encoder,
            &kernels,
            dtype,
            (b, m, n, k),
            &[m * k, k, 1],
            0,
            &lhs,
            &[n * k, n, 1],
            0,
            &rhs,
            &output,
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += 1;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let gflops = (2 * n * n * n * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?},      {n:6}      gflops {gflops:.0}");

    Ok(())
}

/// Deterministic, numerically-tame quantized weight bytes: unit scales, zero
/// mins, patterned quants. Timing-neutral vs real weights; never NaN/Inf.
fn q_weight_bytes(dtype: GgmlDType, n: usize, k: usize) -> (Vec<u8>, usize) {
    let (block_elems, block_bytes) = match dtype {
        GgmlDType::Q4K => (256usize, 144usize),
        GgmlDType::Q6K => (256, 210),
        GgmlDType::Q8_0 => (32, 34),
        _ => unimplemented!("bench covers the bf16-direct mv dtypes"),
    };
    assert_eq!(k % block_elems, 0);
    let blocks_per_row = k / block_elems;
    let total = n * blocks_per_row * block_bytes;
    let mut bytes = vec![0u8; total];
    let one_f16 = 0x3C00u16.to_le_bytes();
    for b in 0..n * blocks_per_row {
        let o = b * block_bytes;
        match dtype {
            GgmlDType::Q4K => {
                // half d = 1.0, half dmin = 0.0, scales[12], qs[128]
                bytes[o..o + 2].copy_from_slice(&one_f16);
                for i in 0..12 {
                    bytes[o + 4 + i] = 17 + (i as u8);
                }
                for i in 0..128 {
                    bytes[o + 16 + i] = ((b + i) % 251) as u8;
                }
            }
            GgmlDType::Q6K => {
                // ql[128], qh[64], scales[16] (i8), half d = 1.0
                for i in 0..192 {
                    bytes[o + i] = ((b + i) % 251) as u8;
                }
                for i in 0..16 {
                    bytes[o + 192 + i] = 3;
                }
                bytes[o + 208..o + 210].copy_from_slice(&one_f16);
            }
            GgmlDType::Q8_0 => {
                // half d = 1.0, i8 qs[32]
                bytes[o..o + 2].copy_from_slice(&one_f16);
                for i in 0..32 {
                    bytes[o + 2 + i] = ((b + i) % 251) as u8;
                }
            }
            _ => unreachable!(),
        }
    }
    (bytes, total)
}

/// Quantized matvec / multi-column benchmark on the bf16-activation kernels
/// (the deployed lmbrrr path). One dispatch per timed iteration, wall time
/// from encode to wait_until_completed; reports effective weight GB/s.
fn run_qmv(dtype: GgmlDType, name: &str, n: usize, k: usize, m: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 3;
    const MIN_DUR: f64 = 1.5;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (weights, weight_bytes) = q_weight_bytes(dtype, n, k);
    let rhs = device
        .new_buffer_with_data(
            weights.as_ptr() as *const core::ffi::c_void,
            weights.len(),
            options,
        )
        .unwrap();
    let acts: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i % 89) as f32 - 44.0) / 97.0))
        .collect();
    let lhs = device
        .new_buffer_with_data(
            acts.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(acts.as_slice()),
            options,
        )
        .unwrap();
    let dst = device
        .new_buffer(m * n * core::mem::size_of::<f32>(), options)
        .unwrap();

    // Many dispatches per command buffer: a single-dispatch commit is
    // dominated by the ~1-3 ms commit + wait_until_completed latency.
    // Hazard tracking on dst serializes the dispatches, which is the
    // sequential-execution timing we want.
    let inner = (50_000_000 / weight_bytes).clamp(4, 512);
    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        let start_time = std::time::Instant::now();
        for _ in 0..inner {
            if m == 1 {
                candle_metal_kernels::call_quantized_matmul_mv_t(
                    &device,
                    &encoder,
                    &kernels,
                    dtype,
                    true,
                    (1, m, n, k),
                    &lhs,
                    0,
                    &rhs,
                    0,
                    &dst,
                )?;
            } else {
                candle_metal_kernels::call_quantized_matmul_mv_mc(
                    &device,
                    &encoder,
                    &kernels,
                    dtype,
                    true,
                    (1, m, n, k),
                    &lhs,
                    0,
                    &rhs,
                    0,
                    &dst,
                )?;
            }
        }
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += inner;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let ms = 1e3 * sum_dt / iters as f64;
    let gbs = (weight_bytes * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?} {name:>10} n={n:6} k={k:5} m={m}  {ms:8.3} ms  {gbs:6.1} GB/s");
    Ok(())
}

/// Same measurement, but through the simdgroup mm kernel (f32 activations —
/// the mm template has no bf16-src1 variant). Answers whether the existing
/// 64x32-tile matrix kernel already beats the mc path at verify-chunk widths
/// by amortizing dequant across the full tile.
fn run_qmm(dtype: GgmlDType, name: &str, n: usize, k: usize, m: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 3;
    const MIN_DUR: f64 = 1.5;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (weights, weight_bytes) = q_weight_bytes(dtype, n, k);
    let row_bytes = weight_bytes / n;
    let rhs = device
        .new_buffer_with_data(
            weights.as_ptr() as *const core::ffi::c_void,
            weights.len(),
            options,
        )
        .unwrap();
    let acts: Vec<f32> = (0..m * k).map(|i| ((i % 89) as f32 - 44.0) / 97.0).collect();
    let lhs = device
        .new_buffer_with_data(
            acts.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(acts.as_slice()),
            options,
        )
        .unwrap();
    let dst = device
        .new_buffer(m * n * core::mem::size_of::<f32>(), options)
        .unwrap();

    let src0_shape = [1usize, 1, n, k];
    let src0_stride = [weight_bytes, weight_bytes, row_bytes, 0];
    let src1_shape = [1usize, 1, m, k];
    let src1_stride = [4 * k * m, 4 * k * m, 4 * k, 4];
    let dst_shape = [1usize, 1, m, n];

    let inner = (50_000_000 / weight_bytes).clamp(4, 512);
    let mut sum_dt = 0f64;
    let mut iters = 0usize;
    for idx in 0.. {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        let start_time = std::time::Instant::now();
        for _ in 0..inner {
            candle_metal_kernels::call_quantized_matmul_mm_t(
                &device,
                &encoder,
                &kernels,
                dtype,
                &src0_shape,
                &src0_stride,
                &rhs,
                &src1_shape,
                &src1_stride,
                &lhs,
                0,
                &dst_shape,
                0,
                &dst,
            )?;
        }
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let dt = start_time.elapsed().as_secs_f64();
        if idx < WARMUP_ITERS {
            continue;
        }
        sum_dt += dt;
        iters += inner;
        if sum_dt > MIN_DUR {
            break;
        }
    }
    let ms = 1e3 * sum_dt / iters as f64;
    let gbs = (weight_bytes * iters) as f64 / (1e9 * sum_dt);
    println!("{dtype:?} {name:>10} n={n:6} k={k:5} m={m} [mm]  {ms:8.3} ms  {gbs:6.1} GB/s");
    Ok(())
}

/// Wraps one bounded command buffer of qmv dispatches in an Xcode .gputrace
/// capture — the occupancy/limiter evidence the q4_K SoA-repack decision is
/// gated on. Requires METAL_CAPTURE_ENABLED=1 in the environment; open the
/// resulting bundle in Xcode (GPU trace) for per-encoder counters
/// (per-dispatch timestamps are unavailable on Apple Silicon).
fn run_qmv_capture(dtype: GgmlDType, name: &str, n: usize, k: usize, m: usize) -> Result<()> {
    use objc2_foundation::NSURL;
    use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

    if std::env::var("METAL_CAPTURE_ENABLED").is_err() {
        anyhow::bail!(
            "GPU capture needs METAL_CAPTURE_ENABLED=1 in the environment \
             (undocumented Metal requirement)"
        );
    }
    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (weights, weight_bytes) = q_weight_bytes(dtype, n, k);
    let rhs = device
        .new_buffer_with_data(
            weights.as_ptr() as *const core::ffi::c_void,
            weights.len(),
            options,
        )
        .unwrap();
    let acts: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i % 89) as f32 - 44.0) / 97.0))
        .collect();
    let lhs = device
        .new_buffer_with_data(
            acts.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(acts.as_slice()),
            options,
        )
        .unwrap();
    let dst = device
        .new_buffer(m * n * core::mem::size_of::<f32>(), options)
        .unwrap();

    let dispatch = |count: usize| -> Result<()> {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        for _ in 0..count {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                &device,
                &encoder,
                &kernels,
                dtype,
                true,
                (1, m, n, k),
                &lhs,
                0,
                &rhs,
                0,
                &dst,
            )?;
        }
        drop(encoder);
        commands.wait_until_completed().unwrap();
        Ok(())
    };

    // Occupancy evidence readable without Xcode: maxTotalThreadsPerThreadgroup
    // reflects the compiled kernel's register pressure (1024 = unconstrained;
    // lower = registers cap resident simdgroups per core).
    {
        use objc2_metal::MTLComputePipelineState;
        let pipeline = kernels.load_pipeline(
            &device,
            candle_metal_kernels::source::Source::Quantized,
            "kernel_mul_mv_q4_K_bf16",
        )?;
        let raw = pipeline.as_ref();
        println!(
            "kernel_mul_mv_q4_K_bf16: maxTotalThreadsPerThreadgroup={} threadExecutionWidth={} staticThreadgroupMemory={}B",
            pipeline.max_total_threads_per_threadgroup(),
            raw.threadExecutionWidth(),
            raw.staticThreadgroupMemoryLength(),
        );
    }

    // Warm up past the shader-compile transient so the capture shows
    // steady-state execution.
    for _ in 0..3 {
        dispatch(8)?;
    }

    let path = std::env::current_dir()?.join(format!("qmv-{name}-m{m}.gputrace"));
    if path.exists() {
        std::fs::remove_dir_all(&path)?;
    }
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    let descriptor = MTLCaptureDescriptor::new();
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    descriptor.set_capture_device(device.as_ref());
    let url = NSURL::from_file_path(&path);
    descriptor.setOutputURL(url.as_deref());
    manager
        .startCaptureWithDescriptor_error(&descriptor)
        .map_err(|e| anyhow::anyhow!("startCapture failed: {e}"))?;

    // One bounded buffer: enough dispatches to show steady occupancy,
    // small enough that Xcode can load the trace.
    dispatch(32)?;

    manager.stopCapture();
    println!(
        "{dtype:?} {name} n={n} k={k} m={m}: captured 32 dispatches ({} MB weights) -> {}",
        weight_bytes / 1_000_000,
        path.display()
    );
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Gemm,
    Qmv,
    Qmm,
    /// GPU capture of the q4_K mv kernel (lm_head + mlp_wide shapes at m=1).
    QmvCapture,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Gemm => {
            for f32 in [false, true] {
                for n in [512, 1024, 2048, 4096] {
                    run_gemm(f32, n)?;
                }
            }
        }
        Task::Qmv => {
            for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0, GgmlDType::Q6K] {
                for (name, n, k) in [("lm_head", 248094usize, 1024usize), ("mlp_wide", 6144, 1024)]
                {
                    for m in [1usize, 4] {
                        run_qmv(dtype, name, n, k, m)?;
                    }
                }
            }
        }
        Task::QmvCapture => {
            for (name, n, k) in [("lm_head", 248094usize, 1024usize), ("mlp_wide", 6144, 1024)] {
                run_qmv_capture(GgmlDType::Q4K, name, n, k, 1)?;
            }
        }
        Task::Qmm => {
            for dtype in [GgmlDType::Q4K, GgmlDType::Q8_0] {
                for (name, n) in [
                    ("lm_head", 248094usize),
                    ("dn_qkvz", 12288),
                    ("mlp_fused", 6144),
                    ("attn_qkv", 3072),
                    ("o_or_down", 1024),
                ] {
                    for m in [8usize, 12] {
                        run_qmv(dtype, name, n, 1024, m)?;
                        run_qmm(dtype, name, n, 1024, m)?;
                    }
                }
            }
        }
    }
    Ok(())
}
