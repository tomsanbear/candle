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
                    false,
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
                    false,
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

/// Barrier-semantics decision experiment: N independent GEMV chains, each a
/// strict same-buffer dependency chain, dispatched (a) sequentially chain
/// after chain, (b) interleaved A1 B1 A2 B2... In (b) the encoder's
/// auto-barrier fires on every same-chain hazard; with GLOBAL barriers those
/// drains also stall the other chain and (b) ≈ (a). If the interleave runs
/// materially faster than sequential, independent work flows past barriers
/// and scoped-barrier work in the backend is worth building.
fn run_barrier_probe(n: usize, k: usize, depth: usize, chains: usize) -> Result<()> {
    const WARMUP: usize = 3;
    const MIN_DUR: f64 = 1.0;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    let (weights, weight_bytes) = q_weight_bytes(GgmlDType::Q4K, n, k);
    let rhs = device
        .new_buffer_with_data(
            weights.as_ptr() as *const core::ffi::c_void,
            weights.len(),
            options,
        )
        .unwrap();
    // Each chain re-dispatches onto its own dst: the WAW hazard on dst makes
    // the chain strictly serial through the encoder's auto-barrier, while
    // different chains share nothing (rhs is read-only).
    let mut lhs = Vec::new();
    let mut dst = Vec::new();
    for _ in 0..chains {
        let acts: Vec<bf16> = (0..k)
            .map(|i| bf16::from_f32(((i % 89) as f32 - 44.0) / 977.0))
            .collect();
        lhs.push(
            device
                .new_buffer_with_data(
                    acts.as_ptr() as *const core::ffi::c_void,
                    std::mem::size_of_val(acts.as_slice()),
                    options,
                )
                .unwrap(),
        );
        dst.push(device.new_buffer(n * 4, options).unwrap());
    }

    let mut measure = |interleave: bool| -> Result<f64> {
        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let encoder = commands.command_encoder().unwrap();
            let start = std::time::Instant::now();
            let gemv = |c: usize| {
                candle_metal_kernels::call_quantized_matmul_mv_t(
                    &device,
                    &encoder,
                    &kernels,
                    GgmlDType::Q4K,
                    true,
                    false,
                    (1, 1, n, k),
                    &lhs[c],
                    0,
                    &rhs,
                    0,
                    &dst[c],
                )
            };
            if interleave {
                for _ in 0..depth {
                    for c in 0..chains {
                        gemv(c)?;
                    }
                }
            } else {
                for c in 0..chains {
                    for _ in 0..depth {
                        gemv(c)?;
                    }
                }
            }
            drop(encoder);
            commands.wait_until_completed().unwrap();
            let dt = start.elapsed().as_secs_f64();
            if idx < WARMUP {
                continue;
            }
            sum_dt += dt;
            iters += depth * chains;
            if sum_dt > MIN_DUR {
                break;
            }
        }
        Ok(1e3 * sum_dt / iters as f64)
    };

    let seq_ms = measure(false)?;
    let int_ms = measure(true)?;
    println!(
        "barrier-probe q4k n=k={n} depth={depth} chains={chains} ({} MB/chain-step): sequential {seq_ms:.4} ms/dispatch, interleaved {int_ms:.4} ms/dispatch, ratio {:.2} (1.0 = barriers serialize everything; ~1/{chains} = full overlap)",
        weight_bytes / 1_000_000,
        int_ms / seq_ms,
    );
    Ok(())
}

/// V1 (q4k-mv-rewrite-round2): simdgroups-per-threadgroup sweep on the q4_K
/// bf16/bf16 mv kernel. Per-row arithmetic is identical across nsg, so the
/// bf16 outputs must match BITWISE vs nsg=1; then the run_qmv timing protocol
/// per nsg. Targets the launch limiter: nsg=1 launches n/4 single-simdgroup
/// TGs (62k on the lm_head).
fn run_nsg_sweep(name: &str, n: usize, k: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 3;
    const MIN_DUR: f64 = 1.5;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;
    let m = 1usize;

    let (weights, weight_bytes) = q_weight_bytes(GgmlDType::Q4K, n, k);
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
    let dst_ref = device
        .new_buffer(m * n * core::mem::size_of::<bf16>(), options)
        .unwrap();
    let dst = device
        .new_buffer(m * n * core::mem::size_of::<bf16>(), options)
        .unwrap();

    // Correctness: each nsg vs nsg=1, bitwise on the bf16 outputs.
    for nsg in [2usize, 4, 8] {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_nsg(
            &device, &encoder, &kernels, 1, (1, m, n, k), &lhs, 0, &rhs, 0, &dst_ref,
        )?;
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_nsg(
            &device, &encoder, &kernels, nsg, (1, m, n, k), &lhs, 0, &rhs, 0, &dst,
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let a = unsafe { std::slice::from_raw_parts(dst_ref.contents() as *const u16, m * n) };
        let b = unsafe { std::slice::from_raw_parts(dst.contents() as *const u16, m * n) };
        let diffs = (0..m * n).filter(|&i| a[i] != b[i]).count();
        anyhow::ensure!(diffs == 0, "nsg={nsg} diverges from nsg=1 on {diffs} outputs");
    }
    println!("q4_K {name}: nsg 2/4/8 bitwise-identical to nsg=1 over {} outputs", m * n);

    // Correctness for the row-tile (V5 lifetime) variants: bitwise vs nsg=1.
    for nt in [2usize, 4, 8, 16, 32] {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_nsg(
            &device, &encoder, &kernels, 1, (1, m, n, k), &lhs, 0, &rhs, 0, &dst_ref,
        )?;
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_rowtile(
            &device, &encoder, &kernels, nt, (1, m, n, k), &lhs, 0, &rhs, 0, &dst,
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let a = unsafe { std::slice::from_raw_parts(dst_ref.contents() as *const u16, m * n) };
        let b = unsafe { std::slice::from_raw_parts(dst.contents() as *const u16, m * n) };
        let diffs = (0..m * n).filter(|&i| a[i] != b[i]).count();
        anyhow::ensure!(diffs == 0, "rt{nt} diverges from baseline on {diffs} outputs");
    }
    println!("q4_K {name}: rt 2/4/8/16/32 bitwise-identical to baseline over {} outputs", m * n);

    // Timing: row-tile variants (V5).
    let inner = (50_000_000 / weight_bytes).clamp(4, 512);
    for nt in [2usize, 4, 8, 16, 32] {
        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let encoder = commands.command_encoder().unwrap();
            let start_time = std::time::Instant::now();
            for _ in 0..inner {
                candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_rowtile(
                    &device, &encoder, &kernels, nt, (1, m, n, k), &lhs, 0, &rhs, 0, &dst,
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
        println!("q4_K {name:>10} n={n:6} k={k:5} rt={nt:2}   {ms:8.3} ms  {gbs:6.1} GB/s");
    }

    // Timing per nsg, run_qmv protocol.
    for nsg in [1usize, 2, 4, 8] {
        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let encoder = commands.command_encoder().unwrap();
            let start_time = std::time::Instant::now();
            for _ in 0..inner {
                candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_nsg(
                    &device, &encoder, &kernels, nsg, (1, m, n, k), &lhs, 0, &rhs, 0, &dst,
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
        println!("q4_K {name:>10} n={n:6} k={k:5} nsg={nsg}  {ms:8.3} ms  {gbs:6.1} GB/s");
    }

    // Round-3 geometry arms: llama.cpp's nr0=2 x nsg=2 (64-thread TGs), the
    // nr0=2 single-simdgroup control (separates NDST from TG size), and the
    // f32-activation arm of the baseline geometry (dtype discriminator; the
    // f32 y buffer carries the SAME values as the bf16 one, so outputs stay
    // bitwise comparable).
    let acts_f32: Vec<f32> = acts.iter().map(|v| v.to_f32()).collect();
    let lhs_f32 = device
        .new_buffer_with_data(
            acts_f32.as_ptr() as *const core::ffi::c_void,
            std::mem::size_of_val(acts_f32.as_slice()),
            options,
        )
        .unwrap();
    let geo_arms: [(&str, (usize, usize, bool)); 3] = [
        ("nr2sg2", (2, 2, false)),
        ("nr2sg1", (1, 2, false)),
        ("f32y  ", (1, 4, true)),
    ];
    for (label, geo) in geo_arms {
        let y = if geo.2 { &lhs_f32 } else { &lhs };
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_nsg(
            &device, &encoder, &kernels, 1, (1, m, n, k), &lhs, 0, &rhs, 0, &dst_ref,
        )?;
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_geo(
            &device, &encoder, &kernels, geo, (1, m, n, k), y, 0, &rhs, 0, &dst,
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let a = unsafe { std::slice::from_raw_parts(dst_ref.contents() as *const u16, m * n) };
        let b = unsafe { std::slice::from_raw_parts(dst.contents() as *const u16, m * n) };
        let diffs = (0..m * n).filter(|&i| a[i] != b[i]).count();
        anyhow::ensure!(diffs == 0, "{label} diverges from baseline on {diffs} outputs");

        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let encoder = commands.command_encoder().unwrap();
            let start_time = std::time::Instant::now();
            for _ in 0..inner {
                candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_geo(
                    &device, &encoder, &kernels, geo, (1, m, n, k), y, 0, &rhs, 0, &dst,
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
        println!("q4_K {name:>10} n={n:6} k={k:5} {label} {ms:8.3} ms  {gbs:6.1} GB/s  (bitwise OK)");
    }
    Ok(())
}

/// Splits AoS q4_K bytes into the SoA planes the `_soa` kernel reads:
/// [n*nb 16B headers | n*nb 128B quant blocks].
fn q4k_soa_bytes(aos: &[u8], n: usize, k: usize) -> Vec<u8> {
    let nb = k / 256;
    let blocks = n * nb;
    assert_eq!(aos.len(), blocks * 144);
    let mut soa = vec![0u8; blocks * 144];
    let (hdrs, quants) = soa.split_at_mut(blocks * 16);
    for b in 0..blocks {
        let o = b * 144;
        hdrs[b * 16..(b + 1) * 16].copy_from_slice(&aos[o..o + 16]);
        quants[b * 128..(b + 1) * 128].copy_from_slice(&aos[o + 16..o + 144]);
    }
    soa
}

/// q4_K mv AoS vs SoA plane-split: correctness cross-check (identical
/// arithmetic — results must match bitwise) then the same dispatch-level
/// timing as run_qmv on both layouts.
fn run_qmv_soa(name: &str, n: usize, k: usize) -> Result<()> {
    const WARMUP_ITERS: usize = 3;
    const MIN_DUR: f64 = 1.5;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;
    let m = 1usize;

    let (aos, weight_bytes) = q_weight_bytes(GgmlDType::Q4K, n, k);
    let soa = q4k_soa_bytes(&aos, n, k);
    let rhs_aos = device
        .new_buffer_with_data(aos.as_ptr() as *const core::ffi::c_void, aos.len(), options)
        .unwrap();
    let rhs_soa = device
        .new_buffer_with_data(soa.as_ptr() as *const core::ffi::c_void, soa.len(), options)
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
    let dst_a = device
        .new_buffer(m * n * core::mem::size_of::<f32>(), options)
        .unwrap();
    let dst_s = device
        .new_buffer(m * n * core::mem::size_of::<f32>(), options)
        .unwrap();

    // Correctness: one dispatch per layout, bitwise-compare dst.
    {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        candle_metal_kernels::call_quantized_matmul_mv_t(
            &device,
            &encoder,
            &kernels,
            GgmlDType::Q4K,
            true,
            false,
            (1, m, n, k),
            &lhs,
            0,
            &rhs_aos,
            0,
            &dst_a,
        )?;
        candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_soa(
            &device,
            &encoder,
            &kernels,
            (1, m, n, k),
            &lhs,
            0,
            &rhs_soa,
            0,
            &dst_s,
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();
        let a = unsafe { std::slice::from_raw_parts(dst_a.contents() as *const f32, m * n) };
        let s = unsafe { std::slice::from_raw_parts(dst_s.contents() as *const f32, m * n) };
        let mut max_abs = 0f32;
        for i in 0..m * n {
            max_abs = max_abs.max((a[i] - s[i]).abs());
        }
        anyhow::ensure!(
            max_abs == 0.0,
            "SoA kernel diverges from AoS: max |diff| = {max_abs}"
        );
        println!("q4_K {name}: SoA vs AoS bitwise-identical over {} outputs", m * n);
    }

    // Timing, both layouts, same protocol as run_qmv.
    let inner = (50_000_000 / weight_bytes).clamp(4, 512);
    for (layout, rhs, dst) in [("aos", &rhs_aos, &dst_a), ("soa", &rhs_soa, &dst_s)] {
        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let encoder = commands.command_encoder().unwrap();
            let start_time = std::time::Instant::now();
            for _ in 0..inner {
                if layout == "aos" {
                    candle_metal_kernels::call_quantized_matmul_mv_t(
                        &device,
                        &encoder,
                        &kernels,
                        GgmlDType::Q4K,
                        true,
                        false,
                        (1, m, n, k),
                        &lhs,
                        0,
                        rhs,
                        0,
                        dst,
                    )?;
                } else {
                    candle_metal_kernels::call_quantized_matmul_mv_q4k_bf16_soa(
                        &device,
                        &encoder,
                        &kernels,
                        (1, m, n, k),
                        &lhs,
                        0,
                        rhs,
                        0,
                        dst,
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
        println!("Q4K-{layout} {name:>10} n={n:6} k={k:5} m={m}  {ms:8.3} ms  {gbs:6.1} GB/s");
    }
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
                false,
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


/// Fused DSpark Markov chain: bitwise token verification against a CPU
/// reference that mirrors the kernel's arithmetic exactly (same per-row
/// sequential f32 accumulation, same bf16 rounding points, same
/// first-index-on-tie argmax), then chain latency per gamma. The legacy
/// serial-dispatch chain measured ~1.17 ms/step on M3; this prints the
/// fused replacement's cost.
fn run_markov_chain() -> Result<()> {
    use candle_metal_kernels::{call_markov_chain, MarkovChainArgs, MARKOV_NTG};
    use half::f16;

    const VOCAB_FULL: usize = 248094;
    const VD: usize = 32768;
    const R: usize = 256;
    const GAMMA: usize = 6;

    let device = Device::system_default().unwrap();
    let kernels = candle_metal_kernels::Kernels::new();
    let residency_set = std::sync::Arc::new(ResidencySet::new(&device));
    let options = RESOURCE_OPTIONS;

    // Deterministic synthetic tensors (no RNG: index-derived, non-degenerate).
    let w1: Vec<bf16> = (0..VOCAB_FULL * R)
        .map(|i| bf16::from_f32((((i * 37 + 11) % 197) as f32 - 98.0) / 391.0))
        .collect();
    let base: Vec<bf16> = (0..GAMMA * VD)
        .map(|i| bf16::from_f32((((i * 53 + 29) % 401) as f32 - 200.0) / 87.0))
        .collect();
    // q8_0 rows: 8 blocks of (f16 d | 32 x i8) per row.
    let blocks_per_row = R / 32;
    let mut w2_q8 = vec![0u8; VD * blocks_per_row * 34];
    for row in 0..VD {
        for b in 0..blocks_per_row {
            let off = (row * blocks_per_row + b) * 34;
            let d = f16::from_f32(0.011 + ((row * 7 + b) % 13) as f32 * 0.0035);
            w2_q8[off..off + 2].copy_from_slice(&d.to_le_bytes());
            for j in 0..32 {
                let q = ((row * 31 + b * 17 + j * 5 + 3) % 251) as i32 - 125;
                w2_q8[off + 2 + j] = (q as i8) as u8;
            }
        }
    }
    let w2_bf16: Vec<bf16> = (0..VD * R)
        .map(|i| bf16::from_f32((((i * 71 + 5) % 311) as f32 - 155.0) / 623.0))
        .collect();
    // Draft->global map: spread, in-range, deterministic.
    let ids: Vec<u32> = (0..VD).map(|i| ((i * 7 + 3) % VOCAB_FULL) as u32).collect();
    let anchor: u32 = 42_137;

    // CPU reference mirroring the kernel arithmetic exactly.
    let cpu_chain = |q8: bool, remap: bool| -> (Vec<u32>, Vec<u32>) {
        let mut prev = anchor as usize;
        let mut tokens = Vec::with_capacity(GAMMA);
        let mut chain_inputs = Vec::with_capacity(GAMMA);
        for k in 0..GAMMA {
            chain_inputs.push(prev as u32);
            let pe: Vec<f32> = (0..R).map(|j| f32::from(w1[prev * R + j])).collect();
            let mut best = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            for row in 0..VD {
                let acc = if q8 {
                    let mut acc = 0f32;
                    for b in 0..blocks_per_row {
                        let off = (row * blocks_per_row + b) * 34;
                        let d = f32::from(f16::from_le_bytes([w2_q8[off], w2_q8[off + 1]]));
                        let mut bsum = 0f32;
                        for j in 0..32 {
                            bsum += (w2_q8[off + 2 + j] as i8) as f32 * pe[b * 32 + j];
                        }
                        acc += d * bsum;
                    }
                    acc
                } else {
                    let mut acc = 0f32;
                    for j in 0..R {
                        acc += f32::from(w2_bf16[row * R + j]) * pe[j];
                    }
                    acc
                };
                let v = f32::from(bf16::from_f32(acc));
                let v = f32::from(bf16::from_f32(v + f32::from(base[k * VD + row])));
                if v > best {
                    best = v;
                    best_idx = row;
                }
            }
            let global = if remap { ids[best_idx] as usize } else { best_idx };
            tokens.push(global as u32);
            prev = global;
        }
        (tokens, chain_inputs)
    };

    let buf = |bytes: &[u8]| {
        device
            .new_buffer_with_data(bytes.as_ptr() as *const core::ffi::c_void, bytes.len(), options)
            .unwrap()
    };
    let as_bytes = |v: &[bf16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let w1_buf = buf(&as_bytes(&w1));
    let base_buf = buf(&as_bytes(&base));
    let w2q8_buf = buf(&w2_q8);
    let w2bf_buf = buf(&as_bytes(&w2_bf16));
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ids_buf = buf(&ids_bytes);
    let partials = device.new_buffer(MARKOV_NTG * 8, options).unwrap();
    let tokens_buf = device.new_buffer(GAMMA * 4, options).unwrap();
    let prev_embs_buf = device.new_buffer(GAMMA * R * 2, options).unwrap();
    let mut chain_init = vec![0u32; GAMMA + 1];
    chain_init[0] = anchor;
    let chain_bytes: Vec<u8> = chain_init.iter().flat_map(|x| x.to_le_bytes()).collect();
    let chain_buf = buf(&chain_bytes);

    for (label, q8, remap) in [
        ("q8+remap", true, true),
        ("q8", true, false),
        ("bf16+remap", false, true),
        ("bf16", false, false),
    ] {
        let command_queue = device.new_command_queue().unwrap();
        let commands = Commands::new(command_queue, &residency_set).unwrap();
        let encoder = commands.command_encoder().unwrap();
        call_markov_chain(
            &device,
            &encoder,
            &kernels,
            MarkovChainArgs {
                gamma: GAMMA,
                draft_vocab: VD,
                rank: R,
                w2_q8: q8,
                w1: (&w1_buf, 0),
                w2: (if q8 { &w2q8_buf } else { &w2bf_buf }, 0),
                base: (&base_buf, 0),
                chain: &chain_buf,
                partials: &partials,
                ids: (&ids_buf, 0),
                remap,
                tokens: &tokens_buf,
                prev_embs: &prev_embs_buf,
            },
        )?;
        drop(encoder);
        commands.wait_until_completed().unwrap();

        let got = unsafe { std::slice::from_raw_parts(tokens_buf.contents() as *const u32, GAMMA) }
            .to_vec();
        let (expected, chain_inputs) = cpu_chain(q8, remap);
        anyhow::ensure!(
            got == expected,
            "markov {label}: tokens {got:?} != cpu reference {expected:?}"
        );
        // prev_embs[k] must be w1[chain_input_k] bitwise.
        let embs = unsafe {
            std::slice::from_raw_parts(prev_embs_buf.contents() as *const u16, GAMMA * R)
        };
        for (k, &inp) in chain_inputs.iter().enumerate() {
            for j in 0..R {
                let want = w1[inp as usize * R + j].to_bits();
                anyhow::ensure!(
                    embs[k * R + j] == want,
                    "markov {label}: prev_embs[{k},{j}] mismatch"
                );
            }
        }
        println!(
            "markov-chain {label}: tokens + prev_embs match CPU reference bitwise ({GAMMA} steps)"
        );
    }

    // Timing: full fused chain (2*gamma dispatches) per iteration.
    const WARMUP_ITERS: usize = 3;
    const MIN_DUR: f64 = 1.0;
    for gamma in [1usize, 3, 6] {
        let mut sum_dt = 0f64;
        let mut iters = 0usize;
        for idx in 0.. {
            let command_queue = device.new_command_queue().unwrap();
            let commands = Commands::new(command_queue, &residency_set).unwrap();
            let inner = 64usize;
            let encoder = commands.command_encoder().unwrap();
            let start_time = std::time::Instant::now();
            for _ in 0..inner {
                call_markov_chain(
                    &device,
                    &encoder,
                    &kernels,
                    MarkovChainArgs {
                        gamma,
                        draft_vocab: VD,
                        rank: R,
                        w2_q8: true,
                        w1: (&w1_buf, 0),
                        w2: (&w2q8_buf, 0),
                        base: (&base_buf, 0),
                        chain: &chain_buf,
                        partials: &partials,
                        ids: (&ids_buf, 0),
                        remap: true,
                        tokens: &tokens_buf,
                        prev_embs: &prev_embs_buf,
                    },
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
        let us = 1e6 * sum_dt / iters as f64;
        println!(
            "markov-chain q8+remap gamma={gamma}: {us:8.1} us/chain ({:.1} us/step) — legacy serial chain was ~1170 us/step on M3",
            us / gamma as f64
        );
    }
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Gemm,
    Qmv,
    Qmm,
    /// GPU capture of the q4_K mv kernel (lm_head + mlp_wide shapes at m=1).
    QmvCapture,
    /// q4_K mv SoA plane-split vs AoS: correctness + GB/s (the repack gate).
    QmvSoa,
    /// Do independent chains overlap past global auto-barriers? (scoped-barrier gate)
    BarrierProbe,
    /// V1 (q4k-mv-rewrite-round2): simdgroups-per-TG sweep on q4_K bf16 mv —
    /// bitwise correctness vs nsg=1 + GB/s per nsg on the deployed shapes.
    NsgSweep,
    /// Fused DSpark Markov-chain kernels: BITWISE token verification against
    /// a same-accumulation-order CPU reference (q8_0 + bf16 w2 variants,
    /// with/without draft-vocab remap) and chain timing per gamma.
    MarkovChain,
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
        Task::QmvSoa => {
            for (name, n, k) in [
                ("lm_head", 248094usize, 1024usize),
                ("dn_qkvz", 12288, 1024),
                ("mlp_wide", 6144, 1024),
                ("attn_qkv", 3072, 1024),
                ("o_or_down", 1024, 3072),
            ] {
                run_qmv_soa(name, n, k)?;
            }
        }
        Task::BarrierProbe => {
            // Decode-representative small shapes: chains of skinny GEMVs.
            for (n, k) in [(3072usize, 1024usize), (1024, 3072), (6144, 1024)] {
                for chains in [2usize, 4] {
                    run_barrier_probe(n, k, 64, chains)?;
                }
            }
        }
        Task::MarkovChain => {
            run_markov_chain()?;
        }
        Task::NsgSweep => {
            // The deployed decode shapes: huge-n head (launch-limited),
            // mid-n body projections, and the small-n reductions where the
            // machine is occupancy-starved at m=1.
            for (name, n, k) in [
                ("lm_head", 248094usize, 1024usize),
                ("dn_qkvz", 8192, 1024),
                ("mlp_gate_up", 7168, 1024),
                ("out_proj", 1024, 2048),
                ("mlp_down", 1024, 3584),
            ] {
                run_nsg_sweep(name, n, k)?;
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
