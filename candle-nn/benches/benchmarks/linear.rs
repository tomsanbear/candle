use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Tensor};
use candle_nn::ops::MatmulActivation;
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

/// (m, k, n) — a decode row against a wide projection, and a small prefill.
const SHAPES: [(usize, usize, usize); 2] = [(1, 1024, 4096), (355, 512, 512)];

#[derive(Copy, Clone)]
enum Mode {
    /// broadcast_matmul + broadcast_add
    Composed,
    /// matmul_bias (fused bias epilogue)
    Fused,
    /// matmul_bias + a separate silu dispatch
    SiluUnfused,
    /// matmul_bias_act (bias + silu in the epilogue)
    SiluFused,
}

fn run(mode: Mode, lhs: &Tensor, rhs: &Tensor, bias: &Tensor) {
    let _ = match mode {
        Mode::Composed => lhs
            .broadcast_matmul(rhs)
            .unwrap()
            .broadcast_add(bias)
            .unwrap(),
        Mode::Fused => candle_nn::ops::matmul_bias(lhs, rhs, bias).unwrap(),
        Mode::SiluUnfused => candle_nn::ops::matmul_bias(lhs, rhs, bias)
            .unwrap()
            .silu()
            .unwrap(),
        Mode::SiluFused => {
            candle_nn::ops::matmul_bias_act(lhs, rhs, bias, MatmulActivation::Silu).unwrap()
        }
    };
}

fn run_linear_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    name: &str,
    (m, k, n): (usize, usize, usize),
    mode: Mode,
) {
    let lhs = Tensor::ones((1, m, k), DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let rhs = Tensor::ones((k, n), DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = Tensor::ones(n, DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let bytes = (m * k + k * n + n) * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(mode, black_box(&lhs), black_box(&rhs), black_box(&bias));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    let modes = [
        (Mode::Composed, "composed"),
        (Mode::Fused, "fused"),
        (Mode::SiluUnfused, "silu_unfused"),
        (Mode::SiluFused, "silu_fused"),
    ];
    for d in device.devices {
        for &(m, k, n) in SHAPES.iter() {
            for &(dtype, dt) in [(DType::F32, "f32"), (DType::BF16, "bf16")].iter() {
                for &(mode, mode_name) in modes.iter() {
                    run_linear_benchmark(
                        c,
                        &d,
                        dtype,
                        &format!("linear_{mode_name}_m{m}k{k}n{n}_{dt}"),
                        (m, k, n),
                        mode,
                    );
                }
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
