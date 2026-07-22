use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{LayerNorm, RmsNorm};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Op {
    RmsNorm,
    LayerNorm,
    LayerNormNoBias,
}

/// (rows, cols) regimes: a single decode row, the many-small-rows shape of a
/// transformer prefill at hidden size 512, and a large square.
const SHAPES: [(usize, usize); 3] = [(1, 512), (1024, 512), (1024, 1024)];

fn run(op: Op, input: &Tensor, weight: &Tensor, bias: &Tensor) {
    let _ = match op {
        Op::RmsNorm => RmsNorm::new(weight.clone(), 1e-5).forward(input),
        Op::LayerNorm => LayerNorm::new(weight.clone(), bias.clone(), 1e-5).forward(input),
        Op::LayerNormNoBias => LayerNorm::new_no_bias(weight.clone(), 1e-5).forward(input),
    };
}

fn run_norm_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    op: Op,
    name: &str,
    (rows, cols): (usize, usize),
) {
    let weight = Tensor::arange(0.0, cols as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = weight.ones_like().unwrap();
    let input = Tensor::ones((1, rows, cols), DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let bytes = rows * cols * dtype.size_in_bytes();
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(op, black_box(&input), black_box(&weight), black_box(&bias));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    let ops = [
        (Op::RmsNorm, "rms_norm"),
        (Op::LayerNorm, "layer_norm"),
        (Op::LayerNormNoBias, "layer_norm_no_bias"),
    ];
    let dtypes = [
        (DType::F32, "f32"),
        (DType::BF16, "bf16"),
        (DType::F16, "f16"),
    ];
    for d in device.devices {
        for &(op, op_name) in ops.iter() {
            for &(dtype, dtype_name) in dtypes.iter() {
                for &(rows, cols) in SHAPES.iter() {
                    let name = format!("{op_name}_r{rows}x{cols}_{dtype_name}");
                    run_norm_benchmark(c, &d, dtype, op, &name, (rows, cols));
                }
            }
        }
    }
}

criterion_group!(benches, criterion_benchmark);
