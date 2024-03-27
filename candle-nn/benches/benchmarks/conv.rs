use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{DType, Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use criterion::{black_box, criterion_group, Criterion};
use std::time::Instant;

const B: usize = 1;
const C: usize = 1;
const M: usize = 50;
const K: usize = 4096;

fn run(input: Tensor, weight: Tensor, bias: Tensor, config: Conv2dConfig) {
    Conv2d::new(weight, Some(bias), config)
        .forward(&input)
        .unwrap();
}

fn run_conv2d_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let elements = B * C * M * K;

    let weight = Tensor::arange(0.0, elements as f32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let bias = weight.ones_like().unwrap();
    let weight = weight.reshape((B, C, M, K)).unwrap();
    let input = weight.ones_like().unwrap().reshape((B, C, M, K)).unwrap();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(
                    black_box(input.clone()),
                    black_box(weight.clone()),
                    black_box(bias.clone()),
                    Conv2dConfig {
                        ..Conv2dConfig::default()
                    },
                );
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let device = BenchDeviceHandler::new().unwrap();
    for d in device.devices {
        run_conv2d_benchmark(c, &d, DType::F32, "conv2d_f32");
        // no matmul support for bf16
        // run_conv2d_benchmark(c, &d, DType::BF16, "conv2d_bf16");
        run_conv2d_benchmark(c, &d, DType::F16, "conv2d_f16");
    }
}

criterion_group!(benches, criterion_benchmark);
