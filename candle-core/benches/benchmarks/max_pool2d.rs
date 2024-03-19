use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(x: &Tensor) {
    x.max_pool2d(4).unwrap();
}

fn run_bench(c: &mut Criterion, device: &Device) {
    let b = 1;
    let m = 1;
    let n = 2048;
    let k = 2048;

    let dtype = DType::F32;
    let lhs = Tensor::arange(0.0f32, 2048.0 * 2048.0, &device)
        .unwrap()
        .reshape((b, m, n, k))
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let flops = b * m * n * k;

    let mut group = c.benchmark_group(device.bench_name("max_pool2d"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&lhs));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_bench(c, &device);
    }
}

criterion_group!(benches, criterion_benchmark);
