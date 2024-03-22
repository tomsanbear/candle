use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, Criterion, Throughput};
use std::time::Instant;

fn run(x: &Tensor, ids: &Tensor, dim: usize) {
    x.index_select(ids, dim).unwrap();
}

fn run_benchmark(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let t = Tensor::arange(0.0f32, 10000.0, device)
        .unwrap()
        .reshape((1, 1, 50, 200))
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let ids = Tensor::new(&[0u8, 100, 199], device).unwrap();

    let flops = t.dims().iter().product::<usize>() * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("index_select", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&t), black_box(&ids), 3);
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
        run_benchmark(c, &device, DType::F32, "ia_u8_f32");
        run_benchmark(c, &device, DType::F16, "ia_u8_f16");
        run_benchmark(c, &device, DType::BF16, "ia_u8_bf16");
    }
}

criterion_group!(benches, criterion_benchmark);
