use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

/// (c_in, l_in, c_out, k_size, stride, padding) — the three upsampling
/// stages of a HiFi-GAN-family vocoder, all with padding = (k - s) / 2.
const SHAPES: [(usize, usize, usize, usize, usize, usize); 3] = [
    (512, 210, 256, 16, 8, 4),
    (256, 1680, 128, 11, 5, 3),
    (128, 8400, 64, 7, 3, 2),
];

fn run(x: &Tensor, k: &Tensor, padding: usize, stride: usize) {
    x.conv_transpose1d(k, padding, 0, stride, 1, 1).unwrap();
}

#[allow(clippy::many_single_char_names)]
fn run_benchmark(
    c: &mut Criterion,
    device: &Device,
    dtype: DType,
    name: &str,
    shape: (usize, usize, usize, usize, usize, usize),
) {
    let (c_in, l_in, c_out, k, s, p) = shape;
    let t = Tensor::zeros((1, c_in, l_in), DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let kernel = Tensor::zeros((c_in, c_out, k), DType::F32, device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();

    let bytes = t.dims().iter().product::<usize>() * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run(black_box(&t), black_box(&kernel), p, s);
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
        for (i, &shape) in SHAPES.iter().enumerate() {
            run_benchmark(
                c,
                &device,
                DType::F32,
                &format!("conv_transpose1d_u{i}_f32"),
                shape,
            );
            run_benchmark(
                c,
                &device,
                DType::BF16,
                &format!("conv_transpose1d_u{i}_bf16"),
                shape,
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
