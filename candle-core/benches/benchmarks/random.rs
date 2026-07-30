use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle_core::{DType, Device, Tensor};
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

fn rand_uniform(a: &Tensor) {
    a.rand_like(-1.0, 123.0).unwrap();
}

fn rand_normal(a: &Tensor) {
    a.randn_like(100.0, 15.0).unwrap();
}

fn run_random_bench(c: &mut Criterion, device: &Device) {
    let b = 1;

    let rows = 2048;
    let cols = 2048;

    let dtype = DType::F32;
    let tensor = Tensor::zeros((b, rows, cols), dtype, device).unwrap();

    let flops = b * rows * cols * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name("random_uniform"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_uniform(black_box(&tensor));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();

    let tensor = Tensor::zeros((b, rows, cols), dtype, device).unwrap();

    let mut group = c.benchmark_group(device.bench_name("random_normal"));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |benches| {
        benches.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                rand_normal(black_box(&tensor));
            }
            device.sync().unwrap();
            start.elapsed()
        })
    });
    group.finish();

    if device.is_cuda() {
        // Explicit seeded CUDA generation is deliberately unsupported until a
        // stateless kernel exists; do not benchmark a stateful emulation.
        return;
    }

    for (label, elements) in [
        ("flow_short", 50_240usize),
        ("decoder_representative", 998_720),
        ("decoder_max", 9_026_880),
    ] {
        for distribution in ["uniform", "normal"] {
            let mut group =
                c.benchmark_group(device.bench_name(format!("seeded_{distribution}_{label}")));
            group.throughput(Throughput::Bytes(
                (elements * DType::F32.size_in_bytes()) as u64,
            ));
            group.bench_with_input(
                BenchmarkId::new("stateful", elements),
                &elements,
                |benches, &elements| {
                    benches.iter_custom(|iters| {
                        let start = Instant::now();
                        for _ in 0..iters {
                            let tensor = if distribution == "uniform" {
                                Tensor::rand(0f32, 1f32, elements, device).unwrap()
                            } else {
                                Tensor::randn(0f32, 1f32, elements, device).unwrap()
                            };
                            black_box(tensor);
                        }
                        device.sync().unwrap();
                        start.elapsed()
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("explicit", elements),
                &elements,
                |benches, &elements| {
                    benches.iter_custom(|iters| {
                        let start = Instant::now();
                        for iteration in 0..iters {
                            let tensor = if distribution == "uniform" {
                                Tensor::rand_seeded(0f32, 1f32, elements, iteration, device)
                                    .unwrap()
                            } else {
                                Tensor::randn_seeded(0f32, 1f32, elements, iteration, device)
                                    .unwrap()
                            };
                            black_box(tensor);
                        }
                        device.sync().unwrap();
                        start.elapsed()
                    })
                },
            );
            if !device.is_cpu() {
                group.bench_with_input(
                    BenchmarkId::new("host_seeded_upload", elements),
                    &elements,
                    |benches, &elements| {
                        benches.iter_custom(|iters| {
                            let start = Instant::now();
                            for iteration in 0..iters {
                                let host = if distribution == "uniform" {
                                    Tensor::rand_seeded(
                                        0f32,
                                        1f32,
                                        elements,
                                        iteration,
                                        &Device::Cpu,
                                    )
                                    .unwrap()
                                } else {
                                    Tensor::randn_seeded(
                                        0f32,
                                        1f32,
                                        elements,
                                        iteration,
                                        &Device::Cpu,
                                    )
                                    .unwrap()
                                };
                                black_box(host.to_device(device).unwrap());
                            }
                            device.sync().unwrap();
                            start.elapsed()
                        })
                    },
                );
            }
            group.finish();
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let handler = BenchDeviceHandler::new().unwrap();
    for device in handler.devices {
        run_random_bench(c, &device);
    }
}

criterion_group!(benches, criterion_benchmark);
