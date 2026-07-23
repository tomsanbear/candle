use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

/// Deformable-DETR/RT-DETR-style multiscale deformable attention shape.
const N: usize = 1;
const HEADS: usize = 8;
const HEAD_DIM: usize = 32;
const POINTS: usize = 4;
const SHAPES: [(usize, usize); 4] = [(64, 64), (32, 32), (16, 16), (8, 8)];

struct Inputs {
    value: Tensor,
    loc: Tensor,
    attn: Tensor,
}

fn inputs(device: &Device) -> Inputs {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(299792458);
    let len_v: usize = SHAPES.iter().map(|&(h, w)| h * w).sum();
    let len_q = len_v;
    let levels = SHAPES.len();
    let mut rand = |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random()).collect() };
    Inputs {
        value: Tensor::from_vec(
            rand(N * len_v * HEADS * HEAD_DIM),
            (N, len_v, HEADS, HEAD_DIM),
            device,
        )
        .unwrap(),
        loc: Tensor::from_vec(
            rand(N * len_q * HEADS * levels * POINTS * 2),
            (N, len_q, HEADS, levels, POINTS, 2),
            device,
        )
        .unwrap(),
        attn: Tensor::from_vec(
            rand(N * len_q * HEADS * levels * POINTS),
            (N, len_q, HEADS, levels, POINTS),
            device,
        )
        .unwrap(),
    }
}

fn run_device(i: &Inputs) {
    let _ = candle_nn::ops::ms_deform_attn(
        black_box(&i.value),
        &SHAPES,
        black_box(&i.loc),
        black_box(&i.attn),
    )
    .unwrap();
}

/// The host round-trip this replaces: pull the operands to the CPU, sample
/// there, push the result back to the device.
fn run_host(i: &Inputs, device: &Device) {
    let cpu = Device::Cpu;
    let value = i.value.to_device(&cpu).unwrap();
    let loc = i.loc.to_device(&cpu).unwrap();
    let attn = i.attn.to_device(&cpu).unwrap();
    let out = candle_nn::ops::ms_deform_attn(&value, &SHAPES, &loc, &attn).unwrap();
    let _ = black_box(out.to_device(device).unwrap());
}

fn run_msda_benchmark(c: &mut Criterion, device: &Device, name: &str, host: bool) {
    let i = inputs(device);
    let len_v: usize = SHAPES.iter().map(|&(h, w)| h * w).sum();
    let bytes = N * len_v * HEADS * HEAD_DIM * 4;
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    let device = device.clone();
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                if host {
                    run_host(&i, &device);
                } else {
                    run_device(&i);
                }
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
        run_msda_benchmark(c, &d, "msda_device", false);
        run_msda_benchmark(c, &d, "msda_host_roundtrip", true);
    }
}

criterion_group!(benches, criterion_benchmark);
