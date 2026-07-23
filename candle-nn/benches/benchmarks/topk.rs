use crate::benchmarks::{BenchDevice, BenchDeviceHandler};
use candle::{Device, Tensor};
use criterion::{criterion_group, Criterion, Throughput};
use std::hint::black_box;
use std::time::Instant;

/// RT-DETR-like selection: top 300 of num_queries x classes scores.
const ROWS: usize = 8;
const NCOLS: usize = 24000;
const K: usize = 300;

fn run_device(xs: &Tensor) {
    let _ = candle_nn::ops::topk(black_box(xs), K).unwrap();
}

/// The host round-trip this replaces: pull the scores to the CPU, sort
/// there, keep the top k — what a detection pipeline has to do when the
/// device has no top-k op.
fn run_host(xs: &Tensor) {
    let rows = xs.to_vec2::<f32>().unwrap();
    let mut out = Vec::with_capacity(ROWS * K);
    for row in rows {
        let mut perm: Vec<u32> = (0..row.len() as u32).collect();
        perm.sort_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
        out.extend_from_slice(&perm[..K]);
    }
    black_box(out);
}

fn run_topk_benchmark(c: &mut Criterion, device: &Device, name: &str, host: bool) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(299792458);
    let data: Vec<f32> = (0..ROWS * NCOLS).map(|_| rng.random()).collect();
    let xs = Tensor::from_vec(data, (ROWS, NCOLS), device).unwrap();

    let bytes = ROWS * NCOLS * 4;
    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("iter", move |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                if host {
                    run_host(&xs);
                } else {
                    run_device(&xs);
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
        run_topk_benchmark(c, &d, &format!("topk_device_r{ROWS}x{NCOLS}k{K}"), false);
        run_topk_benchmark(c, &d, &format!("topk_host_r{ROWS}x{NCOLS}k{K}"), true);
    }
}

criterion_group!(benches, criterion_benchmark);
