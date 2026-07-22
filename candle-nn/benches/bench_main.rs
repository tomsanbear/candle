mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::norm::benches,
    benchmarks::linear::benches,
    benchmarks::softmax::benches,
    benchmarks::conv::benches,
    benchmarks::topk::benches
);
