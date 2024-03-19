mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::affine::benches,
    benchmarks::matmul::benches,
    benchmarks::random::benches,
    benchmarks::where_cond::benches,
    benchmarks::max_pool2d::benches,
    benchmarks::avg_pool2d::benches,
);
