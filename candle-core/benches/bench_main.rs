mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::affine::benches,
    benchmarks::matmul::benches,
    benchmarks::random::benches,
    benchmarks::where_cond::benches,
    benchmarks::conv_transpose2d::benches,
    benchmarks::index_select::benches,
);
