#include <metal_stdlib>
using namespace metal;

template<typename T>
METAL_FUNC void welford_combine(
    thread T val,
    thread T &mean,
    thread T &m2,
    thread T &count
) {
    count += 1.0;
    T delta1 = val - mean;
    mean += delta1 / count;
    T delta2 = val - mean;
    m2 += delta1 * delta2;
}

template<typename T>
METAL_FUNC void block_welford_combine(
    thread T b_mean,
    thread T b_m2,
    thread T b_count,
    thread T &mean,
    thread T &m2,
    thread T &count
) {
    if (b_count == 0.0) {
        return;
    }
    T new_count = count + b_count; 
    T nb_over_n = b_count / new_count;
    T delta = b_mean - mean;
    mean += delta * nb_over_n;
    m2 += b_m2 + delta * delta * (count) * nb_over_n;
    count = new_count;
}

template<typename T>
METAL_FUNC void welford_warp_reduce(
    uint subgrp_size,
    thread T thread_mean,
    thread T thread_m2,
    thread T thread_count,
    thread T &mean,
    thread T &m2,
    thread T &count
) {
    mean = thread_mean;
    m2 = thread_m2;
    count = thread_count;
    for (uint offset = subgrp_size >> 1u; offset > 0u; offset >>= 1u) {
        T b_mean = simd_shuffle_down(mean, offset);
        T b_m2 = simd_shuffle_down(m2, offset);
        T b_count = simd_shuffle_down(count, offset);
        block_welford_combine(b_mean, b_m2, b_count, mean, m2, count);
    }
}

template<typename T>
METAL_FUNC void welford_warp_all_reduce(
    uint subgrp_size,
    thread T thread_mean,
    thread T thread_m2,
    thread T thread_count,
    thread T &mean,
    thread T &m2,
    thread T &count
) {
    welford_warp_reduce(subgrp_size, thread_mean, thread_m2, thread_count, mean, m2, count);

    mean = simd_broadcast(mean, 0u);
    m2 = simd_broadcast(m2, 0u);
    count = simd_broadcast(count, 0u);
}

kernel void welford_f32(
    device const float *X [[buffer(0)]],
    device const float *S [[buffer(1)]],
    device const float *B [[buffer(2)]],
    device float *Y [[buffer(3)]],
    device uint *shape [[buffer(4)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint local_size [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint subgrp_size = local_size;
    uint M = shape[1];
    uint N = shape[2];
    float mu = 0.0;
    float sigma = 0.0;

    uint anchor = global_id;
    thread float thread_var = 0.0;
    thread float thread_mean = 0.0;
    thread float thread_count = 0.0;

    // for (var i = local_id.x; i < metadata.N; i+= {{ workgroup_size_x }}u) {
    //     welford_combine(X[anchor + i], &threadMean, &threadVar, &threadCount);
    // }
    // TODO: probably a bug here
    for (uint i = local_id; i < N; i += local_size) {
        welford_combine<float>(X[anchor + i], thread_mean, thread_var, thread_count);
    }

    thread float mean = 0.0;
    thread float m2 = 0.0;
    thread float count = 0.0;

    welford_warp_all_reduce<float>(subgrp_size, thread_mean, thread_var, thread_count, mean, m2, count);
    if (simd_group_id == 0u) {
        mu = mean;
        sigma = rsqrt(m2 / count + 1e-5);
    }

    simdgroup_barrier(mem_flags::mem_none);

    // for (var i = local_id.x; i < metadata.N; i+= {{ workgroup_size_x }}u) {
    //     let val = X[anchor + i];
    //     let normalized = (val - mu) * sigma;
    //     Y[anchor + i] = fma(normalized, S[i], B[i]); 
    // }
    // TODO: probably a bug here
    for (uint i = local_id; i < N; i += local_size) {
        float val = X[anchor + i];
        float normalized = (val - mu) * sigma;
        Y[anchor + i] = fma(normalized, S[i], B[i]);
    }
}