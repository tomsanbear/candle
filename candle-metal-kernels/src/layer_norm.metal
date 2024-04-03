#include <metal_stdlib>
using namespace metal;

METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template<typename T>
METAL_FUNC void welford_combine(
    thread T val,
    thread T &mean,
    thread T &m2,
    thread T &count
) {
    count += 1;
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
    if (b_count == 0) {
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

kernel void welford_scalar_f32(
    device const float *X [[buffer(0)]],
    device const float *S [[buffer(1)]],
    device const float *B [[buffer(2)]],
    device float *Y [[buffer(3)]],
    constant size_t &num_dims [[buffer(4)]],
    constant const size_t *shape [[buffer(5)]],
    constant const size_t *strides [[buffer(6)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint local_size [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    uint anchor = get_strided_index(global_id, num_dims, shape, strides);

    // Shape: [B, M, N]
    uint M = shape[1];
    uint N = shape[2];

    threadgroup float mu = 0.0;
    threadgroup float sigma = 0.0;

    thread float thread_var = 0.0;
    thread float thread_mean = 0.0;
    thread float thread_count = 0.0;

    welford_combine<float>(X[anchor], thread_mean, thread_var, thread_count);
    
    thread float mean = 0.0;
    thread float m2 = 0.0;
    thread float count = 0.0;

    welford_warp_all_reduce<float>(local_size, thread_mean, thread_var, thread_count, mean, m2, count);
    if (simd_lane_id == 0u) {
        mu = mean;
        sigma = rsqrt(m2 / count + 1e-5);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint dst_idx = global_id;
    float val = X[dst_idx];
    float normalized = (val - mu) * sigma;
    Y[dst_idx] = fma(normalized, S[dst_idx], B[dst_idx]);
}
