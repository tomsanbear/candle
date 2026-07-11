#include <metal_stdlib>
using namespace metal;

// Skinny GEMM for 2 <= M <= 12: C[M,N] = A[M,K] @ B[N,K]^T (nt layout,
// bf16 in/out, f32 accumulation).
//
// The mlx GEMM tile kernels read the big B matrix at roughly half bandwidth
// when M is far below the tile height — exactly the shape of speculative
// verify chunks and small decode batches. Here each simdgroup owns one
// output column (a row of B): lanes split K contiguously (coalesced B
// reads, B is streamed exactly once), and every lane applies its B element
// to all M activation rows from a threadgroup-cached A tile, holding M
// accumulators in registers.
//
// Grid: ceil(N / SG_PER_TG) threadgroups x (SG_PER_TG * 32) threads.

#define SK_MAX_M 12
#define SK_K_TILE 512
#define SK_SG_PER_TG 8

kernel void skinny_gemm_nt_bf16(
    device const bfloat *a [[buffer(0)]],   // [M, K] row-major
    device const bfloat *b [[buffer(1)]],   // [N, K] row-major
    device bfloat *c       [[buffer(2)]],   // [M, N] row-major
    constant uint &m_dim   [[buffer(3)]],
    constant uint &n_dim   [[buffer(4)]],
    constant uint &k_dim   [[buffer(5)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float a_sh[SK_MAX_M * SK_K_TILE];

    const uint n_i = tg_id * SK_SG_PER_TG + simd_group;
    const bool active = n_i < n_dim;

    float acc[SK_MAX_M];
    for (uint j = 0; j < SK_MAX_M; j++) {
        acc[j] = 0.0f;
    }

    const uint n_threads = SK_SG_PER_TG * 32;
    for (uint kt = 0; kt < k_dim; kt += SK_K_TILE) {
        const uint tile = min((uint)SK_K_TILE, k_dim - kt);
        // Cooperative A-tile load (all threads, barrier-synchronized).
        for (uint idx = tid; idx < m_dim * tile; idx += n_threads) {
            const uint row = idx / tile;
            const uint col = idx % tile;
            a_sh[row * SK_K_TILE + col] = float(a[row * k_dim + kt + col]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            device const bfloat *b_row = b + (ulong)n_i * k_dim + kt;
            for (uint kk = simd_lane; kk < tile; kk += 32) {
                const float bv = float(b_row[kk]);
                for (uint j = 0; j < m_dim; j++) {
                    acc[j] = fma(bv, a_sh[j * SK_K_TILE + kk], acc[j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        for (uint j = 0; j < m_dim; j++) {
            const float total = simd_sum(acc[j]);
            if (simd_lane == 0) {
                c[(ulong)j * n_dim + n_i] = bfloat(total);
            }
        }
    }
}
