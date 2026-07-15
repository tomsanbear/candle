// Tensor-op (matmul2d) path for q4_K-quantized linears at m in [1,8],
// exact q4_K semantics via per-32-sub-block scale folding on the C tile.
//
// This source REQUIRES the MetalPerformancePrimitives framework header and
// therefore CANNOT be compiled by the runtime Metal compiler: it is built
// offline (scripts/build_mm2d_q4k.sh, -std=metal4.0, needs a macOS 26.4+
// SDK) into mm2d_q4k.metallib, which is checked in beside this file and
// loaded via newLibraryWithData. Pipeline creation fails on pre-26.4 OSes
// and callers fall back by routing.
//
// Weight layout (produced by the lmbrrr pack sidecar, NOT ggml q4_K):
//   nibbles: [K, Npad] little-endian 4-bit, n innermost, Npad = ceil(N/64)*64,
//            padding nibbles zero.
//   dsc:     [Npad, K/32] fp16, d * sc_j   (per-row, per-32-value sub-block)
//   dmm:     [Npad, K/32] fp16, dmin * m_j
// Effective weight w[k, n] = dsc[n, k/32] * q[k, n] - dmm[n, k/32]; padding
// rows have dsc = dmm = 0 so their (unwritten) outputs are inert. The dmin
// term's per-32-slice A row sums are computed in-kernel (threadgroup
// memory), so a linear is exactly one dispatch.
//
// The matmul2d M tile is hardware-fixed at 8 (framework static_assert):
// m_real < 8 rides the same tile; A's tensor extents are (K, m_real) and the
// framework edge-checks reads against them, C writes are guarded by m_real.
// dims = (K, Npad, N_real, m_real).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

// Row-sum capacity: NJ = K/32 entries per activation row, 8 rows max.
// K caps at 8192 here (larger K would blow the 8KB threadgroup budget);
// every model shape is far below.
#define MM2D_MAX_NJ 256

// Two tile geometries (probe12 receipts): 64-wide/4-simdgroup wins the
// mid-N shapes (qkv, o_proj); 32-wide/1-simdgroup halves per-TG work and
// doubles TG count, winning wide-N (mlp up/gate, lm_head) where latency is
// occupancy-bound. Host routing picks per shape.
#define MM2D_Q4K_KERNEL(NAME, TILE_N, SCOPE, TG_THREADS)                      \
kernel void NAME(                                                             \
    device const bfloat * a_p   [[ buffer(0) ]],                              \
    device const uchar  * b_p   [[ buffer(1) ]],                              \
    device const half   * dsc_p [[ buffer(2) ]],                              \
    device const half   * dmm_p [[ buffer(3) ]],                              \
    device bfloat       * c_p   [[ buffer(4) ]],                              \
    constant int4       & dims  [[ buffer(5) ]],                              \
    uint2 tgid [[threadgroup_position_in_grid]],                              \
    uint  tidx [[thread_index_in_threadgroup]]) {                             \
  const int K = dims.x;                                                       \
  const int Npad = dims.y;                                                    \
  const int Nreal = dims.z;                                                   \
  const int Mreal = dims.w;                                                   \
  const int NJ = K / 32;                                                      \
  const int n0 = int(tgid.x) * TILE_N;                                        \
                                                                              \
  tensor<device bfloat, dextents<int, 2>, tensor_inline>                      \
      a((device bfloat *)a_p, dextents<int, 2>(K, Mreal));                    \
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline>               \
      b((device uchar *)b_p, dextents<int, 2>(Npad, K));                      \
                                                                              \
  constexpr auto d = tensor_ops::matmul2d_descriptor(8, TILE_N, 32);          \
  tensor_ops::matmul2d<d, SCOPE> op;                                          \
                                                                              \
  /* Per-32-slice row sums of A for the dmin term, in threadgroup memory:   */\
  /* the A tile is SLC-resident (the matmul reads it anyway), and a second  */\
  /* dispatch's hazard barrier serialized the concurrent encoder.           */\
  threadgroup float rs_tg[8 * MM2D_MAX_NJ];                                   \
  const int entries = Mreal * NJ;                                             \
  for (int e = int(tidx); e < entries; e += TG_THREADS) {                     \
    const int m = e / NJ;                                                     \
    const int j = e % NJ;                                                     \
    device const bfloat * row = a_p + m * K + 32 * j;                         \
    float s = 0.0f;                                                           \
    for (int i = 0; i < 32; ++i) {                                            \
      s += float(row[i]);                                                     \
    }                                                                         \
    rs_tg[e] = s;                                                             \
  }                                                                           \
  threadgroup_barrier(mem_flags::mem_threadgroup);                            \
                                                                              \
  auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
  for (ushort i = 0; i < acc.get_capacity(); ++i) {                           \
    acc[i] = 0.0f;                                                            \
  }                                                                           \
                                                                              \
  for (int j = 0; j < NJ; ++j) {                                              \
    auto mA = a.slice(32 * j, 0);                                             \
    auto mB = b.slice(n0, 32 * j);                                            \
    auto p = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(mA, mB, p);                                                        \
    for (ushort i = 0; i < p.get_capacity(); ++i) {                           \
      auto mdi = p.get_multidimensional_index(i);                             \
      const int n = n0 + mdi[0];                                              \
      const int m = mdi[1];                                                   \
      const float w = float(dsc_p[n * NJ + j]);                               \
      const float mn = float(dmm_p[n * NJ + j]);                              \
      /* Tile rows beyond Mreal are dead (never stored) but must not read  */ \
      /* out of bounds.                                                    */ \
      const float r = m < Mreal ? rs_tg[m * NJ + j] : 0.0f;                   \
      acc[i] = fma(w, p[i], fma(-mn, r, acc[i]));                             \
    }                                                                         \
  }                                                                           \
                                                                              \
  /* Direct bf16 stores: real-N/real-m guards double as tail handling. */    \
  for (ushort i = 0; i < acc.get_capacity(); ++i) {                           \
    auto mdi = acc.get_multidimensional_index(i);                             \
    const int n = n0 + mdi[0];                                                \
    const int m = mdi[1];                                                     \
    if (n < Nreal && m < Mreal) {                                             \
      c_p[m * Nreal + n] = bfloat(acc[i]);                                    \
    }                                                                         \
  }                                                                           \
}

MM2D_Q4K_KERNEL(kernel_mul_mm2d_q4k_bf16, 64, execution_simdgroups<4>, 128)
MM2D_Q4K_KERNEL(kernel_mul_mm2d_q4k_bf16_t32, 32, execution_simdgroup, 32)

// Split-K pair for small-N shapes: N/64 threadgroups under-occupy the GPU
// when each runs the whole K loop serially (mlp_down measured 37.8 GB/s at
// 16 TGs), so grid dim y partitions the K/32 slices across nsplit groups
// writing f32 partials, and a tiny second dispatch reduces them. The fold
// (dsc*P - dmm*rowsum) is linear over slices, so partial sums are exact.
// Partial layout: part[(ks*8 + m) * Npad + n]; rows m >= Mreal are never
// written (nor read by the reduce).
kernel void kernel_mul_mm2d_q4k_bf16_splitk(
    device const bfloat * a_p    [[ buffer(0) ]],
    device const uchar  * b_p    [[ buffer(1) ]],
    device const half   * dsc_p  [[ buffer(2) ]],
    device const half   * dmm_p  [[ buffer(3) ]],
    device float        * part_p [[ buffer(4) ]],
    constant int4       & dims   [[ buffer(5) ]],
    constant int        & nsplit [[ buffer(6) ]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tidx [[thread_index_in_threadgroup]]) {
  const int K = dims.x;
  const int Npad = dims.y;
  const int Mreal = dims.w;
  const int NJ = K / 32;
  const int n0 = int(tgid.x) * 64;
  const int ks = int(tgid.y);
  const int j0 = (NJ * ks) / nsplit;
  const int j1 = (NJ * (ks + 1)) / nsplit;
  const int span = j1 - j0;

  tensor<device bfloat, dextents<int, 2>, tensor_inline>
      a((device bfloat *)a_p, dextents<int, 2>(K, Mreal));
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
      b((device uchar *)b_p, dextents<int, 2>(Npad, K));

  constexpr auto d = tensor_ops::matmul2d_descriptor(8, 64, 32);
  tensor_ops::matmul2d<d, execution_simdgroups<4>> op;

  threadgroup float rs_tg[8 * MM2D_MAX_NJ];
  const int entries = Mreal * span;
  for (int e = int(tidx); e < entries; e += 128) {
    const int m = e / span;
    const int j = j0 + e % span;
    device const bfloat * row = a_p + m * K + 32 * j;
    float s = 0.0f;
    for (int i = 0; i < 32; ++i) {
      s += float(row[i]);
    }
    rs_tg[e] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  for (int j = j0; j < j1; ++j) {
    auto mA = a.slice(32 * j, 0);
    auto mB = b.slice(n0, 32 * j);
    auto p = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(mA, mB, p);
    for (ushort i = 0; i < p.get_capacity(); ++i) {
      auto mdi = p.get_multidimensional_index(i);
      const int n = n0 + mdi[0];
      const int m = mdi[1];
      const float w = float(dsc_p[n * NJ + j]);
      const float mn = float(dmm_p[n * NJ + j]);
      const float r = m < Mreal ? rs_tg[m * span + (j - j0)] : 0.0f;
      acc[i] = fma(w, p[i], fma(-mn, r, acc[i]));
    }
  }

  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    auto mdi = acc.get_multidimensional_index(i);
    const int n = n0 + mdi[0];
    const int m = mdi[1];
    if (m < Mreal) {
      part_p[(ks * 8 + m) * Npad + n] = acc[i];
    }
  }
}

kernel void kernel_mm2d_q4k_splitk_reduce(
    device const float * part_p [[ buffer(0) ]],
    device bfloat      * c_p    [[ buffer(1) ]],
    constant int4      & dims   [[ buffer(2) ]],
    constant int       & nsplit [[ buffer(3) ]],
    uint2 tid [[thread_position_in_grid]]) {
  const int Npad = dims.y;
  const int Nreal = dims.z;
  const int n = int(tid.x);
  const int m = int(tid.y);
  if (n >= Nreal) {
    return;
  }
  float s = 0.0f;
  for (int ks = 0; ks < nsplit; ++ks) {
    s += part_p[(ks * 8 + m) * Npad + n];
  }
  c_p[m * Nreal + n] = bfloat(s);
}

// ---------------------------------------------------------------------------
// Fused m-row head argmax: the verify chunk only needs argmax(logits) per
// row, so the 64-wide head kernel folds as usual, bf16-rounds each logit
// (matching what candle's fast_argmax would see on the stored tensor —
// byte-identity depends on it), keeps the per-TG best per row, and a tiny
// reduce takes the global winner with the LOWEST-INDEX tie rule. The m x V
// logits tensor is never materialized.
// Partials layout: val[(tile*8 + m)], idx[(tile*8 + m)].
// ---------------------------------------------------------------------------
kernel void kernel_mul_mm2d_q4k_argmax(
    device const bfloat * a_p    [[ buffer(0) ]],
    device const uchar  * b_p    [[ buffer(1) ]],
    device const half   * dsc_p  [[ buffer(2) ]],
    device const half   * dmm_p  [[ buffer(3) ]],
    device float        * pval_p [[ buffer(4) ]],
    device uint         * pidx_p [[ buffer(5) ]],
    constant int4       & dims   [[ buffer(6) ]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tidx [[thread_index_in_threadgroup]]) {
  const int K = dims.x;
  const int Npad = dims.y;
  const int Nreal = dims.z;
  const int Mreal = dims.w;
  const int NJ = K / 32;
  const int n0 = int(tgid.x) * 64;

  tensor<device bfloat, dextents<int, 2>, tensor_inline>
      a((device bfloat *)a_p, dextents<int, 2>(K, Mreal));
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline>
      b((device uchar *)b_p, dextents<int, 2>(Npad, K));

  constexpr auto d = tensor_ops::matmul2d_descriptor(8, 64, 32);
  tensor_ops::matmul2d<d, execution_simdgroups<4>> op;

  threadgroup float rs_tg[8 * MM2D_MAX_NJ];
  const int entries = Mreal * NJ;
  for (int e = int(tidx); e < entries; e += 128) {
    const int m = e / NJ;
    const int j = e % NJ;
    device const bfloat * row = a_p + m * K + 32 * j;
    float s = 0.0f;
    for (int i = 0; i < 32; ++i) {
      s += float(row[i]);
    }
    rs_tg[e] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }
  for (int j = 0; j < NJ; ++j) {
    auto mA = a.slice(32 * j, 0);
    auto mB = b.slice(n0, 32 * j);
    auto p = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(mA, mB, p);
    for (ushort i = 0; i < p.get_capacity(); ++i) {
      auto mdi = p.get_multidimensional_index(i);
      const int n = n0 + mdi[0];
      const int m = mdi[1];
      const float w = float(dsc_p[n * NJ + j]);
      const float mn = float(dmm_p[n * NJ + j]);
      const float r = m < Mreal ? rs_tg[m * NJ + j] : 0.0f;
      acc[i] = fma(w, p[i], fma(-mn, r, acc[i]));
    }
  }

  // Per-thread best per row over this thread's coop elements (bf16-rounded
  // values, lowest index wins ties), then a threadgroup reduce.
  threadgroup float tv[8 * 128];
  threadgroup uint tix[8 * 128];
  for (int m = 0; m < 8; ++m) {
    tv[m * 128 + tidx] = -INFINITY;
    tix[m * 128 + tidx] = 0xffffffffu;
  }
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    auto mdi = acc.get_multidimensional_index(i);
    const int n = n0 + mdi[0];
    const int m = mdi[1];
    if (n >= Nreal || m >= Mreal) {
      continue;
    }
    const float v = float(bfloat(acc[i]));
    const int slot = m * 128 + int(tidx);
    if (v > tv[slot] || (v == tv[slot] && uint(n) < tix[slot])) {
      tv[slot] = v;
      tix[slot] = uint(n);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tidx < 8) {
    const int m = int(tidx);
    float best = -INFINITY;
    uint best_i = 0xffffffffu;
    for (int t = 0; t < 128; ++t) {
      const float v = tv[m * 128 + t];
      const uint ix = tix[m * 128 + t];
      if (v > best || (v == best && ix < best_i)) {
        best = v;
        best_i = ix;
      }
    }
    pval_p[(ulong)tgid.x * 8 + m] = best;
    pidx_p[(ulong)tgid.x * 8 + m] = best_i;
  }
}

// Grid: (n_tiles-threads x, m y) is wasteful; one TG per row instead, 256
// threads striding the tiles. Lowest-index tie rule preserved.
kernel void kernel_mm2d_q4k_argmax_reduce(
    device const float * pval_p [[ buffer(0) ]],
    device const uint  * pidx_p [[ buffer(1) ]],
    device uint        * out_p  [[ buffer(2) ]],
    constant int       & ntiles [[ buffer(3) ]],
    uint m    [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
  threadgroup float tv[256];
  threadgroup uint tix[256];
  float best = -INFINITY;
  uint best_i = 0xffffffffu;
  for (int t = int(tidx); t < ntiles; t += 256) {
    const float v = pval_p[(ulong)t * 8 + m];
    const uint ix = pidx_p[(ulong)t * 8 + m];
    if (v > best || (v == best && ix < best_i)) {
      best = v;
      best_i = ix;
    }
  }
  tv[tidx] = best;
  tix[tidx] = best_i;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tidx == 0) {
    for (int t = 1; t < 256; ++t) {
      if (tv[t] > best || (tv[t] == best && tix[t] < best_i)) {
        best = tv[t];
        best_i = tix[t];
      }
    }
    out_p[m] = best_i;
  }
}
