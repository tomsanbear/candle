// Tensor-op (matmul2d) path for Q2_0-quantized (ternary) linears at m in
// [1,8], via the hardware 2-bit weight operand. This is the ternary mirror of
// mm2d_q4k.metal: activations are the bfloat A operand, the packed ternary
// codes are the uint2b_format B operand, and the tensor/matrix unit unpacks
// the 2-bit lanes in hardware (no software staging — that was the 38 GB/s
// ceiling of the hand-rolled planar kernel).
//
// REQUIRES Metal 4.1: the uint2b_format packed type and its matmul2d operand
// support landed in the Metal Toolchain shipped with Xcode 27 beta 3
// (metalfe-32023.918); Xcode 26.6 (32023.883) has only the 4-bit types. Like
// mm2d_q4k this needs the MetalPerformancePrimitives framework header the
// runtime compiler cannot see, so it is built offline
// (scripts/build_mm2d_q2_0.sh, -std=metal4.1) into mm2d_q2_0.metallib, checked
// in beside this file and loaded via newLibraryWithData. Pipeline creation
// fails on toolchains without 2-bit support and callers fall back by routing.
//
// Weight layout (produced by candle-core q2_0_mm2d_planes):
//   codes: [K, Npad] little-endian 2-bit, n innermost, Npad = ceil(N/64)*64,
//          padding codes zero.
//   d:     [K/128, Npad] fp16 per-128-block scale, block-major (n innermost so
//          the epilogue's per-n read is coalesced).
// Q2_0 code c in {0,1,2,3} encodes ternary value (c-1)*d. The dot identity
//   sum_k (c_k - 1)*d*a_k = d*(sum_k c_k*a_k - sum_k a_k) = d*(P - rowsum)
// so matmul2d yields P = sum c*a (raw 2-bit code as an unsigned operand) and
// the epilogue folds a single per-128 scale over both P and the A row sum
// (vs q4_K's separate dsc*P - dmm*rowsum). rowsum is accumulated per-32 slice
// in threadgroup memory and shares the per-128 d across its 4 sub-slices.
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
// K caps at 8192 here (8KB threadgroup budget); every model shape is below.
#define MM2D_MAX_NJ 256

// Two tile geometries (mirror mm2d_q4k): 64-wide/4-simdgroup default;
// 32-wide/1-simdgroup halves per-TG work / doubles TG count for wide-N.
#define MM2D_Q2_0_KERNEL(NAME, TILE_N, SCOPE, TG_THREADS)                    \
kernel void NAME(                                                            \
    device const bfloat * a_p  [[ buffer(0) ]],                             \
    device const uchar  * b_p  [[ buffer(1) ]],                             \
    device const half   * d_p  [[ buffer(2) ]],                             \
    device bfloat       * c_p  [[ buffer(3) ]],                             \
    constant int4       & dims [[ buffer(4) ]],                             \
    uint2 tgid [[threadgroup_position_in_grid]],                            \
    uint  tidx [[thread_index_in_threadgroup]]) {                           \
  const int K = dims.x;                                                      \
  const int Npad = dims.y;                                                   \
  const int Nreal = dims.z;                                                  \
  const int Mreal = dims.w;                                                  \
  const int NJ = K / 32;                                                     \
  const int n0 = int(tgid.x) * TILE_N;                                       \
                                                                            \
  tensor<device bfloat, dextents<int, 2>, tensor_inline>                    \
      a((device bfloat *)a_p, dextents<int, 2>(K, Mreal));                  \
  tensor<device uint2b_format, dextents<int, 2>, tensor_inline>             \
      b((device uchar *)b_p, dextents<int, 2>(Npad, K));                    \
                                                                            \
  constexpr auto desc = tensor_ops::matmul2d_descriptor(8, TILE_N, 32);      \
  tensor_ops::matmul2d<desc, SCOPE> op;                                      \
                                                                            \
  /* Per-32-slice row sums of A (for the -d*rowsum term), threadgroup mem. */\
  threadgroup float rs_tg[8 * MM2D_MAX_NJ];                                  \
  const int entries = Mreal * NJ;                                            \
  for (int e = int(tidx); e < entries; e += TG_THREADS) {                    \
    const int m = e / NJ;                                                    \
    const int j = e % NJ;                                                    \
    device const bfloat * row = a_p + m * K + 32 * j;                        \
    float s = 0.0f;                                                          \
    for (int i = 0; i < 32; ++i) {                                           \
      s += float(row[i]);                                                    \
    }                                                                        \
    rs_tg[e] = s;                                                            \
  }                                                                          \
  threadgroup_barrier(mem_flags::mem_threadgroup);                          \
                                                                            \
  auto acc = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
  for (ushort i = 0; i < acc.get_capacity(); ++i) {                          \
    acc[i] = 0.0f;                                                           \
  }                                                                          \
                                                                            \
  for (int j = 0; j < NJ; ++j) {                                             \
    auto mA = a.slice(32 * j, 0);                                            \
    auto mB = b.slice(n0, 32 * j);                                           \
    auto p = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>(); \
    op.run(mA, mB, p);                                                       \
    const int blk = j >> 2; /* per-128 scale spans 4 per-32 sub-slices */    \
    for (ushort i = 0; i < p.get_capacity(); ++i) {                          \
      auto mdi = p.get_multidimensional_index(i);                           \
      const int n = n0 + mdi[0];                                             \
      const int m = mdi[1];                                                  \
      const float dd = float(d_p[blk * Npad + n]);                           \
      /* Tile rows beyond Mreal are dead (never stored) but must not read  */ \
      /* out of bounds.                                                    */ \
      const float r = m < Mreal ? rs_tg[m * NJ + j] : 0.0f;                  \
      acc[i] = fma(dd, p[i], fma(-dd, r, acc[i]));                           \
    }                                                                        \
  }                                                                          \
                                                                            \
  for (ushort i = 0; i < acc.get_capacity(); ++i) {                          \
    auto mdi = acc.get_multidimensional_index(i);                           \
    const int n = n0 + mdi[0];                                               \
    const int m = mdi[1];                                                    \
    if (n < Nreal && m < Mreal) {                                            \
      c_p[m * Nreal + n] = bfloat(acc[i]);                                   \
    }                                                                        \
  }                                                                          \
}

MM2D_Q2_0_KERNEL(kernel_mul_mm2d_q2_0_bf16, 64, execution_simdgroups<4>, 128)
MM2D_Q2_0_KERNEL(kernel_mul_mm2d_q2_0_bf16_t32, 32, execution_simdgroup, 32)
