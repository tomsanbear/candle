// Tensor-op (matmul2d) path for Q2_0-quantized (ternary) linears at m in
// [1,8], via the hardware 2-bit weight operand. Ternary mirror of mm2d_q4k:
// activations are the bfloat A operand, the packed ternary codes are the
// uint2b_format B operand, and the tensor/matrix unit unpacks the 2-bit lanes
// in hardware.
//
// REQUIRES Metal 4.1 (uint2b_format matmul2d operand): the Metal Toolchain in
// Xcode 27 beta 3 (metalfe-32023.918); Xcode 26.6 (32023.883) has only 4-bit.
// Like mm2d_q4k this needs the MetalPerformancePrimitives framework header the
// runtime compiler cannot see, so it is built offline
// (scripts/build_mm2d_q2_0.sh, -std=metal4.1) into mm2d_q2_0.metallib.
//
// Weight layout (candle-core q2_0_mm2d_planes):
//   codes: [K, Npad] little-endian 2-bit, n innermost, Npad = ceil(N/64)*64.
//   d:     [K/128, Npad] fp16 per-128-block scale, block-major (n innermost).
// Q2_0 code c in {0,1,2,3} encodes value (c-1)*d; the dot folds as
//   sum_k (c_k-1)*d*a_k = d*(sum c*a - sum a) = d*(P - rowsum)
// so matmul2d yields P = sum c*a (raw 2-bit code) and the epilogue applies the
// single per-128 scale to both P and the A row sum.
//
// Templated on the tile geometry so variants sweep in one metallib:
//   TILE_N  output-N tile per threadgroup (64 or 32)
//   BK      matmul2d K-tile / reduction chunk. MUST divide 128 (the scale
//           block) so one d spans an integer number of K-tiles. Larger BK =>
//           fewer op.run() calls (K/BK) and fewer epilogue folds.
//   NSIMD   simdgroups per threadgroup (4 for TILE_N=64, 1 for TILE_N=32).
//   RELAXED matmul2d relaxed_precision (trade accuracy for speed).
// The matmul2d M tile is hardware-fixed at 8; m<8 rides it, A extents (K,m)
// edge-check reads, C writes guard on m. dims = (K, Npad, N_real, m_real).

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

// Row-sum threadgroup capacity is right-sized PER BK: NJ = K/BK <= 8192/BK
// (the wrapper asserts K <= 8192), 8 activation rows. Sizing it 8*(8192/BK)
// instead of a fixed 8*256 keeps the static threadgroup allocation small for
// the large-BK variants (k128: 2 KB, not 8 KB) so it does not cap occupancy.
// NOTE: no max_total_threads_per_threadgroup attribute — pinning it to 128 (=
// NSIMD*32) let the compiler bloat registers per thread (maxTPT 128 vs q4_K's
// 1024); the matmul2d scope already fixes the simdgroup count, and the host
// dispatches exactly NSIMD*32 threads.
#define MM2D_MAX_NJ(bk) (8192 / (bk))

template <int TILE_N, int BK, int NSIMD, bool RELAXED>
[[kernel]]
void mm2d_q2_0(
    device const bfloat * a_p  [[ buffer(0) ]],
    device const uchar  * b_p  [[ buffer(1) ]],
    device const half   * d_p  [[ buffer(2) ]],
    device bfloat       * c_p  [[ buffer(3) ]],
    constant int4       & dims [[ buffer(4) ]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tidx [[thread_index_in_threadgroup]]) {
  const int K = dims.x;
  const int Npad = dims.y;
  const int Nreal = dims.z;
  const int Mreal = dims.w;
  const int NJ = K / BK;
  constexpr int TG_THREADS = NSIMD * 32;
  const int n0 = int(tgid.x) * TILE_N;

  tensor<device bfloat, dextents<int, 2>, tensor_inline>
      a((device bfloat *)a_p, dextents<int, 2>(K, Mreal));
  tensor<device uint2b_format, dextents<int, 2>, tensor_inline>
      b((device uchar *)b_p, dextents<int, 2>(Npad, K));

  constexpr auto desc =
      tensor_ops::matmul2d_descriptor(8, TILE_N, BK, false, false, RELAXED);
  tensor_ops::matmul2d<desc, execution_simdgroups<NSIMD>> op;

  // Per-BK-slice row sums of A (for the -d*rowsum term), threadgroup memory.
  threadgroup float rs_tg[8 * MM2D_MAX_NJ(BK)];
  const int entries = Mreal * NJ;
  for (int e = int(tidx); e < entries; e += TG_THREADS) {
    const int m = e / NJ;
    const int j = e % NJ;
    device const bfloat * row = a_p + m * K + BK * j;
    float s = 0.0f;
    for (int i = 0; i < BK; ++i) {
      s += float(row[i]);
    }
    rs_tg[e] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto acc =
      op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    acc[i] = 0.0f;
  }

  for (int j = 0; j < NJ; ++j) {
    auto mA = a.slice(BK * j, 0);
    auto mB = b.slice(n0, BK * j);
    auto p =
        op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    op.run(mA, mB, p);
    const int blk = (j * BK) >> 7; // 128-block index of this K-tile
    for (ushort i = 0; i < p.get_capacity(); ++i) {
      auto mdi = p.get_multidimensional_index(i);
      const int n = n0 + mdi[0];
      const int m = mdi[1];
      const float dd = float(d_p[blk * Npad + n]);
      const float r = m < Mreal ? rs_tg[m * NJ + j] : 0.0f;
      acc[i] = fma(dd, p[i], fma(-dd, r, acc[i]));
    }
  }

  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    auto mdi = acc.get_multidimensional_index(i);
    const int n = n0 + mdi[0];
    const int m = mdi[1];
    if (n < Nreal && m < Mreal) {
      c_p[m * Nreal + n] = bfloat(acc[i]);
    }
  }
}

// Explicit instantiations (mlx decltype pattern; see gemv.metal). Host name
// kernel_mul_mm2d_q2_0_<suffix>; the wrapper picks by suffix.
#define instantiate_mm2d_q2_0(tile_n, bk, nsimd, relaxed, suffix)             \
  template [[host_name("kernel_mul_mm2d_q2_0_" #suffix)]] [[kernel]]          \
  decltype(mm2d_q2_0<tile_n, bk, nsimd, (bool)relaxed>)                       \
      mm2d_q2_0<tile_n, bk, nsimd, (bool)relaxed>;

instantiate_mm2d_q2_0(64, 32, 4, 0, t64_k32)
instantiate_mm2d_q2_0(64, 64, 4, 0, t64_k64)
instantiate_mm2d_q2_0(64, 128, 4, 0, t64_k128)
instantiate_mm2d_q2_0(64, 128, 4, 1, t64_k128_relaxed)
instantiate_mm2d_q2_0(32, 128, 1, 0, t32_k128)
instantiate_mm2d_q2_0(32, 128, 1, 1, t32_k128_relaxed)
