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
// rows have dsc = dmm = 0 so their (unwritten) outputs are inert.
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

// Per-32-element row sums of A, feeding the dmin term:
// rs[m, j] = sum_{k in [32j, 32j+32)} A[m, k]. Grid: (K/32, m_real).
kernel void kernel_mm2d_q4k_rowsums(
    device const bfloat * a  [[ buffer(0) ]],
    device float        * rs [[ buffer(1) ]],
    constant int        & k_dim [[ buffer(2) ]],
    uint2 tid [[thread_position_in_grid]]) {
  const int nj = k_dim / 32;
  device const bfloat * row = a + tid.y * k_dim + tid.x * 32;
  float s = 0.0f;
  for (int i = 0; i < 32; ++i) {
    s += float(row[i]);
  }
  rs[tid.y * nj + tid.x] = s;
}

kernel void kernel_mul_mm2d_q4k_bf16(
    device const bfloat * a_p   [[ buffer(0) ]],
    device const uchar  * b_p   [[ buffer(1) ]],
    device const half   * dsc_p [[ buffer(2) ]],
    device const half   * dmm_p [[ buffer(3) ]],
    device const float  * rs_p  [[ buffer(4) ]],
    device bfloat       * c_p   [[ buffer(5) ]],
    constant int4       & dims  [[ buffer(6) ]],
    uint2 tgid [[threadgroup_position_in_grid]]) {
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
      // rs is Mreal x NJ: tile rows beyond Mreal are dead (never stored)
      // but must not read out of bounds.
      const float r = m < Mreal ? rs_p[m * NJ + j] : 0.0f;
      acc[i] = fma(w, p[i], fma(-mn, r, acc[i]));
    }
  }

  // Direct bf16 stores: real-N/real-m guards double as tail handling, so no
  // padded C allocation is needed.
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    auto mdi = acc.get_multidimensional_index(i);
    const int n = n0 + mdi[0];
    const int m = mdi[1];
    if (n < Nreal && m < Mreal) {
      c_p[m * Nreal + n] = bfloat(acc[i]);
    }
  }
}
