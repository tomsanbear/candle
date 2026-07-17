#include <metal_stdlib>
using namespace metal;

// Bit-plane popcount ternary GEMV (DSpark verify, B3 spike).
//
// Weights are ternary {-1,0,+1} stored as two K-bit planes per output column
// (wpos/wneg: bit j of word w = weight at k = 32w+j is +1 / -1) with a per-128
// fp16 scale d: value_{n,k} = d_blk(n) * (pos - neg). Activations are int4
// bit-sliced per row: a ~= s_m * q, q in [-7,7], u = q + 8 in [1,15] sliced
// into 4 offset planes u_b. The dot folds as
//   sum_k d w a = s_m * sum_blk d_blk * (sum_b 2^b (pc(u_b&wp) - pc(u_b&wn))
//                                        - 8 * (pc(wp) - pc(wn)))
// so the per-weight work is AND+popcount — no 2-bit unpack, no multiply, no
// matrix-unit tile. Chases the instruction-throughput wall that caps mc at
// ~20 GB/s (m=8) and mm2d at ~43 GB/s; the roof is the m=1 mv's ~106 GB/s.
//
// Geometry: one simdgroup per output column n (two per threadgroup); lanes
// stride the K/128 blocks so the per-128 d fold happens in registers, then a
// simd_sum per activation row. Registers stay lean per metal_notes 15.C
// (occupancy over ILP): 8 float accs + 8 weight words + temps.
kernel void kernel_ternary_bitplane_qmv(
    device const uint  * wpos   [[ buffer(0) ]],  // [n][k/32]
    device const uint  * wneg   [[ buffer(1) ]],  // [n][k/32]
    device const half  * dscale [[ buffer(2) ]],  // [n][k/128]
    device const uint  * aplane [[ buffer(3) ]],  // [m][4][k/32], u = q+8 slices
    device const float * ascale [[ buffer(4) ]],  // [m]
    device float       * dst    [[ buffer(5) ]],  // [m][n]
    constant int4      & dims   [[ buffer(6) ]],  // (m, n, k, unused)
    uint2 tgid  [[ threadgroup_position_in_grid ]],
    uint  sgitg [[ simdgroup_index_in_threadgroup ]],
    uint  lane  [[ thread_index_in_simdgroup ]]) {
  const int M = dims.x;
  const int N = dims.y;
  const int K = dims.z;
  const int KW = K >> 5;  // 32-bit words per column
  const int NB = K >> 7;  // 128-weight blocks per column
  const int n = int(tgid.x) * 2 + int(sgitg);
  if (n >= N) {
    return;
  }

  float facc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  device const uint * wp_col = wpos + n * KW;
  device const uint * wn_col = wneg + n * KW;
  device const half * d_col = dscale + n * NB;

  for (int blk = int(lane); blk < NB; blk += 32) {
    const int w0 = blk << 2;
    const uint wp0 = wp_col[w0 + 0];
    const uint wp1 = wp_col[w0 + 1];
    const uint wp2 = wp_col[w0 + 2];
    const uint wp3 = wp_col[w0 + 3];
    const uint wn0 = wn_col[w0 + 0];
    const uint wn1 = wn_col[w0 + 1];
    const uint wn2 = wn_col[w0 + 2];
    const uint wn3 = wn_col[w0 + 3];
    const int wsum = int(popcount(wp0) + popcount(wp1) + popcount(wp2) + popcount(wp3))
                   - int(popcount(wn0) + popcount(wn1) + popcount(wn2) + popcount(wn3));
    const float db = float(d_col[blk]);
    for (int m = 0; m < M; ++m) {
      int bsum = 0;
      for (int b = 0; b < 4; ++b) {
        device const uint * ap = aplane + (m * 4 + b) * KW + w0;
        const int pp = int(popcount(ap[0] & wp0) + popcount(ap[1] & wp1)
                         + popcount(ap[2] & wp2) + popcount(ap[3] & wp3));
        const int pn = int(popcount(ap[0] & wn0) + popcount(ap[1] & wn1)
                         + popcount(ap[2] & wn2) + popcount(ap[3] & wn3));
        bsum += (pp - pn) << b;
      }
      facc[m] = fma(db, float(bsum - 8 * wsum), facc[m]);
    }
  }

  for (int m = 0; m < M; ++m) {
    const float t = simd_sum(facc[m]);
    if (lane == 0) {
      dst[m * N + n] = ascale[m] * t;
    }
  }
}
