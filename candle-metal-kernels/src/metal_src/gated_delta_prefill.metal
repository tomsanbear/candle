#include <metal_stdlib>
using namespace metal;

// Streaming / persistent GatedDeltaNet prefill: ONE dispatch handles the whole
// sequence (any l), one threadgroup per value head, dv threads. Walks the
// sequence in fixed GDP_TILE-position tiles, carrying the depthwise-conv window
// in registers across tiles. The recurrent state S (dk*dv*4 = 64KB/head) is too
// big for threadgroup memory AND spills registers if held per-thread (a first
// pass carrying S[:, tid] in a 128-float register array regressed to 4.11s vs
// the chunk loop's 3.76s), so S stays DEVICE-resident: each thread owns column
// tid, so its reads/writes are coalesced and need no barrier. The win over the
// host-looped chunk kernel is one dispatch + the conv window carried in
// registers, removing the per-dispatch LAUNCH + conv round-trip (the CAP sweep
// attributes ~2/3 of the per-dispatch overhead to those, ~1/3 to S traffic).
// No compile-time dependence on l; no rollback capture. Same algebra as
// gated_delta_chunk; gcs is relative to each tile start, S holding the absolute
// decayed state. S lives in state_out (seeded from state_in), updated in place.
//
// Layouts match gated_delta_chunk (minus the cap_* capture outputs):
//   proj      bf16 [l, conv_dim + value_dim + 2*heads]
//   conv_in   bf16 [conv_dim, ksz]; conv_out likewise (window after last pos)
//   state_*   f32  [heads, dk, dv]
//   out       bf16 [l, value_dim]

#define GDP_TILE 12
#define GDP_DIM 128
#define GDP_MAX_KSZ 8

static inline float tg_sum_gdp(float x, threadgroup float *scratch,
                               uint simd_group, uint simd_lane, uint n_simd_groups) {
    float partial = simd_sum(x);
    if (simd_lane == 0) scratch[simd_group] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < n_simd_groups; s++) total += scratch[s];
    return total;
}

kernel void gated_delta_prefill_bf16(
    device const bfloat *proj      [[buffer(0)]],
    device const bfloat *conv_in   [[buffer(1)]],
    device const float  *state_in  [[buffer(2)]],
    device const bfloat *conv_w    [[buffer(3)]],
    device const float  *dt_bias   [[buffer(4)]],
    device const float  *a_log_exp [[buffer(5)]],
    device const float  *norm_w    [[buffer(6)]],
    device bfloat       *out       [[buffer(7)]],
    device bfloat       *conv_out  [[buffer(8)]],
    device float        *state_out [[buffer(9)]],
    constant uint  &heads       [[buffer(10)]],
    constant uint  &dk          [[buffer(11)]],
    constant uint  &dv          [[buffer(12)]],
    constant uint  &conv_dim    [[buffer(13)]],
    constant uint  &key_dim     [[buffer(14)]],
    constant uint  &value_dim   [[buffer(15)]],
    constant uint  &ksz         [[buffer(16)]],
    constant uint  &seq_len     [[buffer(17)]],
    constant float &l2_eps      [[buffer(18)]],
    constant float &norm_eps    [[buffer(19)]],
    constant uint  &num_k_heads [[buffer(20)]],
    uint h          [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint row_stride = conv_dim + value_dim + 2 * heads;
  const uint l = seq_len;
  const uint kq = num_k_heads == heads ? h : h % num_k_heads;
  const uint n_simd_groups = dv / 32;

  // S stays device-resident (64KB/head won't fit on-chip; register-carry
  // spills). Seed the working copy state_out[:, tid] from state_in; each thread
  // owns column tid, so all S reads/writes below are per-thread and coalesced.
  device float *s_dev = state_out + (ulong)h * dk * dv;
  device const float *s_seed = state_in + (ulong)h * dk * dv;
  for (uint i = 0; i < dk; i++) {
    s_dev[i * dv + tid] = s_seed[i * dv + tid];
  }

  // Conv window (ksz slots, same layout as conv_in) for this thread's 3
  // channels, carried in registers. Reads use slots [1..ksz-1] as the retained
  // inputs, matching gated_delta_chunk's conv_in convention.
  const uint chans[3] = {
      kq * dk + tid, key_dim + kq * dk + tid, 2 * key_dim + h * dv + tid};
  float cwin[3][GDP_MAX_KSZ];
  for (uint cc = 0; cc < 3; cc++) {
    for (uint t = 0; t < ksz; t++) {
      cwin[cc][t] = float(conv_in[chans[cc] * ksz + t]);
    }
  }

  threadgroup float k_sh[GDP_TILE * GDP_DIM];
  threadgroup float q_sh[GDP_TILE * GDP_DIM];
  threadgroup float v_sh[GDP_TILE * GDP_DIM];
  threadgroup float delta_sh[GDP_TILE * GDP_DIM];
  threadgroup float kk_sh[GDP_TILE * GDP_TILE];
  threadgroup float qk_sh[GDP_TILE * GDP_TILE];
  threadgroup float gcs_sh[GDP_TILE];
  threadgroup float beta_sh[GDP_TILE];
  threadgroup float scratch[8];

  for (uint base = 0; base < l; base += GDP_TILE) {
    const uint c = min((uint)GDP_TILE, l - base);

    // ---- Phase 1: conv + silu across the tile for this head's channels.
    for (uint cc = 0; cc < 3; cc++) {
      const uint ch = chans[cc];
      float win[GDP_MAX_KSZ];
      for (uint t = 0; t + 1 < ksz; t++) win[t] = cwin[cc][t + 1];
      for (uint pos = 0; pos < c; pos++) {
        win[ksz - 1] = float(proj[(base + pos) * row_stride + ch]);
        float acc = 0.0f;
        for (uint t = 0; t < ksz; t++) acc += float(conv_w[ch * ksz + t]) * win[t];
        const float y = acc / (1.0f + metal::precise::exp(-acc));
        if (cc == 0) q_sh[pos * dk + tid] = y;
        else if (cc == 1) k_sh[pos * dk + tid] = y;
        else v_sh[pos * dv + tid] = y;
        for (uint t = 0; t + 1 < ksz; t++) win[t] = win[t + 1];
      }
      // Rebuild the carried window = last ksz inputs of the sequence so far
      // (positions base+c-ksz .. base+c-1); read proj where in-range, else the
      // pre-tile window. Mirrors gated_delta_chunk's conv_out rebuild.
      float next[GDP_MAX_KSZ];
      for (uint t = 0; t < ksz; t++) {
        const int src = int(c) - int(ksz) + int(t);
        if (src >= 0) {
          next[t] = float(proj[(base + uint(src)) * row_stride + ch]);
        } else {
          next[t] = cwin[cc][uint(int(ksz) + src)];
        }
      }
      for (uint t = 0; t < ksz; t++) cwin[cc][t] = next[t];
    }

    // ---- Phase 2: per-position gate scalars + decay cumsum (thread 0).
    if (tid == 0) {
      float run = 0.0f;
      for (uint pos = 0; pos < c; pos++) {
        const float b_in = float(proj[(base + pos) * row_stride + conv_dim + value_dim + h]);
        const float a_in = float(proj[(base + pos) * row_stride + conv_dim + value_dim + heads + h]);
        const float g = -a_log_exp[h]
            * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
        run += g;
        gcs_sh[pos] = run;
        beta_sh[pos] = 1.0f / (1.0f + metal::precise::exp(-b_in));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: l2norm q (scaled by 1/sqrt(dk)) and k per position.
    for (uint pos = 0; pos < c; pos++) {
      const float qv = q_sh[pos * dk + tid];
      const float kv = k_sh[pos * dk + tid];
      const float q2 = tg_sum_gdp(qv * qv, scratch, simd_group, simd_lane, n_simd_groups);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float k2 = tg_sum_gdp(kv * kv, scratch, simd_group, simd_lane, n_simd_groups);
      q_sh[pos * dk + tid] = qv * metal::precise::powr(q2 + l2_eps, -0.5f)
          * metal::precise::rsqrt(float(dk));
      k_sh[pos * dk + tid] = kv * metal::precise::powr(k2 + l2_eps, -0.5f);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Phase 3.5: kk/qk pair dots + S0 contributions (s_col in registers).
    for (uint p = tid; p < c * c; p += dv) {
      const uint t = p / c;
      const uint j = p % c;
      float kk = 0.0f;
      float qk = 0.0f;
      for (uint i = 0; i < dk; i++) {
        const float kj = k_sh[j * dk + i];
        kk += k_sh[t * dk + i] * kj;
        qk += q_sh[t * dk + i] * kj;
      }
      kk_sh[p] = kk;
      qk_sh[p] = qk;
    }
    float ks0[GDP_TILE];
    float qs0[GDP_TILE];
    for (uint t = 0; t < c; t++) {
      ks0[t] = 0.0f;
      qs0[t] = 0.0f;
    }
    for (uint i = 0; i < dk; i++) {
      const float s0 = s_dev[i * dv + tid];
      for (uint t = 0; t < c; t++) {
        ks0[t] = fma(k_sh[t * dk + i], s0, ks0[t]);
        qs0[t] = fma(q_sh[t * dk + i], s0, qs0[t]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 4: forward substitution for the WY pseudo-values.
    for (uint t = 0; t < c; t++) {
      const float gamma_t = metal::precise::exp(gcs_sh[t]);
      float acc = beta_sh[t] * (v_sh[t * dv + tid] - gamma_t * ks0[t]);
      for (uint j = 0; j < t; j++) {
        const float b_tj = beta_sh[t] * metal::precise::exp(gcs_sh[t] - gcs_sh[j])
            * kk_sh[t * c + j];
        acc -= b_tj * delta_sh[j * dv + tid];
      }
      delta_sh[t * dv + tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 5: outputs (group RMSNorm needs one reduction per position).
    for (uint t = 0; t < c; t++) {
      float o = metal::precise::exp(gcs_sh[t]) * qs0[t];
      for (uint j = 0; j <= t; j++) {
        o += metal::precise::exp(gcs_sh[t] - gcs_sh[j]) * qk_sh[t * c + j]
            * delta_sh[j * dv + tid];
      }
      const float o2 = tg_sum_gdp(o * o, scratch, simd_group, simd_lane, n_simd_groups);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv_rms = metal::precise::powr(o2 / float(dv) + norm_eps, -0.5f);
      const float z_in = float(proj[(base + t) * row_stride + conv_dim + h * dv + tid]);
      const float z_silu = z_in / (1.0f + metal::precise::exp(-z_in));
      out[(base + t) * value_dim + h * dv + tid] = bfloat(o * inv_rms * norm_w[tid] * z_silu);
    }

    // ---- Update S column (device, in place) to the tile-end state.
    const float g_last = gcs_sh[c - 1];
    float decay_j[GDP_TILE];
    for (uint j = 0; j < c; j++) decay_j[j] = metal::precise::exp(g_last - gcs_sh[j]);
    const float g_last_exp = metal::precise::exp(g_last);
    for (uint i = 0; i < dk; i++) {
      float s = g_last_exp * s_dev[i * dv + tid];
      for (uint j = 0; j < c; j++) {
        s = fma(decay_j[j] * k_sh[j * dk + i], delta_sh[j * dv + tid], s);
      }
      s_dev[i * dv + tid] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // S is already in state_out (updated in place). Write the conv window back.
  for (uint cc = 0; cc < 3; cc++) {
    const uint ch = chans[cc];
    for (uint t = 0; t < ksz; t++) {
      conv_out[ch * ksz + t] = bfloat(cwin[cc][t]);
    }
  }
}
