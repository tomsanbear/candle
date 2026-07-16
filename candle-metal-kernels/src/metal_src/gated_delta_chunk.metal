#include <metal_stdlib>
using namespace metal;

// Fused GatedDeltaNet chunk step (l <= 32 positions in one dispatch).
//
// Covers the same span as the decode kernel — depthwise conv + silu, gate
// scalars, q/k l2norm, the WY-form gated delta rule (forward substitution,
// exact), group RMSNorm and silu(z) gating — for a whole chunk, and emits
// the reconstruction intermediates (normed k, pseudo-values delta, decay
// cumsum) the host uses for closed-form speculative rollback.
//
// Grid: one threadgroup per value head, dv threads. The sequential
// dependency across positions runs as an l-step loop with threadgroup
// barriers; every step is dv-parallel.
//
// Layouts (contiguous):
//   proj      bf16 [l, conv_dim + value_dim + 2*heads]  (qkv | z | b | a per position)
//   conv_in   bf16 [conv_dim, ksz]; conv_out likewise (window after last position)
//   state_*   f32  [heads, dk, dv]
//   out       bf16 [l, value_dim]
//   cap_k     f32  [heads, l, dk]   l2-normed keys
//   cap_delta f32  [heads, l, dv]   WY pseudo-values
//   cap_gcs   f32  [heads, l]       inclusive log-decay cumsum

// Threadgroup memory bounds the chunk: 4 arrays x GDC_MAX_L x GDC_DIM x 4B
// must stay under the 32KB threadgroup budget, so GDC_MAX_L=12 with
// GDC_DIM=128 uses ~24.6KB. Verify chunks are gamma+1 <= 9; prefill keeps
// the tensor path. The host enforces l <= GDC_MAX_L and dk == dv == GDC_DIM.
#define GDC_MAX_L 12
#define GDC_DIM 128
#define GDC_MAX_KSZ 8

static inline float tg_sum_gdc(float x,
                               threadgroup float *scratch,
                               uint simd_group,
                               uint simd_lane,
                               uint n_simd_groups) {
    float partial = simd_sum(x);
    if (simd_lane == 0) {
        scratch[simd_group] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < n_simd_groups; s++) {
        total += scratch[s];
    }
    return total;
}

kernel void gated_delta_chunk_bf16(
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
    device float        *cap_k     [[buffer(10)]],
    device float        *cap_delta [[buffer(11)]],
    device float        *cap_gcs   [[buffer(12)]],
    constant uint  &heads      [[buffer(13)]],
    constant uint  &dk         [[buffer(14)]],
    constant uint  &dv         [[buffer(15)]],
    constant uint  &conv_dim   [[buffer(16)]],
    constant uint  &key_dim    [[buffer(17)]],
    constant uint  &value_dim  [[buffer(18)]],
    constant uint  &ksz        [[buffer(19)]],
    constant uint  &seq_len    [[buffer(20)]],
    constant float &l2_eps     [[buffer(21)]],
    constant float &norm_eps   [[buffer(22)]],
    constant uint  &num_k_heads [[buffer(23)]],
    uint h          [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row_stride = conv_dim + value_dim + 2 * heads;
    const uint l = seq_len;
    // GQA: value head h reads its group's shared q/k channels (Bonsai: 48
    // value heads over 16 k-heads). Sibling threadgroups redo the same q/k
    // conv + write identical conv_out bytes for those channels — benign.
    const uint kq = h / (heads / num_k_heads);

    threadgroup float k_sh[GDC_MAX_L * GDC_DIM];   // normed k, [t][dk]
    threadgroup float q_sh[GDC_MAX_L * GDC_DIM];   // normed+scaled q, [t][dk]
    threadgroup float v_sh[GDC_MAX_L * GDC_DIM];   // conv'd v, [t][dv]
    threadgroup float delta_sh[GDC_MAX_L * GDC_DIM]; // pseudo-values, [t][dv]
    threadgroup float gcs_sh[GDC_MAX_L];
    threadgroup float beta_sh[GDC_MAX_L];
    threadgroup float scratch[8];
    const uint n_simd_groups = dv / 32;

    // ---- Phase 1: conv + silu across the chunk for this head's channels.
    // Window walks the ksz-1 retained inputs then the chunk's own inputs.
    const uint chans[3] = {
        kq * dk + tid,
        key_dim + kq * dk + tid,
        2 * key_dim + h * dv + tid,
    };
    for (uint c = 0; c < 3; c++) {
        const uint ch = chans[c];
        float win[GDC_MAX_KSZ];
        for (uint t = 0; t + 1 < ksz; t++) {
            win[t] = float(conv_in[ch * ksz + t + 1]);
        }
        for (uint pos = 0; pos < l; pos++) {
            win[ksz - 1] = float(proj[pos * row_stride + ch]);
            float acc = 0.0f;
            for (uint t = 0; t < ksz; t++) {
                acc += float(conv_w[ch * ksz + t]) * win[t];
            }
            const float y = acc / (1.0f + metal::precise::exp(-acc));
            if (c == 0) q_sh[pos * dk + tid] = y;
            else if (c == 1) k_sh[pos * dk + tid] = y;
            else v_sh[pos * dv + tid] = y;
            // slide
            for (uint t = 0; t + 1 < ksz; t++) win[t] = win[t + 1];
        }
        // Final window = last ksz inputs consumed on this channel.
        // win[0..ksz-2] currently holds inputs l-ksz+2..l-1 shifted; rebuild
        // from the tail to keep it simple and exact.
        for (uint t = 0; t < ksz; t++) {
            // Window slot t corresponds to input position l - ksz + t
            // (negative -> from conv_in at offset ksz + (l - ksz + t) ... ).
            const int src = int(l) - int(ksz) + int(t);
            float val;
            if (src >= 0) {
                // Recompute from proj (raw input, pre-conv).
                val = float(proj[uint(src) * row_stride + ch]);
            } else {
                // From the retained window: conv_in[ch][ksz + src].
                val = float(conv_in[ch * ksz + uint(int(ksz) + src)]);
            }
            conv_out[ch * ksz + t] = bfloat(val);
        }
    }

    // ---- Phase 2: per-position gate scalars + decay cumsum (thread 0).
    if (tid == 0) {
        float run = 0.0f;
        for (uint pos = 0; pos < l; pos++) {
            const float b_in = float(proj[pos * row_stride + conv_dim + value_dim + h]);
            const float a_in = float(proj[pos * row_stride + conv_dim + value_dim + heads + h]);
            const float g = -a_log_exp[h]
                * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
            run += g;
            gcs_sh[pos] = run;
            beta_sh[pos] = 1.0f / (1.0f + metal::precise::exp(-b_in));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 3: l2norm q (scaled by 1/sqrt(dk)) and k per position.
    for (uint pos = 0; pos < l; pos++) {
        const float qv = q_sh[pos * dk + tid];
        const float kv = k_sh[pos * dk + tid];
        const float q2 = tg_sum_gdc(qv * qv, scratch, simd_group, simd_lane, n_simd_groups);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k2 = tg_sum_gdc(kv * kv, scratch, simd_group, simd_lane, n_simd_groups);
        q_sh[pos * dk + tid] = qv * metal::precise::powr(q2 + l2_eps, -0.5f)
            * metal::precise::rsqrt(float(dk));
        k_sh[pos * dk + tid] = kv * metal::precise::powr(k2 + l2_eps, -0.5f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float *s_in = state_in + (ulong)h * dk * dv;
    device float *s_out = state_out + (ulong)h * dk * dv;

    // ---- Phase 3.5: cooperative pair dots + fused S0 pass. Replaces the
    // 2*l^2 barrier-chained tg_sum reductions of v1 with one barrier: each
    // thread owns whole (t, j) pairs; ks0/qs0 accumulate in register arrays
    // over a single sweep of the state.
    threadgroup float kk_sh[GDC_MAX_L * GDC_MAX_L];
    threadgroup float qk_sh[GDC_MAX_L * GDC_MAX_L];
    for (uint p = tid; p < l * l; p += dv) {
        const uint t = p / l;
        const uint j = p % l;
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
    float ks0[GDC_MAX_L];
    float qs0[GDC_MAX_L];
#pragma unroll
    for (uint t = 0; t < GDC_MAX_L; t++) {
        ks0[t] = 0.0f;
        qs0[t] = 0.0f;
    }
    for (uint i = 0; i < dk; i++) {
        const float s0 = s_in[i * dv + tid];
#pragma unroll
        for (uint t = 0; t < GDC_MAX_L; t++) {
            if (t < l) {
                ks0[t] = fma(k_sh[t * dk + i], s0, ks0[t]);
                qs0[t] = fma(q_sh[t * dk + i], s0, qs0[t]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 4: forward substitution for the WY pseudo-values.
    // delta_t = beta_t * (v_t - gamma_t * k_t^T S0) - sum_{j<t} B[t,j] delta_j
    // with B[t,j] = beta_t * exp(G_t - G_j) * (k_t . k_j). Each thread only
    // touches its own column of delta_sh, so the t loop needs no barriers.
    for (uint t = 0; t < l; t++) {
        const float gamma_t = metal::precise::exp(gcs_sh[t]);
        float acc = beta_sh[t] * (v_sh[t * dv + tid] - gamma_t * ks0[t]);
        for (uint j = 0; j < t; j++) {
            const float b_tj = beta_sh[t] * metal::precise::exp(gcs_sh[t] - gcs_sh[j])
                * kk_sh[t * l + j];
            acc -= b_tj * delta_sh[j * dv + tid];
        }
        delta_sh[t * dv + tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 5: outputs (group RMSNorm needs one reduction per position).
    // o_t = gamma_t * q_t^T S0 + sum_{j<=t} exp(G_t - G_j) (q_t . k_j) delta_j
    for (uint t = 0; t < l; t++) {
        float o = metal::precise::exp(gcs_sh[t]) * qs0[t];
        for (uint j = 0; j <= t; j++) {
            o += metal::precise::exp(gcs_sh[t] - gcs_sh[j]) * qk_sh[t * l + j]
                * delta_sh[j * dv + tid];
        }
        const float o2 = tg_sum_gdc(o * o, scratch, simd_group, simd_lane, n_simd_groups);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = metal::precise::powr(o2 / float(dv) + norm_eps, -0.5f);
        const float z_in = float(proj[t * row_stride + conv_dim + h * dv + tid]);
        const float z_silu = z_in / (1.0f + metal::precise::exp(-z_in));
        out[t * value_dim + h * dv + tid] = bfloat(o * inv_rms * norm_w[tid] * z_silu);
    }

    // Final state: S = exp(G_last) S0 + sum_j exp(G_last - G_j) k_j (x) delta_j
    const float g_last = gcs_sh[l - 1];
    float decay_j[GDC_MAX_L];
#pragma unroll
    for (uint j = 0; j < GDC_MAX_L; j++) {
        decay_j[j] = j < l ? metal::precise::exp(g_last - gcs_sh[j]) : 0.0f;
    }
    const float g_last_exp = metal::precise::exp(g_last);
    for (uint i = 0; i < dk; i++) {
        float s = g_last_exp * s_in[i * dv + tid];
#pragma unroll
        for (uint j = 0; j < GDC_MAX_L; j++) {
            if (j < l) {
                s = fma(decay_j[j] * k_sh[j * dk + i], delta_sh[j * dv + tid], s);
            }
        }
        s_out[i * dv + tid] = s;
    }

    // Capture for closed-form rollback (host-side select_verify_state).
    for (uint t = 0; t < l; t++) {
        cap_k[((ulong)h * l + t) * dk + tid] = k_sh[t * dk + tid];
        cap_delta[((ulong)h * l + t) * dv + tid] = delta_sh[t * dv + tid];
    }
    if (tid == 0) {
        for (uint t = 0; t < l; t++) {
            cap_gcs[(ulong)h * l + t] = gcs_sh[t];
        }
    }
}
