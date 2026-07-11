#include <metal_stdlib>
using namespace metal;

// Fused GatedDeltaNet single-token decode step (Qwen3.5-hybrid layer).
//
// One dispatch replaces the layer's ~95-op tensor chain: depthwise causal
// conv + silu, gate scalars, q/k l2-normalization, the gated delta rule
// state update, group RMSNorm and silu(z) output gating. Grid is one
// threadgroup per value head, one thread per value column; no cross-
// threadgroup dependency exists because a head owns its conv channels,
// gate scalars, state slab and output slice outright.
//
// Layouts (all contiguous):
//   proj      bf16 [conv_dim + value_dim + heads + heads]
//             = qkv-conv input | z | b_in | a_in (one packed projection)
//   conv_*    bf16 [conv_dim, ksz]   rolling window, oldest first
//   state_*   f32  [heads, dk, dv]
//   conv_w    bf16 [conv_dim, ksz]   taps, ascending
//   out       bf16 [value_dim]
//
// States are written to distinct output buffers (never in place) so the
// host side keeps its replace-by-assignment snapshot semantics.

#define GD_MAX_KSZ 8

// Sum `x` over the threadgroup (dv threads, dv a multiple of 32).
static inline float tg_sum(float x,
                           threadgroup float *scratch,
                           uint tid,
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

struct GdP {
    uint heads;
    uint dk;
    uint dv;
    uint conv_dim;
    uint key_dim;
    uint value_dim;
    uint ksz;
    float l2_eps;
    float norm_eps;
};

kernel void gated_delta_decode_bf16(
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
    constant uint  &heads      [[buffer(10)]],
    constant uint  &dk         [[buffer(11)]],
    constant uint  &dv         [[buffer(12)]],
    constant uint  &conv_dim   [[buffer(13)]],
    constant uint  &key_dim    [[buffer(14)]],
    constant uint  &value_dim  [[buffer(15)]],
    constant uint  &ksz        [[buffer(16)]],
    constant float &l2_eps     [[buffer(17)]],
    constant float &norm_eps   [[buffer(18)]],
    uint h          [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const GdP p = {heads, dk, dv, conv_dim, key_dim, value_dim, ksz, l2_eps, norm_eps};
    threadgroup float q_sh[256];
    threadgroup float k_sh[256];
    threadgroup float scratch[8];
    const uint n_simd_groups = p.dv / 32;

    // ---- Phase 1: depthwise conv + silu on this head's q/k/v channels.
    // Window semantics match the tensor path exactly: the state holds the
    // last ksz inputs; the new window is [state[1..], x_new] and is also
    // the next conv state.
    float qkv[3];
    const uint chans[3] = {
        h * p.dk + tid,
        p.key_dim + h * p.dk + tid,
        2 * p.key_dim + h * p.dv + tid,
    };
    for (uint c = 0; c < 3; c++) {
        const uint ch = chans[c];
        float win[GD_MAX_KSZ];
        for (uint t = 0; t + 1 < p.ksz; t++) {
            win[t] = float(conv_in[ch * p.ksz + t + 1]);
        }
        win[p.ksz - 1] = float(proj[ch]);
        float acc = 0.0f;
        for (uint t = 0; t < p.ksz; t++) {
            acc += float(conv_w[ch * p.ksz + t]) * win[t];
            conv_out[ch * p.ksz + t] = bfloat(win[t]);
        }
        qkv[c] = acc / (1.0f + metal::precise::exp(-acc)); // silu
    }

    // ---- Phase 2: per-head gate scalars (uniform across the TG).
    const float b_in = float(proj[p.conv_dim + p.value_dim + h]);
    const float a_in = float(proj[p.conv_dim + p.value_dim + p.heads + h]);
    // g = -a_log_exp * softplus(a_in + dt_bias); overflow degrades to
    // decay = 0 exactly like the tensor chain.
    const float g = -a_log_exp[h]
        * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
    const float decay = metal::precise::exp(g);
    const float beta = 1.0f / (1.0f + metal::precise::exp(-b_in));

    // ---- Phase 3: l2norm of q and k over dk; q additionally scaled by
    // 1/sqrt(dk). Matches x * (sum(x^2) + eps)^-0.5.
    const float q2 = tg_sum(qkv[0] * qkv[0], scratch, tid, simd_group, simd_lane, n_simd_groups);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k2 = tg_sum(qkv[1] * qkv[1], scratch, tid, simd_group, simd_lane, n_simd_groups);
    const float qn = qkv[0] * metal::precise::powr(q2 + p.l2_eps, -0.5f)
        * metal::precise::rsqrt(float(p.dk));
    const float kn = qkv[1] * metal::precise::powr(k2 + p.l2_eps, -0.5f);

    q_sh[tid] = qn;
    k_sh[tid] = kn;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- Phase 4: gated delta rule; thread tid owns state column tid.
    // Loads/stores are coalesced: at step i all threads touch row i.
    device const float *s_in = state_in + (ulong)h * p.dk * p.dv;
    device float *s_out = state_out + (ulong)h * p.dk * p.dv;
    float kv_mem = 0.0f;
    for (uint i = 0; i < p.dk; i++) {
        kv_mem += k_sh[i] * s_in[i * p.dv + tid];
    }
    kv_mem *= decay;
    const float delta = (qkv[2] - kv_mem) * beta;
    float o = 0.0f;
    for (uint i = 0; i < p.dk; i++) {
        const float s = decay * s_in[i * p.dv + tid] + k_sh[i] * delta;
        s_out[i * p.dv + tid] = s;
        o += q_sh[i] * s;
    }

    // ---- Phase 5: group RMSNorm over the head's output + silu(z) gate.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float o2 = tg_sum(o * o, scratch, tid, simd_group, simd_lane, n_simd_groups);
    const float inv_rms = metal::precise::powr(o2 / float(p.dv) + p.norm_eps, -0.5f);
    const float z_in = float(proj[p.conv_dim + h * p.dv + tid]);
    const float z_silu = z_in / (1.0f + metal::precise::exp(-z_in));
    out[h * p.dv + tid] = bfloat(o * inv_rms * norm_w[tid] * z_silu);
}
