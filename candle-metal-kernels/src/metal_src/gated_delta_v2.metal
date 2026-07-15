#include <metal_stdlib>
using namespace metal;

// Fused GatedDeltaNet v2 (unified decode/chunk, 1 <= l <= 12).
//
// v1 ran one threadgroup per value head (16 TGs on a 32-core GPU: half the
// machine idle) with each thread walking dk=128 sequentially. v2 re-grids the
// delta-rule core to the MLX gated_delta geometry: one simdgroup per
// (head, value-column), dk split across the 32 lanes (4 registers each,
// simd_sum for the dots), 512 threadgroups at heads=16, dv=128. The state is
// stored TRANSPOSED relative to v1 — [heads, dv, dk] — so a simdgroup's
// column loads are contiguous and the column lives in registers across the
// whole l-loop.
//
// The layer splits into three dispatches (v1 fused everything into one):
//   prep     — conv + silu + l2norm + gate scalars (cheap, head-parallel);
//              stages normed q/k, conv'd v, per-step decay, beta; also emits
//              the k / log-decay-cumsum rollback captures and the rolling
//              conv window.
//   core     — the recurrent delta rule; emits pre-norm outputs and the
//              pseudo-value (delta) capture.
//   epilogue — group RMSNorm + silu(z) output gating (needs the head's full
//              dv row, which the split grid can't reduce across).
//
// The sequential-recurrence deltas equal the WY forward-substitution
// pseudo-values of the v1 chunk kernel, so the captures keep their layouts
// and the host-side closed-form rollback is unchanged.
//
// Layouts (contiguous):
//   proj      bf16 [l, conv_dim + value_dim]  (qkv | z per position — the raw
//             in_proj_qkvz GEMV output, no host-side cat)
//   ba        bf16 [l, 2*heads]  (b | a per position — the fused in_proj_ba
//             dense GEMV output)
//   conv_*    bf16 [conv_dim, ksz]  rolling window, oldest first
//   state_*   f32  [heads, dv, dk]  (v2 layout — transposed vs v1)
//   qn/kn     f32  [heads, l, dk]   (kn doubles as cap_k)
//   vc        f32  [heads, l, dv]
//   g_step    f32  [heads, l]       per-step log decay
//   beta_s    f32  [heads, l]
//   cap_gcs   f32  [heads, l]       inclusive log-decay cumsum
//   cap_delta f32  [heads, l, dv]
//   o_pre     f32  [l, value_dim]
//   out       bf16 [l, value_dim]

#define GD2_MAX_L 12
#define GD2_MAX_KSZ 8

static inline float tg_sum_128(float x,
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

// ---------------------------------------------------------------------------
// prep: grid (heads * batch) TGs x dk threads. Thread tid owns conv channel
// tid within each of the head's three channel blocks (q, k, v), exactly like
// the v1 phase-1 loop, then the per-position l2 norms run as TG reductions.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_prep_bf16(
    device const bfloat *proj      [[buffer(0)]],
    device const bfloat *ba        [[buffer(1)]],
    device const bfloat *conv_in   [[buffer(2)]],
    device const bfloat *conv_w    [[buffer(3)]],
    device const float  *dt_bias   [[buffer(4)]],
    device const float  *a_log_exp [[buffer(5)]],
    device bfloat       *conv_out  [[buffer(6)]],
    device float        *kn        [[buffer(7)]],  // = cap_k
    device float        *qn        [[buffer(8)]],
    device float        *vc        [[buffer(9)]],
    device float        *g_step    [[buffer(10)]],
    device float        *beta_s    [[buffer(11)]],
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
    uint bh         [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint bi = bh / heads;
    const uint h = bh % heads;
    const uint l = seq_len;
    const uint row_stride = conv_dim + value_dim;
    device const bfloat *proj_b = proj + (ulong)bi * l * row_stride;
    device const bfloat *ba_b = ba + (ulong)bi * l * 2 * heads;
    device const bfloat *conv_in_b = conv_in + (ulong)bi * conv_dim * ksz;
    device bfloat *conv_out_b = conv_out + (ulong)bi * conv_dim * ksz;
    const ulong hb = (ulong)(bi * heads + h);

    threadgroup float q_raw[GD2_MAX_L * 128];
    threadgroup float k_raw[GD2_MAX_L * 128];
    threadgroup float scratch[4];
    const uint n_simd_groups = dk / 32;

    // Conv + silu for this thread's q, k and v channels across the chunk;
    // window semantics identical to v1 (state holds the last ksz inputs).
    const uint chans[3] = {
        h * dk + tid,
        key_dim + h * dk + tid,
        2 * key_dim + h * dv + tid,
    };
    for (uint c = 0; c < 3; c++) {
        const uint ch = chans[c];
        float win[GD2_MAX_KSZ];
        for (uint t = 0; t + 1 < ksz; t++) {
            win[t] = float(conv_in_b[ch * ksz + t + 1]);
        }
        for (uint pos = 0; pos < l; pos++) {
            win[ksz - 1] = float(proj_b[pos * row_stride + ch]);
            float acc = 0.0f;
            for (uint t = 0; t < ksz; t++) {
                acc += float(conv_w[ch * ksz + t]) * win[t];
            }
            const float y = acc / (1.0f + metal::precise::exp(-acc));
            if (c == 0) q_raw[pos * dk + tid] = y;
            else if (c == 1) k_raw[pos * dk + tid] = y;
            else vc[(hb * l + pos) * dv + tid] = y;
            for (uint t = 0; t + 1 < ksz; t++) win[t] = win[t + 1];
        }
        // Final rolling window = last ksz inputs on this channel.
        for (uint t = 0; t < ksz; t++) {
            const int src = int(l) - int(ksz) + int(t);
            float val;
            if (src >= 0) {
                val = float(proj_b[uint(src) * row_stride + ch]);
            } else {
                val = float(conv_in_b[ch * ksz + uint(int(ksz) + src)]);
            }
            conv_out_b[ch * ksz + t] = bfloat(val);
        }
    }

    // Gate scalars + decay cumsum (thread 0; uniform per head).
    if (tid == 0) {
        float run = 0.0f;
        for (uint pos = 0; pos < l; pos++) {
            const float b_in = float(ba_b[pos * 2 * heads + h]);
            const float a_in = float(ba_b[pos * 2 * heads + heads + h]);
            const float g = -a_log_exp[h]
                * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
            g_step[hb * l + pos] = g;
            run += g;
            cap_gcs[hb * l + pos] = run;
            beta_s[hb * l + pos] = 1.0f / (1.0f + metal::precise::exp(-b_in));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-position l2 norms; q additionally scaled by 1/sqrt(dk).
    for (uint pos = 0; pos < l; pos++) {
        const float qv = q_raw[pos * dk + tid];
        const float kv = k_raw[pos * dk + tid];
        const float q2 =
            tg_sum_128(qv * qv, scratch, tid, simd_group, simd_lane, n_simd_groups);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k2 =
            tg_sum_128(kv * kv, scratch, tid, simd_group, simd_lane, n_simd_groups);
        qn[(hb * l + pos) * dk + tid] = qv
            * metal::precise::powr(q2 + l2_eps, -0.5f)
            * metal::precise::rsqrt(float(dk));
        kn[(hb * l + pos) * dk + tid] = kv
            * metal::precise::powr(k2 + l2_eps, -0.5f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// core: grid (dv/4, heads * batch) TGs, TG (32, 4). One simdgroup per value
// column; lane owns 4 contiguous dk slots of the transposed state column in
// registers across the whole l-loop.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_core(
    device const float *state_in  [[buffer(0)]],
    device float       *state_out [[buffer(1)]],
    device const float *kn        [[buffer(2)]],
    device const float *qn        [[buffer(3)]],
    device const float *vc        [[buffer(4)]],
    device const float *g_step    [[buffer(5)]],
    device const float *beta_s    [[buffer(6)]],
    device float       *cap_delta [[buffer(7)]],
    device float       *o_pre     [[buffer(8)]],
    constant uint &heads     [[buffer(9)]],
    constant uint &dk        [[buffer(10)]],
    constant uint &dv        [[buffer(11)]],
    constant uint &value_dim [[buffer(12)]],
    constant uint &seq_len   [[buffer(13)]],
    uint2 tg        [[threadgroup_position_in_grid]],
    uint2 tp        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]) {
    const uint bh = tg.y;           // batch * heads index
    const uint h = bh % heads;
    const uint col = tg.x * 4 + tp.y;
    const uint lane = tp.x;
    const uint l = seq_len;
    const uint n_per_lane = dk / 32; // 4 at dk=128

    device const float *s_in = state_in + ((ulong)bh * dv + col) * dk;
    device float *s_out = state_out + ((ulong)bh * dv + col) * dk;

    float state[4];
    for (uint i = 0; i < n_per_lane; i++) {
        state[i] = s_in[lane * n_per_lane + i];
    }

    const ulong hb = (ulong)bh;
    for (uint t = 0; t < l; t++) {
        const float decay = metal::precise::exp(g_step[hb * l + t]);
        device const float *k_t = kn + (hb * l + t) * dk + lane * n_per_lane;
        device const float *q_t = qn + (hb * l + t) * dk + lane * n_per_lane;
        float kv_mem = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] *= decay;
            kv_mem = fma(state[i], k_t[i], kv_mem);
        }
        kv_mem = simd_sum(kv_mem);

        const float delta =
            (vc[(hb * l + t) * dv + col] - kv_mem) * beta_s[hb * l + t];

        float o = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] = fma(k_t[i], delta, state[i]);
            o = fma(state[i], q_t[i], o);
        }
        o = simd_sum(o);
        if (simd_lane == 0) {
            cap_delta[(hb * l + t) * dv + col] = delta;
            // o_pre is [b, l, value_dim].
            o_pre[((ulong)(bh / heads) * l + t) * value_dim + h * dv + col] = o;
        }
    }

    for (uint i = 0; i < n_per_lane; i++) {
        s_out[lane * n_per_lane + i] = state[i];
    }
}

// ---------------------------------------------------------------------------
// decode: fused prep+core for the single-token step (l = 1). The v1 decode
// kernel runs `heads` threadgroups and a threadgroup cannot span GPU cores,
// so it caps at 16 of ~40 cores regardless of thread count; the per-head
// RMSNorm is what pins that geometry. Splitting the norm off instead lets
// the state stream run at the core grid's full occupancy: grid
// (dv/4, batch*heads) TGs of (32, 4), one simdgroup per value column, lane
// owns 4 contiguous dk slots of the transposed state column in registers.
// All reductions are simd-scope (per-lane partials + simd_sum) — no
// threadgroup barriers, no staged kn/qn/vc round-trip; conv and gate
// scalars are recomputed redundantly per lane (a handful of MACs). Pairs
// with gated_delta_v2_epilogue_bf16 at l = 1 for the norm + z gate.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_decode_bf16(
    device const bfloat *proj      [[buffer(0)]],
    device const bfloat *ba        [[buffer(1)]],
    device const bfloat *conv_in   [[buffer(2)]],
    device const float  *state_in  [[buffer(3)]],
    device const bfloat *conv_w    [[buffer(4)]],
    device const float  *dt_bias   [[buffer(5)]],
    device const float  *a_log_exp [[buffer(6)]],
    device bfloat       *conv_out  [[buffer(7)]],
    device float        *state_out [[buffer(8)]],
    device float        *o_pre     [[buffer(9)]],
    constant uint  &heads      [[buffer(10)]],
    constant uint  &dk         [[buffer(11)]],
    constant uint  &dv         [[buffer(12)]],
    constant uint  &conv_dim   [[buffer(13)]],
    constant uint  &key_dim    [[buffer(14)]],
    constant uint  &value_dim  [[buffer(15)]],
    constant uint  &ksz        [[buffer(16)]],
    constant float &l2_eps     [[buffer(17)]],
    uint2 tg        [[threadgroup_position_in_grid]],
    uint2 tp        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]) {
    const uint bh = tg.y;
    const uint bi = bh / heads;
    const uint h = bh % heads;
    const uint col = tg.x * 4 + tp.y;
    const uint lane = tp.x;
    const uint n_per_lane = dk / 32; // 4 at dk=128

    device const bfloat *proj_b = proj + (ulong)bi * (conv_dim + value_dim);
    device const bfloat *ba_b = ba + (ulong)bi * 2 * heads;
    device const bfloat *conv_in_b = conv_in + (ulong)bi * conv_dim * ksz;
    device bfloat *conv_out_b = conv_out + (ulong)bi * conv_dim * ksz;

    // Conv + silu for this lane's q/k channel slots. Window = last ksz
    // inputs: retained conv_in tail + the new token's raw input.
    float qc[4];
    float kc[4];
    for (uint i = 0; i < n_per_lane; i++) {
        const uint slot = lane * n_per_lane + i;
        const uint chq = h * dk + slot;
        const uint chk = key_dim + h * dk + slot;
        float accq = 0.0f;
        float acck = 0.0f;
        for (uint t = 0; t + 1 < ksz; t++) {
            accq += float(conv_w[chq * ksz + t]) * float(conv_in_b[chq * ksz + t + 1]);
            acck += float(conv_w[chk * ksz + t]) * float(conv_in_b[chk * ksz + t + 1]);
        }
        accq += float(conv_w[chq * ksz + ksz - 1]) * float(proj_b[chq]);
        acck += float(conv_w[chk * ksz + ksz - 1]) * float(proj_b[chk]);
        qc[i] = accq / (1.0f + metal::precise::exp(-accq));
        kc[i] = acck / (1.0f + metal::precise::exp(-acck));
    }
    // This simdgroup's value column channel (redundant per lane).
    const uint chv = 2 * key_dim + h * dv + col;
    float accv = 0.0f;
    for (uint t = 0; t + 1 < ksz; t++) {
        accv += float(conv_w[chv * ksz + t]) * float(conv_in_b[chv * ksz + t + 1]);
    }
    accv += float(conv_w[chv * ksz + ksz - 1]) * float(proj_b[chv]);
    const float v_c = accv / (1.0f + metal::precise::exp(-accv));

    // Gate scalars (redundant per lane; identical inputs).
    const float b_in = float(ba_b[h]);
    const float a_in = float(ba_b[heads + h]);
    const float g = -a_log_exp[h]
        * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
    const float decay = metal::precise::exp(g);
    const float beta = 1.0f / (1.0f + metal::precise::exp(-b_in));

    // l2 norms over dk via simd_sum of per-lane partials.
    float q2p = 0.0f;
    float k2p = 0.0f;
    for (uint i = 0; i < n_per_lane; i++) {
        q2p = fma(qc[i], qc[i], q2p);
        k2p = fma(kc[i], kc[i], k2p);
    }
    const float q2 = simd_sum(q2p);
    const float k2 = simd_sum(k2p);
    const float qs = metal::precise::powr(q2 + l2_eps, -0.5f)
        * metal::precise::rsqrt(float(dk));
    const float ks = metal::precise::powr(k2 + l2_eps, -0.5f);
    for (uint i = 0; i < n_per_lane; i++) {
        qc[i] *= qs;
        kc[i] *= ks;
    }

    // Delta rule on this column; state slots live in registers.
    device const float *s_in = state_in + ((ulong)bh * dv + col) * dk;
    device float *s_out = state_out + ((ulong)bh * dv + col) * dk;
    float state[4];
    float kvp = 0.0f;
    for (uint i = 0; i < n_per_lane; i++) {
        state[i] = s_in[lane * n_per_lane + i] * decay;
        kvp = fma(state[i], kc[i], kvp);
    }
    const float kv_mem = simd_sum(kvp);
    const float delta = (v_c - kv_mem) * beta;
    float op = 0.0f;
    for (uint i = 0; i < n_per_lane; i++) {
        state[i] = fma(kc[i], delta, state[i]);
        op = fma(state[i], qc[i], op);
        s_out[lane * n_per_lane + i] = state[i];
    }
    const float o = simd_sum(op);
    if (simd_lane == 0) {
        o_pre[(ulong)bi * value_dim + h * dv + col] = o;
    }

    // Rolling conv window write-back, single writer per channel: the
    // (tg.x == 0, tp.y == 0) simdgroup covers this head's q/k channels
    // (lane-owned slots); each column's simdgroup lane 0 covers its v
    // channel.
    if (tg.x == 0 && tp.y == 0) {
        for (uint i = 0; i < n_per_lane; i++) {
            const uint slot = lane * n_per_lane + i;
            const uint chans2[2] = {h * dk + slot, key_dim + h * dk + slot};
            for (uint c = 0; c < 2; c++) {
                const uint ch = chans2[c];
                for (uint t = 0; t + 1 < ksz; t++) {
                    conv_out_b[ch * ksz + t] = conv_in_b[ch * ksz + t + 1];
                }
                conv_out_b[ch * ksz + ksz - 1] = proj_b[ch];
            }
        }
    }
    if (simd_lane == 0) {
        for (uint t = 0; t + 1 < ksz; t++) {
            conv_out_b[chv * ksz + t] = conv_in_b[chv * ksz + t + 1];
        }
        conv_out_b[chv * ksz + ksz - 1] = proj_b[chv];
    }
}

// ---------------------------------------------------------------------------
// prep_tree: tree-verify variant of prep. The flattened chunk is
// [anchor, a_1..a_w | b_1..b_w]: main segment rows [0, seg1), alternate
// segment rows [seg1, seg1+alt_len), branching after row branch_after-1 of
// the main segment. One dispatch stages BOTH segments: the alternate's conv
// window re-seeds from the branch ancestry (retained conv_in tail plus main
// rows < branch_after — pure row arithmetic, no sequential dependency), and
// the log-decay cumsum restarts at the segment boundary so each segment's
// captures are identical to what its own separate dispatch would emit (the
// host rollback math is unchanged). Single stream; grid `heads` TGs x dk.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_prep_tree_bf16(
    device const bfloat *proj      [[buffer(0)]],
    device const bfloat *ba        [[buffer(1)]],
    device const bfloat *conv_in   [[buffer(2)]],
    device const bfloat *conv_w    [[buffer(3)]],
    device const float  *dt_bias   [[buffer(4)]],
    device const float  *a_log_exp [[buffer(5)]],
    device bfloat       *conv_out  [[buffer(6)]],
    device float        *kn        [[buffer(7)]],
    device float        *qn        [[buffer(8)]],
    device float        *vc        [[buffer(9)]],
    device float        *g_step    [[buffer(10)]],
    device float        *beta_s    [[buffer(11)]],
    device float        *cap_gcs   [[buffer(12)]],
    constant uint  &heads        [[buffer(13)]],
    constant uint  &dk           [[buffer(14)]],
    constant uint  &dv           [[buffer(15)]],
    constant uint  &conv_dim     [[buffer(16)]],
    constant uint  &key_dim      [[buffer(17)]],
    constant uint  &value_dim    [[buffer(18)]],
    constant uint  &ksz          [[buffer(19)]],
    constant uint  &seg1         [[buffer(20)]],
    constant uint  &alt_len      [[buffer(21)]],
    constant uint  &branch_after [[buffer(22)]],
    constant float &l2_eps       [[buffer(23)]],
    uint h          [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint l_total = seg1 + alt_len;
    const uint row_stride = conv_dim + value_dim;
    const ulong hb = (ulong)h;

    threadgroup float q_raw[GD2_MAX_L * 128];
    threadgroup float k_raw[GD2_MAX_L * 128];
    threadgroup float scratch[4];
    const uint n_simd_groups = dk / 32;

    const uint chans[3] = {
        h * dk + tid,
        key_dim + h * dk + tid,
        2 * key_dim + h * dv + tid,
    };
    for (uint c = 0; c < 3; c++) {
        const uint ch = chans[c];
        float win[GD2_MAX_KSZ];
        // Main segment: window slides from the live conv_in exactly like prep.
        for (uint t = 0; t + 1 < ksz; t++) {
            win[t] = float(conv_in[ch * ksz + t + 1]);
        }
        for (uint pos = 0; pos < seg1; pos++) {
            win[ksz - 1] = float(proj[pos * row_stride + ch]);
            float acc = 0.0f;
            for (uint t = 0; t < ksz; t++) {
                acc += float(conv_w[ch * ksz + t]) * win[t];
            }
            const float y = acc / (1.0f + metal::precise::exp(-acc));
            if (c == 0) q_raw[pos * dk + tid] = y;
            else if (c == 1) k_raw[pos * dk + tid] = y;
            else vc[(hb * l_total + pos) * dv + tid] = y;
            for (uint t = 0; t + 1 < ksz; t++) win[t] = win[t + 1];
        }
        // Live conv window continues from the MAIN segment's end (the host
        // keeps main as the live state; the alternate's window is capture-
        // reconstructed on rollback if it wins).
        for (uint t = 0; t < ksz; t++) {
            const int src = int(seg1) - int(ksz) + int(t);
            float val;
            if (src >= 0) {
                val = float(proj[uint(src) * row_stride + ch]);
            } else {
                val = float(conv_in[ch * ksz + uint(int(ksz) + src)]);
            }
            conv_out[ch * ksz + t] = bfloat(val);
        }
        // Alternate segment: re-seed the window with the last ksz-1 inputs
        // before b_1 = tail of [conv_in slots 1..ksz-1, main rows 0..branch_after).
        for (uint t = 0; t + 1 < ksz; t++) {
            const uint j = branch_after + t; // index into that combined sequence
            win[t] = j < ksz - 1
                ? float(conv_in[ch * ksz + 1 + j])
                : float(proj[(j - (ksz - 1)) * row_stride + ch]);
        }
        for (uint pos = seg1; pos < l_total; pos++) {
            win[ksz - 1] = float(proj[pos * row_stride + ch]);
            float acc = 0.0f;
            for (uint t = 0; t < ksz; t++) {
                acc += float(conv_w[ch * ksz + t]) * win[t];
            }
            const float y = acc / (1.0f + metal::precise::exp(-acc));
            if (c == 0) q_raw[pos * dk + tid] = y;
            else if (c == 1) k_raw[pos * dk + tid] = y;
            else vc[(hb * l_total + pos) * dv + tid] = y;
            for (uint t = 0; t + 1 < ksz; t++) win[t] = win[t + 1];
        }
    }

    // Gate scalars + decay cumsum; the running sum restarts at the segment
    // boundary so cap_gcs is per-segment, matching two-dispatch semantics.
    if (tid == 0) {
        float run = 0.0f;
        for (uint pos = 0; pos < l_total; pos++) {
            if (pos == seg1) run = 0.0f;
            const float b_in = float(ba[pos * 2 * heads + h]);
            const float a_in = float(ba[pos * 2 * heads + heads + h]);
            const float g = -a_log_exp[h]
                * metal::precise::log(1.0f + metal::precise::exp(a_in + dt_bias[h]));
            g_step[hb * l_total + pos] = g;
            run += g;
            cap_gcs[hb * l_total + pos] = run;
            beta_s[hb * l_total + pos] = 1.0f / (1.0f + metal::precise::exp(-b_in));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint pos = 0; pos < l_total; pos++) {
        const float qv = q_raw[pos * dk + tid];
        const float kv = k_raw[pos * dk + tid];
        const float q2 =
            tg_sum_128(qv * qv, scratch, tid, simd_group, simd_lane, n_simd_groups);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k2 =
            tg_sum_128(kv * kv, scratch, tid, simd_group, simd_lane, n_simd_groups);
        qn[(hb * l_total + pos) * dk + tid] = qv
            * metal::precise::powr(q2 + l2_eps, -0.5f)
            * metal::precise::rsqrt(float(dk));
        kn[(hb * l_total + pos) * dk + tid] = kv
            * metal::precise::powr(k2 + l2_eps, -0.5f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// core_tree: tree-verify variant of core. Registers carry the state through
// the main segment; at the branch point each lane snapshots its slots (also
// written to state_mid — the alternate capture's S0 for host rollback), and
// after the main segment's end state is written the registers reload the
// snapshot and the alternate segment runs in the same dispatch. The branch
// seed is bit-for-bit the kernel's own recurrence at the branch position —
// the closed form the host used to recompute. Single stream.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_core_tree(
    device const float *state_in  [[buffer(0)]],
    device float       *state_out [[buffer(1)]],
    device const float *kn        [[buffer(2)]],
    device const float *qn        [[buffer(3)]],
    device const float *vc        [[buffer(4)]],
    device const float *g_step    [[buffer(5)]],
    device const float *beta_s    [[buffer(6)]],
    device float       *cap_delta [[buffer(7)]],
    device float       *o_pre     [[buffer(8)]],
    device float       *state_mid [[buffer(9)]],
    constant uint &heads        [[buffer(10)]],
    constant uint &dk           [[buffer(11)]],
    constant uint &dv           [[buffer(12)]],
    constant uint &value_dim    [[buffer(13)]],
    constant uint &seg1         [[buffer(14)]],
    constant uint &alt_len      [[buffer(15)]],
    constant uint &branch_after [[buffer(16)]],
    uint2 tg        [[threadgroup_position_in_grid]],
    uint2 tp        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]) {
    const uint bh = tg.y; // head index; single stream
    const uint h = bh % heads;
    const uint col = tg.x * 4 + tp.y;
    const uint lane = tp.x;
    const uint l_total = seg1 + alt_len;
    const uint n_per_lane = dk / 32;

    device const float *s_in = state_in + ((ulong)bh * dv + col) * dk;
    device float *s_out = state_out + ((ulong)bh * dv + col) * dk;
    device float *s_mid = state_mid + ((ulong)bh * dv + col) * dk;

    float state[4];
    float state_br[4];
    for (uint i = 0; i < n_per_lane; i++) {
        state[i] = s_in[lane * n_per_lane + i];
        state_br[i] = state[i];
    }

    const ulong hb = (ulong)bh;
    for (uint t = 0; t < seg1; t++) {
        const float decay = metal::precise::exp(g_step[hb * l_total + t]);
        device const float *k_t = kn + (hb * l_total + t) * dk + lane * n_per_lane;
        device const float *q_t = qn + (hb * l_total + t) * dk + lane * n_per_lane;
        float kv_mem = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] *= decay;
            kv_mem = fma(state[i], k_t[i], kv_mem);
        }
        kv_mem = simd_sum(kv_mem);

        const float delta =
            (vc[(hb * l_total + t) * dv + col] - kv_mem) * beta_s[hb * l_total + t];

        float o = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] = fma(k_t[i], delta, state[i]);
            o = fma(state[i], q_t[i], o);
        }
        o = simd_sum(o);
        if (simd_lane == 0) {
            cap_delta[(hb * l_total + t) * dv + col] = delta;
            o_pre[(ulong)t * value_dim + h * dv + col] = o;
        }
        if (t + 1 == branch_after) {
            for (uint i = 0; i < n_per_lane; i++) {
                state_br[i] = state[i];
            }
        }
    }

    // Main-segment end state = the live state the host keeps.
    for (uint i = 0; i < n_per_lane; i++) {
        s_out[lane * n_per_lane + i] = state[i];
    }
    // Branch-point state = the alternate capture's S0.
    for (uint i = 0; i < n_per_lane; i++) {
        s_mid[lane * n_per_lane + i] = state_br[i];
        state[i] = state_br[i];
    }

    for (uint t = seg1; t < l_total; t++) {
        const float decay = metal::precise::exp(g_step[hb * l_total + t]);
        device const float *k_t = kn + (hb * l_total + t) * dk + lane * n_per_lane;
        device const float *q_t = qn + (hb * l_total + t) * dk + lane * n_per_lane;
        float kv_mem = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] *= decay;
            kv_mem = fma(state[i], k_t[i], kv_mem);
        }
        kv_mem = simd_sum(kv_mem);

        const float delta =
            (vc[(hb * l_total + t) * dv + col] - kv_mem) * beta_s[hb * l_total + t];

        float o = 0.0f;
        for (uint i = 0; i < n_per_lane; i++) {
            state[i] = fma(k_t[i], delta, state[i]);
            o = fma(state[i], q_t[i], o);
        }
        o = simd_sum(o);
        if (simd_lane == 0) {
            cap_delta[(hb * l_total + t) * dv + col] = delta;
            o_pre[(ulong)t * value_dim + h * dv + col] = o;
        }
    }
    // The alternate segment's end state is never installed (rollback
    // reconstructs the winning prefix from captures), so it is not written.
}

// ---------------------------------------------------------------------------
// rollback state reconstruction: one dispatch replaces the per-layer f32
// broadcast/exp/GEMM chain (measured ~20% of a spec round at m=4):
//   out[h, a, b] = exp(gcs[h, j]) * s0[h, a, b]
//                + sum_{i <= j} F[h, i, a] * exp(gcs[h, j] - gcs[h, i]) * G[h, i, b]
// with j = prefix - 1; (F, G) = (kc, delta) for the v1 layout (a = k_dim,
// b = v_dim) and (delta, kc) for the v2 transposed layout — the formula is
// symmetric, so the host just swaps the factor pointers. F/G/gcs index with
// the capture's FULL chunk stride (c_total), so callers pass untrimmed
// tensors and only rows i < prefix are read. Grid: (da*db, heads*batch).
// ---------------------------------------------------------------------------
#define GDN_RECONSTRUCT(NAME, PTR_T, STORE)                                   \
kernel void NAME(                                                             \
    device const float *s0   [[buffer(0)]],                                   \
    device const float *fmat [[buffer(1)]],                                   \
    device const float *gmat [[buffer(2)]],                                   \
    device const float *gcs  [[buffer(3)]],                                   \
    device PTR_T       *out  [[buffer(4)]],                                   \
    constant uint &da       [[buffer(5)]],                                    \
    constant uint &db       [[buffer(6)]],                                    \
    constant uint &prefix   [[buffer(7)]],                                    \
    constant uint &c_total  [[buffer(8)]],                                    \
    uint2 tid [[thread_position_in_grid]]) {                                  \
    const uint ab = tid.x;                                                    \
    const uint hb = tid.y;                                                    \
    if (ab >= da * db) {                                                      \
        return;                                                               \
    }                                                                         \
    const uint a = ab / db;                                                   \
    const uint b = ab % db;                                                   \
    device const float *g_row = gcs + (ulong)hb * c_total;                    \
    const float gj = g_row[prefix - 1];                                       \
    float acc = metal::precise::exp(gj) * s0[((ulong)hb * da + a) * db + b];  \
    for (uint i = 0; i < prefix; i++) {                                       \
        const float rel = metal::precise::exp(gj - g_row[i]);                 \
        const float f = fmat[((ulong)hb * c_total + i) * da + a];             \
        const float g = gmat[((ulong)hb * c_total + i) * db + b];             \
        acc = fma(f * rel, g, acc);                                           \
    }                                                                         \
    out[((ulong)hb * da + a) * db + b] = STORE(acc);                          \
}

// bf16-rounded f32 store: exactly the values the bf16 kernel plus a
// cast_bf16_f32 dispatch would produce, minus the cast's dispatch + traffic.
#define GDN_BF16R(x) float(bfloat(x))

GDN_RECONSTRUCT(gated_delta_v2_reconstruct_f32, float, float)
GDN_RECONSTRUCT(gated_delta_v2_reconstruct_bf16, bfloat, bfloat)
GDN_RECONSTRUCT(gated_delta_v2_reconstruct_bf16r_f32, float, GDN_BF16R)

// ---------------------------------------------------------------------------
// epilogue: grid (heads * batch) TGs x dv threads. Group RMSNorm over the
// head's output row + silu(z) gating, per position.
// ---------------------------------------------------------------------------
kernel void gated_delta_v2_epilogue_bf16(
    device const float  *o_pre  [[buffer(0)]],
    device const bfloat *proj   [[buffer(1)]],
    device const float  *norm_w [[buffer(2)]],
    device bfloat       *out    [[buffer(3)]],
    constant uint  &heads     [[buffer(4)]],
    constant uint  &dv        [[buffer(5)]],
    constant uint  &conv_dim  [[buffer(6)]],
    constant uint  &value_dim [[buffer(7)]],
    constant uint  &seq_len   [[buffer(8)]],
    constant float &norm_eps  [[buffer(9)]],
    uint bh         [[threadgroup_position_in_grid]],
    uint tid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint bi = bh / heads;
    const uint h = bh % heads;
    const uint l = seq_len;
    const uint row_stride = conv_dim + value_dim;
    device const float *o_pre_b = o_pre + (ulong)bi * l * value_dim;
    device const bfloat *proj_b = proj + (ulong)bi * l * row_stride;
    device bfloat *out_b = out + (ulong)bi * l * value_dim;

    threadgroup float scratch[4];
    const uint n_simd_groups = dv / 32;

    for (uint t = 0; t < l; t++) {
        const float o = o_pre_b[(ulong)t * value_dim + h * dv + tid];
        const float o2 =
            tg_sum_128(o * o, scratch, tid, simd_group, simd_lane, n_simd_groups);
        const float inv_rms =
            metal::precise::powr(o2 / float(dv) + norm_eps, -0.5f);
        const float z_in = float(proj_b[t * row_stride + conv_dim + h * dv + tid]);
        const float z_silu = z_in / (1.0f + metal::precise::exp(-z_in));
        out_b[(ulong)t * value_dim + h * dv + tid] =
            bfloat(o * inv_rms * norm_w[tid] * z_silu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
