#include <metal_stdlib>
using namespace metal;

// Fused DSpark Markov-chain proposal steps.
//
// The legacy chain runs gamma data-dependent iterations of ~5-6 tiny kernels
// (gather -> rank-R gemv over the draft vocab -> add base logits -> f32 cast
// -> argmax -> id remap), a serial dependency chain measured at ~1.17 ms per
// step on M3-class hardware — launch latency, not arithmetic. These kernels
// collapse each step to two dispatches: a grid-wide fused
// gather+gemv+add+partial-argmax, and a single-threadgroup reduce that remaps
// the winner and feeds the next step's token through a device-side chain
// buffer (no host involvement between steps).
//
// Argmax tie rule: maximum value, LOWEST index (thread-strided row loops use
// strict '>' so each thread keeps its earliest row; cross-thread and
// cross-threadgroup reductions break ties by smaller index). Logit values
// mirror the legacy dtype path (f32 accumulate -> bf16 round, add in f32 ->
// bf16 round -> f32 compare) but the accumulation ORDER differs from the qmv
// kernel, so ulp-level ties may resolve differently — draft tokens never
// affect committed output (verification is exact), and the parity gate
// counts any divergence.

constant uint MARKOV_TPG = 256;

struct markov_partial {
    float val;
    uint idx;
};

// ggml block_q8_0: f16 scale + 32 int8 quants, 34 bytes packed.
constant uint Q8_BLOCK = 32;
constant uint Q8_BLOCK_BYTES = 34;

template <bool Q8>
void markov_step_partial_impl(
        device const bfloat * w1,          // [vocab_full, r] bf16
        device const uchar  * w2,          // Q8: q8_0 rows; else bf16 rows
        device const bfloat * base,        // [gamma, vd] bf16
        device const uint   * chain,       // [gamma+1]; chain[k] = this step's prev id
        device markov_partial * partials,  // [ntg]
        constant uint & k,
        constant uint & vd,
        constant uint & r,
        uint tgid,
        uint tid,
        uint ntg) {
    threadgroup float pe[MARKOV_TPG];
    threadgroup float red_val[MARKOV_TPG];
    threadgroup uint  red_idx[MARKOV_TPG];

    const uint prev = chain[k];
    if (tid < r) {
        pe[tid] = float(w1[prev * r + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float best = -INFINITY;
    uint best_idx = 0xFFFFFFFFu;
    for (uint row = tgid * MARKOV_TPG + tid; row < vd; row += ntg * MARKOV_TPG) {
        float acc = 0.0f;
        if (Q8) {
            device const uchar * rp = w2 + (ulong)row * (r / Q8_BLOCK) * Q8_BLOCK_BYTES;
            for (uint b = 0; b < r / Q8_BLOCK; ++b) {
                const float d = float(*(device const half *)rp);
                device const char * qs = (device const char *)(rp + 2);
                float bsum = 0.0f;
                for (uint j = 0; j < Q8_BLOCK; ++j) {
                    bsum += float(qs[j]) * pe[b * Q8_BLOCK + j];
                }
                acc += d * bsum;
                rp += Q8_BLOCK_BYTES;
            }
        } else {
            device const bfloat * wrow = (device const bfloat *)w2 + (ulong)row * r;
            for (uint j = 0; j < r; ++j) {
                acc += float(wrow[j]) * pe[j];
            }
        }
        // Mirror the legacy dtype path: gemv stores bf16, the add runs at
        // f32 and stores bf16, argmax compares the f32 upcast.
        float v = float(bfloat(acc));
        v = float(bfloat(v + float(base[(ulong)k * vd + row])));
        if (v > best) {
            best = v;
            best_idx = row;
        }
    }

    red_val[tid] = best;
    red_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = MARKOV_TPG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const bool take = red_val[tid + s] > red_val[tid]
                || (red_val[tid + s] == red_val[tid] && red_idx[tid + s] < red_idx[tid]);
            if (take) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[tgid].val = red_val[0];
        partials[tgid].idx = red_idx[0];
    }
}

#define MARKOV_PARTIAL(NAME, Q8)                                             \
[[host_name(#NAME)]]                                                         \
kernel void NAME(                                                            \
        device const bfloat * w1        [[buffer(0)]],                       \
        device const uchar  * w2        [[buffer(1)]],                       \
        device const bfloat * base      [[buffer(2)]],                       \
        device const uint   * chain     [[buffer(3)]],                       \
        device markov_partial * partials [[buffer(4)]],                      \
        constant uint & k               [[buffer(5)]],                       \
        constant uint & vd              [[buffer(6)]],                       \
        constant uint & r               [[buffer(7)]],                       \
        uint tgid [[threadgroup_position_in_grid]],                          \
        uint tid  [[thread_position_in_threadgroup]],                        \
        uint ntg  [[threadgroups_per_grid]]) {                               \
    markov_step_partial_impl<Q8>(w1, w2, base, chain, partials,              \
                                 k, vd, r, tgid, tid, ntg);                  \
}

MARKOV_PARTIAL(markov_step_partial_q8, true)
MARKOV_PARTIAL(markov_step_partial_bf16, false)

// Final reduce: pick the global winner (max value, lowest index), remap the
// draft-vocab index to a global token id, feed the chain, and stash this
// step's INPUT embedding row (w1[chain[k]]) for the confidence-head features.
[[host_name("markov_step_reduce")]]
kernel void markov_step_reduce(
        device const markov_partial * partials [[buffer(0)]],
        device const uint   * ids       [[buffer(1)]], // draft->global map; identity when use_remap == 0
        device uint         * chain     [[buffer(2)]], // writes chain[k+1]
        device uint         * tokens    [[buffer(3)]], // [gamma] global ids
        device const bfloat * w1        [[buffer(4)]],
        device bfloat       * prev_embs [[buffer(5)]], // [gamma, r]
        constant uint & k               [[buffer(6)]],
        constant uint & ntg             [[buffer(7)]],
        constant uint & r               [[buffer(8)]],
        constant uint & use_remap       [[buffer(9)]],
        uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float red_val[MARKOV_TPG];
    threadgroup uint  red_idx[MARKOV_TPG];

    float best = -INFINITY;
    uint best_idx = 0xFFFFFFFFu;
    for (uint i = tid; i < ntg; i += MARKOV_TPG) {
        const markov_partial p = partials[i];
        const bool take = p.val > best || (p.val == best && p.idx < best_idx);
        if (take) {
            best = p.val;
            best_idx = p.idx;
        }
    }
    red_val[tid] = best;
    red_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = MARKOV_TPG / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const bool take = red_val[tid + s] > red_val[tid]
                || (red_val[tid + s] == red_val[tid] && red_idx[tid + s] < red_idx[tid]);
            if (take) {
                red_val[tid] = red_val[tid + s];
                red_idx[tid] = red_idx[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint sub = red_idx[0];
    const uint global_id = use_remap != 0 ? ids[sub] : sub;
    if (tid == 0) {
        tokens[k] = global_id;
        chain[k + 1] = global_id;
    }
    // This step's feature embedding is the INPUT id's row.
    const uint prev = chain[k];
    if (tid < r) {
        prev_embs[(ulong)k * r + tid] = w1[prev * r + tid];
    }
}
