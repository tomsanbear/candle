// Fused attention-input preparation for hybrid-attention decode.
//
// The unfused chain per full-attention layer is ten dispatches of tiny grids
// (2 strided-narrow copies + 2 rmsnorms + 2 partial ropes + a v copy + 2
// cache slice_sets + a contiguous fix-up), each a serialization point on the
// GPU timeline. These kernels collapse that to two dispatches:
//
//   attn_q_prep_bf16:  head-norm + partial rope on q, read straight from the
//                      packed qkv projection row, written in (h, t, d) layout
//                      (the transpose the attention kernels want).
//   attn_kv_prep_bf16: head-norm + partial rope on k plus the raw v copy,
//                      both written DIRECTLY into the KV-cache slots at the
//                      append position — no intermediate k/v tensors at all.
//
// Bitwise contract: outputs must match the unfused composition exactly,
// mirroring rms_norm (reduce.metal) semantics precisely — f32 sum-of-squares
// accumulated per thread in load order, folded with the same index pairing
// as block_reducer (s = 64, 32, 16, ..., 1 combines partial[i] with
// partial[i+s]; the shared-memory tree here pairs identically to its
// simd_shuffle tail), `rsqrt(fast::divide(sum, d) + eps)` for the scale, and
// the OUTPUT MATH IN BF16: the scale is rounded to bf16 first, then
// (x * scale) * alpha with per-op bf16 rounding. The rope reproduces
// rope_partial's per-op bf16 arithmetic on those rounded outputs. Copies are
// byte-exact by construction. The contract is enforced bit-for-bit by
// tests::attn_prep_matches_unfused_chain against the production kernels.

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)

// One threadgroup per (head, token) row; matches rmsnorm's block width for
// d == 256 so the reduction order (and therefore the bits) are identical.
constant constexpr uint ATTN_PREP_BLOCK = 128;
constant constexpr uint ATTN_PREP_MAX_D = 256;

// norm(src_row) * alpha, then partial rope at one position, into dst_row.
METAL_FUNC void attn_row_norm_rope(
    const device bfloat *src_row,
    const device bfloat *alpha,
    const device bfloat *cos_row,
    const device bfloat *sin_row,
    device bfloat *dst_row,
    uint d,
    uint rd,
    float eps,
    uint tid,
    threadgroup float *red,
    threadgroup bfloat *nbuf
) {
    float tmp = 0;
    for (uint j = tid; j < d; j += ATTN_PREP_BLOCK) {
        float x = float(src_row[j]);
        tmp = tmp + x * x;
    }
    red[tid] = tmp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_PREP_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bfloat scale = static_cast<bfloat>(rsqrt(fast::divide(red[0], float(d)) + eps));
    for (uint j = tid; j < d; j += ATTN_PREP_BLOCK) {
        bfloat val = src_row[j] * scale;
        val *= alpha[j];
        nbuf[j] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint half_rd = rd / 2;
    for (uint j = tid; j < d; j += ATTN_PREP_BLOCK) {
        bfloat out;
        if (j < half_rd) {
            out = nbuf[j] * cos_row[j] - nbuf[j + half_rd] * sin_row[j];
        } else if (j < rd) {
            uint i = j - half_rd;
            out = nbuf[i] * sin_row[i] + nbuf[j] * cos_row[i];
        } else {
            out = nbuf[j];
        }
        dst_row[j] = out;
    }
}

// Grid: heads * t_len threadgroups of ATTN_PREP_BLOCK threads. The q head
// row sits inside the packed [q | gate] block, hence head_in_stride = 2*d.
kernel void attn_q_prep_bf16(
    constant uint &t_len,
    constant uint &d,
    constant uint &rd,
    constant uint &qkv_row_stride,
    constant uint &head_in_stride,
    constant uint &q_base,
    constant uint &pos_base,
    constant float &eps,
    device const bfloat *qkv,
    device const bfloat *alpha,
    device const bfloat *cos_t,
    device const bfloat *sin_t,
    device bfloat *dst,
    uint gid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    threadgroup float red[ATTN_PREP_BLOCK];
    threadgroup bfloat nbuf[ATTN_PREP_MAX_D];
    uint h = gid / t_len;
    uint t = gid - h * t_len;
    const device bfloat *src_row = qkv + t * qkv_row_stride + q_base + h * head_in_stride;
    uint pos = pos_base + t;
    const device bfloat *cos_row = cos_t + pos * (rd / 2);
    const device bfloat *sin_row = sin_t + pos * (rd / 2);
    device bfloat *dst_row = dst + (h * t_len + t) * d;
    attn_row_norm_rope(src_row, alpha, cos_row, sin_row, dst_row, d, rd, eps, tid, red, nbuf);
}

// Grid: kv_heads * t_len threadgroups. k is normed + roped into its cache
// slot; v is copied raw into its slot. cache layout is (kv_heads, cap, d)
// contiguous, rows written at write_pos + t.
kernel void attn_kv_prep_bf16(
    constant uint &t_len,
    constant uint &d,
    constant uint &rd,
    constant uint &qkv_row_stride,
    constant uint &k_base,
    constant uint &v_base,
    constant uint &cache_cap,
    constant uint &write_pos,
    constant uint &pos_base,
    constant float &eps,
    device const bfloat *qkv,
    device const bfloat *alpha,
    device const bfloat *cos_t,
    device const bfloat *sin_t,
    device bfloat *cache_k,
    device bfloat *cache_v,
    uint gid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]]
) {
    threadgroup float red[ATTN_PREP_BLOCK];
    threadgroup bfloat nbuf[ATTN_PREP_MAX_D];
    uint h = gid / t_len;
    uint t = gid - h * t_len;
    uint slot = h * cache_cap + write_pos + t;

    const device bfloat *v_row = qkv + t * qkv_row_stride + v_base + h * d;
    device bfloat *v_dst = cache_v + slot * d;
    for (uint j = tid; j < d; j += ATTN_PREP_BLOCK) {
        v_dst[j] = v_row[j];
    }

    const device bfloat *k_row = qkv + t * qkv_row_stride + k_base + h * d;
    uint pos = pos_base + t;
    const device bfloat *cos_row = cos_t + pos * (rd / 2);
    const device bfloat *sin_row = sin_t + pos * (rd / 2);
    device bfloat *k_dst = cache_k + slot * d;
    attn_row_norm_rope(k_row, alpha, cos_row, sin_row, k_dst, d, rd, eps, tid, red, nbuf);
}

#endif
