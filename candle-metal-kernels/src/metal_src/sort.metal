// Imported from https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal
#include <metal_stdlib>
using namespace metal;

#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define SORT_ASC 1
#define SORT_DESC 0

template<int order, typename T>
METAL_FUNC void argsort(
        device const T        * x,
        device       uint32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup uint32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const T        * x_row   = x + row * ncols;
    threadgroup uint32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == SORT_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ARGSORT(T, RUST_T) \
kernel void asort_asc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    constant     int64_t & ncols_pad, \
    threadgroup uint32_t  * shared_values [[threadgroup(0)]], \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_ASC, T>(x, dst, ncols, ncols_pad, shared_values, tgpig, tpitg); \
} \
kernel void asort_desc_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    constant     int64_t & ncols_pad, \
    threadgroup uint32_t  * shared_values [[threadgroup(0)]], \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    argsort<SORT_DESC, T>(x, dst, ncols, ncols_pad, shared_values, tgpig, tpitg); \
} \

ARGSORT(float, f32)
ARGSORT(half, f16)
ARGSORT(uint8_t, u8)
ARGSORT(uint32_t, u32)

#if __METAL_VERSION__ >= 220
ARGSORT(int64_t, i64)
#endif
#if defined(__HAVE_BFLOAT__)
ARGSORT(bfloat, bf16)
#endif

// ---------------------------------------------------------------------------
// Per-row top-k for arbitrary row widths. The argsort above is capped at
// ncols_pad <= max threads per threadgroup; RT-DETR-class rows (tens of
// thousands of columns) need this instead. One threadgroup per row walks the
// row in TOPK_TILE-wide tiles: each tile is bitonic-sorted descending in
// shared memory, then its top k merge with the running top-k carry. The
// carry (descending) concatenated with the tile's reversed top-k (ascending)
// is a bitonic sequence, so each merge is a single log2(2k)-stage bitonic
// merge rather than a full sort. Requires k <= TOPK_TILE and k a power of
// two (the host wrapper pads k up and the caller narrows the result).
//
// Values compare as float (order-preserving for half/bfloat); rows
// containing NaN have unspecified NaN placement, like the argsort. Output is
// the k column indices of the largest values, descending by value. Ties are
// broken by whichever candidate a merge stage keeps — no index-order
// guarantee, again like the argsort.

#define TOPK_TILE 1024

METAL_FUNC void topk_cmp_swap_desc(
        threadgroup float    *sv,
        threadgroup uint32_t *si,
        uint a,
        uint b
) {
    if (sv[a] < sv[b]) {
        SWAP(sv[a], sv[b]);
        SWAP(si[a], si[b]);
    }
}

template<typename T>
METAL_FUNC void topk_row(
        device const T        *x_row,
        device       uint32_t *dst_row,
        const int64_t ncols,
        const int64_t k,
        threadgroup float     *sv,   // [TOPK_TILE + 2 * k]
        threadgroup uint32_t  *si,   // [TOPK_TILE + 2 * k]
        uint tid
) {
    const uint npad = TOPK_TILE;
    // Carry region: sv[npad .. npad + k), maintained descending; the merge
    // scratch extends it to sv[npad .. npad + 2k).
    for (uint i = tid; i < uint(2 * k); i += npad) {
        sv[npad + i] = -INFINITY;
        si[npad + i] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ntiles = uint((ncols + npad - 1) / npad);
    for (uint t = 0; t < ntiles; t++) {
        // Load one tile; out-of-row slots sort to the bottom.
        const uint col = t * npad + tid;
        sv[tid] = col < uint(ncols) ? float(x_row[col]) : -INFINITY;
        si[tid] = col;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Bitonic sort of the tile, descending, one element per thread.
        for (uint kk = 2; kk <= npad; kk *= 2) {
            for (uint j = kk / 2; j > 0; j /= 2) {
                const uint ixj = tid ^ j;
                if (ixj > tid) {
                    const bool up = (tid & kk) == 0;
                    if (up ? (sv[tid] < sv[ixj]) : (sv[tid] > sv[ixj])) {
                        SWAP(sv[tid], sv[ixj]);
                        SWAP(si[tid], si[ixj]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // The tile's top k (descending at sv[0..k)) reversed into the second
        // half of the merge buffer; [carry desc][tile asc] is bitonic.
        if (tid < uint(k)) {
            sv[npad + uint(k) + tid] = sv[uint(k) - 1 - tid];
            si[npad + uint(k) + tid] = si[uint(k) - 1 - tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Bitonic merge of the 2k candidates, descending; the new carry is
        // the first k.
        for (uint s = uint(k); s > 0; s >>= 1) {
            if (tid < uint(k)) {
                const uint i = (tid / s) * 2 * s + (tid % s);
                topk_cmp_swap_desc(sv, si, npad + i, npad + i + s);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid < uint(k)) {
        dst_row[tid] = si[npad + tid];
    }
}

#define TOPK(T, RUST_T) \
kernel void topk_##RUST_T( \
    device const T        * x, \
    device       uint32_t * dst, \
    constant     int64_t & ncols, \
    constant     int64_t & k, \
    threadgroup float     * shared_values  [[threadgroup(0)]], \
    threadgroup uint32_t  * shared_indices [[threadgroup(1)]], \
    uint3 tgpig[[threadgroup_position_in_grid]], \
    uint3 tpitg[[thread_position_in_threadgroup]] \
) {  \
    topk_row<T>(x + tgpig[0] * ncols, dst + tgpig[0] * k, ncols, k, \
                shared_values, shared_indices, tpitg[0]); \
} \

TOPK(float, f32)
TOPK(half, f16)
#if defined(__HAVE_BFLOAT__)
TOPK(bfloat, bf16)
#endif
