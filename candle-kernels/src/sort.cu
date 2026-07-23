// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0
#include "cuda_utils.cuh"
#include<stdint.h>

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<int order, typename T>
static __device__ void k_argsort(const T * x, uint32_t * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int row = blockIdx.x;

    const T * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices - each thread handles multiple elements if ncols_pad > blockDim.x
    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols && (order == SORT_ORDER_ASC ?
                                x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                        ) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols && (order == SORT_ORDER_ASC ?
                                x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                                x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                        ) {
                            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ASORT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_ASC>(x, dst, ncols, ncols_pad); \
} \
extern "C" __global__ void asort_desc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_DESC>(x, dst, ncols, ncols_pad); \
} \
 
#if __CUDA_ARCH__ >= 800
ASORT_OP(__nv_bfloat16, bf16)

// NOTE: No sort ops for f8
// ASORT_OP(__nv_fp8_e4m3, fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
ASORT_OP(__half, f16)
#endif

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)

// ---------------------------------------------------------------------------
// Per-row top-k (descending) for arbitrary row widths. The bitonic asort above
// sizes shared memory as ncols_pad * sizeof(u32), which exceeds the default
// 48 KiB block limit once ncols_pad >= 16384 (e.g. Heron RT-DETR encoder
// tokens ≈ 8400 → pad 16384 → 64 KiB). topk only needs the k largest, so a
// tile-and-merge algorithm keeps shared memory O(TILE + k) independent of
// row width — same design as the Metal topk kernel in sort.metal.
//
// One block per row, 1024 threads. Each tile is bitonic-sorted descending in
// shared memory; its top k merges with a running descending carry via a
// bitonic merge of 2k candidates. k must be a power of two and k <= TILE.
// Host pads k and narrows. NaN placement is unspecified (like asort).
// ---------------------------------------------------------------------------

#define TOPK_TILE 1024

template<typename T>
static inline __device__ void topk_swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

static inline __device__ void topk_cmp_swap_desc(
    float *sv, uint32_t *si, int a, int b
) {
    if (sv[a] < sv[b]) {
        topk_swap(sv[a], sv[b]);
        topk_swap(si[a], si[b]);
    }
}

template<typename T>
static __device__ void k_topk(
    const T *x, uint32_t *dst, const int ncols, const int k
) {
    // Dynamic shared: [TOPK_TILE + 2k] floats then [TOPK_TILE + 2k] u32s.
    extern __shared__ char smem[];
    const int npad = TOPK_TILE;
    const int carry_base = npad;
    float *sv = reinterpret_cast<float *>(smem);
    uint32_t *si = reinterpret_cast<uint32_t *>(
        smem + (npad + 2 * k) * (int)sizeof(float)
    );

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T *x_row = x + row * ncols;
    uint32_t *dst_row = dst + row * k;

    // Initialize carry to -inf so the first tile becomes the carry.
    for (int i = tid; i < 2 * k; i += npad) {
        sv[carry_base + i] = -INFINITY;
        si[carry_base + i] = 0xffffffffu;
    }
    __syncthreads();

    const int ntiles = (ncols + npad - 1) / npad;
    for (int t = 0; t < ntiles; t++) {
        const int col = t * npad + tid;
        if (col < ncols) {
            sv[tid] = (float)x_row[col];
            si[tid] = (uint32_t)col;
        } else {
            sv[tid] = -INFINITY;
            si[tid] = 0xffffffffu;
        }
        __syncthreads();

        // Bitonic sort of the tile, descending, one element per thread.
        for (int kk = 2; kk <= npad; kk *= 2) {
            for (int j = kk / 2; j > 0; j /= 2) {
                const int ixj = tid ^ j;
                if (ixj > tid) {
                    const bool up = (tid & kk) == 0;
                    if (up ? (sv[tid] < sv[ixj]) : (sv[tid] > sv[ixj])) {
                        topk_swap(sv[tid], sv[ixj]);
                        topk_swap(si[tid], si[ixj]);
                    }
                }
                __syncthreads();
            }
        }

        // Tile top-k (desc at sv[0..k)) reversed into second half of merge
        // buffer; [carry desc][tile asc] is bitonic of length 2k.
        if (tid < k) {
            sv[carry_base + k + tid] = sv[k - 1 - tid];
            si[carry_base + k + tid] = si[k - 1 - tid];
        }
        __syncthreads();

        for (int s = k; s > 0; s >>= 1) {
            if (tid < k) {
                const int i = (tid / s) * 2 * s + (tid % s);
                topk_cmp_swap_desc(sv, si, carry_base + i, carry_base + i + s);
            }
            __syncthreads();
        }
    }

    if (tid < k) {
        dst_row[tid] = si[carry_base + tid];
    }
}

#define TOPK_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void topk_##RUST_NAME( \
    const TYPENAME *x, uint32_t *dst, const int ncols, const int k \
) { \
    k_topk<TYPENAME>(x, dst, ncols, k); \
}

#if __CUDA_ARCH__ >= 800
TOPK_OP(__nv_bfloat16, bf16)
#endif
#if __CUDA_ARCH__ >= 530
TOPK_OP(__half, f16)
#endif
TOPK_OP(float, f32)
