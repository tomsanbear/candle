// MSL 4.1 packed_numeric variant of the q4_K matvec (m3-macos26-eval-suite).
//
// This source compiles under MTLLanguageVersion 4.1 ONLY (its own Source in
// kernel.rs) — macOS < 27 fails the library load and routing falls back.
//
// Why: per-dispatch counters show the shipped mv kernel spends 61% of its
// 3.26B ALU instructions on integer mask/shift dequantization — the
// documented issue-width limiter at ~92% of DRAM roof. MSL 4.1's
// packed_numeric_type<uint4b_format, 16> unpack converts a thread's whole
// 8-byte plane slice in one construct; if it lowers to a real widening op,
// the integer stream collapses.
//
// Geometry and per-thread slice mirror kernel_mul_mv_q4_K_impl_t (nr2sg2:
// NDST = 2 rows/simdgroup, NSG = 2) so A/B differences are inner-loop-only.
// q4_K nibble layout: byte b of a 32-byte plane region holds value b (low
// nibble) and value b+32 (high nibble); one uint = bytes 4k..4k+4 unpacks
// to lanes {v_k0L, v_k0H, v_k1L, v_k1H, ...} — even lanes are the low plane
// (scale sc_even), odd lanes the high plane (sc_odd). The dmin term uses
// the same algebraic factoring as the shipped kernel. Unpack yields TRUE
// values (0..15), so the 1/256 and 1/16 fixups of the masked form vanish.
//
// The struct/scale helpers are duplicated from quantized.metal (sources
// compile standalone); the equality test in candle-core keeps them honest.

#include <metal_stdlib>
#include <metal_packed_numeric>
using namespace metal;

#define QK_K 256
#define K_SCALE_SIZE 12

// 16 nibbles per 64-bit word — a thread's full 8-byte plane slice in ONE
// unpack (Table 2.19 caps uint4b_format at N=16). The constructor takes
// storage_type (packed_vec<uchar, 8>), NOT a scalar — passing an integer
// directly scalar-converts to uchar and broadcasts one byte (Table 2.18).
// Lane order is little-endian nibbles: lane 2b = byte b low nibble,
// lane 2b+1 = byte b high nibble.
using pnu4x16_t = packed_numeric_type<uint4b_format, 16>;

inline vec<float, 16> unpack_q4x16(ulong w) {
    return unpack<float>(pnu4x16_t(as_type<pnu4x16_t::storage_type>(w)));
}

typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} block_q4_K;

static_assert(sizeof(block_q4_K) == 144, "wrong q4_K block size");

inline void load_y8(device const bfloat * p, thread float * dst) {
    const ushort4 u0 = *(device const ushort4 *)(p + 0);
    const ushort4 u1 = *(device const ushort4 *)(p + 4);
    const float4 f0 = as_type<float4>(uint4(u0) << 16);
    const float4 f1 = as_type<float4>(uint4(u1) << 16);
    dst[0] = f0.x; dst[1] = f0.y; dst[2] = f0.z; dst[3] = f0.w;
    dst[4] = f1.x; dst[5] = f1.y; dst[6] = f1.z; dst[7] = f1.w;
}

kernel void kernel_mul_mv_q4_K_bf16_bf16_unpk(
        device const   void * src0,
        device const bfloat * src1,
        device       bfloat * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant    int64_t & ne02,
        constant   uint64_t & nb00,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne10,
        constant    int64_t & ne11,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
        constant    int64_t & ne1,
        constant    uint    & r2,
        constant    uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int NSG = 2;
    constexpr int NDST = 2;

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg / 8;
    const int it = tiisg % 8;
    const int iq = it / 4;
    const int ir = it % 4;

    const int nb = ne00 / QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int first_row = (r0 * NSG + sgitg) * NDST;
    const int ib_row = first_row * nb;

    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row;
    device const bfloat     * y = src1 + r1 * ne10;

    float yl[16];
    float yh[16];
    float sumf[NDST] = {0.f};

    const int step = sizeof(block_q4_K) * nb;

    device const bfloat * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        load_y8(y4 +   0, yl + 0);
        load_y8(y4 +  32, yl + 8);
        load_y8(y4 + 128, yh + 0);
        load_y8(y4 + 160, yh + 8);
        for (int i = 0; i < 8; ++i) {
            sumy[0] += yl[i + 0];
            sumy[1] += yl[i + 8];
            sumy[2] += yh[i + 0];
            sumy[3] += yh[i + 8];
        }

        device const uint8_t * scb = x[ib].scales;
        // Thread slice: 8 bytes of the q1 plane pair and 8 of q2 (see the
        // shipped kernel); one 64-bit word each. 8-byte aligned: qs sits at
        // offset 16 in the 144-byte block and the slice offset is 8*ir.
        device const ulong * q1 = (device const ulong *)(x[ib].qs + 32 * iq + 8 * ir);
        device const ulong * q2 = q1 + 8; // +64 bytes

        for (int row = 0; row < NDST; row++) {
            if (first_row + row >= ne01) break;

            device const uint16_t * sc = (device const uint16_t *)scb + iq;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const half * dh = &((device const block_q4_K *)((device const uint8_t *)x + row * step))[ib].d;
            // Row-strided views of the same slice for row 1.
            device const ulong * q1r = (device const ulong *)((device const uint8_t *)q1 + row * step);
            device const ulong * q2r = (device const ulong *)((device const uint8_t *)q2 + row * step);

            // One unpack per plane slice: even lanes are the low plane
            // (values b -> yl[b]), odd lanes the high plane (values b+32
            // -> yl[b+8]); q2 covers the +128 region paired with yh.
            const vec<float, 16> v1 = unpack_q4x16(q1r[0]);
            const vec<float, 16> v2 = unpack_q4x16(q2r[0]);

            float acc_l1 = 0.f, acc_h1 = 0.f;
            float acc_l2 = 0.f, acc_h2 = 0.f;
            for (int b = 0; b < 8; ++b) {
                acc_l1 = fma(yl[b + 0], v1[2 * b + 0], acc_l1);
                acc_h1 = fma(yl[b + 8], v1[2 * b + 1], acc_h1);
                acc_l2 = fma(yh[b + 0], v2[2 * b + 0], acc_l2);
                acc_h2 = fma(yh[b + 8], v2[2 * b + 1], acc_h2);
            }

            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (acc_l1 * sc8[0] +
                                 acc_h1 * sc8[1] +
                                 acc_l2 * sc8[4] +
                                 acc_h2 * sc8[5]) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                 sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            scb += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < NDST; ++row) {
        float all_sum = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1 * ne0 + first_row + row] = static_cast<bfloat>(all_sum);
        }
    }
}
