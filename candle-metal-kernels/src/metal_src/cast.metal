#include <metal_stdlib>
using namespace metal;

// Utils
METAL_FUNC uint get_strided_index(
    uint idx,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides
) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        strided_i += (idx % dims[dim_idx]) * strides[dim_idx];
        idx /= dims[dim_idx];
    }
    return strided_i;
}

template<uint Y>
constexpr uint div_ceil(uint x) {
    return x / Y + (x % Y > 0);
}

template<uint X, uint Y>
constexpr uint div_ceil() {
    return X / Y + (X % Y > 0);
}

template<typename T>
constexpr uint work_per_thread() {
    return div_ceil<8, sizeof(T)>();
}

// Kernels
template <
    typename T,
    typename U,
    typename IR = T,
    int W = work_per_thread<T>()
>
[[kernel]] void cast_kernel(
    constant size_t &dim,
    device const T* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = static_cast<U>(static_cast<IR>(input[i]));
    }
}

template <typename T, typename U, typename IR = T>
[[kernel]] void cast_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        static_cast<IR>(input[get_strided_index(tid, num_dims, dims, strides)])
    );
}

// f64 casts. MSL has no double type at any feature level ("Metal does not
// support the double ... data types", MSL spec 2.1), so f64 buffers are read
// and written as ulong bit patterns and converted in software. The pieces
// this relies on are all supported: long/ulong are 64-bit scalar types (the
// spec excludes `long long`, not `long`), buffers of them need Metal 2.3+
// (as the i64 kernels above already do), and 64-bit integer math incl. clz
// is a Metal3/Apple3 feature.

// f64 bits -> f32, round-to-nearest-even (no hardware conversion exists).
METAL_FUNC float double_bits_to_float(ulong bits) {
    const uint sign32 = uint(bits >> 63) << 31;
    const int exp = int((bits >> 52) & 0x7ff);
    const ulong mant = bits & 0xfffffffffffffull;
    if (exp == 0x7ff) {
        // Inf / NaN (payload not preserved, quiet bit set).
        return as_type<float>(sign32 | 0x7f800000u | (mant != 0 ? 0x400000u : 0u));
    }
    if (exp == 0) {
        // f64 denormals are far below the f32 denormal range.
        return as_type<float>(sign32);
    }
    int e = exp - 1023;
    if (e > 127) {
        return as_type<float>(sign32 | 0x7f800000u);
    }
    const ulong sig = (1ull << 52) | mant;
    // 53-bit significand -> 24 bits; f32-denormal results shift further.
    int shift = 29;
    if (e < -126) {
        shift += -126 - e;
        if (shift > 63) {
            return as_type<float>(sign32);
        }
        e = -127; // biased exponent field 0 (denormal)
    }
    ulong kept = sig >> ulong(shift);
    const ulong rem = sig & ((1ull << ulong(shift)) - 1);
    const ulong halfway = 1ull << ulong(shift - 1);
    if (rem > halfway || (rem == halfway && (kept & 1))) {
        kept += 1;
        if (kept == (1ull << 24)) { // carry renormalizes
            kept >>= 1;
            e += 1;
            if (e > 127) {
                return as_type<float>(sign32 | 0x7f800000u);
            }
        }
    }
    if (e == -127) {
        // Denormal: exponent field 0; a carry to 1<<23 lands on the smallest
        // normal through the exponent field, which is exactly right.
        return as_type<float>(sign32 | uint(kept));
    }
    return as_type<float>(sign32 | (uint(e + 127) << 23) | uint(kept - (1ull << 23)));
}

// f32 -> f64 bits, exact.
METAL_FUNC ulong float_to_double_bits(float x) {
    const uint fb = as_type<uint>(x);
    const ulong sign = ulong(fb >> 31) << 63;
    const uint exp = (fb >> 23) & 0xffu;
    const uint mant = fb & 0x7fffffu;
    if (exp == 0xffu) {
        return sign | (0x7ffull << 52) | (mant != 0 ? (1ull << 51) : 0ull);
    }
    if (exp == 0) {
        if (mant == 0) {
            return sign;
        }
        // f32 denormal (mant * 2^-149) normalizes into the f64 range.
        const int h = 31 - int(clz(mant));
        const ulong sig = ulong(mant) << ulong(52 - h);
        return sign | (ulong(h - 149 + 1023) << 52) | (sig & 0xfffffffffffffull);
    }
    return sign | (ulong(int(exp) - 127 + 1023) << 52) | (ulong(mant) << 29);
}

// Integer magnitude -> f64 bits; exact for magnitudes <= 2^53, RNE above.
METAL_FUNC ulong int_mag_to_double_bits(ulong mag, bool neg) {
    if (mag == 0) {
        return 0;
    }
    const int h = 63 - int(clz(mag));
    int e = h;
    ulong sig;
    if (h <= 52) {
        sig = mag << ulong(52 - h);
    } else {
        const int shift = h - 52;
        const ulong rem = mag & ((1ull << ulong(shift)) - 1);
        const ulong halfway = 1ull << ulong(shift - 1);
        sig = mag >> ulong(shift);
        if (rem > halfway || (rem == halfway && (sig & 1))) {
            sig += 1;
            if (sig == (1ull << 53)) {
                sig >>= 1;
                e += 1;
            }
        }
    }
    return (neg ? (1ull << 63) : 0ull) | (ulong(e + 1023) << 52) | (sig & 0xfffffffffffffull);
}

// f64 bits -> magnitude truncated toward zero. NaN -> 0, |value| >= 2^64
// saturates to ULONG_MAX; callers clamp per destination type to match the
// CPU reference (Rust's saturating `as`).
METAL_FUNC ulong double_bits_to_int_mag(ulong bits, thread bool &neg) {
    neg = (bits >> 63) != 0;
    const int exp = int((bits >> 52) & 0x7ff);
    const ulong mant = bits & 0xfffffffffffffull;
    if (exp == 0x7ff) {
        if (mant != 0) {
            neg = false;
            return 0;
        }
        return 0xffffffffffffffffull;
    }
    const int e = exp - 1023;
    if (e < 0) {
        return 0;
    }
    if (e >= 64) {
        return 0xffffffffffffffffull;
    }
    const ulong sig = (1ull << 52) | mant;
    return e <= 52 ? sig >> ulong(52 - e) : sig << ulong(e - 52);
}

template <typename U>
METAL_FUNC U double_bits_to_int(ulong bits) {
    bool neg;
    const ulong mag = double_bits_to_int_mag(bits, neg);
    const bool is_signed = U(-1) < U(0);
    if (is_signed) {
        const ulong lim_pos = (1ull << (8 * sizeof(U) - 1)) - 1;
        const ulong m = min(mag, neg ? lim_pos + 1 : lim_pos);
        // Two's-complement negate in the unsigned domain; the narrowing
        // conversion wraps, which encodes the type's minimum correctly.
        return neg ? U(~m + 1) : U(m);
    }
    if (neg) {
        return U(0);
    }
    const ulong lim =
        sizeof(U) == 8 ? 0xffffffffffffffffull : (1ull << (8 * sizeof(U))) - 1;
    return U(min(mag, lim));
}

template <typename T>
METAL_FUNC ulong int_to_double_bits(T x) {
    const bool neg = x < T(0);
    // ulong(long(x)) sign-extends; the unsigned negate yields |x| without
    // overflow, including at the type's minimum.
    const ulong u = ulong(long(x));
    return int_mag_to_double_bits(neg ? ~u + 1 : u, neg);
}

// f64 source: 8-byte elements, so one element per thread (W = 1), matching
// what the Rust wrapper derives from the source dtype size.
template <typename U>
[[kernel]] void cast_f64_float_kernel(
    constant size_t &dim,
    device const ulong* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    // f64 -> f16/bf16 goes through f32; a 1-ulp double rounding is possible
    // in rare halfway cases.
    output[tid] = static_cast<U>(double_bits_to_float(input[tid]));
}

template <typename U>
[[kernel]] void cast_f64_int_kernel(
    constant size_t &dim,
    device const ulong* input,
    device U* output,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    output[tid] = double_bits_to_int<U>(input[tid]);
}

template <typename U>
[[kernel]] void cast_f64_float_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const ulong *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] = static_cast<U>(
        double_bits_to_float(input[get_strided_index(tid, num_dims, dims, strides)]));
}

template <typename U>
[[kernel]] void cast_f64_int_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const ulong *input,
    device U *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] =
        double_bits_to_int<U>(input[get_strided_index(tid, num_dims, dims, strides)]);
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void cast_float_f64_kernel(
    constant size_t &dim,
    device const T* input,
    device ulong* output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = float_to_double_bits(static_cast<float>(input[i]));
    }
}

template <typename T, int W = work_per_thread<T>()>
[[kernel]] void cast_int_f64_kernel(
    constant size_t &dim,
    device const T* input,
    device ulong* output,
    uint tid [[thread_position_in_grid]]
) {
    const uint step = div_ceil<W>(dim);
    #pragma clang loop unroll(full)
    for (uint i = tid; i < dim; i += step) {
        output[i] = int_to_double_bits(input[i]);
    }
}

template <typename T>
[[kernel]] void cast_float_f64_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device ulong *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] = float_to_double_bits(
        static_cast<float>(input[get_strided_index(tid, num_dims, dims, strides)]));
}

template <typename T>
[[kernel]] void cast_int_f64_kernel_strided(
    constant size_t &dim,
    constant size_t &num_dims,
    constant size_t *dims,
    constant size_t *strides,
    constant const T *input,
    device ulong *output,
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= dim) return;
    output[tid] =
        int_to_double_bits(input[get_strided_index(tid, num_dims, dims, strides)]);
}

// Macros to help initialize kernels
#define init_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#define init_cast(tname, t, uname, u)                                           \
    init_kernel("cast_" #tname "_" #uname, cast_kernel, t, u)                   \
    init_kernel("cast_" #tname "_" #uname "_strided", cast_kernel_strided, t, u)

// f64 is absent from this matrix: MSL has no double type at any feature
// level ("Metal does not support the double ... data types", MSL spec 2.1),
// so no kernel can do f64 arithmetic. Casts are the exception — the f64
// row/column is covered by the software bit-conversion kernels above.
#if defined(__HAVE_BFLOAT__)
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, bf16, bfloat)   \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, i32, int32_t)   \
    init_cast(tname, t, i16, int16_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#else
#define init_cast_all(tname, t)         \
    init_cast(tname, t, f32, float)     \
    init_cast(tname, t, f16, half)      \
    init_cast(tname, t, i64, int64_t)   \
    init_cast(tname, t, i32, int32_t)   \
    init_cast(tname, t, i16, int16_t)   \
    init_cast(tname, t, u32, uint32_t)  \
    init_cast(tname, t, u8, uint8_t)
#endif


init_cast_all(f32, float);
init_cast_all(f16, half);
#if defined(__HAVE_BFLOAT__)
init_cast_all(bf16, bfloat);
#endif
init_cast_all(i64, int64_t);
init_cast_all(i32, int32_t);
init_cast_all(i16, int16_t);
init_cast_all(u32, uint32_t);
init_cast_all(u8, uint8_t);

// The f64 row and column, via the software bit-conversion kernels.
#define init_cast_f64_pair(uname, u, from_k, to_k)                    \
    init_kernel("cast_f64_" #uname, from_k, u)                        \
    init_kernel("cast_f64_" #uname "_strided", from_k##_strided, u)   \
    init_kernel("cast_" #uname "_f64", to_k, u)                       \
    init_kernel("cast_" #uname "_f64_strided", to_k##_strided, u)

init_cast_f64_pair(f32, float, cast_f64_float_kernel, cast_float_f64_kernel)
init_cast_f64_pair(f16, half, cast_f64_float_kernel, cast_float_f64_kernel)
#if defined(__HAVE_BFLOAT__)
init_cast_f64_pair(bf16, bfloat, cast_f64_float_kernel, cast_float_f64_kernel)
#endif
init_cast_f64_pair(i64, int64_t, cast_f64_int_kernel, cast_int_f64_kernel)
init_cast_f64_pair(i32, int32_t, cast_f64_int_kernel, cast_int_f64_kernel)
init_cast_f64_pair(i16, int16_t, cast_f64_int_kernel, cast_int_f64_kernel)
init_cast_f64_pair(u32, uint32_t, cast_f64_int_kernel, cast_int_f64_kernel)
init_cast_f64_pair(u8, uint8_t, cast_f64_int_kernel, cast_int_f64_kernel)
