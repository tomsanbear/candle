#include <metal_stdlib>
#include <metal_integer>
#include <metal_atomic>

using namespace metal;

// Constants
// 2^32 and 1/2^32. Useful for converting between float and uint.
static constexpr constant ulong UNIF01_NORM32 = 4294967296;
static constexpr constant float UNIF01_INV32 = 2.328306436538696289e-10;
// 2 * pi
static constexpr constant float TWO_PI = 2.0 * M_PI_F;
static constexpr constant int3 S1 = {13, 19, 12};
static constexpr constant int3 S2 = {2, 25, 4};
static constexpr constant int3 S3 = {3, 11, 17};

// Used to prevent bad seeds.
static constexpr constant uint64_t PHI[16] = {
    0x9E3779B97F4A7C15,
    0xF39CC0605CEDC834,
    0x1082276BF3A27251,
    0xF86C6A11D0C18E95,
    0x2767F0B153D27B7F,
    0x0347045B5BF1827F,
    0x01886F0928403002,
    0xC1D64BA40F335E36,
    0xF06AD7AE9717877E,
    0x85839D6EFFBD7DC6,
    0x64D325D1C5371682,
    0xCADD0CCCFDFFBBE1,
    0x626E33B8D04B4331,
    0xBBF73C790D94F79D,
    0x471C4AB3ED3D82A5,
    0xFEC507705E4AE6E5,
};

// Combined Tausworthe and LCG Random Number Generator.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
// https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf
struct HybridTaus {

    float state;

    HybridTaus() thread = default;
    HybridTaus() threadgroup = default;
    HybridTaus() device = default;
    HybridTaus() constant = default;

    // Generate seeds for each thread.
    METAL_FUNC static uint4 seed_per_thread(const ulong4 seeds) {
        return uint4(ulong4(seeds) * ulong4(PHI[0], PHI[1], PHI[2], PHI[3]) * ulong4(1099087573UL));
    }

    // Tausworthe generator.
    METAL_FUNC static uint taus(const uint z, const int3 s, const uint M) {
        uint b = (((z << s.x) ^ z) >> s.y);
        return (((z & M) << s.z) ^ b);
    }

    // LCG generator.
    METAL_FUNC static uint lcg(const uint z) {
        return (1664525 * z + 1013904223UL);
    }

    // Initialize the RNG state.
    METAL_FUNC static HybridTaus init(const ulong4 seeds) {
        uint4 seed = seed_per_thread(seeds);

        // Seed #1
        uint z1 = taus(seed.x, S1, 4294967294UL);
        uint z2 = taus(seed.y, S2, 4294967288UL);
        uint z3 = taus(seed.z, S3, 4294967280UL);
        uint z4 = lcg(seed.x);

        // Seed #2
        uint r1 = (z1^z2^z3^z4^seed.y);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #3
        r1 = (z1^z2^z3^z4^seed.z);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        // Seed #4
        r1 = (z1^z2^z3^z4^seed.w);
        z1 = taus(r1, S1, 429496729UL);
        z2 = taus(r1, S2, 4294967288UL);
        z3 = taus(r1, S3, 429496280UL);
        z4 = lcg(r1);

        HybridTaus rng;
        rng.state = (z1^z2^z3^z4) * UNIF01_INV32;
        return rng;
    }

    METAL_FUNC float rand() {
        uint seed = this->state * UNIF01_NORM32;
        uint z1 = taus(seed, S1, 429496729UL);
        uint z2 = taus(seed, S2, 4294967288UL);
        uint z3 = taus(seed, S3, 429496280UL);
        uint z4 = lcg(seed);

        thread float result = this->state;
        this->state = (z1^z2^z3^z4) * UNIF01_INV32;
        return result;
    }
};
typedef struct
{
    atomic_uint seed[2];
} seed_buffer;

// Stateless Philox4x32-10. The seed is a two-word key and the output position
// is a four-word counter, so every operation owns its complete random state and
// all 64 seed bits participate. Ten rounds follows Random123 and PyTorch's
// generic Philox engine; spell out the final round so an off-by-one loop cannot
// silently weaken it.
METAL_FUNC uint2 philox_mulhilo(uint a, uint b) {
    ulong product = static_cast<ulong>(a) * static_cast<ulong>(b);
    return uint2(static_cast<uint>(product >> 32), static_cast<uint>(product));
}

METAL_FUNC uint4 philox_round(uint4 counter, uint2 key) {
    constexpr uint PHILOX_M0 = 0xD2511F53;
    constexpr uint PHILOX_M1 = 0xCD9E8D57;
    uint2 p0 = philox_mulhilo(PHILOX_M0, counter.x);
    uint2 p1 = philox_mulhilo(PHILOX_M1, counter.z);
    return uint4(p1.x ^ counter.y ^ key.x, p1.y, p0.x ^ counter.w ^ key.y, p0.y);
}

METAL_FUNC uint4 philox(uint64_t seed, uint64_t index) {
    constexpr uint2 WEYL = uint2(0x9E3779B9, 0xBB67AE85);
    uint2 key = uint2(static_cast<uint>(seed), static_cast<uint>(seed >> 32));
    uint4 counter = uint4(
        static_cast<uint>(index),
        static_cast<uint>(index >> 32),
        0,
        0
    );
    for (uint round = 0; round < 9; ++round) {
        counter = philox_round(counter, key);
        key += WEYL;
    }
    return philox_round(counter, key);
}

METAL_FUNC float philox_uniform(uint value) {
    // Use the high 24 bits: exactly representable and strictly below one.
    return static_cast<float>(value >> 8) * 0x1.0p-24f;
}


METAL_FUNC ulong atomic_load_seed(device seed_buffer *sb) {
    uint x = atomic_load_explicit(&sb->seed[0], memory_order_relaxed);
    uint y = atomic_load_explicit(&sb->seed[1], memory_order_relaxed);
    return static_cast<ulong>(x) << 32 | y;
}

METAL_FUNC void atomic_store_seed(device seed_buffer *sb, ulong desired) {
    uint x = static_cast<uint>(desired >> 32);
    uint y = static_cast<uint>(desired & 0xFFFFFFFF);
    atomic_store_explicit(&sb->seed[0], x, memory_order_relaxed);
    atomic_store_explicit(&sb->seed[1], y, memory_order_relaxed);
}

template<typename T> METAL_FUNC void rand_uniform(
    constant size_t &size,
    constant float &min,
    constant float &max,
    device seed_buffer *sb,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) {
        return;
    }

    // Evenly sized vectors need an offset when writing the mirror element.
    uint off = 1 - size % 2;
    float diff = abs(min - max);
    ulong s = atomic_load_seed(sb);
    HybridTaus rng = HybridTaus::init({s, tid, 1, 1});
    out[tid] = static_cast<T>(rng.rand() * diff + min);
    if (tid == 0) {
        atomic_store_seed(sb, rng.rand() * UNIF01_NORM32);
        // Return early if tid == 0 && off == 0, otherwise we will write to out[size].
        if (off == 0)
            return;
    }
    // Use symmetry to fill the other half of the array.
    out[size - off - tid] = static_cast<T>(rng.rand() * diff + min);
}

template<typename T> METAL_FUNC void rand_uniform_seeded(
    constant size_t &size,
    constant float &min,
    constant float &max,
    constant ulong &seed,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    size_t base = static_cast<size_t>(tid) * 4;
    if (base >= size) {
        return;
    }

    float diff = abs(min - max);
    uint4 random = philox(seed, tid);
    out[base] = static_cast<T>(philox_uniform(random.x) * diff + min);
    if (base + 1 < size) {
        out[base + 1] = static_cast<T>(philox_uniform(random.y) * diff + min);
    }
    if (base + 2 < size) {
        out[base + 2] = static_cast<T>(philox_uniform(random.z) * diff + min);
    }
    if (base + 3 < size) {
        out[base + 3] = static_cast<T>(philox_uniform(random.w) * diff + min);
    }
}

// Create Gaussian normal distribution using Box-Muller transform:
// https://en.wikipedia.org/wiki/Box–Muller_transform
template<typename T> METAL_FUNC void normal(
    constant size_t &size,
    constant float &mean,
    constant float &stddev,
    device seed_buffer *sb,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) {
        return;
    }
    // Evenly sized vectors need an offset when writing the mirror element.
    uint off = 1 - size % 2;
    ulong s = atomic_load_seed(sb);
    HybridTaus rng = HybridTaus::init({s, tid, 1, 1});
    float u1 = rng.rand();
    float u2 = rng.rand();

    float cosval;
    float sinval = sincos(TWO_PI * u2, cosval);
    float mag = stddev * sqrt(-2.0 * log(u1));
    float z0  = mag * cosval + mean;
    float z1  = mag * sinval + mean;

    out[tid] = static_cast<T>(z0);

    if (tid == 0) {
        atomic_store_seed(sb, rng.rand() * UNIF01_NORM32);
        // Return early if tid == 0 && off == 0, otherwise we will write to out[size].
        if (off == 0)
            return;
    }
    // Use symmetry to fill the other half of the array.
    out[size - off - tid] = static_cast<T>(z1);
}

template<typename T> METAL_FUNC void normal_seeded(
    constant size_t &size,
    constant float &mean,
    constant float &stddev,
    constant ulong &seed,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    size_t base = static_cast<size_t>(tid) * 4;
    if (base >= size) {
        return;
    }

    uint4 random = philox(seed, tid);
    float u0 = 1.0 - philox_uniform(random.x);
    float u1 = philox_uniform(random.y);
    float u2 = 1.0 - philox_uniform(random.z);
    float u3 = philox_uniform(random.w);

    float cos0;
    float sin0 = sincos(TWO_PI * u1, cos0);
    float mag0 = stddev * sqrt(-2.0 * log(u0));
    float cos1;
    float sin1 = sincos(TWO_PI * u3, cos1);
    float mag1 = stddev * sqrt(-2.0 * log(u2));
    float z0 = mag0 * cos0 + mean;
    float z1 = mag0 * sin0 + mean;
    float z2 = mag1 * cos1 + mean;
    float z3 = mag1 * sin1 + mean;

    out[base] = static_cast<T>(z0);
    if (base + 1 < size) {
        out[base + 1] = static_cast<T>(z1);
    }
    if (base + 2 < size) {
        out[base + 2] = static_cast<T>(z2);
    }
    if (base + 3 < size) {
        out[base + 3] = static_cast<T>(z3);
    }
}

#define UNIFORM_OP(NAME, T)                             \
kernel void rand_uniform_##NAME(                        \
    constant size_t &size,                              \
    constant float &min,                                \
    constant float &max,                                \
    device seed_buffer *sb,                             \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    rand_uniform<T>(size, min, max, sb, out, tid);      \
}                                                       \

#define UNIFORM_SEEDED_OP(NAME, T)                      \
kernel void rand_uniform_seeded_##NAME(                 \
    constant size_t &size,                              \
    constant float &min,                                \
    constant float &max,                                \
    constant ulong &seed,                               \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    rand_uniform_seeded<T>(size, min, max, seed, out, tid); \
}                                                       \

#define NORMAL_OP(NAME, T)                              \
kernel void rand_normal_##NAME(                         \
    constant size_t &size,                              \
    constant float &mean,                               \
    constant float &stddev,                             \
    device seed_buffer *sb,                             \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    normal<T>(size, mean, stddev, sb, out, tid);        \
}                                                       \

#define NORMAL_SEEDED_OP(NAME, T)                       \
kernel void rand_normal_seeded_##NAME(                  \
    constant size_t &size,                              \
    constant float &mean,                               \
    constant float &stddev,                             \
    constant ulong &seed,                               \
    device T *out,                                      \
    uint tid [[thread_position_in_grid]]                \
) {                                                     \
    normal_seeded<T>(size, mean, stddev, seed, out, tid); \
}                                                       \


#define RANDOM_OPS(NAME, T) \
UNIFORM_OP(NAME, T)         \
UNIFORM_SEEDED_OP(NAME, T)  \
NORMAL_OP(NAME, T)          \
NORMAL_SEEDED_OP(NAME, T)   \

RANDOM_OPS(f32, float)
RANDOM_OPS(f16, half)

#if __METAL_VERSION__ >= 310
RANDOM_OPS(bf16, bfloat)
#endif
