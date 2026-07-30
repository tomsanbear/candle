#include <stdint.h>

// Stateless Philox4x32-10. Each thread owns one four-word logical counter, so
// operation-local output is independent of CUDA stream order and mutable
// cuRAND state. The two-word key preserves all 64 seed bits.
__device__ __forceinline__ uint4 philox_round(uint4 counter, uint2 key) {
    constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
    constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
    const uint32_t p0_hi = __umulhi(PHILOX_M0, counter.x);
    const uint32_t p0_lo = PHILOX_M0 * counter.x;
    const uint32_t p1_hi = __umulhi(PHILOX_M1, counter.z);
    const uint32_t p1_lo = PHILOX_M1 * counter.z;
    return make_uint4(
        p1_hi ^ counter.y ^ key.x,
        p1_lo,
        p0_hi ^ counter.w ^ key.y,
        p0_lo
    );
}

__device__ __forceinline__ uint4 philox(uint64_t seed, uint64_t index) {
    constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
    constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;
    uint2 key = make_uint2(
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32)
    );
    uint4 counter = make_uint4(
        static_cast<uint32_t>(index),
        static_cast<uint32_t>(index >> 32),
        0u,
        0u
    );
    for (uint32_t round = 0; round < 9; ++round) {
        counter = philox_round(counter, key);
        key.x += PHILOX_W0;
        key.y += PHILOX_W1;
    }
    return philox_round(counter, key);
}

__device__ __forceinline__ float philox_uniform(uint32_t value) {
    // The high 24 bits are exactly representable as f32 and strictly below 1.
    return static_cast<float>(value >> 8) * 0x1.0p-24f;
}

extern "C" __global__ void rand_uniform_seeded_f32(
    size_t size,
    float min,
    float max,
    uint64_t seed,
    float *out
) {
    const uint64_t counter =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t base = static_cast<size_t>(counter) * 4;
    if (base >= size) {
        return;
    }

    const uint4 random = philox(seed, counter);
    const float diff = max - min;
    out[base] = philox_uniform(random.x) * diff + min;
    if (base + 1 < size) {
        out[base + 1] = philox_uniform(random.y) * diff + min;
    }
    if (base + 2 < size) {
        out[base + 2] = philox_uniform(random.z) * diff + min;
    }
    if (base + 3 < size) {
        out[base + 3] = philox_uniform(random.w) * diff + min;
    }
}

extern "C" __global__ void rand_normal_seeded_f32(
    size_t size,
    float mean,
    float stddev,
    uint64_t seed,
    float *out
) {
    constexpr float TWO_PI = 6.283185307179586476925286766559f;
    const uint64_t counter =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t base = static_cast<size_t>(counter) * 4;
    if (base >= size) {
        return;
    }

    const uint4 random = philox(seed, counter);
    const float u0 = 1.0f - philox_uniform(random.x);
    const float u1 = philox_uniform(random.y);
    const float u2 = 1.0f - philox_uniform(random.z);
    const float u3 = philox_uniform(random.w);

    float sin0;
    float cos0;
    sincosf(TWO_PI * u1, &sin0, &cos0);
    const float mag0 = stddev * sqrtf(-2.0f * logf(u0));
    float sin1;
    float cos1;
    sincosf(TWO_PI * u3, &sin1, &cos1);
    const float mag1 = stddev * sqrtf(-2.0f * logf(u2));

    out[base] = mag0 * cos0 + mean;
    if (base + 1 < size) {
        out[base + 1] = mag0 * sin0 + mean;
    }
    if (base + 2 < size) {
        out[base + 2] = mag1 * cos1 + mean;
    }
    if (base + 3 < size) {
        out[base + 3] = mag1 * sin1 + mean;
    }
}
