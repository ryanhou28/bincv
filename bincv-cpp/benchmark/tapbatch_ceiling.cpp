// X-36's CEILING. Four popcounts against ONE mask -- the shape of batching across
// taps -- scalar versus vector. Under 1.4x and the arm is not written.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "measure_util.hpp"

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define TB_NEON 1
#else
#define TB_NEON 0
#endif

namespace {
#if TB_NEON
/// Scalar: what slicedSignedSum does at N = 1 for the four displaced taps.
uint64_t scalarTaps(const uint32_t* t, const uint32_t* m, size_t n, uint64_t (&acc)[4]) {
    for (size_t i = 0; i + 4 <= n; i += 4) {
        const uint32_t mask = m[i >> 2];
        acc[0] += static_cast<uint64_t>(__builtin_popcount(t[i + 0] & mask));
        acc[1] += static_cast<uint64_t>(__builtin_popcount(t[i + 1] & mask));
        acc[2] += static_cast<uint64_t>(__builtin_popcount(t[i + 2] & mask));
        acc[3] += static_cast<uint64_t>(__builtin_popcount(t[i + 3] & mask));
    }
    return acc[0] + acc[1] + acc[2] + acc[3];
}

/// Vector: the four taps share a register, the mask is broadcast, counts land in
/// lanes, and the register domain is crossed once at the end.
uint64_t neonTaps(const uint32_t* t, const uint32_t* m, size_t n, uint64_t (&acc)[4]) {
    uint32x4_t lanes = vdupq_n_u32(0);
    for (size_t i = 0; i + 4 <= n; i += 4) {
        const uint32x4_t vt = vld1q_u32(t + i);
        const uint32x4_t vm = vdupq_n_u32(m[i >> 2]);
        const uint8x16_t both = vreinterpretq_u8_u32(vandq_u32(vt, vm));
        lanes = vaddq_u32(lanes, vpaddlq_u16(vpaddlq_u8(vcntq_u8(both))));
    }
    acc[0] += vgetq_lane_u32(lanes, 0); acc[1] += vgetq_lane_u32(lanes, 1);
    acc[2] += vgetq_lane_u32(lanes, 2); acc[3] += vgetq_lane_u32(lanes, 3);
    return acc[0] + acc[1] + acc[2] + acc[3];
}
#endif
} // namespace

int main() {
    std::printf("=== X-36 CEILING: four taps against one mask ===\n");
#if !TB_NEON
    std::printf("  NEON unavailable here -- the reference device's measurement.\n");
    return 0;
#else
    const size_t n = 31 * 4 * 20;
    std::vector<uint32_t> t(n), m(n / 4);
    uint64_t st = 5;
    for (size_t i = 0; i < n; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        t[i] = static_cast<uint32_t>(st >> 33);
    }
    for (size_t i = 0; i < m.size(); ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        m[i] = static_cast<uint32_t>(st >> 33);
    }
    uint64_t a[4] = {0,0,0,0}, b[4] = {0,0,0,0};
    const uint64_t r1 = scalarTaps(t.data(), m.data(), n, a);
    const uint64_t r2 = neonTaps(t.data(), m.data(), n, b);
    std::printf("  EQUALITY: %llu vs %llu -- %s\n", (unsigned long long)r1,
                (unsigned long long)r2, r1 == r2 ? "identical" : "MISMATCH");
    if (r1 != r2) return 1;
    std::vector<measure::Bench> bs = {
        {"scalar, four popcounts", [&](int){ uint64_t c[4]={0,0,0,0};
            measure::g_sink += (size_t)scalarTaps(t.data(), m.data(), n, c); }},
        {"NEON, four taps in lanes", [&](int){ uint64_t c[4]={0,0,0,0};
            measure::g_sink += (size_t)neonTaps(t.data(), m.data(), n, c); }},
    };
    const auto tt = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-30s %10s %8s\n", "arm", "ns", "vs scalar");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-30s %10.1f %7.3fx\n", bs[i].name.c_str(), tt[i].medianNs,
                    tt[0].medianNs / tt[i].medianNs);
    std::printf("\n  X-36's rule: under 1.4x and the arm is not written.\n");
    return 0;
#endif
}
