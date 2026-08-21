// ===========================================================================
// X-33's CEILING, measured BEFORE the real kernel is written.
//
// If batching popcounts into NEON lanes cannot beat the scalar path by 1.5x even
// with everything else stripped away -- no taps, no masks, no weighting, no
// accumulator structure -- then the real kernel cannot either, and X-33's arm is
// not written. X-32 was a day spent on an optimisation whose ceiling was 1.07x.
//
// Both arms consume the SAME words and produce the SAME sums, so a mismatch is a
// bug and is reported.
// ===========================================================================
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "measure_util.hpp"

#if defined(BINCV_HAVE_NEON) || defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define CEILING_HAVE_NEON 1
#else
#define CEILING_HAVE_NEON 0
#endif

namespace {

#if CEILING_HAVE_NEON
/// The scalar shape: one popcount per word, four accumulators, as the kernel does.
/// Inside the NEON guard because on a platform without NEON there is nothing to
/// compare it against and `-Wunused-function` is fatal (scripts/verify.sh). The
/// gate caught this; a targeted `cmake --build --target` did not, because it never
/// compiled this file on x86 at all.
uint64_t scalarSum(const uint32_t* a, const uint32_t* b, size_t n, uint64_t (&acc)[4]) {
    for (size_t i = 0; i + 4 <= n; i += 4) {
        acc[0] += static_cast<uint64_t>(__builtin_popcount(a[i + 0] & b[i + 0]));
        acc[1] += static_cast<uint64_t>(__builtin_popcount(a[i + 1] & b[i + 1]));
        acc[2] += static_cast<uint64_t>(__builtin_popcount(a[i + 2] & b[i + 2]));
        acc[3] += static_cast<uint64_t>(__builtin_popcount(a[i + 3] & b[i + 3]));
    }
    return acc[0] + acc[1] + acc[2] + acc[3];
}

/// The vector shape: four words per iteration, counts accumulated IN LANES, and the
/// register domain crossed once at the end instead of once per word.
uint64_t neonSum(const uint32_t* a, const uint32_t* b, size_t n, uint64_t (&acc)[4]) {
    uint32x4_t lanes = vdupq_n_u32(0);
    for (size_t i = 0; i + 4 <= n; i += 4) {
        const uint32x4_t va = vld1q_u32(a + i);
        const uint32x4_t vb = vld1q_u32(b + i);
        const uint8x16_t both = vreinterpretq_u8_u32(vandq_u32(va, vb));
        // CNT is per byte; widen twice to get one count per 32-bit lane.
        const uint16x8_t c16 = vpaddlq_u8(vcntq_u8(both));
        lanes = vaddq_u32(lanes, vpaddlq_u16(c16));
    }
    acc[0] += vgetq_lane_u32(lanes, 0);
    acc[1] += vgetq_lane_u32(lanes, 1);
    acc[2] += vgetq_lane_u32(lanes, 2);
    acc[3] += vgetq_lane_u32(lanes, 3);
    return acc[0] + acc[1] + acc[2] + acc[3];
}
#endif

} // namespace

int main() {
    // 31x31 window at uint32 is ~2 words/row x 31 rows = 62 word-visits, and
    // residualSums touches each 20N^2/20 times. Size the buffer to that order so
    // the measurement sits in the same cache regime as the kernel.
    const size_t n = 64 * 20;
    std::vector<uint32_t> a(n), b(n);
    uint64_t st = 99;
    for (size_t i = 0; i < n; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        a[i] = static_cast<uint32_t>(st >> 33);
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        b[i] = static_cast<uint32_t>(st >> 33);
    }

    std::printf("=== X-33 CEILING: batched NEON popcount vs scalar ===\n");
#if !CEILING_HAVE_NEON
    std::printf("  NEON unavailable on this platform -- this is the reference device's\n"
                "  measurement to make. Nothing is reported from here.\n");
    return 0;
#else
    uint64_t s1[4] = {0, 0, 0, 0}, s2[4] = {0, 0, 0, 0};
    const uint64_t r1 = scalarSum(a.data(), b.data(), n, s1);
    const uint64_t r2 = neonSum(a.data(), b.data(), n, s2);
    std::printf("  EQUALITY: scalar %llu, neon %llu -- %s\n",
                static_cast<unsigned long long>(r1), static_cast<unsigned long long>(r2),
                r1 == r2 ? "identical" : "MISMATCH");
    if (r1 != r2) return 1;

    std::vector<measure::Bench> bs = {
        {"scalar popcount, 4 accumulators", [&](int) {
             uint64_t acc[4] = {0, 0, 0, 0};
             measure::g_sink += static_cast<size_t>(scalarSum(a.data(), b.data(), n, acc));
         }},
        {"NEON vcnt, lane accumulators", [&](int) {
             uint64_t acc[4] = {0, 0, 0, 0};
             measure::g_sink += static_cast<size_t>(neonSum(a.data(), b.data(), n, acc));
         }},
    };
    const auto t = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-34s %10s %8s\n", "arm", "ns", "vs scalar");
    for (size_t i = 0; i < bs.size(); ++i) {
        std::printf("  %-34s %10.1f %7.3fx\n", bs[i].name.c_str(), t[i].medianNs,
                    t[0].medianNs / t[i].medianNs);
    }
    std::printf("\n  X-33's rule: if this CEILING is under 1.5x, the real kernel is not\n"
                "  written -- everything else in residualSums only dilutes it.\n");
    return 0;
#endif
}
