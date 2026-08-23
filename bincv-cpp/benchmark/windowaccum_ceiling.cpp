// X-40's CEILING for E-18: window-carried vector accumulators at N = 2.
//
// Two shapes computing THE SAME TEN INTEGERS -- the five tap sums for each of the
// two gradient components -- over one 31-row aligned window.
//
//   A  shipped: slicedSignedSum's shape. The four (i, j) PLANE PAIRS share a
//      register and the horizontal add runs ONCE PER CALL: ten calls per row,
//      31 rows, ~310 register-domain crossings per window.
//   B  proposed: the four TAPS share a register, the plane pairs are folded inside
//      the row with vmlaq_n_s32, and ONE int32x4_t accumulator per component runs
//      to the end of the window. `self` keeps A's shape in both arms, exactly as
//      D-33's N = 1 path leaves it scalar, so the measured difference is the tap
//      work and nothing else.
//
// The arms are compared for EQUALITY before they are timed. A ceiling that is also
// a correctness check -- X-36's was not.
//
// Rule (EXPERIMENTS.md X-40, committed before this file): >= 2.0x write the arm;
// 1.4-2.0x write it and report the frontend effect as modest; < 1.4x do not write
// it and close E-18 negative; SLOWER means the domain-crossing cost model is wrong.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "measure_util.hpp"

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define WA_NEON 1
#else
#define WA_NEON 0
#endif

namespace {
#if WA_NEON

constexpr size_t kRows = 31;      ///< the shipped 31x31 window, one word per row (D-31)
constexpr size_t kTaps = 4;       ///< t00 t01 t10 t11; `self` is handled separately
constexpr size_t kN = 2;          ///< three of the four levels of the 1/2/2/2 ladder

/// One window's worth of rows, laid out as the kernel sees them.
struct Window {
    uint32_t tap[kRows][kTaps][kN];   ///< four displaced taps, N planes each
    uint32_t self[kRows][kN];
    uint32_t mag[2][kRows][kN];       ///< [component][row][plane], already masked
    uint32_t sgn[2][kRows];
};

struct Sums { long long t[kTaps]; long long self; };

/// The shipped inner sum: `N^2` plane pairs in lanes, one horizontal add per call.
inline long long pairLanes(const uint32_t (&mag)[kN], uint32_t sgn, const uint32_t (&val)[kN]) {
    const uint32_t both[4] = {val[0] & mag[0], val[1] & mag[0], val[0] & mag[1], val[1] & mag[1]};
    const uint32x4_t vb = vld1q_u32(both);
    const uint32x4_t vs = vandq_u32(vb, vdupq_n_u32(sgn));
    const uint32x4_t cTotal = vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(vb))));
    const uint32x4_t cOpp = vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(vs))));
    const int32x4_t diff = vsubq_s32(vreinterpretq_s32_u32(cTotal),
                                     vshlq_n_s32(vreinterpretq_s32_u32(cOpp), 1));
    const int32_t w[4] = {1, 2, 2, 4};
    return static_cast<long long>(vaddvq_s32(vmulq_s32(diff, vld1q_s32(w))));
}

/// ARM A -- the shipped shape. Ten calls per row; every one of them reduces.
void armPairPerCall(const Window& w, Sums (&out)[2]) {
    for (size_t c = 0; c < 2; ++c) {
        out[c] = Sums{{0, 0, 0, 0}, 0};
    }
    for (size_t y = 0; y < kRows; ++y) {
        for (size_t c = 0; c < 2; ++c) {
            for (size_t t = 0; t < kTaps; ++t) {
                out[c].t[t] += pairLanes(w.mag[c][y], w.sgn[c][y], w.tap[y][t]);
            }
            out[c].self += pairLanes(w.mag[c][y], w.sgn[c][y], w.self[y]);
        }
    }
}

/// ARM B -- taps in lanes, plane pairs folded inside the row, ONE accumulator per
/// component carried to the end of the window.
void armTapAccum(const Window& w, Sums (&out)[2]) {
    int32x4_t acc[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
    long long self[2] = {0, 0};
    for (size_t y = 0; y < kRows; ++y) {
        // The four taps' plane i, in lanes. Two loads, reused by both components.
        uint32_t p0[4], p1[4];
        for (size_t t = 0; t < kTaps; ++t) { p0[t] = w.tap[y][t][0]; p1[t] = w.tap[y][t][1]; }
        const uint32x4_t vp[kN] = {vld1q_u32(p0), vld1q_u32(p1)};
        for (size_t c = 0; c < 2; ++c) {
            const uint32x4_t vsgn = vdupq_n_u32(w.sgn[c][y]);
            for (size_t j = 0; j < kN; ++j) {
                const uint32x4_t vm = vdupq_n_u32(w.mag[c][y][j]);
                for (size_t i = 0; i < kN; ++i) {
                    const uint32x4_t b = vandq_u32(vp[i], vm);
                    const uint32x4_t s = vandq_u32(b, vsgn);
                    const uint32x4_t ct =
                        vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(b))));
                    const uint32x4_t co =
                        vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(s))));
                    const int32x4_t d = vsubq_s32(vreinterpretq_s32_u32(ct),
                                                  vshlq_n_s32(vreinterpretq_s32_u32(co), 1));
                    // The weight is constant across rows, so folding it here and
                    // reducing later is EXACT: sum_rows sum_pairs w*d == sum_pairs w*sum_rows d.
                    acc[c] = vmlaq_n_s32(acc[c], d, static_cast<int32_t>(1 << (i + j)));
                }
            }
            self[c] += pairLanes(w.mag[c][y], w.sgn[c][y], w.self[y]);
        }
    }
    // ONE crossing per component, not one per call.
    for (size_t c = 0; c < 2; ++c) {
        out[c].t[0] = vgetq_lane_s32(acc[c], 0);
        out[c].t[1] = vgetq_lane_s32(acc[c], 1);
        out[c].t[2] = vgetq_lane_s32(acc[c], 2);
        out[c].t[3] = vgetq_lane_s32(acc[c], 3);
        out[c].self = self[c];
    }
}

bool same(const Sums (&a)[2], const Sums (&b)[2]) {
    for (size_t c = 0; c < 2; ++c) {
        if (a[c].self != b[c].self) return false;
        for (size_t t = 0; t < kTaps; ++t) if (a[c].t[t] != b[c].t[t]) return false;
    }
    return true;
}
#endif
} // namespace

int main() {
    std::printf("=== X-40 CEILING: window-carried accumulators at N = 2 ===\n");
#if !WA_NEON
    std::printf("  NEON unavailable here -- this is the reference device's measurement.\n");
    return 0;
#else
    // Twenty windows, so the timing is not one window's worth of noise.
    constexpr size_t kWindows = 20;
    std::vector<Window> ws(kWindows);
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(st >> 33);
    };
    for (Window& w : ws) {
        for (size_t y = 0; y < kRows; ++y) {
            for (size_t t = 0; t < kTaps; ++t)
                for (size_t p = 0; p < kN; ++p) w.tap[y][t][p] = next();
            for (size_t p = 0; p < kN; ++p) w.self[y][p] = next();
            for (size_t c = 0; c < 2; ++c) {
                // Sparse, like a real edge map's gradient magnitude, and masked to
                // 31 columns as D-31's aligned path leaves it.
                for (size_t p = 0; p < kN; ++p) w.mag[c][y][p] = (next() & next()) & 0x7FFFFFFFu;
                w.sgn[c][y] = next();
            }
        }
    }

    Sums a[2], b[2];
    bool equal = true;
    for (const Window& w : ws) {
        armPairPerCall(w, a);
        armTapAccum(w, b);
        if (!same(a, b)) { equal = false; break; }
    }
    std::printf("  EQUALITY over %zu windows: %s\n", kWindows,
                equal ? "all ten sums identical" : "MISMATCH");
    if (!equal) {
        std::printf("    A: t00=%lld t01=%lld t10=%lld t11=%lld self=%lld\n",
                    a[0].t[0], a[0].t[1], a[0].t[2], a[0].t[3], a[0].self);
        std::printf("    B: t00=%lld t01=%lld t10=%lld t11=%lld self=%lld\n",
                    b[0].t[0], b[0].t[1], b[0].t[2], b[0].t[3], b[0].self);
        return 1;
    }

    std::vector<measure::Bench> bs = {
        {"A  shipped: reduce per call", [&](int) {
             Sums s[2];
             for (const Window& w : ws) { armPairPerCall(w, s); measure::g_sink += (size_t)s[0].t[0]; }
         }},
        {"B  proposed: reduce per window", [&](int) {
             Sums s[2];
             for (const Window& w : ws) { armTapAccum(w, s); measure::g_sink += (size_t)s[0].t[0]; }
         }},
    };
    const auto tt = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-34s %12s %10s\n", "arm", "ns/20 windows", "vs A");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-34s %12.1f %9.3fx\n", bs[i].name.c_str(), tt[i].medianNs,
                    tt[0].medianNs / tt[i].medianNs);
    std::printf("\n  X-40's rule: >=2.0x write the arm; 1.4-2.0x write it and price the\n"
                "  frontend effect as modest; <1.4x do NOT write it and close E-18\n"
                "  negative; slower than A means the domain-crossing cost model is wrong.\n");
    return 0;
#endif
}
