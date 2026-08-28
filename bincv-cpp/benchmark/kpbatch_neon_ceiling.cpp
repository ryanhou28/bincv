// ===========================================================================
// X-87 / E-45 -- WOULD D-66'S KEYPOINT BATCH TRANSFER TO NEON?
//
// D-66 put eight keypoints in AVX2 lanes and won 1.37x on x86 `track`. aarch64 got
// nothing from it. X-82 registered a PREDICTION AGAINST porting it, and D-6 in reverse
// is the reason:
//
//   x86 needed keypoint batching because AVX2 HAS NO POPCOUNT AT ALL -- the nibble-table
//   emulation costs six operations per thirty-two bytes and only pays off spread across
//   eight keypoints. aarch64 HAS a vector popcount, and the shipped NEON kernel already
//   fills all four lanes with THE FOUR TAPS. Batching four keypoints would need one
//   vector per (tap, plane pair) instead of one per plane pair -- the same lane-work,
//   rearranged.
//
// THE TWO ARMS, ON ONE WINDOW ROW OF A 31-PIXEL WINDOW AT N = 2:
//
//   (A) the SHIPPED row body, four keypoints in sequence, each reading the CONTIGUOUS
//       per-keypoint layout `StagedWindow` actually has;
//   (B) four keypoints in lanes, reading `[row][plane][lane]`.
//
// X-82 named the denominator traps in advance and all three are honoured here: arm A is
// the shipped kernel's shape including X-86's byte accumulation, it reads the contiguous
// layout and not the batch's, and ALL TEN of its sums are consumed so the compiler
// cannot delete eight of them (which flattered the vector arm by 4x in X-79).
//
// Usage: kpbatch_neon_ceiling
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bincv-cpp/ops/opticalFlow.hpp"

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kRows = 31;
constexpr size_t kLanes = 4;   // uint32 lanes in a 128-bit register
constexpr size_t kN = 2;

struct Sums {
    long long t00 = 0, t01 = 0, t10 = 0, t11 = 0, self = 0;
};

/// ARM A: the shipped kernel's row body. Four taps in the four lanes, byte
/// accumulation to the end of the window (X-86), one keypoint at a time.
void armShipped(const uint32_t* taps, const uint32_t* mag, const uint32_t* sgn,
                const uint32_t* self, Sums& sx, Sums& sy) {
    uint8x16_t tX[4], oX[4], tY[4], oY[4], tS[2], oS[2];
    for (int k = 0; k < 4; ++k) {
        tX[k] = vdupq_n_u8(0); oX[k] = vdupq_n_u8(0);
        tY[k] = vdupq_n_u8(0); oY[k] = vdupq_n_u8(0);
    }
    tS[0] = vdupq_n_u8(0); oS[0] = vdupq_n_u8(0);
    tS[1] = vdupq_n_u8(0); oS[1] = vdupq_n_u8(0);

    for (size_t i = 0; i < kRows; ++i) {
        // `[plane][tap]`, contiguous -- one load a plane (X-85).
        const uint32x4_t vp[2] = {vld1q_u32(taps + i * 8), vld1q_u32(taps + i * 8 + 4)};
        const uint32_t* m = mag + i * 4;      // {magX0, magX1, magY0, magY1}
        const uint32x4_t sgX = vdupq_n_u32(sgn[i * 2]);
        const uint32x4_t sgY = vdupq_n_u32(sgn[i * 2 + 1]);
        const uint32x4_t mx[2] = {vdupq_n_u32(m[0]), vdupq_n_u32(m[1])};
        const uint32x4_t my[2] = {vdupq_n_u32(m[2]), vdupq_n_u32(m[3])};
        for (int k = 0; k < 4; ++k) {
            const uint32x4_t bx = vandq_u32(vp[k & 1], mx[k >> 1]);
            tX[k] = vaddq_u8(tX[k], vcntq_u8(vreinterpretq_u8_u32(bx)));
            oX[k] = vaddq_u8(oX[k], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(bx, sgX))));
            const uint32x4_t by = vandq_u32(vp[k & 1], my[k >> 1]);
            tY[k] = vaddq_u8(tY[k], vcntq_u8(vreinterpretq_u8_u32(by)));
            oY[k] = vaddq_u8(oY[k], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(by, sgY))));
        }
        const uint32x2_t sp = vld1_u32(self + i * 2);
        const uint32x4_t ss = vcombine_u32(sp, sp);
        const uint32x2_t px = vld1_u32(m);
        const uint32x4_t dx = vcombine_u32(px, px);
        const uint32x4_t vbx = vandq_u32(ss, vzip1q_u32(dx, dx));
        const uint32x2_t py = vld1_u32(m + 2);
        const uint32x4_t dy = vcombine_u32(py, py);
        const uint32x4_t vby = vandq_u32(ss, vzip1q_u32(dy, dy));
        tS[0] = vaddq_u8(tS[0], vcntq_u8(vreinterpretq_u8_u32(vbx)));
        oS[0] = vaddq_u8(oS[0], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(vbx, sgX))));
        tS[1] = vaddq_u8(tS[1], vcntq_u8(vreinterpretq_u8_u32(vby)));
        oS[1] = vaddq_u8(oS[1], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(vby, sgY))));
    }

    static const int32_t kW[4] = {1, 2, 2, 4};
    const auto widen = [](uint8x16_t v) { return vpaddlq_u16(vpaddlq_u8(v)); };
    int32x4_t accX = vdupq_n_s32(0), accY = vdupq_n_s32(0);
    for (int k = 0; k < 4; ++k) {
        accX = vmlaq_n_s32(accX,
                           vsubq_s32(vreinterpretq_s32_u32(widen(tX[k])),
                                     vshlq_n_s32(vreinterpretq_s32_u32(widen(oX[k])), 1)),
                           kW[k]);
        accY = vmlaq_n_s32(accY,
                           vsubq_s32(vreinterpretq_s32_u32(widen(tY[k])),
                                     vshlq_n_s32(vreinterpretq_s32_u32(widen(oY[k])), 1)),
                           kW[k]);
    }
    sx.t00 = vgetq_lane_s32(accX, 0); sx.t01 = vgetq_lane_s32(accX, 1);
    sx.t10 = vgetq_lane_s32(accX, 2); sx.t11 = vgetq_lane_s32(accX, 3);
    sy.t00 = vgetq_lane_s32(accY, 0); sy.t01 = vgetq_lane_s32(accY, 1);
    sy.t10 = vgetq_lane_s32(accY, 2); sy.t11 = vgetq_lane_s32(accY, 3);
    const int32x4_t vw = vld1q_s32(kW);
    const int32x4_t dS0 = vsubq_s32(vreinterpretq_s32_u32(widen(tS[0])),
                                    vshlq_n_s32(vreinterpretq_s32_u32(widen(oS[0])), 1));
    const int32x4_t dS1 = vsubq_s32(vreinterpretq_s32_u32(widen(tS[1])),
                                    vshlq_n_s32(vreinterpretq_s32_u32(widen(oS[1])), 1));
    sx.self = vaddvq_s32(vmulq_s32(dS0, vw));
    sy.self = vaddvq_s32(vmulq_s32(dS1, vw));
}

/// ARM B: FOUR KEYPOINTS IN THE FOUR LANES, `[row][plane][lane]`. Every (tap, plane
/// pair) now needs its own vector, because the lanes are spent on keypoints instead.
void armBatched(const uint32_t* taps, const uint32_t* mag, const uint32_t* sgn,
                const uint32_t* self, Sums* sx, Sums* sy) {
    // 4 taps x 4 plane pairs x 2 components, plus 4 plane pairs x 2 for `self`.
    uint8x16_t tX[16], oX[16], tY[16], oY[16], tS[8], oS[8];
    for (int k = 0; k < 16; ++k) {
        tX[k] = vdupq_n_u8(0); oX[k] = vdupq_n_u8(0);
        tY[k] = vdupq_n_u8(0); oY[k] = vdupq_n_u8(0);
    }
    for (int k = 0; k < 8; ++k) { tS[k] = vdupq_n_u8(0); oS[k] = vdupq_n_u8(0); }

    for (size_t i = 0; i < kRows; ++i) {
        // [row][tap][plane][lane]
        const uint32_t* tp = taps + i * 4 * kN * kLanes;
        const uint32_t* mp = mag + i * 2 * kN * kLanes;
        const uint32x4_t sgX = vld1q_u32(sgn + i * 2 * kLanes);
        const uint32x4_t sgY = vld1q_u32(sgn + i * 2 * kLanes + kLanes);
        const uint32x4_t mx[2] = {vld1q_u32(mp), vld1q_u32(mp + kLanes)};
        const uint32x4_t my[2] = {vld1q_u32(mp + 2 * kLanes), vld1q_u32(mp + 3 * kLanes)};
        for (int t = 0; t < 4; ++t) {
            const uint32x4_t vp[2] = {vld1q_u32(tp + (static_cast<size_t>(t) * kN) * kLanes),
                                      vld1q_u32(tp + (static_cast<size_t>(t) * kN + 1) * kLanes)};
            for (int k = 0; k < 4; ++k) {
                const int idx = t * 4 + k;
                const uint32x4_t bx = vandq_u32(vp[k & 1], mx[k >> 1]);
                tX[idx] = vaddq_u8(tX[idx], vcntq_u8(vreinterpretq_u8_u32(bx)));
                oX[idx] = vaddq_u8(oX[idx],
                                   vcntq_u8(vreinterpretq_u8_u32(vandq_u32(bx, sgX))));
                const uint32x4_t by = vandq_u32(vp[k & 1], my[k >> 1]);
                tY[idx] = vaddq_u8(tY[idx], vcntq_u8(vreinterpretq_u8_u32(by)));
                oY[idx] = vaddq_u8(oY[idx],
                                   vcntq_u8(vreinterpretq_u8_u32(vandq_u32(by, sgY))));
            }
        }
        const uint32x4_t sp[2] = {vld1q_u32(self + i * kN * kLanes),
                                  vld1q_u32(self + i * kN * kLanes + kLanes)};
        for (int k = 0; k < 4; ++k) {
            const uint32x4_t bx = vandq_u32(sp[k & 1], mx[k >> 1]);
            tS[k] = vaddq_u8(tS[k], vcntq_u8(vreinterpretq_u8_u32(bx)));
            oS[k] = vaddq_u8(oS[k], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(bx, sgX))));
            const uint32x4_t by = vandq_u32(sp[k & 1], my[k >> 1]);
            tS[4 + k] = vaddq_u8(tS[4 + k], vcntq_u8(vreinterpretq_u8_u32(by)));
            oS[4 + k] = vaddq_u8(oS[4 + k],
                                 vcntq_u8(vreinterpretq_u8_u32(vandq_u32(by, sgY))));
        }
    }

    static const int32_t kW[4] = {1, 2, 2, 4};
    const auto widen = [](uint8x16_t v) { return vpaddlq_u16(vpaddlq_u8(v)); };
    const auto lanes = [&](uint8x16_t tv, uint8x16_t ov) {
        return vsubq_s32(vreinterpretq_s32_u32(widen(tv)),
                         vshlq_n_s32(vreinterpretq_s32_u32(widen(ov)), 1));
    };
    long long outX[4][5] = {}, outY[4][5] = {};
    for (int t = 0; t < 4; ++t) {
        int32x4_t ax = vdupq_n_s32(0), ay = vdupq_n_s32(0);
        for (int k = 0; k < 4; ++k) {
            ax = vmlaq_n_s32(ax, lanes(tX[t * 4 + k], oX[t * 4 + k]), kW[k]);
            ay = vmlaq_n_s32(ay, lanes(tY[t * 4 + k], oY[t * 4 + k]), kW[k]);
        }
        for (int L = 0; L < 4; ++L) {
            outX[L][t] = vgetq_lane_s32(ax, 0); outY[L][t] = vgetq_lane_s32(ay, 0);
            ax = vextq_s32(ax, ax, 1); ay = vextq_s32(ay, ay, 1);
        }
    }
    int32x4_t sxv = vdupq_n_s32(0), syv = vdupq_n_s32(0);
    for (int k = 0; k < 4; ++k) {
        sxv = vmlaq_n_s32(sxv, lanes(tS[k], oS[k]), kW[k]);
        syv = vmlaq_n_s32(syv, lanes(tS[4 + k], oS[4 + k]), kW[k]);
    }
    for (int L = 0; L < 4; ++L) {
        outX[L][4] = vgetq_lane_s32(sxv, 0); outY[L][4] = vgetq_lane_s32(syv, 0);
        sxv = vextq_s32(sxv, sxv, 1); syv = vextq_s32(syv, syv, 1);
    }
    for (int L = 0; L < 4; ++L) {
        sx[L].t00 = outX[L][0]; sx[L].t01 = outX[L][1]; sx[L].t10 = outX[L][2];
        sx[L].t11 = outX[L][3]; sx[L].self = outX[L][4];
        sy[L].t00 = outY[L][0]; sy[L].t01 = outY[L][1]; sy[L].t10 = outY[L][2];
        sy[L].t11 = outY[L][3]; sy[L].self = outY[L][4];
    }
}

long long consume(const Sums& a) {
    return a.t00 + a.t01 + a.t10 + a.t11 + a.self;
}

}  // namespace

int main() {
    std::srand(20260827u);
    auto rnd = []() {
        return (static_cast<uint32_t>(std::rand()) << 17) ^ static_cast<uint32_t>(std::rand());
    };
    // Arm A's operands: contiguous per keypoint, four keypoints side by side.
    std::vector<uint32_t> aTaps(4 * kRows * 8), aMag(4 * kRows * 4), aSgn(4 * kRows * 2),
        aSelf(4 * kRows * 2);
    for (auto& v : aTaps) v = rnd();
    for (auto& v : aMag) v = rnd();
    for (auto& v : aSgn) v = rnd();
    for (auto& v : aSelf) v = rnd();
    // Arm B's operands: the same numbers, transposed into [row][...][lane].
    std::vector<uint32_t> bTaps(kRows * 4 * kN * kLanes), bMag(kRows * 2 * kN * kLanes),
        bSgn(kRows * 2 * kLanes), bSelf(kRows * kN * kLanes);
    for (size_t L = 0; L < kLanes; ++L) {
        for (size_t i = 0; i < kRows; ++i) {
            for (size_t t = 0; t < 4; ++t) {
                for (size_t k = 0; k < kN; ++k) {
                    bTaps[((i * 4 + t) * kN + k) * kLanes + L] =
                        aTaps[L * kRows * 8 + i * 8 + k * 4 + t];
                }
            }
            for (size_t k = 0; k < 4; ++k) {
                bMag[(i * 4 + k) * kLanes + L] = aMag[L * kRows * 4 + i * 4 + k];
            }
            bSgn[(i * 2) * kLanes + L] = aSgn[L * kRows * 2 + i * 2];
            bSgn[(i * 2 + 1) * kLanes + L] = aSgn[L * kRows * 2 + i * 2 + 1];
            for (size_t k = 0; k < kN; ++k) {
                bSelf[(i * kN + k) * kLanes + L] = aSelf[L * kRows * 2 + i * 2 + k];
            }
        }
    }

    constexpr int kRounds = 12, kReps = 20000;
    std::vector<double> ta, tb;
    long long sink = 0;
    for (int r = 0; r < kRounds; ++r) {
        auto t0 = Clock::now();
        for (int rep = 0; rep < kReps; ++rep) {
            for (size_t L = 0; L < kLanes; ++L) {
                Sums sx, sy;
                armShipped(aTaps.data() + L * kRows * 8, aMag.data() + L * kRows * 4,
                           aSgn.data() + L * kRows * 2, aSelf.data() + L * kRows * 2, sx, sy);
                sink += consume(sx) + consume(sy);
            }
        }
        ta.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / kReps);
        t0 = Clock::now();
        for (int rep = 0; rep < kReps; ++rep) {
            Sums sx[4], sy[4];
            armBatched(bTaps.data(), bMag.data(), bSgn.data(), bSelf.data(), sx, sy);
            for (int L = 0; L < 4; ++L) sink += consume(sx[L]) + consume(sy[L]);
        }
        tb.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / kReps);
    }
    const double A = *std::min_element(ta.begin(), ta.end());
    const double B = *std::min_element(tb.begin(), tb.end());
    std::printf("=== X-87 / E-45: would the keypoint batch transfer to NEON? ===\n");
    std::printf("one 31-row window at N=2, four keypoints, %d interleaved rounds, minimum\n\n",
                kRounds);
    std::printf("  (A) shipped kernel, 4 keypoints in sequence   %8.1f ns\n", A);
    std::printf("  (B) 4 keypoints in NEON lanes                 %8.1f ns\n", B);
    std::printf("  ceiling on the residual arithmetic            %8.2fx\n\n", A / B);
    const char* band = A / B >= 2.0   ? "A -- prediction WRONG, write the port"
                     : A / B >= 1.5   ? "B -- only if track > 60% of the device frontend"
                     : A / B >= 1.2   ? "C -- DO NOT write it (X-79: a 3.1x kernel gave 1.37x)"
                                      : "D -- prediction CONFIRMED, E-45 closes NEGATIVE";
    std::printf("  BAND: %s\n", band);
    std::printf("  (sink %lld)\n", sink);
    return 0;
}
#else
int main() {
    std::printf("X-87 / E-45: aarch64 NEON only; nothing to measure here.\n");
    return 0;
}
#endif
