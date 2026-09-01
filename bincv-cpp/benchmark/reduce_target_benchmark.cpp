// Two questions about ops/reduce.hpp that only the REFERENCE DEVICE can answer,
// each with its decision rule written here before anything was measured
// (CLAUDE.md, "How performance and footprint decisions get made").
//
// This file compares binCV against BINCV -- alternative implementations of the
// same reduction -- so unlike benchmark/reduce_benchmark.cpp it has no OpenCV
// denominator and needs none. the design notes's denominator answers "is binCV
// worth using"; these two questions are "is this file's interface buying what it
// claims" and "does composing the covariance out of it cost anything", and OpenCV
// is not a party to either.
//
// ===========================================================================
// Q1 -- the design rule’s claim, on the target this was derived from
// ===========================================================================
//
// the design notes says reductions are bulk-only "so the implementation keeps
// data in vector registers and accumulates with cnt + uaddlv without crossing
// back". also says the implementation is scalar __builtin_popcountll for
// now. Both cannot describe the same code, and on aarch64 the difference is the
// entire argument for earlier work. So: measure the shipped bulk entry point against the
// per-word popcount loop the design rule forbids exposing, at the same load width, on the
// same data.
//
// DECISION RULE (written first):
// * bulk >= 1.15x the per-word loop -> 6.2's present tense is defensible; fix
// only the instruction sequence it quotes.
// * bulk within +/-15% of the per-word loop -> the INTERFACE decision
// stands, but 6.2's IMPLEMENTATION claim is false today. Separate the two in
// 6.2 and in ops/reduce.hpp, record the numbers in, and
// change no kernel: vectorization is Phase 5, and this is a documentation
// defect, not a kernel defect.
// * Either way, no -march flag and no intrinsics enter the LIBRARY -- the same
// standing decision that measurement’s x86_64 half already recorded.
//
// A third row runs on aarch64 only and is a HEADROOM PROBE, not a candidate
// implementation: the identical 64-bit loads with a VECTOR accumulator (cnt into
// a running total via vpadal_u8), so exactly one register-domain crossing per
// ROW instead of one per word. It exists so the Phase 5 task starts from a
// measured number on this device instead of from an argument. Nothing in
// include/ is changed by this file, and nothing here is included by include/.
//
// ===========================================================================
// Q2 -- what composing the 2x2 covariance out of these primitives costs
// ===========================================================================
//
// the design notes's covariance needs four numbers over one window. Through the
// primitives that is THREE calls -- countNonZero(mag_x), countNonZero(mag_y),
// countAndSplit(mag_x, mag_y, sign_x^sign_y) -- and therefore three traversals of
// the same window, issuing the same popcounts a single fused traversal would.
// countAndSplit is single-pass, as requires; the COMPOSITION is not, and no
// experiment has looked at that axis. is scoped to
// incremental-versus-recompute and would not measure it.
//
// DECISION RULE (written first):
// * composition within 15% of a fused traversal -> the composition is free
// enough; record it and close the question.
// * composition costs > 15% more -> widen that work’s brief to measure a
// covariance-shaped entry point against the composition BEFORE this is
// written against either, and register that in TASKS.md and the design notes.
// Do NOT add the entry point here: choosing that work’s interface on the strength
// of one measurement with no decision rule is the thing forbids for
// incremental state, and the same reasoning binds this.
// * 15% is that work’s own existing threshold, adopted rather than invented, so
// that two questions about the same interface are not judged on two scales.
//
// The fused traversal below is MEASUREMENT CODE. It reaches into impl:: -- which
// carries no stability promise and is exactly what tests/test_reduce.cpp already
// drives directly -- and it is not proposed for include/.
//
// ===========================================================================
// MEASUREMENT VALIDITY -- the four hazards benchmark/reduce_benchmark.cpp lists
//
// 1. DEAD CODE: every result is folded into a volatile sink and printed.
// 2. CONSTANT FOLDING: inputs rotate through four distinct random images.
// 3. CALIBRATED BATCHES: iteration counts are calibrated to a time budget and
// the reported figure is the minimum over several batches.
// 4. THE SIDES MUST AGREE: every implementation is compared against the shipped
// one before anything is timed; a disagreement exits non-zero rather than
// printing a table under a caveat.
//
// x86_64 numbers from this file are INDICATIVE ONLY. Both questions are about the
// primary target and close on the reference device (scripts/run_on_pi.sh).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::Rect;
using bincv::SplitCount;

constexpr int kInputs = 4;
constexpr int kRepeats = 7;
constexpr double kTargetMs = 40.0;

volatile size_t g_sink = 0;

using Clock = std::chrono::high_resolution_clock;

uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

template <typename Body>
int calibrate(Body body, double targetMs) {
    int iterations = 8;
    for (int attempt = 0; attempt < 24; ++attempt) {
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (ms >= targetMs || iterations >= (1 << 22)) return iterations;
        const double scale = (ms > 0.0) ? (targetMs / ms) : 8.0;
        iterations = static_cast<int>(static_cast<double>(iterations) *
                                      std::min(scale * 1.3, 16.0)) + 1;
    }
    return iterations;
}

template <typename Body>
double measureNs(Body body, int repeats, double targetMs) {
    const int iterations = calibrate(body, targetMs);
    double bestNs = -1.0;
    for (int r = 0; r < repeats; ++r) {
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ns =
            std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
            static_cast<double>(iterations);
        if (bestNs < 0.0 || ns < bestNs) bestNs = ns;
    }
    return bestNs;
}

// ---------------------------------------------------------------------------
// Q1: the three ways to count the same L1-resident image
// ---------------------------------------------------------------------------
//
// Width is a whole number of 64-bit words on purpose, so no implementation pays
// for a partial trailing word and the comparison is about the LOOP rather than
// about the masking around it. The shipped kernel still applies its one AND per
// row; that is part of what it costs and is not subtracted out.

/// @brief The per-word popcount loop the design rule forbids the PUBLIC API from making
/// possible. Written here as the thing being measured against, and
/// nowhere else in the project.
size_t countPerWordLoop(const BinMatConstView<uint64_t>& v, size_t wordsPerRow) {
    size_t total = 0;
    for (size_t y = 0; y < v.height; ++y) {
        const uint64_t* row = v.row(y);
        for (size_t i = 0; i < wordsPerRow; ++i) {
            total += static_cast<size_t>(__builtin_popcountll(row[i]));
        }
    }
    return total;
}

#if defined(__aarch64__)
/// @brief HEADROOM PROBE, aarch64 only -- NOT a proposed implementation.
/// @note Identical 64-bit loads; the only difference is that the running total
/// stays in a NEON register (vpadal_u8) and crosses to a GPR once per ROW
/// instead of once per word. Per-lane totals cannot overflow: a row here is
/// 64 words, each u16 lane accumulates two u8 lanes of at most 8, so the
/// largest lane value is 64 * 16 = 1024.
size_t countVectorAccum(const BinMatConstView<uint64_t>& v, size_t wordsPerRow) {
    size_t total = 0;
    for (size_t y = 0; y < v.height; ++y) {
        const uint8_t* row = reinterpret_cast<const uint8_t*>(v.row(y));
        uint16x4_t acc = vdup_n_u16(0);
        for (size_t i = 0; i < wordsPerRow; ++i) {
            acc = vpadal_u8(acc, vcnt_u8(vld1_u8(row + i * 8)));
        }
        total += static_cast<size_t>(vaddlv_u16(acc));
    }
    return total;
}
#endif

bool runQ1() {
    // 4096 x 8 at one bit per pixel is 4 KiB -- comfortably L1-resident on the
    // reference device (32 KiB L1D), so this measures the loop and not the memory
    // system. 4096 is 64 whole uint64 words.
    const int width = 4096;
    const int height = 8;
    const size_t wordsPerRow = static_cast<size_t>(width) / 64u;
    const double pixels = static_cast<double>(width) * static_cast<double>(height);

    std::vector<BinMat<uint64_t>> images;
    images.reserve(kInputs);
    for (int i = 0; i < kInputs; ++i) {
        BinMat<uint64_t> m(width, height);
        uint64_t state = UINT64_C(0xD6D6D6) + static_cast<uint64_t>(i) * UINT64_C(7919);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if ((nextRandom(state) >> 63) != 0) m.set(y, x, true);
            }
        }
        images.push_back(std::move(m));
    }

    std::printf("\n=== Q1: does the bulk entry point beat the per-word loop forbids? ===\n");
    std::printf(" %d x %d uint64, %.0f pixels, %zu B per image (L1-resident)\n", width, height,
                pixels, static_cast<size_t>(width / 8) * static_cast<size_t>(height));

    // Hazard 4: agreement before timing.
    for (const BinMat<uint64_t>& m : images) {
        const BinMatConstView<uint64_t> v = m.constView();
        const size_t shipped = bincv::countNonZero(v);
        const size_t perWord = countPerWordLoop(v, wordsPerRow);
        if (shipped != perWord) {
            std::printf(" DISAGREEMENT: shipped %zu vs per-word loop %zu\n", shipped, perWord);
            return false;
        }
#if defined(__aarch64__)
        const size_t vec = countVectorAccum(v, wordsPerRow);
        if (shipped != vec) {
            std::printf(" DISAGREEMENT: shipped %zu vs vector accumulator %zu\n", shipped, vec);
            return false;
        }
#endif
    }
    std::printf(" all implementations agree (%zu set pixels in image 0)\n",
                bincv::countNonZero(images[0].constView()));

    const double nsBulk = measureNs(
        [&](int i) { g_sink += bincv::countNonZero(images[static_cast<size_t>(i % kInputs)].constView()); },
        kRepeats, kTargetMs);
    const double nsPerWord = measureNs(
        [&](int i) {
            g_sink += countPerWordLoop(images[static_cast<size_t>(i % kInputs)].constView(), wordsPerRow);
        },
        kRepeats, kTargetMs);

    std::printf("[BENCH] %-34s %10.5f ns/px\n", "countNonZero (shipped, bulk API)", nsBulk / pixels);
    std::printf("[BENCH] %-34s %10.5f ns/px\n", "caller-written per-word loop", nsPerWord / pixels);
    std::printf(" bulk / per-word: %.2fx (>1 means the bulk API is faster)\n", nsPerWord / nsBulk);

#if defined(__aarch64__)
    const double nsVec = measureNs(
        [&](int i) {
            g_sink += countVectorAccum(images[static_cast<size_t>(i % kInputs)].constView(), wordsPerRow);
        },
        kRepeats, kTargetMs);
    std::printf("[BENCH] %-34s %10.5f ns/px <- HEADROOM PROBE, not shipped\n",
                "vector accumulator (Phase 5)", nsVec / pixels);
    std::printf(" headroom available to Phase 5 at the same load width: %.2fx\n", nsBulk / nsVec);
#else
    std::printf(" (the vector-accumulator headroom probe is aarch64-only)\n");
#endif
    return true;
}

// ---------------------------------------------------------------------------
// Q2: the covariance, composed out of the primitives versus fused
// ---------------------------------------------------------------------------

struct Covariance {
    size_t xx = 0;
    size_t yy = 0;
    SplitCount xy;

    bool operator==(const Covariance& o) const {
        return xx == o.xx && yy == o.yy && xy.whenClear == o.xy.whenClear &&
               xy.whenSet == o.xy.whenSet;
    }
};

/// @brief the design notes through the primitives -- three calls, and
/// therefore three traversals of the same window.
Covariance covarianceComposed(const BinMatConstView<uint64_t>& magX,
                              const BinMatConstView<uint64_t>& magY,
                              const BinMatConstView<uint64_t>& signXor, const Rect& window) {
    Covariance out;
    out.xx = bincv::countNonZero(magX, window);
    out.yy = bincv::countNonZero(magY, window);
    out.xy = bincv::countAndSplit(magX, magY, signXor, window);
    return out;
}

/// @brief The same four numbers from ONE traversal. MEASUREMENT CODE ONLY -- see
/// the header comment; this is not proposed for include/.
Covariance covarianceFused(const BinMatConstView<uint64_t>& magX,
                           const BinMatConstView<uint64_t>& magY,
                           const BinMatConstView<uint64_t>& signXor, const Rect& window) {
    Covariance out;
    const bincv::impl::RegionWords<uint64_t> r =
        bincv::impl::clipRegion<uint64_t>(magX.width, magX.height, window);
    if (r.isEmpty) return out;

    for (size_t y = r.y0; y < r.y1; ++y) {
        const uint64_t* rx = magX.row(y);
        const uint64_t* ry = magY.row(y);
        const uint64_t* rc = signXor.row(y);
        bincv::impl::visitRowWords<uint64_t>(r, [&](size_t i, uint64_t mask) {
            const uint64_t wx = rx[i] & mask;
            const uint64_t wy = ry[i] & mask;
            const uint64_t both = wx & wy;
            const size_t total = bincv::impl::popcountWord<uint64_t>(both);
            const size_t set = bincv::impl::popcountWord<uint64_t>(both & rc[i]);
            out.xx += bincv::impl::popcountWord<uint64_t>(wx);
            out.yy += bincv::impl::popcountWord<uint64_t>(wy);
            out.xy.whenSet += set;
            out.xy.whenClear += total - set;
        });
    }
    return out;
}

bool runQ2() {
    // 640x480 and 200 keypoints: the frame size and the keypoint count TASKS.md
    // names (the reference gftt_max_corners), with the 31x31 window
    // the design notes specifies.
    const int width = 640;
    const int height = 480;
    const int windowSize = 31;
    const int keypoints = 200;

    std::vector<bincv::TernaryMat<uint64_t>> dxs;
    std::vector<bincv::TernaryMat<uint64_t>> dys;
    std::vector<BinMat<uint64_t>> xors;
    dxs.reserve(kInputs);
    dys.reserve(kInputs);
    xors.reserve(kInputs);
    for (int i = 0; i < kInputs; ++i) {
        bincv::TernaryMat<uint64_t> dx(width, height);
        bincv::TernaryMat<uint64_t> dy(width, height);
        uint64_t state = UINT64_C(0x7F5C0) + static_cast<uint64_t>(i) * UINT64_C(104729);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                dx.set(y, x, static_cast<int>(nextRandom(state) % 3) - 1);
                dy.set(y, x, static_cast<int>(nextRandom(state) % 3) - 1);
            }
        }
        BinMat<uint64_t> sx(width, height);
        bincv::bitwiseXor(dx.constSign(), dy.constSign(), sx.view());
        dxs.push_back(std::move(dx));
        dys.push_back(std::move(dy));
        xors.push_back(std::move(sx));
    }

    // Keypoints, including ones near enough to an edge that the window clips --
    // which is the realistic case (the design notes) and not a corner case.
    std::vector<Rect> windows;
    windows.reserve(static_cast<size_t>(keypoints));
    {
        uint64_t state = UINT64_C(0xC0FFEE);
        for (int i = 0; i < keypoints; ++i) {
            const int cx = static_cast<int>(nextRandom(state) % static_cast<uint64_t>(width));
            const int cy = static_cast<int>(nextRandom(state) % static_cast<uint64_t>(height));
            windows.push_back(Rect(cx - windowSize / 2, cy - windowSize / 2, windowSize, windowSize));
        }
    }

    std::printf("\n=== Q2: the 2x2 covariance composed out of versus one fused pass ===\n");
    std::printf(" %d x %d uint64, %d keypoints, %dx%d windows (ARCHITECTURE 7.5)\n", width, height,
                keypoints, windowSize, windowSize);

    // Hazard 4: agreement on every window before anything is timed.
    for (int i = 0; i < kInputs; ++i) {
        const size_t k = static_cast<size_t>(i);
        for (const Rect& w : windows) {
            const Covariance a = covarianceComposed(dxs[k].constMagnitude(0),
                                                    dys[k].constMagnitude(0),
                                                    xors[k].constView(), w);
            const Covariance b = covarianceFused(dxs[k].constMagnitude(0),
                                                 dys[k].constMagnitude(0),
                                                 xors[k].constView(), w);
            if (!(a == b)) {
                std::printf(" DISAGREEMENT on a window: composed {%zu,%zu,%zu,%zu} "
                            "fused {%zu,%zu,%zu,%zu}\n",
                            a.xx, a.yy, a.xy.whenClear, a.xy.whenSet, b.xx, b.yy, b.xy.whenClear,
                            b.xy.whenSet);
                return false;
            }
        }
    }
    std::printf(" composed and fused agree on all %d windows x %d images\n", keypoints, kInputs);

    const double nsComposed = measureNs(
        [&](int i) {
            const size_t k = static_cast<size_t>(i % kInputs);
            size_t acc = 0;
            for (const Rect& w : windows) {
                const Covariance c = covarianceComposed(dxs[k].constMagnitude(0),
                                                        dys[k].constMagnitude(0),
                                                        xors[k].constView(), w);
                acc += c.xx + c.yy + c.xy.whenClear + c.xy.whenSet;
            }
            g_sink += acc;
        },
        kRepeats, kTargetMs);

    const double nsFused = measureNs(
        [&](int i) {
            const size_t k = static_cast<size_t>(i % kInputs);
            size_t acc = 0;
            for (const Rect& w : windows) {
                const Covariance c = covarianceFused(dxs[k].constMagnitude(0),
                                                     dys[k].constMagnitude(0),
                                                     xors[k].constView(), w);
                acc += c.xx + c.yy + c.xy.whenClear + c.xy.whenSet;
            }
            g_sink += acc;
        },
        kRepeats, kTargetMs);

    const double perKp = static_cast<double>(keypoints);
    std::printf("[BENCH] %-34s %8.4f ms/frame %8.1f ns/keypoint\n",
                "covariance composed (as shipped)", nsComposed / 1e6, nsComposed / perKp);
    std::printf("[BENCH] %-34s %8.4f ms/frame %8.1f ns/keypoint\n",
                "covariance fused (one traversal)", nsFused / 1e6, nsFused / perKp);
    std::printf(" composed / fused: %.2fx (decision rule: > 1.15x widens the brief)\n",
                nsComposed / nsFused);
    return true;
}

}  // namespace

int main() {
    std::printf("=== binCV reduction: target-specific questions ( triage) ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE for both questions\n");
#else
    std::printf("target: not aarch64 -- INDICATIVE ONLY; both questions close on the "
                "reference device\n");
#endif
    std::printf("Decision rules are in this file's header, written before measuring.\n");

    const bool ok1 = runQ1();
    const bool ok2 = runQ2();
    std::printf("\nsink: %zu (printed so no reduction above can be optimized away)\n",
                static_cast<size_t>(g_sink));
    return (ok1 && ok2) ? 0 : 1;
}
