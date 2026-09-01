// ===========================================================================
// earlier work -- WHAT DOES AN INITIAL-FLOW GUESS ACTUALLY BUY, AND WHERE DOES IT
// STOP BUYING IT?
//
// `LKParams::useInitialFlow` seeds each point's search from `nextPts` instead of
// `prevPts`. A VIO frontend fills that array from an IMU pose prediction; HybVIO does
// exactly this. The tempting sentence is "a good guess cuts the iteration count, and
// the iteration loop is ~55% of `track`" -- which is a PERFORMANCE CLAIM, so it is
// measured before it is written, and it is measured against a range of guess errors
// rather than at the one point where it is guaranteed to look good.
//
// THE WORKLOAD IS SYNTHETIC ON PURPOSE. A guess-error curve needs the truth to be
// known exactly, and only a rendered warp gives that: `frame1` is the binarization of
// the TRANSLATED field, not a translation of the binarization, so ground truth is the
// warp's own arithmetic. On a real sequence the "perfect guess" arm cannot be built at
// all, and the number that matters here is the SHAPE of the curve between perfect and
// useless.
//
// Reported per arm: mean iterations per point-level (the thing a guess is supposed to
// change), wall-clock `track`, and RMS endpoint error against ground truth. THE ERROR
// COLUMN IS NOT DECORATION -- a change that speeds `track` up by giving up on
// points has not helped anyone, and a bad guess does exactly that.
//
// Usage: lk_initial_flow
// ===========================================================================

// Before the include: the hook is off by default and this is one of two consumers.
#define BINCV_LK_ITERATION_HISTOGRAM 1

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"

using Clock = std::chrono::steady_clock;
using bincv::Point2f;
using W = uint32_t;

namespace {

constexpr int kWidth = 640, kHeight = 480;
// TWO TRUE MOTIONS, AND THE SMALL ONE IS THE CONTROL. At (5, 3) px the coarsest
// level of a 1/2/2/2 ladder sees (0.63, 0.38) px, so the tracker starts within a
// pixel of the answer WITHOUT any guess and there is nothing for a guess to save --
// which is itself the finding for slow motion. (24, 16) px is where an IMU earns its
// keep: level 3 sees (3, 2) px and the pyramid has real distance to walk.
constexpr int kRounds = 15;

/// test_opticalflow.cpp's texture, verbatim -- four incommensurate terms at four
/// orientations, so the binarized level set has edges at many angles.
double field(double x, double y) {
    return std::sin(x / 7.3 + 0.4) * std::cos(y / 5.1) + 0.6 * std::sin((x + y) / 11.7) +
           0.5 * std::cos((x - 2.0 * y) / 9.3) + 0.4 * std::sin(x / 3.1) * std::sin(y / 3.7);
}

struct Rng {
    uint64_t s = 0x13198A2E03707344ull;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    /// A unit vector times `mag`, so a guess error of `mag` px really is `mag` px
    /// regardless of direction -- a per-axis uniform would make the actual error
    /// depend on the angle and blur the curve this benchmark exists to draw.
    void offset(float mag, float& ox, float& oy) {
        const double a = 6.283185307179586 * (static_cast<double>(next() & 0xFFFF) / 65536.0);
        ox = static_cast<float>(std::cos(a) * static_cast<double>(mag));
        oy = static_cast<float>(std::sin(a) * static_cast<double>(mag));
    }
};

double minOf(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

}  // namespace

int main() {
    std::printf("=== what an initial-flow guess buys ===\n");
    std::printf("640x480, ladder 1/2/2/2, 31x31 window, %d interleaved rounds, minimum\n\n",
                kRounds);
#if defined(BINCV_X86_LK_BATCH)
    std::printf("shipped path: %s; the iteration counter only sees the SCALAR path,\n"
                "because the keypoint batch does not go through `trackOnePoint` -- so the\n"
                "table times BOTH and counts iterations on the one that can be counted.\n\n",
                bincv::lkPathName<bincv::LKLevelN<1, W>>());
#else
    std::printf("shipped path: %s (no keypoint batch on this architecture, so the\n"
                "shipped and scalar columns are the same code).\n\n",
                bincv::lkPathName<bincv::LKLevelN<1, W>>());
#endif

    for (const auto& motion : {std::pair<double, double>{5.0, 3.0},
                              std::pair<double, double>{24.0, 16.0}}) {
    const double kTx = motion.first, kTy = motion.second;
    bincv::Pyramid<W, 1, 2, 2, 2> prev(kWidth, kHeight), next(kWidth, kHeight);
    bincv::SignedQuantMat<1, W> dx0(kWidth, kHeight), dy0(kWidth, kHeight);
    bincv::SignedQuantMat<2, W> dx1(320, 240), dy1(320, 240), dx2(160, 120), dy2(160, 120),
        dx3(80, 60), dy3(80, 60);

    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            prev.level<0>().set(y, x, field(x, y) > 0.0 ? 1u : 0u);
            next.level<0>().set(y, x, field(static_cast<double>(x) - kTx,
                                            static_cast<double>(y) - kTy) > 0.0 ? 1u : 0u);
        }
    }
    prev.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
    next.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
    bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
    bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
    bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
    bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);

    bincv::LKLevels<W, 1, 2, 2, 2> levels;
    levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
    levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
    levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
    levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);

    std::vector<Point2f> pts;
    for (int y = 48; y < kHeight - 48; y += 16) {
        for (int x = 48; x < kWidth - 48; x += 16) {
            pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    const size_t n = pts.size();

    std::printf("TRUE MOTION (%.0f, %.0f) px -- level 3 sees (%.2f, %.2f)\n", kTx, kTy,
                kTx / 8.0, kTy / 8.0);
    std::vector<unsigned> counts(4 * n);
    bincv::impl::iterationTrace().counts = counts.data();
    bincv::impl::iterationTrace().pointCount = n;

    std::printf("%-24s %11s %11s %11s %9s %8s\n", "arm", "shipped ns", "scalar ns",
                "iters/pt-lvl", "rms px", "tracked");
    std::printf("%s\n", "--------------------------------------------------------------------"
                        "--------");

    struct Row { const char* label; float err; bool guess; };
    const Row rows[] = {
        {"no guess (prevPts)",       0.0f,  false},
        {"guess: exact",             0.0f,  true},
        {"guess: 0.25 px off",       0.25f, true},
        {"guess: 0.5 px off",        0.5f,  true},
        {"guess: 1 px off",          1.0f,  true},
        {"guess: 2 px off",          2.0f,  true},
        {"guess: 4 px off",          4.0f,  true},
        {"guess: 8 px off",          8.0f,  true},
    };

    double baseNs = 0.0, baseIters = 0.0;
    for (const Row& row : rows) {
        bincv::LKParams params;
        params.useInitialFlow = row.guess;

        // Built once so every round tracks the SAME starting estimates -- a per-round
        // redraw would put the RNG inside the timing loop and change the workload
        // between the rounds whose minimum is being taken.
        std::vector<Point2f> seed(n);
        Rng rng;
        for (size_t i = 0; i < n; ++i) {
            float ox = 0.0f, oy = 0.0f;
            if (row.err > 0.0f) rng.offset(row.err, ox, oy);
            seed[i] = Point2f{pts[i].x + static_cast<float>(kTx) + ox,
                              pts[i].y + static_cast<float>(kTy) + oy};
        }

        std::vector<Point2f> out(n);
        std::vector<uint8_t> status(n);
        std::vector<double> tShipped, tScalar;
        for (int r = 0; r < kRounds; ++r) {
#if defined(BINCV_X86_LK_BATCH)
            bincv::impl::lkBatchEnabled() = true;
#endif
            if (row.guess) out = seed; else out.assign(n, Point2f{});
            auto t0 = Clock::now();
            bincv::calcOpticalFlowPyrLK(levels, pts.data(), out.data(), status.data(), nullptr, n,
                                        params);
            tShipped.push_back(
                std::chrono::duration<double, std::nano>(Clock::now() - t0).count());

            // THE ITERATION COUNTER ONLY SEES THIS PATH. The AVX2 keypoint batch does
            // not call `trackOnePoint`, so with the batch on the histogram records
            // nothing at all -- the first run of this benchmark reported 0.000
            // iterations for every arm and it read like a converged tracker.
#if defined(BINCV_X86_LK_BATCH)
            bincv::impl::lkBatchEnabled() = false;
#endif
            if (row.guess) out = seed; else out.assign(n, Point2f{});
            t0 = Clock::now();
            bincv::calcOpticalFlowPyrLK(levels, pts.data(), out.data(), status.data(), nullptr, n,
                                        params);
            tScalar.push_back(std::chrono::duration<double, std::nano>(Clock::now() - t0).count());
#if defined(BINCV_X86_LK_BATCH)
            bincv::impl::lkBatchEnabled() = true;
#endif
        }

        // ONE MORE CALL, WITH THE COUNTER ZEROED FIRST. `iterationTrace` ACCUMULATES:
        // it is written by every call, and the first run of this benchmark read the
        // running total over all 15 rounds and all previous arms -- reporting 242
        // iterations per point-level against a cap of 20, which is impossible and was
        // the tell. Counting once, after the timing, keeps the timed loop clean too.
        std::fill(counts.begin(), counts.end(), 0u);
#if defined(BINCV_X86_LK_BATCH)
        bincv::impl::lkBatchEnabled() = false;
#endif
        if (row.guess) out = seed; else out.assign(n, Point2f{});
        bincv::calcOpticalFlowPyrLK(levels, pts.data(), out.data(), status.data(), nullptr, n,
                                    params);
#if defined(BINCV_X86_LK_BATCH)
        bincv::impl::lkBatchEnabled() = true;
#endif

        unsigned long long iterSum = 0;
        size_t pointLevels = 0;
        for (size_t i = 0; i < 4 * n; ++i) {
            iterSum += counts[i];
            if (counts[i] > 0) ++pointLevels;
        }
        double sumSq = 0.0;
        size_t tracked = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!status[i]) continue;
            const double ex = static_cast<double>(out[i].x) - (static_cast<double>(pts[i].x) + kTx);
            const double ey = static_cast<double>(out[i].y) - (static_cast<double>(pts[i].y) + kTy);
            sumSq += ex * ex + ey * ey;
            ++tracked;
        }
        const double ns = minOf(tShipped);
        const double sc = minOf(tScalar);
        const double iters = pointLevels ? static_cast<double>(iterSum) /
                                               static_cast<double>(pointLevels)
                                         : 0.0;
        const double rms = tracked ? std::sqrt(sumSq / static_cast<double>(tracked)) : 0.0;
        if (baseNs == 0.0) { baseNs = ns; baseIters = iters; }
        std::printf("%-24s %11.0f %11.0f %11.3f %9.4f %8zu %5.2fx time %5.2fx iters\n",
                    row.label, ns, sc, iters, rms, tracked, baseNs / ns,
                    iters > 0.0 ? baseIters / iters : 0.0);
    }
    bincv::impl::iterationTrace().counts = nullptr;
    std::printf("\n");
    }

    std::printf("\nThe first row is the denominator: it is the shipped tracker, unchanged.\n"
                "A guess is only a win where BOTH the time ratio is above 1.00 AND the rms\n"
                "column has not moved -- buying speed by dropping points is not a win.\n");
    return 0;
}
