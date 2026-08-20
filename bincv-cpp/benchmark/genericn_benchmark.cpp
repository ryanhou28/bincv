// T3.9 / E-4 -- EXPERIMENTS.md X-21. Does generic-N cost the specialized N = 1
// and ternary paths anything?
//
// THREE ARMS, one per translation unit (see benchmark/genericn_arms.hpp):
//
//   generic-N      the library's generic route forced on at N = 1: the
//                  ripple-borrow subtract that serves any N, and a compile-time
//                  plane loop around the reductions.
//   specialized    what ships: derivativeX / derivativeY on a QuantMat<1> into a
//                  TernaryMat, and the single-plane reductions.
//   hand-written   binary-only C++ that includes no binCV header at all -- no
//                  container, no view, no template over N or over the word type,
//                  no route selector, no argument contract.
//
// The third arm is what makes the comparison mean anything. Arms 1 and 2 are both
// binCV, so between them they show only whether the specialization is SELECTED.
// The decision rule in TASKS.md T3.9 is written against arm 3 by name.
//
// WORKLOAD   T3.5's derivative (dx + dy of one frame) and T2.5's reductions (a
//            whole-frame count, and the T3.6 covariance over 200 31x31 windows --
//            200 is X-20's keypoint count and 31x31 is the LK window
//            seal_params.yaml sets). 640x480 and 94x60, the frame and the pyramid
//            level X-11 and X-16 already report at, so these numbers sit beside
//            theirs. uint32_t, D-14's default and the width the hand-written arm
//            is written in.
// METRIC     ns/pixel, median of the batches with the min and max beside it, and
//            the ratio of each arm to the hand-written one. Code size is the
//            other half of the metric and is NOT measured here -- it is `size` on
//            the three arm objects, run on the device; see the footer this
//            program prints.
//
// VALIDITY (EXPERIMENTS.md "Rules"): every result is folded into a volatile sink,
// the inputs are four distinct pseudo-random frames rotated across calls, and all
// three arms must produce BIT-IDENTICAL destination buffers and identical
// reduction values on every input before anything is timed. A disagreement is
// reported as a defect and the case is not timed at all.
//
// NO OPENCV. All three arms are binCV or plain C++, so ARCHITECTURE 10.3's
// denominator does not apply -- this is not a claim about binCV against OpenCV --
// and the binary builds in the reference device's DEFAULT core-only build.
//
// On x86_64 this is INDICATIVE ONLY (EXPERIMENTS.md, "Measurement platforms").
// The authoritative run is
//
//   ./scripts/run_on_pi.sh pi4 './benchmark/genericn_benchmark'

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "genericn_arms.hpp"
#include "measure_util.hpp"

namespace {

using t39::Arm;
using t39::Cov;
using t39::Word;

constexpr int kInputs = 4;       // distinct frames, rotated across calls
constexpr int kRepeats = 15;     // batches per arm; the spread across them is reported
constexpr double kTargetMs = 50.0;
constexpr int kWindow = 31;      // the LK window seal_params.yaml uses
constexpr int kWindows = 200;    // keypoints per frame, from X-20's footprint table

/// @brief One frame's storage: a binary source and the two ternary derivatives
///        each arm writes, laid out as SignedQuantMat<1> expects.
struct Frame {
    std::vector<Word> src;
    size_t stride = 0;
    int width = 0;
    int height = 0;
};

/// @brief Destination pair for one arm: dx and dy, two planes each.
struct Dest {
    std::vector<Word> dx;
    std::vector<Word> dy;
};

Frame makeFrame(int width, int height, uint64_t seed) {
    Frame f;
    f.width = width;
    f.height = height;
    f.stride = (static_cast<size_t>(width) + 31u) / 32u;
    f.src.assign(f.stride * static_cast<size_t>(height), 0u);

    uint64_t state = seed;
    const size_t words = (static_cast<size_t>(width) + 31u) / 32u;
    const uint32_t tail = (width % 32 == 0)
                              ? UINT32_C(0xFFFFFFFF)
                              : static_cast<uint32_t>((UINT32_C(1) << (width % 32)) - 1u);
    for (size_t y = 0; y < static_cast<size_t>(height); ++y) {
        for (size_t i = 0; i < words; ++i) {
            const uint32_t w = static_cast<uint32_t>(measure::nextRandom(state));
            // Padding past `width` starts and stays clear (D-13): every arm counts
            // word-wise, and a dirty padding bit would be an over-count in one arm
            // and a masked zero in another.
            f.src[y * f.stride + i] = (i + 1u == words) ? static_cast<uint32_t>(w & tail) : w;
        }
    }
    return f;
}

Dest makeDest(const Frame& f) {
    Dest d;
    const size_t planePair = 2u * f.stride * static_cast<size_t>(f.height);
    d.dx.assign(planePair, 0u);
    d.dy.assign(planePair, 0u);
    return d;
}

/// @brief The windows the covariance sweep visits. Fixed for the whole run so
///        every arm sees the same positions in the same order.
std::vector<std::pair<int, int>> makeWindows(int width, int height, uint64_t seed) {
    std::vector<std::pair<int, int>> out;
    out.reserve(static_cast<size_t>(kWindows));
    uint64_t state = seed;
    const int spanX = (width > kWindow) ? width - kWindow : 1;
    const int spanY = (height > kWindow) ? height - kWindow : 1;
    for (int i = 0; i < kWindows; ++i) {
        const int x = static_cast<int>(measure::nextRandom(state) % static_cast<uint64_t>(spanX));
        const int y = static_cast<int>(measure::nextRandom(state) % static_cast<uint64_t>(spanY));
        out.emplace_back(x, y);
    }
    return out;
}

/// @brief Sums the covariance over the whole window sweep, so one timed call is
///        200 windows rather than one -- 961 pixels is far below a clock tick.
Cov sweepCovariance(const Arm& arm, const Dest& d, const Frame& f,
                    const std::vector<std::pair<int, int>>& windows) {
    Cov total;
    for (const auto& w : windows) {
        const Cov c = arm.covarianceWindow(d.dx.data(), d.dy.data(), f.stride, f.width, f.height,
                                           w.first, w.second, kWindow);
        total.xx += c.xx;
        total.yy += c.yy;
        total.whenClear += c.whenClear;
        total.whenSet += c.whenSet;
    }
    return total;
}

void printRow(const char* workload, const char* armName, const measure::Timing& t, double pixels,
              double handMedian) {
    const double nsPerPixel = t.medianNs / pixels;
    const double ratio = (handMedian > 0.0) ? t.medianNs / handMedian : 0.0;
    std::printf("  %-22s %-13s %8.4f  [%7.4f, %7.4f]  %5.1f%%   %5.3fx\n", workload, armName,
                nsPerPixel, t.minNs / pixels, t.maxNs / pixels, t.spreadPct(), ratio);
}

/// @brief One frame size, all three arms, all three workloads.
/// @return false if any two arms disagree -- a defect, and nothing is timed.
bool runSize(int width, int height) {
    const Arm* arms[3] = {&t39::handWrittenArm(), &t39::specializedArm(), &t39::genericArm()};

    std::vector<Frame> frames;
    frames.reserve(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        frames.push_back(makeFrame(width, height,
                                   UINT64_C(0x3907) + static_cast<uint64_t>(i) * UINT64_C(7919)));
    }
    const std::vector<std::pair<int, int>> windows =
        makeWindows(width, height, UINT64_C(0xC0FFEE));

    // One destination pair per arm, so the agreement check below compares whole
    // buffers rather than trusting that a later arm overwrote an earlier one.
    // Four: three arms, plus one for whichever decomposition point is running.
    Dest dests[4] = {makeDest(frames[0]), makeDest(frames[0]), makeDest(frames[0]),
                     makeDest(frames[0])};

    std::printf("\n  %dx%d, uint32_t, stride %zu words\n", width, height, frames[0].stride);

    // ---- VALIDITY: the three arms must agree before anything is timed --------
    for (int f = 0; f < kInputs; ++f) {
        const Frame& fr = frames[static_cast<size_t>(f)];
        size_t counts[3];
        Cov covs[3];
        for (int a = 0; a < 3; ++a) {
            std::memset(dests[a].dx.data(), 0, dests[a].dx.size() * sizeof(Word));
            std::memset(dests[a].dy.data(), 0, dests[a].dy.size() * sizeof(Word));
            arms[a]->derivative(fr.src.data(), fr.stride, fr.width, fr.height, dests[a].dx.data(),
                                dests[a].dy.data());
            counts[a] = arms[a]->countWhole(fr.src.data(), fr.stride, fr.width, fr.height);
            covs[a] = sweepCovariance(*arms[a], dests[a], fr, windows);
        }
        for (int a = 1; a < 3; ++a) {
            if (dests[a].dx != dests[0].dx || dests[a].dy != dests[0].dy) {
                std::printf("  DEFECT: arm '%s' computes a different derivative from '%s'"
                            " on input %d -- nothing timed.\n",
                            arms[a]->name, arms[0]->name, f);
                return false;
            }
            if (counts[a] != counts[0]) {
                std::printf("  DEFECT: arm '%s' counts %zu where '%s' counts %zu"
                            " on input %d -- nothing timed.\n",
                            arms[a]->name, counts[a], arms[0]->name, counts[0], f);
                return false;
            }
            if (covs[a] != covs[0]) {
                std::printf("  DEFECT: arm '%s' disagrees on the covariance sweep with '%s'"
                            " on input %d -- nothing timed.\n",
                            arms[a]->name, arms[0]->name, f);
                return false;
            }
        }
    }
    // The two decomposition points are held to the same standard: a diagnostic
    // that computes something else localizes nothing.
    for (int f = 0; f < kInputs; ++f) {
        const Frame& fr = frames[static_cast<size_t>(f)];
        std::memset(dests[0].dx.data(), 0, dests[0].dx.size() * sizeof(Word));
        std::memset(dests[0].dy.data(), 0, dests[0].dy.size() * sizeof(Word));
        std::memset(dests[1].dx.data(), 0, dests[1].dx.size() * sizeof(Word));
        std::memset(dests[1].dy.data(), 0, dests[1].dy.size() * sizeof(Word));
        arms[0]->derivative(fr.src.data(), fr.stride, fr.width, fr.height, dests[0].dx.data(),
                            dests[0].dy.data());
        t39::derivativeViewsOnly(fr.src.data(), fr.stride, fr.width, fr.height,
                                 dests[1].dx.data(), dests[1].dy.data());
        if (dests[1].dx != dests[0].dx || dests[1].dy != dests[0].dy) {
            std::printf("  DEFECT: the 'views only' decomposition point computes a different"
                        " derivative on input %d -- nothing timed.\n",
                        f);
            return false;
        }
        Cov viewCov;
        for (const auto& w : windows) {
            const Cov c = t39::covarianceWindowViewsOnly(dests[1].dx.data(), dests[1].dy.data(),
                                                         fr.stride, fr.width, fr.height, w.first,
                                                         w.second, kWindow);
            viewCov.xx += c.xx;
            viewCov.yy += c.yy;
            viewCov.whenClear += c.whenClear;
            viewCov.whenSet += c.whenSet;
        }
        if (viewCov != sweepCovariance(*arms[0], dests[0], fr, windows)) {
            std::printf("  DEFECT: the 'views only' decomposition point disagrees on the"
                        " covariance sweep on input %d -- nothing timed.\n",
                        f);
            return false;
        }

        std::memset(dests[3].dx.data(), 0, dests[3].dx.size() * sizeof(Word));
        std::memset(dests[3].dy.data(), 0, dests[3].dy.size() * sizeof(Word));
        t39::derivativeScalarized(fr.src.data(), fr.stride, fr.width, fr.height,
                                  dests[3].dx.data(), dests[3].dy.data());
        if (dests[3].dx != dests[0].dx || dests[3].dy != dests[0].dy) {
            std::printf("  DEFECT: the 'scalarized' decomposition point computes a different"
                        " derivative on input %d -- nothing timed.\n",
                        f);
            return false;
        }

        // The accumulator twins must agree with the arm they are copies of, and
        // with each other: summing per row cannot change a total, and if it did
        // the pair would be measuring two operations rather than one variable.
        const size_t handCount =
            arms[0]->countWhole(fr.src.data(), fr.stride, fr.width, fr.height);
        if (t39::countWholeOneChain(fr.src.data(), fr.stride, fr.width, fr.height) != handCount ||
            t39::countWholePerRow(fr.src.data(), fr.stride, fr.width, fr.height) != handCount) {
            std::printf("  DEFECT: an accumulator decomposition point disagrees on the count"
                        " on input %d -- nothing timed.\n",
                        f);
            return false;
        }
        Cov chainCov;
        Cov perRowCov;
        for (const auto& w : windows) {
            const Cov c1 = t39::covarianceWindowOneChain(dests[0].dx.data(), dests[0].dy.data(),
                                                         fr.stride, fr.width, fr.height, w.first,
                                                         w.second, kWindow);
            const Cov c2 = t39::covarianceWindowPerRow(dests[0].dx.data(), dests[0].dy.data(),
                                                       fr.stride, fr.width, fr.height, w.first,
                                                       w.second, kWindow);
            chainCov.xx += c1.xx;
            chainCov.yy += c1.yy;
            chainCov.whenClear += c1.whenClear;
            chainCov.whenSet += c1.whenSet;
            perRowCov.xx += c2.xx;
            perRowCov.yy += c2.yy;
            perRowCov.whenClear += c2.whenClear;
            perRowCov.whenSet += c2.whenSet;
        }
        const Cov handCov = sweepCovariance(*arms[0], dests[0], fr, windows);
        if (chainCov != handCov || perRowCov != handCov) {
            std::printf("  DEFECT: an accumulator decomposition point disagrees on the"
                        " covariance sweep on input %d -- nothing timed.\n",
                        f);
            return false;
        }
    }

    std::printf("  agreement: all three arms and all five decomposition points produce"
                " identical dx, dy, counts and covariances on %d inputs\n\n",
                kInputs);
    std::printf("  %-22s %-13s %8s  %-18s %6s   %6s\n", "workload", "arm", "ns/px",
                "[min, max] ns/px", "spread", "vs hand");

    const double framePixels = static_cast<double>(width) * static_cast<double>(height);
    const double windowPixels =
        static_cast<double>(kWindows) * static_cast<double>(kWindow) * kWindow;

    // ---- derivative ---------------------------------------------------------
    {
        std::vector<measure::Bench> benches;
        for (int a = 0; a < 3; ++a) {
            const Arm* arm = arms[a];
            Dest* dst = &dests[a];
            benches.push_back({arm->name, [arm, dst, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   arm->derivative(fr.src.data(), fr.stride, fr.width, fr.height,
                                                   dst->dx.data(), dst->dy.data());
                                   measure::g_sink += dst->dx[0] + dst->dy[0];
                               }});
        }
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (int a = 0; a < 3; ++a) {
            printRow("derivative dx+dy", arms[a]->name, t[static_cast<size_t>(a)], framePixels,
                     t[0].medianNs);
        }
    }

    // ---- whole-frame count --------------------------------------------------
    {
        std::vector<measure::Bench> benches;
        for (int a = 0; a < 3; ++a) {
            const Arm* arm = arms[a];
            benches.push_back({arm->name, [arm, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   measure::g_sink += arm->countWhole(fr.src.data(), fr.stride,
                                                                      fr.width, fr.height);
                               }});
        }
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (int a = 0; a < 3; ++a) {
            printRow("count whole frame", arms[a]->name, t[static_cast<size_t>(a)], framePixels,
                     t[0].medianNs);
        }
        // Sanity against a physical bound: one bit per pixel read, so the count
        // moves width*height/8 bytes. A figure above DRAM bandwidth on a frame
        // that does not fit in cache would mean the loop was deleted, not that
        // the kernel is fast.
        const double bytes = framePixels / 8.0;
        const double gbPerSec = bytes / t[0].medianNs;  // B/ns == GB/s
        std::printf("  %-22s %-13s %8s   reads %.1f KiB at %.2f GB/s\n", "  (bound check)",
                    arms[0]->name, "", bytes / 1024.0, gbPerSec);
    }

    // ---- decomposition: where the derivative gap lives -----------------------
    // Not part of the rule comparison. `views only` is the SAME kernel the
    // specialized arm calls, reached through the public view entry points with
    // the container removed, so the two differences below add up to the gap:
    //   views only - hand-written  =  the kernel's generic SHAPE at N = 1
    //   specialized - views only   =  the container
    {
        std::vector<measure::Bench> benches;
        {
            const Arm* arm = arms[0];
            Dest* dst = &dests[0];
            benches.push_back({"hand-written", [arm, dst, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   arm->derivative(fr.src.data(), fr.stride, fr.width, fr.height,
                                                   dst->dx.data(), dst->dy.data());
                                   measure::g_sink += dst->dx[0] + dst->dy[0];
                               }});
        }
        {
            Dest* dst = &dests[3];
            benches.push_back({"scalarized", [dst, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   t39::derivativeScalarized(fr.src.data(), fr.stride, fr.width,
                                                             fr.height, dst->dx.data(),
                                                             dst->dy.data());
                                   measure::g_sink += dst->dx[0] + dst->dy[0];
                               }});
        }
        {
            Dest* dst = &dests[1];
            benches.push_back({"views only", [dst, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   t39::derivativeViewsOnly(fr.src.data(), fr.stride, fr.width,
                                                            fr.height, dst->dx.data(),
                                                            dst->dy.data());
                                   measure::g_sink += dst->dx[0] + dst->dy[0];
                               }});
        }
        {
            const Arm* arm = arms[1];
            Dest* dst = &dests[2];
            benches.push_back({"specialized", [arm, dst, &frames](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   arm->derivative(fr.src.data(), fr.stride, fr.width, fr.height,
                                                   dst->dx.data(), dst->dy.data());
                                   measure::g_sink += dst->dx[0] + dst->dy[0];
                               }});
        }
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (size_t a = 0; a < benches.size(); ++a) {
            printRow("  decomp: derivative", benches[a].name.c_str(), t[a], framePixels,
                     t[0].medianNs);
        }
    }

    // The derivative timings above left each buffer holding whichever input ran
    // last, which differs per arm. Popcount cost is content-dependent, so the
    // covariance rows below are given IDENTICAL content in all three buffers
    // first -- otherwise a timing difference could be a difference in the data.
    for (int a = 0; a < 4; ++a) {
        arms[0]->derivative(frames[0].src.data(), frames[0].stride, frames[0].width,
                            frames[0].height, dests[a].dx.data(), dests[a].dy.data());
    }

    // ---- covariance over 200 31x31 windows ----------------------------------
    {
        std::vector<measure::Bench> benches;
        for (int a = 0; a < 3; ++a) {
            const Arm* arm = arms[a];
            const Dest* dst = &dests[a];
            benches.push_back({arm->name, [arm, dst, &frames, &windows](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   const Cov c = sweepCovariance(*arm, *dst, fr, windows);
                                   measure::g_sink += c.xx + c.yy + c.whenClear + c.whenSet;
                               }});
        }
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (int a = 0; a < 3; ++a) {
            printRow("covariance 31x31 x200", arms[a]->name, t[static_cast<size_t>(a)],
                     windowPixels, t[0].medianNs);
        }
    }

    // ---- decomposition: where the covariance gap lives ----------------------
    // ops/reduce.hpp never took a container (D-5), so `views only` here differs
    // from the specialized arm ONLY in the TernaryMat construction and the
    // magnitude() / sign() calls that name its planes.
    {
        std::vector<measure::Bench> benches;
        {
            const Arm* arm = arms[0];
            const Dest* dst = &dests[0];
            benches.push_back({"hand-written", [arm, dst, &frames, &windows](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   const Cov c = sweepCovariance(*arm, *dst, fr, windows);
                                   measure::g_sink += c.xx + c.yy + c.whenClear + c.whenSet;
                               }});
        }
        {
            const Dest* dst = &dests[1];
            benches.push_back({"views only", [dst, &frames, &windows](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   Cov total;
                                   for (const auto& w : windows) {
                                       const Cov c = t39::covarianceWindowViewsOnly(
                                           dst->dx.data(), dst->dy.data(), fr.stride, fr.width,
                                           fr.height, w.first, w.second, kWindow);
                                       total.xx += c.xx;
                                       total.yy += c.yy;
                                       total.whenClear += c.whenClear;
                                       total.whenSet += c.whenSet;
                                   }
                                   measure::g_sink +=
                                       total.xx + total.yy + total.whenClear + total.whenSet;
                               }});
        }
        {
            const Arm* arm = arms[1];
            const Dest* dst = &dests[2];
            benches.push_back({"specialized", [arm, dst, &frames, &windows](int i) {
                                   const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                                   const Cov c = sweepCovariance(*arm, *dst, fr, windows);
                                   measure::g_sink += c.xx + c.yy + c.whenClear + c.whenSet;
                               }});
        }
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (size_t a = 0; a < 3; ++a) {
            printRow("  decomp: covariance", benches[a].name.c_str(), t[a], windowPixels,
                     t[0].medianNs);
        }
    }

    // ---- decomposition: the ACCUMULATOR, isolated ---------------------------
    // Added at triage. X-21 attributed the library's count WIN to
    // impl::visitRowWords' head/interior/tail skeleton, but the hand-written arm
    // already has that skeleton -- what it does not have is D-15's PER-ROW partial
    // sum. These two pairs of twins differ in that and nothing else, and both twins
    // of a pair live in one object so the A/B is not a code-layout artefact
    // (genericn_diag_accum.cpp's header). The `one chain` rows are exact copies of
    // the hand-written arm's bodies and double as the layout control: they should
    // reproduce that arm's row above.
    {
        std::vector<measure::Bench> benches;
        benches.push_back({"one chain", [&frames](int i) {
                               const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                               measure::g_sink += t39::countWholeOneChain(
                                   fr.src.data(), fr.stride, fr.width, fr.height);
                           }});
        benches.push_back({"per row (D-15)", [&frames](int i) {
                               const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                               measure::g_sink += t39::countWholePerRow(fr.src.data(), fr.stride,
                                                                        fr.width, fr.height);
                           }});
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (size_t a = 0; a < benches.size(); ++a) {
            printRow("  decomp: count accum", benches[a].name.c_str(), t[a], framePixels,
                     t[0].medianNs);
        }
    }
    {
        std::vector<measure::Bench> benches;
        const Dest* dst = &dests[0];
        benches.push_back({"one chain", [dst, &frames, &windows](int i) {
                               const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                               Cov total;
                               for (const auto& w : windows) {
                                   const Cov c = t39::covarianceWindowOneChain(
                                       dst->dx.data(), dst->dy.data(), fr.stride, fr.width,
                                       fr.height, w.first, w.second, kWindow);
                                   total.xx += c.xx;
                                   total.yy += c.yy;
                                   total.whenClear += c.whenClear;
                                   total.whenSet += c.whenSet;
                               }
                               measure::g_sink +=
                                   total.xx + total.yy + total.whenClear + total.whenSet;
                           }});
        benches.push_back({"per row (D-15)", [dst, &frames, &windows](int i) {
                               const Frame& fr = frames[static_cast<size_t>(i % kInputs)];
                               Cov total;
                               for (const auto& w : windows) {
                                   const Cov c = t39::covarianceWindowPerRow(
                                       dst->dx.data(), dst->dy.data(), fr.stride, fr.width,
                                       fr.height, w.first, w.second, kWindow);
                                   total.xx += c.xx;
                                   total.yy += c.yy;
                                   total.whenClear += c.whenClear;
                                   total.whenSet += c.whenSet;
                               }
                               measure::g_sink +=
                                   total.xx + total.yy + total.whenClear + total.whenSet;
                           }});
        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        for (size_t a = 0; a < benches.size(); ++a) {
            printRow("  decomp: cov accum", benches[a].name.c_str(), t[a], windowPixels,
                     t[0].medianNs);
        }
    }

    return true;
}

}  // namespace

int main() {
    std::printf("\n");
    std::printf("========================================================================\n");
    std::printf("  T3.9 / E-4 -- generic-N against the specialized paths (X-21)\n");
    std::printf("========================================================================\n");
    std::printf("  arms:    hand-written (no binCV at all) | specialized (ships) | generic-N\n");
    std::printf("  metric:  ns/pixel, median of %d batches; [min, max] and spread beside it\n",
                kRepeats);
    std::printf("  rule:    within 5%% of hand-written -> arbitrary N confirmed free;\n");
    std::printf("           a regression > 5%% -> report before acting (TASKS.md T3.9)\n");
    std::printf("  NOTE:    x86_64 is indicative only. The Pi 4 closes this.\n");

    bool ok = true;
    if (!runSize(640, 480)) ok = false;
    if (!runSize(94, 60)) ok = false;

    std::printf("\n  CODE SIZE IS THE OTHER HALF OF THE METRIC and is not printed here.\n");
    std::printf("  Run, on the same device and against the same build:\n");
    std::printf("    size    benchmark/CMakeFiles/genericn_arms.dir/genericn_arm_*.cpp.o\n");
    std::printf("    size -A benchmark/CMakeFiles/genericn_arms.dir/genericn_arm_*.cpp.o\n");
    std::printf("  The -A breakdown matters: Berkeley `text` folds .text, .rodata and the\n");
    std::printf("  exception tables together, and the container arms carry tables the\n");
    std::printf("  hand-written one has no throw sites to need.\n");
    std::printf("  D-2 and ARCHITECTURE 2 both name code size as often binding before RAM\n");
    std::printf("  on Tier 2, so a speed result alone cannot close E-4.\n");
    std::printf("\n  sink: %zu\n", static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
