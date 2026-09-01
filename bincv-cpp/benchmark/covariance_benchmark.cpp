// -- the LK gradient covariance, fused against composed, AT THIS LEVEL.
//
// WHY THIS FILE EXISTS WHEN window_benchmark.cpp ALREADY MEASURED "fused versus
// composed"
//
// the axis 2 measured that question one level down, on the REDUCTION entry points
// and with a precomputed `sign_x ^ sign_y` plane on both sides. ships neither
// of those things: it ships `gradientCovariance`, which calls the FOUR-ARGUMENT
// countCovariance -- the form that XORs the two sign planes inside the word loop
// and needs no plane at all (the axis 3, memory wins CLAUDE.md's tiebreak). The
// four-argument form loads a fourth stream per word, so it is a different mix of
// loads to popcounts than the plane form a measurement timed, and the redundancy a
// composition pays is a different fraction of a bigger number.
//
// So that work’s Done-when asks for the ratio to be CONFIRMED at this level rather than
// inherited from a measurement of something adjacent. That is the whole content of
// this file.
//
// THE RULE, WRITTEN BEFORE MEASURING (CLAUDE.md: "write the decision rule before
// measuring"):
//
// * Fused beats composed at W=31 -> the axis 2 holds at the level;
// ops/covariance.hpp's "reach for the fused entry point" note is confirmed and
// nothing moves.
// * Fused within noise of composed, or SLOWER -> that CONTRADICTS a documented
// claim (the axis 2, the design notes, ops/reduce.hpp). CLAUDE.md's rule for
// that case is explicit: report it, do not adjust the code to fit the doc.
// that work’s implementation would then be resting on a ratio that does not exist
// at its own level, and the spec's "built on the fused entry point" would need
// re-deciding rather than re-measuring.
//
// No threshold is attached to that rule, deliberately. The 15% line in earlier work
// selected an interface that did not exist yet; this file is checking that an
// interface already selected behaves as recorded where it is actually called, so
// the question is direction and magnitude against the measured spread, not a gate.
//
// WHAT IS MEASURED
//
// FUSED gradientCovariance(dx, dy, window) -- WHAT SHIPS. One
// traversal, four popcounts per word, three loads plus the selector
// XOR of two more, no scratch.
// COMPOSED countNonZero(magX, w) + countNonZero(magY, w) +
// countAndSplit(magX, magY, signX, signY, w). The same four numbers,
// the same popcounts, THREE traversals, and 6 word loads per word
// index against the fused pass's 4. Also no scratch -- so this
// comparison is speed against speed with memory held equal, which is
// what makes it a clean confirmation of axis 2 rather than a mixture
// of axes 2 and 3.
// FUSED+PLANE / COMPOSED+PLANE
// The same two, with a caller-held `sign_x ^ sign_y` plane. They are
// here because CLAUDE.md requires memory and speed to be reported
// TOGETHER: the plane forms are faster and cost a frame-sized plane
// per pyramid level, and a reader weighing that work’s choice needs both
// numbers on one page. The plane's formation cost is reported
// separately and is NOT charged to the timed loop, which flatters the
// plane forms on purpose -- the conclusion survives being generous to
// the alternative.
//
// THE WORKLOAD IS THE LK ONE: 200 keypoints (the reference pipeline's
// gftt_max_corners), one window each, at 640x480, scattered so that windows near
// the border clip. Windows are NOT swept in a column here -- a caller that sweeps a
// column should be calling SlidingWindowCount for sumXX and sumYY instead (
// axis 1: 5.96x-15.9x, which are single-plane countNonZero sweeps; the cross term
// has no incremental form and is recomputed per position), and
// ops/covariance.hpp says so in its docstring.
//
// Validity: measure_util.hpp's protocol -- volatile sink, four rotating inputs,
// calibrated batches, interleaved variants, spread reported next to the median. And
// every variant's ANSWER is checked equal before any of them is timed; a faster
// wrong answer is not a result.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

// ---------------------------------------------------------------------------
// THE ALLOCATION COUNTER -- so the memory column is MEASURED on the same binary
// that produced the speed column.
//
// The "0 B" beside `fused` is the whole reason ships the slower of the two
// selector forms (the axis 3: the plane is 11-14% faster and costs a fifth
// frame-sized plane per pyramid level; CLAUDE.md's tiebreak takes the memory).
// Printed as a literal it was an assertion about the code rather than an
// observation of it: this table would have read "fused 0 B" unchanged if
// gradientCovariance allocated on every call. The counter is the idiom from
// tests/test_covariance.cpp, including the C++17 OVER-ALIGNED forms -- scratch for
// a vectorized kernel is exactly the kind that takes that path, and a counter
// replacing only the plain forms cannot see it.
// ---------------------------------------------------------------------------
#include <cstdlib>
#include <new>

namespace {
std::size_t g_newCount = 0;

void* benchAllocate(std::size_t bytes) {
    ++g_newCount;
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}

void* benchAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, rounded);
    if (p == nullptr) std::abort();
    return p;
}

void benchFree(void* p) noexcept { std::free(p); }
} // namespace

void* operator new(std::size_t bytes)   { return benchAllocate(bytes); }
void* operator new[](std::size_t bytes) { return benchAllocate(bytes); }
void operator delete(void* p) noexcept                 { benchFree(p); }
void operator delete[](void* p) noexcept               { benchFree(p); }
void operator delete(void* p, std::size_t) noexcept    { benchFree(p); }
void operator delete[](void* p, std::size_t) noexcept  { benchFree(p); }

void* operator new(std::size_t bytes, std::align_val_t a) {
    return benchAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return benchAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept                { benchFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { benchFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { benchFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { benchFree(p); }

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::GradientCovariance;
using bincv::Rect;
using bincv::TernaryMat;

// 640x480 and 200 keypoints, the same frame and keypoint count a measurement used, so the
// two measurements are directly comparable.
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kKeypoints = 200;
constexpr int kInputs = 4;
constexpr int kRepeats = 11;
constexpr double kTargetMs = 50.0;
const int kWindows[] = {7, 15, 31};

// ---------------------------------------------------------------------------
// The compositions being compared. Both produce GradientCovariance, so the
// agreement check is on the shipped type and not on an intermediate.
// ---------------------------------------------------------------------------

/// @brief the design notes through the primitives, with no plane: three
/// calls, therefore three traversals of one window. This is what a caller
/// who has not read writes, and it is the denominator.
template <typename Word>
GradientCovariance covarianceComposed(const BinMatConstView<Word>& magX,
                                      const BinMatConstView<Word>& magY,
                                      const BinMatConstView<Word>& signX,
                                      const BinMatConstView<Word>& signY, const Rect& window) {
    GradientCovariance out;
    out.sumXX = static_cast<int64_t>(bincv::countNonZero(magX, window));
    out.sumYY = static_cast<int64_t>(bincv::countNonZero(magY, window));
    out.sumXY = bincv::countAndSplit(magX, magY, signX, signY, window).crossTerm();
    return out;
}

/// @brief The same three calls with a caller-held selector plane.
template <typename Word>
GradientCovariance covarianceComposedPlane(const BinMatConstView<Word>& magX,
                                           const BinMatConstView<Word>& magY,
                                           const BinMatConstView<Word>& signXor,
                                           const Rect& window) {
    GradientCovariance out;
    out.sumXX = static_cast<int64_t>(bincv::countNonZero(magX, window));
    out.sumYY = static_cast<int64_t>(bincv::countNonZero(magY, window));
    out.sumXY = bincv::countAndSplit(magX, magY, signXor, window).crossTerm();
    return out;
}

/// @brief The fused entry point with a caller-held plane -- one traversal, one
/// fewer stream, and a frame-sized plane per pyramid level.
template <typename Word>
GradientCovariance covarianceFusedPlane(const BinMatConstView<Word>& magX,
                                        const BinMatConstView<Word>& magY,
                                        const BinMatConstView<Word>& signXor, const Rect& window) {
    const bincv::CovarianceCount c = bincv::countCovariance<Word>(magX, magY, signXor, window);
    GradientCovariance out;
    out.sumXX = static_cast<int64_t>(c.xx);
    out.sumYY = static_cast<int64_t>(c.yy);
    out.sumXY = c.crossTerm();
    return out;
}

bool same(const GradientCovariance& a, const GradientCovariance& b) {
    return a.sumXX == b.sumXX && a.sumYY == b.sumYY && a.sumXY == b.sumXY;
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// @brief Ternary derivative pairs plus the precomputed sign_x ^ sign_y plane.
/// @note Sparse on purpose -- a binarized derivative is mostly zero -- but the
/// kernels are content-independent (every word is loaded and counted
/// whatever it holds), so the fill ratio moves no ratio here.
template <typename Word>
struct DerivativeSet {
    std::vector<TernaryMat<Word>> dx;
    std::vector<TernaryMat<Word>> dy;
    std::vector<BinMat<Word>> sel;

    DerivativeSet() {
        dx.reserve(static_cast<size_t>(kInputs));
        dy.reserve(static_cast<size_t>(kInputs));
        sel.reserve(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            TernaryMat<Word> a(kWidth, kHeight);
            TernaryMat<Word> b(kWidth, kHeight);
            uint64_t state = UINT64_C(0x7F5C0) + static_cast<uint64_t>(i) * UINT64_C(104729);
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    a.set(y, x, static_cast<int>(measure::nextRandom(state) % 3) - 1);
                    b.set(y, x, static_cast<int>(measure::nextRandom(state) % 3) - 1);
                }
            }
            BinMat<Word> s(kWidth, kHeight);
            bincv::bitwiseXor(a.constSign(), b.constSign(), s.view());
            dx.push_back(std::move(a));
            dy.push_back(std::move(b));
            sel.push_back(std::move(s));
        }
    }
};

/// @brief 200 scattered keypoint windows, centerd, so edge windows clip.
std::vector<Rect> keypointWindows(int W) {
    std::vector<Rect> out;
    out.reserve(static_cast<size_t>(kKeypoints));
    uint64_t state = UINT64_C(0xC0FFEE);
    for (int k = 0; k < kKeypoints; ++k) {
        const int cx = static_cast<int>(measure::nextRandom(state) % static_cast<uint64_t>(kWidth));
        const int cy = static_cast<int>(measure::nextRandom(state) % static_cast<uint64_t>(kHeight));
        out.push_back(Rect(cx - W / 2, cy - W / 2, W, W));
    }
    return out;
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

template <typename Word>
bool runWordType(const char* wordName, const DerivativeSet<Word>& d) {
    std::printf("\n %-9s %-4s %12s %12s %11s %12s %12s %11s %18s\n", "word", "W", "fused",
                "composed", "composed/", "fused+plane", "comp+plane", "plane/", "spread f / c");
    std::printf(" %-9s %-4s %12s %12s %11s %12s %12s %11s\n", "", "", "ns/window", "ns/window",
                "fused", "ns/window", "ns/window", "4arg");

    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        const std::vector<Rect> windows = keypointWindows(W);

        // Validity hazard 4: everything agrees before anything is timed.
        for (int i = 0; i < kInputs; ++i) {
            const size_t k = static_cast<size_t>(i);
            for (const Rect& w : windows) {
                const GradientCovariance fused = bincv::gradientCovariance(d.dx[k], d.dy[k], w);
                const GradientCovariance composed = covarianceComposed<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.dx[k].constSign(),
                    d.dy[k].constSign(), w);
                const GradientCovariance fusedPlane = covarianceFusedPlane<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.sel[k].constView(), w);
                const GradientCovariance composedPlane = covarianceComposedPlane<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.sel[k].constView(), w);
                if (!same(fused, composed) || !same(fused, fusedPlane) ||
                    !same(fused, composedPlane)) {
                    std::printf(" DISAGREEMENT at W=%d: fused {%lld %lld %lld} composed "
                                "{%lld %lld %lld}\n",
                                W, static_cast<long long>(fused.sumXX),
                                static_cast<long long>(fused.sumYY),
                                static_cast<long long>(fused.sumXY),
                                static_cast<long long>(composed.sumXX),
                                static_cast<long long>(composed.sumYY),
                                static_cast<long long>(composed.sumXY));
                    return false;
                }
            }
        }

        const DerivativeSet<Word>* dp = &d;
        const std::vector<Rect>* wp = &windows;
        std::vector<measure::Bench> benches;

        benches.push_back({"fused", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               int64_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const GradientCovariance c =
                                       bincv::gradientCovariance(dp->dx[k], dp->dy[k], w);
                                   acc += c.sumXX + c.sumYY + c.sumXY;
                               }
                               measure::g_sink += static_cast<size_t>(acc);
                           }});
        benches.push_back({"composed", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               int64_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const GradientCovariance c = covarianceComposed<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->dx[k].constSign(), dp->dy[k].constSign(), w);
                                   acc += c.sumXX + c.sumYY + c.sumXY;
                               }
                               measure::g_sink += static_cast<size_t>(acc);
                           }});
        benches.push_back({"fused+plane", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               int64_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const GradientCovariance c = covarianceFusedPlane<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->sel[k].constView(), w);
                                   acc += c.sumXX + c.sumYY + c.sumXY;
                               }
                               measure::g_sink += static_cast<size_t>(acc);
                           }});
        benches.push_back({"composed+plane", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               int64_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const GradientCovariance c = covarianceComposedPlane<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->sel[k].constView(), w);
                                   acc += c.sumXX + c.sumYY + c.sumXY;
                               }
                               measure::g_sink += static_cast<size_t>(acc);
                           }});

        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        const double n = static_cast<double>(kKeypoints);
        std::printf(" %-9s %-4d %12.1f %12.1f %10.2fx %12.1f %12.1f %10.2fx %8.1f%% / %.1f%%\n",
                    wordName, W, t[0].medianNs / n, t[1].medianNs / n,
                    t[1].medianNs / t[0].medianNs, t[2].medianNs / n, t[3].medianNs / n,
                    t[0].medianNs / t[2].medianNs, t[0].spreadPct(), t[1].spreadPct());
    }
    return true;
}

/// @brief One pass of a variant over every keypoint window, with the allocation
/// counter armed. Returns allocations OBSERVED, not allocations expected.
/// @note The counter is armed after the windows exist -- building the window list
/// is the harness's allocation, not the kernel's -- and the result is
/// accumulated into the volatile sink so the pass cannot be elided.
template <typename Word, typename Fn>
size_t observedAllocations(const std::vector<Rect>& windows, Fn&& fn) {
    const size_t before = g_newCount;
    int64_t acc = 0;
    for (const Rect& w : windows) {
        const GradientCovariance c = fn(w);
        acc += c.sumXX + c.sumYY + c.sumXY;
    }
    measure::g_sink += static_cast<size_t>(acc);
    return g_newCount - before;
}

/// @brief What each form needs beyond the four derivative planes it must read.
/// @note CLAUDE.md: report memory and speed together. Two of the four forms need a
/// frame-sized plane and two need nothing, and that is the entire reason the
/// slower pair is what ships.
/// @note **The "allocs/pass" column is measured on this binary, not asserted.**
/// The plane bytes are arithmetic -- a plane's size is not in doubt -- but
/// the 0 B is a claim about the KERNEL, and printed as a literal it would
/// read 0 B for a gradientCovariance that allocated scratch on every call.
/// That is the one number the axis 3 traded 11-14% of speed for, so it is
/// counted here rather than stated. The counter covers the over-aligned
/// path too; see the note beside operator new at the top of this file.
template <typename Word>
void reportMemory(const char* wordName, const DerivativeSet<Word>& d) {
    const size_t planeWords =
        (static_cast<size_t>(kWidth) + BinMat<Word>::WordBits - 1) / BinMat<Word>::WordBits;
    const size_t planeBytes = planeWords * sizeof(Word) * static_cast<size_t>(kHeight);

    const std::vector<Rect> windows = keypointWindows(31);
    const size_t fusedAllocs = observedAllocations<Word>(windows, [&d](const Rect& w) {
        return bincv::gradientCovariance(d.dx[0], d.dy[0], w);
    });
    const size_t composedAllocs = observedAllocations<Word>(windows, [&d](const Rect& w) {
        return covarianceComposed<Word>(d.dx[0].constMagnitude(0), d.dy[0].constMagnitude(0),
                                        d.dx[0].constSign(), d.dy[0].constSign(), w);
    });
    const size_t fusedPlaneAllocs = observedAllocations<Word>(windows, [&d](const Rect& w) {
        return covarianceFusedPlane<Word>(d.dx[0].constMagnitude(0), d.dy[0].constMagnitude(0),
                                          d.sel[0].constView(), w);
    });
    const size_t composedPlaneAllocs = observedAllocations<Word>(windows, [&d](const Rect& w) {
        return covarianceComposedPlane<Word>(d.dx[0].constMagnitude(0), d.dy[0].constMagnitude(0),
                                             d.sel[0].constView(), w);
    });

    // The counter's own teeth: an instrument that cannot register a reading makes
    // a zero meaningless, so one allocation of each kind is put through it here.
    const size_t plainBefore = g_newCount;
    char* plain = new char[16];
    const size_t plainSeen = g_newCount - plainBefore;
    delete[] plain;
    struct alignas(64) OverAligned {
        char bytes[64];
    };
    const size_t alignedBefore = g_newCount;
    OverAligned* over = new OverAligned;
    const size_t alignedSeen = g_newCount - alignedBefore;
    delete over;

    std::printf("\n MEMORY (%s), beyond the four derivative planes every form reads.\n",
                wordName);
    std::printf(" \"allocs/pass\" is operator new calls COUNTED over one pass of %zu windows\n",
                windows.size());
    std::printf(" at W=31 -- the 0 B is a measurement of this binary, not a printed claim.\n");
    std::printf(" (counter self-check: plain new counted %zu, over-aligned new counted %zu; "
                "both must be 1)\n",
                plainSeen, alignedSeen);
    std::printf(" %-16s %9s %11s\n", "", "scratch", "allocs/pass");
    std::printf(" %-16s %7zu B %11zu <- this SHIPS THIS\n", "fused", size_t(0),
                fusedAllocs);
    std::printf(" %-16s %7zu B %11zu\n", "composed", size_t(0), composedAllocs);
    std::printf(" %-16s %7zu B %11zu one sign_x^sign_y plane at %dx%d\n", "fused+plane",
                planeBytes, fusedPlaneAllocs, kWidth, kHeight);
    std::printf(" %-16s %7zu B %11zu\n", "composed+plane", planeBytes, composedPlaneAllocs);
    std::printf(" A plane is needed at EVERY pyramid level, not once: ~%zu B over a\n",
                planeBytes + planeBytes / 4 + planeBytes / 16 + planeBytes / 64);
    std::printf(" four-level pyramid, a FIFTH plane on top of the four the covariance\n");
    std::printf(" already reads -- +25%% of the derivative working set, held for the\n");
    std::printf(" frame's lifetime.\n");
}

} // namespace

int main() {
    std::printf("=== the LK gradient covariance -- fused against composed ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE (the reference device is where this closes)\n");
#else
    std::printf("target: not aarch64 -- INDICATIVE ONLY. Every variant here is popcount-bound,\n"
                " and the x86 popcount lowering ( a libgcc CALL per word on the\n"
                " shipped baseline) can change the ranking outright.\n");
#endif
    std::printf("%dx%d, %d keypoints, one window each -- the LK access pattern of\n", kWidth,
                kHeight, kKeypoints);
    std::printf("ARCHITECTURE 7.5. The rule this is measured against is in this file's\n");
    std::printf("header, written before measuring.\n");
    std::printf("\n\"composed/fused\" > 1.00x means the FUSED entry point this ships is faster,\n");
    std::printf("confirming axis 2 at this level. \"plane/4arg\" > 1.00x means the\n");
    std::printf("precomputed selector plane is faster than the four-argument form -- which it\n");
    std::printf("is expected to be, and is the speed spends to save the plane.\n");

    DerivativeSet<uint32_t> d32;
    DerivativeSet<uint64_t> d64;

    bool ok = runWordType<uint32_t>("uint32_t", d32);
    ok = runWordType<uint64_t>("uint64_t", d64) && ok;

    reportMemory<uint32_t>("uint32_t", d32);

    std::printf("\nsink: %zu (printed so nothing above can be optimized away)\n",
                static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
