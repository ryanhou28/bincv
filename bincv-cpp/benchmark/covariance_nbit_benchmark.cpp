// -- what an N-BIT pyramid level costs per LK window.
//
// WHY THIS MEASUREMENT IS A DELIVERABLE AND NOT AN AFTERTHOUGHT
//
// a measurement measured the hybrid LK tracker missing its accuracy tolerance on the
// reference pipeline's own edge-map content and separated the causes: on windows
// that never clip, four 1-BIT pyramid levels are still ~600x worse than one,
// because a level whose pixels are bits cannot localise sub-pixel motion better
// than its own quantization. a measurement measured the levels needing 1/3/4/5 bits. So the
// fix is N-bit levels, is the task that has to CHOOSE a bit depth per
// level -- and a choice needs a price. This file is the price.
//
// THE COST MODEL, WRITTEN OUT BEFORE MEASURING
//
// The covariance of two N-bit values is a sum over plane PAIRS, so it is quadratic
// in N where that work’s derivative is linear (the design rule’s scope limit says exactly this, and
// flagged it). Counting the popcounts ops/covariance.hpp issues per word:
//
// N(N+1)/2 for sumXX (the diagonal is symmetric: upper triangle, doubled)
// N(N+1)/2 for sumYY
// 2*N^2 for sumXY (a total and a selected count per ORDERED pair)
// ----------------------------------------------------------------
// 3N^2 + N popcounts per word, with 2N+2 word loads and one selector XOR
//
// N = 1: 4 N = 2: 14 N = 3: 30 N = 4: 52
// ratio: 1.00 3.50 7.50 13.00
//
// At N = 1 that is exactly countCovariance's four popcounts, which is the
// arithmetic statement of "ternary is the N = 1 instance".
//
// THE RULE, WRITTEN BEFORE MEASURING (CLAUDE.md: "write the decision rule before
// measuring"). **Nothing here chooses between two implementations** -- a product of
// two N-bit values IS a sum over N^2 plane pairs, and anything linear in N computes
// a different quantity. So the rule is a falsifiable prediction about the cost
// curve rather than a selection between arms:
//
// BAND A -- measured ns/window ratios within +/-25% of 1.00 / 3.50 / 7.50 / 13.00:
// the popcount count IS the cost model. this may price a per-level bit depth
// with 3N^2 + N and this file's table, and no code moves.
// BAND B -- ratios systematically BELOW the prediction: the kernel is not purely
// popcount-bound at these N (the N^2 pairs come off 2N+2 loads, so there is
// instruction-level parallelism the count does not model). Report the MEASURED
// curve as the price and mark 3N^2 + N an upper bound. Still no code moves.
// BAND C -- ratios ABOVE the prediction: something is quadratic that should not
// be -- register spills out of the N^2 counters, or the per-row combine growing
// with N. That CONTRADICTS the documented cost of the shipped kernel and
// CLAUDE.md's rule applies: report it, do not adjust the doc to fit.
//
// WHAT IS MEASURED
//
// ternary N=1 the five-view entry point -- the shipped level-0 path, and
// the denominator a bit-depth decision is taken against.
// bit-sliced the plane-array entry point at N = 1, 2, 3, 4. The N = 1
// arm is the SAME KERNEL as the N = 2..4 arms with N = 1, not the
// ternary one, so the ratios in the table are one kernel's curve
// and not a change of kernel at the first column.
// window-acc THE SHIPPED KERNEL WITH EXACTLY ONE THING CHANGED: the per-row
// partial accumulator replaced by a single window-wide one. Same
// popcounts, same words, same masks, same combine -- so any
// difference is the accumulator and nothing else.
//
// It is here because the per-row split is a MEASURED decision taken
// at N = 1 ( item 4; a measurement measured 1.08x at W=31 and a 5-6%
// loss at W=7) and its cost is O(N^2) PER ROW -- 4N^2 counters
// zeroed and 4N^2 added, 128 operations per row at N = 4 -- while
// the work it is amortized over is O(N^2) per WORD. A 31-pixel
// window is 1-2 `uint64_t` words per row, so at large N and wide
// words the fixed per-row cost stops being small. Whether it
// actually does is a question this file can answer with one arm
// rather than a paragraph, and it is exactly the corner this will
// be choosing in. **This arm decides nothing** -- the shipped
// kernel is not changed here -- it exists so the report says
// "measured" where it would otherwise say "presumably".
//
// Both spellings are VIEW spellings, so no arm pays container plumbing the others
// do not. Memory is reported beside the time, as CLAUDE.md requires: an N-bit level
// costs (N+1) bits per pixel per derivative against ternary's 2, and that is the
// other half of the trade this is taking.
//
// THE WORKLOAD IS THE LK ONE: 200 keypoints (the reference pipeline's
// gftt_max_corners), one window each, at 640x480, scattered so border windows clip
// -- the same shape covariance_benchmark.cpp and use, so the numbers are
// comparable across the three.
//
// Validity: measure_util.hpp's protocol -- volatile sink, four rotating inputs,
// calibrated batches, interleaved variants, spread reported next to the median. And
// every arm's ANSWER is checked against an independent per-pixel reference before
// any of them is timed; a faster wrong answer is not a result.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

// ---------------------------------------------------------------------------
// THE ALLOCATION COUNTER -- so the "no scratch at N bits" column is MEASURED on
// the same binary that produced the speed column. The N-bit kernel holds 4*N^2
// per-pair counters; the claim is that they are automatic storage and that nothing
// reaches the heap. Printed as a literal that would be an assertion about the code
// rather than an observation of it. Idiom and rationale: covariance_benchmark.cpp,
// including the C++17 OVER-ALIGNED forms.
// ---------------------------------------------------------------------------
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

using bincv::BinMatConstView;
using bincv::GradientCovariance;
using bincv::Rect;

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kKeypoints = 200;
constexpr int kInputs = 4;
constexpr int kRepeats = 11;
constexpr double kTargetMs = 50.0;
const int kWindows[] = {7, 15, 31};

/// @brief Popcounts per word at bit depth N -- the model the rule is written
/// against. Derived in this file's header; NOT read out of the kernel.
constexpr double popcountsPerWord(size_t n) {
    return 3.0 * static_cast<double>(n) * static_cast<double>(n) + static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Inputs: N-bit levels through the REAL derivative, not a generator
// ---------------------------------------------------------------------------

/// @brief `kInputs` N-bit derivative pairs, and the plane-array views the
/// entry point takes.
/// @note The content comes from that work’s `derivativeX` / `derivativeY` over a random
/// `QuantMat<N>` level, which is what an N-bit pyramid level actually feeds
/// the covariance. The kernels are content-independent -- every word in the
/// window is loaded and counted whatever it holds -- so the fill ratio moves
/// no ratio here, but a generator that produced impossible sign/magnitude
/// combinations would make the ANSWER check below meaningless.
/// @note The views are built AFTER every container exists. Building them inside
/// the fill loop would leave them pointing into storage a `push_back`
/// reallocation had moved.
template <size_t N, typename Word>
struct LevelSet {
    /// The plane arrays one input's kernel call needs, in the exact array form
    /// `gradientCovariance<N, Word>` takes.
    struct Views {
        BinMatConstView<Word> magX[N];
        BinMatConstView<Word> magY[N];
        BinMatConstView<Word> signX;
        BinMatConstView<Word> signY;
    };

    std::vector<bincv::SignedQuantMat<N, Word>> dx;
    std::vector<bincv::SignedQuantMat<N, Word>> dy;
    std::vector<Views> views;

    LevelSet() {
        dx.reserve(static_cast<size_t>(kInputs));
        dy.reserve(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            bincv::QuantMat<N, Word> src(kWidth, kHeight);
            uint64_t state = UINT64_C(0x9E3B10) + static_cast<uint64_t>(i) * UINT64_C(104729) +
                             static_cast<uint64_t>(N) * UINT64_C(7919);
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    src.set(y, x, static_cast<unsigned>(measure::nextRandom(state) &
                                                        bincv::QuantMat<N, Word>::MaxValue));
                }
            }
            bincv::SignedQuantMat<N, Word> a(kWidth, kHeight);
            bincv::SignedQuantMat<N, Word> b(kWidth, kHeight);
            bincv::derivativeX(src, a);
            bincv::derivativeY(src, b);
            dx.push_back(std::move(a));
            dy.push_back(std::move(b));
        }
        views.resize(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            const size_t k = static_cast<size_t>(i);
            for (size_t p = 0; p < N; ++p) {
                views[k].magX[p] = dx[k].constMagnitude(p);
                views[k].magY[p] = dy[k].constMagnitude(p);
            }
            views[k].signX = dx[k].constSign();
            views[k].signY = dy[k].constSign();
        }
    }

    /// @brief Bytes of derivative plane the covariance must hold for ONE level.
    /// @note (N + 1) planes per derivative, two derivatives. This is the footprint
    /// half of the trade: ternary is 2 bits per pixel per derivative, N = 4
    /// is 5.
    size_t bytesPerLevel() const {
        return 2 * bincv::SignedQuantMat<N, Word>::Planes * dx[0].planeWords() * sizeof(Word);
    }
};

/// @brief 200 scattered keypoint windows, centerd, so edge windows clip.
/// @note The same generator covariance_benchmark.cpp uses, seeded identically, so
/// the two files time the same windows.
std::vector<Rect> keypointWindows(int W) {
    std::vector<Rect> out;
    out.reserve(static_cast<size_t>(kKeypoints));
    uint64_t state = UINT64_C(0xC0FFEE);
    for (int k = 0; k < kKeypoints; ++k) {
        const int cx = static_cast<int>(measure::nextRandom(state) % static_cast<uint64_t>(kWidth));
        const int cy = static_cast<int>(measure::nextRandom(state) %
                                        static_cast<uint64_t>(kHeight));
        out.push_back(Rect(cx - W / 2, cy - W / 2, W, W));
    }
    return out;
}

// ---------------------------------------------------------------------------
// THE ONE-VARIABLE ALTERNATIVE: a window-wide accumulator instead of per-row
// ---------------------------------------------------------------------------

/// @brief ops/covariance.hpp's N-bit kernel with the per-row partial accumulator
/// removed, and nothing else changed.
/// @note It calls the SAME `impl::bitSlicedPairRowRegion` and the SAME
/// `impl::combineBitSlicedPairs`, so the popcounts, the masks, the clip and
/// the weighting are literally the shipped code. The only difference is that
/// every row adds into one `BitSlicedPairCounts<N>` rather than into a fresh
/// one that is then folded in -- which removes 8N^2 operations per row and
/// lengthens the dependency chain through each counter.
/// @note Its ANSWER is identical by construction (integer addition is
/// associative), and the check below requires it rather than assuming it.
template <size_t N, typename Word>
GradientCovariance covarianceWindowAccumulator(const BinMatConstView<Word> (&magX)[N],
                                               const BinMatConstView<Word> (&magY)[N],
                                               const BinMatConstView<Word>& signX,
                                               const BinMatConstView<Word>& signY,
                                               const Rect& window) {
    const bincv::impl::RegionWords<Word> r =
        bincv::impl::clipRegion<Word>(magX[0].width, magX[0].height, window);
    if (r.isEmpty) return GradientCovariance();

    bincv::impl::BitSlicedPairCounts<N> total;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const Word* rowX[N];
        const Word* rowY[N];
        for (size_t p = 0; p < N; ++p) {
            rowX[p] = magX[p].row(y);
            rowY[p] = magY[p].row(y);
        }
        bincv::impl::bitSlicedPairRowRegion<N, Word>(rowX, rowY, signX.row(y), signY.row(y), r,
                                                     total);
    }
    return bincv::impl::combineBitSlicedPairs<N>(total);
}

// ---------------------------------------------------------------------------
// The independent answer check (validity hazard 4)
// ---------------------------------------------------------------------------

/// @brief The per-pixel covariance, with multiplies, through SignedQuantMat::at.
/// @note Not a second spelling of the kernel: it knows nothing about planes, pairs,
/// masks or popcounts. Slow and only run once per arm before timing starts.
template <size_t N, typename Word>
GradientCovariance referenceCovariance(const bincv::SignedQuantMat<N, Word>& dx,
                                       const bincv::SignedQuantMat<N, Word>& dy, const Rect& w) {
    GradientCovariance out;
    const int x0 = (w.x > 0) ? w.x : 0;
    const int y0 = (w.y > 0) ? w.y : 0;
    const int x1 = (w.x + w.width < dx.cols()) ? (w.x + w.width) : dx.cols();
    const int y1 = (w.y + w.height < dx.rows()) ? (w.y + w.height) : dx.rows();
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const int64_t a = dx.at(y, x);
            const int64_t b = dy.at(y, x);
            out.sumXX += a * a;
            out.sumYY += b * b;
            out.sumXY += a * b;
        }
    }
    return out;
}

bool same(const GradientCovariance& a, const GradientCovariance& b) {
    return a.sumXX == b.sumXX && a.sumYY == b.sumYY && a.sumXY == b.sumXY;
}

/// @brief Checks one bit depth's arm against the per-pixel reference at every
/// timed window, and returns false loudly if any disagrees.
template <size_t N, typename Word>
bool checkAnswers(const LevelSet<N, Word>& set, const std::vector<Rect>& windows,
                  const char* wordName, int W) {
    for (int i = 0; i < kInputs; ++i) {
        const size_t k = static_cast<size_t>(i);
        for (const Rect& w : windows) {
            const GradientCovariance got = bincv::gradientCovariance<N, Word>(
                set.views[k].magX, set.views[k].magY, set.views[k].signX, set.views[k].signY, w);
            const GradientCovariance want = referenceCovariance<N, Word>(set.dx[k], set.dy[k], w);
            const GradientCovariance alt = covarianceWindowAccumulator<N, Word>(
                set.views[k].magX, set.views[k].magY, set.views[k].signX, set.views[k].signY, w);
            if (!same(got, alt)) {
                std::printf(" DISAGREEMENT %s N=%zu W=%d at [%d %d]: the window-accumulator "
                            "arm is not the shipped kernel's answer\n",
                            wordName, N, W, w.x, w.y);
                return false;
            }
            if (!same(got, want)) {
                std::printf(" DISAGREEMENT %s N=%zu W=%d at [%d %d]: kernel "
                            "{%lld %lld %lld} reference {%lld %lld %lld}\n",
                            wordName, N, W, w.x, w.y, static_cast<long long>(got.sumXX),
                            static_cast<long long>(got.sumYY),
                            static_cast<long long>(got.sumXY),
                            static_cast<long long>(want.sumXX),
                            static_cast<long long>(want.sumYY),
                            static_cast<long long>(want.sumXY));
                return false;
            }
        }
    }
    return true;
}

/// @brief One timed arm: every keypoint window at one bit depth, folded into the
/// volatile sink so nothing can be deleted.
template <size_t N, typename Word>
measure::Bench bitSlicedArm(const LevelSet<N, Word>& set, const std::vector<Rect>& windows,
                            const std::string& name) {
    measure::Bench b;
    b.name = name;
    b.body = [&set, &windows](int i) {
        const size_t k = static_cast<size_t>(i % kInputs);
        int64_t acc = 0;
        for (const Rect& w : windows) {
            const GradientCovariance c = bincv::gradientCovariance<N, Word>(
                set.views[k].magX, set.views[k].magY, set.views[k].signX, set.views[k].signY, w);
            acc += c.sumXX + c.sumYY + c.sumXY;
        }
        measure::g_sink += static_cast<size_t>(acc);
    };
    return b;
}

/// @brief The window-accumulator arm at one bit depth.
template <size_t N, typename Word>
measure::Bench windowAccArm(const LevelSet<N, Word>& set, const std::vector<Rect>& windows,
                            const std::string& name) {
    measure::Bench b;
    b.name = name;
    b.body = [&set, &windows](int i) {
        const size_t k = static_cast<size_t>(i % kInputs);
        int64_t acc = 0;
        for (const Rect& w : windows) {
            const GradientCovariance c = covarianceWindowAccumulator<N, Word>(
                set.views[k].magX, set.views[k].magY, set.views[k].signX, set.views[k].signY, w);
            acc += c.sumXX + c.sumYY + c.sumXY;
        }
        measure::g_sink += static_cast<size_t>(acc);
    };
    return b;
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

template <typename Word>
bool runWordType(const char* wordName) {
    LevelSet<1, Word> l1;
    LevelSet<2, Word> l2;
    LevelSet<3, Word> l3;
    LevelSet<4, Word> l4;

    std::printf("\n === %s ===\n", wordName);
    std::printf(" %-4s %-14s %12s %10s %10s %9s %12s\n", "W", "arm", "ns/window", "vs N=1",
                "predicted", "spread", "bits/px/deriv");

    bool ok = true;
    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        const std::vector<Rect> windows = keypointWindows(W);

        ok = checkAnswers<1, Word>(l1, windows, wordName, W) && ok;
        ok = checkAnswers<2, Word>(l2, windows, wordName, W) && ok;
        ok = checkAnswers<3, Word>(l3, windows, wordName, W) && ok;
        ok = checkAnswers<4, Word>(l4, windows, wordName, W) && ok;

        // The the arm is the shipped level-0 path, spelled as five loose views so
        // that it pays exactly the call structure the bit-sliced arms do.
        measure::Bench ternary;
        ternary.name = "ternary";
        ternary.body = [&l1, &windows](int i) {
            const size_t k = static_cast<size_t>(i % kInputs);
            int64_t acc = 0;
            for (const Rect& w : windows) {
                const GradientCovariance c = bincv::gradientCovariance<Word>(
                    l1.views[k].magX[0], l1.views[k].magY[0], l1.views[k].signX,
                    l1.views[k].signY, w);
                acc += c.sumXX + c.sumYY + c.sumXY;
            }
            measure::g_sink += static_cast<size_t>(acc);
        };

        std::vector<measure::Bench> benches;
        benches.push_back(ternary);
        benches.push_back(bitSlicedArm<1, Word>(l1, windows, "bit-sliced N=1"));
        benches.push_back(bitSlicedArm<2, Word>(l2, windows, "bit-sliced N=2"));
        benches.push_back(bitSlicedArm<3, Word>(l3, windows, "bit-sliced N=3"));
        benches.push_back(bitSlicedArm<4, Word>(l4, windows, "bit-sliced N=4"));
        benches.push_back(windowAccArm<1, Word>(l1, windows, "window-acc N=1"));
        benches.push_back(windowAccArm<2, Word>(l2, windows, "window-acc N=2"));
        benches.push_back(windowAccArm<3, Word>(l3, windows, "window-acc N=3"));
        benches.push_back(windowAccArm<4, Word>(l4, windows, "window-acc N=4"));

        // THE NO-SCRATCH READING, taken around the KERNEL CALLS and not around
        // measureInterleaved. The harness itself allocates -- std::vector, the
        // std::function bodies -- so a counter wrapped around the timed region
        // reads its allocations, not the kernel's, and would report a false
        // positive on every run. One pass of every arm, outside the timing, is the
        // reading that is about the kernel: 200 windows per arm at four bit depths,
        // so a kernel that allocated once per frame rather than once per window
        // would still show up.
        const size_t newsBefore = g_newCount;
        for (size_t b = 0; b < benches.size(); ++b) benches[b].body(0);
        const size_t newsDuring = g_newCount - newsBefore;

        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);

        const double perWindow = static_cast<double>(kKeypoints);
        const double base = t[1].medianNs / perWindow;  // bit-sliced N = 1
        const size_t bitsPerPixel[5] = {0, 2, 3, 4, 5};
        // arm 0 is ternary (N = 1); arms 1..4 are bit-sliced N = 1..4; arms 5..8
        // are the window-accumulator variant at the same four depths.
        const size_t armDepth[9] = {1, 1, 2, 3, 4, 1, 2, 3, 4};
        for (size_t b = 0; b < benches.size(); ++b) {
            const double ns = t[b].medianNs / perWindow;
            const size_t n = armDepth[b];
            const double predicted =
                (b == 0) ? 1.0 : popcountsPerWord(n) / popcountsPerWord(1);
            std::printf(" %-4d %-14s %12.1f %10.2fx %9.2fx %8.1f%% %12zu\n", W,
                        benches[b].name.c_str(), ns, ns / base, predicted, t[b].spreadPct(),
                        bitsPerPixel[n]);
        }
        if (newsDuring != 0) {
            std::printf(" W=%d: operator new was called %zu time(s) across %d kernel calls "
                        "-- the no-scratch claim is FALSE here\n",
                        W, newsDuring, kKeypoints * static_cast<int>(benches.size()));
            ok = false;
        } else {
            std::printf(" W=%d: operator new across %d kernel calls: 0 (no scratch at any "
                        "N)\n", W, kKeypoints * static_cast<int>(benches.size()));
        }
    }

    std::printf("\n footprint at %dx%d, both derivatives, one level:\n", kWidth, kHeight);
    std::printf(" N=1 (ternary) %8zu B N=2 %8zu B N=3 %8zu B N=4 %8zu B\n",
                l1.bytesPerLevel(), l2.bytesPerLevel(), l3.bytesPerLevel(), l4.bytesPerLevel());
    return ok;
}

} // namespace

int main() {
    std::printf("=== what an N-bit level costs per LK window ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE (the reference device is where this closes)\n");
#else
    std::printf("target: not aarch64 -- INDICATIVE ONLY. Every arm here is popcount-bound and\n"
                " the x86 popcount lowering can change the shape of the curve.\n");
#endif
    std::printf("%dx%d, %d keypoints, one window each -- the LK access pattern of\n", kWidth,
                kHeight, kKeypoints);
    std::printf("ARCHITECTURE 7.5. The cost model and the rule are in this file's header,\n");
    std::printf("written before measuring. 'predicted' is (3N^2+N)/4, the popcount count\n");
    std::printf("per word; 'vs N=1' is measured against the BIT-SLICED N=1 arm, so the\n");
    std::printf("column is one kernel's curve in N and not a change of kernel.\n");

    bool ok = runWordType<uint32_t>("uint32_t");
    ok = runWordType<uint64_t>("uint64_t") && ok;

    std::printf("\nsink: %zu (printed so nothing above can be optimized away)\n",
                static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
