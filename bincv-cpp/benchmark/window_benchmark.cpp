// E-3 / T2.10 -- incremental versus recomputed window reductions, and the two
// other T2.6 interface questions that were registered alongside it.
//
// T3.6 cannot start until this closes, because all three axes decide the SHAPE of
// ops/reduce.hpp rather than its speed.
//
// ===========================================================================
// WHAT CHANGED WHEN T2.11 LANDED, AND WHY THIS FILE HAD TO BE RE-RUN
// ===========================================================================
//
// E-3 closed against measurement copies: the winning variants lived in this file,
// because writing them into ops/reduce.hpp in the same commit as the measurement
// that gated them is the inversion EXPERIMENTS.md exists to prevent. T2.11 then
// landed them for real, so this file now times the SHIPPED entry points --
// bincv::SlidingWindowCount, bincv::countCovariance and the four-argument
// bincv::countAndSplit -- and a copy survives here only where nothing shipped
// (INC-COL, which axis 1 explicitly declines to expose).
//
// T2.11 also landed a FOURTH change that no axis asked for: impl::countViewRegion
// carried one accumulator across a whole region, one dependency chain through the
// popcount latency, and its row bodies now each return their own partial sum. That
// landed FIRST of the four, so the recompute baseline every ratio below is divided
// by is up to 1.32x faster than the one X-11 measured. The axis-1 ratios here are
// therefore SMALLER than X-11's, by design and not by regression: X-11 predicted
// roughly 5.6x and 15x where it had measured 7.3x and 20x. EXPERIMENTS.md X-11
// records both sets side by side; neither replaces the other, because they answer
// "what did the accumulator buy" and "what does it buy in the shipped library".
//
// ===========================================================================
// AXIS 1 -- recompute per window versus a sliding accumulator
// ===========================================================================
//
// DECISION RULE, verbatim from TASKS.md T2.10 and recorded in EXPERIMENTS.md X-11
// before this file was written:
//
//   * Recompute within 15% of incremental at 31x31 -> keep the simpler recompute
//     API, close E-3, and record that incremental state was rejected on data
//   * Incremental wins by > 15% at 31x31 -> extend T2.6 with incremental state
//     BEFORE T3.6 is written against the simpler form
//
// The rule names a window size but not an access pattern, and the two things it
// asks for -- "~200 keypoints, per the reference gftt_max_corners" and "include
// the heavy-overlap case, since that is what favours incremental" -- are not the
// same workload. So all three patterns the MVP actually contains are measured and
// reported separately, and the rule is applied to each rather than to an average
// that would hide the disagreement:
//
//   SPARSE   200 isolated windows at scattered keypoints. ARCHITECTURE 7.5: the
//            LK covariance per tracked keypoint. Windows barely overlap.
//   SEARCH   200 keypoints x an 8x8 sweep of window positions = 12800 windows.
//            Heavy LOCAL overlap; a block-matching / search-region pattern.
//   DENSE    every window position in the frame. ARCHITECTURE 7.6's corner
//            response is computed from the same covariance machinery over the
//            whole image, so this is the frontend's real maximum-overlap case
//            rather than a synthetic upper bound.
//
// TWO incremental forms are measured, because "a sliding accumulator" over
// bit-packed rows is not one design, and they differ in exactly the resource
// T2.10 asks about (extra memory):
//
//   INC-COL  the classic separable box accumulator: per-column running sums over
//            the window's rows, slid one pixel at a time in x and in y. It issues
//            NO popcounts at all -- it reads individual bits -- which is the form
//            most likely to win where popcount is expensive (see X-7). Costs an
//            array of (sweepWidth + W - 1) uint32 counters, which is the extra
//            memory the interface would have to let a caller provide (no heap
//            allocation inside kernels, CLAUDE.md).
//   INC-ROW  slides vertically only, keeping ONE scalar accumulator: the window
//            sum gains the incoming row's windowed popcount and loses the
//            outgoing row's. Word-parallel like the shipped kernel, and needs
//            essentially no extra memory.
//
// INC-ROW won and is now bincv::SlidingWindowCount, so this file times the shipped
// class rather than a copy of it. INC-COL remains MEASUREMENT CODE -- axis 1
// declined to expose it, so there is nothing shipped to time, and it stays here
// because a rejected alternative with no number next to it is an assertion rather
// than a decision.
//
// A FOURTH variant, recompute-1acc, is the recompute path with T2.11 item 4
// UNDONE: one accumulator across the whole region. It is timed interleaved with
// the other three so that item 4's own effect is measured the same way everything
// else here is, rather than inferred by comparing absolute ns against a run from
// another session. It also keeps X-11's original denominator alive, so the
// pre-split ratios that entry quotes can be reproduced from this binary.
//
// Windows are placed fully inside the image on this axis. Edge clipping is real
// (ARCHITECTURE 7.5) and axes 2 and 3 below include it; the shipped
// SlidingWindowCount clips exactly (tests/test_reduce.cpp sweeps whole frames
// checking it position by position), but INC-COL here does not, and a comparison
// between two implementations of clipping is not the question E-3 asks.
//
// ===========================================================================
// AXIS 2 -- the 2x2 covariance composed out of T2.6 versus one fused pass
// ===========================================================================
//
// Registered as T2.10's second axis after X-8 measured 1.30x on the reference
// device. Rule, verbatim, same threshold: a covariance-shaped entry point
// (returning xx, yy, whenClear, whenSet from one visitRowWords pass) beats the
// composition by > 15% at 31x31 -> add it to T2.6 before T3.6 is written; within
// 15% -> keep the composition and record that the fused form was rejected on data.
//
// The fused side is now bincv::countCovariance. BOTH sides carry T2.11's per-row
// accumulator split, so this ratio is still redundant traversal and nothing else:
// splitting only the fused side would have made it a mixture of two effects.
//
// Re-measured here at three window sizes and at two word widths, in the same
// session as axis 1, rather than resting on X-8's single 31x31 uint64_t point.
//
// ===========================================================================
// AXIS 3 -- frame-sized selector plane versus a four-argument countAndSplit
// ===========================================================================
//
// countAndSplit's selector `c` is sign_x ^ sign_y, which the three-argument form
// takes as a whole extra plane, formed once per pyramid level. The four-argument
// form takes the two sign planes and XORs them inside the word loop, needing no
// plane at all. Both are now shipped overloads of bincv::countAndSplit, so this
// axis times the library rather than a copy.
//
// TASKS.md states NO numeric threshold for this axis -- it requires that both
// memory and speed be reported, "since this is precisely a case where the two
// goals may disagree". No threshold is invented here. Both numbers are printed,
// including the plane's formation cost amortized over the keypoints that use it,
// and the weighing is against CLAUDE.md's stated tiebreak: memory wins when the
// goals conflict and no explicit choice has been made.
//
// ===========================================================================
// X-7 CAVEAT -- strongest on this file
//
// Every variant here is popcount-bound except INC-COL, which issues none. binCV
// builds with no -march flags, so __builtin_popcountll is `call __popcountdi2@PLT`
// on x86_64 and fmov/cnt/uaddlv/fmov on aarch64. The ratio between a popcounting
// variant and a bit-reading one IS the thing that lowering changes, so x86 cannot
// rank these at all and this experiment closes only on the reference device. No
// -march flag is added: that is a dispatch decision (ROADMAP 2.3) that no
// experiment has settled, and changing it mid-experiment would confound exactly
// these comparisons.
//
// VALIDITY: measure::g_sink consumes every result; four distinct random images
// rotate through each timed body, on a call counter that runs on across batches --
// which matters here more than anywhere else, because DENSE recompute at W=31 is
// one call per batch and would otherwise have timed image 0 forever while the
// cheap variant it is being divided by rotated through all four; batches are
// calibrated, interleaved across the variants being compared and repeated with the
// spread reported; and every variant is checked against the shipped kernel on
// every window before anything is timed.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/logic.hpp"
#include "bincv-cpp/ops/reduce.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

namespace {

using bincv::BinMat;
using bincv::BinMatConstView;
using bincv::Rect;
using bincv::SplitCount;

constexpr int kInputs = 4;
constexpr int kRepeats = 7;
constexpr double kTargetMs = 40.0;
constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kKeypoints = 200;  // the reference gftt_max_corners
const int kWindows[] = {7, 15, 31};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

template <typename Word>
void fillRandom(BinMat<Word>& m, uint64_t seed) {
    uint64_t state = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            m.set(y, x, (measure::nextRandom(state) >> 63) != 0);
        }
    }
}

/// @brief One pixel, read as a bit. INC-COL is built out of this and nothing else.
template <typename Word>
inline uint32_t bitAt(const BinMatConstView<Word>& v, int y, int x) {
    constexpr size_t kBits = sizeof(Word) * 8;
    const Word w = v.row(static_cast<size_t>(y))[static_cast<size_t>(x) / kBits];
    return static_cast<uint32_t>((w >> (static_cast<size_t>(x) % kBits)) & Word(1));
}

/// @brief A rectangular sweep of window positions: top-left corners
///        (x0 + i, y0 + j) for i in [0, sx), j in [0, sy).
struct Sweep {
    int x0;
    int y0;
    int sx;
    int sy;
};

// ---------------------------------------------------------------------------
// AXIS 1 -- the three implementations
// ---------------------------------------------------------------------------

/// @brief The recompute path AS IT WAS BEFORE T2.11 item 4: ONE size_t
///        accumulator carried across every row and every word of the region.
///        MEASUREMENT CODE -- the only copy of a shipped kernel left in this file.
/// @note It is here because item 4's own effect is otherwise measurable only by
///       comparing absolute ns across two sessions, which is the one comparison
///       this harness is built to avoid: every other number in this file is
///       produced by variants timed INTERLEAVED on identical inputs in one
///       process. Item 4 is the change the other three are divided by, so what it
///       is worth had better not rest on a weaker method than they do.
/// @note Byte for byte the pre-split loop: clip, then one accumulator over the
///       whole region. The shipped kernel differs from it in exactly one respect,
///       which is where the sum lands.
template <typename Word>
size_t sweepRecomputeOneAccumulator(const BinMatConstView<Word>& v, const Sweep& s, int W) {
    size_t total = 0;
    for (int j = 0; j < s.sy; ++j) {
        for (int i = 0; i < s.sx; ++i) {
            const bincv::impl::RegionWords<Word> r =
                bincv::impl::clipRegion<Word>(v.width, v.height, Rect(s.x0 + i, s.y0 + j, W, W));
            if (r.isEmpty) continue;
            size_t region = 0;  // ONE chain through the popcount latency
            for (size_t y = r.y0; y < r.y1; ++y) {
                const Word* rs = v.row(y);
                bincv::impl::visitRowWords<Word>(r, [&](size_t k, Word mask) {
                    region += bincv::impl::popcountWord<Word>(static_cast<Word>(rs[k] & mask));
                });
            }
            total += region;
        }
    }
    return total;
}

/// @brief What the API ships: one countNonZero per window position, with T2.11
///        item 4's per-row partial sums inside it.
template <typename Word>
size_t sweepRecompute(const BinMatConstView<Word>& v, const Sweep& s, int W) {
    size_t total = 0;
    for (int j = 0; j < s.sy; ++j) {
        for (int i = 0; i < s.sx; ++i) {
            total += bincv::countNonZero(v, Rect(s.x0 + i, s.y0 + j, W, W));
        }
    }
    return total;
}

/// @brief INC-COL: the separable box accumulator. Per-column sums over the
///        window's rows, slid in x and then in y. Issues no popcount.
/// @param colSum Caller-provided scratch -- kernels do not allocate (CLAUDE.md).
///        Its size, (sx + W - 1) counters, IS the extra memory this design costs.
template <typename Word>
size_t sweepIncrementalColumns(const BinMatConstView<Word>& v, const Sweep& s, int W,
                               std::vector<uint32_t>& colSum) {
    const int cols = s.sx + W - 1;
    colSum.assign(static_cast<size_t>(cols), 0u);
    for (int r = 0; r < W; ++r) {
        for (int t = 0; t < cols; ++t) {
            colSum[static_cast<size_t>(t)] += bitAt(v, s.y0 + r, s.x0 + t);
        }
    }

    size_t total = 0;
    for (int j = 0; j < s.sy; ++j) {
        uint32_t win = 0;
        for (int t = 0; t < W; ++t) win += colSum[static_cast<size_t>(t)];
        total += win;
        for (int i = 1; i < s.sx; ++i) {
            win += colSum[static_cast<size_t>(i + W - 1)];
            win -= colSum[static_cast<size_t>(i - 1)];
            total += win;
        }
        if (j + 1 < s.sy) {
            const int rowOut = s.y0 + j;
            const int rowIn = s.y0 + j + W;
            for (int t = 0; t < cols; ++t) {
                colSum[static_cast<size_t>(t)] += bitAt(v, rowIn, s.x0 + t);
                colSum[static_cast<size_t>(t)] -= bitAt(v, rowOut, s.x0 + t);
            }
        }
    }
    return total;
}

/// @brief INC-ROW: the SHIPPED bincv::SlidingWindowCount, one accumulator per
///        column of window positions. The window sum gains the incoming row's
///        windowed popcount and loses the outgoing row's, so it stays
///        word-parallel and needs no scratch array.
/// @note One construction per x offset, which is where the column masks are
///       clipped -- the same amortization the measurement copy did by hoisting
///       impl::clipRegion out of the y loop, except that this is the library's own
///       and includes the row clipping the shipped class does per position.
template <typename Word>
size_t sweepIncrementalRows(const BinMatConstView<Word>& v, const Sweep& s, int W) {
    size_t total = 0;
    for (int i = 0; i < s.sx; ++i) {
        bincv::SlidingWindowCount<Word> acc(v, Rect(s.x0 + i, s.y0, W, W));
        for (int j = 0; j < s.sy; ++j) {
            total += acc.count();
            if (j + 1 < s.sy) acc.slideDown();
        }
    }
    return total;
}

struct Pattern {
    const char* name;
    const char* note;
    std::vector<Sweep> sweeps;  // filled per window size
    int windowsPerCall = 0;
};

/// @brief Builds the three access patterns for one window size. Every window is
///        fully inside the image; see the header for why.
std::vector<Pattern> buildPatterns(int W) {
    std::vector<Pattern> out;

    const int maxX = kWidth - W;
    const int maxY = kHeight - W;

    Pattern sparse;
    sparse.name = "SPARSE";
    sparse.note = "200 isolated keypoints (LK, ARCHITECTURE 7.5)";
    {
        uint64_t state = UINT64_C(0xC0FFEE);
        for (int k = 0; k < kKeypoints; ++k) {
            const int x = static_cast<int>(measure::nextRandom(state) %
                                           static_cast<uint64_t>(maxX + 1));
            const int y = static_cast<int>(measure::nextRandom(state) %
                                           static_cast<uint64_t>(maxY + 1));
            sparse.sweeps.push_back(Sweep{x, y, 1, 1});
        }
        sparse.windowsPerCall = kKeypoints;
    }
    out.push_back(sparse);

    Pattern search;
    search.name = "SEARCH";
    search.note = "200 keypoints x 8x8 sweep = 12800 windows, heavy local overlap";
    {
        const int S = 8;
        uint64_t state = UINT64_C(0x5EA12C);
        for (int k = 0; k < kKeypoints; ++k) {
            const int x = static_cast<int>(measure::nextRandom(state) %
                                           static_cast<uint64_t>(maxX - S + 2));
            const int y = static_cast<int>(measure::nextRandom(state) %
                                           static_cast<uint64_t>(maxY - S + 2));
            search.sweeps.push_back(Sweep{x, y, S, S});
        }
        search.windowsPerCall = kKeypoints * S * S;
    }
    out.push_back(search);

    Pattern dense;
    dense.name = "DENSE";
    dense.note = "every position in the frame (corner response, ARCHITECTURE 7.6)";
    dense.sweeps.push_back(Sweep{0, 0, maxX + 1, maxY + 1});
    dense.windowsPerCall = (maxX + 1) * (maxY + 1);
    out.push_back(dense);

    return out;
}

bool runAxis1() {
    using Word = uint32_t;  // the shipped default; E-2 owns the width axis

    std::printf("\n===========================================================\n");
    std::printf("AXIS 1 -- recompute per window versus a sliding accumulator\n");
    std::printf("===========================================================\n");
    std::printf("  %dx%d, uint32_t, %d keypoints. Rule: incremental must beat "
                "recompute by > 15%% at 31x31.\n",
                kWidth, kHeight, kKeypoints);

    std::vector<BinMat<Word>> imgs;
    imgs.reserve(static_cast<size_t>(kInputs));
    for (int i = 0; i < kInputs; ++i) {
        BinMat<Word> m(kWidth, kHeight);
        fillRandom(m, UINT64_C(0xD00D) + static_cast<uint64_t>(i) * UINT64_C(7919));
        imgs.push_back(std::move(m));
    }

    std::printf("\n  %-8s %-4s %13s %13s %13s %13s %9s %9s %9s %10s\n", "pattern", "W",
                "recomp-1acc", "recompute", "INC-COL", "INC-ROW", "item 4", "INC-COL",
                "INC-ROW", "INC-COL mem");
    std::printf("  %-8s %-4s %13s %13s %13s %13s %9s %9s %9s %10s\n", "", "", "ns/window",
                "ns/window", "ns/window", "ns/window", "1acc/rec", "vs recomp", "vs recomp",
                "bytes");

    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        std::vector<Pattern> patterns = buildPatterns(W);

        for (Pattern& p : patterns) {
            std::vector<uint32_t> scratch;

            // Hazard 4: all three implementations must agree on every image before
            // any of them is timed.
            for (int i = 0; i < kInputs; ++i) {
                const BinMatConstView<Word> v = imgs[static_cast<size_t>(i)].constView();
                size_t a = 0, b = 0, c = 0, d = 0;
                for (const Sweep& s : p.sweeps) {
                    a += sweepRecompute(v, s, W);
                    b += sweepIncrementalColumns(v, s, W, scratch);
                    c += sweepIncrementalRows(v, s, W);
                    d += sweepRecomputeOneAccumulator(v, s, W);
                }
                if (a != b || a != c || a != d) {
                    std::printf("  DISAGREEMENT %s W=%d image %d: recompute %zu, "
                                "INC-COL %zu, INC-ROW %zu, recompute-1acc %zu\n",
                                p.name, W, i, a, b, c, d);
                    return false;
                }
            }

            const Pattern* pp = &p;
            const std::vector<BinMat<Word>>* im = &imgs;
            std::vector<uint32_t>* sc = &scratch;
            std::vector<measure::Bench> benches;
            benches.push_back({"recompute-1acc", [pp, im, W](int i) {
                                   const BinMatConstView<Word> v =
                                       (*im)[static_cast<size_t>(i % kInputs)].constView();
                                   size_t acc = 0;
                                   for (const Sweep& s : pp->sweeps)
                                       acc += sweepRecomputeOneAccumulator(v, s, W);
                                   measure::g_sink += acc;
                               }});
            benches.push_back({"recompute", [pp, im, W](int i) {
                                   const BinMatConstView<Word> v =
                                       (*im)[static_cast<size_t>(i % kInputs)].constView();
                                   size_t acc = 0;
                                   for (const Sweep& s : pp->sweeps) acc += sweepRecompute(v, s, W);
                                   measure::g_sink += acc;
                               }});
            benches.push_back({"inc-col", [pp, im, sc, W](int i) {
                                   const BinMatConstView<Word> v =
                                       (*im)[static_cast<size_t>(i % kInputs)].constView();
                                   size_t acc = 0;
                                   for (const Sweep& s : pp->sweeps)
                                       acc += sweepIncrementalColumns(v, s, W, *sc);
                                   measure::g_sink += acc;
                               }});
            benches.push_back({"inc-row", [pp, im, W](int i) {
                                   const BinMatConstView<Word> v =
                                       (*im)[static_cast<size_t>(i % kInputs)].constView();
                                   size_t acc = 0;
                                   for (const Sweep& s : pp->sweeps)
                                       acc += sweepIncrementalRows(v, s, W);
                                   measure::g_sink += acc;
                               }});

            const std::vector<measure::Timing> t =
                measure::measureInterleaved(benches, kRepeats, kTargetMs);
            const double n = static_cast<double>(p.windowsPerCall);

            // The widest sweep decides how much scratch INC-COL needs.
            size_t widestSweep = 0;
            for (const Sweep& s : p.sweeps) {
                widestSweep = std::max(widestSweep, static_cast<size_t>(s.sx + W - 1));
            }

            std::printf("  %-8s %-4d %13.1f %13.1f %13.1f %13.1f %8.2fx %8.2fx %8.2fx %10zu\n",
                        p.name, W, t[0].medianNs / n, t[1].medianNs / n, t[2].medianNs / n,
                        t[3].medianNs / n, t[0].medianNs / t[1].medianNs,
                        t[1].medianNs / t[2].medianNs, t[1].medianNs / t[3].medianNs,
                        widestSweep * sizeof(uint32_t));
            std::printf("           spread (max-min)/median: recomp-1acc %.1f%%, recompute %.1f%%, "
                        "INC-COL %.1f%%, INC-ROW %.1f%%   [%s]\n",
                        t[0].spreadPct(), t[1].spreadPct(), t[2].spreadPct(), t[3].spreadPct(),
                        p.note);
        }
    }
    std::printf("\n  \"vs recomp\" > 1.00x means the accumulator is FASTER than the SHIPPED\n");
    std::printf("  recompute -- i.e. than the post-item-4 baseline, not X-11's pre-split one.\n");
    std::printf("  \"item 4\" > 1.00x means the per-row accumulator split made recompute faster.\n");
    std::printf("  axis 3's \"plane/4arg\" > 1.00x means the four-argument form is faster.\n");
    return true;
}

// ---------------------------------------------------------------------------
// AXIS 2 -- composed versus fused covariance
// ---------------------------------------------------------------------------

/// @brief Value equality on the shipped result type, for the agreement check.
bool sameCovariance(const bincv::CovarianceCount& a, const bincv::CovarianceCount& b) {
    return a.xx == b.xx && a.yy == b.yy && a.xy.whenClear == b.xy.whenClear &&
           a.xy.whenSet == b.xy.whenSet;
}

/// @brief ARCHITECTURE 7.5 through the T2.5/T2.6 primitives: three calls, therefore
///        three traversals of one window. Still a shipped composition -- a caller
///        who has not read X-11 writes exactly this -- so it is the denominator.
template <typename Word>
bincv::CovarianceCount covarianceComposed(const BinMatConstView<Word>& magX,
                                          const BinMatConstView<Word>& magY,
                                          const BinMatConstView<Word>& signXor,
                                          const Rect& window) {
    bincv::CovarianceCount out;
    out.xx = bincv::countNonZero(magX, window);
    out.yy = bincv::countNonZero(magY, window);
    out.xy = bincv::countAndSplit(magX, magY, signXor, window);
    return out;
}

/// @brief Keypoint windows for axes 2 and 3. Unlike axis 1 these DO include
///        windows that clip at the frame edge, which is the realistic case and
///        which both sides of these two comparisons handle identically.
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

/// @brief Ternary derivative inputs plus the precomputed sign_x ^ sign_y plane,
///        shared by axes 2 and 3.
template <typename Word>
struct DerivativeSet {
    std::vector<bincv::TernaryMat<Word>> dx;
    std::vector<bincv::TernaryMat<Word>> dy;
    std::vector<BinMat<Word>> sel;

    DerivativeSet() {
        dx.reserve(static_cast<size_t>(kInputs));
        dy.reserve(static_cast<size_t>(kInputs));
        sel.reserve(static_cast<size_t>(kInputs));
        for (int i = 0; i < kInputs; ++i) {
            bincv::TernaryMat<Word> a(kWidth, kHeight);
            bincv::TernaryMat<Word> b(kWidth, kHeight);
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

template <typename Word>
bool runAxis2(const char* wordName, const DerivativeSet<Word>& d) {
    std::printf("\n  %-9s %-4s %14s %14s %12s %22s\n", "word", "W", "composed",
                "fused", "composed/", "spread comp / fused");
    std::printf("  %-9s %-4s %14s %14s %12s\n", "", "", "ns/keypoint", "ns/keypoint", "fused");

    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        const std::vector<Rect> windows = keypointWindows(W);

        for (int i = 0; i < kInputs; ++i) {
            const size_t k = static_cast<size_t>(i);
            for (const Rect& w : windows) {
                const bincv::CovarianceCount a = covarianceComposed<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.sel[k].constView(), w);
                const bincv::CovarianceCount b = bincv::countCovariance<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.sel[k].constView(), w);
                if (!sameCovariance(a, b)) {
                    std::printf("  DISAGREEMENT composed vs fused at W=%d\n", W);
                    return false;
                }
            }
        }

        const DerivativeSet<Word>* dp = &d;
        const std::vector<Rect>* wp = &windows;
        std::vector<measure::Bench> benches;
        benches.push_back({"composed", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               size_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const bincv::CovarianceCount c = covarianceComposed<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->sel[k].constView(), w);
                                   acc += c.xx + c.yy + c.xy.whenClear + c.xy.whenSet;
                               }
                               measure::g_sink += acc;
                           }});
        benches.push_back({"fused", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               size_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const bincv::CovarianceCount c = bincv::countCovariance<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->sel[k].constView(), w);
                                   acc += c.xx + c.yy + c.xy.whenClear + c.xy.whenSet;
                               }
                               measure::g_sink += acc;
                           }});

        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        const double n = static_cast<double>(kKeypoints);
        std::printf("  %-9s %-4d %14.1f %14.1f %11.2fx %10.1f%% / %.1f%%\n", wordName, W,
                    t[0].medianNs / n, t[1].medianNs / n, t[0].medianNs / t[1].medianNs,
                    t[0].spreadPct(), t[1].spreadPct());
    }
    // CLAUDE.md requires memory and speed together, and "there is none" is an
    // answer that still has to be stated: both forms read the same views and
    // return by value, so neither needs scratch. Axis 1 and axis 3 both carry a
    // byte column; this one would look like an omission without the line.
    std::printf("  EXTRA MEMORY: 0 B for both forms -- neither needs scratch; the fused "
                "pass returns four counters in registers.\n");
    return true;
}

template <typename Word>
bool runAxis3(const char* wordName, DerivativeSet<Word>& d) {
    const size_t planeBytes = d.sel[0].sizeInWords() * sizeof(Word);

    std::printf("\n  %-9s %-4s %16s %16s %10s %14s\n", "word", "W", "plane (shipped)",
                "four-arg XOR", "plane/4arg", "spread p / 4");
    std::printf("  %-9s %-4s %16s %16s\n", "", "", "ns/keypoint", "ns/keypoint");

    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        const std::vector<Rect> windows = keypointWindows(W);

        for (int i = 0; i < kInputs; ++i) {
            const size_t k = static_cast<size_t>(i);
            for (const Rect& w : windows) {
                const SplitCount a = bincv::countAndSplit<Word>(d.dx[k].constMagnitude(0),
                                                                d.dy[k].constMagnitude(0),
                                                                d.sel[k].constView(), w);
                const SplitCount b = bincv::countAndSplit<Word>(
                    d.dx[k].constMagnitude(0), d.dy[k].constMagnitude(0), d.dx[k].constSign(),
                    d.dy[k].constSign(), w);
                if (a.whenClear != b.whenClear || a.whenSet != b.whenSet) {
                    std::printf("  DISAGREEMENT plane vs four-arg at W=%d\n", W);
                    return false;
                }
            }
        }

        const DerivativeSet<Word>* dp = &d;
        const std::vector<Rect>* wp = &windows;
        std::vector<measure::Bench> benches;
        benches.push_back({"plane", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               size_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const SplitCount c = bincv::countAndSplit<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->sel[k].constView(), w);
                                   acc += c.whenClear + c.whenSet;
                               }
                               measure::g_sink += acc;
                           }});
        benches.push_back({"four-arg", [dp, wp](int i) {
                               const size_t k = static_cast<size_t>(i % kInputs);
                               size_t acc = 0;
                               for (const Rect& w : *wp) {
                                   const SplitCount c = bincv::countAndSplit<Word>(
                                       dp->dx[k].constMagnitude(0), dp->dy[k].constMagnitude(0),
                                       dp->dx[k].constSign(), dp->dy[k].constSign(), w);
                                   acc += c.whenClear + c.whenSet;
                               }
                               measure::g_sink += acc;
                           }});

        const std::vector<measure::Timing> t =
            measure::measureInterleaved(benches, kRepeats, kTargetMs);
        const double n = static_cast<double>(kKeypoints);
        std::printf("  %-9s %-4d %16.1f %16.1f %9.2fx %7.1f%% / %.1f%%\n", wordName, W,
                    t[0].medianNs / n, t[1].medianNs / n, t[0].medianNs / t[1].medianNs,
                    t[0].spreadPct(), t[1].spreadPct());
    }

    // The other half of this axis, which no threshold covers and which is
    // therefore reported next to the speed rather than folded into it.
    DerivativeSet<Word>* dp = &d;
    std::vector<measure::Bench> formBench;
    formBench.push_back({"form plane", [dp](int i) {
                             const size_t k = static_cast<size_t>(i % kInputs);
                             // Rewrites the plane with the value it already holds, so
                             // the agreement checks above stay valid.
                             bincv::bitwiseXor(dp->dx[k].constSign(), dp->dy[k].constSign(),
                                               dp->sel[k].view());
                             measure::g_sink += dp->sel[k].data()[k];
                         }});
    const std::vector<measure::Timing> ft =
        measure::measureInterleaved(formBench, kRepeats, kTargetMs);

    // The plane is one bit per pixel OF THE LEVEL IT BELONGS TO, so it shrinks by 4x
    // per level exactly as the derivative planes do. Printing only the 640x480
    // figure and calling it "per pyramid level" overstates the pyramid-wide cost by
    // about 3x -- in favour of the memory side of this axis, which is the direction
    // an error must never go unremarked. The level-invariant statement is the
    // relative one: a fifth plane against the four the covariance already reads.
    const size_t levels[] = {1, 4, 16, 64};  // area divisors for L0..L3
    size_t planeLadder = 0;
    for (size_t li = 0; li < sizeof(levels) / sizeof(levels[0]); ++li) {
        planeLadder += planeBytes / levels[li];
    }
    std::printf("\n  MEMORY (%s): the selector plane is %zu B at %dx%d and scales with the "
                "level\n",
                wordName, planeBytes, kWidth, kHeight);
    std::printf("  (%zu B at L1, %zu B at L2, ~%zu B summed over a 4-level pyramid), held "
                "for the frame's lifetime.\n",
                planeBytes / 4, planeBytes / 16, planeLadder);
    std::printf("  Level-invariant form: +25%% on the derivative working set of every "
                "level -- a fifth plane against dx/dy's four.\n");
    std::printf("  Forming it costs %.1f us at %dx%d (%.1f ns amortized over %d "
                "keypoints); the four-argument form allocates nothing, at any level.\n",
                ft[0].medianNs / 1000.0, kWidth, kHeight,
                ft[0].medianNs / static_cast<double>(kKeypoints), kKeypoints);
    return true;
}

}  // namespace

int main() {
    std::printf("=== E-3 / T2.10: window reductions -- three interface axes ===\n");
#if defined(__aarch64__)
    std::printf("target: aarch64 -- AUTHORITATIVE (the reference device closes E-3)\n");
#else
    std::printf("target: not aarch64 -- INDICATIVE ONLY. Every variant here is "
                "popcount-bound except INC-COL,\n"
                "        which issues none, so the x86 popcount lowering (X-7) can "
                "invert the ranking outright.\n");
#endif
    std::printf("Decision rules are in this file's header, written before measuring "
                "(EXPERIMENTS.md X-11).\n");

    bool ok = runAxis1();

    std::printf("\n===========================================================\n");
    std::printf("AXIS 2 -- the 2x2 covariance composed out of T2.6 versus fused\n");
    std::printf("===========================================================\n");
    std::printf("  Rule: fused beats composed by > 15%% at 31x31 -> add a covariance "
                "entry point to T2.6.\n");
    DerivativeSet<uint32_t> d32;
    DerivativeSet<uint64_t> d64;
    ok = runAxis2<uint32_t>("uint32_t", d32) && ok;
    ok = runAxis2<uint64_t>("uint64_t", d64) && ok;
    std::printf("  (uint64_t at W=31 is X-8's measurement, repeated here in the same "
                "session for comparability.)\n");

    std::printf("\n===========================================================\n");
    std::printf("AXIS 3 -- selector plane versus a four-argument countAndSplit\n");
    std::printf("===========================================================\n");
    std::printf("  No numeric threshold exists for this axis. Both memory and speed "
                "are reported; the\n  weighing is against CLAUDE.md's tiebreak -- "
                "memory wins when the goals conflict.\n");
    ok = runAxis3<uint32_t>("uint32_t", d32) && ok;

    std::printf("\nsink: %zu (printed so nothing above can be optimized away)\n",
                static_cast<size_t>(measure::g_sink));
    return ok ? 0 : 1;
}
