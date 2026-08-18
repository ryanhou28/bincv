// E-3 / T2.10 -- incremental versus recomputed window reductions, and the two
// other T2.6 interface questions that were registered alongside it.
//
// T3.6 cannot start until this closes, because all three axes decide the SHAPE of
// ops/reduce.hpp rather than its speed.
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
//            essentially no extra memory -- so if it is the one that wins, the
//            interface change T2.6 would need is much smaller.
//
// Both incremental forms are MEASUREMENT CODE. INC-ROW reaches into impl:: for the
// row masks (as benchmark/reduce_target_benchmark.cpp already does) so that the
// comparison is between ALGORITHMS and not between one algorithm and the region
// clipping the other would pay per call. Neither is proposed for include/ by this
// file; that is the decision the numbers inform.
//
// Windows are placed fully inside the image on this axis. Edge clipping is real
// (ARCHITECTURE 7.5) and axes 2 and 3 below include it, but the accumulators
// would need their own clipping logic to agree with the shipped kernel at the
// border, and a comparison between two implementations of clipping is not the
// question E-3 asks.
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
// Re-measured here at three window sizes and at two word widths, in the same
// session as axis 1, rather than resting on X-8's single 31x31 uint64_t point.
//
// ===========================================================================
// AXIS 3 -- frame-sized selector plane versus a four-argument countAndSplit
// ===========================================================================
//
// countAndSplit's selector `c` is sign_x ^ sign_y, which today the caller forms as
// a whole extra plane, once per pyramid level. A four-argument form taking the two
// sign planes and XOR-ing them inside the word loop needs no plane at all.
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
// rotate through each timed body; batches are calibrated, interleaved across the
// variants being compared and repeated with the spread reported; and every variant
// is checked against the shipped kernel on every window before anything is timed.

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

/// @brief What the API ships today: one countNonZero per window position.
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

/// @brief Windowed popcount of ONE row, through the same masks the shipped kernel
///        uses. MEASUREMENT CODE: it reaches into impl:: so that INC-ROW is
///        compared as an algorithm rather than as a call-overhead difference.
template <typename Word>
inline size_t rowWindowCount(const BinMatConstView<Word>& v, int y,
                             const bincv::impl::RegionWords<Word>& r) {
    const Word* rs = v.row(static_cast<size_t>(y));
    size_t total = 0;
    bincv::impl::visitRowWords<Word>(r, [&](size_t i, Word mask) {
        total += bincv::impl::popcountWord<Word>(static_cast<Word>(rs[i] & mask));
    });
    return total;
}

/// @brief INC-ROW: slide vertically with ONE scalar accumulator. The window sum
///        gains the incoming row's windowed popcount and loses the outgoing row's,
///        so it stays word-parallel and needs no scratch array.
template <typename Word>
size_t sweepIncrementalRows(const BinMatConstView<Word>& v, const Sweep& s, int W) {
    size_t total = 0;
    for (int i = 0; i < s.sx; ++i) {
        // The column mask set is the same for every row of this x offset, so it is
        // built once per offset rather than once per window.
        const bincv::impl::RegionWords<Word> r =
            bincv::impl::clipRegion<Word>(v.width, v.height, Rect(s.x0 + i, s.y0, W, W));
        size_t win = 0;
        for (int t = 0; t < W; ++t) win += rowWindowCount(v, s.y0 + t, r);
        total += win;
        for (int j = 1; j < s.sy; ++j) {
            win += rowWindowCount(v, s.y0 + j + W - 1, r);
            win -= rowWindowCount(v, s.y0 + j - 1, r);
            total += win;
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

    std::printf("\n  %-8s %-4s %14s %14s %14s %10s %10s %12s\n", "pattern", "W",
                "recompute", "INC-COL", "INC-ROW", "INC-COL", "INC-ROW", "INC-COL mem");
    std::printf("  %-8s %-4s %14s %14s %14s %10s %10s %12s\n", "", "", "ns/window",
                "ns/window", "ns/window", "vs recomp", "vs recomp", "bytes");

    for (size_t wi = 0; wi < sizeof(kWindows) / sizeof(kWindows[0]); ++wi) {
        const int W = kWindows[wi];
        std::vector<Pattern> patterns = buildPatterns(W);

        for (Pattern& p : patterns) {
            std::vector<uint32_t> scratch;

            // Hazard 4: all three implementations must agree on every image before
            // any of them is timed.
            for (int i = 0; i < kInputs; ++i) {
                const BinMatConstView<Word> v = imgs[static_cast<size_t>(i)].constView();
                size_t a = 0, b = 0, c = 0;
                for (const Sweep& s : p.sweeps) {
                    a += sweepRecompute(v, s, W);
                    b += sweepIncrementalColumns(v, s, W, scratch);
                    c += sweepIncrementalRows(v, s, W);
                }
                if (a != b || a != c) {
                    std::printf("  DISAGREEMENT %s W=%d image %d: recompute %zu, "
                                "INC-COL %zu, INC-ROW %zu\n",
                                p.name, W, i, a, b, c);
                    return false;
                }
            }

            const Pattern* pp = &p;
            const std::vector<BinMat<Word>>* im = &imgs;
            std::vector<uint32_t>* sc = &scratch;
            std::vector<measure::Bench> benches;
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

            std::printf("  %-8s %-4d %14.1f %14.1f %14.1f %9.2fx %9.2fx %12zu\n", p.name, W,
                        t[0].medianNs / n, t[1].medianNs / n, t[2].medianNs / n,
                        t[0].medianNs / t[1].medianNs, t[0].medianNs / t[2].medianNs,
                        widestSweep * sizeof(uint32_t));
            std::printf("           spread (max-min)/median: recompute %.1f%%, "
                        "INC-COL %.1f%%, INC-ROW %.1f%%   [%s]\n",
                        t[0].spreadPct(), t[1].spreadPct(), t[2].spreadPct(), p.note);
        }
    }
    std::printf("\n  \"vs recomp\" > 1.00x means the accumulator is FASTER than recompute;\n");
    std::printf("  axis 3's \"plane/4arg\" > 1.00x means the four-argument form is faster.\n");
    return true;
}

// ---------------------------------------------------------------------------
// AXIS 2 -- composed versus fused covariance
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

/// @brief ARCHITECTURE 7.5 through the T2.6 primitives: three calls, therefore
///        three traversals of one window.
template <typename Word>
Covariance covarianceComposed(const BinMatConstView<Word>& magX, const BinMatConstView<Word>& magY,
                              const BinMatConstView<Word>& signXor, const Rect& window) {
    Covariance out;
    out.xx = bincv::countNonZero(magX, window);
    out.yy = bincv::countNonZero(magY, window);
    out.xy = bincv::countAndSplit(magX, magY, signXor, window);
    return out;
}

/// @brief The same four numbers from ONE traversal. MEASUREMENT CODE -- the shape
///        a covariance entry point would have, not a proposal in itself.
template <typename Word>
Covariance covarianceFused(const BinMatConstView<Word>& magX, const BinMatConstView<Word>& magY,
                           const BinMatConstView<Word>& signXor, const Rect& window) {
    Covariance out;
    const bincv::impl::RegionWords<Word> r =
        bincv::impl::clipRegion<Word>(magX.width, magX.height, window);
    if (r.isEmpty) return out;

    for (size_t y = r.y0; y < r.y1; ++y) {
        const Word* rx = magX.row(y);
        const Word* ry = magY.row(y);
        const Word* rc = signXor.row(y);
        bincv::impl::visitRowWords<Word>(r, [&](size_t i, Word mask) {
            const Word wx = static_cast<Word>(rx[i] & mask);
            const Word wy = static_cast<Word>(ry[i] & mask);
            const Word both = static_cast<Word>(wx & wy);
            const size_t total = bincv::impl::popcountWord<Word>(both);
            const size_t set =
                bincv::impl::popcountWord<Word>(static_cast<Word>(both & rc[i]));
            out.xx += bincv::impl::popcountWord<Word>(wx);
            out.yy += bincv::impl::popcountWord<Word>(wy);
            out.xy.whenSet += set;
            out.xy.whenClear += total - set;
        });
    }
    return out;
}

/// @brief The four-argument alternative of axis 3: no selector plane, the XOR
///        happens in the word loop. MEASUREMENT CODE.
template <typename Word>
SplitCount countAndSplitXor(const BinMatConstView<Word>& a, const BinMatConstView<Word>& b,
                            const BinMatConstView<Word>& c0, const BinMatConstView<Word>& c1,
                            const Rect& window) {
    SplitCount out;
    const bincv::impl::RegionWords<Word> r =
        bincv::impl::clipRegion<Word>(a.width, a.height, window);
    if (r.isEmpty) return out;

    for (size_t y = r.y0; y < r.y1; ++y) {
        const Word* ra = a.row(y);
        const Word* rb = b.row(y);
        const Word* r0 = c0.row(y);
        const Word* r1 = c1.row(y);
        bincv::impl::visitRowWords<Word>(r, [&](size_t i, Word mask) {
            const Word both = static_cast<Word>(static_cast<Word>(ra[i] & rb[i]) & mask);
            const Word sel = static_cast<Word>(r0[i] ^ r1[i]);
            const size_t total = bincv::impl::popcountWord<Word>(both);
            const size_t set = bincv::impl::popcountWord<Word>(static_cast<Word>(both & sel));
            out.whenSet += set;
            out.whenClear += total - set;
        });
    }
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
                const Covariance a = covarianceComposed<Word>(d.dx[k].constMagnitude(0),
                                                              d.dy[k].constMagnitude(0),
                                                              d.sel[k].constView(), w);
                const Covariance b = covarianceFused<Word>(d.dx[k].constMagnitude(0),
                                                           d.dy[k].constMagnitude(0),
                                                           d.sel[k].constView(), w);
                if (!(a == b)) {
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
                                   const Covariance c = covarianceComposed<Word>(
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
                                   const Covariance c = covarianceFused<Word>(
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
                const SplitCount b = countAndSplitXor<Word>(
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
                                   const SplitCount c = countAndSplitXor<Word>(
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

    std::printf("\n  MEMORY (%s): the selector plane is %zu B per pyramid level, "
                "allocated for the frame's lifetime.\n",
                wordName, planeBytes);
    std::printf("  Forming it costs %.1f us per level (%.1f ns amortized over %d "
                "keypoints); the four-argument form allocates nothing.\n",
                ft[0].medianNs / 1000.0, ft[0].medianNs / static_cast<double>(kKeypoints),
                kKeypoints);
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
