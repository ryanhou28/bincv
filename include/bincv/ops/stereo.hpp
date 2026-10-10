#pragma once

/// @file stereo.hpp
/// @brief Sparse rectified stereo matching: descriptor search along the epipolar
/// row, then Hamming window refinement to sub-pixel disparity. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHAT THIS IS
///
/// A stereo pipeline turns a left keypoint into a depth by finding it in the
/// right image. On a RECTIFIED pair that search is one-dimensional -- same row,
/// disparity in a known range -- and both stages of the standard method are
/// operations this library is built from:
///
/// 1. **Coarse**: the left keypoint's descriptor against right-image candidates
///    in the same row band -- `popcount(a ^ b)`, the matcher ops/descriptor.hpp
///    already ships, with the row-and-range constraint it does not have.
/// 2. **Fine**: a small window slid along the row around the coarse hit, scored
///    by `popcount((L ^ R) & mask)` per word -- the exact inner loop
///    ops/blockMatch.hpp runs, pointed at one axis -- with the parabola fit that
///    file already carries for the sub-pixel offset.
///
/// The ORB-SLAM family does this shape with SAD on bytes. On packed frames the
/// window score is an XOR and a bulk count, which is the whole thesis.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND WHY THE STAGES ARE SEPARABLE
///
/// `cv::StereoBM` / `cv::StereoSGBM` are dense; a sparse per-keypoint rectified
/// search has no OpenCV equivalent and borrows no OpenCV name.
///
/// The two stages are separate entry points because they have separate callers:
/// a pipeline with descriptors runs both; a caller with an initial disparity
/// from anywhere else (a projection of a map point, a previous frame) fills
/// `StereoMatch::disparity` itself and runs refinement alone.
///
/// ---------------------------------------------------------------------------
/// CONTRACTS -- ops/blockMatch.hpp's, unchanged
///
/// * Views, never containers; no heap, no scratch, no throw. The refinement
///   keeps a running minimum, never a cost surface.
/// * **It is a ONE-BIT refinement**, like the block matcher and for the same
///   reason: Hamming distance is defined on bits.
/// * Windows clip at the frame edge; right-frame reads outside the image
///   replicate -- through the same impl machinery, so the border behavior
///   cannot drift from the tracker's.
/// * Rectification is the CALLER's (it turns one wide image into another --
///   the input boundary). `rowTolerance` absorbs the residual error of a real
///   rectification in the DESCRIPTOR stage; the refinement stage trusts the
///   epipolar geometry and scans one axis only.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"
#include "blockMatch.hpp"   // BlockMatchLevel, impl::hammingAt, impl::parabolicOffset
#include "descriptor.hpp"   // hammingDistance
#include "reduce.hpp"

// The staged refinement: selected by default where a population count is software --
// the targets Lucas-Kanade's window-count arm is selected on -- and COMPILED only
// there unless a translation unit asks for it, so a core with a count instruction runs
// the per-row path instruction for instruction. It changes inline bodies, so like
// BINCV_LK_WINDOW_COUNT it must be the same in every translation unit of a program.
#define BINCV_STEREO_WINDOW_COUNT_DEFAULT BINCV_LK_WINDOW_COUNT_DEFAULT
#if !defined(BINCV_STEREO_WINDOW_COUNT)
#define BINCV_STEREO_WINDOW_COUNT BINCV_STEREO_WINDOW_COUNT_DEFAULT
#endif
#if BINCV_STEREO_WINDOW_COUNT_DEFAULT && !BINCV_STEREO_WINDOW_COUNT
#error "BINCV_STEREO_WINDOW_COUNT=0 on a target whose default selects the staged refinement"
#endif
#if BINCV_STEREO_WINDOW_COUNT
#include <type_traits>
#endif
// The descriptor stage's row test on the coordinates' bit patterns: selected, and
// COMPILED, only on M-profile, where a float comparison reaches a branch only through
// the FPU's flags (`vmrs`), and the two-sided test's branch goes either way at random.
#if defined(__ARM_ARCH_PROFILE) && (__ARM_ARCH_PROFILE == 'M')
#define BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT 1
#else
#define BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT 0
#endif
#if !defined(BINCV_STEREO_INTEGER_ROW_GATE)
#define BINCV_STEREO_INTEGER_ROW_GATE BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT
#endif
#if BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT && !BINCV_STEREO_INTEGER_ROW_GATE
#error "BINCV_STEREO_INTEGER_ROW_GATE=0 on a target whose default selects it"
#endif
#if BINCV_STEREO_INTEGER_ROW_GATE
#include <cstring>
#include <limits>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Search and window parameters for the sparse rectified stereo matcher.
/// **API TIER 3.**
struct StereoMatchParams {
    /// @brief Disparity range, level-0 pixels, `0 <= min < max`. Disparity is
    /// `leftX - rightX`, so the range encodes "how close can a point be".
    int minDisparity = 0;
    int maxDisparity = 64;

    /// @brief Rows the DESCRIPTOR stage will look above and below the left
    /// keypoint's row. A perfect rectification needs 0; real ones do not,
    /// and the caller knows theirs.
    int rowTolerance = 2;

    /// @brief Largest descriptor Hamming distance the coarse stage accepts.
    /// @note An ABSOLUTE bit count, and the default encodes ONE configuration:
    /// 100 is the ORB-SLAM family's high threshold at 256 bits. A caller
    /// with shorter descriptors must scale it down themselves -- at 128
    /// bits the default gates almost nothing, and wrong coarse matches
    /// come out of refinement as confident sub-pixel depths. A ratio test
    /// is deliberately NOT applied here: along an epipolar row the
    /// second-best candidate is often the true match's neighbour, and
    /// rejecting on that ratio throws away exactly the matches the
    /// refinement stage exists to sharpen.
    unsigned maxHamming = 100;

    /// @brief Refinement window, pixels, > 2 on a side.
    int winWidth = 11;
    int winHeight = 11;

    /// @brief Half-extent of the refinement scan around the coarse disparity.
    int refineRadius = 4;

    /// @brief Parabolic sub-pixel fit on the disparity axis, three window scores.
    /// The integer floor is the block matcher's 0.2887 px per axis; depth
    /// is 1/disparity, so the fit matters more here than anywhere.
    bool subPixel = true;
};

/// @brief One left keypoint's stereo result. **API TIER 3.**
struct StereoMatch {
    float disparity = 0.0f;    ///< `leftX - rightX`, level-0 pixels
    unsigned distance = 0;     ///< descriptor Hamming of the accepted candidate
    size_t rightIndex = 0;     ///< index into the right keypoint set
    uint8_t valid = 0;         ///< 0: no candidate survived; the other fields are void
};

#if BINCV_STEREO_INTEGER_ROW_GATE
namespace impl {

/// @brief Whether `stereoDescriptorMatch` tests rows on bit patterns. **INTERNAL.**
/// Exists where it is compiled (BINCV_STEREO_INTEGER_ROW_GATE); on by default on
/// M-profile. The tests and the board flip it to hold the two to each other and to
/// time them.
inline bool& stereoIntegerRowGateEnabled() {
    static bool on = BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT != 0;
    return on;
}

/// @brief A float's bits as an integer that orders as the float does. **INTERNAL.**
/// A negative float's magnitude bits are flipped, so its key falls as it grows. `-0`
/// sorts just below `+0`; the two compare equal, so they sit on the same side of any
/// interval a float comparison draws.
inline int32_t stereoOrderedKey(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return static_cast<int32_t>(b ^ ((0u - (b >> 31)) >> 1));
}

/// @brief The float whose key is `k`; the inverse of `stereoOrderedKey`. **INTERNAL.**
inline float stereoKeyFloat(int32_t k) {
    uint32_t b = static_cast<uint32_t>(k);
    b ^= (b & 0x80000000u) != 0u ? 0x7FFFFFFFu : 0u;
    float f;
    std::memcpy(&f, &b, sizeof f);
    return f;
}

/// @brief The last key from `pass` towards `fail` that `passes`, given that it passes at
/// `pass` and fails at `fail` and changes its answer once between them. **INTERNAL.**
/// The search starts at `guess` and gallops outward, so an end that lies a few keys from
/// it costs a few evaluations; one that does not costs a bisection of what is left. In
/// 64 bits, because an infinity's key and a finite one's can be 2^32 apart.
template <typename Passes>
inline int64_t stereoKeyBoundary(int64_t pass, int64_t fail, int64_t guess,
                                 const Passes& passes) {
    const int64_t dir = fail > pass ? 1 : -1;
    if ((guess - pass) * dir > 0 && (fail - guess) * dir > 0) {
        if (passes(guess)) {
            pass = guess;
            for (int64_t step = 1; (fail - (pass + dir * step)) * dir > 0; step *= 2) {
                const int64_t probe = pass + dir * step;
                if (!passes(probe)) {
                    fail = probe;
                    break;
                }
                pass = probe;
            }
        } else {
            fail = guess;
            for (int64_t step = 1; (fail - dir * step - pass) * dir > 0; step *= 2) {
                const int64_t probe = fail - dir * step;
                if (passes(probe)) {
                    pass = probe;
                    break;
                }
                fail = probe;
            }
        }
    }
    while ((fail - pass) * dir > 1) {
        const int64_t mid = pass + (fail - pass) / 2;
        (passes(mid) ? pass : fail) = mid;
    }
    return pass;
}

/// @brief The keys `[lo, hi]` of every y the published row test passes beside `yL`.
/// **INTERNAL.** `y - yL` rounds monotonically in `y`, so the y with
/// `-rowTol <= y - yL <= rowTol` are one interval of floats, hence of keys. Each end is
/// searched for between a key the test passes (`yL`'s own) and one it fails (an
/// infinity's), from where float arithmetic puts it, `yL -+ rowTol`, evaluating the
/// published comparison itself -- so the interval is that comparison's to the last bit.
/// `yL` must be finite and `rowTol >= 0`.
inline void stereoRowKeys(float yL, float rowTol, int32_t& lo, int32_t& hi) {
    const auto passes = [yL, rowTol](int64_t k) {
        const float dy = stereoKeyFloat(static_cast<int32_t>(k)) - yL;
        return !(dy > rowTol || dy < -rowTol);
    };
    const float inf = std::numeric_limits<float>::infinity();
    const int64_t inside = stereoOrderedKey(yL);
    lo = static_cast<int32_t>(stereoKeyBoundary(inside, stereoOrderedKey(-inf),
                                                stereoOrderedKey(yL - rowTol), passes));
    hi = static_cast<int32_t>(stereoKeyBoundary(inside, stereoOrderedKey(inf),
                                                stereoOrderedKey(yL + rowTol), passes));
}

/// @brief Whether `v` is a NaN or an infinity, from its bits. **INTERNAL.**
inline bool stereoNotFinite(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    return (b & 0x7F800000u) == 0x7F800000u;
}

} // namespace impl
#endif

/// @brief COARSE stage: each left descriptor against the right keypoints in its
/// row band and disparity range. **API TIER 3.**
/// @param leftXY / rightXY (x, y) pairs, interleaved, level-0 pixels -- the same
/// raw-array contract as the descriptor family.
/// @param leftDesc / rightDesc `words` words per keypoint, `computeBrief`'s or
/// `computeBriefSteered`'s layout.
/// @param out One `StereoMatch` per LEFT keypoint, every entry written.
/// @note Brute force over the right set per left keypoint -- O(L * R) distance
/// tests, each gated by two float comparisons before any descriptor word is
/// read. At pipeline counts (hundreds against hundreds) the gate leaves a
/// few candidates per keypoint; a row-bucketed index would need scratch,
/// and the no-scratch rule outranks a constant factor at those counts.
template <typename WordType>
inline void stereoDescriptorMatch(const float* leftXY, size_t leftCount,
                                  const WordType* leftDesc, const float* rightXY,
                                  size_t rightCount, const WordType* rightDesc,
                                  size_t words, StereoMatch* out,
                                  const StereoMatchParams& params = StereoMatchParams()) {
    if (leftCount == 0) return;
    BINCV_ASSERT(leftXY != nullptr && leftDesc != nullptr && out != nullptr,
                 "stereoDescriptorMatch: null left argument");
    BINCV_ASSERT(rightCount == 0 || (rightXY != nullptr && rightDesc != nullptr),
                 "stereoDescriptorMatch: null right argument");
    BINCV_ASSERT(params.minDisparity >= 0 && params.maxDisparity > params.minDisparity,
                 "stereoDescriptorMatch: need 0 <= minDisparity < maxDisparity");
    BINCV_ASSERT(params.rowTolerance >= 0, "stereoDescriptorMatch: rowTolerance < 0");

    const float rowTol = static_cast<float>(params.rowTolerance);
    const float minD = static_cast<float>(params.minDisparity);
    const float maxD = static_cast<float>(params.maxDisparity);
#if BINCV_STEREO_INTEGER_ROW_GATE
    // A right y that is a NaN passes the published test -- both comparisons are false --
    // and no interval of keys holds a NaN, so a call with one takes the published loop.
    bool keyGate = impl::stereoIntegerRowGateEnabled();
    for (size_t j = 0; keyGate && j < rightCount; ++j) {
        if (rightXY[2 * j + 1] != rightXY[2 * j + 1]) keyGate = false;
    }
#endif

    for (size_t i = 0; i < leftCount; ++i) {
        const float xL = leftXY[2 * i];
        const float yL = leftXY[2 * i + 1];
        unsigned best = 0xFFFFFFFFu;
        size_t bestIdx = 0;
#if BINCV_STEREO_INTEGER_ROW_GATE
        // ON BIT PATTERNS, where the compiler's own macros say the float test is the
        // cost: the right rows this left keypoint's test passes are one interval of keys,
        // found once, so a candidate is one integer range check instead of a subtraction
        // and two float comparisons through the FPU's flags. A left y that is not finite
        // takes the published test, which is what defines its answer.
        if (keyGate && !impl::stereoNotFinite(yL)) {
            int32_t keyLo = 0, keyHi = 0;
            impl::stereoRowKeys(yL, rowTol, keyLo, keyHi);
            const uint32_t span = static_cast<uint32_t>(keyHi) - static_cast<uint32_t>(keyLo);
            const auto consider = [&](size_t j) {
                const float d = xL - rightXY[2 * j];
                if (d < minD || d > maxD) return;
                const unsigned dist = hammingDistance<WordType>(leftDesc + i * words,
                                                                rightDesc + j * words, words);
                if (dist < best) {
                    best = dist;
                    bestIdx = j;
                }
            };
            if (keyLo >= 0 || keyHi < 0) {
                // On one side of zero the keys are the floats' own bits -- reversed below
                // it, so the range's low end is the high key's -- and the pairs compare
                // the bits as they lie in memory.
                const uint32_t rawLo = static_cast<uint32_t>(keyLo >= 0 ? keyLo
                                                                        : keyHi ^ 0x7FFFFFFF);
                for (size_t j = 0; j < rightCount; ++j) {
                    uint32_t b;
                    std::memcpy(&b, &rightXY[2 * j + 1], sizeof b);
                    if (b - rawLo <= span) consider(j);
                }
            } else {
                for (size_t j = 0; j < rightCount; ++j) {
                    const uint32_t k =
                        static_cast<uint32_t>(impl::stereoOrderedKey(rightXY[2 * j + 1]));
                    if (k - static_cast<uint32_t>(keyLo) <= span) consider(j);
                }
            }
        } else
#endif
        for (size_t j = 0; j < rightCount; ++j) {
            const float dy = rightXY[2 * j + 1] - yL;
            if (dy > rowTol || dy < -rowTol) continue;
            const float d = xL - rightXY[2 * j];
            if (d < minD || d > maxD) continue;
            const unsigned dist = hammingDistance<WordType>(leftDesc + i * words,
                                                            rightDesc + j * words, words);
            if (dist < best) {
                best = dist;
                bestIdx = j;
            }
        }
        StereoMatch m;
        // The no-candidate sentinel is excluded EXPLICITLY, not by the Hamming
        // gate: a caller who raises maxHamming to "accept everything" must get
        // "no candidate", never a match fabricated from rightXY[0] -- and with
        // an empty right set, never a read through a null rightXY.
        if (best != 0xFFFFFFFFu && best <= params.maxHamming) {
            m.disparity = xL - rightXY[2 * bestIdx];
            m.distance = best;
            m.rightIndex = bestIdx;
            m.valid = 1;
        }
        out[i] = m;
    }
}

#if BINCV_STEREO_WINDOW_COUNT
namespace impl {

/// @brief The refinement's selection rule over the staged scores: the integer scan
/// over `[lo, hi]` (the first minimum wins), the parabola through the best score's
/// neighbours, and the clamp to the stated range. **INTERNAL.** `cost(d)` is the
/// window's Hamming distance at disparity `d`.
/// @note A COPY of the rule `stereoRefineDisparity` spells inline for the per-row
/// path, kept apart because routing that path through this template changes what GCC
/// inlines there -- the x86-64 build of it shrank from 800 instructions to 347, with
/// `hammingAt` split out -- and the per-row path is what a core with a count
/// instruction runs and has published figures for. `Stereo.StagedArmIsBitExact`
/// holds the two copies to one another.
template <typename Cost>
inline double stereoScanAndFit(long long lo, long long hi, long long minD, long long maxD,
                               bool subPixel, const Cost& cost) {
    long long bestCost = -1, bestD = lo;
    for (long long d = lo; d <= hi; ++d) {
        const long long c = cost(d);
        if (bestCost < 0 || c < bestCost) {
            bestCost = c;
            bestD = d;
        }
    }

    double sub = 0.0;
    if (subPixel) {
        // Neighbours may sit outside [lo, hi]; the fit wants the true local
        // shape, and one extra score per side is cheaper than a biased vertex.
        const long long cm = cost(bestD - 1);
        const long long cp = cost(bestD + 1);
        sub = parabolicOffset(cm, bestCost, cp);
    }
    double refined = static_cast<double>(bestD) + sub;
    if (refined < static_cast<double>(minD)) refined = static_cast<double>(minD);
    if (refined > static_cast<double>(maxD)) refined = static_cast<double>(maxD);
    return refined;
}

/// @brief Whether `stereoRefineDisparity` stages its windows. **INTERNAL.** Exists where
/// the arm is compiled (BINCV_STEREO_WINDOW_COUNT); on by default where a population
/// count is software. The tests and the board flip it to hold the two arms to each
/// other and to time them.
inline bool& stereoWindowCountEnabled() {
    static bool on = BINCV_STEREO_WINDOW_COUNT_DEFAULT != 0;
    return on;
}

/// Rows the staged arm holds; a taller window takes the per-row path. A byte lane
/// counts 8 a row, so 31 rows are what it carries without a fold.
inline constexpr size_t kStereoStagedMaxRows = 31;

/// @brief Where one keypoint's staged refinement reads, or `ok == false`. **INTERNAL.**
/// The left window and a right STRIP holding the window at every disparity the
/// refinement can score -- `[lo - 1, hi + 1]`, the parabola's neighbours included --
/// both inside their frames, so nothing replicates, and the strip in one 32-bit word.
struct StereoStagedPlan {
    bool ok = false;
    size_t x0 = 0;           ///< the window's first column
    size_t y0 = 0;           ///< the window's first row
    size_t stripX0 = 0;      ///< the strip's first column, `x0 - (hi + 1)`
    size_t stripWidth = 0;   ///< `winW + (hi - lo) + 2`, at most 32
};

template <typename WordType>
inline StereoStagedPlan stereoStagedPlan(size_t width, size_t height, long long anchorX,
                                         long long anchorY, int winW, int winH,
                                         long long lo, long long hi) {
    StereoStagedPlan p;
    if (!std::is_same<WordType, uint32_t>::value) return p;
    if (winH > static_cast<int>(kStereoStagedMaxRows)) return p;
    const long long w = static_cast<long long>(width);
    const long long stripX0 = anchorX - hi - 1;
    const long long stripW = static_cast<long long>(winW) + (hi - lo) + 2;
    if (stripW > 32) return p;
    if (anchorY < 0 || anchorY + winH > static_cast<long long>(height)) return p;
    if (anchorX + winW > w) return p;
    if (stripX0 < 0 || stripX0 + stripW > w) return p;
    // stripX0 >= 0 puts the window's first column at hi + 1 >= 1 or beyond.
    p.ok = true;
    p.x0 = static_cast<size_t>(anchorX);
    p.y0 = static_cast<size_t>(anchorY);
    p.stripX0 = static_cast<size_t>(stripX0);
    p.stripWidth = static_cast<size_t>(stripW);
    return p;
}

/// @brief Pixels `[x, x + n)` of a row, aligned to bit 0, for `1 <= n <= 32` and
/// `x + n` inside the row. **INTERNAL.**
inline uint32_t stereoRowBits(const uint32_t* row, size_t x, size_t n) {
    const size_t q = x >> 5;
    const size_t s = x & 31u;
    uint32_t v = row[q] >> s;
    if (s + n > 32u) v |= row[q + 1] << (32u - s);
    return n == 32u ? v : static_cast<uint32_t>(v & ((uint32_t{1} << n) - 1u));
}

/// @brief `v`, unchanged, in a register the optimizer cannot see into. **INTERNAL.** A
/// mask held in a register lets Thumb-2 shift the other operand of the AND for free;
/// an immediate cannot, and the count's `(x >> s) & m` becomes two instructions.
inline uint32_t stereoInRegister(uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__("" : "+r"(v));
#endif
    return v;
}

/// @brief The staged window's Hamming distance at strip shift `sh`. **INTERNAL.** Each
/// row's count stops at four byte lanes, the lanes are added down the window -- 8 a
/// row, so 31 rows fit -- and collapsed once: one collapse a window where the per-row
/// path makes a whole count a row.
inline long long stereoStagedCost(const uint32_t* left, const uint32_t* strip, size_t rows,
                                  size_t sh, uint32_t winMask) {
    const uint32_t m1 = stereoInRegister(0x55555555u);
    const uint32_t m2 = stereoInRegister(0x33333333u);
    const uint32_t m4 = stereoInRegister(0x0F0F0F0Fu);
    uint32_t lanes = 0;
    for (size_t r = 0; r < rows; ++r) {
        uint32_t v = (left[r] ^ (strip[r] >> sh)) & winMask;
        v = v - ((v >> 1) & m1);
        v = (v & m2) + ((v >> 2) & m2);
        lanes += (v + (v >> 4)) & m4;
    }
    lanes = (lanes & 0x00FF00FFu) + ((lanes >> 8) & 0x00FF00FFu);
    return static_cast<long long>((lanes & 0xFFFFu) + (lanes >> 16));
}

} // namespace impl
#endif

/// @brief FINE stage: slide a window along the epipolar row around each valid
/// match's disparity, score by Hamming distance on the packed frames, and
/// refine to sub-pixel. **API TIER 3.**
/// @param left / right The RECTIFIED pair, one bit per pixel, same extent.
/// @param leftXY The left keypoints the windows anchor on.
/// @param inout One entry per left keypoint. On the way in, `valid` and
/// `disparity` are read -- from `stereoDescriptorMatch`, or filled by a
/// caller refining an initial disparity of their own. On the way out,
/// `disparity` is refined and `valid` reports whether a window could be
/// scored at all (a keypoint whose window misses the frame entirely
/// reports 0, never a clamped answer).
/// @note The scan is HORIZONTAL only: rectification promised the row, and this
/// stage holds it to that. The result is clamped to the params' disparity
/// range -- a refinement that wandered out of the range the caller stated
/// would be reporting a depth the caller already excluded.
template <typename WordType>
inline void stereoRefineDisparity(const BinMatConstView<WordType>& left,
                                  const BinMatConstView<WordType>& right,
                                  const float* leftXY, size_t count, StereoMatch* inout,
                                  const StereoMatchParams& params = StereoMatchParams()) {
    if (count == 0) return;
    BINCV_ASSERT(leftXY != nullptr && inout != nullptr,
                 "stereoRefineDisparity: null argument");
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "stereoRefineDisparity: the pair must share its extent");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "stereoRefineDisparity: the window must be more than 2 pixels on a side");
    BINCV_ASSERT(params.refineRadius >= 1, "stereoRefineDisparity: refineRadius < 1");

    // The block matcher's level type names the two frames; `prev` anchors the
    // window, `next` is read displaced -- here by MINUS the disparity, because
    // the right camera sees everything shifted left.
    BlockMatchLevel<WordType> lv;
    lv.prev = left;
    lv.next = right;

    const int winW = params.winWidth;
    const int winH = params.winHeight;
    const float halfWinX = static_cast<float>(winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(winH - 1) * 0.5f;
    const long long minD = static_cast<long long>(params.minDisparity);
    const long long maxD = static_cast<long long>(params.maxDisparity);

    for (size_t i = 0; i < count; ++i) {
        if (!inout[i].valid) continue;
        const long long anchorX = impl::floorToLL(leftXY[2 * i] - halfWinX);
        const long long anchorY = impl::floorToLL(leftXY[2 * i + 1] - halfWinY);
        const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), winW, winH);
        const impl::RegionWords<WordType> region =
            impl::clipRegion<WordType>(left.width, left.height, window);
        if (region.isEmpty) {
            inout[i].valid = 0;
            continue;
        }

        // Integer scan around the coarse disparity, clamped to the stated range.
        const long long d0 =
            impl::floorToLL(static_cast<float>(inout[i].disparity) + 0.5f);
        long long lo = d0 - params.refineRadius, hi = d0 + params.refineRadius;
        if (lo < minD) lo = minD;
        if (hi > maxD) hi = maxD;
        if (lo > hi) {
            inout[i].valid = 0;
            continue;
        }

#if BINCV_STEREO_WINDOW_COUNT
        // STAGED, where a population count is software: the left window and a right
        // strip covering every disparity scored, one word a row each, read once; a
        // score is then a shift, an XOR and a count carried down the window. The same
        // integers as the per-row path, which re-extracts both rows and counts each
        // word whole for every disparity.
        if constexpr (std::is_same<WordType, uint32_t>::value) {
            if (impl::stereoWindowCountEnabled()) {
                const impl::StereoStagedPlan plan = impl::stereoStagedPlan<WordType>(
                    left.width, left.height, anchorX, anchorY, winW, winH, lo, hi);
                if (plan.ok) {
                    const size_t rows = static_cast<size_t>(winH);
                    const size_t w = static_cast<size_t>(winW);
                    uint32_t l[impl::kStereoStagedMaxRows];
                    uint32_t strip[impl::kStereoStagedMaxRows];
                    for (size_t r = 0; r < rows; ++r) {
                        l[r] = impl::stereoRowBits(left.row(plan.y0 + r), plan.x0, w);
                        strip[r] = impl::stereoRowBits(right.row(plan.y0 + r), plan.stripX0,
                                                       plan.stripWidth);
                    }
                    const uint32_t winMask = static_cast<uint32_t>((uint32_t{1} << w) - 1u);
                    inout[i].disparity = static_cast<float>(impl::stereoScanAndFit(
                        lo, hi, minD, maxD, params.subPixel, [&](long long d) {
                            return impl::stereoStagedCost(l, strip, rows,
                                                          static_cast<size_t>(hi + 1 - d),
                                                          winMask);
                        }));
                    continue;
                }
            }
        }
#endif
        long long bestCost = -1, bestD = lo;
        for (long long d = lo; d <= hi; ++d) {
            const long long cost = impl::hammingAt<WordType>(lv, region, -d, 0);
            if (bestCost < 0 || cost < bestCost) {
                bestCost = cost;
                bestD = d;
            }
        }

        double sub = 0.0;
        if (params.subPixel) {
            // Neighbours may sit outside [lo, hi]; the fit wants the true local
            // shape, and one extra score per side is cheaper than a biased vertex.
            const long long cm = impl::hammingAt<WordType>(lv, region, -(bestD - 1), 0);
            const long long cp = impl::hammingAt<WordType>(lv, region, -(bestD + 1), 0);
            sub = impl::parabolicOffset(cm, bestCost, cp);
        }
        double refined = static_cast<double>(bestD) + sub;
        if (refined < static_cast<double>(minD)) refined = static_cast<double>(minD);
        if (refined > static_cast<double>(maxD)) refined = static_cast<double>(maxD);
        inout[i].disparity = static_cast<float>(refined);
    }
}

/// @brief Both stages: descriptor search, then window refinement. **API TIER 3.**
/// @note Exactly `stereoDescriptorMatch` into `stereoRefineDisparity`; it exists
/// so the common caller cannot get the hand-off wrong, not because the
/// composition adds anything.
template <typename WordType>
inline void stereoMatchRectified(const BinMatConstView<WordType>& left,
                                 const BinMatConstView<WordType>& right,
                                 const float* leftXY, size_t leftCount,
                                 const WordType* leftDesc, const float* rightXY,
                                 size_t rightCount, const WordType* rightDesc, size_t words,
                                 StereoMatch* out,
                                 const StereoMatchParams& params = StereoMatchParams()) {
    stereoDescriptorMatch<WordType>(leftXY, leftCount, leftDesc, rightXY, rightCount,
                                    rightDesc, words, out, params);
    stereoRefineDisparity<WordType>(left, right, leftXY, leftCount, out, params);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
