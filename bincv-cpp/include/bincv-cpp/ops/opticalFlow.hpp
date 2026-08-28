#pragma once

/// @file opticalFlow.hpp
/// @brief Pyramidal Lucas-Kanade keypoint tracking over bit-packed frames (T3.8).
///        **API TIER 2** -- `cv::calcOpticalFlowPyrLK`'s role and call shape with
///        deliberately different numerics. NOT bit-exact against OpenCV, and no
///        such promise is available even in principle (see THE BOUNDARY below).
///
/// This is [ARCHITECTURE 7.9](../../../ARCHITECTURE.md)'s **known hard problem**,
/// and route **(b)** of the two it names. Lucas-Kanade warps its window to
/// subpixel positions and bilinearly interpolates; that is continuous and does not
/// bit-parallelize. Route (a) -- census/Hamming block matching at integer pixels,
/// fully bit-parallel -- is a different algorithm and is E-6 / T4.2, not this
/// file. Route (b) keeps LK's numerics where LK's accuracy comes from and puts
/// everything else on population counts.
///
/// ===========================================================================
/// THE BOUNDARY. WHICH SIDE OF THE BIT-PARALLEL / FLOATING-POINT LINE EACH PIECE
/// FALLS ON -- THIS IS THE ARCHITECTURAL CLAIM THE FILE EXISTS TO TEST
/// ===========================================================================
///
/// **BIT-PARALLEL, EXACT INTEGERS, NO PER-PIXEL FLOAT AT ALL:**
///
///  1. **Window extraction.** The 31x31 window is a `Rect` clipped against the
///     level through `impl::clipRegion` and walked by `impl::visitRowWords` --
///     the same two helpers every reduction in ops/reduce.hpp uses. No patch is
///     ever copied out. The reference copies the warped patch into `IWinBuf` and
///     `derivIWinBuf` (`winSize.area() * 3` shorts = 5766 B at 31x31, per
///     invoker); this file copies nothing and needs no scratch buffer.
///  2. **The 2x2 normal-equations matrix `A`.** One call to T3.6's
///     `gradientCovariance` -- four masked popcounts in one traversal, no scratch
///     (ARCHITECTURE 7.5, D-15). Its three entries are EXACT INTEGERS, which is
///     what lets the singularity test below be exact rather than epsilon-guarded.
///  3. **THE RESIDUAL `b = sum(diff * grad)`, WHICH IS ALSO POPCOUNTS.** This is
///     the one place this file goes further than [ARCHITECTURE 7.9] promised, and
///     the reason is an identity rather than an optimization. Bilinear
///     interpolation is LINEAR in the four taps, and the taps of a BINARY next
///     frame are bits:
///
///         diff(z) = w00*T00(z) + w01*T01(z) + w10*T10(z) + w11*T11(z) - I(z)
///
///     with `w..` the four bilinear weights -- CONSTANT over the whole window,
///     because every pixel of the window is displaced by the same vector -- and
///     `T00..T11` the next frame's four tap planes, each one bit per pixel. The
///     gradient is ternary, so `sum(M * Ix)` over any bit-plane `M` is a SIGNED
///     MASKED POPCOUNT, `popcount(magX & M) - 2*popcount(magX & M & signX)`.
///     Therefore
///
///         b1 = w00*S(T00) + w01*S(T01) + w10*S(T10) + w11*S(T11) - S(I)
///
///     with `S(M) = sum over the window of M * Ix`, five exact integers. The
///     bilinear interpolation is *performed*, at full precision, by five integer
///     counts and four floating-point multiplies **per window per iteration** --
///     not per pixel. Nothing is approximated and nothing is rounded before the
///     four weights are applied; the reference, by contrast, rounds each
///     interpolated sample to 14-bit fixed point (`CV_DESCALE`) at every pixel.
///  4. **The tracking error `err`.** Same identity: `|diff|` collapses because
///     `I` is a bit, so `|Jinterp - I| = I + (1 - 2I)*Jinterp`, and the window sum
///     is again five popcounts per tap plane. Bit-parallel, exact.
///  5. **The four tap planes themselves.** `T01` is `T00` shifted one pixel and
///     `T1x` is one row down, so all four are word-aligned reads of the SAME two
///     next-frame rows with a cross-word bit shift -- `impl::ReplicatedShiftedRow`
///     below. No gather, no per-pixel address arithmetic.
///
/// **FLOATING POINT, ONCE PER WINDOW PER ITERATION -- NEVER PER PIXEL:**
///
///  6. The subpixel position itself, and the split of the flow into an integer
///     tap offset and a fraction `(a, b)` in `[0, 1)`.
///  7. The four bilinear weights `w00 = (1-a)(1-b)` ... `w11 = ab`.
///  8. The 2x2 solve `delta = -A^-1 b`, its determinant, and the minimum
///     eigenvalue used to reject a degenerate window.
///  9. The iteration: `nextPt += delta`, the epsilon test, the oscillation test.
///
/// So the per-pixel cost of this tracker is **integers and popcounts only**, and
/// the floating point is O(iterations), not O(iterations * window area). That is a
/// STRONGER result than [ARCHITECTURE 7.9] specified route (b) with -- it named
/// only the window and the covariance -- and it is stated here rather than buried
/// because the boundary IS the claim. What could not be moved across the line is
/// items 6-9: the position, the weights and the solve are irreducibly continuous,
/// exactly as 7.9 says. D-20 records the placement.
///
/// ===========================================================================
/// WHAT IS DELIBERATELY DIFFERENT FROM THE REFERENCE, AND WHY
/// ===========================================================================
/// Reference: `SEAL/src/keypoint_tracking/SparsePyrLKOpticalFlowSealImpl.cpp` and
/// `SEAL/opencv_internal/src/LKTrackerInvoker.cpp`, with
/// `SEAL/seal_params.yaml`'s parameters (win 31x31, maxLevel 3, maxCount 20,
/// eps 0.03, minEig 0.001, BINARIZED derivative, BOX_2x2 pyrDown) -- all of which
/// are this file's defaults, verbatim.
///
/// **(i) THE PREVIOUS WINDOW SITS ON THE INTEGER GRID.** The reference warps BOTH
/// windows: it bilinearly interpolates the previous frame and its derivative at
/// the fractional `prevPt`, then interpolates the next frame at `nextPt`. A
/// bit-plane derivative cannot be interpolated -- the popcount identity of
/// ARCHITECTURE 7.5 is exact only for values in {-1, 0, +1} -- so the previous
/// window is anchored at `floor(prevPt - halfWin)` and the ENTIRE subpixel
/// displacement is carried on the next-frame side. The estimated flow is
/// unaffected: the residual still compares `I(z)` with `J(z + d)` for `d` the
/// full-precision float flow, so `d` is what is being solved for either way. What
/// moves is only WHICH pixels the aperture covers -- by at most half a pixel.
/// **This is the concrete thing route (b) trades away**, and it is the reason
/// this operation is Tier 2 rather than Tier 1.
///
/// **(ii) THE WINDOW IS CLIPPED, NOT PADDED.** `buildOpticalFlowPyramid` allocates
/// every pyramid level with a `winSize`-wide border on all four sides and fills it
/// with BORDER_REFLECT_101 -- at 640x480 and a 31x31 window that is a 702x542
/// buffer for a 640x480 level, **1.24x the level's own footprint, at every
/// level**. binCV declines it: the window is intersected with the level exactly as
/// D-13 and ops/reduce.hpp intersect every other region, so `A` and `b` are
/// accumulated over the same clipped pixel set and the solve stays consistent.
/// A window wholly outside gives `A = {0,0,0}`, a zero determinant, and a lost
/// point -- a value, not an error.
///
/// **(iii) NEXT-FRAME TAPS OUTSIDE THE LEVEL REPLICATE.** A tap can fall outside
/// even when the window pixel is inside, because the flow displaces it. Those bits
/// read as the nearest edge pixel (BORDER_REPLICATE). Reflect-101 -- the
/// reference's choice -- is a per-pixel index mapping and is not word-parallel;
/// replicate is two mask-selects per word. Replicate also cannot manufacture a
/// gradient outside the frame, which is the property D-19 chose reflect-101 for
/// on the derivative.
///
/// **(iv) THE SINGULARITY TEST IS EXACT.** The reference guards `D < FLT_EPSILON`
/// because its `D` is a float product of float box-filtered float Sobel outputs.
/// Here `det = xx*yy - xy*xy` is a difference of products of exact popcounts: it
/// is 0 or at least 1, and nothing lies between. The test is `det <= 0`, with no
/// epsilon, for the same reason ops/corner.hpp's `> threshold` needs none.
///
/// **(v) ONE BIT PER PYRAMID LEVEL.** Every level here is binary, because the
/// popcount covariance is exact only for a TERNARY derivative and a ternary
/// derivative is what a ONE-bit level produces (ops/covariance.hpp promise 1). An
/// N-bit level needs the bit-sliced weighted-sum covariance that ARCHITECTURE 7.5
/// describes and nothing in binCV implements. **How many bits each level actually
/// needs is E-7 / T4.1, which depends on this task and is where that choice gets
/// made and measured** -- it is not settled here, and 1/1/1/1 is what the shipped
/// kernels can express, not a claim that it is right.
///
/// **(vi) A LEVEL NO LARGER THAN THE WINDOW IS IGNORED, NOT REFUSED.** The
/// reference stops BUILDING levels at the first one that is not strictly larger
/// than `winSize` (`SparsePyrLKOpticalFlowSealImpl.cpp`'s `if (sz.width <=
/// winSize.width || sz.height <= winSize.height)`) and truncates `maxLevel`. Here
/// the caller owns the pyramid (D-5), so the same rule is applied at USE: levels
/// are consumed as a prefix and one at or below the window size ends it. This is
/// a deviation only in WHERE the rule lives. It is load-bearing for binCV in a way
/// it is not for the reference: on such a level every window covers nearly the
/// whole level, so every point gets nearly the same `A` and `b` -- and because
/// binCV clips instead of padding, there is no border to make the windows differ.
///
/// **(vii) THE FINAL-POSITION RANGE TEST IS UNCONDITIONAL.** The reference
/// re-tests the returned position and can set `status` false there, but only
/// inside `if (status && err && level == 0 ...)` -- so its `status` depends on
/// whether the caller asked for an error value. That is not reproduced: the test
/// runs whether or not `err` is null, because `status` describes the position the
/// caller is handed.
///
/// ===========================================================================
/// THE ITERATION AND THE TERMINATION RULE, TAKEN FROM THE REFERENCE
/// ===========================================================================
/// `LKTrackerInvoker::operator()` decides these, and all five are reproduced:
///
///   * **Coarse to fine, propagating by doubling.** At the coarsest level the
///     estimate starts at the point itself (`lk_use_initial_flow: 0`); at every
///     finer level it starts at twice the level above's answer. The propagation
///     happens for EVERY point before ANY point is tracked, and it happens even
///     for points that were skipped at the level above -- so a point that leaves
///     the frame at level 2 still arrives at level 1 with a doubled estimate.
///   * **A point is LOST in exactly three ways**, and status is written **only at
///     level 0** -- at coarser levels the point is skipped for that level and
///     tried again at the next one:
///       1. the previous window's origin is out of range
///          (`ip.x < -winWidth || ip.x >= width`, and likewise in y);
///       2. the window is degenerate -- `minEig < minEigThreshold` or a singular
///          `det`;
///       3. the current estimate's origin goes out of range -- tested at the top
///          of every iteration, and once more on the position finally returned
///          (deviation (vii)).
///   * **Up to `maxIterations` iterations**, clamped to [0, 100] as the reference
///     clamps `criteria.maxCount`.
///   * **Converged when `|delta|^2 <= epsilon^2`.** The reference squares
///     `criteria.epsilon` once up front and compares `delta.ddot(delta)`; 0.03 px
///     is `seal_params.yaml`'s value and the default here.
///   * **The oscillation rule, which is NOT the epsilon rule.** From iteration 1
///     on, if `|delta + prevDelta| < 0.01` in both components -- i.e. this step
///     almost exactly undoes the last one -- the point is declared converged, the
///     estimate is backed off by HALF the last step, and the loop breaks. Without
///     it a point that lands between two pixels ping-pongs for all 20 iterations
///     and finishes on whichever side the last step took it to.
///
/// ===========================================================================
/// UNITS: WHY THERE IS A FACTOR OF 2 AND A CONSTANT NAMED FOR THE REFERENCE
/// ===========================================================================
/// binCV's pixels are {0, 1} and its binarized derivative is the raw `[-1, 0, 1]`
/// tap, so `Ix` is `I(x+1) - I(x-1)`, which is TWICE the central difference
/// `dI/dx`. The reference reaches the same place by a different route -- pixels in
/// {0, 255}, derivative scaled by 16 (`calcBinarizedDeriv`), intensity descaled by
/// `W_BITS1-5` -- whose net effect is a gradient of `Ix/2` per unit intensity,
/// identical. Substituting `g = Ix/2` into `delta = -A^-1 b` scales `A` by 1/4 and
/// `b` by 1/2, so the step computed from the raw taps must be multiplied by
/// **exactly 2** (`impl::kCentralDifferenceScale`). Dropping it would not diverge
/// -- it is a damped Gauss-Newton step -- it would halve every step and stop early
/// on the epsilon test, which is the kind of bug that looks like "slightly worse
/// accuracy".
///
/// `minEigThreshold` is quoted in the REFERENCE's units so that
/// `seal_params.yaml`'s 0.001 can be used verbatim.
/// `impl::kReferenceMinEigScale` is the conversion, and it is a derivation and not
/// a fitted constant: the reference's `A11` is `FLT_SCALE * sum(ixval^2)` with
/// `ixval = 16*255*Ix`, i.e. `(16*255)^2 / 2^20 = 65025/4096 = 15.875244...` times
/// binCV's integer `sumXX`; and its `minEig` divides the eigenvalue SUM by
/// `2*w*h`, which is this eigenvalue divided by `w*h`.
///
/// ===========================================================================
/// WHAT THIS FILE DOES NOT CLAIM
/// ===========================================================================
/// **Tracking across real image sequences is T4.3a, not this task.** ADVIO ships
/// `.mov`; frame extraction and a comparison harness against the reference
/// frontend are pipeline validation, on the far side of ARCHITECTURE 1's
/// boundary. What is claimed here is measured against SYNTHETIC WARPS with
/// EXACTLY KNOWN ground-truth displacement -- see tests/test_opticalflow.cpp for
/// the tolerance, which is stated with its derivation BEFORE any error is
/// measured.
///
/// ===========================================================================
/// CONTRACTS
/// ===========================================================================
///  * **Views, never containers** (D-5). Every per-level argument is a
///    `BinMatConstView`; `lkLevel()` is the one convenience that names a
///    container's planes into them.
///  * **No heap, anywhere.** The keypoint arrays, the status array and the error
///    array are the caller's. There is no scratch buffer of any kind -- not one
///    byte -- because item 1 above never copies a patch out.
///  * **Never throws** (ARCHITECTURE 5.3). Inconsistent arguments are programming
///    errors reported by BINCV_ASSERT in debug and undefined in release. A track
///    that fails reports `status[i] == 0`; that is a value, not an error.
///  * **Read-only inputs**, so aliasing between any two inputs is unrestricted.
///    The OUT arrays are a different matter: `nextPts` must not overlap `prevPts`.
///    The level loop writes every `nextPts` entry before it reads any `prevPts`
///    entry, so an in-place call tracks from the wrong anchor -- a BINCV_ASSERT,
///    in the shape of D-11, rather than a documented hazard.
///  * `nextPts` is written for every point at every level and holds LEVEL
///    coordinates while the level loop runs; on return it holds level-0
///    coordinates. This is the reference's own use of the same array.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#ifdef BINCV_LK_STAGE_TIMING
#include <chrono>
#endif

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "../core/error.hpp"
#include "../core/parallel.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../quantMat.hpp"
#include "../impl/kernel_util.hpp"
#include "../impl/lkBatch_impl.hpp"
// gradientCovariance: the 2x2 matrix, one fused traversal, zero scratch (T3.6).
#include "covariance.hpp"
// impl::minEigenValue -- the SAME (S - sqrt(D))/2 from exact integer operands
// that ops/corner.hpp selects on. One definition, two operations: the corner
// detector's "is this trackable" and the tracker's "is this still trackable" are
// the same question and must not be able to answer it differently.
#include "corner.hpp"
// impl::clipRegion / visitRowWords / popcountWord / RegionWords -- the region
// walk every reduction in the library shares (D-13's clipping and padding
// contract lives there, not here).
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief A subpixel point. **API TIER 2**: `cv::Point2f`'s role and field names,
///        because this operation's whole job is to produce fractional positions.
/// @note Deliberately NOT reused for ops/corner.hpp's `Corner`, which is integer
///       by construction and carries a response.
struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

/// @brief The tracker's parameters, defaulted to `SEAL/seal_params.yaml` verbatim.
/// @note `lk_win_size_width/height: 31`, `lk_term_criteria_max_count: 20`,
///       `lk_term_criteria_eps: 0.03`, `lk_min_eig_threshold: 0.001`.
///       `lk_max_level: 3` is not a field: the number of levels is `levelCount`,
///       because the caller owns the pyramid (D-5).
///       `lk_use_initial_flow: 0` is not a field either -- there is no initial-flow
///       mode, since an unused mode is an untested one.
/// @brief Which pyramid level a keypoint ENTERS at. **X-25 / E-14.**
/// @note Not a border and not a padding scheme: it is a policy about which levels
///       a given point is allowed to be tracked on at all. binCV clips where the
///       reference pads (deviation (ii)), and a window that is mostly outside a
///       coarse level yields an ill-conditioned `A` and a one-sided `b` whose
///       error is then multiplied by `2^level` on the way down. X-24 measured that
///       cost: `1/2/2/2` is 0.8356 px over all 141 real-frame keypoints and
///       **0.0010 px over the 58 that never clip**.
enum class LKEntryLevel {
    /// Every point enters at the coarsest usable level. **What ships**, and the
    /// reference's behaviour given a padded pyramid it always has enough of.
    Coarsest,
    /// A point enters at the coarsest level whose window lies **fully inside that
    /// level**, and is not tracked above it. Costs **no memory and no keypoints**:
    /// a point near the edge gets a shallower pyramid rather than a padded one or
    /// a rejection. Level 0 is always used, whatever it clips -- it is the frame.
    DeepestFitting,
};

struct LKParams {
    int winWidth = 31;     ///< window width in pixels, > 2
    int winHeight = 31;    ///< window height in pixels, > 2
    int maxIterations = 20;      ///< per level, clamped to [0, 100] as the reference clamps
    float epsilon = 0.03f;       ///< converged when |delta| <= this, in pixels
    float minEigThreshold = 0.001f;  ///< IN THE REFERENCE'S UNITS; see UNITS above

    /// Which level each keypoint enters the pyramid at (X-25 / E-14). Defaults to
    /// the shipped behaviour; nothing changes unless a caller asks for it.
    LKEntryLevel entryLevel = LKEntryLevel::Coarsest;
};

/// @brief One pyramid level's six planes: both frames, and the previous frame's
///        ternary derivative.
/// @note Views, not containers (D-5). All six must share the level's dimensions.
/// @note The derivative belongs to the PREVIOUS frame only. LK linearizes about
///       the previous frame, so the next frame's derivative is never formed --
///       which halves the derivative footprint against a naive reading of the
///       algorithm.
template <typename WordType>
struct LKLevel {
    BinMatConstView<WordType> prev;    ///< previous frame at this level, 1 bit/pixel
    BinMatConstView<WordType> next;    ///< next frame at this level, 1 bit/pixel
    BinMatConstView<WordType> dxMag;   ///< `dx.constMagnitude(0)`
    BinMatConstView<WordType> dxSign;  ///< `dx.constSign()`; a set bit is NEGATIVE (D-3)
    BinMatConstView<WordType> dyMag;   ///< `dy.constMagnitude(0)`
    BinMatConstView<WordType> dySign;  ///< `dy.constSign()`

    /// Bits per pixel at this level -- one, by construction. `LKLevelN<N>` below
    /// is the same level at N, and the tracking body is written against this
    /// surface so that ONE body serves both (D-21: genericity in N is not paid
    /// for at N == 1).
    static constexpr size_t Bits = 1;

    /// The storage word type, so a body templated on the LEVEL can still name it.
    using Word = WordType;

    size_t width() const { return prev.width; }
    size_t height() const { return prev.height; }
};

/// @brief Names a level's containers into an LKLevel. **API TIER 2.**
/// @note The one container-shaped entry point in this file, in the shape
///       ops/derivative.hpp and ops/covariance.hpp use: the container knows which
///       plane is magnitude and which is sign; the kernel does not (D-5).
template <typename WordType>
inline LKLevel<WordType> lkLevel(const BinMat<WordType>& prev, const BinMat<WordType>& next,
                                 const TernaryMat<WordType>& dx, const TernaryMat<WordType>& dy) {
    LKLevel<WordType> lv;
    lv.prev = prev.constView();
    lv.next = next.constView();
    lv.dxMag = dx.constMagnitude(0);
    lv.dxSign = dx.constSign();
    lv.dyMag = dy.constMagnitude(0);
    lv.dySign = dy.constSign();
    return lv;
}

/// @brief One pyramid level at **N bits per pixel**: both frames' bit-planes, and
///        the previous frame's N-bit signed derivative. The generic-N form of
///        `LKLevel`, which is this at `N == 1`.
/// @note Views, not containers (D-5). All `4N + 2` planes must share the level's
///       dimensions.
/// @note `prev[i]` / `next[i]` are bit-plane `i`, **plane 0 being the LEAST
///       significant bit** -- QuantMat's own convention, unchanged. `dxMag[j]` is
///       magnitude plane `j` and `dxSign` is the shared sign plane, a set bit
///       meaning NEGATIVE (D-3).
/// @note The derivative of an N-bit level is N-bit: `I(x+1) - I(x-1)` over an
///       alphabet of `2^N` values lands in `[-(2^N - 1), 2^N - 1]`, which is
///       exactly `SignedQuantMat<N, WordType>`'s range. So one `N` describes the
///       whole level and there is no second depth parameter.
template <size_t N, typename WordType>
struct LKLevelN {
    static_assert(N >= 1 && N <= 7, "LKLevelN: N outside SignedQuantMat's supported range");

    /// Bits per pixel at this level. The tracker reads it to scale `minEigThreshold`.
    static constexpr size_t Bits = N;

    /// The storage word type, so a body templated on the LEVEL can still name it.
    using Word = WordType;

    BinMatConstView<WordType> prev[N];    ///< previous frame's planes, LSB first
    BinMatConstView<WordType> next[N];    ///< next frame's planes, LSB first
    BinMatConstView<WordType> dxMag[N];   ///< `dx.constMagnitude(j)`
    BinMatConstView<WordType> dxSign;     ///< `dx.constSign()`; set bit is NEGATIVE (D-3)
    BinMatConstView<WordType> dyMag[N];   ///< `dy.constMagnitude(j)`
    BinMatConstView<WordType> dySign;     ///< `dy.constSign()`

    size_t width() const { return prev[0].width; }
    size_t height() const { return prev[0].height; }
};

/// @brief Names an N-bit level's containers into an LKLevelN. **API TIER 2.**
/// @note The generic-N counterpart of the `lkLevel` above, in the same shape: the
///       container knows which plane is which, the kernel does not (D-5).
template <size_t N, typename WordType>
inline LKLevelN<N, WordType> lkLevel(const QuantMat<N, WordType>& prev,
                                     const QuantMat<N, WordType>& next,
                                     const SignedQuantMat<N, WordType>& dx,
                                     const SignedQuantMat<N, WordType>& dy) {
    LKLevelN<N, WordType> lv;
    for (size_t i = 0; i < N; ++i) {
        lv.prev[i] = prev.constPlane(i);
        lv.next[i] = next.constPlane(i);
        lv.dxMag[i] = dx.constMagnitude(i);
        lv.dyMag[i] = dy.constMagnitude(i);
    }
    lv.dxSign = dx.constSign();
    lv.dySign = dy.constSign();
    return lv;
}

namespace impl {

#ifdef BINCV_LK_STAGE_TIMING
/// @brief Where `track`'s time actually goes, by stage. **DIAGNOSTIC ONLY — off by
///        default and compiled out otherwise.** (X-83.)
///
/// **THIS EXISTS BECAUSE THREE GUESSES IN A ROW MISSED.** An iteration-cap sweep on the
/// reference device put roughly **45% of `track` OUTSIDE the iteration loop** —
/// staging, the covariance, the clip — and nothing in this project had ever measured
/// which of those it is. Guessing produced a 1.9% win and a 0.0% win before this was
/// written; [X-67](../../../EXPERIMENTS.md)/[D-59](../../../ARCHITECTURE.md) is the
/// same lesson from the frontend's side.
///
/// Nanoseconds, accumulated over every point and level. Four clock reads against a
/// ~2 300 ns point-level is a few percent, and it is compared only against itself.
struct StageTiming {
    unsigned long long setup = 0;        ///< propagation, bounds, `clipRegion`
    unsigned long long staging = 0;      ///< `stageWindow`
    unsigned long long covariance = 0;   ///< `levelCovariance` and the eigen test
    unsigned long long residual = 0;     ///< the iteration loop: taps and `residualSums`
    unsigned long long points = 0;
    unsigned long long tapRows = 0;      ///< window rows whose taps were EXTRACTED
    unsigned long long iterations = 0;   ///< `residualSums` calls
};
inline StageTiming& stageTiming() {
    static StageTiming t;
    return t;
}
inline unsigned long long stageNow() {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}
#define BINCV_STAGE_MARK() unsigned long long bincvStageMark = impl::stageNow()
#define BINCV_STAGE_LAP(field)                                  \
    do {                                                        \
        const unsigned long long bincvNow = impl::stageNow();   \
        impl::stageTiming().field += bincvNow - bincvStageMark; \
        bincvStageMark = bincvNow;                              \
    } while (0)
#define BINCV_STAGE_POINT() ++impl::stageTiming().points
#else
#define BINCV_STAGE_MARK() ((void)0)
#define BINCV_STAGE_LAP(field) ((void)0)
#define BINCV_STAGE_POINT() ((void)0)
#endif


/// @brief The factor the raw `[-1, 0, 1]` tap needs to become a central
///        difference. See UNITS at the top of the file -- this is a derivation,
///        not a tuning knob.
constexpr double kCentralDifferenceScale = 2.0;

/// @brief binCV's integer minimum eigenvalue, in the units the reference's
///        `minEigThreshold` is quoted in: `(16*255)^2 / 2^20`.
/// @note Written as the quotient of the two integers it comes from, so that a
///       reader can check the derivation rather than recognise 15.875244140625.
constexpr double kReferenceMinEigScale = (16.0 * 255.0) * (16.0 * 255.0) / 1048576.0;

/// @brief `kReferenceMinEigScale` at an arbitrary bit depth.
/// @param bits Bits per pixel at the level, `>= 1`.
///
/// **THIS IS WHAT KEEPS `minEigThreshold` MEANING THE SAME THING AT EVERY N, AND
/// WITHOUT IT E-7's ACCURACY CURVE WOULD COMPARE DIFFERENT POINT SETS.** The
/// derivation above fixes binCV's full-scale intensity at 1 because a bit is the
/// whole alphabet; at N bits the alphabet runs `[0, 2^N - 1]` and the SAME
/// physical intensity is now `2^N - 1` LSBs rather than 1. The reference's
/// `ixval = 16 * 255 * Ix` is quoted per unit of ITS full scale, so the conversion
/// is `16 * 255 / (2^N - 1)` per binCV LSB, squared because `A11` is a sum of
/// squares, and divided by `2^20` exactly as before.
///
/// Getting this wrong is not a small error and it is not a visible one: an N-bit
/// level's `sumXX` is roughly `(2^N - 1)^2` times a 1-bit level's for the same
/// image content, so holding the 1-bit constant would raise the effective
/// rejection threshold by that factor -- 225x at N = 4 -- and REJECT MORE POINTS
/// AT HIGHER N. The accuracy curve would then be measured over a different (and
/// progressively better-conditioned) subset at every depth, which would make deeper
/// levels look better for a reason that has nothing to do with their depth.
constexpr double referenceMinEigScale(size_t bits) {
    return (16.0 * 255.0 / static_cast<double>((size_t{1} << bits) - 1)) *
           (16.0 * 255.0 / static_cast<double>((size_t{1} << bits) - 1)) / 1048576.0;
}

// The generic form must reproduce the derived 1-bit constant exactly, not nearly:
// both are the same expression with the same operands at bits == 1.
static_assert(referenceMinEigScale(1) == kReferenceMinEigScale,
              "referenceMinEigScale(1) must be kReferenceMinEigScale");

/// @brief `floor(a / b)` for integers with `b > 0`, rounding toward MINUS
///        infinity.
/// @note C++ integer division truncates toward zero, which for a negative
///       numerator is one word off in the tap extraction below -- and off by one
///       WORD, i.e. up to 64 pixels, not off by one pixel. Spelled out rather than
///       written inline for that reason.
inline long long floorDiv(long long a, long long b) {
    const long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/// @brief One row of a binary plane, readable at an arbitrary PIXEL offset, with
///        columns outside `[0, width)` reading as the nearest edge pixel.
///        **INTERNAL.**
///
/// @note This is the whole of the "bilinear tap" machinery, and it is a bit shift.
///       `word(i)` returns the `WordBits` bits that sit under word `i` of the
///       window's own coordinate grid after displacement by `off` pixels, so a
///       caller ANDs it straight against a derivative mask word (see the residual
///       below) with no per-pixel address arithmetic anywhere.
/// @note **The replicate fill is why this is not simply a shift.** `leftFill` and
///       `rightFill` are all-ones or all-zeros words holding the value of pixel 0
///       and pixel `width - 1`; the two mask-selects at the end substitute them
///       for the bits whose source column fell outside. Deviation (iii) at the top
///       of the file is what that implements.
/// @note Bits past `width` in the last word are cleared by `tailMask` before any
///       shift, so a source plane with dirty padding gives the clean plane's
///       answer -- D-13's rule, applied one level down from the reductions that
///       state it.
template <typename WordType>
struct ReplicatedShiftedRow {
    const WordType* row = nullptr;  ///< the source row, or null for an empty plane
    size_t words = 0;               ///< words that hold pixels: minRowWords(width)
    size_t width = 0;               ///< the plane's width in pixels
    WordType tailMask = 0;          ///< valid bits of word `words - 1`
    long long off = 0;              ///< displacement in pixels; may be negative
    WordType leftFill = 0;          ///< all ones iff pixel 0 is set
    WordType rightFill = 0;         ///< all ones iff pixel width-1 is set

    /// @brief The source word `k`, with the trailing partial word masked and any
    ///        index outside the row reading as zero (the replicate fill covers it).
    WordType sourceWord(long long k) const {
        if (k < 0 || static_cast<unsigned long long>(k) >= words) return 0;
        const size_t ku = static_cast<size_t>(k);
        const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
        return static_cast<WordType>(row[ku] & ((ku + 1 == words) ? tailMask : allOnes));
    }

    /// @brief Bits of the displaced row lying under word `i` of the window grid.
    WordType word(size_t i) const {
        constexpr long long bits = static_cast<long long>(bitsPerWord<WordType>());
        const long long start = static_cast<long long>(i) * bits + off;
        const long long q = floorDiv(start, bits);
        const size_t s = static_cast<size_t>(start - q * bits);

        WordType raw = sourceWord(q);
        if (s != 0) {
            const WordType hi = sourceWord(q + 1);
            raw = static_cast<WordType>(static_cast<WordType>(raw >> s) |
                                        static_cast<WordType>(hi << (bitsPerWord<WordType>() - s)));
        }

        // Bits whose source column was < 0, and bits whose source column was
        // >= width. Both counts are clamped to a whole word, which is what makes a
        // window displaced entirely off the plane read as a solid edge value
        // rather than as an out-of-range shift.
        long long lowOutside = -start;
        if (lowOutside < 0) lowOutside = 0;
        if (lowOutside > bits) lowOutside = bits;
        long long highOutside = start + bits - static_cast<long long>(width);
        if (highOutside < 0) highOutside = 0;
        if (highOutside > bits) highOutside = bits;

        if (lowOutside > 0) {
            const WordType m = lowBitsMask<WordType>(static_cast<size_t>(lowOutside));
            raw = static_cast<WordType>(static_cast<WordType>(raw & static_cast<WordType>(~m)) |
                                        static_cast<WordType>(leftFill & m));
        }
        if (highOutside > 0) {
            const WordType m = static_cast<WordType>(
                ~lowBitsMask<WordType>(static_cast<size_t>(bits - highOutside)));
            raw = static_cast<WordType>(static_cast<WordType>(raw & static_cast<WordType>(~m)) |
                                        static_cast<WordType>(rightFill & m));
        }
        return raw;
    }
};

/// @brief Reads one pixel of a plane, for the two edge values a replicate fill
///        needs. **INTERNAL**, and called twice per row, never per pixel.
template <typename WordType>
inline WordType edgeFill(const WordType* row, size_t column) {
    const bool set = (row[wordIndex<WordType>(column)] & bitMask<WordType>(column)) != 0;
    return set ? static_cast<WordType>(~static_cast<WordType>(0)) : static_cast<WordType>(0);
}

/// @brief Builds a displaced reader for row `y` of `plane`, clamped vertically.
/// @note The vertical half of the replicate border is this clamp; the horizontal
///       half is inside ReplicatedShiftedRow. Two axes, two mechanisms, because a
///       row displacement moves no bits (T2.4) and a column displacement moves all
///       of them.
template <typename WordType>
inline ReplicatedShiftedRow<WordType> displacedRow(const BinMatConstView<WordType>& plane,
                                                   long long y, long long off) {
    ReplicatedShiftedRow<WordType> r;
    if (plane.height == 0 || plane.width == 0) return r;
    long long clamped = y;
    if (clamped < 0) clamped = 0;
    const long long lastRow = static_cast<long long>(plane.height) - 1;
    if (clamped > lastRow) clamped = lastRow;

    r.row = plane.row(static_cast<size_t>(clamped));
    r.words = minRowWords<WordType>(plane.width);
    r.width = plane.width;
    r.tailMask = rowTailMask<WordType>(plane.width);
    r.off = off;
    r.leftFill = edgeFill<WordType>(r.row, 0);
    r.rightFill = edgeFill<WordType>(r.row, plane.width - 1);
    return r;
}

/// @brief `sum over the window of M(z) * Ix(z)` for a ternary derivative, from two
///        popcounts. **INTERNAL.**
/// @return `popcount(mag & M) - 2 * popcount(mag & M & sign)`, i.e. the count of
///         agreeing pixels minus the count of opposing ones.
/// @note The `total - 2*set` spelling rather than
///       `popcount(mag & M & ~sign) - popcount(mag & M & sign)`: one popcount
///       cheaper on a target where the popcount is the expensive operation (D-6),
///       and it never forms `~sign`, which would set every padding bit of a
///       trailing word (D-13). The same argument as `impl::splitRowRegion`'s.
template <typename WordType>
inline long long signedMaskedSum(WordType mag, WordType sign, WordType m) {
    const WordType both = static_cast<WordType>(mag & m);
    const long long total = static_cast<long long>(popcountWord<WordType>(both));
    const long long opposing =
        static_cast<long long>(popcountWord<WordType>(static_cast<WordType>(both & sign)));
    return total - 2 * opposing;
}

/// @brief `sum over the window of V(z) * G(z)` for an N-bit value `V` against an
///        N-bit SIGN-MAGNITUDE gradient `G`, from `2N^2` popcounts. **INTERNAL.**
/// @param maskedMag The gradient's N magnitude plane words, ALREADY masked to the
///        region -- masked once by the caller rather than `N^2` times here.
/// @param sign The gradient's sign plane word, unmasked: it is only ever ANDed
///        with an already-masked magnitude.
/// @param val The value's N plane words, LSB first.
///
/// `V = sum_i 2^i V_i` and `G = +/- sum_j 2^j M_j`, so
///
///     sum V*G = sum_{i,j} 2^(i+j) * [ popcount(V_i & M_j) - 2*popcount(V_i & M_j & S) ]
///
/// -- the same `total - 2*set` spelling `signedMaskedSum` uses, for the same two
/// reasons (one popcount cheaper than forming both halves, and it never forms
/// `~sign`, which would set every padding bit of a trailing word, D-13). This is
/// the generalization of `signedMaskedSum`, which it reduces to exactly at N = 1.
///
/// @note The weight is a MULTIPLY, not a shift. `(total - 2*opposing)` is signed
///       and routinely negative, and left-shifting a negative value is undefined
///       before C++20. ops/covariance.hpp's `combineBitSlicedPairs` multiplies by
///       `int64_t(1) << (i + j)` for the same reason.
/// @note One accumulator, not `N^2` of them. The per-(i,j) counts are weighted and
///       folded immediately, so the register footprint does NOT grow with N and
///       E-13's O(N^2)-per-row accumulator concern -- which is about
///       `BitSlicedPairCounts<N>` in ops/covariance.hpp -- does not arise here.
/// @note The plane operands are POINTERS, not array references, so a caller holding
///       them in a staged buffer or a tap cache passes them IN PLACE (E-39's
///       `RowOperands`). Array arguments decay, so every existing call site is
///       unchanged; `N` was already explicit at all of them.
/// @tparam UseNeon False forces the portable scalar path even where NEON exists.
///         That is not a tuning knob: it is how the vector path is held to
///         BIT-EXACTNESS, by giving the benchmark and the tests both spellings to
///         compare on the same machine (X-33).
template <size_t N, typename WordType, bool UseNeon = true>
inline long long slicedSignedSum(const WordType* maskedMag, WordType sign,
                                 const WordType* val) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    if constexpr (UseNeon) {
    // ===================================================================
    // NEON: THE PLANE-PAIR POPCOUNTS BATCHED INTO LANES (X-33, Phase 5.1).
    //
    // This is [D-6](../../../ARCHITECTURE.md)'s reservation being cashed in.
    // aarch64 has NO SCALAR POPCOUNT: `CNT` lives in the vector registers, so
    // every scalar `popcountWord` pays `fmov` in and `fmov` out. This call issues
    // `2N^2` of them -- EIGHT at N = 2, the depth three of four levels of the
    // adopted 1/2/2/2 ladder run at (D-23). Batching the four plane pairs into
    // lanes crosses the register domain ONCE, at the horizontal add, instead of
    // eight times.
    //
    // D-6 forbade a per-word popcount in the public API precisely so that the
    // reductions would be shaped to allow this; nothing here would be possible if
    // callers had been handed `popcountWord`.
    //
    // BIT-EXACT: the same integers, weighted the same way. The scalar path below
    // is the portable one AND the equality oracle, and
    // tests/test_opticalflow.cpp compares them.
    //
    // N == 2 only. At N == 1 there are two popcounts and nothing to batch; above
    // 2 the pair count is not a multiple of the lane count and the scalar path is
    // clearer than the packing would be.
    if constexpr (N == 2 && sizeof(WordType) == 4) {
        // The four ordered plane pairs (i, j), weights 2^(i+j) = 1, 2, 2, 4.
        const uint32_t both[4] = {
            static_cast<uint32_t>(val[0] & maskedMag[0]),
            static_cast<uint32_t>(val[1] & maskedMag[0]),
            static_cast<uint32_t>(val[0] & maskedMag[1]),
            static_cast<uint32_t>(val[1] & maskedMag[1])};
        const uint32x4_t vb = vld1q_u32(both);
        const uint32x4_t vs = vandq_u32(vb, vdupq_n_u32(static_cast<uint32_t>(sign)));
        // CNT is per byte; two pairwise widenings give one count per 32-bit lane.
        const uint32x4_t cTotal =
            vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(vb))));
        const uint32x4_t cOpp =
            vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(vs))));
        // total - 2*opposing, per lane, then weighted and reduced ONCE.
        const int32x4_t diff = vsubq_s32(vreinterpretq_s32_u32(cTotal),
                                         vshlq_n_s32(vreinterpretq_s32_u32(cOpp), 1));
        const int32_t w[4] = {1, 2, 2, 4};
        return static_cast<long long>(vaddvq_s32(vmulq_s32(diff, vld1q_s32(w))));
    }
    }
#endif
    long long acc = 0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            const WordType both = static_cast<WordType>(val[i] & maskedMag[j]);
            const long long total = static_cast<long long>(popcountWord<WordType>(both));
            const long long opposing =
                static_cast<long long>(popcountWord<WordType>(static_cast<WordType>(both & sign)));
            acc += (total - 2 * opposing) * (1LL << (i + j));
        }
    }
    return acc;
}

/// Forward declarations: `residualSums`' 1-bit overload names these in its signature
/// and they are defined below, beside the reader that consumes them (E-39).
template <size_t N, typename WordType>
struct StagedWindow;
template <size_t N, typename WordType>
struct TapCache;

/// @brief The five integer sums one gradient component's residual needs.
/// @note `sum(T * Ix)` for each of the four tap planes, and `sum(I * Ix)` for the
///       previous frame. The four bilinear weights combine them ONCE per window --
///       see the residual identity at the top of the file.
struct TapSums {
    long long t00 = 0;
    long long t01 = 0;
    long long t10 = 0;
    long long t11 = 0;
    long long self = 0;

    /// @brief `w00*t00 + w01*t01 + w10*t10 + w11*t11 - self`.
    double combine(double w00, double w01, double w10, double w11) const {
        return w00 * static_cast<double>(t00) + w01 * static_cast<double>(t01) +
               w10 * static_cast<double>(t10) + w11 * static_cast<double>(t11) -
               static_cast<double>(self);
    }
};

/// @brief The right-hand side `b = sum(diff * grad)` of the normal equations, as
///        ten exact integers. **INTERNAL, and the heart of the file.**
/// @param lv The level's six planes.
/// @param r The window, already clipped to the level (impl::clipRegion).
/// @param tapX, tapY The INTEGER part of the displacement, in pixels.
/// @param sumsX, sumsY Out: the five sums for each gradient component.
/// @note One traversal, four tap words and one previous-frame word per word index,
///       twenty popcounts per word. Nothing float is touched inside the loop.
template <typename WordType>
inline void residualSums(const LKLevel<WordType>& lv, const RegionWords<WordType>& r,
                         long long tapX, long long tapY, TapSums& sumsX, TapSums& sumsY,
                         const StagedWindow<1, WordType>* = nullptr,
                         TapCache<1, WordType>* = nullptr) {
    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* mx = lv.dxMag.row(y);
        const WordType* sx = lv.dxSign.row(y);
        const WordType* my = lv.dyMag.row(y);
        const WordType* sy = lv.dySign.row(y);
        const WordType* ip = lv.prev.row(y);

        const long long srcY = static_cast<long long>(y) + tapY;
        const ReplicatedShiftedRow<WordType> row0 = displacedRow<WordType>(lv.next, srcY, tapX);
        const ReplicatedShiftedRow<WordType> row0s =
            displacedRow<WordType>(lv.next, srcY, tapX + 1);
        const ReplicatedShiftedRow<WordType> row1 =
            displacedRow<WordType>(lv.next, srcY + 1, tapX);
        const ReplicatedShiftedRow<WordType> row1s =
            displacedRow<WordType>(lv.next, srcY + 1, tapX + 1);

        // Per-row partial sums, for the reason ops/reduce.hpp's row bodies return
        // one: a single accumulator across every row of a window is one serialized
        // dependency chain through the popcount latency (D-15, X-11b).
        TapSums rowX;
        TapSums rowY;
        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            const WordType t00 = row0.word(i);
            const WordType t01 = row0s.word(i);
            const WordType t10 = row1.word(i);
            const WordType t11 = row1s.word(i);

            const WordType magX = static_cast<WordType>(mx[i] & mask);
            const WordType signX = sx[i];
            rowX.t00 += signedMaskedSum<WordType>(magX, signX, t00);
            rowX.t01 += signedMaskedSum<WordType>(magX, signX, t01);
            rowX.t10 += signedMaskedSum<WordType>(magX, signX, t10);
            rowX.t11 += signedMaskedSum<WordType>(magX, signX, t11);
            rowX.self += signedMaskedSum<WordType>(magX, signX, ip[i]);

            const WordType magY = static_cast<WordType>(my[i] & mask);
            const WordType signY = sy[i];
            rowY.t00 += signedMaskedSum<WordType>(magY, signY, t00);
            rowY.t01 += signedMaskedSum<WordType>(magY, signY, t01);
            rowY.t10 += signedMaskedSum<WordType>(magY, signY, t10);
            rowY.t11 += signedMaskedSum<WordType>(magY, signY, t11);
            rowY.self += signedMaskedSum<WordType>(magY, signY, ip[i]);
        });
        sumsX.t00 += rowX.t00; sumsX.t01 += rowX.t01; sumsX.t10 += rowX.t10;
        sumsX.t11 += rowX.t11; sumsX.self += rowX.self;
        sumsY.t00 += rowY.t00; sumsY.t01 += rowY.t01; sumsY.t10 += rowY.t10;
        sumsY.t11 += rowY.t11; sumsY.self += rowY.self;
    }
}


/// @brief Bits `[x0, x0 + bitsPerWord)` of a row, aligned to bit 0. **INTERNAL.**
/// @note The `s == 0` guard is not decoration: shifting a word by its own width is
///       undefined, and that case is 1 in 32 rather than exotic.
template <typename WordType>
inline WordType alignedWord(const WordType* row, size_t words, size_t x0) {
    constexpr size_t bits = bitsPerWord<WordType>();
    const size_t w0 = x0 / bits;
    const size_t s = x0 % bits;
    const WordType lo = (w0 < words) ? row[w0] : static_cast<WordType>(0);
    if (s == 0) return lo;
    const WordType hi = (w0 + 1 < words) ? row[w0 + 1] : static_cast<WordType>(0);
    return static_cast<WordType>((lo >> s) | (hi << (bits - s)));
}

/// @brief Rows the staging path handles. The shipped window is 31 (D-31).
/// @note A bound, not a tuning knob: it fixes the size of a STACK buffer, and
///       CLAUDE.md forbids a kernel allocating. At the shipped `N = 2` the two
///       structures together are 4 KB; at the `N = 8` ceiling about 15 KB, which is a
///       lot for a Cortex-M and is stated rather than hidden (E-38). A taller window
///       declines and takes the unstaged path.
constexpr size_t kStagedMaxRows = 64;

/// @brief One window's ITERATION-INVARIANT words, extracted once. **INTERNAL** (X-69).
///
/// Of the twelve words a row needs, **eight belong to the PREVIOUS frame** -- `self`,
/// `magX`, `magY`, `signX`, `signY` -- and LK linearises about the previous frame, so
/// they are identical on every one of X-68's **4.29 mean iterations** and were being
/// re-extracted on all of them. `region` is fixed per point per level, so one staging
/// serves the whole iteration.
template <size_t N, typename WordType>
struct StagedWindow {
    WordType self[kStagedMaxRows][N];
    WordType magX[kStagedMaxRows][N];   ///< already masked to the region
    WordType magY[kStagedMaxRows][N];
    WordType signX[kStagedMaxRows];
    WordType signY[kStagedMaxRows];
};

/// @brief The four TAP words per row, cached against the integer displacement they
///        were read at. **INTERNAL** (X-70).
///
/// The taps move, which is why X-69 could not stage them -- but they move as
/// `floor(offX)`, and the iteration is *shrinking* `off`. Once the estimate settles
/// inside a pixel the integer part stops changing and the same four words are
/// re-extracted every remaining iteration. **Sound by construction:** the tap words
/// are a pure function of `lv.next`, `region` and `(tapX, tapY)`; the first two are
/// fixed for the point and the third is the key.
template <size_t N, typename WordType>
struct TapCache {
    /// `[row][plane][tap]`, tap order `t00, t01, t10, t11`. **THE INNERMOST AXIS IS THE
    /// TAP AND THAT IS THE WHOLE POINT** (X-85): the NEON kernels put the four taps of
    /// one plane in the four lanes of a vector, so this layout makes that a single
    /// `vld1q_u32`. Stored as four separate arrays it was **eight stores and two loads
    /// a row** to marshal them, each load waiting on its stores — the same store-to-load
    /// round trip X-83 found costing 1.5× in the covariance.
    WordType taps[kStagedMaxRows][N][4];
    long long tapX = 0;
    long long tapY = 0;
    bool valid = false;
};

/// @brief Fill a `StagedWindow`, or decline. **INTERNAL** (X-69).
/// @return False when this window cannot be staged, leaving the caller on the
///         unstaged path. Declines are not failures: a window wider than a word uses
///         `ReplicatedShiftedRow` spans rather than single words, and a taller one
///         would overrun a fixed stack buffer.
template <size_t N, typename WordType>
inline bool stageWindow(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                        StagedWindow<N, WordType>& s) {
    const size_t width = r.x1 - r.x0;
    if (width == 0 || width > bitsPerWord<WordType>()) return false;
    const size_t rows = r.y1 - r.y0;
    if (rows > kStagedMaxRows) return false;
    const size_t words = minRowWords<WordType>(lv.prev[0].width);
    const WordType mask = lowBitsMask<WordType>(width);
    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        for (size_t k = 0; k < N; ++k) {
            s.self[i][k] = alignedWord<WordType>(lv.prev[k].row(y), words, r.x0);
            s.magX[i][k] = static_cast<WordType>(
                alignedWord<WordType>(lv.dxMag[k].row(y), words, r.x0) & mask);
            s.magY[i][k] = static_cast<WordType>(
                alignedWord<WordType>(lv.dyMag[k].row(y), words, r.x0) & mask);
        }
        s.signX[i] = alignedWord<WordType>(lv.dxSign.row(y), words, r.x0);
        s.signY[i] = alignedWord<WordType>(lv.dySign.row(y), words, r.x0);
    }
    return true;
}

/// @brief The 1-bit `LKLevel` declines unconditionally. **INTERNAL.**
/// @note `stageWindow` is written against `LKLevelN`'s plane arrays. Declining keeps
///       one tracking body serving both level types (D-21) without a compile error.
template <typename WordType>
inline bool stageWindow(const LKLevel<WordType>&, const RegionWords<WordType>&,
                        StagedWindow<1, WordType>&) {
    return false;
}

/// @brief The 2x2 covariance from the ALREADY-STAGED window. **INTERNAL** (X-84).
///
/// **`levelCovariance` WALKS THE SAME WINDOW `stageWindow` HAS JUST FINISHED WALKING.**
/// The staged buffer holds `magX`, `magY`, `signX`, `signY` per row — which is exactly
/// and only what the covariance reads — so the second traversal of the level's planes
/// is pure repetition. [X-83](../../../EXPERIMENTS.md) measured the covariance at
/// **27.5% of `track` on the reference device**; this removes its memory traffic
/// entirely and leaves the arithmetic reading a ~2 KB stack buffer.
///
/// **BIT-EXACT, AND THE REASON IS THAT POPCOUNTS DO NOT CARE WHERE A BIT SITS.** The
/// staged word is the region extracted to bit 0 and masked; `bitSlicedPairRowRegion`
/// reads it in place under `visitRowWords`' mask. Every operand is shifted by the same
/// amount, so every `popcount(a & b)` is the same integer. The sign words are unmasked
/// in both spellings and only ever ANDed with a masked product (D-13).
///
/// @param rows `region.y1 - region.y0`. A staged window is one word wide by
///        construction, so this is one word per row and no `visitRowWords` at all.
template <size_t N, typename WordType>
inline GradientCovariance stagedCovariance(const StagedWindow<N, WordType>& s, size_t rows) {
    BitSlicedPairCounts<N> total;
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    if constexpr ((N == 1 || N == 2) && sizeof(WordType) == 4) {
        // X-83's lane kernel, on the staged buffer: the counts stay in lanes to the end
        // of the window and the register domain is crossed once per point per level.
        const auto counts = [](uint32x4_t v) {
            return vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(v))));
        };
        uint32x4_t accA = vdupq_n_u32(0), accB = vdupq_n_u32(0);
        uint32x4_t accC = vdupq_n_u32(0), accD = vdupq_n_u32(0);
        for (size_t i = 0; i < rows; ++i) {
            const uint32_t sel = static_cast<uint32_t>(s.signX[i] ^ s.signY[i]);
            if constexpr (N == 1) {
                const uint32_t ax = static_cast<uint32_t>(s.magX[i][0]);
                const uint32_t ay = static_cast<uint32_t>(s.magY[i][0]);
                const uint32_t lanes[4] = {ax, ay, ax & ay,
                                           static_cast<uint32_t>(ax & ay & sel)};
                accA = vaddq_u32(accA, counts(vld1q_u32(lanes)));
            } else {
                const uint32_t base[4] = {static_cast<uint32_t>(s.magX[i][0]),
                                          static_cast<uint32_t>(s.magX[i][1]),
                                          static_cast<uint32_t>(s.magY[i][0]),
                                          static_cast<uint32_t>(s.magY[i][1])};
                const uint32x4_t v = vld1q_u32(base);
                accA = vaddq_u32(accA, counts(v));
                accB = vaddq_u32(accB, counts(vandq_u32(v, vextq_u32(v, v, 1))));
                const uint32x4_t cross =
                    vandq_u32(vzip1q_u32(v, v),
                              vcombine_u32(vget_high_u32(v), vget_high_u32(v)));
                accC = vaddq_u32(accC, counts(cross));
                accD = vaddq_u32(accD, counts(vandq_u32(cross, vdupq_n_u32(sel))));
            }
        }
        if constexpr (N == 1) {
            total.xx[0][0] = vgetq_lane_u32(accA, 0);
            total.yy[0][0] = vgetq_lane_u32(accA, 1);
            total.xyTotal[0][0] = vgetq_lane_u32(accA, 2);
            total.xySet[0][0] = vgetq_lane_u32(accA, 3);
        } else {
            total.xx[0][0] = vgetq_lane_u32(accA, 0);
            total.xx[1][1] = vgetq_lane_u32(accA, 1);
            total.yy[0][0] = vgetq_lane_u32(accA, 2);
            total.yy[1][1] = vgetq_lane_u32(accA, 3);
            total.xx[0][1] = vgetq_lane_u32(accB, 0);
            total.yy[0][1] = vgetq_lane_u32(accB, 2);
            total.xyTotal[0][0] = vgetq_lane_u32(accC, 0);
            total.xyTotal[0][1] = vgetq_lane_u32(accC, 1);
            total.xyTotal[1][0] = vgetq_lane_u32(accC, 2);
            total.xyTotal[1][1] = vgetq_lane_u32(accC, 3);
            total.xySet[0][0] = vgetq_lane_u32(accD, 0);
            total.xySet[0][1] = vgetq_lane_u32(accD, 1);
            total.xySet[1][0] = vgetq_lane_u32(accD, 2);
            total.xySet[1][1] = vgetq_lane_u32(accD, 3);
        }
        return combineBitSlicedPairs<N>(total);
    }
#endif
    for (size_t i = 0; i < rows; ++i) {
        const WordType selector = static_cast<WordType>(s.signX[i] ^ s.signY[i]);
        for (size_t a = 0; a < N; ++a) {
            for (size_t b = a; b < N; ++b) {
                total.xx[a][b] += popcountWord<WordType>(
                    static_cast<WordType>(s.magX[i][a] & s.magX[i][b]));
                total.yy[a][b] += popcountWord<WordType>(
                    static_cast<WordType>(s.magY[i][a] & s.magY[i][b]));
            }
        }
        for (size_t a = 0; a < N; ++a) {
            for (size_t b = 0; b < N; ++b) {
                const WordType both =
                    static_cast<WordType>(s.magX[i][a] & s.magY[i][b]);
                total.xyTotal[a][b] += popcountWord<WordType>(both);
                total.xySet[a][b] +=
                    popcountWord<WordType>(static_cast<WordType>(both & selector));
            }
        }
    }
    return combineBitSlicedPairs<N>(total);
}

/// @brief One row's twelve operands, as POINTERS. **INTERNAL** (E-39).
/// @note **Pointers, and that was measured.** They let a staged or cached operand be
///       used IN PLACE, which is the whole point of X-69/X-70. Copying them into a
///       value struct instead gave back what staging bought -- X-72 measured both,
///       and only pointers PLUS a compile-time `Staged` reached parity.
template <size_t N, typename WordType>
struct RowOperands {
    const WordType (*taps)[4];   ///< `[plane][tap]`; see `TapCache`
    const WordType* self;
    const WordType* magX;   ///< already masked to the region
    const WordType* magY;
    WordType signX;
    WordType signY;
    WordType tapScratch[N][4];   ///< where the unstaged path materialises them
    WordType scratch[3][N];
};

/// @brief The ONE place a window row's operands are read. **INTERNAL** (E-39).
///
/// **WHY THIS EXISTS.** X-41 recorded **three copies** of this extraction block and
/// recommended collapsing them *for maintenance*. X-69 and X-70 made it worth doing
/// *for speed*: staging and tap-caching have to reach the NEON paths too, and writing
/// them into each copy separately would have made **five**. One reader serves scalar
/// and NEON, staged and unstaged -- and X-34's `+1`-tap-is-a-shift and X-35's interior
/// fast path live here once instead of four times.
///
/// @tparam Staged Compile-time, NOT a runtime pointer test. X-72 measured a single
///         body branching on `staged != nullptr` per row costing **17% of `track` on
///         x86**: the compiler stops specialising the row loop. Two instantiations of
///         one source is the price of that 17%.
template <size_t N, typename WordType, bool Staged>
class RowReader {
public:
    RowReader(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r, long long tapX,
              long long tapY, const StagedWindow<N, WordType>* staged,
              TapCache<N, WordType>* taps)
        : lv_(lv), staged_(staged), taps_(taps), x0_(r.x0), width_(r.x1 - r.x0),
          words_(minRowWords<WordType>(lv.prev[0].width)),
          mask_(lowBitsMask<WordType>(r.x1 - r.x0)),
          tapIsShift_(r.x1 - r.x0 < bitsPerWord<WordType>()),
          srcX_(static_cast<long long>(r.x0) + tapX), tapY_(tapY) {
        const long long lastCol = static_cast<long long>(lv.next[0].width) - 1;
        colsInside_ = srcX_ >= 0 && srcX_ + static_cast<long long>(width_) <= lastCol;
        if constexpr (Staged) {
            tapsFresh_ = taps_->valid && taps_->tapX == tapX && taps_->tapY == tapY;
            if (!tapsFresh_) {
                taps_->tapX = tapX;
                taps_->tapY = tapY;
                taps_->valid = true;
            }
        }
    }

    /// @param y Absolute row in the level. @param i Row index within the window.
    void load(size_t y, size_t i, RowOperands<N, WordType>& o) {
        if constexpr (Staged) {
            if (!tapsFresh_) {
                extractTaps(y, taps_->taps[i]);
#ifdef BINCV_LK_STAGE_TIMING
                ++stageTiming().tapRows;
#endif
            }
            // ALIAS, do not copy. `Staged` is compile-time, so this branch is the
            // whole body here and the operands are used where they already live.
            o.taps = taps_->taps[i];
            o.self = staged_->self[i];
            o.magX = staged_->magX[i];
            o.magY = staged_->magY[i];
            o.signX = staged_->signX[i];
            o.signY = staged_->signY[i];
        } else {
            extractTaps(y, o.tapScratch);
            extractInvariants(y, o.scratch[0], o.scratch[1], o.scratch[2], o.signX, o.signY);
            o.taps = o.tapScratch;
            o.self = o.scratch[0];
            o.magX = o.scratch[1];
            o.magY = o.scratch[2];
        }
    }

private:
    /// The four displaced taps. X-34's `+1`-is-a-shift and X-35's interior fast path.
    ///
    /// **ROW i's LOWER TAP IS ROW i+1's UPPER TAP** — they name the same level row at
    /// the same displacement, and the window is walked in increasing `y`, so carrying
    /// it forward halves the reads of `lv.next`. X-84.
    ///
    /// Sound because the two spellings agree wherever both apply: `alignedWord` is
    /// X-35's interior fast path for exactly the case `displacedRow(...).word(0)` would
    /// compute the same bits more slowly, and outside it both take `displacedRow`.
    WordType readNext(long long sy, size_t k, long long sx) const {
        if (colsInside_ && sx == srcX_ && sy >= 0 &&
            sy < static_cast<long long>(lv_.next[0].height)) {
            return alignedWord<WordType>(lv_.next[k].row(static_cast<size_t>(sy)), words_,
                                         static_cast<size_t>(sx));
        }
        return displacedRow<WordType>(lv_.next[k], sy, sx).word(0);
    }

    void extractTaps(size_t y, WordType (*out)[4]) {
        const long long srcY = static_cast<long long>(y) + tapY_;
        const bool carryOk = haveCarry_ && y == carryRow_ + 1;
        for (size_t k = 0; k < N; ++k) {
            const WordType upper = carryOk ? carry_[k] : readNext(srcY, k, srcX_);
            const WordType lower = readNext(srcY + 1, k, srcX_);
            carry_[k] = lower;
            out[k][0] = upper;
            out[k][2] = lower;
            if (tapIsShift_) {
                out[k][1] = static_cast<WordType>(upper >> 1);
                out[k][3] = static_cast<WordType>(lower >> 1);
            } else {
                const WordType upperR =
                    carryOk ? carryShift_[k] : readNext(srcY, k, srcX_ + 1);
                const WordType lowerR = readNext(srcY + 1, k, srcX_ + 1);
                carryShift_[k] = lowerR;
                out[k][1] = upperR;
                out[k][3] = lowerR;
            }
        }
        carryRow_ = y;
        haveCarry_ = true;
    }

    /// The eight previous-frame words -- what X-69 stages when it can.
    void extractInvariants(size_t y, WordType* self, WordType* magX, WordType* magY,
                           WordType& signX, WordType& signY) const {
        for (size_t k = 0; k < N; ++k) {
            self[k] = alignedWord<WordType>(lv_.prev[k].row(y), words_, x0_);
            magX[k] = static_cast<WordType>(
                alignedWord<WordType>(lv_.dxMag[k].row(y), words_, x0_) & mask_);
            magY[k] = static_cast<WordType>(
                alignedWord<WordType>(lv_.dyMag[k].row(y), words_, x0_) & mask_);
        }
        signX = alignedWord<WordType>(lv_.dxSign.row(y), words_, x0_);
        signY = alignedWord<WordType>(lv_.dySign.row(y), words_, x0_);
    }

    const LKLevelN<N, WordType>& lv_;
    const StagedWindow<N, WordType>* staged_;
    TapCache<N, WordType>* taps_;
    size_t x0_, width_, words_;
    WordType mask_;
    bool tapIsShift_, colsInside_ = false, tapsFresh_ = false;
    WordType carry_[N] = {};
    WordType carryShift_[N] = {};
    bool haveCarry_ = false;
    size_t carryRow_ = 0;
    long long srcX_, tapY_;
};

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
/// @brief The aligned residual at **N == 1**, batching across the FOUR TAPS with
///        accumulators carried across the WHOLE WINDOW. **INTERNAL** (D-33 / X-36).
/// @note `Staged` is compile-time; see `RowReader`. This is E-39: the NEON paths get
///       X-69's staging and X-70's tap cache without a fifth copy of the extraction
///       block X-41 counted three of.
template <typename WordType, bool Staged>
inline void alignedResidualSumsNeon1Impl(const LKLevelN<1, WordType>& lv,
                                         const RegionWords<WordType>& r, long long tapX,
                                         long long tapY, TapSums& sumsX, TapSums& sumsY,
                                         const StagedWindow<1, WordType>* staged,
                                         TapCache<1, WordType>* taps) {
    RowReader<1, WordType, Staged> rd(lv, r, tapX, tapY, staged, taps);
    RowOperands<1, WordType> o;

    // X-86: byte-domain accumulation, as in the N == 2 kernel. `vcntq_u8` gives per-byte
    // counts and a byte count is at most 8, so 31 rows fit in a byte and the two
    // `vpaddlq` widenings move out of the row loop entirely.
    uint8x16_t bTotX = vdupq_n_u8(0), bOppX = vdupq_n_u8(0);
    uint8x16_t bTotY = vdupq_n_u8(0), bOppY = vdupq_n_u8(0);
    uint8x16_t bSelf = vdupq_n_u8(0);
    uint32x4_t totX = vdupq_n_u32(0), oppX = vdupq_n_u32(0);
    uint32x4_t totY = vdupq_n_u32(0), oppY = vdupq_n_u32(0);
    uint32x4_t accSelf = vdupq_n_u32(0);

    const auto widen = [](uint8x16_t v) { return vpaddlq_u16(vpaddlq_u8(v)); };
    const auto flush = [&]() {
        totX = vaddq_u32(totX, widen(bTotX)); bTotX = vdupq_n_u8(0);
        oppX = vaddq_u32(oppX, widen(bOppX)); bOppX = vdupq_n_u8(0);
        totY = vaddq_u32(totY, widen(bTotY)); bTotY = vdupq_n_u8(0);
        oppY = vaddq_u32(oppY, widen(bOppY)); bOppY = vdupq_n_u8(0);
        accSelf = vaddq_u32(accSelf, widen(bSelf)); bSelf = vdupq_n_u8(0);
    };

    size_t sinceFlush = 0;
    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        rd.load(y, i, o);
        const WordType selfW = o.self[0];
        const WordType magX = o.magX[0];
        const WordType magY = o.magY[0];
        const WordType sgnX = o.signX;
        const WordType sgnY = o.signY;
        // X-85: `[plane][tap]` -- one load, nothing to marshal.
        const uint32x4_t vt = vld1q_u32(o.taps[0]);

        const uint32x4_t bx = vandq_u32(vt, vdupq_n_u32(static_cast<uint32_t>(magX)));
        bTotX = vaddq_u8(bTotX, vcntq_u8(vreinterpretq_u8_u32(bx)));
        const uint32x4_t sx = vandq_u32(bx, vdupq_n_u32(static_cast<uint32_t>(sgnX)));
        bOppX = vaddq_u8(bOppX, vcntq_u8(vreinterpretq_u8_u32(sx)));

        const uint32x4_t by = vandq_u32(vt, vdupq_n_u32(static_cast<uint32_t>(magY)));
        bTotY = vaddq_u8(bTotY, vcntq_u8(vreinterpretq_u8_u32(by)));
        const uint32x4_t sy = vandq_u32(by, vdupq_n_u32(static_cast<uint32_t>(sgnY)));
        bOppY = vaddq_u8(bOppY, vcntq_u8(vreinterpretq_u8_u32(sy)));

        // {total_X, opposing_X, total_Y, opposing_Y} -- one `cnt` for all four.
        const WordType bsX = static_cast<WordType>(selfW & magX);
        const WordType bsY = static_cast<WordType>(selfW & magY);
        const uint32_t selfLanes[4] = {static_cast<uint32_t>(bsX),
                                       static_cast<uint32_t>(bsX & sgnX),
                                       static_cast<uint32_t>(bsY),
                                       static_cast<uint32_t>(bsY & sgnY)};
        bSelf = vaddq_u8(bSelf, vcntq_u8(vreinterpretq_u8_u32(vld1q_u32(selfLanes))));

        if (++sinceFlush == 31) {
            flush();
            sinceFlush = 0;
        }
    }
    if (sinceFlush != 0) flush();

    // ONE domain crossing per window per component, not one per row.
    auto lane = [](uint32x4_t tv, uint32x4_t ov, int i) {
        return static_cast<long long>(vgetq_lane_u32(tv, i)) -
               2 * static_cast<long long>(vgetq_lane_u32(ov, i));
    };
    sumsX.t00 += lane(totX, oppX, 0); sumsX.t01 += lane(totX, oppX, 1);
    sumsX.t10 += lane(totX, oppX, 2); sumsX.t11 += lane(totX, oppX, 3);
    sumsX.self += static_cast<long long>(vgetq_lane_u32(accSelf, 0)) -
                  2 * static_cast<long long>(vgetq_lane_u32(accSelf, 1));
    sumsY.t00 += lane(totY, oppY, 0); sumsY.t01 += lane(totY, oppY, 1);
    sumsY.t10 += lane(totY, oppY, 2); sumsY.t11 += lane(totY, oppY, 3);
    sumsY.self += static_cast<long long>(vgetq_lane_u32(accSelf, 2)) -
                  2 * static_cast<long long>(vgetq_lane_u32(accSelf, 3));
}

template <typename WordType>
inline void alignedResidualSumsNeon1(const LKLevelN<1, WordType>& lv,
                                     const RegionWords<WordType>& r, long long tapX,
                                     long long tapY, TapSums& sumsX, TapSums& sumsY,
                                     const StagedWindow<1, WordType>* staged,
                                     TapCache<1, WordType>* taps) {
    if (staged != nullptr) {
        alignedResidualSumsNeon1Impl<WordType, true>(lv, r, tapX, tapY, sumsX, sumsY, staged,
                                                     taps);
    } else {
        alignedResidualSumsNeon1Impl<WordType, false>(lv, r, tapX, tapY, sumsX, sumsY,
                                                      nullptr, nullptr);
    }
}

/// @brief The aligned residual at **N == 2**, the plane pairs folded inside the row
///        into one accumulator per component. **INTERNAL** (X-40).
/// @note See `alignedResidualSumsNeon1Impl` for why `Staged` is compile-time.
template <typename WordType, bool Staged>
inline void alignedResidualSumsNeon2Impl(const LKLevelN<2, WordType>& lv,
                                         const RegionWords<WordType>& r, long long tapX,
                                         long long tapY, TapSums& sumsX, TapSums& sumsY,
                                         const StagedWindow<2, WordType>* staged,
                                         TapCache<2, WordType>* taps) {
    constexpr size_t N = 2;
    RowReader<N, WordType, Staged> rd(lv, r, tapX, tapY, staged, taps);
    RowOperands<N, WordType> o;

    // X-86: THE COUNTS STAY IN BYTES UNTIL THE END OF THE WINDOW.
    //
    // `vcntq_u8` counts per byte. Turning that into a per-TAP total takes two
    // `vpaddlq` widenings — and the old kernel paid them **on every row**, then
    // subtracted, shifted and multiply-accumulated: eleven operations per plane pair
    // per row. A byte count is at most 8 and a window is 31 rows, so **248 fits in a
    // byte**: the widening can wait for the window's end and the row body collapses to
    // AND, `cnt`, byte-add.
    //
    // Six operations where there were eleven, and it is the same trick
    // [X-80](../../../EXPERIMENTS.md) used to make the bit-plane FAST worth having.
    //
    // Sixteen byte accumulators — total and opposing for each of the four plane pairs,
    // twice for the two components — plus four for the previous-frame term. aarch64
    // has thirty-two vector registers and they fit; this is the shape that would NOT
    // have fitted on x86, which is why the AVX2 batch is a different kernel.
    uint8x16_t tX[4], oX[4], tY[4], oY[4];
    for (int k = 0; k < 4; ++k) {
        tX[k] = vdupq_n_u8(0); oX[k] = vdupq_n_u8(0);
        tY[k] = vdupq_n_u8(0); oY[k] = vdupq_n_u8(0);
    }
    uint8x16_t tS[2], oS[2];
    for (int k = 0; k < 2; ++k) { tS[k] = vdupq_n_u8(0); oS[k] = vdupq_n_u8(0); }

    int32x4_t accX = vdupq_n_s32(0), accY = vdupq_n_s32(0);
    int32x4_t accSelfX = vdupq_n_s32(0), accSelfY = vdupq_n_s32(0);
    static const int32_t kPairW[4] = {1, 2, 2, 4};

    // Widen the byte accumulators into the running 32-bit ones and clear them. Called
    // every 31 rows and once at the end -- 31 * 8 = 248, and a 32nd row would overflow.
    const auto flushPairs = [&](uint8x16_t (&tp)[4], uint8x16_t (&op)[4], int32x4_t& acc) {
        for (int k = 0; k < 4; ++k) {
            const uint32x4_t ct = vpaddlq_u16(vpaddlq_u8(tp[k]));
            const uint32x4_t co = vpaddlq_u16(vpaddlq_u8(op[k]));
            const int32x4_t d = vsubq_s32(vreinterpretq_s32_u32(ct),
                                          vshlq_n_s32(vreinterpretq_s32_u32(co), 1));
            acc = vmlaq_n_s32(acc, d, kPairW[k]);
            tp[k] = vdupq_n_u8(0);
            op[k] = vdupq_n_u8(0);
        }
    };
    const auto flushSelf = [&](uint8x16_t& tp, uint8x16_t& op, int32x4_t& acc) {
        const uint32x4_t ct = vpaddlq_u16(vpaddlq_u8(tp));
        const uint32x4_t co = vpaddlq_u16(vpaddlq_u8(op));
        acc = vaddq_s32(acc, vsubq_s32(vreinterpretq_s32_u32(ct),
                                       vshlq_n_s32(vreinterpretq_s32_u32(co), 1)));
        tp = vdupq_n_u8(0);
        op = vdupq_n_u8(0);
    };

    size_t sinceFlush = 0;
    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        rd.load(y, i, o);
        // X-85: `[plane][tap]`, so this is two loads and no marshalling.
        const uint32x4_t vp0 = vld1q_u32(o.taps[0]), vp1 = vld1q_u32(o.taps[1]);
        const uint32x4_t sgX = vdupq_n_u32(static_cast<uint32_t>(o.signX));
        const uint32x4_t sgY = vdupq_n_u32(static_cast<uint32_t>(o.signY));

        // pairs (i, j) = (0,0) (1,0) (0,1) (1,1); weights 1, 2, 2, 4.
        const uint32x4_t mx[2] = {vdupq_n_u32(static_cast<uint32_t>(o.magX[0])),
                                  vdupq_n_u32(static_cast<uint32_t>(o.magX[1]))};
        const uint32x4_t my[2] = {vdupq_n_u32(static_cast<uint32_t>(o.magY[0])),
                                  vdupq_n_u32(static_cast<uint32_t>(o.magY[1]))};
        const uint32x4_t vp[2] = {vp0, vp1};
        for (int k = 0; k < 4; ++k) {
            const uint32x4_t bx = vandq_u32(vp[k & 1], mx[k >> 1]);
            tX[k] = vaddq_u8(tX[k], vcntq_u8(vreinterpretq_u8_u32(bx)));
            oX[k] = vaddq_u8(oX[k],
                             vcntq_u8(vreinterpretq_u8_u32(vandq_u32(bx, sgX))));
            const uint32x4_t by = vandq_u32(vp[k & 1], my[k >> 1]);
            tY[k] = vaddq_u8(tY[k], vcntq_u8(vreinterpretq_u8_u32(by)));
            oY[k] = vaddq_u8(oY[k],
                             vcntq_u8(vreinterpretq_u8_u32(vandq_u32(by, sgY))));
        }

        // The previous-frame term: the same four plane pairs against `self`, in lanes —
        // and built by SHUFFLE, not through a stack array. X-86: `self`, `magX` and
        // `magY` are each two CONTIGUOUS words in the staged row (and in the unstaged
        // scratch), so a 64-bit load and two lane moves give `{s0,s1,s0,s1}` against
        // `{m0,m0,m1,m1}`. The array spelling was four ANDs, four stores and a load
        // that waited on all of them — the same store-to-load round trip that has now
        // cost this project four separate times (X-83, X-85, and twice here).
        const uint32x4_t ss = vcombine_u32(vld1_u32(o.self), vld1_u32(o.self));
        const auto spread = [](const WordType* m) {
            const uint32x2_t p = vld1_u32(m);
            const uint32x4_t d = vcombine_u32(p, p);
            return vzip1q_u32(d, d);   // {m0, m0, m1, m1}
        };
        const uint32x4_t vbx = vandq_u32(ss, spread(o.magX));
        const uint32x4_t vby = vandq_u32(ss, spread(o.magY));
        tS[0] = vaddq_u8(tS[0], vcntq_u8(vreinterpretq_u8_u32(vbx)));
        oS[0] = vaddq_u8(oS[0], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(vbx, sgX))));
        tS[1] = vaddq_u8(tS[1], vcntq_u8(vreinterpretq_u8_u32(vby)));
        oS[1] = vaddq_u8(oS[1], vcntq_u8(vreinterpretq_u8_u32(vandq_u32(vby, sgY))));

        if (++sinceFlush == 31) {
            flushPairs(tX, oX, accX);
            flushPairs(tY, oY, accY);
            flushSelf(tS[0], oS[0], accSelfX);
            flushSelf(tS[1], oS[1], accSelfY);
            sinceFlush = 0;
        }
    }
    if (sinceFlush != 0) {
        flushPairs(tX, oX, accX);
        flushPairs(tY, oY, accY);
        flushSelf(tS[0], oS[0], accSelfX);
        flushSelf(tS[1], oS[1], accSelfY);
    }

    // ONE domain crossing per component, not one per call.
    sumsX.t00 += vgetq_lane_s32(accX, 0); sumsX.t01 += vgetq_lane_s32(accX, 1);
    sumsX.t10 += vgetq_lane_s32(accX, 2); sumsX.t11 += vgetq_lane_s32(accX, 3);
    sumsY.t00 += vgetq_lane_s32(accY, 0); sumsY.t01 += vgetq_lane_s32(accY, 1);
    sumsY.t10 += vgetq_lane_s32(accY, 2); sumsY.t11 += vgetq_lane_s32(accY, 3);
    const int32x4_t vw = vld1q_s32(kPairW);
    sumsX.self += static_cast<long long>(vaddvq_s32(vmulq_s32(accSelfX, vw)));
    sumsY.self += static_cast<long long>(vaddvq_s32(vmulq_s32(accSelfY, vw)));
}

template <typename WordType>
inline void alignedResidualSumsNeon2(const LKLevelN<2, WordType>& lv,
                                     const RegionWords<WordType>& r, long long tapX,
                                     long long tapY, TapSums& sumsX, TapSums& sumsY,
                                     const StagedWindow<2, WordType>* staged,
                                     TapCache<2, WordType>* taps) {
    if (staged != nullptr) {
        alignedResidualSumsNeon2Impl<WordType, true>(lv, r, tapX, tapY, sumsX, sumsY, staged,
                                                     taps);
    } else {
        alignedResidualSumsNeon2Impl<WordType, false>(lv, r, tapX, tapY, sumsX, sumsY,
                                                      nullptr, nullptr);
    }
}
#endif

/// @brief `residualSums` for a region that fits in ONE word. **INTERNAL.**
/// @note See `alignedResidualSumsNeon1Impl` for why `Staged` is compile-time.
template <size_t N, typename WordType, bool UseNeon, bool Staged>
inline void alignedResidualSumsImpl(const LKLevelN<N, WordType>& lv,
                                    const RegionWords<WordType>& r, long long tapX,
                                    long long tapY, TapSums& sumsX, TapSums& sumsY,
                                    const StagedWindow<N, WordType>* staged,
                                    TapCache<N, WordType>* taps) {
    RowReader<N, WordType, Staged> rd(lv, r, tapX, tapY, staged, taps);
    RowOperands<N, WordType> o;
    for (size_t y = r.y0, i = 0; y < r.y1; ++y, ++i) {
        rd.load(y, i, o);
        // X-85: the taps are stored `[plane][tap]` for the NEON kernels' sake, and
        // `slicedSignedSum` wants a value's N planes contiguous -- so this path
        // transposes the 4xN block once a row. It is the generic fallback (the shipped
        // 1/2/2/2 ladder at `uint32_t` takes the NEON kernels or the AVX2 batch), and
        // 4N moves a row is cheaper than the eight stores a row the old layout cost the
        // paths that DO ship.
        WordType val[4][N];
        for (size_t k = 0; k < N; ++k) {
            val[0][k] = o.taps[k][0];
            val[1][k] = o.taps[k][1];
            val[2][k] = o.taps[k][2];
            val[3][k] = o.taps[k][3];
        }
        sumsX.t00 += slicedSignedSum<N, WordType, UseNeon>(o.magX, o.signX, val[0]);
        sumsX.t01 += slicedSignedSum<N, WordType, UseNeon>(o.magX, o.signX, val[1]);
        sumsX.t10 += slicedSignedSum<N, WordType, UseNeon>(o.magX, o.signX, val[2]);
        sumsX.t11 += slicedSignedSum<N, WordType, UseNeon>(o.magX, o.signX, val[3]);
        sumsY.t00 += slicedSignedSum<N, WordType, UseNeon>(o.magY, o.signY, val[0]);
        sumsY.t01 += slicedSignedSum<N, WordType, UseNeon>(o.magY, o.signY, val[1]);
        sumsY.t10 += slicedSignedSum<N, WordType, UseNeon>(o.magY, o.signY, val[2]);
        sumsY.t11 += slicedSignedSum<N, WordType, UseNeon>(o.magY, o.signY, val[3]);
        sumsX.self += slicedSignedSum<N, WordType, UseNeon>(o.magX, o.signX, o.self);
        sumsY.self += slicedSignedSum<N, WordType, UseNeon>(o.magY, o.signY, o.self);
    }
}

template <size_t N, typename WordType, bool UseNeon>
inline void alignedResidualSums(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                                long long tapX, long long tapY, TapSums& sumsX, TapSums& sumsY,
                                const StagedWindow<N, WordType>* staged = nullptr,
                                TapCache<N, WordType>* taps = nullptr) {
    const size_t width = r.x1 - r.x0;
    if (width == 0) return;
    if (staged != nullptr) {
        alignedResidualSumsImpl<N, WordType, UseNeon, true>(lv, r, tapX, tapY, sumsX, sumsY,
                                                            staged, taps);
    } else {
        alignedResidualSumsImpl<N, WordType, UseNeon, false>(lv, r, tapX, tapY, sumsX, sumsY,
                                                             nullptr, nullptr);
    }
}

/// @brief `b = sum(diff * grad)` at **N bits per pixel**, as ten exact integers.
///        **INTERNAL, and the generic-N form of the function above.**
///
/// **THE RESIDUAL IDENTITY SURVIVES AN N-BIT ALPHABET UNCHANGED, AND THAT IS THE
/// WHOLE REASON AN N-BIT TRACKER IS AFFORDABLE AT ALL.** Item 3 of THE BOUNDARY
/// rests on bilinear interpolation being LINEAR in the four taps -- not on the
/// taps being bits. A linear functional of a bit-sliced value is the weighted sum
/// of its planes' popcounts, so widening the alphabet moves `sum(T * Ix)` from two
/// popcounts to `2N^2` and changes nothing else: the four weights still leave the
/// per-pixel loop, still combine five exact integers ONCE per window per
/// iteration, and still round nothing before they are applied.
///
/// @note **`20 N^2` popcounts per word**, against 20 at N = 1, which this reduces
///       to exactly: five plane sums (four taps and the previous frame) for each
///       of two gradient components, `2N^2` popcounts each. That quadratic is the
///       cost side of E-7 and it is the same `3N^2 + N`-shaped growth
///       ops/covariance.hpp pays -- both come from the SAME source, a product of
///       two bit-sliced values needing every plane pair.
/// @note The per-row partial sums are the same ten `long long`s at every N, for
///       the reason the 1-bit form has them (D-15, X-11b): a single accumulator
///       across every row is one serialized dependency chain through the popcount
///       latency. The count does not grow with N because `slicedSignedSum` folds
///       its `N^2` terms internally.
/// @note `4N` displaced row readers per row, not `4`. They are built once per row
///       and are word-aligned reads of the same two next-frame rows per plane, so
///       item 5 of THE BOUNDARY also survives: still no gather, still no per-pixel
///       address arithmetic.
template <size_t N, typename WordType, bool UseNeon = true>
inline void residualSums(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                         long long tapX, long long tapY, TapSums& sumsX, TapSums& sumsY,
                         const StagedWindow<N, WordType>* staged = nullptr,
                         TapCache<N, WordType>* taps = nullptr) {
    // ===================================================================
    // THE ALIGNED FAST PATH (X-34). A 31-pixel window at an arbitrary offset
    // spans 1.94 `uint32_t` words on average -- it fits in one only when
    // `x0 % 32 <= 1`, two cases in thirty-two -- so the general path below issues
    // TWICE THE POPCOUNTS IT NEEDS, each covering 15.5 useful pixels instead of
    // 31. Extracting the region into bits [0, width) of a single word makes every
    // popcount cover the whole window, and removes the per-word loop and its
    // head/tail masking with it.
    //
    // THE TAPS COST NOTHING EXTRA: `ReplicatedShiftedRow` already shifts, so
    // asking it for `word(0)` with `off = x0 + tapX` returns exactly the source
    // bits at the window's left edge, aligned. Only the previous-frame planes need
    // an explicit extraction, and the region is already clipped, so every bit that
    // survives the mask is inside the frame and no border handling arises.
    //
    // Measured on the reference device: **2.13x**, bit-exact.
    //
    // Guarded on the window fitting a word -- `LKParams` allows any `winWidth`,
    // and at 31x31 (`seal_params.yaml`) it fits at every word type binCV supports.
    // ===================================================================
    if (r.x1 - r.x0 <= bitsPerWord<WordType>()) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
        // N == 1 is level 0 of every ladder and got NOTHING from the plane-pair
        // batching, because at N == 1 there is one pair. Batching across the four
        // TAPS is what applies here (X-36).
        if constexpr (UseNeon && N == 1 && sizeof(WordType) == 4) {
            alignedResidualSumsNeon1<WordType>(lv, r, tapX, tapY, sumsX, sumsY, staged,
                                              taps);
            return;
        }
        // N == 2 is levels 1-3 of the shipped ladder, and had the plane pairs in
        // lanes but still reduced once per call. X-40 folds the pairs into a
        // window-carried accumulator instead.
        if constexpr (UseNeon && N == 2 && sizeof(WordType) == 4) {
            if (r.x1 > r.x0) {
                alignedResidualSumsNeon2<WordType>(lv, r, tapX, tapY, sumsX, sumsY, staged,
                                                   taps);
                return;
            }
        }
#endif
        alignedResidualSums<N, WordType, UseNeon>(lv, r, tapX, tapY, sumsX, sumsY, staged,
                                                  taps);
        return;
    }
    // The general path reads `ReplicatedShiftedRow` SPANS, not single words, so a
    // staged buffer cannot serve it. `stageWindow` already declines for these
    // windows; the pointers are ignored here rather than silently half-used.
    (void)staged;
    (void)taps;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* mx[N];
        const WordType* my[N];
        const WordType* ip[N];
        for (size_t k = 0; k < N; ++k) {
            mx[k] = lv.dxMag[k].row(y);
            my[k] = lv.dyMag[k].row(y);
            ip[k] = lv.prev[k].row(y);
        }
        const WordType* sx = lv.dxSign.row(y);
        const WordType* sy = lv.dySign.row(y);

        const long long srcY = static_cast<long long>(y) + tapY;
        ReplicatedShiftedRow<WordType> tapRows[4][N];
        for (size_t k = 0; k < N; ++k) {
            tapRows[0][k] = displacedRow<WordType>(lv.next[k], srcY, tapX);
            tapRows[1][k] = displacedRow<WordType>(lv.next[k], srcY, tapX + 1);
            tapRows[2][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX);
            tapRows[3][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX + 1);
        }

        TapSums rowX;
        TapSums rowY;
        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            WordType t00[N], t01[N], t10[N], t11[N], self[N];
            for (size_t k = 0; k < N; ++k) {
                t00[k] = tapRows[0][k].word(i);
                t01[k] = tapRows[1][k].word(i);
                t10[k] = tapRows[2][k].word(i);
                t11[k] = tapRows[3][k].word(i);
                self[k] = ip[k][i];
            }

            // Masked ONCE per plane here, then reused across every value plane
            // inside slicedSignedSum -- N masks rather than N^2.
            WordType magX[N], magY[N];
            for (size_t k = 0; k < N; ++k) {
                magX[k] = static_cast<WordType>(mx[k][i] & mask);
                magY[k] = static_cast<WordType>(my[k][i] & mask);
            }
            const WordType signX = sx[i];
            const WordType signY = sy[i];

            rowX.t00 += slicedSignedSum<N, WordType, UseNeon>(magX, signX, t00);
            rowX.t01 += slicedSignedSum<N, WordType, UseNeon>(magX, signX, t01);
            rowX.t10 += slicedSignedSum<N, WordType, UseNeon>(magX, signX, t10);
            rowX.t11 += slicedSignedSum<N, WordType, UseNeon>(magX, signX, t11);
            rowX.self += slicedSignedSum<N, WordType, UseNeon>(magX, signX, self);

            rowY.t00 += slicedSignedSum<N, WordType, UseNeon>(magY, signY, t00);
            rowY.t01 += slicedSignedSum<N, WordType, UseNeon>(magY, signY, t01);
            rowY.t10 += slicedSignedSum<N, WordType, UseNeon>(magY, signY, t10);
            rowY.t11 += slicedSignedSum<N, WordType, UseNeon>(magY, signY, t11);
            rowY.self += slicedSignedSum<N, WordType, UseNeon>(magY, signY, self);
        });
        sumsX.t00 += rowX.t00; sumsX.t01 += rowX.t01; sumsX.t10 += rowX.t10;
        sumsX.t11 += rowX.t11; sumsX.self += rowX.self;
        sumsY.t00 += rowY.t00; sumsY.t01 += rowY.t01; sumsY.t10 += rowY.t10;
        sumsY.t11 += rowY.t11; sumsY.self += rowY.self;
    }
}

/// @brief Mean absolute residual over the window, in binCV's {0, 1} intensity
///        units. **INTERNAL.**
/// @note `|Jinterp - I| = I + (1 - 2I)*Jinterp` because `I` is a BIT and
///       `Jinterp` lies in [0, 1]. So the absolute value -- the one nonlinearity
///       in the whole operation -- still collapses to popcounts: the window sum is
///       `count(I) + sum(Jinterp) - 2*sum over I of Jinterp`, and each `sum` is
///       the four tap planes' counts weighted by the four bilinear weights.
/// @note **THE DENOMINATOR IS THE CLIPPED PIXEL COUNT**, i.e. the number of pixels
///       actually compared, not the window's area. The reference divides by the
///       FULL window area (`errval / (32*winSize.width*cn*winSize.height)`),
///       because it pads every level and so always has `w*h` pixels to compare.
///       The two agree exactly for a window that does not clip; for one that does,
///       the reference would report the same total spread over more pixels. Since
///       binCV declines the padded level (deviation (ii)), the pixels outside are
///       not merely unknown, they do not exist -- so the mean is taken over the
///       set the residual was actually computed on, which is also the set `A` and
///       `b` were accumulated over. For an unclipped window, multiply by 255 to
///       reach the reference's {0,255} scale.
template <typename WordType>
inline float windowMeanAbsDiff(const LKLevel<WordType>& lv, const RegionWords<WordType>& r,
                               long long tapX, long long tapY, double w00, double w01, double w10,
                               double w11) {
    size_t pixels = 0;
    size_t countI = 0;
    long long allT[4] = {0, 0, 0, 0};
    long long selfT[4] = {0, 0, 0, 0};

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* ip = lv.prev.row(y);
        const long long srcY = static_cast<long long>(y) + tapY;
        const ReplicatedShiftedRow<WordType> taps[4] = {
            displacedRow<WordType>(lv.next, srcY, tapX),
            displacedRow<WordType>(lv.next, srcY, tapX + 1),
            displacedRow<WordType>(lv.next, srcY + 1, tapX),
            displacedRow<WordType>(lv.next, srcY + 1, tapX + 1)};

        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            pixels += popcountWord<WordType>(mask);
            const WordType iw = static_cast<WordType>(ip[i] & mask);
            countI += popcountWord<WordType>(iw);
            for (int k = 0; k < 4; ++k) {
                const WordType t = static_cast<WordType>(taps[k].word(i) & mask);
                allT[k] += static_cast<long long>(popcountWord<WordType>(t));
                selfT[k] += static_cast<long long>(
                    popcountWord<WordType>(static_cast<WordType>(t & iw)));
            }
        });
    }
    if (pixels == 0) return 0.0f;

    const double w[4] = {w00, w01, w10, w11};
    double sumJ = 0.0;
    double sumJoverI = 0.0;
    for (int k = 0; k < 4; ++k) {
        sumJ += w[k] * static_cast<double>(allT[k]);
        sumJoverI += w[k] * static_cast<double>(selfT[k]);
    }
    const double total = static_cast<double>(countI) + sumJ - 2.0 * sumJoverI;
    return static_cast<float>(total / static_cast<double>(pixels));
}

/// @brief Mean absolute residual over the window at **N bits per pixel**.
///        **INTERNAL.**
///
/// ===========================================================================
/// THIS IS THE ONE PIECE OF THE BIT-PARALLEL BOUNDARY THAT DOES NOT SURVIVE
/// N > 1, AND IT IS STATED RATHER THAN QUIETLY APPROXIMATED
/// ===========================================================================
/// Items 1, 2, 3 and 5 of THE BOUNDARY at the top of this file all carry over to
/// an N-bit alphabet unchanged, because every one of them is LINEAR in the
/// intensity and a linear functional of a bit-sliced value is a weighted sum of
/// its planes' popcounts. **Item 4 is the exception, and it is the only one.**
/// `|Jinterp - I|` collapsed to `I + (1 - 2I)*Jinterp` because `I` is a BIT: with
/// `Jinterp` in `[0, 1]` the SIGN of `Jinterp - I` is fixed by `I` alone, so the
/// one nonlinearity in the operation never has to be evaluated. At N > 1 the sign
/// depends on both operands, and bit-slicing does not recover it -- `Jinterp` is a
/// convex combination of four integers with floating-point weights, so it is not
/// an integer and there is no comparator to slice. Rounding it to fixed point
/// first, as the reference does per pixel, would make a comparator available but
/// would need a bit-sliced multiply-accumulate to build its planes, which costs
/// more than the scalar loop it would replace.
///
/// So this is computed PER PIXEL, and that is a real and acknowledged asymmetry
/// with the 1-bit form above. It is not a hole in the tracker's claim, for four
/// reasons that are properties of where `err` sits rather than excuses: it is an
/// OPTIONAL output (`err == nullptr` is the frontend's own call and skips this
/// entirely), it is computed at level 0 ONLY, it runs once per point per frame and
/// never inside the iteration loop, and nothing in the solve reads it -- `status`
/// does not depend on it, deliberately (deviation (vii)). **The per-pixel cost of
/// TRACKING at N bits remains integers and popcounts only.**
///
/// @note **UNITS: LSBs, so the scale depends on N.** The 1-bit form returns a mean
///       in `{0, 1}` intensity units; this returns one in `[0, 2^N - 1]` units,
///       because that is the level's alphabet. Divide by `2^N - 1` to compare
///       across depths, or multiply by `255 / (2^N - 1)` to reach the reference's
///       `{0, 255}` scale. At N = 1 both statements are the old one.
/// @note The denominator is the CLIPPED pixel count, exactly as above and for the
///       same reason.
template <size_t N, typename WordType>
inline float windowMeanAbsDiff(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                               long long tapX, long long tapY, double w00, double w01, double w10,
                               double w11) {
    const double w[4] = {w00, w01, w10, w11};
    size_t pixels = 0;
    double total = 0.0;

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* ip[N];
        for (size_t k = 0; k < N; ++k) ip[k] = lv.prev[k].row(y);

        const long long srcY = static_cast<long long>(y) + tapY;
        ReplicatedShiftedRow<WordType> tapRows[4][N];
        for (size_t k = 0; k < N; ++k) {
            tapRows[0][k] = displacedRow<WordType>(lv.next[k], srcY, tapX);
            tapRows[1][k] = displacedRow<WordType>(lv.next[k], srcY, tapX + 1);
            tapRows[2][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX);
            tapRows[3][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX + 1);
        }

        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            WordType iw[N];
            WordType tw[4][N];
            for (size_t k = 0; k < N; ++k) {
                iw[k] = ip[k][i];
                for (size_t t = 0; t < 4; ++t) tw[t][k] = tapRows[t][k].word(i);
            }
            for (size_t b = 0; b < bitsPerWord<WordType>(); ++b) {
                const WordType bit = bitMask<WordType>(b);
                if ((mask & bit) == 0) continue;
                unsigned iv = 0;
                for (size_t k = 0; k < N; ++k) {
                    if ((iw[k] & bit) != 0) iv |= (1u << k);
                }
                double j = 0.0;
                for (size_t t = 0; t < 4; ++t) {
                    unsigned jv = 0;
                    for (size_t k = 0; k < N; ++k) {
                        if ((tw[t][k] & bit) != 0) jv |= (1u << k);
                    }
                    j += w[t] * static_cast<double>(jv);
                }
                total += std::fabs(j - static_cast<double>(iv));
                ++pixels;
            }
        });
    }
    if (pixels == 0) return 0.0f;
    return static_cast<float>(total / static_cast<double>(pixels));
}

/// @brief `floor(v)` as a `long long`, for a value already known to be finite and
///        within the frame's range.
inline long long floorToLL(float v) { return static_cast<long long>(std::floor(v)); }

} // namespace impl

namespace impl {

/// The most pyramid levels the tracker will consume. `seal_params.yaml`'s
/// `lk_max_level: 3` is four; a 640x480 frame stalls at 1 pixel after 10. Fixed so
/// that `LKContext` can carry every level's extent WITHOUT a scratch allocation.
constexpr size_t kMaxLevels = 16;


#ifdef BINCV_LK_ITERATION_HISTOGRAM
/// @brief X-78's iteration counter. **DIAGNOSTIC ONLY — off by default, and it must
///        stay that way.**
///
/// T5.16's AVX2 keypoint batch runs eight lanes in lockstep, so a batch costs the
/// **maximum** iteration count over its eight points, not the mean X-68 measured. That
/// ratio decides whether the batch is worth writing at all, and it is knowable from a
/// histogram before any AVX2 exists — which is the entire reason this hook exists.
///
/// `counts` is caller-owned and `levelCount * pointCount` wide; entry `li * n + p` is
/// the number of iterations point `p` actually executed at level `li`, zero for a point
/// rejected before the loop. **Written from inside `parallelFor`**, so a caller that has
/// installed a parallel backend must not read it — indices are distinct, but the
/// benchmark that consumes this runs serial anyway.
struct IterationTrace {
    unsigned* counts = nullptr;
    size_t pointCount = 0;
};
inline IterationTrace& iterationTrace() {
    static IterationTrace t;
    return t;
}
#endif

/// A level's extent, so `usableLevelCount` needs no <utility>.
struct LevelDims {
    size_t width = 0;
    size_t height = 0;
};

/// @brief Everything the per-level tracking body needs that does not vary with the
///        level. **INTERNAL.**
/// @note Not templated on WordType: nothing in it is a view or a word. The level
///       is the only WordType-dependent argument `trackOneLevel` takes, which is
///       what lets one body serve a ladder whose levels have DIFFERENT types.
struct LKContext {
    const Point2f* prevPts = nullptr;
    Point2f* nextPts = nullptr;
    uint8_t* status = nullptr;
    float* err = nullptr;
    size_t pointCount = 0;
    size_t usableLevels = 0;
    int winW = 0;
    int winH = 0;
    float halfWinX = 0.0f;
    float halfWinY = 0.0f;
    int maxIterations = 0;
    double eps2 = 0.0;
    double minEigThreshold = 0.0;
    LKEntryLevel entryLevel = LKEntryLevel::Coarsest;

    /// Every usable level's extent, so that a point's entry level can be RECOMPUTED
    /// rather than cached. A per-point array would be scratch, and this operation
    /// has none -- not one byte (CLAUDE.md: no heap allocation inside kernels).
    /// The recomputation is at most `kMaxLevels` comparisons per point per level,
    /// against a 31x31 window's worth of popcounts.
    LevelDims dims[kMaxLevels];
};

/// @brief Is point `p`'s window entirely inside level `li`?
/// @note **This is `unclippedAtEveryLevel`'s predicate in tests/test_opticalflow.cpp,
///       deliberately spelled the same way.** X-25 measures a policy against a
///       point set defined by clipping; if the kernel's notion of "fits" and the
///       harness's disagreed, the experiment would be comparing two definitions
///       rather than two policies.
inline bool windowFitsAtLevel(const LKContext& c, const Point2f& point, size_t li) {
    const double scale = 1.0 / static_cast<double>(size_t{1} << li);
    const double x = static_cast<double>(point.x) * scale;
    const double y = static_cast<double>(point.y) * scale;
    return x - static_cast<double>(c.halfWinX) >= 0.0 &&
           y - static_cast<double>(c.halfWinY) >= 0.0 &&
           x + static_cast<double>(c.halfWinX) < static_cast<double>(c.dims[li].width) &&
           y + static_cast<double>(c.halfWinY) < static_cast<double>(c.dims[li].height);
}

/// @brief The coarsest usable level whose window contains point `p`, or 0.
/// @note Fitting is MONOTONE in the level index -- the window's half-extent is
///       fixed in pixels while the level halves, so `x >= halfWin * 2^l` tightens
///       as `l` grows -- which is what makes "the coarsest level that fits" also
///       "every level below it fits", and therefore makes a single scan correct.
/// @note Returns 0 when nothing fits: level 0 is always tracked, whatever it clips,
///       because it is the frame. So this policy loses NO keypoints.
inline size_t entryLevelFor(const LKContext& c, size_t p) {
    if (c.entryLevel == LKEntryLevel::Coarsest) return c.usableLevels - 1;
    for (size_t li = c.usableLevels; li-- > 0;) {
        if (windowFitsAtLevel(c, c.prevPts[p], li)) return li;
    }
    return 0;
}

/// @brief A level's planes all share its dimensions. **INTERNAL.**
template <typename WordType>
inline void checkLevelPlanes(const LKLevel<WordType>& lv) {
    BINCV_ASSERT(lv.prev.width == lv.next.width && lv.prev.height == lv.next.height &&
                     lv.prev.width == lv.dxMag.width && lv.prev.height == lv.dxMag.height &&
                     lv.prev.width == lv.dxSign.width && lv.prev.height == lv.dxSign.height &&
                     lv.prev.width == lv.dyMag.width && lv.prev.height == lv.dyMag.height &&
                     lv.prev.width == lv.dySign.width && lv.prev.height == lv.dySign.height,
                 "opticalFlow: a level's six planes must share its dimensions");
    (void)lv;
}

/// @brief The same check over an N-bit level's `4N + 2` planes. **INTERNAL.**
template <size_t N, typename WordType>
inline void checkLevelPlanes(const LKLevelN<N, WordType>& lv) {
    const size_t w = lv.prev[0].width;
    const size_t h = lv.prev[0].height;
    for (size_t k = 0; k < N; ++k) {
        BINCV_ASSERT(lv.prev[k].width == w && lv.prev[k].height == h &&
                         lv.next[k].width == w && lv.next[k].height == h &&
                         lv.dxMag[k].width == w && lv.dxMag[k].height == h &&
                         lv.dyMag[k].width == w && lv.dyMag[k].height == h,
                     "opticalFlow: a level's planes must share its dimensions");
    }
    BINCV_ASSERT(lv.dxSign.width == w && lv.dxSign.height == h && lv.dySign.width == w &&
                     lv.dySign.height == h,
                 "opticalFlow: a level's sign planes must share its dimensions");
    (void)w;
    (void)h;
}

/// @brief The 2x2 normal-equations matrix for a level of any depth. **INTERNAL.**
/// @note Two overloads over one name so that `trackOneLevel` names the operation
///       rather than the representation. The N-bit one is ops/covariance.hpp's
///       bit-sliced weighted-sum form (T3.10), which is bit-identical to the
///       ternary one at N == 1 -- so this dispatch changes which code runs but
///       cannot change the answer at the depth both can express.
template <typename WordType>
inline GradientCovariance levelCovariance(const LKLevel<WordType>& lv, Rect window) {
    return gradientCovariance<WordType>(lv.dxMag, lv.dyMag, lv.dxSign, lv.dySign, window);
}
template <size_t N, typename WordType>
inline GradientCovariance levelCovariance(const LKLevelN<N, WordType>& lv, Rect window) {
    return gradientCovariance<N, WordType>(lv.dxMag, lv.dyMag, lv.dxSign, lv.dySign, window);
}

/// @brief One point, one level — the whole tracking body for a single keypoint.
///        **INTERNAL.**
///
/// **THIS WAS THE `parallelFor` LAMBDA AND IT IS UNCHANGED.** It was lifted out so
/// that [X-79](../../../EXPERIMENTS.md)'s batched path has something to fall back
/// TO: a window the batch cannot stage — wider than a word, or taller than
/// `kLkBatchMaxRows` — must still be tracked, and tracked identically. Naming the
/// body once is what keeps "identically" a property of the code rather than of two
/// copies staying in step.
template <typename LevelT, typename WordType = typename LevelT::Word>
inline void trackOnePoint(const LevelT& lv, size_t li, const LKContext& c, size_t p,
                          bool finest, float scale, long long levelWidth,
                          long long levelHeight, double kLevelMinEigScale) {
    if (li > entryLevelFor(c, p)) return;
    BINCV_STAGE_MARK();
    BINCV_STAGE_POINT();
    const float prevX = c.prevPts[p].x * scale - c.halfWinX;
    const float prevY = c.prevPts[p].y * scale - c.halfWinY;
    const long long anchorX = floorToLL(prevX);
    const long long anchorY = floorToLL(prevY);

    // LOSS RULE 1 -- the window's origin is out of range. The reference's
    // own bounds, which allow a window almost entirely outside; what is
    // outside is then clipped away rather than padded (deviation (ii)).
    if (anchorX < -static_cast<long long>(c.winW) || anchorX >= levelWidth ||
        anchorY < -static_cast<long long>(c.winH) || anchorY >= levelHeight) {
        if (finest) c.status[p] = 0;
        return;
    }

    const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), c.winW, c.winH);
    const RegionWords<WordType> region =
        clipRegion<WordType>(lv.width(), lv.height(), window);
    if (region.isEmpty) {
        if (finest) c.status[p] = 0;
        return;
    }

    // X-66: extract the window's ITERATION-INVARIANT words ONCE. X-68 measured
    // a mean of 4.29 iterations per point per level, every one of which was
    // re-reading the same eight previous-frame words per row. `region` is fixed
    // for the whole iteration, so one staging serves all of them.
    //
    // The buffer is a STACK local -- 2 048 B at the shipped N = 2 -- because
    // CLAUDE.md forbids a kernel allocating and this operation has no caller
    // scratch. `stageWindow` declines rather than overrunning it.
    BINCV_STAGE_LAP(setup);
    StagedWindow<LevelT::Bits, WordType> stagedWindow;
    TapCache<LevelT::Bits, WordType> tapCache;   // X-70; invalid until first use
    const bool staged = stageWindow(lv, region, stagedWindow);
    BINCV_STAGE_LAP(staging);

    // BIT-PARALLEL: the 2x2 matrix. X-84: from the STAGED window when there is one,
    // because `levelCovariance` reads exactly the planes `stageWindow` has just
    // finished reading and nothing else.
    const GradientCovariance a =
        staged ? stagedCovariance<LevelT::Bits, WordType>(stagedWindow, region.y1 - region.y0)
               : levelCovariance(lv, window);
    const double a11 = static_cast<double>(a.sumXX);
    const double a22 = static_cast<double>(a.sumYY);
    const double a12 = static_cast<double>(a.sumXY);
    const double det = a11 * a22 - a12 * a12;

    // LOSS RULE 2 -- a degenerate window. `det` is a difference of
    // products of exact popcounts, so it is 0 or at least 1 and the test
    // needs no epsilon (deviation (iv)).
    const double minEig = static_cast<double>(minEigenValue(a.sumXX, a.sumYY,
                                                                  a.sumXY));
    const double referenceMinEig = kLevelMinEigScale * minEig /
                                   static_cast<double>(c.winW * c.winH);
    if (det <= 0.0 || referenceMinEig < static_cast<double>(c.minEigThreshold)) {
        if (finest) c.status[p] = 0;
        return;
    }

    BINCV_STAGE_LAP(covariance);
    float nextX = c.nextPts[p].x - c.halfWinX;
    float nextY = c.nextPts[p].y - c.halfWinY;
    double prevDeltaX = 0.0;
    double prevDeltaY = 0.0;
    bool inRange = true;

    // The tap offset and the four weights live INSIDE the loop, and
    // deliberately do not survive it. They used to be declared here so
    // that the error term below could reuse them, which made `err` the
    // residual at the previous ITERATE rather than at the position
    // actually returned -- measured 134% high at c.maxIterations = 1, and
    // wrong by a whole half-step whenever the oscillation rule fired.
    // The reference recomputes them in a separate pass after the loop
    // (LKTrackerInvoker.cpp:222-259); so does this, below.
    for (int it = 0; it < c.maxIterations; ++it) {
#ifdef BINCV_LK_ITERATION_HISTOGRAM
        {
            const IterationTrace& tr = iterationTrace();
            if (tr.counts != nullptr) ++tr.counts[li * tr.pointCount + p];
        }
#endif
        const long long originX = floorToLL(nextX);
        const long long originY = floorToLL(nextY);

        // LOSS RULE 3 -- the estimate walked out of range mid-iteration.
        if (originX < -static_cast<long long>(c.winW) || originX >= levelWidth ||
            originY < -static_cast<long long>(c.winH) || originY >= levelHeight) {
            if (finest) c.status[p] = 0;
            inRange = false;
            break;
        }

        // FLOAT, once per iteration: split the displacement of the whole
        // window into an integer tap offset and a fraction. Every pixel of
        // the window shares both, which is exactly why the four weights
        // can leave the per-pixel loop.
        // THE DISPLACEMENT IS MEASURED FROM `prevX`, NOT FROM THE
        // INTEGER ANCHOR. `nextX - anchorX` looks equivalent and is not:
        // it differs by `frac(prevX)`, which is zero at level 0 for
        // integer keypoints and is NOT zero at any coarser level, where
        // `prevPt / 2^level` is fractional. Anchoring the window on the
        // grid moves which pixels the aperture covers (deviation (i));
        // it must not move what the residual is a residual OF, which is
        // `I(z)` against `J(z + d)` for `d` the full-precision flow.
        // Measured on the repo's real frame with prev == next: the wrong
        // spelling put a stationary point up to 1.4 px off through four
        // levels, because each level converged to `d - frac` and handed
        // twice that error to the next one down.
        const double offX = static_cast<double>(nextX) - static_cast<double>(prevX);
        const double offY = static_cast<double>(nextY) - static_cast<double>(prevY);
        const long long tapX = static_cast<long long>(std::floor(offX));
        const long long tapY = static_cast<long long>(std::floor(offY));
        const double fx = offX - static_cast<double>(tapX);
        const double fy = offY - static_cast<double>(tapY);
        const double w00 = (1.0 - fx) * (1.0 - fy);
        const double w01 = fx * (1.0 - fy);
        const double w10 = (1.0 - fx) * fy;
        const double w11 = fx * fy;

            // BIT-PARALLEL: ten exact integers, twenty popcounts per word --
        // reading the previous frame's eight words from `staged` when this
        // window could be staged (X-66). Bit-exact either way; the staged
        // path differs only in where the words come from.
        TapSums sumsX;
        TapSums sumsY;
#ifdef BINCV_LK_STAGE_TIMING
        ++impl::stageTiming().iterations;
#endif
        residualSums(lv, region, tapX, tapY, sumsX, sumsY,
                     staged ? &stagedWindow : nullptr, staged ? &tapCache : nullptr);

        const double b1 = sumsX.combine(w00, w01, w10, w11);
        const double b2 = sumsY.combine(w00, w01, w10, w11);

        // FLOAT, once per iteration: the 2x2 solve. The factor of 2 turns
        // the raw [-1, 0, 1] tap into a central difference; see UNITS.
        const double deltaX = kCentralDifferenceScale * (a12 * b2 - a22 * b1) / det;
        const double deltaY = kCentralDifferenceScale * (a12 * b1 - a11 * b2) / det;

        nextX += static_cast<float>(deltaX);
        nextY += static_cast<float>(deltaY);
        c.nextPts[p].x = nextX + c.halfWinX;
        c.nextPts[p].y = nextY + c.halfWinY;

        // TERMINATION 1 -- converged.
        if (deltaX * deltaX + deltaY * deltaY <= c.eps2) break;

        // TERMINATION 2 -- oscillation: this step almost exactly undoes
        // the last one. Back off by half a step and stop. The reference's
        // rule, thresholds included.
        if (it > 0 && std::fabs(deltaX + prevDeltaX) < 0.01 &&
            std::fabs(deltaY + prevDeltaY) < 0.01) {
            c.nextPts[p].x -= static_cast<float>(deltaX * 0.5);
            c.nextPts[p].y -= static_cast<float>(deltaY * 0.5);
            nextX = c.nextPts[p].x - c.halfWinX;
            nextY = c.nextPts[p].y - c.halfWinY;
            break;
        }
        prevDeltaX = deltaX;
        prevDeltaY = deltaY;
    }

    BINCV_STAGE_LAP(residual);
    // THE FINAL PASS, AND IT IS THE REFERENCE'S OWN SEPARATE PASS. Two
    // things happen here and both are ABOUT THE POSITION THAT IS
    // RETURNED, not about the last iterate: the range test is re-applied
    // to it, and -- only if it survives -- the error term is measured
    // there, from taps and weights recomputed from `c.nextPts[p]`.
    if (finest && c.status[p] != 0 && inRange) {
        const float finalX = c.nextPts[p].x - c.halfWinX;
        const float finalY = c.nextPts[p].y - c.halfWinY;
        const long long finalOriginX = floorToLL(finalX);
        const long long finalOriginY = floorToLL(finalY);
        if (finalOriginX < -static_cast<long long>(c.winW) || finalOriginX >= levelWidth ||
            finalOriginY < -static_cast<long long>(c.winH) || finalOriginY >= levelHeight) {
            // LOSS RULE 3, applied to the RETURNED estimate. The last
            // iteration's step can carry the point out of range after the
            // in-loop test has already passed, and `status` describes the
            // position the caller gets. The reference makes this same test
            // in this same place -- but only when `err` was requested,
            // which makes its `status` depend on whether the caller wanted
            // an error value. That quirk is not reproduced (deviation
            // (vii)); the test is unconditional here.
            c.status[p] = 0;
        } else if (c.err != nullptr) {
            const double offX = static_cast<double>(finalX) - static_cast<double>(prevX);
            const double offY = static_cast<double>(finalY) - static_cast<double>(prevY);
            const long long tapX = static_cast<long long>(std::floor(offX));
            const long long tapY = static_cast<long long>(std::floor(offY));
            const double fx = offX - static_cast<double>(tapX);
            const double fy = offY - static_cast<double>(tapY);
            c.err[p] = windowMeanAbsDiff(lv, region, tapX, tapY, (1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
                (1.0 - fx) * fy, fx * fy);
        }
    }
}

/// @brief Force the scalar path, for the tests. **INTERNAL, and not a tuning knob.**
///
/// It is how the batched path is held to BIT-EXACTNESS: the batch and `trackOnePoint`
/// are two spellings of the same arithmetic, and the only way to hold them equal is to
/// run both on the same input and compare. `slicedSignedSum`'s `UseNeon` exists for
/// exactly this reason and says exactly this (X-33).
///
/// A shipped build never touches it. It is a plain `bool` and therefore **not
/// thread-safe to flip while tracking** — tests flip it between calls.
inline bool& lkBatchEnabled() {
    static bool on = true;
    return on;
}

/// @brief Is this level type the N-bit one, with plane ARRAYS? **INTERNAL.**
/// @note `LKLevel<W>` and `LKLevelN<1, W>` both report `Bits == 1` and are not
///       interchangeable: only the second has `prev[]`, which is what the batch stages
///       from. `Bits` cannot tell them apart, so this does.
template <typename T>
struct IsLevelN : std::false_type {};
template <size_t N, typename WordType>
struct IsLevelN<LKLevelN<N, WordType>> : std::true_type {};

/// @brief Can this level type be batched AT ALL — asked at COMPILE time. **INTERNAL.**
/// @note The shipped 1/2/2/2 ladder (D-23) at `uint32_t`. This has to be a compile-time
///       question and not only a runtime one: `lkBatchResidual` is instantiated for the
///       level's depth, and a level at `N = 4` would fail to COMPILE a kernel it would
///       never have called. Whether the machine has AVX2 is the separate, runtime half.
template <typename T>
inline constexpr bool kBatchableLevel = IsLevelN<T>::value && (T::Bits == 1 || T::Bits == 2) &&
                                        sizeof(typename T::Word) == 4;

#if defined(BINCV_X86_LK_BATCH)

/// @brief One lane's tracking state. **INTERNAL** (X-79).
/// @note Everything here is per-KEYPOINT and scalar, and it stays scalar on purpose.
///       Only the ten window sums are vectorised; the 2x2 solve, the tap split, the
///       convergence test and the oscillation rule are computed in `double` exactly as
///       `trackOnePoint` computes them. **That is what makes the batch bit-exact rather
///       than merely close** -- there is no second spelling of the floating-point
///       arithmetic to drift from the first.
template <typename WordType>
struct LkLane {
    RegionWords<WordType> region{};
    size_t p = 0;
    size_t rows = 0;
    float prevX = 0.0f, prevY = 0.0f, nextX = 0.0f, nextY = 0.0f;
    double a11 = 0.0, a12 = 0.0, a22 = 0.0, det = 0.0;
    double prevDeltaX = 0.0, prevDeltaY = 0.0;
    double w00 = 0.0, w01 = 0.0, w10 = 0.0, w11 = 0.0;
    long long tapX = 0, tapY = 0;
    int it = 0;
    bool active = false;
    bool inRange = true;
    bool tapValid = false;
};

/// @brief Eight windows' worth of staged operands, in the ONE layout that makes the
///        batch worth having. **INTERNAL** (X-79).
///
/// `[row][plane][lane]`: eight keypoints' words at the same row and plane are eight
/// adjacent `uint32_t`, so a `__m256i` load fetches one word from each of eight
/// keypoints. [X-61](../../../EXPERIMENTS.md) tried the other arrangement and lost --
/// its vector arithmetic won on operation count and its **gathers** gave the win back.
/// The fix was never a better gather; it was arranging not to need one.
template <size_t N, typename WordType>
struct LkBatchArrays {
    WordType self[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType magX[kLkBatchMaxRows][N][kLkBatchLanes];   ///< masked to each lane's region
    WordType magY[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType signX[kLkBatchMaxRows][kLkBatchLanes];
    WordType signY[kLkBatchMaxRows][kLkBatchLanes];
    WordType t00[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType t01[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType t10[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType t11[kLkBatchMaxRows][N][kLkBatchLanes];
    WordType splitP[kLkBatchMaxRows][N][kLkBatchLanes];   ///< kernel scratch
    WordType splitN[kLkBatchMaxRows][N][kLkBatchLanes];
};

/// @brief Stage ONE lane's iteration-invariant words, and zero-pad it to `winRows`.
///        **INTERNAL** (X-79).
///
/// The scalar `stageWindow` written into a lane of the batch layout — same eight words
/// per row, same masking, scattered instead of packed.
///
/// **THE ZERO PADDING IS WHY CLIPPED WINDOWS NEED NO SPECIAL CASE.** Lanes in a batch
/// have different heights, and the vector kernel runs the TALLEST. A short lane's
/// remaining rows are given **zero magnitude**, and a zero magnitude contributes
/// exactly zero to every one of the ten sums — `popcount(V & 0)` is 0 whatever `V` is.
/// So a half-clipped window batches with a full one and the answer is unchanged, with
/// no per-lane row masking anywhere in the kernel.
///
/// Column clipping needs nothing at all: the region mask is applied to `magX`/`magY`
/// here, once, and every product is taken against a masked magnitude.
template <size_t N, typename WordType>
inline void stageLane(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                      LkBatchArrays<N, WordType>& b, size_t lane, size_t winRows) {
    const size_t words = minRowWords<WordType>(lv.prev[0].width);
    const WordType mask = lowBitsMask<WordType>(r.x1 - r.x0);
    size_t i = 0;
    for (size_t y = r.y0; y < r.y1; ++y, ++i) {
        for (size_t k = 0; k < N; ++k) {
            b.self[i][k][lane] = alignedWord<WordType>(lv.prev[k].row(y), words, r.x0);
            b.magX[i][k][lane] = static_cast<WordType>(
                alignedWord<WordType>(lv.dxMag[k].row(y), words, r.x0) & mask);
            b.magY[i][k][lane] = static_cast<WordType>(
                alignedWord<WordType>(lv.dyMag[k].row(y), words, r.x0) & mask);
        }
        b.signX[i][lane] = alignedWord<WordType>(lv.dxSign.row(y), words, r.x0);
        b.signY[i][lane] = alignedWord<WordType>(lv.dySign.row(y), words, r.x0);
    }
    for (; i < winRows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            b.magX[i][k][lane] = 0;
            b.magY[i][k][lane] = 0;
        }
    }
}

/// @brief Refresh ONE lane's four tap words per row at a new integer displacement.
///        **INTERNAL** (X-79). `RowReader::extractTaps`, scattered into the lane.
/// @note X-34's `+1`-tap-is-a-shift and X-35's interior fast path, both preserved:
///       the taps for row `i+1`'s upper pair are row `i`'s lower pair, and a window
///       narrower than a word gets its `+1` column by shifting rather than re-reading.
template <size_t N, typename WordType>
inline void extractLaneTaps(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                            long long tapX, long long tapY, LkBatchArrays<N, WordType>& b,
                            size_t lane) {
    const size_t words = minRowWords<WordType>(lv.prev[0].width);
    const size_t width = r.x1 - r.x0;
    const bool tapIsShift = width < bitsPerWord<WordType>();
    const long long srcX = static_cast<long long>(r.x0) + tapX;
    const long long lastCol = static_cast<long long>(lv.next[0].width) - 1;
    const bool colsInside = srcX >= 0 && srcX + static_cast<long long>(width) <= lastCol;
    const size_t height = lv.next[0].height;
    // ROW i's LOWER TAP IS ROW i+1's UPPER TAP -- they name the same level row at the
    // same displacement. Reading it twice is what `RowReader` does, and there it costs
    // nothing worth naming because the two reads are adjacent in one row's work. Here
    // the window is walked once per iteration for eight lanes, so halving the reads of
    // `lv.next` halves the scalar half of the batch's inner loop.
    //
    // Sound because the two spellings agree wherever both apply: `alignedWord` is
    // X-35's interior fast path for exactly the case `displacedRow(...).word(0)` would
    // compute the same bits more slowly, and outside it both sides take `displacedRow`.
    const auto readRow = [&](long long sy, size_t k, long long sx) -> WordType {
        if (colsInside && sy >= 0 && sy < static_cast<long long>(height) && sx == srcX) {
            return alignedWord<WordType>(lv.next[k].row(static_cast<size_t>(sy)), words,
                                         static_cast<size_t>(sx));
        }
        return displacedRow<WordType>(lv.next[k], sy, sx).word(0);
    };

    WordType carry[N];
    WordType carryShift[N];
    bool haveCarry = false;
    size_t i = 0;
    for (size_t y = r.y0; y < r.y1; ++y, ++i) {
        const long long srcY = static_cast<long long>(y) + tapY;
        for (size_t k = 0; k < N; ++k) {
            const WordType upper = haveCarry ? carry[k] : readRow(srcY, k, srcX);
            const WordType lower = readRow(srcY + 1, k, srcX);
            carry[k] = lower;
            b.t00[i][k][lane] = upper;
            b.t10[i][k][lane] = lower;
            if (tapIsShift) {
                b.t01[i][k][lane] = static_cast<WordType>(upper >> 1);
                b.t11[i][k][lane] = static_cast<WordType>(lower >> 1);
            } else {
                const WordType upperR =
                    haveCarry ? carryShift[k] : readRow(srcY, k, srcX + 1);
                const WordType lowerR = readRow(srcY + 1, k, srcX + 1);
                carryShift[k] = lowerR;
                b.t01[i][k][lane] = upperR;
                b.t11[i][k][lane] = lowerR;
            }
        }
        haveCarry = true;
    }
}

/// @brief The 2x2 covariance from ONE LANE of an already-staged batch. **INTERNAL**
///        (X-84). `stagedCovariance`'s reason, in the batch's `[row][plane][lane]`
///        layout: the words are already here, so reading the level's planes a second
///        time is pure repetition.
template <size_t N, typename WordType>
inline GradientCovariance stagedCovarianceLane(const LkBatchArrays<N, WordType>& b,
                                               size_t lane, size_t rows) {
    BitSlicedPairCounts<N> total;
    for (size_t i = 0; i < rows; ++i) {
        const WordType selector =
            static_cast<WordType>(b.signX[i][lane] ^ b.signY[i][lane]);
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = x; y < N; ++y) {
                total.xx[x][y] += popcountWord<WordType>(
                    static_cast<WordType>(b.magX[i][x][lane] & b.magX[i][y][lane]));
                total.yy[x][y] += popcountWord<WordType>(
                    static_cast<WordType>(b.magY[i][x][lane] & b.magY[i][y][lane]));
            }
        }
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < N; ++y) {
                const WordType both =
                    static_cast<WordType>(b.magX[i][x][lane] & b.magY[i][y][lane]);
                total.xyTotal[x][y] += popcountWord<WordType>(both);
                total.xySet[x][y] +=
                    popcountWord<WordType>(static_cast<WordType>(both & selector));
            }
        }
    }
    return combineBitSlicedPairs<N>(total);
}

/// @brief Track a RANGE of points through one level, eight at a time, with **lane
///        refill**. **INTERNAL** (X-79, E-36).
///
/// **THE REFILL IS THE DESIGN, NOT A REFINEMENT ON TOP OF IT.**
/// [X-78](../../../EXPERIMENTS.md) counted the iteration distribution before any of
/// this was written: **72.6% of point-levels finish in two iterations or fewer, and a
/// 3.6% tail runs the cap of twenty.** Eight lanes in lockstep cost the MAXIMUM of
/// eight draws from that distribution, which measured **5.20 against a mean of 3.24 --
/// 39.9% of every lane slot wasted.** A naive lockstep batch would have turned a 3.1x
/// kernel into 1.31x on `track`.
///
/// So a lane that finishes takes **the next untracked point** instead of idling. That
/// costs one re-staging, which is the same staging the scalar path does for every
/// point anyway: the work is not new, it happens at a different time.
///
/// @param pFirst,pEnd The half-open range of points this call owns. Splitting the
///        ARRAY is how the operation threads (T5.1) — each range refills from its own
///        cursor and writes only its own points' outputs.
///
/// @note **Points the batch cannot hold are not skipped, they are tracked here**, by
///       `trackOnePoint`, at the moment the refill reaches them. A window wider than a
///       word or taller than `kLkBatchMaxRows` is rare (the shipped window is 31x31)
///       and correct either way.
template <size_t N, typename WordType>
inline void trackRangeBatched(const LKLevelN<N, WordType>& lv, size_t li, const LKContext& c,
                              bool finest, float scale, long long levelWidth,
                              long long levelHeight, double kLevelMinEigScale,
                              size_t pFirst, size_t pEnd) {
    LkBatchArrays<N, WordType> b;
    // Zeroed once, not per refill. Padded rows keep whatever a previous occupant of
    // the lane left in the TAP arrays -- which is harmless, because their magnitude is
    // zero -- but reading never-written memory is not, so the first pass over each
    // array must find it defined.
    std::memset(&b, 0, sizeof(b));

    LkLane<WordType> lane[kLkBatchLanes];
    const size_t winRows =
        static_cast<size_t>(c.winH) < kLkBatchMaxRows ? static_cast<size_t>(c.winH)
                                                      : kLkBatchMaxRows;
    size_t cursor = pFirst;

    // The tail of `trackOnePoint`: the RETURNED estimate's range test, and the error
    // term measured there. Same place, same conditions, same deviations.
    const auto finishLane = [&](size_t L) {
        LkLane<WordType>& s = lane[L];
        if (finest && c.status[s.p] != 0 && s.inRange) {
            const float finalX = c.nextPts[s.p].x - c.halfWinX;
            const float finalY = c.nextPts[s.p].y - c.halfWinY;
            const long long fx0 = floorToLL(finalX);
            const long long fy0 = floorToLL(finalY);
            if (fx0 < -static_cast<long long>(c.winW) || fx0 >= levelWidth ||
                fy0 < -static_cast<long long>(c.winH) || fy0 >= levelHeight) {
                c.status[s.p] = 0;
            } else if (c.err != nullptr) {
                const double offX = static_cast<double>(finalX) - static_cast<double>(s.prevX);
                const double offY = static_cast<double>(finalY) - static_cast<double>(s.prevY);
                const long long tx = static_cast<long long>(std::floor(offX));
                const long long ty = static_cast<long long>(std::floor(offY));
                const double fx = offX - static_cast<double>(tx);
                const double fy = offY - static_cast<double>(ty);
                c.err[s.p] = windowMeanAbsDiff(lv, s.region, tx, ty, (1.0 - fx) * (1.0 - fy),
                                               fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
            }
        }
        s.active = false;
    };

    // Everything `trackOnePoint` does BEFORE its iteration loop, for the next point
    // that qualifies. Returns false when the range is exhausted.
    const auto refill = [&](size_t L) {
        while (cursor < pEnd) {
            const size_t p = cursor++;
            if (li > entryLevelFor(c, p)) continue;
            const float prevX = c.prevPts[p].x * scale - c.halfWinX;
            const float prevY = c.prevPts[p].y * scale - c.halfWinY;
            const long long anchorX = floorToLL(prevX);
            const long long anchorY = floorToLL(prevY);
            if (anchorX < -static_cast<long long>(c.winW) || anchorX >= levelWidth ||
                anchorY < -static_cast<long long>(c.winH) || anchorY >= levelHeight) {
                if (finest) c.status[p] = 0;
                continue;
            }
            const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), c.winW,
                              c.winH);
            const RegionWords<WordType> region =
                clipRegion<WordType>(lv.width(), lv.height(), window);
            if (region.isEmpty) {
                if (finest) c.status[p] = 0;
                continue;
            }
            const size_t width = region.x1 - region.x0;
            const size_t rows = region.y1 - region.y0;
            if (width == 0 || width > bitsPerWord<WordType>() || rows > kLkBatchMaxRows) {
                trackOnePoint<LKLevelN<N, WordType>, WordType>(
                    lv, li, c, p, finest, scale, levelWidth, levelHeight, kLevelMinEigScale);
                continue;
            }
            // X-84: stage FIRST, then take the covariance off the staged lane. The
            // order used to be the other way round because the covariance did not need
            // the staging; now it does, and a rejected point pays one staging it did
            // not use -- against a whole second traversal of the level's planes for
            // every point that IS accepted.
            stageLane<N, WordType>(lv, region, b, L, winRows);
            const GradientCovariance a =
                stagedCovarianceLane<N, WordType>(b, L, rows);
            const double a11 = static_cast<double>(a.sumXX);
            const double a22 = static_cast<double>(a.sumYY);
            const double a12 = static_cast<double>(a.sumXY);
            const double det = a11 * a22 - a12 * a12;
            const double minEig =
                static_cast<double>(minEigenValue(a.sumXX, a.sumYY, a.sumXY));
            const double referenceMinEig =
                kLevelMinEigScale * minEig / static_cast<double>(c.winW * c.winH);
            if (det <= 0.0 || referenceMinEig < static_cast<double>(c.minEigThreshold)) {
                if (finest) c.status[p] = 0;
                continue;
            }
            LkLane<WordType>& s = lane[L];
            s.region = region;
            s.p = p;
            s.rows = rows;
            s.prevX = prevX;
            s.prevY = prevY;
            s.nextX = c.nextPts[p].x - c.halfWinX;
            s.nextY = c.nextPts[p].y - c.halfWinY;
            s.a11 = a11;
            s.a12 = a12;
            s.a22 = a22;
            s.det = det;
            s.prevDeltaX = 0.0;
            s.prevDeltaY = 0.0;
            s.it = 0;
            s.active = true;
            s.inRange = true;
            s.tapValid = false;
            return;
        }
    };

    int32_t outX[5 * kLkBatchLanes];
    int32_t outY[5 * kLkBatchLanes];

    for (;;) {
        bool any = false;
        for (size_t L = 0; L < kLkBatchLanes; ++L) {
            if (!lane[L].active) refill(L);
            any = any || lane[L].active;
        }
        if (!any) break;

        // PER-LANE, SCALAR: the range test and the tap split, both of which are
        // per-keypoint float arithmetic and neither of which is worth vectorising.
        size_t rowsMax = 0;
        bool lost = false;
        for (size_t L = 0; L < kLkBatchLanes; ++L) {
            LkLane<WordType>& s = lane[L];
            if (!s.active) continue;
            const long long originX = floorToLL(s.nextX);
            const long long originY = floorToLL(s.nextY);
            // LOSS RULE 3 -- the estimate walked out of range mid-iteration.
            if (originX < -static_cast<long long>(c.winW) || originX >= levelWidth ||
                originY < -static_cast<long long>(c.winH) || originY >= levelHeight) {
                if (finest) c.status[s.p] = 0;
                s.inRange = false;
                finishLane(L);
                lost = true;
                continue;
            }
            // THE DISPLACEMENT IS MEASURED FROM `prevX`, NOT FROM THE INTEGER ANCHOR --
            // see `trackOnePoint`, where getting this wrong put a stationary point 1.4
            // px off through four levels.
            const double offX = static_cast<double>(s.nextX) - static_cast<double>(s.prevX);
            const double offY = static_cast<double>(s.nextY) - static_cast<double>(s.prevY);
            const long long tapX = static_cast<long long>(std::floor(offX));
            const long long tapY = static_cast<long long>(std::floor(offY));
            const double fx = offX - static_cast<double>(tapX);
            const double fy = offY - static_cast<double>(tapY);
            s.w00 = (1.0 - fx) * (1.0 - fy);
            s.w01 = fx * (1.0 - fy);
            s.w10 = (1.0 - fx) * fy;
            s.w11 = fx * fy;
            // X-70's tap cache, per lane: the iteration SHRINKS the displacement, so
            // once the estimate settles inside a pixel the integer part stops moving
            // and the same four words would be re-extracted every remaining iteration.
            if (!s.tapValid || s.tapX != tapX || s.tapY != tapY) {
                extractLaneTaps<N, WordType>(lv, s.region, tapX, tapY, b, L);
                s.tapX = tapX;
                s.tapY = tapY;
                s.tapValid = true;
            }
            if (s.rows > rowsMax) rowsMax = s.rows;
        }
        // A lane lost here would idle through the kernel call. Refill first -- the
        // whole point of X-78 is that an idle lane is the expensive thing.
        if (lost) continue;

        lkBatchResidual<N>(&b.self[0][0][0], &b.t00[0][0][0], &b.t01[0][0][0],
                           &b.t10[0][0][0], &b.t11[0][0][0], &b.magX[0][0][0],
                           &b.signX[0][0], &b.magY[0][0][0], &b.signY[0][0], rowsMax,
                           &b.splitP[0][0][0], &b.splitN[0][0][0], outX, outY);

        // PER-LANE, SCALAR AGAIN: the 2x2 solve and both termination rules, in
        // `double`, exactly as `trackOnePoint` computes them.
        for (size_t L = 0; L < kLkBatchLanes; ++L) {
            LkLane<WordType>& s = lane[L];
            if (!s.active) continue;
            TapSums sumsX, sumsY;
            sumsX.t00 = outX[0 * kLkBatchLanes + L];
            sumsX.t01 = outX[1 * kLkBatchLanes + L];
            sumsX.t10 = outX[2 * kLkBatchLanes + L];
            sumsX.t11 = outX[3 * kLkBatchLanes + L];
            sumsX.self = outX[4 * kLkBatchLanes + L];
            sumsY.t00 = outY[0 * kLkBatchLanes + L];
            sumsY.t01 = outY[1 * kLkBatchLanes + L];
            sumsY.t10 = outY[2 * kLkBatchLanes + L];
            sumsY.t11 = outY[3 * kLkBatchLanes + L];
            sumsY.self = outY[4 * kLkBatchLanes + L];

            const double b1 = sumsX.combine(s.w00, s.w01, s.w10, s.w11);
            const double b2 = sumsY.combine(s.w00, s.w01, s.w10, s.w11);
            const double deltaX =
                kCentralDifferenceScale * (s.a12 * b2 - s.a22 * b1) / s.det;
            const double deltaY =
                kCentralDifferenceScale * (s.a12 * b1 - s.a11 * b2) / s.det;
            s.nextX += static_cast<float>(deltaX);
            s.nextY += static_cast<float>(deltaY);
            c.nextPts[s.p].x = s.nextX + c.halfWinX;
            c.nextPts[s.p].y = s.nextY + c.halfWinY;
            const int it = s.it;
            ++s.it;

            // TERMINATION 1 -- converged.
            if (deltaX * deltaX + deltaY * deltaY <= c.eps2) {
                finishLane(L);
                continue;
            }
            // TERMINATION 2 -- oscillation: this step almost exactly undoes the last.
            if (it > 0 && std::fabs(deltaX + s.prevDeltaX) < 0.01 &&
                std::fabs(deltaY + s.prevDeltaY) < 0.01) {
                c.nextPts[s.p].x -= static_cast<float>(deltaX * 0.5);
                c.nextPts[s.p].y -= static_cast<float>(deltaY * 0.5);
                s.nextX = c.nextPts[s.p].x - c.halfWinX;
                s.nextY = c.nextPts[s.p].y - c.halfWinY;
                finishLane(L);
                continue;
            }
            s.prevDeltaX = deltaX;
            s.prevDeltaY = deltaY;
            // TERMINATION 3 -- the iteration cap.
            if (s.it >= c.maxIterations) finishLane(L);
        }
    }
}

#endif  // BINCV_X86_LK_BATCH

/// @brief Track every point through ONE pyramid level. **INTERNAL, and the whole
///        of the tracker's per-level logic.**
/// @tparam LevelT `LKLevel<W>` or `LKLevelN<N, W>` -- the body is written against
///         `width()`, `height()`, `Bits` and the two dispatched helpers above, so
///         it is identical for both and cannot drift between them.
/// @note `Bits` enters in exactly one place: the `minEigThreshold` conversion.
///       Everything else about the level's depth is inside `levelCovariance` and
///       `residualSums`.
template <typename LevelT, typename WordType = typename LevelT::Word>
inline void trackOneLevel(const LevelT& lv, size_t li, const LKContext& c) {
    // The one place the level's depth reaches the tracking logic. See
    // referenceMinEigScale for why holding the 1-bit constant here would silently
    // reject more points at higher N and confound E-7's accuracy curve.
    constexpr double kLevelMinEigScale = referenceMinEigScale(LevelT::Bits);

    const bool finest = (li == 0);
    const float scale = 1.0f / static_cast<float>(1u << li);

    checkLevelPlanes(lv);

    // PASS 1 -- propagate every point's estimate into this level's
    // coordinates, before any point is tracked. The reference does this in its
    // own first loop, and it applies to skipped points too.
    //
    // A point's ENTRY level is where the propagation starts rather than continues.
    // Under the shipped policy that is the coarsest usable level for every point
    // and this is exactly the old two-branch loop; under DeepestFitting it is per
    // point, and a point is left alone entirely above its own entry level -- so
    // `nextPts` must not be doubled there either, which is why the skip is in this
    // pass and not only in the tracking one.
    for (size_t p = 0; p < c.pointCount; ++p) {
        const size_t entry = entryLevelFor(c, p);
        if (li > entry) continue;
        if (li == entry) {
            c.nextPts[p].x = c.prevPts[p].x * scale;
            c.nextPts[p].y = c.prevPts[p].y * scale;
        } else {
            c.nextPts[p].x *= 2.0f;
            c.nextPts[p].y *= 2.0f;
        }
    }

    const long long levelWidth = static_cast<long long>(lv.width());
    const long long levelHeight = static_cast<long long>(lv.height());

    // PASS 2 -- track.
    // T5.1: KEYPOINTS ARE INDEPENDENT, AND THAT IS WHY THIS SPLITS SAFELY.
    // Each iteration writes only `nextPts[p]`, `status[p]` and `err[p]` -- distinct
    // indices -- and reads only the level's views, which are const. Nothing is
    // accumulated across points, so a split cannot move a flow vector. That is why
    // X-65 measured 2.60x by splitting the ARRAY with no library change at all; this
    // only makes it askable.
    //
    // Serial unless a caller installs a backend (core/parallel.hpp). On a core-only
    // build this is exactly the loop it replaces.
#if defined(BINCV_X86_LK_BATCH)
    // X-79 / E-36: EIGHT KEYPOINTS PER AVX2 REGISTER, WITH LANE REFILL.
    // Selected at run time, so nothing about the library's baseline ISA changes and a
    // machine without AVX2 takes the loop below. The split is by RANGE rather than by
    // point, because each range carries its own refill cursor -- and by at most one
    // range per thread, since a short final batch is the one thing a range boundary
    // costs and X-78 measured idle lanes to be expensive.
    if constexpr (kBatchableLevel<LevelT>) {
        if (hasLkBatch() && lkBatchEnabled()) {
            size_t groups = c.pointCount / kLkBatchLanes;
            const size_t threads = static_cast<size_t>(getNumThreads());
            if (groups > threads) groups = threads;
            if (groups < 1) groups = 1;
            parallelFor(groups, [&](size_t g) {
                trackRangeBatched<LevelT::Bits, WordType>(
                    lv, li, c, finest, scale, levelWidth, levelHeight, kLevelMinEigScale,
                    c.pointCount * g / groups, c.pointCount * (g + 1) / groups);
            });
            return;
        }
    }
#endif
    parallelFor(c.pointCount, [&](size_t p) {
        trackOnePoint<LevelT, WordType>(lv, li, c, p, finest, scale, levelWidth,
                                        levelHeight, kLevelMinEigScale);
    });
}

/// @brief Shared prologue: argument checks, and the "every entry is written"
///        contract. **INTERNAL.**
/// @return false when the call is already complete -- no points, or no levels.
inline bool lkPrologue(size_t levelCount, const Point2f* prevPts, Point2f* nextPts,
                       uint8_t* status, float* err, size_t pointCount, const LKParams& params,
                       LKContext& c) {
    if (pointCount == 0) return false;
    BINCV_ASSERT(prevPts != nullptr && nextPts != nullptr && status != nullptr,
                 "opticalFlow: prevPts, nextPts and status must be non-null");
    // PASS 1 writes `nextPts[p]` before PASS 2 reads `prevPts[p]`, so an in-place
    // call is not merely inexact, it tracks from the wrong anchor -- measured at up
    // to 23 px of divergence on a three-level frontend. Every other kernel in the
    // library carries an explicit D-11 predicate; this is that predicate, on the
    // destination array rather than on a view.
    BINCV_ASSERT(byteRangesDisjoint(prevPts, pointCount * sizeof(Point2f), nextPts,
                                    pointCount * sizeof(Point2f)),
                 "opticalFlow: nextPts must not overlap prevPts");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "opticalFlow: the window must be more than 2 pixels on a side");

    c.prevPts = prevPts;
    c.nextPts = nextPts;
    c.status = status;
    c.err = err;
    c.pointCount = pointCount;
    c.winW = params.winWidth;
    c.winH = params.winHeight;
    c.halfWinX = static_cast<float>(c.winW - 1) * 0.5f;
    c.halfWinY = static_cast<float>(c.winH - 1) * 0.5f;

    // The reference clamps both criteria before use; so does this, for the same
    // reason -- they arrive from a YAML file.
    c.maxIterations = params.maxIterations;
    if (c.maxIterations < 0) c.maxIterations = 0;
    if (c.maxIterations > 100) c.maxIterations = 100;
    float eps = params.epsilon;
    if (eps < 0.0f) eps = 0.0f;
    if (eps > 10.0f) eps = 10.0f;
    c.eps2 = static_cast<double>(eps) * static_cast<double>(eps);
    c.minEigThreshold = static_cast<double>(params.minEigThreshold);
    c.entryLevel = params.entryLevel;

    // Written BEFORE the degenerate exit below, not after it: "every entry is
    // written" is the documented contract, and a caller that reads `status` after
    // a zero-level call must not be reading its own uninitialised buffer.
    for (size_t i = 0; i < pointCount; ++i) {
        status[i] = 1;
        if (err != nullptr) err[i] = 0.0f;
    }

    // Degenerate but legal, and a VALUE rather than an error (ARCHITECTURE 5.3):
    // with no levels there is nothing to track on, so every point is lost and its
    // last estimate is the point itself.
    if (levelCount == 0) {
        for (size_t i = 0; i < pointCount; ++i) {
            nextPts[i] = prevPts[i];
            status[i] = 0;
        }
        return false;
    }
    return true;
}

/// @brief THE REFERENCE'S PYRAMID CAP, REPRODUCED (deviation (vi)).
///        **INTERNAL.**
///
/// `buildOpticalFlowPyramid` refuses to build a level that is not strictly larger
/// than the window -- `if (sz.width <= winSize.width || sz.height <=
/// winSize.height) { maxLevel = level; break; }` -- and truncates `maxLevel` to
/// what it built. binCV cannot decline to BUILD a level, because the caller owns
/// the pyramid (D-5), so it declines to USE one: levels are consumed as a prefix,
/// coarsest usable first, and anything at or below the window size is ignored. It
/// matters MORE here than in the reference, because binCV clips where the
/// reference pads: on a level no larger than the window every point's window
/// covers nearly the whole level, so every point gets nearly the same `A` and `b`
/// -- one estimate for the entire image, multiplied by 2^level on the way down.
/// Level 0 is always used, whatever its size; it is the frame.
///
/// @note Written against `widthAt`/`heightAt` callables so that the array form and
///       the heterogeneous ladder share ONE copy of the rule.
template <typename DimFn>
inline size_t usableLevelCount(size_t levelCount, int winW, int winH, DimFn dims) {
    size_t usable = 1;
    while (usable < levelCount && dims(usable).width > static_cast<size_t>(winW) &&
           dims(usable).height > static_cast<size_t>(winH)) {
        ++usable;
    }
    return usable;
}
} // namespace impl

/// @brief Pyramidal Lucas-Kanade tracking of sparse keypoints between two binary
///        frames. **API TIER 2** -- `cv::calcOpticalFlowPyrLK`'s role and call
///        shape, deliberately different numerics, NOT bit-exact against OpenCV.
///
/// @tparam WordType The planes' word type (D-1).
/// @param levels `levelCount` per-level view bundles, **level 0 (the finest)
///        first**. Every level's six planes must share that level's dimensions.
/// @param levelCount Number of pyramid levels. `seal_params.yaml`'s
///        `lk_max_level: 3` means four levels. **Levels at or below the window
///        size are ignored** (deviation (vi)), so passing more of them than the
///        frame can carry is harmless rather than silently wrong; 0 is legal and
///        loses every point.
/// @param prevPts `pointCount` keypoints in LEVEL-0 coordinates. Read only.
/// @param nextPts Out: `pointCount` tracked positions in LEVEL-0 coordinates.
///        Written for every point, whether or not it was tracked -- a lost point's
///        entry is its last estimate, exactly as in the reference, and `status` is
///        the only thing that says which is which.
/// @param status Out: 1 if the point was tracked, 0 if it was lost. Every entry is
///        written -- including on the degenerate `levelCount == 0` call, where
///        every point is lost and `nextPts` is a copy of `prevPts`.
/// @param err Out, OPTIONAL (may be null): the mean absolute residual **at the
///        position that was returned** -- taps and weights recomputed from the
///        final `nextPts`, not left over from the last iterate -- over the clipped
///        window at level 0, in {0, 1} intensity units. Zero for a lost point. See
///        `impl::windowMeanAbsDiff` for the denominator, which is the clipped
///        pixel count and not the window area.
/// @param pointCount Number of keypoints.
/// @param params Window, iteration limit, epsilon and minimum-eigenvalue
///        threshold; defaulted to `seal_params.yaml`.
///
/// @note **No allocation, no scratch, no throw.** The four arrays are the
///       caller's and nothing else is needed; tests/test_opticalflow.cpp counts
///       `operator new` across this call -- the plain and the over-aligned forms --
///       and requires zero.
/// @note **Every level of THIS overload is one bit deep.** For deeper levels see
///       the `LKLevelN<N, WordType>` overload below (a ladder all at one depth)
///       and `LKLevels<WordType, LevelBits...>` (a ladder of MIXED depths, which
///       is the form E-7 / T4.1 needs). All three run the same body; the depth
///       reaches it in exactly two places, `impl::referenceMinEigScale` and
///       `impl::levelCovariance`.
/// @note **Windows clip at the frame edge** (deviation (ii)) and next-frame taps
///       outside the frame replicate (deviation (iii)). Neither is the reference's
///       border, and both are consequences of declining its `winSize`-wide padded
///       copy of every level.
/// @note Points are propagated coarse-to-fine even when they are skipped at a
///       level, and `status` is written only from level 0 -- the reference's rule,
///       reproduced deliberately, not incidentally.
template <typename WordType>
inline void calcOpticalFlowPyrLK(const LKLevel<WordType>* levels, size_t levelCount,
                                 const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                 float* err, size_t pointCount,
                                 const LKParams& params = LKParams()) {
    impl::LKContext c;
    if (!impl::lkPrologue(levelCount, prevPts, nextPts, status, err, pointCount, params, c)) return;
    BINCV_ASSERT(levels != nullptr, "opticalFlow: levels must be non-null");
    c.usableLevels = impl::usableLevelCount(levelCount, c.winW, c.winH, [&](size_t i) {
        return impl::LevelDims{levels[i].width(), levels[i].height()};
    });
    if (c.usableLevels > impl::kMaxLevels) c.usableLevels = impl::kMaxLevels;
    for (size_t i = 0; i < c.usableLevels; ++i) {
        c.dims[i] = impl::LevelDims{levels[i].width(), levels[i].height()};
    }
    for (size_t li = c.usableLevels; li-- > 0;) impl::trackOneLevel(levels[li], li, c);
}


/// @brief Pyramidal Lucas-Kanade over a ladder of levels that are all the SAME
///        depth `N`. **API TIER 2.**
/// @param levels `levelCount` levels, **LEVEL 0 FIRST**, each `N` bits per pixel.
/// @note Identical in every respect to the 1-bit entry point above -- same
///       contracts, same deviations, same loss rules, same `err` denominator --
///       because it runs the same body. The two differences are consequences of
///       the depth and are documented where they live:
///       `impl::referenceMinEigScale` (the `minEigThreshold` conversion, which
///       is what keeps the threshold meaning one thing across depths) and
///       `impl::windowMeanAbsDiff` (the `err` term, the one piece of THE BOUNDARY
///       that does not survive `N > 1`).
/// @note `N == 1` here is the same computation as `LKLevel` above, through the
///       generic code path rather than the hand-written one, and
///       tests/test_opticalflow.cpp requires the two to agree exactly.
template <size_t N, typename WordType>
inline void calcOpticalFlowPyrLK(const LKLevelN<N, WordType>* levels, size_t levelCount,
                                 const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                 float* err, size_t pointCount,
                                 const LKParams& params = LKParams()) {
    impl::LKContext c;
    if (!impl::lkPrologue(levelCount, prevPts, nextPts, status, err, pointCount, params, c)) return;
    BINCV_ASSERT(levels != nullptr, "opticalFlow: levels must be non-null");
    c.usableLevels = impl::usableLevelCount(levelCount, c.winW, c.winH, [&](size_t i) {
        return impl::LevelDims{levels[i].width(), levels[i].height()};
    });
    if (c.usableLevels > impl::kMaxLevels) c.usableLevels = impl::kMaxLevels;
    for (size_t i = 0; i < c.usableLevels; ++i) {
        c.dims[i] = impl::LevelDims{levels[i].width(), levels[i].height()};
    }
    for (size_t li = c.usableLevels; li-- > 0;) impl::trackOneLevel(levels[li], li, c);
}

/// @brief A tracking ladder whose levels have **DIFFERENT bit depths**, level 0
///        first. **API TIER 2.**
/// @tparam LevelBits One depth per level -- `LKLevels<uint32_t, 1, 3, 4, 5>` is
///         the ladder ARCHITECTURE 7.2 measured on the reference pipeline.
///
/// @note **The depths are a template parameter list, not a runtime vector, and
///       that is `Pyramid`'s decision rather than a new one.** `QuantMat` is
///       templated on N, so levels of different depths have different TYPES; a
///       runtime container of them would need type erasure, and the `N^2` inner
///       loops of `impl::slicedSignedSum` would become runtime-bounded, which
///       would confound exactly the ns/pixel axis E-7 has to report. This mirrors
///       `Pyramid<WordType, LevelBits...>` one-for-one so that a pyramid and the
///       tracker that reads it are declared the same way.
/// @note Holds VIEWS, not containers (D-5), so it owns nothing and allocates
///       nothing. The caller owns the two pyramids and the derivative planes.
template <typename WordType, size_t... LevelBits>
struct LKLevels;

template <typename WordType, size_t N0>
struct LKLevels<WordType, N0> {
    static constexpr size_t Levels = 1;
    LKLevelN<N0, WordType> level;

    template <size_t I>
    LKLevelN<N0, WordType>& get() {
        static_assert(I == 0, "LK ladder level index out of range");
        return level;
    }
    template <size_t I>
    const LKLevelN<N0, WordType>& get() const {
        static_assert(I == 0, "LK ladder level index out of range");
        return level;
    }
    impl::LevelDims dimsAt(size_t i) const {
        (void)i;
        return impl::LevelDims{level.width(), level.height()};
    }
    template <typename Fn>
    void visitCoarseToFine(size_t usable, Fn& f, size_t index) const {
        (void)usable;
        f(level, index);
    }
};

template <typename WordType, size_t N0, size_t N1, size_t... Rest>
struct LKLevels<WordType, N0, N1, Rest...> {
    static constexpr size_t Levels = 2 + sizeof...(Rest);
    LKLevelN<N0, WordType> level;
    LKLevels<WordType, N1, Rest...> rest;

    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return level;
        } else {
            return rest.template get<I - 1>();
        }
    }
    template <size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return level;
        } else {
            return rest.template get<I - 1>();
        }
    }
    impl::LevelDims dimsAt(size_t i) const {
        return i == 0 ? impl::LevelDims{level.width(), level.height()} : rest.dimsAt(i - 1);
    }

    /// @note Recurses to the coarsest USABLE level and calls `f` on the way back
    ///       out, so the visit order is coarse-to-fine -- the tracker's order --
    ///       with no array of pointers and no type erasure.
    template <typename Fn>
    void visitCoarseToFine(size_t usable, Fn& f, size_t index) const {
        if (index + 1 < usable) rest.visitCoarseToFine(usable, f, index + 1);
        f(level, index);
    }
};

/// @brief Pyramidal Lucas-Kanade over a ladder of **mixed-depth** levels.
///        **API TIER 2.**
/// @note Same contracts, deviations and loss rules as the two entry points above;
///       each level runs the same body at its own depth. This is the form E-7
///       needs, because the question it asks -- how many bits does EACH level need
///       -- only has an answer a mixed ladder can express.
template <typename WordType, size_t... LevelBits>
inline void calcOpticalFlowPyrLK(const LKLevels<WordType, LevelBits...>& levels,
                                 const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                 float* err, size_t pointCount,
                                 const LKParams& params = LKParams()) {
    impl::LKContext c;
    if (!impl::lkPrologue(sizeof...(LevelBits), prevPts, nextPts, status, err, pointCount, params,
                          c)) {
        return;
    }
    c.usableLevels = impl::usableLevelCount(sizeof...(LevelBits), c.winW, c.winH,
                                            [&](size_t i) { return levels.dimsAt(i); });
    if (c.usableLevels > impl::kMaxLevels) c.usableLevels = impl::kMaxLevels;
    for (size_t i = 0; i < c.usableLevels; ++i) c.dims[i] = levels.dimsAt(i);
    auto visit = [&](const auto& lv, size_t li) { impl::trackOneLevel(lv, li, c); };
    levels.visitCoarseToFine(c.usableLevels, visit, 0);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
