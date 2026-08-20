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

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../quantMat.hpp"
#include "../impl/kernel_util.hpp"
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
struct LKParams {
    int winWidth = 31;     ///< window width in pixels, > 2
    int winHeight = 31;    ///< window height in pixels, > 2
    int maxIterations = 20;      ///< per level, clamped to [0, 100] as the reference clamps
    float epsilon = 0.03f;       ///< converged when |delta| <= this, in pixels
    float minEigThreshold = 0.001f;  ///< IN THE REFERENCE'S UNITS; see UNITS above
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
template <size_t N, typename WordType>
inline long long slicedSignedSum(const WordType (&maskedMag)[N], WordType sign,
                                 const WordType (&val)[N]) {
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
                         long long tapX, long long tapY, TapSums& sumsX, TapSums& sumsY) {
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
template <size_t N, typename WordType>
inline void residualSums(const LKLevelN<N, WordType>& lv, const RegionWords<WordType>& r,
                         long long tapX, long long tapY, TapSums& sumsX, TapSums& sumsY) {
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
        ReplicatedShiftedRow<WordType> taps[4][N];
        for (size_t k = 0; k < N; ++k) {
            taps[0][k] = displacedRow<WordType>(lv.next[k], srcY, tapX);
            taps[1][k] = displacedRow<WordType>(lv.next[k], srcY, tapX + 1);
            taps[2][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX);
            taps[3][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX + 1);
        }

        TapSums rowX;
        TapSums rowY;
        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            WordType t00[N], t01[N], t10[N], t11[N], self[N];
            for (size_t k = 0; k < N; ++k) {
                t00[k] = taps[0][k].word(i);
                t01[k] = taps[1][k].word(i);
                t10[k] = taps[2][k].word(i);
                t11[k] = taps[3][k].word(i);
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

            rowX.t00 += slicedSignedSum<N, WordType>(magX, signX, t00);
            rowX.t01 += slicedSignedSum<N, WordType>(magX, signX, t01);
            rowX.t10 += slicedSignedSum<N, WordType>(magX, signX, t10);
            rowX.t11 += slicedSignedSum<N, WordType>(magX, signX, t11);
            rowX.self += slicedSignedSum<N, WordType>(magX, signX, self);

            rowY.t00 += slicedSignedSum<N, WordType>(magY, signY, t00);
            rowY.t01 += slicedSignedSum<N, WordType>(magY, signY, t01);
            rowY.t10 += slicedSignedSum<N, WordType>(magY, signY, t10);
            rowY.t11 += slicedSignedSum<N, WordType>(magY, signY, t11);
            rowY.self += slicedSignedSum<N, WordType>(magY, signY, self);
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
        ReplicatedShiftedRow<WordType> taps[4][N];
        for (size_t k = 0; k < N; ++k) {
            taps[0][k] = displacedRow<WordType>(lv.next[k], srcY, tapX);
            taps[1][k] = displacedRow<WordType>(lv.next[k], srcY, tapX + 1);
            taps[2][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX);
            taps[3][k] = displacedRow<WordType>(lv.next[k], srcY + 1, tapX + 1);
        }

        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            WordType iw[N];
            WordType tw[4][N];
            for (size_t k = 0; k < N; ++k) {
                iw[k] = ip[k][i];
                for (size_t t = 0; t < 4; ++t) tw[t][k] = taps[t][k].word(i);
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
};

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

    const bool coarsest = (li + 1 == c.usableLevels);
    const bool finest = (li == 0);
    const float scale = 1.0f / static_cast<float>(1u << li);

    checkLevelPlanes(lv);

    // PASS 1 -- propagate every point's estimate into this level's
    // coordinates, before any point is tracked. The reference does this in its
    // own first loop, and it applies to skipped points too.
    for (size_t p = 0; p < c.pointCount; ++p) {
        if (coarsest) {
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
    for (size_t p = 0; p < c.pointCount; ++p) {
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
            continue;
        }

        const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), c.winW, c.winH);
        const RegionWords<WordType> region =
            clipRegion<WordType>(lv.width(), lv.height(), window);
        if (region.isEmpty) {
            if (finest) c.status[p] = 0;
            continue;
        }

        // BIT-PARALLEL: the 2x2 matrix, one fused traversal, zero scratch.
        const GradientCovariance a = levelCovariance(lv, window);
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
            continue;
        }

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

            // BIT-PARALLEL: ten exact integers, twenty popcounts per word.
            TapSums sumsX;
            TapSums sumsY;
            residualSums(lv, region, tapX, tapY, sumsX, sumsY);

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
/// A level's extent, so `usableLevelCount` needs no <utility>.
struct LevelDims {
    size_t width = 0;
    size_t height = 0;
};

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
    auto visit = [&](const auto& lv, size_t li) { impl::trackOneLevel(lv, li, c); };
    levels.visitCoarseToFine(c.usableLevels, visit, 0);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
