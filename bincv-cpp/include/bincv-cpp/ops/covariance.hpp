#pragma once

/// @file covariance.hpp
/// @brief The Lucas-Kanade 2x2 gradient covariance over a window (T3.6).
///        **API TIER 3** -- OpenCV has no operation with these semantics, so
///        nothing here borrows an OpenCV name (ARCHITECTURE 5.1).
///
/// **THE LOAD-BEARING OPERATION** ([ARCHITECTURE 7.5]). Lucas-Kanade needs
/// `[ΣIx², ΣIxIy; ΣIxIy, ΣIy²]` over a window, and the claim this whole project
/// rests on is that for sign-magnitude ternary derivatives (D-3) every entry of
/// that matrix is a POPULATION COUNT OVER A MASK -- no multiply, no accumulator
/// per pixel, WordBits pixels per instruction:
///
///     ΣIx²  = popcount(mag_x)
///     ΣIy²  = popcount(mag_y)
///     ΣIxIy = popcount(mag_x & mag_y & ~(sign_x ^ sign_y))   // agreeing: +1
///           - popcount(mag_x & mag_y &  (sign_x ^ sign_y))   // opposing:  -1
///
/// It is exact rather than approximate, and that is worth saying plainly because
/// a reader expects a bit-parallel trick to cost accuracy somewhere. A ternary
/// value is in {-1, 0, +1}, so Ix² is 1 exactly where mag_x is set; and the
/// product IxIy is +1 where both magnitudes are set and the signs agree, -1 where
/// both are set and they disagree, 0 otherwise. Every entry is an integer and this
/// file computes that integer. tests/test_covariance.cpp checks it against a
/// per-pixel FLOAT reference at every window position of whole frames, at three
/// window sizes and all four word types, and requires equality with no tolerance
/// -- a tolerance would be a bug, not a rounding allowance.
///
/// ---------------------------------------------------------------------------
/// ONE CALL, AND NO SCRATCH AT ALL
///
/// This operation is a naming of `countCovariance`'s four-argument form plus one
/// signed subtraction. It is not a loop, and the fact that it is not a loop is
/// the outcome of an experiment rather than an implementation style:
///
///     const CovarianceCount c = countCovariance(dx.constMagnitude(0),
///                                               dy.constMagnitude(0),
///                                               dx.constSign(), dy.constSign(),
///                                               window);
///
/// E-3 (T2.10, EXPERIMENTS.md X-11 / X-11b, ARCHITECTURE.md D-15) settled two
/// questions that between them fix that spelling, and BOTH went against the
/// simpler shape T3.6 was originally specified with:
///
///   * **Fused, not composed.** `countNonZero(mag_x, w) + countNonZero(mag_y, w) +
///     countAndSplit(...)` produces the same four numbers with the same popcounts,
///     and makes THREE traversals of the window instead of one: 6 word loads per
///     word index against this call's 4, or 5 against 3 in the selector-plane form
///     X-11 timed. Measured 1.27-1.29x pre-split (X-11 axis 2) and 1.20-1.27x at
///     31x31 against the shipped entry points (X-11b). **Both of those are
///     measured ONE LEVEL DOWN, on ops/reduce.hpp's entry points rather than on
///     this one.** X-17 asks the same question at THIS level, on the
///     four-argument form this operation actually calls -- and X-17 is `PARTIAL`:
///     its reference-device number is NOT TAKEN (the device refused preflight with
///     a sticky throttle flag), and its x86 table is marked "INDICATIVE ONLY --
///     not a result, do not quote": every ratio there sits inside its own
///     within-run spread, and re-running the same binary moves the ranking. So
///     the ratios above are what is known, the spelling below is what D-15
///     decided, and the confirmation at this level is outstanding.
///   * **Four arguments, not a selector plane.** The four-argument form XORs the
///     two sign planes inside the word loop, so this operation needs **no scratch
///     of any kind** -- 0 B beyond the derivative planes it must read anyway. The
///     precomputed `sign_x ^ sign_y` plane is 11-14% FASTER per frame even after
///     paying to form it (X-11b axis 3), and costs a fifth frame-sized plane at
///     every pyramid level (+25% of the derivative working set, ~51 kB over four
///     levels). Speed and footprint disagree, and CLAUDE.md's tiebreak takes the
///     memory. tests/test_covariance.cpp does not take the no-scratch claim on
///     trust: it counts `operator new` across these calls -- the plain AND the
///     C++17 over-aligned forms, which is the path a vectorized scratch buffer
///     would take -- and requires zero, with the counter itself exercised on one
///     allocation of each kind so that the zero is a reading and not a blind spot.
///
/// A caller that already holds such a plane for other reasons should call
/// `countCovariance(magX, magY, plane, window)` directly and keep the speed; it is
/// the same four numbers. Nothing here obliges the plane to exist, which is the
/// whole point of D-15's third item.
///
/// ---------------------------------------------------------------------------
/// SWEEPING A COLUMN OF POSITIONS? HALF OF THIS SLIDES AND HALF DOES NOT.
///
/// **T3.7's corner response, and any search sweep, want `SlidingWindowCount` for
/// `sumXX` and `sumYY`** (ops/reduce.hpp, the INC-ROW form) -- AT A LARGE ENOUGH
/// WINDOW. Consecutive windows in a vertical column differ by two rows out of W,
/// and recomputing all W of them re-reads what the previous position already
/// counted. Measured on the reference device against the shipped recompute path:
/// **15.9x** on a dense scan at 31x31 and **5.96x** on an 8x8 search sweep
/// (X-11b axis 1). This function recomputes, by construction.
///
/// **T3.7 then measured the same choice AT ITS OWN LEVEL and it does not hold at
/// every window size (X-18).** Sweeping a covariance is not sweeping one plane's
/// popcount: only two of the three numbers slide, and the accumulator forces a
/// column-major traversal. On the reference device at 640x480 the sliding corner
/// sweep is 1.22x faster at a 31x31 window and **1.20x SLOWER at 3x3** -- the block
/// size SEAL/seal_params.yaml runs -- over four runs whose ranking never changes.
/// The advice above stands
/// for large windows and is wrong for small ones; ops/corner.hpp carries the table.
///
/// **Read those two numbers for what they are.** `SlidingWindowCount` slides ONE
/// plane's popcount -- it is constructed from a single `BinMatConstView` and
/// returns a single `count()` -- so 15.9x and 5.96x are `countNonZero` sweeps, not
/// covariance sweeps. `sumXX` and `sumYY` are exactly that shape and get exactly
/// that speedup, one accumulator each. **`sumXY` has no incremental form in this
/// library**: the cross term needs `magX & magY` split by `signX ^ signY`, and
/// nothing in ops/reduce.hpp slides a split. A column sweep today therefore slides
/// two of the three numbers and RECOMPUTES the third, per position, through the
/// four-argument `countAndSplit`. Materializing `magX & magY` and
/// `magX & magY & (signX ^ signY)` as frame-sized planes would make the third
/// number slide too, at the cost of TWO frame-sized planes per pyramid level --
/// more than the one plane D-15 axis 3 already declined on memory grounds, so it
/// is not a free win and is not registered as one.
///
/// That split is not a defect of this signature -- it is the right shape for the
/// pattern it names. **One window is a countNonZero, not a slide**: at isolated
/// keypoints the incremental form issues exactly the same popcounts over exactly
/// the same words and wins nothing (1.10x, which is call structure and not
/// incremental state). The LK covariance of ARCHITECTURE 7.5 -- one window per
/// tracked keypoint -- is exactly that pattern, and is what this file is for.
///
/// The two are not alternatives to choose between by taste. The access pattern
/// decides, and ops/reduce.hpp's "WHICH SHAPE TO REACH FOR" section is the table.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS OPERATION PROMISES
///
///  1. **Ternary only, i.e. pyramid level 0.** The identity above is exact for
///     values in {-1, 0, +1}. For an N-bit level the same structure holds with
///     BIT-SLICED WEIGHTED SUMS -- each magnitude plane pair contributes at its
///     binary weight, so the covariance becomes a weighted combination of the same
///     masked popcounts (ARCHITECTURE 7.5, last paragraph). That is a different
///     kernel, not a wider version of this one, and it is deliberately not written
///     here. **The CONTAINER spelling refuses N > 1 at compile time** -- it takes
///     `TernaryMat<W>`, which is `SignedQuantMat<1, W>`, so
///     `gradientCovariance(dx3, dy3, w)` on a `SignedQuantMat<3, W>` is "no
///     matching function" rather than a silently wrong matrix. **The VIEW spelling
///     cannot make that promise, and does not**: a `BinMatConstView` carries no
///     plane count, so passing an N-bit level's planes to the five-argument form
///     compiles cleanly and returns the covariance of the LSB magnitude planes
///     alone. Measured on a `SignedQuantMat<3, uint32_t>` filled with values in
///     [-3, 3]: the view form returns `{183, 183, 6}` where the true 3-bit signed
///     covariance is `{1275, 1283, 64}`. That is a caller error the type system
///     cannot catch -- assemble views by hand only from a ternary level.
///  2. **Windows are CLIPPED, not rejected** (D-13, and ops/reduce.hpp's region
///     contract). A 31x31 window centred on a keypoint within 15 pixels of an edge
///     is out of range, and every LK frontend has such keypoints. The window is
///     intersected with the image; the pixels that exist contribute and the rest
///     do not. A window wholly outside gives `{0, 0, 0}`, which is a value and not
///     an error. **A bit at or past `width` is never counted**, whatever it holds
///     -- so a view onto a wider image counts its own pixels and not its
///     neighbours', and a wrapped buffer with dirty padding gives the clean
///     buffer's answer.
///  3. **No allocation, no throw, no scratch** (ARCHITECTURE 5.3). Mismatched
///     plane dimensions are a programming error reported by BINCV_ASSERT in debug
///     builds and undefined in release, exactly as in ops/reduce.hpp. There is no
///     error return: a covariance over valid views cannot fail.
///  4. **Read-only, so aliasing is unrestricted** (ops/reduce.hpp promise 3).
///     `gradientCovariance(dx, dx, w)` is well defined -- it is the ΣIx² case with
///     a cross term equal to ΣIx², since a plane never disagrees in sign with
///     itself.
///  5. **The sign plane is read ONLY where both magnitudes are set**, which is
///     what makes the canonical-zero rule (quantMat.hpp: a sign bit over magnitude
///     zero carries no information) irrelevant to the answer rather than a
///     precondition on the caller. Dirtying the sign planes wherever the magnitude
///     is clear cannot move any of the three numbers, and the test sweeps a frame
///     built exactly that way.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// TernaryMat / SignedQuantMat, for the container spelling -- what T3.5's
// derivativeX / derivativeY write, and therefore what this reads.
#include "../quantMat.hpp"
// countCovariance and CovarianceCount: the fused, scratch-free reduction this
// operation IS. Nothing in this file re-implements a word loop -- see D-15.
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The 2x2 Lucas-Kanade gradient covariance over one window:
///        `[sumXX, sumXY; sumXY, sumYY]`. **API TIER 3.**
///
/// @note **All three fields are SIGNED, including the two that cannot be
///       negative.** `sumXX` and `sumYY` are population counts and `sumXY` is a
///       difference of two of them, so only the last can be less than zero -- but
///       the three are used together, and every use is signed arithmetic: T3.7's
///       minimum eigenvalue is
///       `((sumXX + sumYY) - sqrt((sumXX - sumYY)^2 + 4*sumXY^2)) / 2`, in which
///       `sumXX - sumYY` is negative for half the windows in any real frame. Typed
///       as counts, that subtraction is `size_t` arithmetic and wraps to ~1.8e19
///       silently -- unsigned wraparound is defined behaviour, so `-Wconversion
///       -Wsign-conversion -Werror` does not diagnose it. `SplitCount::crossTerm()`
///       exists for exactly this hazard one level down (ops/reduce.hpp); this
///       struct is the same defence applied to the whole matrix, by making the
///       type of the value announce that its differences are signed.
/// @note `int64_t` and not `int`: the counts are bounded by the window's pixel
///       count, which for a dense sweep of a large frame exceeds `INT_MAX` only in
///       absurd cases -- but the caller's arithmetic squares them. `sumXY^2` at a
///       31x31 window is at most 961^2, comfortable; at a window the size of a
///       640x480 frame it is 3.07e5 squared, which is 9.4e10 and not an `int`.
///       64 bits costs nothing here and removes the question.
/// @note Zero-initialized, so a wholly-clipped window's `{0, 0, 0}` is the same
///       value whether it was computed or defaulted.
struct GradientCovariance {
    int64_t sumXX = 0;  ///< ΣIx² over the window: popcount(mag_x)
    int64_t sumYY = 0;  ///< ΣIy² over the window: popcount(mag_y)
    int64_t sumXY = 0;  ///< ΣIxIy over the window, SIGNED: agreeing minus opposing
};

/// @brief The 2x2 gradient covariance of a ternary derivative pair over `window`,
///        from ONE traversal and with no scratch. **API TIER 3.**
/// @tparam WordType The views' word type (D-1).
/// @param magX Magnitude plane of the x-derivative -- `dx.constMagnitude(0)`.
/// @param magY Magnitude plane of the y-derivative -- `dy.constMagnitude(0)`.
/// @param signX Sign plane of the x-derivative -- `dx.constSign()`. A set bit is
///        NEGATIVE (D-3).
/// @param signY Sign plane of the y-derivative -- `dy.constSign()`.
/// @param window Window in pixels, half-open, **intersected with the image**.
/// @return `{ΣIx², ΣIy², ΣIxIy}` over the intersection; `{0, 0, 0}` when it is
///         empty.
///
/// @note The view spelling, for a caller whose planes do not come from one
///       container -- a window onto a wider frame, a wrapped sensor buffer, a
///       pyramid level assembled by hand. The container spelling below is the one
///       T3.6 specifies and forwards to this (D-5: kernels take views).
/// @note **Spell the arguments `constMagnitude()` / `constSign()`, not
///       `magnitude()` / `sign()`.** Deduction does not consider the
///       BinMatView -> BinMatConstView conversion (D-9), so the short spelling on
///       a non-const container is a deduction failure whose diagnostic does not
///       mention the conversion. ops/reduce.hpp promise 1 has the full note.
/// @note The four views must have the same dimensions -- BINCV_ASSERT, since
///       nothing about a window makes differing frame sizes meaningful.
/// @note **TERNARY PLANES ONLY, and this overload cannot check it.** `magX` and
///       `magY` must be the magnitude planes of a ONE-bit signed level. A view
///       carries no plane count, so handing this `dx3.constMagnitude(0)` from a
///       `SignedQuantMat<3, W>` compiles and returns the LSB plane's covariance --
///       a wrong matrix, not an error. The container overload below is the one
///       that rejects N > 1; prefer it wherever the planes do come from one
///       container. See promise 1 at the top of this file for the measured numbers.
/// @note `~(signX ^ signY)` is never formed. The cross term's agreeing half is
///       `total - opposing`, which is one popcount cheaper and cannot count a
///       padding bit (ops/reduce.hpp, D-13).
template <typename WordType>
inline GradientCovariance gradientCovariance(BinMatConstView<WordType> magX,
                                             BinMatConstView<WordType> magY,
                                             BinMatConstView<WordType> signX,
                                             BinMatConstView<WordType> signY, Rect window) {
    // ONE fused traversal, four popcounts per word, no selector plane. Everything
    // that makes this the right call rather than three composed ones is D-15; the
    // clipping, padding and aliasing contracts are ops/reduce.hpp's and are not
    // restated here as code.
    const CovarianceCount c = countCovariance<WordType>(magX, magY, signX, signY, window);

    GradientCovariance out;
    // The two counts widen; the cross term is already signed. crossTerm() is
    // called rather than `whenClear - whenSet` spelled out, so the one signed
    // subtraction in the operation has exactly one implementation.
    out.sumXX = static_cast<int64_t>(c.xx);
    out.sumYY = static_cast<int64_t>(c.yy);
    out.sumXY = static_cast<int64_t>(c.crossTerm());
    return out;
}

/// @brief The 2x2 gradient covariance of a ternary derivative pair over `window`.
///        **API TIER 3.** This is the spelling T3.6 specifies.
/// @tparam WordType The containers' word type (D-1).
/// @param dx Horizontal derivative, ternary -- what `derivativeX` writes at
///        pyramid level 0 (ops/derivative.hpp).
/// @param dy Vertical derivative, ternary, with `dx`'s dimensions.
/// @param window Window in pixels, half-open, intersected with the image. A
///        keypoint's 31x31 window near an edge is out of range and is clipped
///        here; see promise 2 at the top of this file.
/// @return `{ΣIx², ΣIy², ΣIxIy}` over the intersection.
///
/// @note **Ternary only.** `TernaryMat<W>` is `SignedQuantMat<1, W>`, and N > 1
///       does not match this overload -- deliberately, since an N-bit level needs
///       bit-sliced weighted sums rather than a single masked popcount (promise 1).
/// @note A thin naming of the view form, in the shape ops/derivative.hpp and
///       ops/pyramid.hpp use for their container spellings: the container knows
///       which of its planes are magnitude and which is sign, and the kernel does
///       not have to (D-5).
/// @note **Never throws.** `constMagnitude(0)` and `constSign()` are the checked
///       accessors quantMat.hpp keeps live in every build, but the index here is
///       the literal 0 against N == 1, so the check cannot fire.
/// @note No allocation and no scratch buffer -- 0 B beyond the four planes read.
///       tests/test_covariance.cpp counts `operator new` across this call.
template <typename WordType>
inline GradientCovariance gradientCovariance(const TernaryMat<WordType>& dx,
                                             const TernaryMat<WordType>& dy, Rect window) {
    return gradientCovariance<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                        dy.constSign(), window);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
