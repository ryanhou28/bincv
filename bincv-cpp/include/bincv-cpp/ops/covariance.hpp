#pragma once

/// @file covariance.hpp
/// @brief The Lucas-Kanade 2x2 gradient covariance over a window -- ternary
/// and N-bit.
/// **API TIER 3** -- OpenCV has no operation with these semantics, so
/// nothing here borrows an OpenCV name (the design notes).
///
/// **THE LOAD-BEARING OPERATION** ([the design notes]). Lucas-Kanade needs
/// `[ΣIx², ΣIxIy; ΣIxIy, ΣIy²]` over a window, and the claim this whole project
/// rests on is that for sign-magnitude ternary derivatives every entry of
/// that matrix is a POPULATION COUNT OVER A MASK -- no multiply, no accumulator
/// per pixel, WordBits pixels per instruction:
///
/// ΣIx² = popcount(mag_x)
/// ΣIy² = popcount(mag_y)
/// ΣIxIy = popcount(mag_x & mag_y & ~(sign_x ^ sign_y)) // agreeing: +1
/// - popcount(mag_x & mag_y & (sign_x ^ sign_y)) // opposing: -1
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
/// const CovarianceCount c = countCovariance(dx.constMagnitude(0),
/// dy.constMagnitude(0),
/// dx.constSign, dy.constSign,
/// window);
///
/// (/, ) settled two
/// questions that between them fix that spelling, and BOTH went against the
/// simpler shape this was originally specified with:
///
/// * **Fused, not composed.** `countNonZero(mag_x, w) + countNonZero(mag_y, w) +
/// countAndSplit(...)` produces the same four numbers with the same popcounts,
/// and makes THREE traversals of the window instead of one: 6 word loads per
/// word index against this call's 4, or 5 against 3 in the selector-plane form
/// a measurement timed. Measured 1.27-1.29x pre-split (the axis 2) and 1.20-1.27x at
/// 31x31 against the shipped entry points. **Both of those are
/// measured ONE LEVEL DOWN, on ops/reduce.hpp's entry points rather than on
/// this one.** asks the same question at THIS level, on the
/// four-argument form this operation actually calls -- and this is `PARTIAL`:
/// its reference-device number is NOT TAKEN (the device refused preflight with
/// a sticky throttle flag), and its x86 table is marked "INDICATIVE ONLY --
/// not a result, do not quote": every ratio there sits inside its own
/// within-run spread, and re-running the same binary moves the ranking. So
/// the ratios above are what is known, the spelling below is what
/// decided, and the confirmation at this level is outstanding.
/// * **Four arguments, not a selector plane.** The four-argument form XORs the
/// two sign planes inside the word loop, so this operation needs **no scratch
/// of any kind** -- 0 B beyond the derivative planes it must read anyway. The
/// precomputed `sign_x ^ sign_y` plane is 11-14% FASTER per frame even after
/// paying to form it (the axis 3), and costs a fifth frame-sized plane at
/// every pyramid level (+25% of the derivative working set, ~51 kB over four
/// levels). Speed and footprint disagree, and CLAUDE.md's tiebreak takes the
/// memory. tests/test_covariance.cpp does not take the no-scratch claim on
/// trust: it counts `operator new` across these calls -- the plain AND the
/// C++17 over-aligned forms, which is the path a vectorized scratch buffer
/// would take -- and requires zero, with the counter itself exercised on one
/// allocation of each kind so that the zero is a reading and not a blind spot.
///
/// A caller that already holds such a plane for other reasons should call
/// `countCovariance(magX, magY, plane, window)` directly and keep the speed; it is
/// the same four numbers. Nothing here obliges the plane to exist, which is the
/// whole point of the design rule’s third item.
///
/// ---------------------------------------------------------------------------
/// SWEEPING A COLUMN OF POSITIONS? HALF OF THIS SLIDES AND HALF DOES NOT.
///
/// **that work’s corner response, and any search sweep, want `SlidingWindowCount` for
/// `sumXX` and `sumYY`** (ops/reduce.hpp, the INC-ROW form) -- AT A LARGE ENOUGH
/// WINDOW. Consecutive windows in a vertical column differ by two rows out of W,
/// and recomputing all W of them re-reads what the previous position already
/// counted. Measured on the reference device against the shipped recompute path:
/// **15.9x** on a dense scan at 31x31 and **5.96x** on an 8x8 search sweep
/// (the axis 1). This function recomputes, by construction.
///
/// ** then measured the same choice AT ITS OWN LEVEL and it does not hold at
/// every window size.** Sweeping a covariance is not sweeping one plane's
/// popcount: only two of the three numbers slide, and the accumulator forces a
/// column-major traversal. On the reference device at 640x480 the sliding corner
/// sweep is 1.22x faster at a 31x31 window and **1.20x SLOWER at 3x3** -- the block
/// size SEAL/seal_params.yaml runs -- over four runs whose ranking never changes.
/// The advice above stands
/// for large windows and is wrong for small ones; ops/corner.hpp carries the table.
///
/// **Read those two numbers for what they are.** `SlidingWindowCount` slides ONE
/// plane's popcount -- it is constructed from a single `BinMatConstView` and
/// returns a single `count` -- so 15.9x and 5.96x are `countNonZero` sweeps, not
/// covariance sweeps. `sumXX` and `sumYY` are exactly that shape and get exactly
/// that speedup, one accumulator each. **`sumXY` has no incremental form in this
/// library**: the cross term needs `magX & magY` split by `signX ^ signY`, and
/// nothing in ops/reduce.hpp slides a split. A column sweep today therefore slides
/// two of the three numbers and RECOMPUTES the third, per position, through the
/// four-argument `countAndSplit`. Materializing `magX & magY` and
/// `magX & magY & (signX ^ signY)` as frame-sized planes would make the third
/// number slide too, at the cost of TWO frame-sized planes per pyramid level --
/// more than the one plane the axis 3 already declined on memory grounds, so it
/// is not a free win and is not registered as one.
///
/// That split is not a defect of this signature -- it is the right shape for the
/// pattern it names. **One window is a countNonZero, not a slide**: at isolated
/// keypoints the incremental form issues exactly the same popcounts over exactly
/// the same words and wins nothing (1.10x, which is call structure and not
/// incremental state). The LK covariance of the design notes -- one window per
/// tracked keypoint -- is exactly that pattern, and is what this file is for.
///
/// The two are not alternatives to choose between by taste. The access pattern
/// decides, and ops/reduce.hpp's "WHICH SHAPE TO REACH FOR" section is the table.
///
/// ---------------------------------------------------------------------------
/// THE N-BIT LEVEL: THE SAME POPCOUNTS, WEIGHTED OVER PLANE PAIRS
///
/// **Why this is here at all.** a measurement measured the hybrid LK tracker missing its
/// accuracy tolerance on the reference pipeline's own edge-map content, and
/// separated the causes: on the windows that never clip, four 1-BIT pyramid levels
/// are still ~600x worse than one, because a level whose pixels are BITS cannot
/// localise sub-pixel motion better than its own quantisation and that error is
/// multiplied by 2^level on the way down. a measurement had already measured the levels
/// needing 1/3/4/5 bits -- a frame statistic superseded with the alphabet the
/// arithmetic can REACH, 1/3/5/7. So the fix is N-bit levels -- and before earlier work binCV
/// could not form the LK covariance above one bit AT ALL, which is what blocked
/// from measuring a bit-depth choice.
///
/// **The formulation** (the design notes, last paragraph). For magnitude planes
/// `m[0..N-1]` and sign plane `s`, a pixel is `Ix = +/- SUM_i 2^i * m_x[i]`, so:
///
/// sumXX = SUM_i SUM_j 2^(i+j) * popcount(m_x[i] & m_x[j]) // sign squares away
/// sumYY = SUM_i SUM_j 2^(i+j) * popcount(m_y[i] & m_y[j])
/// sumXY = SUM_i SUM_j 2^(i+j) * [ popcount(m_x[i] & m_y[j] & ~(s_x^s_y))
/// - popcount(m_x[i] & m_y[j] & (s_x^s_y)) ]
///
/// Every term is still a masked popcount over words -- there is no multiply per
/// pixel and no per-pixel accumulator anywhere. What changes is that there are
/// N^2 of them per matrix entry instead of one.
///
/// **The sign planes are still read only where BOTH magnitudes are set**, and for
/// the same reason as at N = 1: `m_x[i] & m_y[j]` can only be set at a pixel whose
/// two magnitudes are both non-zero, so the canonical-zero rule stays irrelevant
/// to the answer rather than becoming a precondition (promise 5).
///
/// **IT IS QUADRATIC IN N, WHERE THE DERIVATIVE IS LINEAR, AND THAT IS INHERENT.**
/// A product of two N-bit values is a sum over plane PAIRS; that work’s derivative is a
/// ripple-borrow subtraction and touches each plane once. Exploiting the symmetry
/// of the two diagonal entries (`m_x[i] & m_x[j]` is symmetric in i and j, so the
/// off-diagonal pair is counted once and doubled) the cost per word is
///
/// N(N+1)/2 popcounts for sumXX (the upper triangle)
/// N(N+1)/2 popcounts for sumYY
/// 2*N^2 popcounts for sumXY (a total and a selected count per ordered pair)
/// ------------------------------
/// 3*N^2 + N popcounts per word, plus 2N+2 word loads and one selector XOR
///
/// N = 1: 4 N = 2: 14 N = 3: 30 N = 4: 52
///
/// **At N = 1 that is 4 popcounts per word -- exactly what `countCovariance`
/// issues** (`popcount(a)`, `popcount(b)`, `popcount(a & b)`,
/// `popcount(a & b & sel)`), which is the arithmetic statement of "ternary is the
/// N = 1 instance". The predicted cost ratios against N = 1 are therefore 3.5x,
/// 7.5x and 13x at N = 2, 3, 4.
///
/// **WHAT THE REFERENCE DEVICE ACTUALLY CHARGES.** At `uint32_t` (the
/// shipped default word width,), 640x480, a 31x31 window, 200 keypoints:
///
/// N = 1 903 ns/window 1.00x (predicted 1.00x) 153600 B/level
/// N = 2 3187 ns/window 3.53x (predicted 3.50x) 230400 B/level
/// N = 3 5907 ns/window 6.54x (predicted 7.50x) 307200 B/level
/// N = 4 11023 ns/window 12.20x (predicted 13.00x) 384000 B/level
///
/// Re-measured after triage tidied the per-row accumulator ( run 4, the kernel
/// that ships): 896 / 2930 / 6189 / 10551 ns, i.e. 1.00x / 3.27x / 6.91x / 11.78x.
/// Across the four runs the W = 31 `uint32_t` ratios span 3.0-3.5x, 6.4-6.9x and
/// 11.8-12.5x, which is the honest width of "3.5x / 6.5x / 12.2x" -- the spread
/// between BINARIES, not within one.
///
/// So at W = 31 `3N^2 + N` is a good model and a slight OVER-estimate, which is the
/// right way for a price to be wrong when a decision is taken against it.
///
/// **THE MODEL'S VALIDITY RANGE IS PART OF THE MODEL, AND IT IS NOT ALL WINDOW
/// SIZES.** The table above is W = 31. The same run's smaller windows (that measurement’s
/// per-cell band table):
///
/// uint32_t W = 7 4.76x / 7.22x / 12.96x at N = 2, 3, 4
/// uint32_t W = 15 4.06x / 6.61x / 12.02x
/// uint32_t W = 31 3.53x / 6.54x / 12.20x
///
/// At W = 7 the model UNDER-estimates N = 2 by 36%, outside that measurement’s pre-registered
/// +/-25% band and in the direction that says something is quadratic that should not
/// be -- reported, not absorbed. It is also the cell that moved most between two
/// binaries built from unchanged source (3.27x to 4.76x), so it is layout-confounded
/// and is NOT attributed here. **Read `3N^2 + N` as measured within +/-25% at
/// W = 15 and W = 31, and as an under-estimate at N = 2 on small windows** -- where
/// the per-window and per-row fixed costs are largest relative to the word work.
/// The bytes
/// column is the other half of the trade and is why it is printed next to the
/// time: an N-bit level costs (N+1) bits per pixel per derivative against
/// ternary's 2. **this is the task that weighs those two columns against that measurement’s
/// accuracy finding; this file takes no bit-depth decision.** Two caveats belong
/// with the numbers rather than after them: at `uint64_t` the curve runs ~20%
/// ABOVE the model at N >= 3 and the 64-bit word is slower in absolute terms than
/// the 32-bit one at N = 4 -- register pressure was the obvious cause and was
/// measured and REJECTED (scripts/covariance_nbit_codegen.sh) -- and the same
/// kernel's absolute cost moved by up to 1.46x between two binaries built from
/// unchanged source, so re-measure in your own binary rather than quoting these.
/// has both in full, and registered an open question for the per-row accumulator, which
/// is O(N^2) per row here where it was O(1) at N = 1.
///
/// **Do not try to make this linear.** Anything linear in N is computing a
/// different quantity.
///
/// **NO HEAP, AND NO CALLER SCRATCH EITHER -- the property is preserved, not
/// traded.** The four-argument `countAndSplit` XORs the two sign planes inside the
/// word loop, and this kernel does the same, so N > 1 needs no selector plane
/// either. What it does need is somewhere to hold the per-pair counts, and that is
/// `impl::BitSlicedPairCounts<N>` -- 4*N^2 `size_t` in AUTOMATIC storage, 512 B at
/// N = 4, zero bytes of heap and zero bytes of caller-provided buffer. It is not a
/// scratch BUFFER in the/no-heap sense (nothing is allocated, nothing is
/// passed in, nothing outlives the call), and tests/test_covariance.cpp counts
/// `operator new` across the N-bit calls exactly as it does across the ternary
/// ones and requires zero.
///
/// **THE ACCUMULATOR IS `int64_t` AND WHAT BOUNDS IT.** Each per-pair counter is a
/// popcount over the window, so it is at most the window's pixel count P. The
/// weights sum to `(SUM_i 2^i)^2 = (2^N - 1)^2`, so
///
/// |sumXX|, |sumYY|, |sumXY| <= (2^N - 1)^2 * P
///
/// which is just `max|I|^2 * P` -- the obvious bound, reached only by a window
/// that is saturated everywhere. At N = 4 and a 31x31 window: the largest single
/// pair term is `2^6 * 961 = 61504` and the whole entry is at most
/// `225 * 961 = 216225`, which needs 18 bits. `int64_t` overflows only for a window
/// of more than `2^63 / (2^N - 1)^2` pixels -- **4.1e16 at N = 4, and 5.7e14 at
/// N = 7, the largest N `SignedQuantMat` admits**, which is the worst case over the
/// whole admissible range. At N = 7 that window is 5.7e14 pixels carrying seven
/// magnitude planes and a sign plane, i.e. about 570 TB of planes, so no
/// addressable image can reach it. The per-pair counters
/// are `size_t` and bounded by P for the same reason `SplitCount`'s halves are.
/// The one signed subtraction still has exactly one implementation:
/// `SplitCount::crossTerm`, called per pair.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS OPERATION PROMISES
///
/// 1. **Two kernels, one name: the ternary one above and the BIT-SLICED one
/// added.** The single-popcount identity is exact for values in {-1, 0, +1},
/// i.e. pyramid level 0. An N-bit level needs the weighted plane-pair form in
/// the next section -- a different kernel, not a wider version of this one --
/// and the two are separate overloads that agree exactly where they overlap.
/// **The CONTAINER spellings dispatch on the plane count**, which is a
/// compile-time property of `SignedQuantMat<N, W>`: `TernaryMat` takes the
/// one-popcount path, `SignedQuantMat<3, W>` takes the bit-sliced path, and
/// neither can be reached with the other's planes. shipped with N > 1
/// rejected outright ("no matching function"), because until there was
/// nothing correct to send it to; made that a blocker rather than a
/// conservatism, since a 1-bit pyramid level cannot localise sub-pixel motion
/// and the fix is N-bit levels.
/// **The five-argument VIEW spelling still promises nothing about N, and
/// cannot**: a `BinMatConstView` carries no plane count, so passing an N-bit
/// level's LSB plane to it compiles cleanly and returns the covariance of that
/// plane alone. Measured on a `SignedQuantMat<3, uint32_t>` filled with values
/// in [-3, 3]: the five-argument view form returns `{183, 183, 6}` where the
/// true 3-bit signed covariance is `{1275, 1283, 64}`. **The N-bit view
/// spelling takes plane ARRAYS** -- `const BinMatConstView<W> (&magX)[N]` --
/// so N is in the type there and the same mistake does not compile. Hand
/// assembly of the five-argument form is for a ternary level only.
/// 2. **Windows are CLIPPED, not rejected** (, and ops/reduce.hpp's region
/// contract). A 31x31 window centred on a keypoint within 15 pixels of an edge
/// is out of range, and every LK frontend has such keypoints. The window is
/// intersected with the image; the pixels that exist contribute and the rest
/// do not. A window wholly outside gives `{0, 0, 0}`, which is a value and not
/// an error. **A bit at or past `width` is never counted**, whatever it holds
/// -- so a view onto a wider image counts its own pixels and not its
/// neighbours', and a wrapped buffer with dirty padding gives the clean
/// buffer's answer.
/// 3. **No allocation, no throw, no scratch** (the design notes). Mismatched
/// plane dimensions are a programming error reported by BINCV_ASSERT in debug
/// builds and undefined in release, exactly as in ops/reduce.hpp. There is no
/// error return: a covariance over valid views cannot fail.
/// 4. **Read-only, so aliasing is unrestricted** (ops/reduce.hpp promise 3).
/// `gradientCovariance(dx, dx, w)` is well defined -- it is the ΣIx² case with
/// a cross term equal to ΣIx², since a plane never disagrees in sign with
/// itself.
/// 5. **The sign plane is read ONLY where both magnitudes are set**, which is
/// what makes the canonical-zero rule (quantMat.hpp: a sign bit over magnitude
/// zero carries no information) irrelevant to the answer rather than a
/// precondition on the caller. Dirtying the sign planes wherever the magnitude
/// is clear cannot move any of the three numbers, and the test sweeps a frame
/// built exactly that way.
// F-5: BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"


#include <cstddef>
#include <cstdint>

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// TernaryMat / SignedQuantMat, for the container spelling -- what that work’s
// derivativeX / derivativeY write, and therefore what this reads.
#include "../quantMat.hpp"
// countCovariance and CovarianceCount: the fused, scratch-free reduction this
// operation IS. Nothing in this file re-implements a word loop -- see.
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The 2x2 Lucas-Kanade gradient covariance over one window:
/// `[sumXX, sumXY; sumXY, sumYY]`. **API TIER 3.**
///
/// @note **All three fields are SIGNED, including the two that cannot be
/// negative.** `sumXX` and `sumYY` are population counts and `sumXY` is a
/// difference of two of them, so only the last can be less than zero -- but
/// the three are used together, and every use is signed arithmetic: that work’s
/// minimum eigenvalue is
/// `((sumXX + sumYY) - sqrt((sumXX - sumYY)^2 + 4*sumXY^2)) / 2`, in which
/// `sumXX - sumYY` is negative for half the windows in any real frame. Typed
/// as counts, that subtraction is `size_t` arithmetic and wraps to ~1.8e19
/// silently -- unsigned wraparound is defined behaviour, so `-Wconversion
/// -Wsign-conversion -Werror` does not diagnose it. `SplitCount::crossTerm`
/// exists for exactly this hazard one level down (ops/reduce.hpp); this
/// struct is the same defence applied to the whole matrix, by making the
/// type of the value announce that its differences are signed.
/// @note `int64_t` and not `int`: the counts are bounded by the window's pixel
/// count, which for a dense sweep of a large frame exceeds `INT_MAX` only in
/// absurd cases -- but the caller's arithmetic squares them. `sumXY^2` at a
/// 31x31 window is at most 961^2, comfortable; at a window the size of a
/// 640x480 frame it is 3.07e5 squared, which is 9.4e10 and not an `int`.
/// 64 bits costs nothing here and removes the question.
/// @note Zero-initialized, so a wholly-clipped window's `{0, 0, 0}` is the same
/// value whether it was computed or defaulted.
struct GradientCovariance {
    int64_t sumXX = 0;  ///< ΣIx² over the window: popcount(mag_x)
    int64_t sumYY = 0;  ///< ΣIy² over the window: popcount(mag_y)
    int64_t sumXY = 0;  ///< ΣIxIy over the window, SIGNED: agreeing minus opposing
};

/// @brief The 2x2 gradient covariance of a ternary derivative pair over `window`,
/// from ONE traversal and with no scratch. **API TIER 3.**
/// @tparam WordType The views' word type.
/// @param magX Magnitude plane of the x-derivative -- `dx.constMagnitude(0)`.
/// @param magY Magnitude plane of the y-derivative -- `dy.constMagnitude(0)`.
/// @param signX Sign plane of the x-derivative -- `dx.constSign`. A set bit is
/// NEGATIVE.
/// @param signY Sign plane of the y-derivative -- `dy.constSign`.
/// @param window Window in pixels, half-open, **intersected with the image**.
/// @return `{ΣIx², ΣIy², ΣIxIy}` over the intersection; `{0, 0, 0}` when it is
/// empty.
///
/// @note The view spelling, for a caller whose planes do not come from one
/// container -- a window onto a wider frame, a wrapped sensor buffer, a
/// pyramid level assembled by hand. The container spelling below is the one
/// specifies and forwards to this ( kernels take views).
/// @note **Spell the arguments `constMagnitude` / `constSign`, not
/// `magnitude` / `sign`.** Deduction does not consider the
/// BinMatView -> BinMatConstView conversion, so the short spelling on
/// a non-const container is a deduction failure whose diagnostic does not
/// mention the conversion. ops/reduce.hpp promise 1 has the full note.
/// @note The four views must have the same dimensions -- BINCV_ASSERT, since
/// nothing about a window makes differing frame sizes meaningful.
/// @note **TERNARY PLANES ONLY, and this overload cannot check it.** `magX` and
/// `magY` must be the magnitude planes of a ONE-bit signed level. A view
/// carries no plane count, so handing this `dx3.constMagnitude(0)` from a
/// `SignedQuantMat<3, W>` compiles and returns the LSB plane's covariance --
/// a wrong matrix, not an error. **The N-bit spellings are the fix, not a
/// rejection**: the plane-ARRAY view form and the `SignedQuantMat<N, W>`
/// container form both carry N in the type. Prefer either wherever
/// the level is not ternary. See promise 1 for the measured wrong matrix.
/// @note `~(signX ^ signY)` is never formed. The cross term's agreeing half is
/// `total - opposing`, which is one popcount cheaper and cannot count a
/// padding bit (ops/reduce.hpp).
template <typename WordType>
inline GradientCovariance gradientCovariance(BinMatConstView<WordType> magX,
                                             BinMatConstView<WordType> magY,
                                             BinMatConstView<WordType> signX,
                                             BinMatConstView<WordType> signY, Rect window) {
    // ONE fused traversal, four popcounts per word, no selector plane. Everything
    // that makes this the right call rather than three composed ones is; the
    // clipping, padding and aliasing contracts are ops/reduce.hpp's and are not
    // restated here as code.
    const CovarianceCount c = countCovariance<WordType>(magX, magY, signX, signY, window);

    GradientCovariance out;
    // The two counts widen; the cross term is already signed. crossTerm is
    // called rather than `whenClear - whenSet` spelled out, so the one signed
    // subtraction in the operation has exactly one implementation.
    out.sumXX = static_cast<int64_t>(c.xx);
    out.sumYY = static_cast<int64_t>(c.yy);
    out.sumXY = static_cast<int64_t>(c.crossTerm());
    return out;
}

/// @brief The 2x2 gradient covariance of a ternary derivative pair over `window`.
/// **API TIER 3.** This is the spelling specifies.
/// @tparam WordType The containers' word type.
/// @param dx Horizontal derivative, ternary -- what `derivativeX` writes at
/// pyramid level 0 (ops/derivative.hpp).
/// @param dy Vertical derivative, ternary, with `dx`'s dimensions.
/// @param window Window in pixels, half-open, intersected with the image. A
/// keypoint's 31x31 window near an edge is out of range and is clipped
/// here; see promise 2 at the top of this file.
/// @return `{ΣIx², ΣIy², ΣIxIy}` over the intersection.
///
/// @note **Ternary only, and N > 1 goes elsewhere rather than nowhere.**
/// `TernaryMat<W>` is `SignedQuantMat<1, W>`, so this overload is the better
/// match at N == 1 by partial ordering and keeps the single-popcount path;
/// `SignedQuantMat<N, W>` for N > 1 matches that work’s overload below instead of
/// failing to compile, which is what made a precondition. The two agree
/// exactly at N == 1 -- checked at every window position, not argued.
/// @note A thin naming of the view form, in the shape ops/derivative.hpp and
/// ops/pyramid.hpp use for their container spellings: the container knows
/// which of its planes are magnitude and which is sign, and the kernel does
/// not have to.
/// @note **Never throws.** `constMagnitude(0)` and `constSign` are the checked
/// accessors quantMat.hpp keeps live in every build, but the index here is
/// the literal 0 against N == 1, so the check cannot fire.
/// @note No allocation and no scratch buffer -- 0 B beyond the four planes read.
/// tests/test_covariance.cpp counts `operator new` across this call.
template <typename WordType>
inline GradientCovariance gradientCovariance(const TernaryMat<WordType>& dx,
                                             const TernaryMat<WordType>& dy, Rect window) {
    return gradientCovariance<WordType>(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign(),
                                        dy.constSign(), window);
}


// ---------------------------------------------------------------------------
// THE N-BIT LEVEL
//
// WHY THE ROW BODY LIVES HERE AND NOT IN ops/reduce.hpp
//
// the design notes says the reduction interface for N-bit levels is "specified
// over plane pairs, not over a single mask", and this is that interface. It is
// kept in this file because nothing else in the library reduces plane PAIRS --
// ops/reduce.hpp's entry points are the bulk reductions exports, and adding a
// pair-weighted one there would export a shape with exactly one caller.
//
// What is NOT re-implemented here is anything ops/reduce.hpp already owns: the
// region clip, the head/tail masks, the single-pass row skeleton and the popcount
// all come from impl:: over there. So the design rule’s "a bit at or past width is never
// counted" has ONE implementation for both kernels in this file, and a padding bug
// cannot be fixed in one and not the other.
// ---------------------------------------------------------------------------

namespace impl {

/// @brief Per-plane-pair popcounts for one window (or for one row of it).
/// **INTERNAL.**
/// @tparam N Magnitude planes per derivative.
///
/// @note **This is the whole of that work’s state, and it is AUTOMATIC storage** --
/// 4*N^2 counters, 512 B at N = 4. No heap, and nothing the caller has to
/// provide: the no-scratch property the axis 3 bought at N = 1 survives at
/// N > 1 unchanged, because the sign planes are still XORed inside the word
/// loop rather than materialized as a plane.
/// @note `xx` and `yy` use the UPPER TRIANGLE only (`i <= j`). `m_x[i] & m_x[j]`
/// is symmetric in its indices, so the lower half would be the same numbers
/// counted a second time; combineBitSlicedPairs doubles the off-diagonal
/// instead. `xyTotal` / `xySet` are full N x N, because `m_x[i] & m_y[j]` is
/// NOT symmetric -- x and y are different images.
/// @note Two counters per xy pair rather than a pre-subtracted difference, for
/// SplitCount's reason: the signed subtraction gets exactly one
/// implementation, in SplitCount::crossTerm.
template <size_t N>
struct BitSlicedPairCounts {
    size_t xx[N][N] = {};       ///< popcount(m_x[i] & m_x[j]), i <= j
    size_t yy[N][N] = {};       ///< popcount(m_y[i] & m_y[j]), i <= j
    size_t xyTotal[N][N] = {};  ///< popcount(m_x[i] & m_y[j])
    size_t xySet[N][N] = {};    ///<...of which the selector (s_x ^ s_y) is set

    /// @brief Adds a row's partial counts into this one.
    /// @note Per-row partials rather than one accumulator carried across the whole
    /// window: the same measured choice as ops/reduce.hpp's row bodies
    /// ( item 4, earlier work), applied here so the N = 1 instance has
    /// the shape the N = 1 kernel has.
    void add(const BitSlicedPairCounts<N>& row) {
        for (size_t i = 0; i < N; ++i) {
            // **xx and yy on the UPPER TRIANGLE ONLY, matching where they are
            // written and read.** bitSlicedPairRowRegion writes `j >= i` and
            // combineBitSlicedPairs reads `j >= i`, so both operands of a lower-half
            // add are provably zero -- N^2 - N adds PER ROW that could only ever add
            // nothing (12 of 64 at N = 4, paid per window row, which at a 31x31
            // uint64_t window is per 1-2 words of real work). Folding them was not a
            // rounding difference: the result is bit-identical by construction, which
            // is why the N = 1..MAX_BIT_DEPTH sweeps are the check on it.
            for (size_t j = i; j < N; ++j) {
                xx[i][j] += row.xx[i][j];
                yy[i][j] += row.yy[i][j];
            }
            // The cross term is NOT symmetric -- x and y are different images -- so
            // this one is the full N x N and stays so.
            for (size_t j = 0; j < N; ++j) {
                xyTotal[i][j] += row.xyTotal[i][j];
                xySet[i][j] += row.xySet[i][j];
            }
        }
    }
};

/// @brief Every plane-pair popcount over ONE row of an already-clipped region.
/// **INTERNAL.**
/// @param mx, my Hoisted row pointers, one per magnitude plane.
/// @param sx, sy Hoisted row pointers of the two sign planes.
/// @param r The clipped region; its masks decide which bits of the first and last
/// word are inside.
/// @param out Zeroed on entry by the caller; this adds into it.
///
/// @note One visit per word, `2N + 2` loads, `3N^2 + N` popcounts -- see the
/// N-BIT section at the top of this file for where that count comes from and
/// why it cannot be made linear.
/// @note The magnitude words are masked ONCE, on load. The selector is not masked
/// at all and does not need to be: it is only ever ANDed with `both`, which
/// is masked, so a padding bit of `s_x ^ s_y` can never reach a counter.
/// `~(s_x ^ s_y)` is never formed, exactly as in splitRowRegion -- it would
/// set every padding bit of a trailing word.
template <size_t N, typename WordType>
inline void bitSlicedPairRowRegion(const WordType* (&mx)[N], const WordType* (&my)[N],
                                   const WordType* sx, const WordType* sy,
                                   const RegionWords<WordType>& r, BitSlicedPairCounts<N>& out) {
    visitRowWords<WordType>(r, [&](size_t w, WordType mask) {
        WordType ax[N];
        WordType ay[N];
        for (size_t p = 0; p < N; ++p) {
            ax[p] = static_cast<WordType>(mx[p][w] & mask);
            ay[p] = static_cast<WordType>(my[p][w] & mask);
        }
        const WordType selector = static_cast<WordType>(sx[w] ^ sy[w]);

        for (size_t i = 0; i < N; ++i) {
            // The diagonal entries, upper triangle only. At i == j this is
            // popcount(ax[i]), which is what makes N == 1 issue countCovariance's
            // four popcounts and not five.
            for (size_t j = i; j < N; ++j) {
                out.xx[i][j] += popcountWord<WordType>(static_cast<WordType>(ax[i] & ax[j]));
                out.yy[i][j] += popcountWord<WordType>(static_cast<WordType>(ay[i] & ay[j]));
            }
            // The cross term, every ordered pair: x's plane i against y's plane j.
            for (size_t j = 0; j < N; ++j) {
                const WordType both = static_cast<WordType>(ax[i] & ay[j]);
                const size_t total = popcountWord<WordType>(both);
                const size_t set =
                    popcountWord<WordType>(static_cast<WordType>(both & selector));
                out.xyTotal[i][j] += total;
                out.xySet[i][j] += set;
            }
        }
    });
}

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
/// @brief The whole clipped region's plane-pair counts, in NEON lanes. **INTERNAL**
///. `N` in {1, 2} at `uint32_t` — the shipped 1/2/2/2 ladder.
///
/// ** WAS WRITTEN FOR THIS AND THIS FUNCTION WAS NOT
/// OBEYING IT.** `bitSlicedPairRowRegion` issues `3N^2 + N` **scalar** `popcountWord`
/// calls per word — fourteen at `N = 2` — and on aarch64 there is no scalar popcount:
/// every one is `fmov` in, `cnt`, `addv`, `fmov` out.
/// measured the consequence on the reference device: the covariance is **27.5% of
/// `track`** there against 18.2% on x86, and **5.9× slower than x86** where the
/// iteration loop — which does have a NEON path — is only 3.6× slower. That gap IS the
/// missing kernel.
///
/// The counts go into LANES and stay there to the end of the window, so the register
/// domain is crossed **once per point per level** instead of `14 * rows` times. This is
/// the shape gave the residual accumulator, applied to
/// the operation that sits next to it and never got it.
///
/// **Bit-exact:** the same integers, counted the same way. The scalar body above is the
/// portable one AND the oracle, and `tests/test_covariance.cpp` compares them.
template <size_t N, typename WordType>
inline void bitSlicedPairRegionNeon(const BinMatConstView<WordType> (&magX)[N],
                                    const BinMatConstView<WordType> (&magY)[N],
                                    BinMatConstView<WordType> signX,
                                    BinMatConstView<WordType> signY,
                                    const RegionWords<WordType>& r,
                                    BitSlicedPairCounts<N>& out) {
    static_assert(N == 1 || N == 2, "bitSlicedPairRegionNeon: shipped depths only");
    const auto counts = [](uint32x4_t v) {
        return vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(v))));
    };
    uint32x4_t accA = vdupq_n_u32(0), accB = vdupq_n_u32(0);
    uint32x4_t accC = vdupq_n_u32(0), accD = vdupq_n_u32(0);

    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* rowX[N];
        const WordType* rowY[N];
        for (size_t p = 0; p < N; ++p) {
            rowX[p] = magX[p].row(y);
            rowY[p] = magY[p].row(y);
        }
        const WordType* sxr = signX.row(y);
        const WordType* syr = signY.row(y);
        visitRowWords<WordType>(r, [&](size_t w, WordType mask) {
            const uint32_t sel = static_cast<uint32_t>(sxr[w] ^ syr[w]);
            if constexpr (N == 1) {
                const uint32_t ax = static_cast<uint32_t>(rowX[0][w] & mask);
                const uint32_t ay = static_cast<uint32_t>(rowY[0][w] & mask);
                // FOUR COUNTS, WHICH IS EXACTLY ONE VECTOR:
                // {xx, yy, xy total, xy set}.
                const uint32_t lanes[4] = {ax, ay, ax & ay,
                                           static_cast<uint32_t>(ax & ay & sel)};
                accA = vaddq_u32(accA, counts(vld1q_u32(lanes)));
            } else {
                // ONE array round trip a word, not four. The first version built each
                // of the four operand vectors through its own stack array -- sixteen
                // stores and four loads a word, each load waiting on its stores -- and
                // a measurement measured that eating most of the win. Everything below is a
                // SHUFFLE of the one vector `{ax0, ax1, ay0, ay1}`.
                const uint32_t base[4] = {static_cast<uint32_t>(rowX[0][w] & mask),
                                          static_cast<uint32_t>(rowX[1][w] & mask),
                                          static_cast<uint32_t>(rowY[0][w] & mask),
                                          static_cast<uint32_t>(rowY[1][w] & mask)};
                const uint32x4_t v = vld1q_u32(base);
                // A = {ax0, ax1, ay0, ay1} IS already xx[0][0], xx[1][1], yy[0][0],
                // yy[1][1] -- `a & a` is `a`, so the diagonal needs no AND at all.
                accA = vaddq_u32(accA, counts(v));
                // B: rotating by one lane puts ax1 under ax0 and ay1 under ay0, so
                // lanes 0 and 2 are the two off-diagonal terms. Lanes 1 and 3 are
                // meaningless and simply not read.
                accB = vaddq_u32(accB, counts(vandq_u32(v, vextq_u32(v, v, 1))));
                // C: every ordered (x plane, y plane) pair -- {ax0,ax0,ax1,ax1} against
                // {ay0,ay1,ay0,ay1}, both shuffles of `v`.
                const uint32x4_t xs = vzip1q_u32(v, v);
                const uint32x4_t ys = vcombine_u32(vget_high_u32(v), vget_high_u32(v));
                const uint32x4_t cross = vandq_u32(xs, ys);
                accC = vaddq_u32(accC, counts(cross));
                accD = vaddq_u32(accD, counts(vandq_u32(cross, vdupq_n_u32(sel))));
            }
        });
    }

    // ONE domain crossing per point per level, not fourteen per word.
    if constexpr (N == 1) {
        out.xx[0][0] += vgetq_lane_u32(accA, 0);
        out.yy[0][0] += vgetq_lane_u32(accA, 1);
        out.xyTotal[0][0] += vgetq_lane_u32(accA, 2);
        out.xySet[0][0] += vgetq_lane_u32(accA, 3);
    } else {
        out.xx[0][0] += vgetq_lane_u32(accA, 0);
        out.xx[1][1] += vgetq_lane_u32(accA, 1);
        out.yy[0][0] += vgetq_lane_u32(accA, 2);
        out.yy[1][1] += vgetq_lane_u32(accA, 3);
        out.xx[0][1] += vgetq_lane_u32(accB, 0);
        out.yy[0][1] += vgetq_lane_u32(accB, 2);
        out.xyTotal[0][0] += vgetq_lane_u32(accC, 0);
        out.xyTotal[0][1] += vgetq_lane_u32(accC, 1);
        out.xyTotal[1][0] += vgetq_lane_u32(accC, 2);
        out.xyTotal[1][1] += vgetq_lane_u32(accC, 3);
        out.xySet[0][0] += vgetq_lane_u32(accD, 0);
        out.xySet[0][1] += vgetq_lane_u32(accD, 1);
        out.xySet[1][0] += vgetq_lane_u32(accD, 2);
        out.xySet[1][1] += vgetq_lane_u32(accD, 3);
    }
}
#endif  // BINCV_HAVE_NEON && __aarch64__

/// @brief Weights the plane-pair counts into the 2x2 matrix. **INTERNAL.**
/// @note This is where `2^(i+j)` enters, once per pair per window rather than once
/// per word: the counts are exact integers, so weighting them at the end is
/// the same number as weighting inside the loop and costs O(N^2) instead of
/// O(N^2 * words).
/// @note The off-diagonal of `xx` / `yy` is doubled because only the upper triangle
/// was counted; `xy` is not, because its two triangles are different numbers.
/// @note MULTIPLICATION, not a shift, for the cross term: `crossTerm` is signed
/// and negative for half the windows in any real frame, and shifting a
/// negative value left is undefined before C++20.
template <size_t N>
inline GradientCovariance combineBitSlicedPairs(const BitSlicedPairCounts<N>& c) {
    GradientCovariance out;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i; j < N; ++j) {
            const int64_t weight = static_cast<int64_t>(1) << (i + j);
            const int64_t both = (i == j) ? weight : 2 * weight;
            out.sumXX += both * static_cast<int64_t>(c.xx[i][j]);
            out.sumYY += both * static_cast<int64_t>(c.yy[i][j]);
        }
        for (size_t j = 0; j < N; ++j) {
            const int64_t weight = static_cast<int64_t>(1) << (i + j);
            // whenClear is total - set, never popcount(both & ~selector) -- see
            // the row body. crossTerm then makes the subtraction signed.
            const SplitCount pair{c.xyTotal[i][j] - c.xySet[i][j], c.xySet[i][j]};
            out.sumXY += weight * static_cast<int64_t>(pair.crossTerm());
        }
    }
    return out;
}

/// @brief The shape contract the N-bit kernel takes. **INTERNAL.**
/// @note Every BINCV_ASSERT discards its condition in a release build, which
/// leaves the parameters unread -- and warnings are fatal in the gate. The
/// same shape ops/derivative.hpp's checkDerivativeArgs uses.
template <size_t N, typename WordType>
inline void checkBitSlicedArgs(const BinMatConstView<WordType> (&magX)[N],
                               const BinMatConstView<WordType> (&magY)[N],
                               BinMatConstView<WordType> signX, BinMatConstView<WordType> signY) {
    static_cast<void>(magX);
    static_cast<void>(magY);
    static_cast<void>(signX);
    static_cast<void>(signY);

    for (size_t p = 0; p < N; ++p) {
        BINCV_ASSERT(magX[p].width == magX[0].width && magX[p].height == magX[0].height &&
                         magY[p].width == magX[0].width && magY[p].height == magX[0].height,
                     "covariance: every magnitude plane must have the same dimensions");
        BINCV_ASSERT(strideCoversARow<WordType>(magX[p].width, magX[p].height, magX[p].stride) &&
                         strideCoversARow<WordType>(magY[p].width, magY[p].height, magY[p].stride),
                     "covariance: every view's stride must cover a whole row");
    }
    BINCV_ASSERT(signX.width == magX[0].width && signX.height == magX[0].height &&
                     signY.width == magX[0].width && signY.height == magX[0].height,
                 "covariance: the sign planes must have the magnitude planes' dimensions");
    BINCV_ASSERT(strideCoversARow<WordType>(signX.width, signX.height, signX.stride) &&
                     strideCoversARow<WordType>(signY.width, signY.height, signY.stride),
                 "covariance: every view's stride must cover a whole row");
}

} // namespace impl

/// @brief The 2x2 gradient covariance of an N-BIT signed derivative pair over
/// `window`, as bit-sliced weighted popcounts. **API TIER 3.**
/// @tparam N Magnitude planes per derivative -- the level's bit depth.
/// @tparam WordType The views' word type.
/// @param magX Magnitude planes of the x-derivative, `magX[0]` the LEAST
/// significant -- `{dx.constMagnitude(0),... dx.constMagnitude(N-1)}`.
/// @param magY Magnitude planes of the y-derivative, same order and dimensions.
/// @param signX Sign plane of the x-derivative. A set bit is NEGATIVE, and
/// there is ONE sign plane however many magnitude planes there are.
/// @param signY Sign plane of the y-derivative.
/// @param window Window in pixels, half-open, **intersected with the image**.
/// @return `{SumIx^2, SumIy^2, SumIxIy}` over the intersection; `{0, 0, 0}` when
/// it is empty.
///
/// @note **The view spelling, and unlike the ternary one it does carry N**: the
/// plane count is in the array type, so a caller cannot hand an N-bit level
/// to a kernel that reads one plane (promise 1). The container spelling
/// below is the one to prefer where the planes come from one object.
/// @note **N == 1 is the ternary case and is not special-cased.** This kernel runs
/// its generic loop at N = 1 and must produce, bit for bit, what the
/// five-argument ternary overload above produces -- 4 popcounts per word,
/// the same words, the same masks. tests/test_covariance.cpp requires that
/// equality at every window position of whole frames at all four word types
/// rather than arguing it: if the two can differ at all, one of them is
/// wrong. Delegating to the ternary path at N == 1 would make the check
/// vacuous, which is why this does not.
/// @note ONE traversal of the window, whatever N is. The N^2 plane pairs are all
/// formed from the `2N + 2` words loaded per word index, so the traversal
/// count does not grow with N even though the popcount count does
/// (`3N^2 + N` per word; see the N-BIT section at the top of this file).
/// @note **No allocation, no throw, no caller scratch, and no selector plane.**
/// The per-pair counters are automatic storage. Mismatched plane dimensions
/// are a BINCV_ASSERT programming error, as everywhere else in this file.
/// @note Clipping, padding and aliasing are ops/reduce.hpp's contracts unchanged
/// -- promises 2, 4 and 5 at the top of this file apply verbatim, because
/// the region machinery is literally the same code.
template <size_t N, typename WordType>
inline GradientCovariance gradientCovariance(const BinMatConstView<WordType> (&magX)[N],
                                             const BinMatConstView<WordType> (&magY)[N],
                                             BinMatConstView<WordType> signX,
                                             BinMatConstView<WordType> signY, Rect window) {
    // The container spellings cannot reach an N outside this range -- QuantMat is
    // 1..8 and SignedQuantMat 1..7 -- but hand-assembled plane arrays can, and this
    // overload's docstring blesses them. Without the bound, combineBitSlicedPairs'
    // `int64_t(1) << (i + j)` is undefined once 2N - 2 >= 64, and the weights
    // overflow well before that. Matches pyrDownRoute's and derivativeX's guards.
    static_assert(N >= 1 && N <= 8, "covariance: N outside QuantMat's supported range");

    impl::checkBitSlicedArgs<N, WordType>(magX, magY, signX, signY);

    const impl::RegionWords<WordType> r =
        impl::clipRegion<WordType>(magX[0].width, magX[0].height, window);
    if (r.isEmpty) return GradientCovariance();

    BINCV_ASSERT(signX.ptr != nullptr && signY.ptr != nullptr,
                 "covariance: a non-empty view needs a non-null pointer");

    impl::BitSlicedPairCounts<N> total;
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    // the shipped ladder's depths get the design rule’s reservation cashed in. Anything else
    // takes the portable body below, which is also the exactness oracle.
    if constexpr ((N == 1 || N == 2) && sizeof(WordType) == 4) {
        impl::bitSlicedPairRegionNeon<N, WordType>(magX, magY, signX, signY, r, total);
        return impl::combineBitSlicedPairs<N>(total);
    }
#endif
    for (size_t y = r.y0; y < r.y1; ++y) {
        // Row pointers hoisted once per row, per ops/reduce.hpp's row bodies: the
        // per-plane indexing is loop-invariant and "each word is read once" should
        // be visible rather than inferable from an inlining argument.
        const WordType* rowX[N];
        const WordType* rowY[N];
        for (size_t p = 0; p < N; ++p) {
            BINCV_ASSERT(magX[p].ptr != nullptr && magY[p].ptr != nullptr,
                         "covariance: a non-empty view needs a non-null pointer");
            rowX[p] = magX[p].row(y);
            rowY[p] = magY[p].row(y);
        }
        // ===================================================================
        // THE ACCUMULATOR SHAPE IS CHOSEN ON N, AND IT IS MEASURED.
        //
        // item 4 gives window reductions a PER-ROW partial accumulator, to
        // break the serialized dependency chain through popcount latency.
        // measured that worth 1.08x -- AT N = 1, where BitSlicedPairCounts is four
        // counters. At N = 4 it is SIXTY-FOUR, and the per-row zero-and-add is
        // ~3N^2+N adds plus 4N^2 words of zeroing against 1-2 uint64_t words of
        // real work per row. a measurement measured both shapes on the reference device:
        //
        // N 1 2 3 4
        // W vs P 0.917x 1.114x 1.348x 1.248x
        //
        // So the per-row shape PAYS at N = 1 and COSTS above it, and the crossover
        // is between 1 and 2 rather than somewhere in the middle. The selection is
        // `if constexpr` and therefore free: N is already a template parameter.
        //
        // Both forms produce BIT-IDENTICAL results -- they add the same integers in
        // a different order, and size_t addition is associative -- so this changes
        // timing and nothing else. tests/test_covariance.cpp's N sweeps are what
        // hold that.
        //
        // The noise floor was measured rather than assumed: compiled the SAME
        // arm into two translation units and timed both. On the device that spread
        // is 0.0-0.3%, so the numbers above are real; on an x86 development machine
        // it reaches 10.6%, which is why declined to close this question on a
        // single-binary A/B and was right to.
        // ===================================================================
        if constexpr (N == 1) {
            impl::BitSlicedPairCounts<N> row;
            impl::bitSlicedPairRowRegion<N, WordType>(rowX, rowY, signX.row(y), signY.row(y), r,
                                                      row);
            total.add(row);
        } else {
            impl::bitSlicedPairRowRegion<N, WordType>(rowX, rowY, signX.row(y), signY.row(y), r,
                                                      total);
        }
    }
    return impl::combineBitSlicedPairs<N>(total);
}

/// @brief The 2x2 gradient covariance of an N-bit signed derivative pair over
/// `window`. **API TIER 3.** This is the spelling specifies.
/// @tparam N The level's bit depth -- 1 for a ternary level, 3/4/5 for the upper
/// pyramid levels a measurement measured.
/// @param dx Horizontal derivative, N-bit -- what `derivativeX` writes from a
/// `QuantMat<N>` level (ops/derivative.hpp).
/// @param dy Vertical derivative, with `dx`'s dimensions and bit depth.
/// @param window Window in pixels, half-open, intersected with the image.
/// @return `{SumIx^2, SumIy^2, SumIxIy}` over the intersection.
///
/// @note **At N == 1 this overload is not the one selected.** `TernaryMat<W>` is
/// `SignedQuantMat<1, W>`, so the ternary overload above is a better match
/// by partial ordering and keeps the single-popcount path. That is a
/// dispatch detail and NOT something a caller has to know, because the two
/// agree exactly -- which is the property tests/test_covariance.cpp pins.
/// @note A thin naming of the view form: the container knows which planes are
/// magnitude and which is sign, and the kernel does not have to. The
/// plane arrays are automatic storage, `N` views each.
/// @note **Never throws.** `constMagnitude(p)` and `constSign` are the checked
/// accessors quantMat.hpp keeps live in every build, and `p` here is a loop
/// variable bounded by N, so the check cannot fire.
template <size_t N, typename WordType>
inline GradientCovariance gradientCovariance(const SignedQuantMat<N, WordType>& dx,
                                             const SignedQuantMat<N, WordType>& dy, Rect window) {
    BINCV_ASSERT(dx.getWidth() == dy.getWidth() && dx.getHeight() == dy.getHeight(),
                 "covariance: the two derivatives must have the same dimensions");
    BinMatConstView<WordType> magX[N];
    BinMatConstView<WordType> magY[N];
    for (size_t p = 0; p < N; ++p) {
        magX[p] = dx.constMagnitude(p);
        magY[p] = dy.constMagnitude(p);
    }
    return gradientCovariance<N, WordType>(magX, magY, dx.constSign(), dy.constSign(), window);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
