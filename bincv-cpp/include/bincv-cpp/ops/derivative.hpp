#pragma once

/// @file derivative.hpp
/// @brief The binarized spatial derivative, `[-1, 0, 1]` on both axes.
/// **API TIER 3** -- see the tier note below.
///
/// The first operation in the project whose output is a MULTI-PLANE SIGNED image,
/// and therefore the first real exercise of `SignedQuantMat` / `TernaryMat`
/// (the design notes). What it produces is what that work’s LK gradient
/// covariance consumes ([the design notes]), so the two planes it writes -- a
/// magnitude stack and a sign plane -- are exactly the arguments
/// `countCovariance(dx.constMagnitude(0), dy.constMagnitude(0), dx.constSign,
/// dy.constSign, window)` takes.
///
/// ---------------------------------------------------------------------------
/// THE OPERATION, READ OUT OF THE REFERENCE RATHER THAN INFERRED
///
/// SEAL/src/keypoint_tracking/gradients.cpp, `calcBinarizedDeriv`:
///
/// cv::Mat kernelX = (cv::Mat_<int>(1, 3) << -1, 0, 1);
/// cv::Mat kernelY = (cv::Mat_<int>(3, 1) << -1, 0, 1);
/// cv::filter2D(src, binarizedX, CV_16S, kernelX);
/// cv::filter2D(src, binarizedY, CV_16S, kernelY);
/// binarizedX *= 16; binarizedY *= 16;
/// cv::merge({binarizedX, binarizedY}, dst);
///
/// Three properties of those four lines are load-bearing, and two of them are the
/// kind that produce a plausible-looking image when they are got backwards. Each
/// was checked against the real `cv::filter2D` rather than against a reading of
/// the documentation; the probes are `Derivative.OpenCvFilter2D_*` in
/// tests/test_derivative.cpp.
///
/// **1. `filter2D` CORRELATES. It does not convolve.** With the anchor at the
/// center of a 1x3 kernel,
///
/// dst(x) = -1*src(x-1) + 0*src(x) + 1*src(x+1) = src(x+1) - src(x-1)
///
/// so the derivative at column x is the RIGHT neighbour minus the LEFT one. A
/// convolution would flip every sign. Measured: a 1x8 row that steps 0 -> 255
/// between columns 3 and 4 gives `dx(3) = +255`, which is `src(4) - src(2)`; a
/// convolution predicts `-255` there.
///
/// **Nothing downstream catches this, and that is why the probe exists.** It is
/// tempting to say that work’s cross term would catch it -- `sumXX` and `sumYY` are
/// popcounts of the MAGNITUDE and a negation cannot touch them, while `sumXY` is
/// the one entry that reads the sign planes. But the inversion negates BOTH
/// derivatives, and `(-Ix)(-Iy) = IxIy`: the whole 2x2 covariance, cross term
/// included, is INVARIANT under it. Measured, on a diagonal edge through the real
/// derivative: a 31x31 window gives `sumXY = -61` before negating both planes and
/// `-61` after, and the whole frame gives `-124` and `-124`
/// (tests/test_covariance.cpp, `Covariance.Identity_*`, the negation case). So
/// `Derivative.OpenCvFilter2D_Direction` is the ONLY thing guarding the tap
/// direction -- not a belt-and-braces check on top of a downstream tripwire.
///
/// **2. `filter2D`'s default border is `BORDER_REFLECT_101`, not zero.**
/// `cv::BORDER_DEFAULT` *is* `BORDER_REFLECT_101` (both are 4, here and in
/// OpenCV). binCV's shifts default to `BORDER_CONSTANT` with fill 0, so
/// taking the default would differ from the reference at every edge column and
/// row. Measured on a 1x8 row with a single set pixel at column 1: the default
/// and an explicit `BORDER_REFLECT_101` both give `dx(0) = 0`, while an explicit
/// `BORDER_CONSTANT` gives `dx(0) = +255`.
///
/// **THIS FILE THEREFORE DEFAULTS TO `BORDER_REFLECT_101`, DELIBERATELY BREAKING
/// WITH the design rule’s DEFAULT**, and the border is still a parameter so a caller can ask
/// for any of the five. Two reasons, and the second is why the deviation is not
/// merely a compatibility tax:
///
/// * It is what the reference does, and this is written against derivatives
/// that agree with the reference at every pixel, borders included.
/// * Reflect-101 makes the derivative EXACTLY ZERO on the first and last
/// column (for d/dx) and row (for d/dy): both taps read the same source
/// pixel, so they cancel. A zero fill instead manufactures a full-strength
/// edge all the way around the frame wherever the second column is set --
/// which the corner response of this would then read as a ring of features
/// along the image border. Reflect-101 is the answer that is both
/// reference-exact and right.
///
/// The one degenerate case is `width == 1` (or `height == 1`), where
/// `impl::borderIndex` returns 0 for both reflect flavours, matching OpenCV; the
/// derivative is then 0 everywhere along that axis. `cv::filter2D` on a 1x1 image
/// returns 0, and so does this.
///
/// **3. THE SCALE FACTOR IS NOT REPRODUCED, AND THAT IS A REPRESENTATIONAL
/// DIFFERENCE RATHER THAN A SEMANTIC ONE.** The reference's source is a
/// binary-VALUED `CV_8U` image holding {0, 255}, and it multiplies the filtered
/// result by 16 into `CV_16S`, so its derivative takes the values
/// {-4080, 0, +4080}. binCV stores the same picture as {0, 1} at one bit per
/// pixel, so its derivative is {-1, 0, +1} -- the same sign and the same
/// zero/non-zero structure, scaled by 4080. Everything downstream in the MVP
/// reads sign and magnitude and nothing else: the design notes's covariance is
/// three population counts over masks, and a common positive factor multiplies
/// every entry of the 2x2 matrix identically, leaving eigenvector directions and
/// the min-eigenvalue ORDERING that selects on unchanged. Reproducing 4080
/// would cost 13 more magnitude planes per pixel to carry no information.
/// tests/test_derivative.cpp compares against the ported reference after dividing
/// its output by 4080, and checks that the division is exact -- i.e. that no
/// other value ever appears.
///
/// ---------------------------------------------------------------------------
/// LEVEL 0: A 1-BIT SOURCE GIVES A TERNARY DERIVATIVE, IN SHIFTS AND MASKS
///
/// With `a = src(x+1)` and `b = src(x-1)` drawn from {0, 1}, `a - b` is
///
/// pos = a & ~b rising edge, +1
/// neg = b & ~a falling edge, -1
/// mag = pos | neg which is a ^ b
/// sign = neg a set sign bit means NEGATIVE (the design notes)
///
/// which is 's spelling and the design notes's, and it is three
/// word operations per 8..64 pixels. `impl::ternaryDifference` is that
/// expression.
///
/// **binCV's `shiftLeft(src, k)` is `dst[c] = src[c + k]`** (ops/shift.hpp's
/// header states the convention and its test sweeps it), so `shiftLeft(src, 1)`
/// IS the `src(x+1)` tap and `shiftRight(src, 1)` is `src(x-1)`. The composed
/// spelling is therefore
///
/// shiftLeft (src, a, 1, BORDER_REFLECT_101);
/// shiftRight(src, b, 1, BORDER_REFLECT_101);
/// pos = a & ~b; neg = b & ~a; mag = pos | neg; sign = neg;
///
/// and tests/test_derivative.cpp runs exactly that, requiring it to agree with
/// this file's fused kernel on every pixel. That is what keeps the inline shift
/// below honest about word boundaries and about the border.
///
/// ---------------------------------------------------------------------------
/// LEVEL >= 1: AN N-BIT SOURCE, AND WHY THE COST IS LINEAR IN N
///
/// Above level 0 the pyramid is not binary -- distinct values grow 2 -> 5 -> 15
/// -> 26 across levels 0..3 (the design notes), i.e. 1 -> 3 -> 4 -> 5 bits -- so
/// the derivative is a bit-sliced subtraction of the two shifted N-bit planes.
/// Its range is [-(2^N - 1), +(2^N - 1)], which is exactly `SignedQuantMat<N>`:
/// N magnitude planes and one sign plane, N+1 planes in all.
///
/// `impl::signedDifferenceRipple` does it in two passes over the plane array:
///
/// 1. **One ripple-borrow subtraction**, N full-subtractor stages. The
/// difference comes out modulo 2^N and the BORROW OUT of the top plane is,
/// by definition, `a < b` -- the sign, obtained for free.
/// 2. **One conditional two's-complement negate**, N half-adder stages, which
/// turns the modular difference into a magnitude in the lanes the borrow
/// marked negative and leaves the others alone.
///
/// That is `derivativeAdderStages(N) == 2*N` adder-class stages: **linear in N**,
/// which is the property this was written to protect. The exponential alternative
/// is the same one rejected for the box sum -- replicate each N-bit operand
/// into unary and feed ops/bitslice.hpp's single-bit adder network -- and it costs
/// `derivativeReplicatedInputs(N) == 2 * (2^N - 1)` single-bit inputs: 2 at N = 1,
/// but 30 at N = 4 and 62 at N = 5, which are the levels the reference pyramid
/// actually reaches. It is not implemented here, and the two constants sit next to
/// each other so the claim is a number a test can assert rather than a sentence.
///
/// **Ternary is the N = 1 instance of this, not a separate operation**, which is
/// what the sign-magnitude convention buys (the design notes). At N = 1 the ripple
/// reduces to `mag = a ^ b`, `sign = ~a & b` algebraically -- the borrow chain has
/// one stage and the conditional negate cancels against it -- so
/// `impl::signedDifference` dispatches to the three-operation ternary spelling and
/// `Derivative.RoutesAgree_*` requires the two to produce identical images at
/// N = 1. The generic route stays reachable as `impl::derivativeXGeneric` /
/// `impl::derivativeYGeneric` precisely so that check can exist.
///
/// **WHAT SIGN-MAGNITUDE COSTS HERE, STATED RATHER THAN HIDDEN.** A two's
/// complement output would be the subtraction alone -- N+1 stages over
/// sign-extended operands -- so costs this operation roughly N-1 extra
/// adder-class stages per destination word, approaching 2x at large N. That is
/// the price of the design notes's covariance being three population counts over
/// masks instead of a bit-sliced multiply, and 7.5 is run 31x31 times per
/// keypoint while this runs once per pixel. The trade is recorded, not measured:
/// no experiment has priced the two-complement alternative end to end, because
/// nothing in the MVP would consume it.
///
/// ---------------------------------------------------------------------------
/// THE CANONICAL-ZERO RULE HOLDS BY CONSTRUCTION, NOT BY A FIX-UP PASS
///
/// `SignedQuantMat`'s convention is that magnitude 0 means zero and the sign bit
/// is then ignored -- "-0" is not a distinct value -- and a kernel that wrote a
/// set sign bit over a zero magnitude would still be legal but would be writing
/// noise into the plane that work’s cross term reads. Nothing here can:
///
/// sign == borrow out of (a - b) == (a < b)
///
/// and `a < b` implies `a != b` implies a non-zero magnitude. The two planes
/// cannot disagree, in either direction, for any input -- so there is no
/// canonicalization pass and none is needed. `Derivative.CanonicalZero_*` sweeps
/// whole frames at every N and word width and asserts that no pixel is ever
/// magnitude 0 with its sign bit set, and the padding bits past `width` are
/// cleared in BOTH the magnitude planes and the sign plane, so a padding pixel is
/// canonical zero too.
///
/// ---------------------------------------------------------------------------
/// WHAT THE KERNEL PROMISES (the ops/ contract)
///
/// 1. **Views, never containers.** The kernels take an array of N source plane
/// views and an array of N magnitude plane views plus one sign plane view,
/// each stride read per row. The `QuantMat` / `SignedQuantMat` spellings at
/// the bottom of this file are one-line wrappers that name the planes.
/// 2. **NO SCRATCH AND NO ALLOCATION**, and this is a choice with a footprint
/// behind it. Written as the composition it is defined as -- two calls into
/// ops/shift.hpp and then the mask -- each axis needs 2N FRAME-SIZED
/// temporaries, which a kernel may not conjure (CLAUDE.md) and which would
/// therefore become caller-provided scratch on the signature. At 640x480,
/// uint32_t, N = 1, that is 2 x 38400 B added to a 38400 B source and a
/// 76800 B destination: the composed form's peak working set is **1.67x the
/// fused form's FOR ONE AXIS** (5 planes against 3). The figure quoted
/// everywhere else -- benchmark/derivative_benchmark.cpp, -- is
/// **1.40x**, which is the same comparison for BOTH AXES (7 planes against
/// 5), because the two scratch frames are reused across the axes and a
/// caller that wants a covariance needs both. 1.40x is the number a footprint
/// claim should use; 1.67x is what one axis in isolation costs. The
/// horizontal neighbours are computed into registers by
/// shifting each source word by one bit; the vertical ones are a row index
/// ( a vertical shift moves no bits). This is the design rule’s finding applied to
/// the operation names as its second caller.
/// 3. **PADDING BITS STAY ZERO** in every destination plane, sign included --
/// the trailing word of each is stored masked, and the sign plane needs that
/// as much as the magnitude planes do, because a padding bit set there is a
/// "negative zero" and therefore a canonical-zero violation.
///
/// The SOURCE's trailing word is deliberately NOT masked, unlike
/// ops/denoise.hpp's. The dirty padding bit that file guards against does
/// move here too -- `cur >> 1` carries bit `width % WordBits` into pixel
/// `width - 1` -- but it lands on precisely the bit the right-border fixup
/// overwrites a moment later, so a mask there is an operation no test can
/// observe. That is measured, not argued, and both halves of the measurement
/// are in the kernel: deleting the mask changes no result (48526 of 48526
/// either way), deleting the fixup from the SHIPPED kernel fails 16772.
///
/// THE TWO ARE COUPLED, AND THE COUPLING IS ITSELF MEASURED. Deleting the
/// fixup while the mask is still present fails 16744; deleting it from the
/// shipped, unmasked kernel fails 16772. The 28-check gap is exactly
/// Derivative.DirtyPadding*, and it is the proof that the mask is dead
/// BECAUSE the fixup is there rather than dead outright.
/// 4. **Never throws.** Mismatched dimensions, a stride shorter than a row, an
/// unknown BorderType and any overlap between a source plane and a
/// destination plane are programming errors, reported by BINCV_ASSERT in
/// debug builds and undefined in release (the design notes). The container
/// wrappers can throw, but only from `plane` / `magnitude`, whose index
/// is a loop variable bounded by N and cannot be out of range.
///
/// ---------------------------------------------------------------------------
/// ALIASING: NO DESTINATION PLANE MAY SHARE A WORD WITH ANY SOURCE PLANE
///
/// The half of earlier work that ops/shift.hpp and ops/denoise.hpp take, and for the same
/// reason: neither axis is pointwise in the word index. Destination word i of
/// `derivativeX` reads source words i-1, i and i+1; destination row y of
/// `derivativeY` reads source rows y-1 and y+1. In place is not supported.
///
/// **The destination planes must also be distinct from each other**, which is a
/// check this operation needs and the single-plane kernels did not: a sign plane
/// that aliased a magnitude plane would compile, run, and quietly produce an
/// image where the canonical-zero rule no longer holds.
///
/// ---------------------------------------------------------------------------
/// PRECONDITION ON EVERY DESTINATION PLANE, identical to ops/logic.hpp's: it must
/// span its image's full width, or end on a word boundary. The trailing partial
/// word is stored masked, so in a sub-width window onto a WIDER image the bits
/// between `width` and the end of that word are zeroed, and nothing diagnoses it.
///
/// Empty views (width or height 0) are a no-op, not an error.
///
/// ---------------------------------------------------------------------------
/// API TIER 3, AND THE NAMES SAY SO
///
/// OpenCV has no operation with these semantics. `cv::Sobel` and `cv::Scharr` are
/// different kernels; `cv::filter2D` is a general correlation over a byte or
/// float image producing a byte or float image, not a sign-magnitude bit-plane
/// pair, and this file deliberately drops the reference's scale factor. So
/// nothing here borrows an OpenCV name (the design notes) and nothing here
/// promises to be a drop-in for one. The BORDER semantics are Tier 1 all the
/// same, being `impl::borderIndex`'s, which is pinned against
/// `cv::borderInterpolate` by tests/test_shift.cpp -- that is what lets the
/// agreement with `cv::filter2D` be exact at the edges as well as the interior.
///
/// The two axes are also kept as two images rather than merged into one
/// two-channel destination the way `calcBinarizedDeriv` does. `cv::merge` exists
/// to feed OpenCV's interleaved-channel kernels; binCV's reductions read plane
/// views, and interleaving would put every second word out of reach of a
/// word-parallel popcount.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow, impl::viewsShareNoWord and the
// impl:: word helpers -- the row-geometry and aliasing vocabulary every
// kernel under ops/ is written in. One copy is the only way the aliasing rule
// cannot drift.
#include "../impl/kernel_util.hpp"
// QuantMat / SignedQuantMat / TernaryMat, for the container spellings at the end.
#include "../quantMat.hpp"
// impl::maj3 -- a full adder's carry, and (with the minuend inverted) a full
// subtractor's borrow. ops/pyramid.hpp's addPlanes says the same at its own
// definition; the gate is enumerated exhaustively by tests/test_bitslice.cpp.
#include "bitslice.hpp"
// impl::borderIndex (cv::borderInterpolate) and impl::isKnownBorderType. The
// border mapping is shared with the shifts rather than restated, which is what
// makes this operation's edges Tier 1 without a second implementation to drift.
#include "shift.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// Cost, as constants a test can assert
// ---------------------------------------------------------------------------

/// @brief Adder-class stages one destination word of the derivative costs.
/// **API TIER 3.**
/// @return `2 * n` -- n full-subtractor stages for `a - b`, then n half-adder
/// stages for the conditional two's-complement negate that turns the
/// modular difference into a magnitude.
/// @note Exists so that "linear in N" is a number rather than a claim in a
/// comment, exactly as `impl::boxSumFullAdders` does for that work’s box sum.
/// Compare `derivativeReplicatedInputs(n)` beside it.
constexpr size_t derivativeAdderStages(size_t n) { return 2 * n; }

/// @brief Single-bit inputs the REJECTED replication route would need.
/// **API TIER 3.**
/// @param n The source depth. **DOMAIN: n < the width of `size_t` in bits.** The
/// kernels `static_assert` N in [1, 8] and `SignedQuantMat` caps N well
/// below that, so every real call is far inside it -- but this is a public
/// `constexpr` taking a `size_t`, and `size_t{1} << 64` is undefined
/// rather than ill-formed, so it would be UB at run time and a silently
/// wrong constant at compile time. Measured with `-fsanitize=undefined`
/// before the guard existed: `n = 64` gave "shift exponent 64 is too large"
/// and returned 0, and `n = 100` returned 137438953470. The guard makes the
/// function TOTAL: out of domain it saturates at `SIZE_MAX`, which is the
/// only answer that keeps "the replication route is worse" true.
/// @return `2 * (2^n - 1)`: 2 at n = 1, 30 at n = 4, 62 at n = 5, 510 at n = 8;
/// `SIZE_MAX` out of domain.
/// @note The same failure mode that work’s `impl::boxSum4ReplicatedInputs` names for
/// the box sum -- unary-encode each operand and hand the result to
/// ops/bitslice.hpp's single-bit adder network. Kept next to
/// `derivativeAdderStages` so the two can be printed side by side; the
/// route itself is not implemented, because nothing would call it.
/// @note These two constants are PUBLIC where `ops/pyramid.hpp`'s identical pair
/// is `impl::`, and the departure is deliberate rather than an oversight:
/// pyramid's pair had one caller (its own file), and this pair has three --
/// this header, tests/test_derivative.cpp and
/// benchmark/derivative_benchmark.cpp, which prints them beside the measured
/// ns/pixel so that measurement’s linear-in-N reading is a comparison against a named
/// number rather than against a sentence.
constexpr size_t derivativeReplicatedInputs(size_t n) {
    return (n >= impl::bitsPerWord<size_t>()) ? static_cast<size_t>(-1)
                                              : 2 * ((size_t{1} << n) - 1);
}

namespace impl {

// ---------------------------------------------------------------------------
// The arithmetic: sign-magnitude difference of two bit-sliced operands
// ---------------------------------------------------------------------------

/// @brief One bit of a packed row, by column. **Internal.**
/// @note Only ever called on the at most two border columns per row, never in
/// the word loop -- the design rule’s lesson is that an edge fixup must cost the
/// boundary and not the row.
template <typename WordType>
inline bool rowBit(const WordType* row, size_t x) {
    return (row[wordIndex<WordType>(x)] & bitMask<WordType>(x)) != 0;
}

/// @brief `a - b` for ONE-BIT operands, as sign and magnitude. **Internal.**
/// @param a The `src(x+1)` tap, `b` the `src(x-1)` tap; both single planes.
/// @param mag Set where the two differ -- the ternary magnitude, |a - b|.
/// @param negative Set where `b` is set and `a` is not, i.e. where a - b == -1.
/// @note 's and the design notes's spelling, kept literally:
/// `pos = a & ~b`, `neg = b & ~a`, `mag = pos | neg`, `sign = neg`.
/// Three word operations over 8..64 pixels.
/// @note It is the N = 1 instance of signedDifferenceRipple below and not a
/// different operation; `Derivative.RoutesAgree_*` requires the two to
/// agree image for image.
template <typename WordType>
inline void ternaryDifference(WordType a, WordType b, WordType& mag, WordType& negative) {
    const WordType pos = static_cast<WordType>(a & static_cast<WordType>(~b));
    const WordType neg = static_cast<WordType>(b & static_cast<WordType>(~a));
    mag = static_cast<WordType>(pos | neg);
    negative = neg;
}

/// @brief `a - b` for N-BIT bit-sliced operands, as sign and magnitude.
/// **Internal.**
/// @param a N planes, least significant first: the `src(x+1)` tap.
/// @param b N planes, the `src(x-1)` tap.
/// @param mag N planes, written in full: |a - b|, which fits in N planes because
/// both operands are in [0, 2^N - 1].
/// @param negative One word: set in the lanes where a < b.
/// @note `mag` must not overlap `a` or `b` -- the subtraction is accumulated in
/// the destination and then negated in place.
/// @note TWO LINEAR PASSES, `derivativeAdderStages(N)` stages in total:
///
/// 1. Ripple-borrow subtract. `diff = a ^ b ^ borrow`,
/// `borrow = maj3(~a, b, borrow)` -- the full-subtractor borrow is
/// maj3 with the minuend inverted, which is the identity
/// ops/pyramid.hpp's multiplyByAllOnes already relies on. After N
/// stages `diff` is `a - b` modulo 2^N and the borrow out is `a < b`.
/// 2. Conditional two's-complement negate. `t = diff ^ negative` then a
/// carry chain seeded with `negative`, so a lane the borrow marked
/// negative gets `~diff + 1 == 2^N - diff == b - a` and every other
/// lane gets `diff` unchanged. No per-lane branch anywhere.
///
/// @note NEITHER CHAIN CARRIES OUT, and both facts are consequences of the same
/// thing rather than assumptions. The negate's carry can only propagate
/// past the top plane when `diff == 0`, and `negative` is set only where
/// `a < b`, which forces `diff != 0`. Not asserted -- it would be a word
/// comparison in the innermost loop of every pyramid level, which is the
/// argument ops/bitslice.hpp makes for its own carry-out invariant --
/// tests/test_derivative.cpp enumerates the values instead.
/// @note THE CANONICAL-ZERO RULE FALLS OUT HERE: `negative` IS the borrow, so it
/// is set only where the operands differ, so no lane can ever leave this
/// function with a set sign over a zero magnitude. See the file header.
template <size_t N, typename WordType>
inline void signedDifferenceRipple(const WordType (&a)[N], const WordType (&b)[N],
                                   WordType (&mag)[N], WordType& negative) {
    WordType borrow = static_cast<WordType>(0);
    for (size_t p = 0; p < N; ++p) {
        const WordType x = a[p];
        const WordType y = b[p];
        mag[p] = static_cast<WordType>(static_cast<WordType>(x ^ y) ^ borrow);
        borrow = maj3<WordType>(static_cast<WordType>(~x), y, borrow);
    }
    negative = borrow;

    WordType carry = negative;
    for (size_t p = 0; p < N; ++p) {
        const WordType t = static_cast<WordType>(mag[p] ^ negative);
        mag[p] = static_cast<WordType>(t ^ carry);
        carry = static_cast<WordType>(t & carry);
    }
}

/// @brief The route selector: the ternary spelling at N == 1, the ripple
/// otherwise. **Internal.**
/// @tparam ForceGeneric True takes the ripple even at N == 1, which is what makes
/// "ternary is the N = 1 instance" a testable claim rather than an
/// algebraic one.
template <size_t N, typename WordType, bool ForceGeneric>
inline void signedDifference(const WordType (&a)[N], const WordType (&b)[N],
                             WordType (&mag)[N], WordType& negative) {
    if constexpr (N == 1 && !ForceGeneric) {
        ternaryDifference<WordType>(a[0], b[0], mag[0], negative);
    } else {
        signedDifferenceRipple<N, WordType>(a, b, mag, negative);
    }
}

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

/// @brief The shape and aliasing contract both derivative kernels take.
/// @note Every BINCV_ASSERT discards its condition in a release build, which
/// leaves the parameters unread -- and warnings are fatal in the gate.
template <size_t N, typename WordType>
inline void checkDerivativeArgs(const BinMatConstView<WordType> (&src)[N],
                                BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                                BorderType borderType) {
    static_cast<void>(src);
    static_cast<void>(mag);
    static_cast<void>(sign);
    static_cast<void>(borderType);

    BINCV_ASSERT(isKnownBorderType(borderType), "derivative: unknown BorderType");

    for (size_t p = 0; p < N; ++p) {
        BINCV_ASSERT(src[p].width == src[0].width && src[p].height == src[0].height,
                     "derivative: every source plane must have the same dimensions");
        BINCV_ASSERT(strideCoversARow<WordType>(src[p].width, src[p].height, src[p].stride),
                     "derivative: a source plane's stride is shorter than one of its own rows");
        BINCV_ASSERT(src[p].width == 0 || src[p].height == 0 || src[p].ptr != nullptr,
                     "derivative: a non-empty source plane needs a non-null pointer");
    }

    BINCV_ASSERT(sign.width == src[0].width && sign.height == src[0].height,
                 "derivative: the sign plane must have the source's dimensions");
    BINCV_ASSERT(strideCoversARow<WordType>(sign.width, sign.height, sign.stride),
                 "derivative: the sign plane's stride is shorter than one of its own rows");
    BINCV_ASSERT(sign.width == 0 || sign.height == 0 || sign.ptr != nullptr,
                 "derivative: a non-empty sign plane needs a non-null pointer");

    for (size_t q = 0; q < N; ++q) {
        BINCV_ASSERT(mag[q].width == src[0].width && mag[q].height == src[0].height,
                     "derivative: every magnitude plane must have the source's dimensions");
        BINCV_ASSERT(strideCoversARow<WordType>(mag[q].width, mag[q].height, mag[q].stride),
                     "derivative: a magnitude plane's stride is shorter than one of its own rows");
        BINCV_ASSERT(mag[q].width == 0 || mag[q].height == 0 || mag[q].ptr != nullptr,
                     "derivative: a non-empty magnitude plane needs a non-null pointer");

        for (size_t p = 0; p < N; ++p) {
            BINCV_ASSERT(viewsShareNoWord(src[p], mag[q]),
                         "derivative: no destination plane may share a word with a source plane "
                         "-- neither axis is pointwise in the word index, so no in-place form "
                         "exists");
        }
        // The destinations must be distinct FROM EACH OTHER as well. A sign plane
        // aliasing a magnitude plane compiles and runs, and quietly produces an
        // image in which the canonical-zero rule no longer holds.
        BINCV_ASSERT(viewsShareNoWord(static_cast<BinMatConstView<WordType>>(mag[q]), sign),
                     "derivative: the sign plane must not share a word with a magnitude plane");
        for (size_t r = q + 1; r < N; ++r) {
            BINCV_ASSERT(viewsShareNoWord(static_cast<BinMatConstView<WordType>>(mag[q]), mag[r]),
                         "derivative: the magnitude planes must not share a word with each other");
        }
    }

    for (size_t p = 0; p < N; ++p) {
        BINCV_ASSERT(viewsShareNoWord(src[p], sign),
                     "derivative: the sign plane must not share a word with a source plane");
    }
}

// ---------------------------------------------------------------------------
// The kernels ( views, never containers)
// ---------------------------------------------------------------------------

/// @brief d/dx: `dst(x, y) = src(x + 1, y) - src(x - 1, y)`. **Internal**; the
/// public entry point is `derivativeX` below, which fixes `ForceGeneric`.
/// @tparam ForceGeneric See impl::signedDifference. False is what ships.
/// @note One pass over the destination, no scratch. The two taps are the source
/// row's words shifted one bit each way, computed into registers; the
/// previous word is carried in a register rather than re-read.
/// @note THE LEFT BORDER IS CARRIED AS A SYNTHETIC "WORD BEFORE WORD 0" whose top
/// bit holds whatever column 0's left neighbour resolves to. That keeps the
/// shift recurrence identical at i == 0 and everywhere else -- the design rule’s rule
/// that an edge fixup must cost the boundary rather than a test per word.
/// The RIGHT border is folded into the trailing word's branch, which the
/// tail mask requires anyway, so it costs nothing extra.
template <size_t N, typename WordType, bool ForceGeneric>
inline void derivativeXRoute(const BinMatConstView<WordType> (&src)[N],
                             BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                             BorderType borderType, bool borderValue) {
    static_assert(N >= 1 && N <= 8, "derivativeX: N outside QuantMat's supported range");

    checkDerivativeArgs<N, WordType>(src, mag, sign, borderType);

    const size_t width = src[0].width;
    const size_t height = src[0].height;
    if (width == 0 || height == 0) return;

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t words = minRowWords<WordType>(width);
    const WordType tailMask = rowTailMask<WordType>(width);
    const WordType lastLiveBit = bitMask<WordType>(width - 1);
    const WordType topBit = static_cast<WordType>(static_cast<WordType>(1) << (B - 1));

    // Which source COLUMN each out-of-image tap reads. Resolved once for the whole
    // image: the border mapping depends on the column and the width, not the row.
    // -1 means BORDER_CONSTANT, i.e. `borderValue` rather than a source pixel.
    const ptrdiff_t leftSrc = borderIndex(static_cast<ptrdiff_t>(-1), width, borderType);
    const ptrdiff_t rightSrc = borderIndex(static_cast<ptrdiff_t>(width), width, borderType);

    const WordType* srcRow[N];
    WordType* magRow[N];
    WordType prev[N];

    for (size_t y = 0; y < height; ++y) {
        for (size_t p = 0; p < N; ++p) {
            srcRow[p] = src[p].row(y);
            magRow[p] = mag[p].row(y);
            const bool bit = (leftSrc < 0)
                                 ? borderValue
                                 : rowBit<WordType>(srcRow[p], static_cast<size_t>(leftSrc));
            prev[p] = bit ? topBit : static_cast<WordType>(0);
        }
        WordType* signRow = sign.row(y);

        for (size_t i = 0; i < words; ++i) {
            const bool last = (i + 1 == words);
            WordType a[N];
            WordType b[N];

            for (size_t p = 0; p < N; ++p) {
                // THE SOURCE'S TRAILING WORD IS NOT MASKED, AND THAT IS MEASURED
                // RATHER THAN ASSUMED. ops/denoise.hpp masks its own because
                // `cur >> 1` moves bit `width % WordBits` -- PADDING, which a
                // wrapped buffer's caller owns and may leave dirty -- into pixel
                // `width - 1`. The same movement happens here, and it lands on
                // exactly the bit the right-border fixup below overwrites: with
                // `width = kB + r` and `r != 0`, the leak arrives at bit r - 1 and
                // `lastLiveBit` IS bit (width - 1) % B == r - 1. At r == 0 the mask
                // is a no-op anyway. So it is dead. Measured: deleting it changes
                // nothing -- 48526 of 48526 checks pass either way, in every
                // configuration, INCLUDING Derivative.DirtyPadding1_* /
                // DirtyPadding3_*, which are built on wrapped buffers whose padding
                // is all ones. Deleting the FIXUP from this kernel as it stands
                // instead fails 16772 of them -- and with the mask restored it
                // fails only 16744. Those 28 checks are DirtyPadding cases, and
                // they are what "the mask is dead BECAUSE the fixup is there"
                // means as a number: remove the fixup and the mask stops being
                // dead in the same instant.
                //
                // The coupling that creates is stated so it cannot be broken
                // silently: in this kernel the last column's right tap is carried
                // by the border fixup alone, and the padding invariant by
                // the store masks below. Anything that narrows the fixup must
                // re-establish both explicitly rather than assume they survived.
                const WordType cur = srcRow[p][i];
                // Bit 0 of word i+1 is pixel (i+1)*WordBits, which is inside
                // `width` for every i that has a successor, so it needs no mask.
                const WordType nxt = last ? static_cast<WordType>(0) : srcRow[p][i + 1];

                a[p] = static_cast<WordType>(static_cast<WordType>(cur >> 1) |
                                             static_cast<WordType>(nxt << (B - 1)));
                b[p] = static_cast<WordType>(static_cast<WordType>(cur << 1) |
                                             static_cast<WordType>(prev[p] >> (B - 1)));
                prev[p] = cur;

                if (last) {
                    // Column width - 1's right tap: the one pixel of this row the
                    // word recurrence cannot supply.
                    const bool bit =
                        (rightSrc < 0) ? borderValue
                                       : rowBit<WordType>(srcRow[p], static_cast<size_t>(rightSrc));
                    a[p] = static_cast<WordType>(
                        static_cast<WordType>(a[p] & static_cast<WordType>(~lastLiveBit)) |
                        (bit ? lastLiveBit : static_cast<WordType>(0)));
                }
            }

            WordType m[N];
            WordType s = static_cast<WordType>(0);
            signedDifference<N, WordType, ForceGeneric>(a, b, m, s);

            for (size_t p = 0; p < N; ++p) {
                magRow[p][i] = last ? static_cast<WordType>(m[p] & tailMask) : m[p];
            }
            // The sign plane is masked too. Padding that reads as "negative zero"
            // is still a violation of the canonical-zero rule, and counts
            // padding bits in every plane, not only the ones carrying magnitude.
            signRow[i] = last ? static_cast<WordType>(s & tailMask) : s;
        }
    }
}

/// @brief d/dy: `dst(x, y) = src(x, y + 1) - src(x, y - 1)`. **Internal**; the
/// public entry point is `derivativeY` below.
/// @note NO BIT MANIPULATION AT ALL: a vertical tap is a row index, so
/// this kernel reads two source rows word for word and never shifts. It is
/// also why the vertical derivative needs no border fixup inside the word
/// loop -- an out-of-image ROW is out of image for every column at once, so
/// it is a whole-row fill.
template <size_t N, typename WordType, bool ForceGeneric>
inline void derivativeYRoute(const BinMatConstView<WordType> (&src)[N],
                             BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                             BorderType borderType, bool borderValue) {
    static_assert(N >= 1 && N <= 8, "derivativeY: N outside QuantMat's supported range");

    checkDerivativeArgs<N, WordType>(src, mag, sign, borderType);

    const size_t width = src[0].width;
    const size_t height = src[0].height;
    if (width == 0 || height == 0) return;

    const size_t words = minRowWords<WordType>(width);
    const WordType tailMask = rowTailMask<WordType>(width);
    // A `true` constant border means every magnitude plane reads all-ones, i.e.
    // the maximum representable value -- the N-bit reading of the design rule’s `bool`, and
    // the same convention cv::erode's default border takes to the depth's maximum.
    const WordType fill =
        borderValue ? static_cast<WordType>(~static_cast<WordType>(0)) : static_cast<WordType>(0);

    const WordType* rowA[N];
    const WordType* rowB[N];
    WordType* magRow[N];

    for (size_t y = 0; y < height; ++y) {
        const ptrdiff_t ya = borderIndex(static_cast<ptrdiff_t>(y) + 1, height, borderType);
        const ptrdiff_t yb = borderIndex(static_cast<ptrdiff_t>(y) - 1, height, borderType);
        const bool haveA = ya >= 0;
        const bool haveB = yb >= 0;

        for (size_t p = 0; p < N; ++p) {
            rowA[p] = haveA ? src[p].row(static_cast<size_t>(ya)) : nullptr;
            rowB[p] = haveB ? src[p].row(static_cast<size_t>(yb)) : nullptr;
            magRow[p] = mag[p].row(y);
        }
        WordType* signRow = sign.row(y);

        for (size_t i = 0; i < words; ++i) {
            const bool last = (i + 1 == words);
            WordType a[N];
            WordType b[N];
            for (size_t p = 0; p < N; ++p) {
                a[p] = haveA ? rowA[p][i] : fill;
                b[p] = haveB ? rowB[p][i] : fill;
            }

            WordType m[N];
            WordType s = static_cast<WordType>(0);
            signedDifference<N, WordType, ForceGeneric>(a, b, m, s);

            // The source's padding bits are NOT masked on the way in here, because
            // nothing moves between lanes on this axis: a dirty padding bit can
            // only produce a wrong PADDING bit, which the store below clears. That
            // is the whole difference from the horizontal kernel, where the same
            // bit would have moved into a live pixel.
            for (size_t p = 0; p < N; ++p) {
                magRow[p][i] = last ? static_cast<WordType>(m[p] & tailMask) : m[p];
            }
            signRow[i] = last ? static_cast<WordType>(s & tailMask) : s;
        }
    }
}

/// @brief The forced-generic routes, for tests/test_derivative.cpp. **Internal.**
/// @note Identical in every other respect, so `Derivative.RoutesAgree_*` compares
/// two FORMULATIONS of one operation rather than two operations -- the same
/// device ops/pyramid.hpp uses for its rejected box sum.
template <size_t N, typename WordType>
inline void derivativeXGeneric(const BinMatConstView<WordType> (&src)[N],
                               BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                               BorderType borderType = BORDER_REFLECT_101,
                               bool borderValue = false) {
    derivativeXRoute<N, WordType, true>(src, mag, sign, borderType, borderValue);
}

template <size_t N, typename WordType>
inline void derivativeYGeneric(const BinMatConstView<WordType> (&src)[N],
                               BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                               BorderType borderType = BORDER_REFLECT_101,
                               bool borderValue = false) {
    derivativeYRoute<N, WordType, true>(src, mag, sign, borderType, borderValue);
}

} // namespace impl

// ---------------------------------------------------------------------------
// The public entry points
// ---------------------------------------------------------------------------

/// @brief Horizontal binarized derivative: `dst(x, y) = src(x+1, y) - src(x-1, y)`,
/// as sign and magnitude. **API TIER 3** (no cv:: equivalent with these
/// semantics), **with TIER 1 BORDER SEMANTICS**.
///
/// @param src N source plane views, least significant plane first, all the same
/// size. N == 1 is the binary case and gives a ternary result.
/// @param mag N destination magnitude plane views, the source's size. Plane 0 is
/// the least significant bit of |dst|.
/// @param sign One destination plane view, the source's size. **A set bit means
/// NEGATIVE** (the design notes), and it is set exactly where
/// `src(x+1) < src(x-1)`.
/// @param borderType How a coordinate outside the image extrapolates. **Defaults
/// to BORDER_REFLECT_101, which is cv::filter2D's default and therefore
/// the reference's** -- deliberately not the design rule’s BORDER_CONSTANT default.
/// See the file header for the measurement and for why reflect-101 is also
/// the right answer rather than only the compatible one.
/// @param borderValue The pixel value outside the image under BORDER_CONSTANT;
/// ignored by the other four types. `true` reads as the maximum
/// representable N-bit value.
///
/// @note **CORRELATION, NOT CONVOLUTION**: the `+1` tap is the RIGHT neighbour.
/// That is `cv::filter2D`'s behavior with `[-1, 0, 1]`, verified by
/// experiment rather than inferred; getting it backwards negates every
/// gradient and silently negates that work’s cross term while leaving `sumXX`
/// and `sumYY` correct. See the file header.
/// @note **The values are {-1, 0, +1}, where the reference's are
/// {-4080, 0, +4080}.** A representational difference, not a semantic one --
/// binCV's pixels are {0, 1} where the reference's are {0, 255}, and the
/// reference then scales by 16. Sign and magnitude structure are identical,
/// and sign and magnitude are all the design notes's covariance reads.
/// @note The result satisfies `SignedQuantMat`'s CANONICAL-ZERO rule by
/// construction: the sign plane is the subtraction's borrow-out, which
/// cannot be set where the operands are equal. No fix-up pass exists or is
/// needed.
/// @note One pass, no scratch buffer, no allocation. Padding bits past `width`
/// are zero in every destination plane on return, sign included, even when
/// the source's are not.
/// @note Never throws (the design notes). Mismatched dimensions, a stride shorter
/// than a row, an unknown BorderType, an overlap between a source and a
/// destination plane, and two destination planes that overlap each other
/// are programming errors: BINCV_ASSERT reports them in debug builds and
/// they are undefined in release.
/// @note Empty views (width or height 0) are a no-op, not an error.
template <size_t N, typename WordType>
inline void derivativeX(const BinMatConstView<WordType> (&src)[N],
                        BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                        BorderType borderType = BORDER_REFLECT_101, bool borderValue = false) {
    impl::derivativeXRoute<N, WordType, false>(src, mag, sign, borderType, borderValue);
}

/// @brief Vertical binarized derivative: `dst(x, y) = src(x, y+1) - src(x, y-1)`,
/// as sign and magnitude. **API TIER 3**, TIER 1 border semantics.
/// @note Everything derivativeX documents applies, with the axis exchanged: the
/// `+1` tap is the row BELOW (larger y), matching `cv::filter2D` with the
/// 3x1 kernel `[-1, 0, 1]`, and the border defaults to BORDER_REFLECT_101
/// so rows 0 and `height - 1` come out exactly zero.
/// @note A vertical tap is a row index and moves no bits, so this kernel
/// does no shifting at all and its per-word cost is the subtraction alone.
template <size_t N, typename WordType>
inline void derivativeY(const BinMatConstView<WordType> (&src)[N],
                        BinMatView<WordType> (&mag)[N], BinMatView<WordType> sign,
                        BorderType borderType = BORDER_REFLECT_101, bool borderValue = false) {
    impl::derivativeYRoute<N, WordType, false>(src, mag, sign, borderType, borderValue);
}

// ---------------------------------------------------------------------------
// The container spellings
// ---------------------------------------------------------------------------

namespace impl {

/// @brief Names `src`'s and `dst`'s planes into the arrays the kernels take.
/// @note A thin wrapper, not a second implementation -- the shape
/// ops/pyramid.hpp and ops/threshold.hpp both use, and for the design rule’s reason.
/// @note This is the only place in the file that can throw, and only from
/// `plane` / `magnitude`, whose bounds check is live in every build
/// (quantMat.hpp says why). The index is a loop variable bounded by N, so
/// it cannot fire.
template <size_t N, typename WordType, bool ForceGeneric, bool Horizontal>
inline void derivativeContainer(const QuantMat<N, WordType>& src,
                                SignedQuantMat<N, WordType>& dst, BorderType borderType,
                                bool borderValue) {
    BinMatConstView<WordType> srcPlanes[N];
    BinMatView<WordType> magPlanes[N];
    for (size_t p = 0; p < N; ++p) {
        srcPlanes[p] = src.plane(p);
        magPlanes[p] = dst.magnitude(p);
    }
    if constexpr (Horizontal) {
        derivativeXRoute<N, WordType, ForceGeneric>(srcPlanes, magPlanes, dst.sign(), borderType,
                                                    borderValue);
    } else {
        derivativeYRoute<N, WordType, ForceGeneric>(srcPlanes, magPlanes, dst.sign(), borderType,
                                                    borderValue);
    }
}

} // namespace impl

/// @brief The QuantMat spelling of derivativeX. **API TIER 3.**
/// @param src Source level, N bits per pixel. `BinMat` IS `QuantMat<1>`
/// (core/types.hpp), so a binary level 0 comes here with no adapter.
/// @param dst Destination, already sized to the source's dimensions. For N == 1
/// that type is `TernaryMat` -- the same type, spelled for the reader.
/// @note `SignedQuantMat<N>` is the exactly-right destination width and not a
/// rounded-up one: the difference of two values in [0, 2^N - 1] lies in
/// [-(2^N - 1), +(2^N - 1)], which is precisely N magnitude planes plus a
/// sign -- N+1 planes in all.
template <size_t N, typename WordType>
inline void derivativeX(const QuantMat<N, WordType>& src, SignedQuantMat<N, WordType>& dst,
                        BorderType borderType = BORDER_REFLECT_101, bool borderValue = false) {
    impl::derivativeContainer<N, WordType, false, true>(src, dst, borderType, borderValue);
}

/// @brief The QuantMat spelling of derivativeY. **API TIER 3.** See derivativeX.
template <size_t N, typename WordType>
inline void derivativeY(const QuantMat<N, WordType>& src, SignedQuantMat<N, WordType>& dst,
                        BorderType borderType = BORDER_REFLECT_101, bool borderValue = false) {
    impl::derivativeContainer<N, WordType, false, false>(src, dst, borderType, borderValue);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
