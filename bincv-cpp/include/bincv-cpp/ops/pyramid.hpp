#pragma once

/// @file pyramid.hpp
/// @brief Pyramid downsample -- 2x2 box mean, then subsample (T3.4,
///        ARCHITECTURE 7.2). **API TIER 2**: `pyrDown` has cv::pyrDown's name and
///        role and deliberately different numerics (ARCHITECTURE 5.1), so it is
///        NOT bit-exact against OpenCV and is not meant to be. What it is checked
///        against is stated in "WHAT CORRECTNESS MEANS HERE" below.
///
///   pyrDownWidth / pyrDownHeight     destination extent, ceil(n / 2)
///   pyrDown(srcPlanes, dstPlanes)    the kernel (views, D-5)
///   pyrDown(QuantMat, QuantMat)      the container wrapper that names the planes
///   Pyramid<W, N0, N1, ...>          a ladder of levels, one bit depth each
///
/// This is the operation where **output precision exceeds input precision**: a
/// 2x2 mean of 1-bit pixels has five values, and five values do not fit in one
/// bit. ARCHITECTURE 7.2 measured the reference pipeline's own ladder as 1, 3, 4
/// and 5 bits, which is why QuantMat<N> exists at all.
///
/// ---------------------------------------------------------------------------
/// THE OPERATION, EXACTLY
///
/// For every destination pixel (y, x):
///
///     S = src(2y, 2x) + src(2y, 2x+1) + src(2y+1, 2x) + src(2y+1, 2x+1)
///     dst(y, x) = round( (S / 4) * (2^NOut - 1) / (2^NIn - 1) )
///               = floor( (S * (2^NOut - 1) + 2 * (2^NIn - 1))
///                        / (4 * (2^NIn - 1)) )                    [half-up]
///
/// The second line is what the code computes, in bit-sliced integer arithmetic,
/// with no division and no per-pixel branch. The two lines are equal for every
/// S in [0, 4 * (2^NIn - 1)]; tests/test_pyramid.cpp checks the kernel against
/// the first spelling, evaluated per pixel through at().
///
/// **The rescale is not decoration.** A QuantMat<N> value v means the intensity
/// v / (2^N - 1) -- 1 is white at N = 1 exactly as 255 is white in CV_8U. Keeping
/// the sum's own scale instead (denominator 4 * (2^NIn - 1) rather than
/// 2^NOut - 1) would make white read as 4/7 of full scale at NOut = 3, and the
/// error compounds down the ladder. The mean has to come back onto the output's
/// scale, and that is where (2^NOut - 1) / (2^NIn - 1) comes from.
///
/// ---------------------------------------------------------------------------
/// PROBLEM 1 OF T3.4's SPEC -- THE SUBSAMPLE. Closed before this file existed.
///
/// Vertical decimation is a row index: destination row y reads source rows 2y and
/// 2y+1, so it moves no bits (ops/resample.hpp's rowsDecimatedBy2 is the same
/// arithmetic written as a view; this kernel needs BOTH rows of each pair, so it
/// indexes them directly rather than taking a stride-doubled view of one of them).
/// Horizontal decimation is the word-local unshuffle E-8 chose:
/// impl::gatherEvenBits, from ops/resample.hpp, measured on the reference device
/// as 8.3x-26.4x faster than either route E-8 actually named (X-14, D-17).
///
/// This file calls the WORD primitive rather than the view-level
/// decimateColumnsBy2(), and that is a deliberate fusion in the sense of D-16.
/// The box needs all FOUR phases of the 2x2 grid -- (even col, even row),
/// (odd, even), (even, odd), (odd, odd) -- and decimateColumnsBy2 produces one of
/// them into a destination view. Composing the box out of it would mean four
/// decimated frames per source PLANE (4 * NIn frames of scratch at quarter size,
/// i.e. NIn full frames) and four passes over memory before the first add. Fused,
/// the four phases are four registers, the arithmetic runs on them, and the
/// kernel needs **no scratch BUFFER** -- nothing frame-sized, nothing caller-
/// provided, nothing on the heap. It does use a bounded stack frame; the number
/// is in promise 2 below, and it is not the one this file used to quote. The odd
/// phase is the same gather over `word >> 1`, which costs one shift.
///
/// ---------------------------------------------------------------------------
/// PROBLEM 2 OF T3.4's SPEC -- THE N-BIT BOX. Closed here, and this is the part
/// worth reading.
///
/// ops/bitslice.hpp's `bitSlicedSum` counts k inputs **each worth one**. Over a
/// 1-bit source the 2x2 box is exactly that at k = 4. Over an NIn-bit source the
/// only way to spell the box with it is to replicate plane p of each pixel 2^p
/// times, which is correct and **exponential**: k = 4 * (2^NIn - 1), so 4 inputs
/// at NIn = 1, 60 at NIn = 4 and 124 at NIn = 5 -- and ARCHITECTURE 7.2 measures
/// NIn = 3, 4 and 5 as the real case for every level above the first. That route
/// is still here, as impl::boxSum4Replicated, because a rejected alternative that
/// cannot be run is not evidence; it is what tests/test_pyramid.cpp and
/// benchmark/pyramid_benchmark.cpp measure the shipped route against.
///
/// **The shipped formulation is a bit-sliced multi-bit ADD, and it is linear in
/// NIn.** Three ripple-carry additions in a tree:
///
///     t1 = a + b            NIn full adders,      NIn + 1 planes
///     t2 = c + d            NIn full adders,      NIn + 1 planes
///     S  = t1 + t2          NIn + 1 full adders,  NIn + 2 planes
///                           ---------------------
///                           3 * NIn + 1 full adders
///
/// A full adder is `sum = a ^ b ^ carry` and `carry = maj3(a, b, carry)` -- seven
/// word operations, the maj3 being ops/bitslice.hpp's, and every one of them is
/// word-parallel over 8 to 64 pixels. So the box costs
/// **impl::boxSumFullAdders(NIn) == 3 * NIn + 1** adder stages where the
/// replication route costs 4 * (2^NIn - 1) single-bit accumulate steps:
///
///     NIn                        1     2     3     4     5     8
///     3*NIn + 1  (shipped)       4     7    10    13    16    25
///     4*(2^NIn - 1)  (replicated)4    12    28    60   124  1020
///
/// Equal at NIn = 1 -- which is why T2.7 could ship the single-bit form and call
/// it the box -- and 40x apart by NIn = 8. The replicated route also puts k words
/// on the stack, so its FOOTPRINT is exponential too; impl::boxSum4Replicated
/// therefore refuses to compile above NIn = 5, where that array is already 124
/// words per destination word.
///
/// The requantization that follows is **quadratic in NOut and linear in NIn**,
/// never exponential in either. It is one constant multiply, one constant add and
/// a restoring division by a constant:
///
///     Y = S * (2^NOut - 1)      = (S << NOut) - S     one bit-sliced subtract
///     Y = Y + 2 * (2^NIn - 1)                         one constant add
///     V = Y / (4 * (2^NIn - 1))                       NOut restoring steps
///
/// Multiplying by 2^NOut - 1 is a subtraction rather than a shift-and-add chain
/// because an all-ones constant is one less than a power of two -- the same
/// identity that makes the scale factor cheap is the one that made it necessary.
/// The division is schoolbook restoring division against a CONSTANT divisor, so
/// each step is `thresholdGE` (ops/bitslice.hpp, two operations per plane) plus a
/// masked constant subtract, and there are exactly NOut steps because the
/// quotient provably fits in NOut bits (S <= 4M gives V <= A).
///
/// impl::pyrDownAdderStages(NIn, NOut) reports the whole count. It is not used to
/// size anything -- the loops carry their own bounds -- it exists so the cost
/// claim above is a value a test can assert on rather than a sentence in a
/// comment.
///
/// ---------------------------------------------------------------------------
/// ODD WIDTHS AND HEIGHTS: THE MISSING PARTNER IS REPLICATED
///
/// The destination is ceil(w/2) x ceil(h/2), which is cv::pyrDown's `dsize` and
/// the reference pipeline's. At an odd width the last destination column's 2x2
/// block wants source column w, which does not exist; likewise row h at an odd
/// height, and both at once in the bottom-right corner.
///
/// **binCV replicates the edge pixel** -- the missing partner takes the value of
/// the neighbour it would have paired with, so the block is (w-1, w-1) rather
/// than (w-1, w). This is BORDER_REPLICATE, and it is chosen over the two
/// alternatives for a reason each:
///
///   * **Zero fill** would divide a real edge pixel by four and darken the last
///     column and row of every level. On a frontend whose keypoints live near
///     edges that is a systematic bias, not a rounding difference.
///   * **Dropping the odd column** would make the destination floor(w/2), break
///     the ceil(w/2) size relation OpenCV and the reference both use, and lose a
///     column of image per level.
///
/// Replication also keeps the divisor at 4 everywhere, which is what lets the
/// requantization be one rule with no per-column special case: the sum's range is
/// [0, 4 * (2^NIn - 1)] for interior and edge pixels alike.
///
/// A consequence worth stating: at an odd width the last destination column reads
/// source column w-1 TWICE, so a source whose padding bits are dirty cannot leak
/// there -- the kernel substitutes the live pixel before the arithmetic sees it.
/// Every other source column at or past w maps to a destination column at or past
/// dst.width, i.e. to destination padding, which the trailing mask clears (D-13).
///
/// ---------------------------------------------------------------------------
/// WHAT CORRECTNESS MEANS HERE, GIVEN THAT TIER 2 PROMISES NO BIT-EXACTNESS
///
/// Three independent references, all in tests/test_pyramid.cpp:
///
///  1. **A per-pixel reference** over at(), written before the kernel and sharing
///     no expression with it, for every (NIn, NOut) pair the suite covers.
///  2. **The replicated route** (impl::boxSum4Replicated), which reaches the same
///     S through ops/bitslice.hpp's single-bit adder network instead of this
///     file's multi-bit one.
///  3. **The reference pipeline's own BOX_2x2 path** for the 1-bit level-0 case
///     -- cv::blur(2x2) then subsample, with the Gaussian disabled, which is what
///     SEAL/src/keypoint_tracking/pyramids.cpp does.
///
/// **Three documented deviations from that reference. The first is a rounding
/// difference and the other two are not:**
///
///   * **The rounding.** cv::blur on CV_8U does not round the exact mean to
///     nearest. MEASURED, over every operand quadruple tests/test_pyramid.cpp
///     visits, its 2x2 box is `ceil((a + b + c + d) / 4)` -- it rounds the mean
///     UP. That is where ARCHITECTURE 7.2's level-1 value set
///     {0, 64, 128, 192, 255} comes from: the exact means are
///     {0, 63.75, 127.5, 191.25, 255}, and rounded to NEAREST the fourth would be
///     191. binCV rounds once, half up, so at NOut = 8 the two agree exactly
///     wherever the mean is an integer and differ by ONE LSB out of 255 where it
///     is not, with OpenCV never the lower of the pair. **That one-LSB bound is
///     the ARITHMETIC compared on the ALIGNED block** -- cv::blur run at anchor
///     (0, 0) with BORDER_REPLICATE, i.e. binCV's own geometry expressed in
///     OpenCV calls -- and NOT the distance to the reference pipeline's whole
///     level, which also carries the phase deviation below. Measured against the
///     reference's actual BOX_2x2 level the two differ on **74.9% of destination
///     pixels (848 of 3380 identical) by up to 192 of 255** -- measured by
///     tests/test_pyramid.cpp check 6, and all of it phase; the third bullet is
///     where that is accounted for. Rounding a mean up is a
///     systematic brightening of every level; rounding to nearest is the better
///     numeric and Tier 2 is what buys the right to differ. The test checks BOTH
///     rules, so if OpenCV's ever changes it fails here rather than being
///     absorbed.
///
///   * **The cap.** The reference lets precision grow into the CV_8U it was
///     already paying for. binCV caps each level at NOut bits, which is the
///     footprint lever E-7 (T4.1) exists to price. At NOut = 3 the level-1 value
///     set is {0, 2, 4, 5, 7}/7 = {0, .286, .571, .714, 1} where the reference's
///     is {0, .25, .5, .75, 1}: the true quarter-steps are not representable in
///     three bits, and the nearest three-bit values are what binCV stores. NOut is
///     a template parameter precisely so this is the caller's decision and can be
///     measured rather than argued.
///   * **The phase.** `cv::blur(src, dst, cv::Size(2, 2))` uses OpenCV's DEFAULT
///     anchor, which for an even kernel size is (1, 1) -- the window for output
///     (y, x) is source rows 2y-1..2y and columns 2x-1..2x, half a source pixel up
///     and to the left of the aligned block, with the reference's border rule
///     supplying row and column -1. binCV uses the ALIGNED, non-overlapping block
///     (rows 2y..2y+1, columns 2x..2x+1), whose centre is source coordinate
///     2*(y + 0.5): the destination grid maps onto the source with a pure factor
///     of two and no offset. The reference's half-pixel shift is an artifact of an
///     even-sized kernel's anchor, not a modelling choice, and reproducing it
///     would cost a border rule in the hot loop for a geometry that is worse.
///     tests/test_pyramid.cpp pins this claim rather than asserting it in prose:
///     under OpenCV it checks that cv::blur's default-anchor output at (2y, 2x)
///     IS the aligned block at (2y-1, 2x-1).
///
/// ---------------------------------------------------------------------------
/// WHAT A KERNEL HERE PROMISES
///
///  1. **Views, never containers** (D-5). The kernel takes NIn source plane views
///     and NOut destination plane views; the QuantMat overload is a wrapper that
///     names them, exactly as ops/threshold.hpp's binarize is.
///  2. **No allocation and no scratch parameter.** There is no scratch parameter
///     because there is nothing to put in it, and no heap allocation on any path
///     -- tests/test_pyramid.cpp pins both by counting `operator new` across
///     pyrDown and Pyramid::build. What the kernel does use is a
///     **compile-time-bounded stack frame**, and this file used to quote the
///     wrong number for it: NIn + NOut + 2 words is the widest SINGLE intermediate
///     (`scaled`), not the total. The source declares
///     `impl::pyrDownAutomaticWords(NIn, NOut) == 8*NIn + 2*NOut + 6` words --
///     four phase arrays at NIn, boxSum4's two partials at NIn+1, `sum` at NIn+2,
///     `scaled` at NIn+NOut+2 and `value` at NOut -- plus 2*NIn + NOut row
///     pointers. MEASURED with `-fstack-usage` at `-O2 -DNDEBUG`, one call's frame
///     is (aarch64 / g++ 14.2 above, x86_64 / g++ 11.4 below, in bytes):
///
///         NIn/NOut       1/3    3/4    4/5    8/8
///         aarch64  u32   224    416    448    640
///                  u64   272    544    592    912
///         x86_64   u32   288    480    544    720
///                  u64   368    640    704   1040
///
///     -- larger than the declared words alone, because the compiler also spills,
///     and smaller on aarch64, which has twice the registers to spill into. The
///     old sentence implied 48 B at 1/3 and 144 B at 8/8, so it was low by
///     5.7x-7.7x across this table. A caller sizing a Cortex-M stack should budget
///     from here, not from the widest intermediate. The frame does not depend on
///     image size.
///  3. **Never throws.** Shape and aliasing violations are programming errors,
///     reported by BINCV_ASSERT in debug and undefined in release
///     (ARCHITECTURE 5.3).
///  4. **Padding bits stay zero** in every destination plane (D-13).
///  5. **No aliasing between any source plane and any destination plane.**
///     Destination word i reads source words 2i and 2i+1 of two rows, so this is
///     not pointwise in the word index and the in-place half of D-11 does not
///     apply -- impl::viewsShareNoWord, the predicate ops/shift.hpp and
///     ops/resample.hpp take, and for the same reason.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
#include "../quantMat.hpp"
// maj3 (the full adder's carry) and thresholdGE (the division's comparison), plus
// bitSlicedSum and bitSlicedSumPlanes for the replicated route kept for
// comparison. The arithmetic here is a multi-bit extension of that file, not a
// replacement for it.
#include "bitslice.hpp"
// impl::gatherEvenBits -- E-8's word-local unshuffle (D-17). The horizontal
// subsample is that primitive and nothing else.
#include "resample.hpp"
// impl::rowTailMask / minRowWords / bitsPerWord, impl::strideCoversARow and
// impl::viewsShareNoWord -- the row-geometry and D-11 vocabulary every kernel
// under ops/ is written in.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// @brief Destination width of one pyramid level: ceil(srcWidth / 2).
///        **API TIER 3** -- the bare extent computation has no cv:: equivalent
///        (cv::pyrDown computes it internally), exactly as ops/resample.hpp's
///        `decimatedWidth` does.
/// @note cv::pyrDown's own `dsize` rule, and the reference pipeline's. An odd
///       width keeps its last column; that column's 2x2 block replicates the edge
///       pixel (see the file header).
/// @note constexpr, so a caller can size a level without a runtime call.
constexpr size_t pyrDownWidth(size_t srcWidth) { return (srcWidth + 1) / 2; }

/// @brief Destination height of one pyramid level: ceil(srcHeight / 2).
///        **API TIER 3**, with pyrDownWidth.
constexpr size_t pyrDownHeight(size_t srcHeight) { return (srcHeight + 1) / 2; }

namespace impl {

// ---------------------------------------------------------------------------
// Bit-sliced multi-bit arithmetic
//
// Every function here works on an ARRAY OF PLANES, least significant first, the
// layout ops/bitslice.hpp and QuantMat both use (ARCHITECTURE 4.1). Each plane is
// one word, so every one of these is word-parallel over 8 to 64 pixels and none
// of them reduces across lanes (D-6).
//
// They are impl:: rather than public because their shape was fixed by ONE caller
// -- this file -- exactly as ops/bitslice.hpp's header said it should be. If a
// second caller ever wants them, that is the point to decide what the public
// spelling is.
// ---------------------------------------------------------------------------

/// @brief Full-adder stages the 2x2 box sum costs at a given source depth.
/// @return 3 * NIn + 1 -- the tree `(a+b) + (c+d)`, whose three additions are NIn,
///         NIn and NIn+1 stages wide.
/// @note Exists so that "linear in NIn" is a number a test can assert rather than
///       a claim in a comment. Compare boxSum4ReplicatedInputs(NIn).
constexpr size_t boxSumFullAdders(size_t nIn) { return 3 * nIn + 1; }

/// @brief Single-bit inputs the REJECTED replication route would need.
/// @return 4 * (2^NIn - 1). 4 at NIn = 1, 60 at NIn = 4, 1020 at NIn = 8.
/// @note Kept next to boxSumFullAdders so the two can be printed side by side.
constexpr size_t boxSum4ReplicatedInputs(size_t nIn) {
    return 4 * ((size_t{1} << nIn) - 1);
}

/// @brief Total adder-class stages one destination word costs: box plus rescale.
/// @note The rescale is `(S << NOut) - S` over NIn + NOut + 2 planes, a constant
///       add over the same width, and NOut restoring-division steps each of which
///       is a compare plus a masked subtract over that width. Quadratic in NOut,
///       linear in NIn, exponential in neither.
constexpr size_t pyrDownAdderStages(size_t nIn, size_t nOut) {
    const size_t width = nIn + nOut + 2;
    return boxSumFullAdders(nIn)      // the 2x2 sum
           + width                    // multiply by 2^NOut - 1
           + width                    // add the rounding constant
           + nOut * width;            // the restoring division
}

/// @brief Words of automatic storage the shipped `pyrDown` DECLARES per
///        destination word.
/// @return 8 * NIn + 2 * NOut + 6, which is every word-valued array on the path:
///
///         topLeft, topRight, bottomLeft, bottomRight   4 * NIn
///         boxSum4's left and right                     2 * (NIn + 1)
///         sum                                          NIn + 2
///         requantizeBoxSum's scaled                    NIn + NOut + 2
///         value                                        NOut
///
///        On top of it the kernel holds 2 * NIn + NOut row POINTERS and a few
///        scalars. None of it scales with the image.
/// @note **NIn + NOut + 2 -- the widest single intermediate -- was documented as
///       the whole budget until a `-fstack-usage` measurement said otherwise.**
///       The file header carries the measured frames; they exceed even this count,
///       because a compiler that inlines the helpers also spills. So this is a
///       source-level inventory a reader can check against the declarations, not a
///       promise about the emitted frame -- and it is deliberately NOT enforced by
///       restructuring the kernel to match: grouping these into one object, or
///       hoisting the helpers' locals into the caller, both MEASURED LARGER
///       (784 B and 816 B against 704 B at NIn = 4 / NOut = 5 / uint64_t, x86_64),
///       because the compiler can no longer overlap the lifetimes that do not
///       meet. Memory wins, so the code stayed and the number was corrected.
/// @note Does NOT describe impl::pyrDownReplicated, whose rejected box sum adds
///       4 * (2^NIn - 1) words of its own -- half of why it lost.
constexpr size_t pyrDownAutomaticWords(size_t nIn, size_t nOut) {
    return 8 * nIn + 2 * nOut + 6;
}

/// @brief `out = a + b`, bit-sliced. `out` holds max(na, nb) + 1 planes.
/// @note One ripple-carry chain of full adders: `sum = a ^ b ^ carry`,
///       `carry = maj3(a, b, carry)`. maj3 IS a full adder's carry
///       (ops/bitslice.hpp says so at its definition), which is why this file
///       needs no gate of its own.
/// @note `out` must not overlap `a` or `b`: the last plane written is index
///       max(na, nb), one past either operand.
template <typename WordType>
inline void addPlanes(const WordType* a, size_t na, const WordType* b, size_t nb,
                      WordType* out) {
    const size_t n = na > nb ? na : nb;
    WordType carry = 0;
    for (size_t p = 0; p < n; ++p) {
        const WordType x = (p < na) ? a[p] : static_cast<WordType>(0);
        const WordType y = (p < nb) ? b[p] : static_cast<WordType>(0);
        out[p] = static_cast<WordType>(static_cast<WordType>(x ^ y) ^ carry);
        carry = maj3<WordType>(x, y, carry);
    }
    out[n] = carry;
}

/// @brief `out = (v << shift) - v`, i.e. `v * (2^shift - 1)`, bit-sliced.
/// @param v `n` planes, least significant first.
/// @param out `n + shift` planes, written in full.
/// @note This is the whole of the output rescale's multiply. An all-ones constant
///       is one less than a power of two, so multiplying by it is ONE borrow chain
///       rather than `shift` shifted additions -- and a bit-sliced shift is free,
///       being an index offset into the plane array.
/// @note The result never borrows out: `v << shift >= v` for every v, so the final
///       borrow is zero. Not asserted -- it would be a word comparison in the
///       innermost loop of every pyramid level, and tests/test_pyramid.cpp proves
///       the values instead, which is the same argument ops/bitslice.hpp makes for
///       its own carry-out invariant.
template <typename WordType>
inline void multiplyByAllOnes(const WordType* v, size_t n, size_t shift, WordType* out) {
    const size_t total = n + shift;
    WordType borrow = 0;
    for (size_t p = 0; p < total; ++p) {
        const WordType hi =
            (p >= shift && (p - shift) < n) ? v[p - shift] : static_cast<WordType>(0);
        const WordType lo = (p < n) ? v[p] : static_cast<WordType>(0);
        out[p] = static_cast<WordType>(static_cast<WordType>(hi ^ lo) ^ borrow);
        // Full-subtractor borrow: (~hi & lo) | (~hi & borrow) | (lo & borrow),
        // which is maj3 with the minuend inverted.
        borrow = maj3<WordType>(static_cast<WordType>(~hi), lo, borrow);
    }
}

/// @brief `v += c` in every lane, bit-sliced, `c` an ordinary integer constant.
/// @param v `n` planes, modified in place. The sum must fit: no carry out of plane
///        n - 1 is kept.
/// @note The loop runs the full width rather than stopping at the top set bit of
///       `c`, because the carry is a WORD -- some lanes may still be propagating
///       long after the constant's bits have run out, and there is no cheap test
///       for "no lane carries".
template <typename WordType>
inline void addConstant(WordType* v, size_t n, unsigned c) {
    constexpr size_t constantBits = sizeof(unsigned) * 8;
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    WordType carry = 0;
    for (size_t p = 0; p < n; ++p) {
        const bool bitSet = (p < constantBits) && (((c >> p) & 1u) != 0u);
        const WordType b = bitSet ? allOnes : static_cast<WordType>(0);
        const WordType x = v[p];
        v[p] = static_cast<WordType>(static_cast<WordType>(x ^ b) ^ carry);
        carry = maj3<WordType>(x, b, carry);
    }
}

/// @brief `v -= c` in the lanes where `mask` is set, bit-sliced.
/// @param mask A full-word lane mask -- typically thresholdGE's answer.
/// @note The subtrahend's plane p is `mask` where the constant's bit p is set and
///       zero elsewhere, so a lane outside the mask subtracts zero and never
///       borrows. A lane inside it is guaranteed by the caller to hold at least
///       `c`, so it does not borrow out either.
template <typename WordType>
inline void subtractConstantWhere(WordType* v, size_t n, unsigned c, WordType mask) {
    constexpr size_t constantBits = sizeof(unsigned) * 8;
    WordType borrow = 0;
    for (size_t p = 0; p < n; ++p) {
        const bool bitSet = (p < constantBits) && (((c >> p) & 1u) != 0u);
        const WordType b = bitSet ? mask : static_cast<WordType>(0);
        const WordType x = v[p];
        v[p] = static_cast<WordType>(static_cast<WordType>(x ^ b) ^ borrow);
        borrow = maj3<WordType>(static_cast<WordType>(~x), b, borrow);
    }
}

/// @brief `quotient = floor(value / divisor)`, bit-sliced, divisor a constant.
/// @param value `n` planes, CLOBBERED -- it is left holding the remainder.
/// @param divisor The constant divisor. Must be non-zero.
/// @param quotient `nq` planes. The caller guarantees the quotient fits.
/// @note Schoolbook restoring division, most significant quotient bit first: at
///       step q, compare the running remainder against `divisor << q` and subtract
///       it where it fits. The comparison is ops/bitslice.hpp's thresholdGE, which
///       is two operations per plane and needs no borrow chain; the conditional
///       subtract is the masked one above. `nq` steps, each O(n) -- no data-
///       dependent branch and no per-lane control flow anywhere.
/// @note Deliberately NOT a reciprocal multiply. A reciprocal accurate enough to
///       round the same way for every input needs a constant wider than the value
///       itself, and a bit-sliced multiply by a wide constant costs one addition
///       per set bit -- more work than this, and correct only after a search that
///       would have to be justified. Restoring division is exact by construction.
template <typename WordType>
inline void divideByConstant(WordType* value, size_t n, unsigned divisor,
                             WordType* quotient, size_t nq) {
    BINCV_ASSERT(divisor != 0u, "divideByConstant: the divisor must be non-zero");
    for (size_t q = nq; q-- > 0;) {
        const unsigned scaled = divisor << q;
        const WordType fits = thresholdGE<WordType>(value, n, scaled);
        quotient[q] = fits;
        subtractConstantWhere<WordType>(value, n, scaled, fits);
    }
}

// ---------------------------------------------------------------------------
// The 2x2 box sum, both formulations
// ---------------------------------------------------------------------------

/// @brief `sum = a + b + c + d`, four NIn-bit bit-sliced operands, NIn+2 planes.
/// @note THE SHIPPED FORMULATION (see the file header): a tree of three
///       ripple-carry additions, 3*NIn + 1 full-adder stages, linear in NIn.
/// @note `sum` must not overlap any operand.
template <size_t NIn, typename WordType>
inline void boxSum4(const WordType (&a)[NIn], const WordType (&b)[NIn],
                    const WordType (&c)[NIn], const WordType (&d)[NIn],
                    WordType (&sum)[NIn + 2]) {
    WordType left[NIn + 1];
    WordType right[NIn + 1];
    addPlanes<WordType>(a, NIn, b, NIn, left);
    addPlanes<WordType>(c, NIn, d, NIn, right);
    addPlanes<WordType>(left, NIn + 1, right, NIn + 1, sum);
}

/// @brief The same sum through ops/bitslice.hpp's SINGLE-BIT adder network.
/// @note THE REJECTED FORMULATION, kept so that the rejection is reproducible
///       rather than asserted -- the same reason ops/resample.hpp keeps E-8's two
///       losing arms. Plane p of each pixel is replicated 2^p times and the whole
///       lot is handed to bitSlicedSum at k = 4 * (2^NIn - 1). Correct, and both
///       its time and its stack are exponential in NIn.
/// @note Capped at NIn <= 5 on purpose. At NIn = 5 the input array is already 124
///       words per destination word -- 992 bytes at uint64_t -- and at NIn = 8 it
///       would be 1020. A comparison arm may be slow; it may not blow the stack of
///       a caller who instantiated it by accident.
template <size_t NIn, typename WordType>
inline void boxSum4Replicated(const WordType (&a)[NIn], const WordType (&b)[NIn],
                              const WordType (&c)[NIn], const WordType (&d)[NIn],
                              WordType (&sum)[NIn + 2]) {
    static_assert(NIn <= 5,
                  "boxSum4Replicated is the exponential comparison arm; above NIn = 5 its "
                  "input array is larger than any caller should put on the stack. Use "
                  "boxSum4, which is linear in NIn.");
    constexpr size_t k = 4 * ((size_t{1} << NIn) - 1);
    static_assert(bitSlicedSumPlanes(k) == NIn + 2,
                  "the replicated sum must occupy the same planes as the direct one");

    const WordType* const operands[4] = {a, b, c, d};
    WordType inputs[k];
    size_t n = 0;
    for (size_t o = 0; o < 4; ++o) {
        for (size_t p = 0; p < NIn; ++p) {
            const size_t weight = size_t{1} << p;
            for (size_t r = 0; r < weight; ++r) inputs[n++] = operands[o][p];
        }
    }
    BINCV_ASSERT(n == k, "boxSum4Replicated: the replication count must be 4 * (2^NIn - 1)");
    bitSlicedSum<WordType>(inputs, k, sum);
}

/// @brief `dst = round(sum / 4 * (2^NOut - 1) / (2^NIn - 1))`, bit-sliced.
/// @param sum The 2x2 box sum, NIn + 2 planes, in [0, 4 * (2^NIn - 1)].
/// @param out NOut planes.
/// @note The three steps from the file header, and the widths are what make them
///       safe: `Y = sum * (2^NOut - 1) + 2 * (2^NIn - 1)` is at most
///       (2^NIn - 1) * (2^(NOut+2) - 2) < 2^(NIn+NOut+2), so NIn + NOut + 2 planes
///       hold it; and `Y / (4 * (2^NIn - 1)) <= (4A + 2) / 4 <= A`, so NOut
///       quotient planes hold the answer.
/// @note At NIn == NOut this is `round(sum / 4)` and the multiply and divide
///       cancel -- the reference pipeline's case, where the mean stays on the
///       source's own scale. Nothing special-cases it; the generic path computes
///       it, and tests/test_pyramid.cpp checks the identity.
template <size_t NOut, size_t NIn, typename WordType>
inline void requantizeBoxSum(const WordType (&sum)[NIn + 2], WordType (&out)[NOut]) {
    constexpr unsigned maxIn = (1u << NIn) - 1u;    // 2^NIn - 1
    constexpr size_t width = NIn + NOut + 2;

    WordType scaled[width];
    multiplyByAllOnes<WordType>(sum, NIn + 2, NOut, scaled);
    addConstant<WordType>(scaled, width, 2u * maxIn);
    divideByConstant<WordType>(scaled, width, 4u * maxIn, out, NOut);
}

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

/// @brief The shape and aliasing contract pyrDown's kernel takes.
template <size_t NOut, size_t NIn, typename WordType>
inline void checkPyrDownArgs(const BinMatConstView<WordType> (&src)[NIn],
                             BinMatView<WordType> (&dst)[NOut]) {
    // Every BINCV_ASSERT below discards its condition in a release build, which
    // leaves both parameters unread -- and warnings are fatal in the gate.
    static_cast<void>(src);
    static_cast<void>(dst);

    for (size_t p = 0; p < NIn; ++p) {
        BINCV_ASSERT(src[p].width == src[0].width && src[p].height == src[0].height,
                     "pyrDown: every source plane must have the same dimensions");
        BINCV_ASSERT(strideCoversARow<WordType>(src[p].width, src[p].height, src[p].stride),
                     "pyrDown: a source plane's stride is shorter than one of its own rows");
        BINCV_ASSERT(src[p].width == 0 || src[p].height == 0 || src[p].ptr != nullptr,
                     "pyrDown: a non-empty source plane needs a non-null pointer");
    }
    for (size_t q = 0; q < NOut; ++q) {
        BINCV_ASSERT(dst[q].width == dst[0].width && dst[q].height == dst[0].height,
                     "pyrDown: every destination plane must have the same dimensions");
        BINCV_ASSERT(strideCoversARow<WordType>(dst[q].width, dst[q].height, dst[q].stride),
                     "pyrDown: a destination plane's stride is shorter than one of its rows");
        BINCV_ASSERT(dst[q].width == 0 || dst[q].height == 0 || dst[q].ptr != nullptr,
                     "pyrDown: a non-empty destination plane needs a non-null pointer");
        for (size_t p = 0; p < NIn; ++p) {
            BINCV_ASSERT(viewsShareNoWord(src[p], dst[q]),
                         "pyrDown: no destination plane may share a word with a source plane "
                         "-- destination word i reads source words 2i and 2i+1 of two rows, "
                         "so no in-place form exists");
        }
    }
    BINCV_ASSERT(dst[0].width == pyrDownWidth(src[0].width),
                 "pyrDown: dst.width must be ceil(src.width / 2)");
    BINCV_ASSERT(dst[0].height == pyrDownHeight(src[0].height),
                 "pyrDown: dst.height must be ceil(src.height / 2)");
}

/// @brief The four 2x2 phases of one destination word, for one source plane.
/// @param w0 Source word 2i of the row, `w1` source word 2i+1 (zero past the row).
/// @param evenPhase Destination-aligned bits from EVEN source columns.
/// @param oddPhase  The same from ODD source columns.
/// @note This is D-17's word-local unshuffle, once per phase: destination word i
///       covers destination columns [i*B, (i+1)*B), which are source columns
///       [2i*B, (2i+2)*B) -- exactly source words 2i and 2i+1, whatever the word
///       width, so no cross-word carry is involved (ops/resample.hpp's header
///       has the arithmetic). The odd phase is the same gather over `word >> 1`.
template <typename WordType>
inline void gatherPhases(WordType w0, WordType w1, WordType& evenPhase,
                         WordType& oddPhase) {
    constexpr size_t B = bitsPerWord<WordType>();
    evenPhase = static_cast<WordType>(
        gatherEvenBits<WordType>(w0) |
        static_cast<WordType>(gatherEvenBits<WordType>(w1) << (B / 2)));
    oddPhase = static_cast<WordType>(
        gatherEvenBits<WordType>(static_cast<WordType>(w0 >> 1)) |
        static_cast<WordType>(gatherEvenBits<WordType>(static_cast<WordType>(w1 >> 1))
                              << (B / 2)));
}

} // namespace impl

// ===========================================================================
// THE DOWNSAMPLING FILTER AXIS (X-39 / E-21)
//
// The reference defines SIX `LKPyrDownFilterType` variants and binCV implemented
// exactly one, `BOX_2x2`, because that is what `seal_params.yaml` selects. Every
// accuracy result in this project was therefore measured at one point of a
// two-dimensional design space -- and [X-39](../../../EXPERIMENTS.md) showed the
// two axes are NOT independent: a 2x2 box sum of four values has five possible
// outcomes, so it saturates at 3 bits and gains +0.82 yield points from N=2 to
// N=7, where a 5x5 Gaussian gains +3.93. **The filter decides how much depth is
// useful.**
//
// FIVE OF THE SIX ARE SEPARABLE WEIGHTED SUMS and share this one framework;
// only the tap offsets and weights change:
//
//     DIRECT_SUBSAMPLE   offsets { 0}            weights [1]           sum   1
//     BOX_2x2            offsets { 0, +1}        weights [1,1]         sum   2
//     BOX_3x3            offsets {-1, 0, +1}     weights [1,1,1]       sum   3
//     GAUSSIAN_3x3       offsets {-1, 0, +1}     weights [1,2,1]       sum   4
//     GAUSSIAN_5x5       offsets {-2..+2}        weights [1,4,6,4,1]   sum  16
//
// (`MEDIAN_3x3` is an order statistic, not a weighted sum, and is not here.
// X-39 measured it 7.53 points BELOW the box anyway: a median of a mostly-zero
// neighbourhood returns zero, so it erodes a sparse edge map rather than blurring
// it. It belongs in the temporal denoiser, which is where SEAL uses it.)
//
// WHY THE TAPS ARE CHEAP. Output column x reads source column 2x + dx, and
// `gatherPhases` already separates a source row into its even and odd column
// phases packed at destination stride. So source column 2x+2k is the EVEN phase at
// output column x+k, and 2x+2k+1 is the ODD phase at x+k -- every tap is one of
// two phase words shifted by a whole number of OUTPUT columns. No gather is needed
// beyond the one `pyrDown` already does.
//
// A bit-sliced shift is free (it is an index offset into the plane array), so a
// weight of 2^k costs nothing and a weight of 6 = 4+2 costs one extra add. That is
// why the weighted sum is barely more expensive than the box.
// ===========================================================================

/// @brief Which downsampling filter `pyrDown` applies. Names match the reference's
///        `LKPyrDownFilterType` so a configuration can be carried across.
enum class PyrDownFilter {
    DirectSubsample,  ///< no lowpass; X-39 measured -19.68 yield points. Aliases badly.
    Box2x2,           ///< the shipped default and the reference's
    Box3x3,           ///< -0.80 from the Gaussian anchor at N=3
    Gaussian3x3,      ///< -1.28
    Gaussian5x5,      ///< the ANCHOR: what cv::buildOpticalFlowPyramid applies
};

namespace impl {

/// @brief Tap offsets and weights for one separable filter.
/// @note `Radius` is the half-extent, so the tap count is `2*Radius + 1` except for
///       the even-sized `Box2x2`, whose taps are {0, +1} and which is handled by
///       `evenOnly`.
struct FilterTaps {
    int lo;            ///< lowest tap offset
    int hi;            ///< highest tap offset
    unsigned w[5];     ///< weights, indexed from `lo`
    unsigned sum;      ///< sum of the weights, per axis
};

constexpr FilterTaps filterTaps(PyrDownFilter f) {
    switch (f) {
        case PyrDownFilter::DirectSubsample: return FilterTaps{0, 0, {1, 0, 0, 0, 0}, 1};
        case PyrDownFilter::Box2x2:          return FilterTaps{0, 1, {1, 1, 0, 0, 0}, 2};
        case PyrDownFilter::Box3x3:          return FilterTaps{-1, 1, {1, 1, 1, 0, 0}, 3};
        case PyrDownFilter::Gaussian3x3:     return FilterTaps{-1, 1, {1, 2, 1, 0, 0}, 4};
        default:                             return FilterTaps{-2, 2, {1, 4, 6, 4, 1}, 16};
    }
}

/// @brief Planes needed to hold one axis of the weighted sum of `NIn`-bit values.
constexpr size_t axisPlanes(size_t nIn, unsigned weightSum) {
    size_t bits = 0;
    // max value is weightSum * (2^nIn - 1); count its bits.
    unsigned long long maxV = static_cast<unsigned long long>(weightSum) *
                              ((1ull << nIn) - 1ull);
    while (maxV > 0) { ++bits; maxV >>= 1; }
    return bits == 0 ? 1 : bits;
}

/// @brief `acc += (v << Shift)`, bit-sliced. A shift is free: it is an index
///        offset, so this is one ripple add.
/// @note X-42: `AccN`, `VN` and `Shift` are TEMPLATE parameters and there is no
///       staging buffer. The earlier signature took all three at runtime and built
///       the shifted operand in a `tmp` array first -- which at `Shift == 0`, the
///       common case, copied the operand onto itself before adding. With the extents
///       known the loop unrolls and the out-of-range planes fold away: below `Shift`
///       the addend is literally zero, so the stage degenerates to
///       `acc[p] ^= carry; carry &= acc_old[p]` with no operand read at all.
template <size_t AccN, size_t VN, size_t Shift, typename WordType>
inline void addShifted(WordType* acc, const WordType* v) {
    WordType carry = 0;
    for (size_t p = 0; p < AccN; ++p) {
        const WordType y = (p >= Shift && p - Shift < VN) ? v[p - Shift]
                                                          : static_cast<WordType>(0);
        const WordType x = acc[p];
        acc[p] = static_cast<WordType>(static_cast<WordType>(x ^ y) ^ carry);
        carry = maj3<WordType>(x, y, carry);
    }
    // The caller sizes `acc` so the top carry cannot be lost; see axisPlanes.
}

/// @brief One (tap, weight-bit) stage of `weightedAxis`, unrolled at compile time.
/// @note The recursion exists because the decomposition is over TWO compile-time
///       ranges -- taps and the set bits of each tap's weight -- and C++17 has no
///       constexpr for-loop that can instantiate `addShifted` per iteration. Weights
///       in `FilterTaps` never exceed 6, so four bit positions cover every filter.
template <PyrDownFilter F, size_t NIn, size_t OutN, size_t Tap, size_t Bit, typename WordType>
inline void weightedAxisStage(const WordType (*taps)[NIn], WordType* out) {
    constexpr FilterTaps T = filterTaps(F);
    constexpr size_t kTapCount = static_cast<size_t>(T.hi - T.lo + 1);
    if constexpr (Tap < kTapCount) {
        if constexpr (Bit < 4) {
            if constexpr (((T.w[Tap] >> Bit) & 1u) != 0u) {
                addShifted<OutN, NIn, Bit, WordType>(out, taps[Tap]);
            }
            weightedAxisStage<F, NIn, OutN, Tap, Bit + 1, WordType>(taps, out);
        } else {
            weightedAxisStage<F, NIn, OutN, Tap + 1, 0, WordType>(taps, out);
        }
    }
}

/// @brief One axis of a separable weighted sum, bit-sliced.
/// @param taps The filter's tap count operands, each `NIn` planes, in tap order.
/// @param out `OutN` planes, zeroed by this function then filled.
/// @note X-42: the tap count, the weights and the output width all come from `F`,
///       which is a template parameter, so none of them are passed at runtime any
///       more. The previous signature took them as arguments and looped over them --
///       decomposing compile-time weights into set bits at run time, once per output
///       word.
template <PyrDownFilter F, size_t NIn, size_t OutN, typename WordType>
inline void weightedAxis(const WordType (*taps)[NIn], WordType* out) {
    for (size_t p = 0; p < OutN; ++p) out[p] = 0;
    weightedAxisStage<F, NIn, OutN, 0, 0, WordType>(taps, out);
}

/// @brief `requantizeBoxSum` for an arbitrary kernel weight sum.
/// @note The box version is this with `KSum == 4`. Same three steps: scale to the
///       output's full range, add half the divisor to round to nearest, divide.
/// @brief One quotient bit of `divideByConstantT`, unrolled at compile time.
template <unsigned Divisor, size_t N, size_t Q, typename WordType>
inline void divideStage(WordType* value, WordType* quotient) {
    if constexpr (Q > 0) {
        constexpr unsigned kScaled = Divisor << (Q - 1);
        const WordType fits = thresholdGE<WordType>(value, N, kScaled);
        quotient[Q - 1] = fits;
        subtractConstantWhere<WordType>(value, N, kScaled, fits);
        divideStage<Divisor, N, Q - 1, WordType>(value, quotient);
    }
}

/// @brief `divideByConstant` with the divisor and the quotient width known at
///        compile time.
/// @note X-42. Same restoring division, same order, same result -- but
///       `Divisor << q` is a literal at every step, so each `thresholdGE` and each
///       `subtractConstantWhere` specialises against its own constant instead of
///       shifting a runtime value. The runtime spelling stays for callers that have
///       a genuinely runtime divisor.
template <unsigned Divisor, size_t N, size_t NQ, typename WordType>
inline void divideByConstantT(WordType* value, WordType* quotient) {
    static_assert(Divisor != 0u, "divideByConstantT: the divisor must be non-zero");
    divideStage<Divisor, N, NQ, WordType>(value, quotient);
}

template <size_t NOut, size_t NIn, size_t SumPlanes, unsigned KSum, typename WordType>
inline void requantizeWeighted(const WordType* sum, WordType* out) {
    constexpr unsigned maxIn = (1u << NIn) - 1u;
    constexpr size_t width = SumPlanes + NOut;
    WordType scaled[width];
    multiplyByAllOnes<WordType>(sum, SumPlanes, NOut, scaled);
    addConstant<WordType>(scaled, width, (KSum / 2) * maxIn);
    divideByConstantT<KSum * maxIn, width, NOut, WordType>(scaled, out);
}

} // namespace impl

// ---------------------------------------------------------------------------
// The kernel (D-5: views, never containers)
// ---------------------------------------------------------------------------

namespace impl {

/// @brief One pyramid level: 2x2 box mean of `src`, subsampled, at NOut bits --
///        the body BOTH box-sum routes share. **Internal**; the public entry
///        point is `pyrDown` below, which fixes `Replicated` to false.
/// @tparam Replicated Which 2x2 sum to use: `false` is the shipped multi-bit
///         adder, linear in NIn; `true` is the rejected replication route through
///         ops/bitslice.hpp's single-bit network, exponential in NIn. Everything
///         else about the two is identical, which is what lets
///         benchmark/pyramid_benchmark.cpp compare the FORMULATIONS rather than
///         two different operations, and tests/test_pyramid.cpp require them to
///         agree pixel for pixel.
/// @tparam NOut Bits per destination pixel. The CAP: this is the footprint lever
///         E-7 (T4.1) exists to price, and it is a parameter rather than a
///         derived value precisely so that it can be measured rather than argued.
/// @tparam NIn Bits per source pixel.
/// @param src NIn source plane views, least significant plane first, all the same
///        size.
/// @param dst NOut destination plane views, `pyrDownWidth(src.width)` by
///        `pyrDownHeight(src.height)`. Must share no word with any source plane.
///
/// @note `dst(y, x) = round( (S / 4) * (2^NOut - 1) / (2^NIn - 1) )` where S is
///       the 2x2 sum at source rows 2y, 2y+1 and columns 2x, 2x+1, rounded half
///       up. Read the file header for why the rescale is there and what it costs.
/// @note **Deviation from the reference pipeline, recorded per T3.4's spec.** The
///       reference (SEAL, `LKPyrDownFilterType::BOX_2x2`) lets precision grow into
///       CV_8U and never re-binarizes, so its levels are 1/3/4/5 bits by
///       measurement (ARCHITECTURE 7.2). binCV caps each level at NOut and stores
///       the nearest representable value. The reference also anchors its 2x2
///       window half a pixel up and to the left, an artifact of cv::blur's default
///       anchor on an even kernel, and rounds its mean UP rather than to nearest.
///       All three deviations are in the file header with their evidence.
/// @note Odd widths and heights replicate the edge pixel into the missing half of
///       the block, so the divisor stays 4 everywhere (file header).
/// @note Empty destinations (width or height 0) are a no-op, not an error.
/// @note Never throws, never allocates, takes no scratch parameter. Its automatic
///       storage is impl::pyrDownAutomaticWords(NIn, NOut) words plus 2*NIn + NOut
///       row pointers, bounded at compile time and independent of image size; the
///       measured frames are in the file header. NIn + NOut + 2 is the widest
///       single intermediate, not the total.
/// @note Padding bits past `dst.width` are zero in every destination plane on
///       return, even when the source's are not (D-13).
template <size_t NOut, size_t NIn, typename WordType, bool Replicated>
inline void pyrDownRoute(const BinMatConstView<WordType> (&src)[NIn],
                         BinMatView<WordType> (&dst)[NOut]) {
    static_assert(NIn >= 1 && NIn <= 8, "pyrDown: NIn outside QuantMat's supported range");
    static_assert(NOut >= 1 && NOut <= 8, "pyrDown: NOut outside QuantMat's supported range");

    impl::checkPyrDownArgs<NOut, NIn, WordType>(src, dst);

    const size_t dstWidth = dst[0].width;
    const size_t dstHeight = dst[0].height;
    if (dstWidth == 0 || dstHeight == 0) return;

    constexpr size_t B = impl::bitsPerWord<WordType>();
    const size_t srcHeight = src[0].height;
    const size_t srcWords = impl::minRowWords<WordType>(src[0].width);
    const size_t dstWords = impl::minRowWords<WordType>(dstWidth);
    const WordType tail = impl::rowTailMask<WordType>(dstWidth);

    // At an odd source width the LAST destination column has no right partner, so
    // its odd phase is replaced by its even phase -- BORDER_REPLICATE, file header.
    // Precomputed here because it is one bit of one word per row.
    const bool oddWidth = (src[0].width % 2) != 0;
    const size_t lastWord = (dstWidth - 1) / B;
    const WordType lastBit = static_cast<WordType>(static_cast<WordType>(1)
                                                   << ((dstWidth - 1) % B));

    const WordType* srcRow0[NIn];
    const WordType* srcRow1[NIn];
    WordType* dstRow[NOut];

    for (size_t y = 0; y < dstHeight; ++y) {
        // The vertical half of the subsample: a row index, not a bit movement.
        // 2y+1 past the last source row replicates 2y, which is the same edge rule
        // the odd-width branch applies horizontally.
        const size_t row0 = 2 * y;
        const size_t row1 = (2 * y + 1 < srcHeight) ? (2 * y + 1) : row0;

        for (size_t p = 0; p < NIn; ++p) {
            srcRow0[p] = src[p].row(row0);
            srcRow1[p] = src[p].row(row1);
        }
        for (size_t q = 0; q < NOut; ++q) dstRow[q] = dst[q].row(y);

        for (size_t i = 0; i < dstWords; ++i) {
            const size_t lo = 2 * i;

            // The four 2x2 phases, one set of NIn planes each. Source words past
            // the row read as zero: they can only supply destination columns at or
            // past dst.width (see the file header), which the tail mask clears.
            WordType topLeft[NIn];
            WordType topRight[NIn];
            WordType bottomLeft[NIn];
            WordType bottomRight[NIn];
            for (size_t p = 0; p < NIn; ++p) {
                const WordType a0 =
                    (lo < srcWords) ? srcRow0[p][lo] : static_cast<WordType>(0);
                const WordType a1 =
                    (lo + 1 < srcWords) ? srcRow0[p][lo + 1] : static_cast<WordType>(0);
                const WordType b0 =
                    (lo < srcWords) ? srcRow1[p][lo] : static_cast<WordType>(0);
                const WordType b1 =
                    (lo + 1 < srcWords) ? srcRow1[p][lo + 1] : static_cast<WordType>(0);
                impl::gatherPhases<WordType>(a0, a1, topLeft[p], topRight[p]);
                impl::gatherPhases<WordType>(b0, b1, bottomLeft[p], bottomRight[p]);
            }

            if (oddWidth && i == lastWord) {
                // The one destination column whose right partner is source column
                // src.width: replicate the left one. Done before the arithmetic, so
                // a dirty source padding bit never reaches a live destination pixel.
                for (size_t p = 0; p < NIn; ++p) {
                    topRight[p] = static_cast<WordType>(
                        static_cast<WordType>(topRight[p] & static_cast<WordType>(~lastBit)) |
                        static_cast<WordType>(topLeft[p] & lastBit));
                    bottomRight[p] = static_cast<WordType>(
                        static_cast<WordType>(bottomRight[p] &
                                              static_cast<WordType>(~lastBit)) |
                        static_cast<WordType>(bottomLeft[p] & lastBit));
                }
            }

            WordType sum[NIn + 2];
            if constexpr (Replicated) {
                impl::boxSum4Replicated<NIn, WordType>(topLeft, topRight, bottomLeft,
                                                       bottomRight, sum);
            } else {
                impl::boxSum4<NIn, WordType>(topLeft, topRight, bottomLeft, bottomRight, sum);
            }

            WordType value[NOut];
            impl::requantizeBoxSum<NOut, NIn, WordType>(sum, value);

            for (size_t q = 0; q < NOut; ++q) {
                dstRow[q][i] = (i + 1 == dstWords)
                                   ? static_cast<WordType>(value[q] & tail)
                                   : value[q];
            }
        }
    }
}

/// @brief pyrDown with the REJECTED exponential box sum. **Internal.**
/// @note Identical in every other respect, so benchmark/pyramid_benchmark.cpp
///       measures the two formulations of the same operation rather than two
///       different operations -- and tests/test_pyramid.cpp can require them to
///       agree pixel for pixel. Capped at NIn <= 5 by boxSum4Replicated.
template <size_t NOut, size_t NIn, typename WordType>
inline void pyrDownReplicated(const BinMatConstView<WordType> (&src)[NIn],
                              BinMatView<WordType> (&dst)[NOut]) {
    pyrDownRoute<NOut, NIn, WordType, true>(src, dst);
}

} // namespace impl

/// @brief One pyramid level: 2x2 box mean of `src`, subsampled, at NOut bits.
///        **API TIER 2** -- cv::pyrDown's role, deliberately different numerics.
///        This is the entry point; `impl::pyrDownRoute` above is the body, and
///        this fixes its box-sum route to the linear one T3.4 chose.
/// @param src NIn source plane views, least significant plane first.
/// @param dst NOut destination plane views, `pyrDownWidth(src.width)` by
///        `pyrDownHeight(src.height)`. Must share no word with any source plane.
template <size_t NOut, size_t NIn, typename WordType>
inline void pyrDown(const BinMatConstView<WordType> (&src)[NIn],
                    BinMatView<WordType> (&dst)[NOut]) {
    impl::pyrDownRoute<NOut, NIn, WordType, false>(src, dst);
}

namespace impl {

/// @brief The same, taking containers, for the benchmark's convenience.
template <size_t NOut, size_t NIn, typename WordType>
inline void pyrDownReplicated(const QuantMat<NIn, WordType>& src,
                              QuantMat<NOut, WordType>& dst) {
    BinMatConstView<WordType> srcPlanes[NIn];
    BinMatView<WordType> dstPlanes[NOut];
    for (size_t p = 0; p < NIn; ++p) srcPlanes[p] = src.plane(p);
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q] = dst.plane(q);
    pyrDownReplicated<NOut, NIn, WordType>(srcPlanes, dstPlanes);
}

} // namespace impl

/// @brief The QuantMat spelling of pyrDown. **API TIER 2.**
/// @param src Source level, NIn bits per pixel.
/// @param dst Destination level, NOut bits per pixel, already sized
///        `pyrDownWidth(src.getWidth()) x pyrDownHeight(src.getHeight())`.
/// @note A thin wrapper that names the planes, not a second implementation -- the
///       same shape ops/threshold.hpp's binarize uses, and for the same reason
///       (D-5: the kernel binds to views so it compiles once per (WordType, N)
///       whatever the container).
/// @note NIn == 1 comes here too: BinMat IS QuantMat<1> (core/types.hpp), so a
///       binary level 0 needs no separate entry point.
template <size_t NOut, size_t NIn, typename WordType>
inline void pyrDown(const QuantMat<NIn, WordType>& src, QuantMat<NOut, WordType>& dst) {
    BinMatConstView<WordType> srcPlanes[NIn];
    BinMatView<WordType> dstPlanes[NOut];
    for (size_t p = 0; p < NIn; ++p) srcPlanes[p] = src.plane(p);
    for (size_t q = 0; q < NOut; ++q) dstPlanes[q] = dst.plane(q);
    pyrDown<NOut, NIn, WordType>(srcPlanes, dstPlanes);
}

// ---------------------------------------------------------------------------
// The ladder
// ---------------------------------------------------------------------------

namespace impl {

/// @brief The I-th value of a `size_t...` pack. **Internal.**
template <size_t I, size_t... Values>
struct NthSize;
template <size_t Value0, size_t... Rest>
struct NthSize<0, Value0, Rest...> {
    static constexpr size_t value = Value0;
};
template <size_t I, size_t Value0, size_t... Rest>
struct NthSize<I, Value0, Rest...> {
    static constexpr size_t value = NthSize<I - 1, Rest...>::value;
};

/// @brief A pyramid's levels: one QuantMat per bit depth, recursively. **Internal.**
/// @note Recursive rather than a tuple because the levels have DIFFERENT types --
///       QuantMat is templated on N -- and because a header that the core-only,
///       dependency-free configuration compiles should not pull in <tuple> for a
///       four-element list.
template <typename WordType, size_t... LevelBits>
struct PyramidLevels;

template <typename WordType, size_t N0>
struct PyramidLevels<WordType, N0> {
    QuantMat<N0, WordType> mat;

    PyramidLevels() = default;
    PyramidLevels(int width, int height, size_t rowAlignment)
        : mat(width, height, rowAlignment) {}

    template <size_t I>
    QuantMat<N0, WordType>& get() {
        static_assert(I == 0, "pyramid level index out of range");
        return mat;
    }
    template <size_t I>
    const QuantMat<N0, WordType>& get() const {
        static_assert(I == 0, "pyramid level index out of range");
        return mat;
    }

    size_t words() const { return mat.sizeInWords(); }
    void buildDown() {}
};

template <typename WordType, size_t N0, size_t N1, size_t... Rest>
struct PyramidLevels<WordType, N0, N1, Rest...> {
    QuantMat<N0, WordType> mat;
    PyramidLevels<WordType, N1, Rest...> rest;

    PyramidLevels() = default;
    PyramidLevels(int width, int height, size_t rowAlignment)
        : mat(width, height, rowAlignment),
          rest(static_cast<int>(pyrDownWidth(width > 0 ? static_cast<size_t>(width) : 0)),
               static_cast<int>(pyrDownHeight(height > 0 ? static_cast<size_t>(height) : 0)),
               rowAlignment) {}

    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return mat;
        } else {
            return rest.template get<I - 1>();
        }
    }
    template <size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return mat;
        } else {
            return rest.template get<I - 1>();
        }
    }

    size_t words() const { return mat.sizeInWords() + rest.words(); }

    void buildDown() {
        pyrDown<N1, N0, WordType>(mat, rest.mat);
        rest.buildDown();
    }
};

} // namespace impl

/// @brief A pyramid: one QuantMat per level, each at its own bit depth.
/// @tparam WordType The storage word type, shared by every level (D-1, D-14).
/// @tparam LevelBits One bit depth per level, level 0 first --
///         `Pyramid<uint32_t, 1, 3, 4, 5>` is the ladder ARCHITECTURE 7.2
///         measured on the reference pipeline.
///
/// @note **The bit depths are a template parameter list, not a runtime vector.**
///       QuantMat is templated on N (D-1's word-type templating and 4.1's plane
///       layout both depend on it), so levels of different depths have different
///       types and a runtime container of them would need type erasure. Making the
///       ladder compile-time also makes the footprint a compile-time consequence
///       of the declaration, which is the property E-7 is going to be weighing.
/// @note **API TIER 2**, with pyrDown. cv::buildPyramid's role; different numerics
///       and a caller-chosen bit depth per level, which OpenCV has no way to
///       express.
/// @note Levels are sized by `pyrDownWidth` / `pyrDownHeight` from level 0, so an
///       odd extent keeps its last column or row (see pyrDown). The ladder stalls
///       at 1 pixel: ceil(1/2) is 1.
/// @note This is a CONTAINER, not a kernel: it owns its levels and allocates them
///       in its constructor. The no-heap rule binds kernels
///       (CLAUDE.md, hard rules), and `build()` allocates nothing.
template <typename WordType, size_t... LevelBits>
class Pyramid {
    static_assert(sizeof...(LevelBits) >= 1, "a pyramid needs at least one level");

public:
    /// Number of levels, level 0 included.
    static constexpr size_t Levels = sizeof...(LevelBits);

    /// @brief Bits per pixel at level I.
    template <size_t I>
    static constexpr size_t levelBits() {
        return impl::NthSize<I, LevelBits...>::value;
    }

    /// @brief The type of level I.
    template <size_t I>
    using Level = QuantMat<impl::NthSize<I, LevelBits...>::value, WordType>;

    /// @brief An empty pyramid that owns nothing.
    Pyramid() = default;

    /// @brief Allocates every level from level 0's extent.
    /// @param width Level 0's width in pixels.
    /// @param height Level 0's height in pixels.
    /// @param rowAlignment Row alignment in bytes, passed to every level (D-4).
    Pyramid(int width, int height,
            size_t rowAlignment = QuantMat<impl::NthSize<0, LevelBits...>::value,
                                           WordType>::DefaultRowAlignment)
        : levels_(width, height, rowAlignment) {}

    /// @brief Level I, mutable. Level 0 is the one a caller fills.
    template <size_t I>
    Level<I>& level() {
        static_assert(I < Levels, "pyramid level index out of range");
        return levels_.template get<I>();
    }
    template <size_t I>
    const Level<I>& level() const {
        static_assert(I < Levels, "pyramid level index out of range");
        return levels_.template get<I>();
    }

    /// @brief Fills levels 1..N-1 by running pyrDown down the ladder.
    /// @note Level 0 is the caller's input and is not touched. Allocates nothing:
    ///       every level already exists, and pyrDown takes no scratch buffer.
    ///       tests/test_pyramid.cpp counts operator new across this call and
    ///       requires zero.
    void build() { levels_.buildDown(); }

    /// @brief Total words across every level -- the pyramid's whole footprint.
    /// @note This is the number E-7 (T4.1) weighs. It is a peak, not a per-buffer
    ///       ratio: the levels coexist, because a tracker reads all of them
    ///       (CLAUDE.md, benchmarking: report peak working set).
    size_t sizeInWords() const { return levels_.words(); }

    /// @brief Total bytes across every level.
    size_t sizeInBytes() const { return sizeInWords() * sizeof(WordType); }

private:
    impl::PyramidLevels<WordType, LevelBits...> levels_;
};

namespace impl {

/// @brief The value at output column x-1, and at x+1, of a phase word.
/// @note Output column x lives at bit `x % B`, so the value at `x-1` is one bit
///       LOWER and reaching it is a left shift. The bit crossing the word boundary
///       comes from the neighbouring phase word.
template <typename WordType>
inline WordType phaseAtMinus1(WordType cur, WordType prev) {
    constexpr size_t B = bitsPerWord<WordType>();
    return static_cast<WordType>((cur << 1) | (prev >> (B - 1)));
}
template <typename WordType>
inline WordType phaseAtPlus1(WordType cur, WordType next) {
    constexpr size_t B = bitsPerWord<WordType>();
    return static_cast<WordType>((cur >> 1) | (next << (B - 1)));
}

/// @brief One pyramid level under an arbitrary SEPARABLE filter. **INTERNAL.**
///
/// See the filter-axis section above for why every tap is a phase word shifted by a
/// whole number of output columns, and why a weight of `2^k` is free.
///
/// **BORDER: source outside the frame reads as ZERO**, which is the rule the shipped
/// `pyrDown` already applies to source words past the row. It is NOT the
/// reference's `BORDER_REFLECT_101`, and the difference shows on a `Radius`-pixel
/// rim of each level. Reflect-101 is a per-pixel index map and is not word-parallel
/// -- the same reason deviation (iii) rejected it for the LK taps. X-39's rule asks
/// each filter to reproduce **its own** definition exactly, and
/// tests/test_pyramid.cpp checks that against a per-pixel integer reference using
/// this same border.
template <PyrDownFilter F, size_t NOut, size_t NIn, typename WordType>
inline void pyrDownFilteredRoute(const BinMatConstView<WordType> (&src)[NIn],
                                 BinMatView<WordType> (&dst)[NOut]) {
    static_assert(NIn >= 1 && NIn <= 8, "pyrDown: NIn outside QuantMat's supported range");
    static_assert(NOut >= 1 && NOut <= 8, "pyrDown: NOut outside QuantMat's supported range");
    checkPyrDownArgs<NOut, NIn, WordType>(src, dst);

    constexpr FilterTaps T = filterTaps(F);
    constexpr size_t kTaps = static_cast<size_t>(T.hi - T.lo + 1);
    constexpr size_t hPlanes = axisPlanes(NIn, T.sum);
    constexpr size_t vPlanes = axisPlanes(hPlanes, T.sum);

    const size_t dstWidth = dst[0].width;
    const size_t dstHeight = dst[0].height;
    if (dstWidth == 0 || dstHeight == 0) return;

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t srcHeight = src[0].height;
    const size_t srcWords = minRowWords<WordType>(src[0].width);
    const size_t dstWords = minRowWords<WordType>(dstWidth);
    const WordType tail = rowTailMask<WordType>(dstWidth);

    auto srcWord = [&](const WordType* row, long long w) -> WordType {
        if (row == nullptr || w < 0 || static_cast<size_t>(w) >= srcWords) {
            return static_cast<WordType>(0);
        }
        return row[static_cast<size_t>(w)];
    };

    for (size_t y = 0; y < dstHeight; ++y) {
        WordType* dstRow[NOut];
        for (size_t q = 0; q < NOut; ++q) dstRow[q] = dst[q].row(y);

        // The vertical support: source rows 2y + dy. Outside the frame reads zero.
        const WordType* vRows[kTaps][NIn];
        for (size_t t = 0; t < kTaps; ++t) {
            const long long r = static_cast<long long>(2 * y) + T.lo + static_cast<long long>(t);
            const bool ok = r >= 0 && r < static_cast<long long>(srcHeight);
            for (size_t p = 0; p < NIn; ++p) {
                vRows[t][p] = ok ? src[p].row(static_cast<size_t>(r)) : nullptr;
            }
        }

        for (size_t i = 0; i < dstWords; ++i) {
            // One horizontally-filtered value per vertical tap.
            WordType hRes[kTaps][hPlanes];
            for (size_t t = 0; t < kTaps; ++t) {
                // Even/odd phases at output words i-1, i, i+1 -- everything the
                // horizontal taps need, from the gather pyrDown already does.
                WordType ev[3][NIn], od[3][NIn];
                for (size_t p = 0; p < NIn; ++p) {
                    for (int k = -1; k <= 1; ++k) {
                        const long long j = static_cast<long long>(i) + k;
                        const long long lo = 2 * j;
                        gatherPhases<WordType>(srcWord(vRows[t][p], lo),
                                               srcWord(vRows[t][p], lo + 1),
                                               ev[k + 1][p], od[k + 1][p]);
                    }
                }
                // Source column 2x+2k is the EVEN phase at x+k; 2x+2k+1 is the ODD
                // phase at x+k. So each tap is one shift of one phase word.
                WordType hTaps[kTaps][NIn];
                for (size_t p = 0; p < NIn; ++p) {
                    for (size_t t2 = 0; t2 < kTaps; ++t2) {
                        const int dx = T.lo + static_cast<int>(t2);
                        const int k = (dx >= 0) ? (dx / 2) : ((dx - 1) / 2);
                        const bool odd = ((dx - 2 * k) != 0);
                        const WordType cur = odd ? od[1][p] : ev[1][p];
                        if (k == 0) {
                            hTaps[t2][p] = cur;
                        } else if (k == -1) {
                            hTaps[t2][p] = phaseAtMinus1<WordType>(cur, odd ? od[0][p] : ev[0][p]);
                        } else {
                            hTaps[t2][p] = phaseAtPlus1<WordType>(cur, odd ? od[2][p] : ev[2][p]);
                        }
                    }
                }
                weightedAxis<F, NIn, hPlanes, WordType>(hTaps, hRes[t]);
            }

            WordType vSum[vPlanes];
            weightedAxis<F, hPlanes, vPlanes, WordType>(hRes, vSum);

            WordType outPlanes[NOut];
            requantizeWeighted<NOut, NIn, vPlanes, T.sum * T.sum, WordType>(vSum, outPlanes);
            const bool last = (i + 1 == dstWords);
            for (size_t q = 0; q < NOut; ++q) {
                dstRow[q][i] = last ? static_cast<WordType>(outPlanes[q] & tail) : outPlanes[q];
            }
        }
    }
    (void)B;
}

} // namespace impl

/// @brief One pyramid level under a chosen downsampling filter. **API TIER 3.**
/// @tparam F Which separable filter (X-39 / E-21). `Box2x2` is the shipped default
///         and the reference's; `Gaussian5x5` is what `cv::buildOpticalFlowPyramid`
///         applies and is X-39's accuracy anchor.
/// @note The existing `pyrDown` is this at `Box2x2`, through a hand-written route
///       that stays because it is what every prior result was measured on and
///       tests/test_pyramid.cpp holds the two to agreement.
template <PyrDownFilter F, size_t NOut, size_t NIn, typename WordType>
inline void pyrDownFiltered(const BinMatConstView<WordType> (&src)[NIn],
                            BinMatView<WordType> (&dst)[NOut]) {
    impl::pyrDownFilteredRoute<F, NOut, NIn, WordType>(src, dst);
}

/// @brief Container spelling of `pyrDownFiltered`.
template <PyrDownFilter F, size_t NOut, size_t NIn, typename WordType>
inline void pyrDownFiltered(const QuantMat<NIn, WordType>& src, QuantMat<NOut, WordType>& dst) {
    BinMatConstView<WordType> s[NIn];
    BinMatView<WordType> d[NOut];
    for (size_t p = 0; p < NIn; ++p) s[p] = src.constPlane(p);
    for (size_t q = 0; q < NOut; ++q) d[q] = dst.plane(q);
    pyrDownFiltered<F, NOut, NIn, WordType>(s, d);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
