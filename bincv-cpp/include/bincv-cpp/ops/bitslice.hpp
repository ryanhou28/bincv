#pragma once

/// @file bitslice.hpp
/// @brief Small-count arithmetic over bit-packed planes (T2.7). **API TIER 3**
///        throughout -- bit-sliced arithmetic has no OpenCV counterpart
///        (ARCHITECTURE 5.1), so nothing here borrows an OpenCV name.
///
/// Three word-level primitives and one view-level kernel:
///
///   maj3(a, b, c)                       bitwise majority of three words
///   bitSlicedSum(inputs, k, outPlanes)  count k one-bit inputs, as PLANES
///   thresholdGE(planes, n, threshold)   compare that count against a constant
///   majority3(a, b, c, dst)             maj3 over three views (D-5)
///
/// This is the arithmetic layer the first two MVP operations sit on
/// (ARCHITECTURE 6.1):
///
///   - **Denoise, median of 3** (ARCHITECTURE 7.1, T3.1). For binary input the
///     median IS the majority, so the whole kernel is maj3 over three views --
///     which is why majority3() exists here rather than being open-coded there.
///   - **Pyramid box 2x2** (ARCHITECTURE 7.2, T3.2/T3.4), *at the first level*.
///     Over a 1-bit source the 2x2 sum is bitSlicedSum() at k = 4 and the
///     requantization that follows is a comparison against constants, i.e.
///     thresholdGE(). That is the level-0 case only, and T3.4's pyrDown needs
///     two further things this file deliberately does not provide -- read the
///     next section before building on it.
///
/// ---------------------------------------------------------------------------
/// WHY A BIT-SLICED SUM AND NOT A POPCOUNT
///
/// D-6 forbids exposing a per-word popcount, and this file is not a way around
/// it -- it is the alternative that makes the prohibition affordable. A popcount
/// answers "how many bits are set in this word", collapsing 64 independent pixels
/// into one scalar; on aarch64 it also pays two register-domain crossings to do
/// it (ARCHITECTURE 6.2). A bit-sliced sum answers a different question: for each
/// of the 64 bit positions independently, how many of the k inputs have that bit
/// set. The answer is not a scalar, it is ceil(log2(k+1)) WORDS -- plane 0 the
/// least significant bit of every lane's count, plane 1 the next, and so on -- so
/// the result is still 64 pixels wide and the next operation is still word
/// parallel. Nothing here crosses to the vector register file and back, and
/// nothing here reduces across lanes.
///
/// So: a reduction (ops/reduce.hpp) counts pixels ACROSS a region and returns a
/// number. This file counts inputs PER PIXEL and returns planes. They are not
/// two spellings of the same thing.
///
/// ---------------------------------------------------------------------------
/// PLANE ORDER, AND HOW MANY PLANES
///
/// `outPlanes[0]` is the least significant bit of the count, `outPlanes[n-1]` the
/// most significant, matching QuantMat's plane order (ARCHITECTURE 4.1). The
/// count is exact and unsaturated: k inputs need bitSlicedSumPlanes(k) ==
/// ceil(log2(k+1)) planes, which is 1 for k = 1, 2 for k = 2 and 3, 3 for k = 4,
/// 4 for k = 9 and 5 for k = 16. **The caller sizes and owns that array** -- no
/// kernel here allocates (CLAUDE.md, hard rules), and the natural call site has
/// it on the stack:
///
///     WordType planes[bincv::bitSlicedSumPlanes(4)];
///     bincv::bitSlicedSum(quad, 4, planes);
///     const WordType atLeastTwo = bincv::thresholdGE(planes, 3, 2);
///
/// `atLeastTwo` is a FULL-WORD mask: every lane of the word is answered,
/// including the lanes past a row's `width`, and at threshold 0 every lane is
/// answered *yes* whatever the planes hold. Storing one into a row's trailing
/// word therefore means masking it, exactly as majority3() does internally:
///
///     row[words - 1] = static_cast<WordType>(
///         atLeastTwo & bincv::impl::rowTailMask<WordType>(width));
///
/// Skip that and padding bits past `width` are left set, which is the D-13
/// failure: every word-wise reduction over the image then over-counts.
///
/// ---------------------------------------------------------------------------
/// WHAT pyrDown STILL NEEDS, AND WHY IT IS NOT HERE
///
/// T3.4's `pyrDown<NOut, NIn>` is "box 2x2 sum, then subsample" over a QuantMat.
/// Two of its three parts have no primitive anywhere in ops/ yet. Both were found
/// by review of THIS file, and are recorded here rather than left for T3.4 to
/// discover, because the header used to read as though the pyramid step were
/// covered:
///
///   1. **The sum is single-bit and equal-weight.** bitSlicedSum counts k inputs
///      each worth exactly one. A 2x2 box over an NIn-bit source adds four values
///      worth up to 2^NIn - 1 each, which this signature cannot express. It can
///      be *faked* by replicating plane p of each pixel 2^p times -- correct, and
///      exponential: k = 4 * (2^NIn - 1), so 4 inputs at NIn = 1 but 124 at
///      NIn = 5, and ARCHITECTURE 7.2 measures NIn = 3, 4 and 5 as the real case
///      for every level above the first. A bit-sliced adder over multi-bit
///      operands is linear in NIn instead. It is not added here because T2.7
///      specifies three single-bit primitives, and the adder's shape -- weighted
///      (word, weight) inputs, or plane-array plus plane-array -- should be fixed
///      by the caller that needs it in T3.4 rather than guessed one task early.
///
///   2. **There is no horizontal decimation, anywhere.** Vertical decimation is
///      free: a BinMatConstView with twice the stride and half the height reads
///      every other row and costs nothing. Horizontal decimation wants output bit
///      j to come from input bit 2j, and no kernel in ops/ expresses it --
///      logic.hpp is pointwise in the lane, shift.hpp moves every lane by the
///      same amount, and this file's primitives are per-lane. The two known
///      routes are a per-pixel at()/set() loop (slow, no extra memory) and a
///      log2(width) big-integer unshuffle (word-parallel, but frame-sized
///      constant masks), which is speed against footprint -- the trade CLAUDE.md
///      forbids settling by argument. It is registered as **E-8**
///      (ARCHITECTURE 9) and gates T3.4.
///
/// ARCHITECTURE 6.1's primitive table says "nearly every operation in the MVP set
/// is a composition of these"; it has no resample row, and now says so.
///
/// ---------------------------------------------------------------------------
/// THE ADDER NETWORK, AND WHY IT IS THE SLOW ONE
///
/// bitSlicedSum accumulates the inputs ONE AT A TIME into the output planes,
/// rippling a carry upward: adding a single bit to a running total is a chain of
/// half adders (`sum = acc ^ carry`, `carry = acc & carry`, two operations each),
/// and the chain is cut at the plane where the running total provably cannot
/// carry any further. That is not the cheapest network. The textbook form for
/// k = 4 is two half adders and one full adder -- 9 operations against this
/// file's 16 -- and for k = 9 a carry-save tree of 3:2 compressors beats the
/// ripple by more.
///
/// The ripple is here because T2.7 says in as many words to prefer a correct
/// reference over a clever minimal network, and the reason is what a wrong adder
/// costs: every pyramid level and every denoised frame in the MVP is built on it,
/// and the corruption would be a few pixels per frame rather than a crash. This
/// form is ONE loop nest that is correct for every k, with an invariant a reader
/// can check in a line (before input i the total is at most i, so it occupies
/// bitSlicedSumPlanes(i) planes and cannot carry out of
/// bitSlicedSumPlanes(i+1)); a per-k tree is a different shape per k, and the
/// shapes that matter -- 4 and 9 -- are exactly the ones the MVP depends on.
///
/// Phase 5 may replace the body with a compressor tree. The interface does not
/// change when it does, and tests/test_bitslice.cpp enumerates every one of the
/// 2^k input patterns for each k it tests, so a replacement is proven rather than
/// argued.
///
/// ---------------------------------------------------------------------------
/// NO HEAP, NO THROW, NO ALIASING BETWEEN inputs AND outPlanes
///
/// Everything here is BINCV_ASSERT-checked and undefined in release, exactly as
/// at() is (ARCHITECTURE 5.3). The one contract worth stating twice:
/// `bitSlicedSum` accumulates IN the destination planes, so `outPlanes` must not
/// overlap `inputs`. The alternative -- a scratch accumulator -- would either
/// allocate or add a caller-provided buffer to the signature, and the call sites
/// in the MVP have no reason to overlap the two.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow, impl::destinationAliasIsSafe and
// impl::byteRangesDisjoint -- the row-geometry and D-11 aliasing vocabulary
// shared with ops/logic.hpp and ops/shift.hpp. majority3() is pointwise in the
// word index, so it takes the same half of D-11 that ops/logic.hpp does.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// Word-level primitives
// ---------------------------------------------------------------------------

/// @brief Planes a bit-sliced sum of `k` one-bit inputs needs: ceil(log2(k+1)).
/// @param k Number of single-bit inputs. 0 is legal and needs no plane at all.
/// @return The number of words `bitSlicedSum` writes. **API TIER 3.**
/// @note Equivalently the bit length of `k`, which is how it is computed -- the
///       largest representable total IS k, and a floating-point log2 at a power
///       of two is exactly the place that expression would go wrong.
/// @note constexpr, so `WordType planes[bitSlicedSumPlanes(9)]` is an array and
///       not an allocation.
constexpr size_t bitSlicedSumPlanes(size_t k) {
    size_t planes = 0;
    while (k > 0) {
        ++planes;
        k >>= 1;
    }
    return planes;
}

/// @brief Bitwise majority of three words: `(a & b) | (b & c) | (a & c)`.
///        **API TIER 3.**
/// @return A word whose bit i is set iff at least two of a, b, c have bit i set.
/// @note For binary pixels this is also the MEDIAN of the three (ARCHITECTURE
///       7.1): with values drawn from {0, 1}, the middle of three sorted values
///       is whichever value appears at least twice. That equivalence is what
///       makes T3.1's denoise one expression, and tests/test_bitslice.cpp checks
///       it against a per-pixel median rather than against this formula restated.
/// @note It is also the CARRY of a full adder over a, b, c -- the same gate the
///       adder network below uses -- which is the other way to see why "at least
///       two of three" is a majority.
/// @note No branches, no memory, 64 pixels per word at uint64_t.
template <typename WordType>
constexpr WordType maj3(WordType a, WordType b, WordType c) {
    // The casts are the -Wconversion tax on integer promotion: `a & b` is an int
    // for uint8_t and uint16_t, and storing it back is a narrowing conversion
    // (see the same note in ops/logic.hpp).
    return static_cast<WordType>(static_cast<WordType>(a & b) |
                                 static_cast<WordType>(b & c) |
                                 static_cast<WordType>(a & c));
}

/// @brief Bit-sliced sum of `k` single-bit inputs, lane by lane. **API TIER 3.**
/// @param inputs `k` words. Bit i of `inputs[j]` is input j for lane i.
/// @param k Number of inputs, **each worth one**. Any k is accepted. The arities
///        with an MVP caller are 4 (a 2x2 box over a 1-BIT source, ARCHITECTURE
///        7.2) and 9 (a 3x3 median); k = 3 is tested as a shape, not called --
///        the three-pixel median is maj3, which is one expression rather than a
///        sum and a compare. A 2x2 box over a multi-bit source is NOT k = 4 here;
///        see the file header on what it needs instead.
/// @param outPlanes bitSlicedSumPlanes(k) words, written in full (the caller need
///        not zero them). `outPlanes[p]` bit i is bit p of lane i's total.
/// @note Every lane is independent, and the count for each is exact: bit i of the
///        result planes, read as an unsigned integer LSB first, equals the number
///        of `inputs[j]` whose bit i is set.
/// @note This is the bit-parallel alternative to a per-word popcount, not a way
///        around D-6 -- see the file header. Nothing is reduced across lanes.
/// @note PRECONDITION: `outPlanes` must not overlap `inputs`. The sum is
///        accumulated in place in the destination planes, so an overlap would
///        rewrite inputs the loop has not read yet. Asserted in debug builds.
/// @note Never throws and never allocates. A null pointer with a non-zero count
///        and an overlapping destination are programming errors, reported by
///        BINCV_ASSERT in debug builds and undefined in release.
template <typename WordType>
inline void bitSlicedSum(const WordType* inputs, size_t k, WordType* outPlanes) {
    const size_t planes = bitSlicedSumPlanes(k);

    BINCV_ASSERT(k == 0 || inputs != nullptr,
                 "bitSlicedSum: a non-zero input count needs a non-null inputs pointer");
    BINCV_ASSERT(planes == 0 || outPlanes != nullptr,
                 "bitSlicedSum: a non-zero plane count needs a non-null outPlanes pointer");
    BINCV_ASSERT(k == 0 || planes == 0 ||
                     impl::byteRangesDisjoint(inputs, k * sizeof(WordType),
                                              outPlanes, planes * sizeof(WordType)),
                 "bitSlicedSum: outPlanes must not overlap inputs");

    for (size_t p = 0; p < planes; ++p) outPlanes[p] = static_cast<WordType>(0);

    for (size_t i = 0; i < k; ++i) {
        // Add one single-bit input to the running total: a half-adder chain, from
        // the least significant plane upward, carrying while it must.
        WordType carry = inputs[i];

        // After this input the total is at most i + 1, so it occupies exactly
        // this many planes -- and therefore the carry out of the top one is zero.
        // Cutting the chain here is what keeps the cost near k*log2(k) half adders
        // rather than k*planes of them, without any data-dependent branch.
        const size_t live = bitSlicedSumPlanes(i + 1);

        for (size_t p = 0; p < live; ++p) {
            const WordType sum = static_cast<WordType>(outPlanes[p] ^ carry);
            carry = static_cast<WordType>(outPlanes[p] & carry);
            outPlanes[p] = sum;
        }
        // carry == 0 here by the invariant above. It is not asserted, because the
        // assertion would be a word comparison in the innermost loop of every
        // pyramid level; tests/test_bitslice.cpp proves it exhaustively instead.
    }
}

/// @brief Lanes whose bit-sliced value is >= `threshold`, as a 1-bit mask.
///        **API TIER 3.**
/// @param planes `nPlanes` words, LSB first -- the output of bitSlicedSum(), or
///        any bit-sliced unsigned value in the same layout (a QuantMat pixel row,
///        for instance).
/// @param nPlanes Number of planes. 0 means the value is 0 in every lane.
/// @param threshold The constant to compare against, as an ordinary integer.
/// @return A word whose bit i is set iff lane i's value is >= `threshold`.
/// @note Defined for EVERY threshold, including the two degenerate ends a caller
///        reaches by arithmetic rather than by choice: `threshold == 0` passes
///        every lane (all ones), and a threshold above what `nPlanes` bits can
///        hold passes none (zero). A caller sweeping thresholds 0..k+1 over a
///        k-input sum -- which is what T3.2's requantization does -- never has to
///        special-case its own loop bounds.
/// @note MSB-first, tracking two masks: `greater` (a more significant bit has
///        already decided this lane ABOVE the threshold) and `notLess` (no more
///        significant bit has decided it BELOW). A lane that reaches the end in
///        neither state is exactly equal, so `>=` is `greater | notLess` -- which
///        is why the comparison needs no subtraction and no borrow chain.
/// @note `notLess` is deliberately NOT narrowed when a value bit beats the
///        threshold: that lane is added to `greater` in the same step, and
///        `greater | notLess` cannot tell the difference. The narrowing was
///        written first, and is measurably dead -- removing it changed no result
///        anywhere in tests/test_bitslice.cpp's exhaustive (value, threshold)
///        enumeration, which is every input this function has for nPlanes <= 5.
///        Two operations per plane, in a primitive every pyramid level runs.
/// @note The per-plane branch is on a bit of `threshold`, not on pixel data, so
///        it is perfectly predicted -- and with the usual compile-time constant
///        threshold it disappears entirely.
/// @note THE RESULT IS A FULL WORD AND HAS NO NOTION OF `width`. Every lane is
///        answered, including lanes past a row's last pixel -- whose planes hold
///        whatever the caller's padding held -- and `threshold == 0` returns all
///        ones whatever the planes hold, which is the case a caller sweeping
///        thresholds from 0 reaches by arithmetic rather than by choice. A caller
///        storing this into a view's trailing word must AND it with
///        `impl::rowTailMask<WordType>(width)` first; otherwise padding bits past
///        `width` are left set and every word-wise reduction over that image
///        over-counts (D-13). majority3() below masks internally because it owns
///        its destination -- this function returns a word and cannot.
/// @note Never throws and never allocates.
template <typename WordType>
inline WordType thresholdGE(const WordType* planes, size_t nPlanes, unsigned threshold) {
    BINCV_ASSERT(nPlanes == 0 || planes != nullptr,
                 "thresholdGE: a non-zero plane count needs a non-null planes pointer");

    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    const WordType none = static_cast<WordType>(0);

    // Every unsigned value is >= 0, whatever the planes hold -- including no
    // planes at all.
    if (threshold == 0u) return allOnes;
    if (nPlanes == 0) return none;

    // A threshold no value of nPlanes bits can reach. Checked before the loop
    // because the loop only ever inspects the low nPlanes bits of the threshold,
    // and would otherwise report `>=` for a threshold whose high bits are set.
    constexpr size_t thresholdBits = sizeof(unsigned) * 8;
    if (nPlanes < thresholdBits && (threshold >> nPlanes) != 0u) return none;

    WordType greater = none;
    WordType notLess = allOnes;

    for (size_t p = nPlanes; p-- > 0;) {
        const WordType bit = planes[p];
        // Planes above the width of `unsigned` see a threshold bit of 0: the
        // range check above already established that the threshold fits.
        const bool thresholdBitSet =
            (p < thresholdBits) && (((threshold >> p) & 1u) != 0u);

        if (thresholdBitSet) {
            // Value bit 0 against threshold bit 1: strictly less, decided, and it
            // can never recover -- every lower bit is worth less than this one.
            notLess = static_cast<WordType>(notLess & bit);
        } else {
            // Value bit 1 against threshold bit 0: strictly greater, decided.
            // Undecided lanes only -- a lane already ruled less is not in
            // `notLess` and must not be revived here.
            greater = static_cast<WordType>(greater | static_cast<WordType>(notLess & bit));
        }
    }

    return static_cast<WordType>(greater | notLess);
}

namespace impl {

/// @brief The majority3 kernel body: dst = maj3(a, b, c), word-wise, padding
///        cleared.
/// @note Structurally identical to ops/logic.hpp's applyBinary -- same stride
///       handling, same tail mask, same aliasing contract -- with three sources
///       instead of two. It is a separate body rather than a third Op in
///       logic.hpp because that file's dispatch is arity-2 and widening it for
///       one kernel would make every AND compile through a shape it does not use.
template <typename WordType>
inline void applyMajority3(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                           BinMatConstView<WordType> c, BinMatView<WordType> dst) {
    BINCV_ASSERT(a.width == dst.width && a.height == dst.height &&
                     b.width == dst.width && b.height == dst.height &&
                     c.width == dst.width && c.height == dst.height,
                 "majority3: a, b, c and dst must have the same dimensions");
    BINCV_ASSERT(strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     strideCoversARow<WordType>(c.width, c.height, c.stride) &&
                     strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "majority3: every view's stride must cover a whole row");
    BINCV_ASSERT(destinationAliasIsSafe(a, dst) && destinationAliasIsSafe(b, dst) &&
                     destinationAliasIsSafe(c, dst),
                 "majority3: dst must alias an input exactly or not overlap it");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && c.ptr != nullptr && dst.ptr != nullptr,
                 "majority3: a non-empty view needs a non-null pointer");

    const size_t words = minRowWords<WordType>(dst.width);
    const WordType tailMask = rowTailMask<WordType>(dst.width);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    // The single contiguous run, on the same terms as ops/logic.hpp: every view
    // dense, and no partial word to mask. Anything else walks row by row.
    if (tailMask == allOnes && a.stride == words && b.stride == words &&
        c.stride == words && dst.stride == words) {
        const size_t total = words * dst.height;
        const WordType* pa = a.ptr;
        const WordType* pb = b.ptr;
        const WordType* pc = c.ptr;
        WordType* pd = dst.ptr;
        for (size_t i = 0; i < total; ++i) {
            pd[i] = maj3<WordType>(pa[i], pb[i], pc[i]);
        }
        return;
    }

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* ra = a.row(y);
        const WordType* rb = b.row(y);
        const WordType* rc = c.row(y);
        WordType* rd = dst.row(y);

        for (size_t i = 0; i + 1 < words; ++i) {
            rd[i] = maj3<WordType>(ra[i], rb[i], rc[i]);
        }
        // The trailing word masked, so that a source whose padding bits are dirty
        // -- a wrapped buffer's are its caller's, and a view onto a wider image
        // has its neighbours' live pixels there (D-13) -- cannot leave majority
        // bits past `width` in the destination.
        rd[words - 1] = static_cast<WordType>(
            maj3<WordType>(ra[words - 1], rb[words - 1], rc[words - 1]) & tailMask);
    }
}

} // namespace impl

// ---------------------------------------------------------------------------
// The view-level kernel (D-5: views, never containers)
// ---------------------------------------------------------------------------

/// @brief dst = the per-pixel MAJORITY of a, b and c -- which for binary pixels
///        is their MEDIAN. **API TIER 3** (no OpenCV equivalent; cv::medianBlur
///        is a spatial filter over one image, not a pointwise median of three).
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param c Third source view; must have a's dimensions.
/// @param dst Destination view; must have a's dimensions. May be `a`, `b` or `c`
///        exactly (in-place), or share no memory with any of them -- the same
///        halves of D-11 ops/logic.hpp takes, and for the same reason: this
///        kernel is pointwise in the word index.
/// @note This is T3.1's denoise once its three neighbour views exist: the
///        reference three-pixel median filter takes the pixel above, the pixel
///        itself and the pixel to its right (ARCHITECTURE 7.1), which are two
///        shifts (ops/shift.hpp) and this call.
/// @note Padding bits past `width` are zero in the destination on return, even
///        when a source's are not.
/// @note PRECONDITION ON `dst`, identical to ops/logic.hpp's: it must span its
///        image's full width, or end on a word boundary. The trailing partial
///        word is stored masked, so in a sub-width window onto a WIDER image the
///        bits between `width` and the end of that word -- the neighbours' live
///        pixels -- are zeroed. Nothing diagnoses it. Sources are unaffected:
///        nothing past `width` is ever read as a pixel.
/// @note Empty views (width or height 0) are a no-op, not an error.
/// @note Never throws and never allocates. Mismatched dimensions, a stride
///        shorter than a row, and overlapping-but-not-identical views are
///        programming errors: BINCV_ASSERT reports them in debug builds and they
///        are undefined in release.
/// @note There is deliberately NO QuantMat overload. A per-plane majority of
///        three N-bit images is not the median of those images -- bit 3 of the
///        median is not the majority of the three bit 3s -- so an overload
///        looping over plane() would compile, run, and be wrong. The N-bit median
///        is a sorting network over bit-sliced values, and nothing in the MVP
///        asks for one.
template <typename WordType>
inline void majority3(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                      BinMatConstView<WordType> c, BinMatView<WordType> dst) {
    impl::applyMajority3<WordType>(a, b, c, dst);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
