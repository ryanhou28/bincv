#pragma once

/// @file resample.hpp
/// @brief Horizontal decimation by two -- destination bit *j* is source bit 2*j*
///        (E-8, measured as EXPERIMENTS.md X-14). **API TIER 3**
///        throughout: OpenCV has no operation that subsamples columns without
///        also filtering, so nothing here borrows an OpenCV name -- `cv::resize`
///        and `cv::pyrDown` both mean something else, and ARCHITECTURE 5.1
///        forbids reusing the name for the different thing.
///
/// ---------------------------------------------------------------------------
/// WHY THIS FILE EXISTS AT ALL
///
/// ARCHITECTURE 6.1's primitive table has no resample row, and T3.4's pyramid
/// step is "box 2x2 sum THEN SUBSAMPLE". The subsample half splits into two
/// halves that are nothing alike:
///
///   VERTICAL   free. A BinMatConstView with twice the stride and half the
///              height reads every other row and costs no instructions and no
///              memory. rowsDecimatedBy2() below is that view, written once so
///              the pyramid does not open-code a stride multiplication.
///
///   HORIZONTAL not expressible by anything in ops/. logic.hpp is pointwise in
///              the lane, shift.hpp moves every lane by the SAME amount, and
///              bitslice.hpp is per-lane. Moving bit 2j to bit j moves each lane
///              by a DIFFERENT amount, which is a different kind of operation.
///
/// So the horizontal half is the primitive, and this file is it.
///
/// ---------------------------------------------------------------------------
/// THE THREE ROUTES, AND WHY ALL THREE ARE STILL HERE
///
/// E-8 registered the choice as speed against footprint -- "a per-pixel gather
/// loop, or a log2(width) word-parallel unshuffle that needs frame-sized constant
/// masks" -- which is the trade CLAUDE.md forbids settling by argument. X-14
/// measured it on the reference device, and the three variants it compares are
/// the three impl:: kernels below:
///
///   A  decimateColumnsBy2Gather      per destination pixel, read source bit 2j
///                                    and accumulate into a local word. Word
///                                    literals only, no scratch.
///   B  decimateColumnsBy2Unshuffle   per destination WORD, deinterleave the even
///                                    bits of two source words with log2(WordBits)
///                                    mask/shift steps in registers. Word literals
///                                    only, no scratch.
///   C  decimateColumnsBy2FrameMasked the row as one big integer: log2(rowBits)
///                                    masked shift-or passes over a caller-provided
///                                    scratch row against a caller-built mask table
///                                    at frame width. This is E-8's "frame-masked"
///                                    route, and the only one that costs bytes.
///
/// All three survive because tests/test_resample.cpp checks each against a
/// per-pixel reference at every word width and at widths that are not word
/// multiples, and benchmark/decimate_benchmark.cpp times them against each other;
/// an experiment whose losing arms cannot be rebuilt is not reproducible. The
/// PUBLIC entry point is decimateColumnsBy2(), which is the winner and the only
/// one a caller outside this experiment should name.
///
/// ---------------------------------------------------------------------------
/// THE PAIRING IS EXACT, WHICH IS WHY NONE OF THIS NEEDS A CROSS-WORD CARRY
///
/// Destination word i covers destination columns [i*WordBits, (i+1)*WordBits),
/// which are source columns [2*i*WordBits, 2*(i+1)*WordBits) -- exactly source
/// words 2i and 2i+1, whatever the word width. That is what separates decimation
/// from ops/shift.hpp, where word i of the destination straddles two source words
/// at an arbitrary bit offset and the carry logic is the whole file.
///
/// ---------------------------------------------------------------------------
/// PADDING BITS (D-13), IN BOTH DIRECTIONS
///
/// Every kernel here stores the destination's trailing word masked, so bits at
/// columns >= dst.width are left zero exactly as ops/logic.hpp leaves them.
///
/// The same arithmetic makes variant C's zero-fill of its padded scratch words
/// DEFENSIVE rather than load-bearing: whatever those words held would gather to
/// positions >= dst.width and be masked away. It is still done, because a
/// recurrence whose result depends on uninitialised memory only where nobody
/// looks is one refactor away from depending on it where somebody does.
///
/// A DIRTY SOURCE cannot reach a live destination pixel, and that is arithmetic
/// rather than luck: a source column c >= src.width lands at destination column
/// floor(c/2) >= floor(src.width/2), and dst.width is ceil(src.width/2), so the
/// only destination columns a source padding bit can reach are >= dst.width --
/// destination padding, which the trailing mask clears. So unlike ops/shift.hpp
/// this file needs no extendedRowWord: it does not have to invent a value for
/// out-of-image columns, because none of them is read into the image.
/// tests/test_resample.cpp sweeps a source whose padding is deliberately all ones
/// rather than leaving that as a claim.
///
/// ---------------------------------------------------------------------------
/// WHAT A KERNEL HERE PROMISES
///
///  1. **Views, never containers** (D-5). Strides are read per row and src and
///     dst may differ.
///  2. **No allocation, no throw** (ARCHITECTURE 5.3). Variant C's scratch and
///     mask table are the CALLER's, which is why they are in its signature.
///  3. **Padding bits stay zero** in the destination (D-13).
///  4. **No aliasing between src and dst.** Destination word i reads source words
///     2i and 2i+1, so this is not pointwise in the word index and the in-place
///     half of D-11 does not apply -- impl::viewsShareNoWord, the same predicate
///     ops/shift.hpp takes, and for the same reason.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// impl::rowTailMask / minRowWords / bitsPerWord / lowBitsMask, impl::strideCoversARow
// and impl::viewsShareNoWord -- the row-geometry and D-11 vocabulary every kernel
// under ops/ is written in.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// @brief Destination width for a horizontal decimation by two. **API TIER 3.**
/// @return ceil(srcWidth / 2) -- source columns 0, 2, 4, ... are kept, so an odd
///         width keeps its last column.
/// @note constexpr, so a caller can size a destination without a runtime call.
constexpr size_t decimatedWidth(size_t srcWidth) { return (srcWidth + 1) / 2; }

/// @brief The FREE half of a 2x2 subsample: every other row, as a view.
///        **API TIER 3.**
/// @return A view over source rows 0, 2, 4, ... -- twice the stride, ceil(height/2)
///         rows, the same width and the same memory.
/// @note This allocates nothing and copies nothing, which is the point: the
///       pyramid's vertical decimation is a view and only its horizontal half is
///       a kernel. Written here rather than open-coded at the call site so that
///       "stride * 2" appears once.
template <typename WordType>
inline BinMatConstView<WordType> rowsDecimatedBy2(BinMatConstView<WordType> src) {
    return BinMatConstView<WordType>{src.ptr, src.width, decimatedWidth(src.height),
                                     src.stride * 2};
}

namespace impl {

// ---------------------------------------------------------------------------
// Preconditions, written once for all three variants
// ---------------------------------------------------------------------------

/// @brief The shape and aliasing contract every decimation kernel here shares.
/// @note One function rather than three copies: the three variants differ in how
///       they compute the destination and not at all in what they accept, and a
///       precondition that drifts between two implementations of one operation is
///       worse than a shared file and line in the diagnostic.
template <typename WordType>
inline void checkDecimateArgs(BinMatConstView<WordType> src, BinMatView<WordType> dst) {
    // In a release build every BINCV_ASSERT below discards its condition, which
    // leaves both parameters unread -- and warnings are fatal in the gate.
    static_cast<void>(src);
    static_cast<void>(dst);
    BINCV_ASSERT(src.width == 0 || src.height == 0 || src.ptr != nullptr,
                 "decimateColumnsBy2: non-empty source needs a non-null pointer");
    BINCV_ASSERT(dst.width == 0 || dst.height == 0 || dst.ptr != nullptr,
                 "decimateColumnsBy2: non-empty destination needs a non-null pointer");
    BINCV_ASSERT(dst.width == decimatedWidth(src.width),
                 "decimateColumnsBy2: dst.width must be ceil(src.width / 2)");
    BINCV_ASSERT(dst.height == src.height,
                 "decimateColumnsBy2: this kernel is HORIZONTAL only -- decimate rows "
                 "with rowsDecimatedBy2() and pass the resulting view as src");
    BINCV_ASSERT(strideCoversARow<WordType>(src.width, src.height, src.stride),
                 "decimateColumnsBy2: src stride is shorter than one of its own rows");
    BINCV_ASSERT(strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "decimateColumnsBy2: dst stride is shorter than one of its own rows");
    BINCV_ASSERT(viewsShareNoWord(src, dst),
                 "decimateColumnsBy2: src and dst must not share a word -- destination "
                 "word i reads source words 2i and 2i+1, so no in-place form exists");
}

// ---------------------------------------------------------------------------
// A -- the gather loop
// ---------------------------------------------------------------------------

/// @brief E-8 variant A: horizontal decimation one destination pixel at a time.
/// @note Two spellings of "a gather loop" were available and this is deliberately
///       the stronger one, because X-14 compares ROUTES and a route measured in
///       its weakest spelling proves nothing about the route. The bit is
///       accumulated into a local word and stored once per destination word
///       rather than read-modify-writing memory per pixel; and it is OR-ed in
///       branchlessly rather than tested with an `if`, which on ~50%-fill image
///       data would be a branch misprediction per pixel and would flatter every
///       other variant in the comparison.
/// @note Only destination bits below dst.width are ever set, so the padding-bit
///       invariant holds without a trailing mask.
template <typename WordType>
inline void decimateColumnsBy2Gather(BinMatConstView<WordType> src,
                                     BinMatView<WordType> dst) {
    checkDecimateArgs(src, dst);
    if (dst.width == 0 || dst.height == 0) return;

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t dstWords = minRowWords<WordType>(dst.width);

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* srcRow = src.row(y);
        WordType* dstRow = dst.row(y);

        for (size_t i = 0; i < dstWords; ++i) {
            const size_t base = i * B;
            const size_t lanes = (dst.width - base < B) ? (dst.width - base) : B;

            WordType acc = 0;
            for (size_t b = 0; b < lanes; ++b) {
                // base + b < dst.width == ceil(src.width / 2), so the source
                // column below is at most src.width - 1: always in range, never
                // a padding bit.
                const size_t c = 2 * (base + b);
                const WordType bit =
                    static_cast<WordType>((srcRow[c / B] >> (c % B)) & static_cast<WordType>(1));
                acc = static_cast<WordType>(acc | static_cast<WordType>(bit << b));
            }
            dstRow[i] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// B -- the word-local unshuffle
// ---------------------------------------------------------------------------

/// @brief Even bits of one word, packed into its low half. **Internal.**
/// @return A word whose bit i is bit 2i of the argument, for i < WordBits/2, and
///         zero above that.
/// @note The classic Morton deinterleave: log2(WordBits) masked shift-ors, each
///       doubling the size of the block whose even bits are already contiguous at
///       the block's base. Three steps at uint8_t, six at uint64_t.
/// @note The masks are WORD LITERALS -- 0x55.., 0x33.., 0x0f.. and so on -- not
///       frame-sized constants. That is the whole of E-8's "word-local" branch:
///       word-parallel and zero auxiliary bytes at the same time.
/// @note The casts are the -Wconversion tax on integer promotion, exactly as in
///       maj3 (ops/bitslice.hpp): `x | (x >> 1)` is an int for uint8_t and
///       uint16_t and storing it back narrows.
template <typename WordType>
inline WordType gatherEvenBits(WordType x) {
    constexpr size_t B = bitsPerWord<WordType>();
    x = static_cast<WordType>(x & static_cast<WordType>(UINT64_C(0x5555555555555555)));
    x = static_cast<WordType>((x | (x >> 1)) &
                              static_cast<WordType>(UINT64_C(0x3333333333333333)));
    x = static_cast<WordType>((x | (x >> 2)) &
                              static_cast<WordType>(UINT64_C(0x0f0f0f0f0f0f0f0f)));
    if (B > 8) {
        x = static_cast<WordType>((x | (x >> 4)) &
                                  static_cast<WordType>(UINT64_C(0x00ff00ff00ff00ff)));
    }
    if (B > 16) {
        x = static_cast<WordType>((x | (x >> 8)) &
                                  static_cast<WordType>(UINT64_C(0x0000ffff0000ffff)));
    }
    if (B > 32) {
        x = static_cast<WordType>((x | (x >> 16)) &
                                  static_cast<WordType>(UINT64_C(0x00000000ffffffff)));
    }
    return x;
}

/// @brief E-8 variant B: horizontal decimation one destination WORD at a time.
/// @note Destination word i is `gather(src[2i]) | gather(src[2i+1]) << WordBits/2`
///       -- see the file header on why that pairing is exact. Source words past
///       the row read as zero, which can only affect destination padding bits.
template <typename WordType>
inline void decimateColumnsBy2Unshuffle(BinMatConstView<WordType> src,
                                        BinMatView<WordType> dst) {
    checkDecimateArgs(src, dst);
    if (dst.width == 0 || dst.height == 0) return;

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t srcWords = minRowWords<WordType>(src.width);
    const size_t dstWords = minRowWords<WordType>(dst.width);
    const WordType tail = rowTailMask<WordType>(dst.width);

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* srcRow = src.row(y);
        WordType* dstRow = dst.row(y);

        for (size_t i = 0; i < dstWords; ++i) {
            const size_t lo = 2 * i;
            const WordType a = (lo < srcWords) ? srcRow[lo] : static_cast<WordType>(0);
            const WordType b = (lo + 1 < srcWords) ? srcRow[lo + 1] : static_cast<WordType>(0);
            const WordType word = static_cast<WordType>(
                gatherEvenBits(a) | static_cast<WordType>(gatherEvenBits(b) << (B / 2)));
            dstRow[i] = (i + 1 == dstWords) ? static_cast<WordType>(word & tail) : word;
        }
    }
}

// ---------------------------------------------------------------------------
// C -- the frame-masked unshuffle
// ---------------------------------------------------------------------------
//
// The same log-depth gather as variant B, done on the ROW as one big integer:
// pass k masks with period 2^(k+1) and shifts right by 2^(k-1) words-and-bits
// across the whole row. Two things follow, and they are what E-8 is asking about:
//
//   * every step is a PASS OVER MEMORY rather than a register operation, and
//     there are log2(rowBits) of them -- ten at 640 columns, against six register
//     steps for a uint64_t word,
//   * the masks are no longer word literals. Once the period exceeds a word they
//     are runs of all-ones and all-zero WORDS, so they have to be materialised at
//     frame width, and the row has to be padded to a power-of-two bit count for
//     the block recurrence to tile it. Hence a mask table and a scratch row, both
//     the caller's (no heap in a kernel, CLAUDE.md).

/// @brief Words in one padded row for variant C. **Internal.**
/// @note The block recurrence doubles its period each pass and must end with one
///       block covering the row, so the row is padded up to a power-of-two bit
///       count. At 640 columns and uint32_t that is 20 words rounded to 32.
template <typename WordType>
inline size_t frameMaskedRowWords(size_t srcWidth) {
    const size_t rowWords = minRowWords<WordType>(srcWidth);
    size_t padded = 1;
    while (padded < rowWords) padded <<= 1;
    return padded;
}

/// @brief Passes -- and therefore mask rows -- variant C needs. **Internal.**
/// @return log2(paddedRowBits): one mask per period 2, 4, ... paddedRowBits.
template <typename WordType>
inline size_t frameMaskedPasses(size_t srcWidth) {
    const size_t bits = frameMaskedRowWords<WordType>(srcWidth) * bitsPerWord<WordType>();
    size_t passes = 0;
    for (size_t period = 2; period <= bits; period <<= 1) ++passes;
    return passes;
}

/// @brief Words the caller must provide for variant C's mask table. **Internal.**
/// @note This is the number X-14 weighs against zero for variants A and B. It
///       depends on the width and the word type only, so one table serves a whole
///       pyramid level -- but it is still frame-scale state that the word-local
///       routes do not have.
template <typename WordType>
inline size_t frameMaskedPlanWords(size_t srcWidth) {
    return frameMaskedPasses<WordType>(srcWidth) * frameMaskedRowWords<WordType>(srcWidth);
}

/// @brief Builds variant C's mask table. **Internal.**
/// @param masks frameMaskedPlanWords<WordType>(srcWidth) words, written in full.
/// @note Mask k keeps the low half of every aligned block of 2^(k+1) bits, which
///       is 0x5555.., 0x3333.., 0x0f0f.. and so on while the period fits a word,
///       and alternating runs of whole words once it does not.
template <typename WordType>
inline void buildFrameMaskedPlan(size_t srcWidth, WordType* masks) {
    BINCV_ASSERT(masks != nullptr, "buildFrameMaskedPlan: null mask table");

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t rowWords = frameMaskedRowWords<WordType>(srcWidth);
    const size_t bits = rowWords * B;

    size_t k = 0;
    for (size_t period = 2; period <= bits; period <<= 1, ++k) {
        WordType* mask = masks + k * rowWords;
        if (period <= B) {
            WordType word = 0;
            for (size_t offset = 0; offset < B; offset += period) {
                word = static_cast<WordType>(
                    word | static_cast<WordType>(lowBitsMask<WordType>(period / 2) << offset));
            }
            for (size_t i = 0; i < rowWords; ++i) mask[i] = word;
        } else {
            for (size_t i = 0; i < rowWords; ++i) {
                mask[i] = ((i * B) % period) < (period / 2)
                              ? static_cast<WordType>(~static_cast<WordType>(0))
                              : static_cast<WordType>(0);
            }
        }
    }
}

/// @brief E-8 variant C: horizontal decimation as a big-integer unshuffle.
/// @param masks frameMaskedPlanWords<WordType>(src.width) words from
///        buildFrameMaskedPlan(). Depends on the width and word type only.
/// @param scratch frameMaskedRowWords<WordType>(src.width) words, clobbered.
/// @note The mask table and the scratch row are the caller's because a kernel may
///       not allocate (CLAUDE.md), and they are the footprint side of E-8's trade.
template <typename WordType>
inline void decimateColumnsBy2FrameMasked(BinMatConstView<WordType> src,
                                          BinMatView<WordType> dst,
                                          const WordType* masks, WordType* scratch) {
    checkDecimateArgs(src, dst);
    if (dst.width == 0 || dst.height == 0) return;

    const size_t rowWords = frameMaskedRowWords<WordType>(src.width);
    BINCV_ASSERT(masks != nullptr && scratch != nullptr,
                 "decimateColumnsBy2FrameMasked: mask table and scratch are required");
    BINCV_ASSERT(byteRangesDisjoint(scratch, rowWords * sizeof(WordType), dst.ptr,
                                    viewSpanWords<WordType>(dst.width, dst.height, dst.stride) *
                                        sizeof(WordType)),
                 "decimateColumnsBy2FrameMasked: scratch must not overlap dst");

    constexpr size_t B = bitsPerWord<WordType>();
    const size_t srcWords = minRowWords<WordType>(src.width);
    const size_t dstWords = minRowWords<WordType>(dst.width);
    const size_t bits = rowWords * B;
    const WordType tail = rowTailMask<WordType>(dst.width);

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* srcRow = src.row(y);
        WordType* dstRow = dst.row(y);

        // Pass 0 is the copy into scratch fused with the even-bit mask, and it is
        // what zeroes the padding words the recurrence needs.
        for (size_t i = 0; i < rowWords; ++i) {
            const WordType word = (i < srcWords) ? srcRow[i] : static_cast<WordType>(0);
            scratch[i] = static_cast<WordType>(word & masks[i]);
        }

        // Pass k >= 1: fold the block above into the block below, then re-mask.
        // Ascending i is what makes the fold safe in place -- index i reads only
        // indices >= i.
        size_t k = 1;
        for (size_t shift = 1; shift * 4 <= bits; shift <<= 1, ++k) {
            const WordType* mask = masks + k * rowWords;
            const size_t wordShift = shift / B;
            const size_t bitShift = shift % B;

            if (bitShift == 0) {
                for (size_t i = 0; i < rowWords; ++i) {
                    const WordType a = (i + wordShift < rowWords)
                                           ? scratch[i + wordShift]
                                           : static_cast<WordType>(0);
                    scratch[i] = static_cast<WordType>(static_cast<WordType>(scratch[i] | a) &
                                                       mask[i]);
                }
            } else {
                // bitShift != 0 here, so `<< (B - bitShift)` is never a shift by
                // the word width -- the undefined case ops/shift.hpp branches on
                // for the same reason.
                for (size_t i = 0; i < rowWords; ++i) {
                    const WordType a = (i + wordShift < rowWords)
                                           ? scratch[i + wordShift]
                                           : static_cast<WordType>(0);
                    const WordType b = (i + wordShift + 1 < rowWords)
                                           ? scratch[i + wordShift + 1]
                                           : static_cast<WordType>(0);
                    const WordType moved = static_cast<WordType>(
                        static_cast<WordType>(a >> bitShift) |
                        static_cast<WordType>(b << (B - bitShift)));
                    scratch[i] = static_cast<WordType>(
                        static_cast<WordType>(scratch[i] | moved) & mask[i]);
                }
            }
        }

        for (size_t i = 0; i < dstWords; ++i) {
            dstRow[i] = (i + 1 == dstWords) ? static_cast<WordType>(scratch[i] & tail)
                                            : scratch[i];
        }
    }
}

} // namespace impl

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
