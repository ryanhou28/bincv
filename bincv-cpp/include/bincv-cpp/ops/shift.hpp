#pragma once

/// @file shift.hpp
/// @brief Horizontal and vertical shifts with OpenCV border semantics.
///
/// The primitive morphology and the binarized derivative are both
/// built out of, and -- as says in as many words -- the easiest
/// operation in the project to get subtly wrong. Nothing here is clever; the value
/// is in the edge cases being enumerated rather than discovered later.
///
/// ---------------------------------------------------------------------------
/// THE BIT CONVENTION, AND WHAT "LEFT" MEANS
///
/// Column `c` lives at bit `c % WordBits` of word `c / WordBits`, LSB first
/// (impl::bitMask). So a column index INCREASES with bit significance, and moving
/// the image LEFT -- towards lower column indices -- is a RIGHT shift of the word:
///
/// shiftLeft (src, dst, k): dst[r][c] = src[r][c + k] words shift >> k
/// shiftRight(src, dst, k): dst[r][c] = src[r][c - k] words shift << k
/// shiftUp (src, dst, k): dst[r][c] = src[r + k][c] row offset, no bit work
/// shiftDown (src, dst, k): dst[r][c] = src[r - k][c] row offset, no bit work
///
/// This matches the design notes, which writes the derivative's rising edge as
/// `(src >> 1) & ~(src << 1)` and writes the same expression as
/// `shiftLeft(src, 1) & ~shiftRight(src, 1)`. Both spellings must mean one thing,
/// and this is it.
///
/// The word recurrence for shiftLeft, with `wordShift = k / WordBits` and
/// `bitShift = k % WordBits`:
///
/// bitShift == 0: dst[i] = src[i + wordShift]
/// otherwise: dst[i] = (src[i + wordShift] >> bitShift)
/// | (src[i + wordShift + 1] << (WordBits - bitShift))
///
/// **`bitShift == 0` IS A SEPARATE BRANCH BECAUSE `x << WordBits` IS UNDEFINED
/// BEHAVIOUR.** Not merely wrong -- undefined, and on x86 the natural encoding
/// masks the shift count and yields `x` where the algebra wants 0, so the bug
/// survives every test written at k = 1 and appears the first time a caller shifts
/// by exactly one word. tests/test_shift.cpp sweeps k from 0 through 2*WordBits+1
/// at every word width, which covers 0, WordBits and 2*WordBits by construction,
/// and the suite is additionally built and run under -fsanitize=undefined so that
/// a shift-exponent violation is a diagnostic rather than a difference of opinion
/// (the two commands are in GETTING_STARTED, "Sanitizers"). That was watched
/// failing, not assumed: with the branch below removed, UBSan reports
/// `shift exponent 32 is too large for 32-bit type` and 1008 checks go red.
///
/// ---------------------------------------------------------------------------
/// OUT-OF-RANGE READS: WORDS *AND* THE PADDING BITS OF THE LAST ONE
///
/// "Out-of-range source words read as zero" is not sufficient on its own, and the
/// gap is exactly where a shift bug would hide. A row's trailing partial word
/// holds `width % WordBits` pixels and `WordBits - width % WordBits` padding bits.
/// Those padding bits are at column indices >= width, i.e. OUTSIDE the image, so a
/// shiftLeft by k moves them to columns >= width - k -- which are LIVE PIXELS of
/// the destination.
///
/// A destination binCV produced has clean padding, but a source may legitimately
/// not: BinMat's wrap constructor documents that a caller's padding bits are the
/// caller's (sensor DMA, a sub-region of a larger frame), and tests/test_logic.cpp
/// already sweeps that construction. So this file extends every source word past
/// `width` with the BORDER value rather than trusting the bits that happen to be
/// there -- see impl::extendedRowWord. Measured on a 5-pixel-wide all-ones wrapped
/// buffer at uint8_t, shiftLeft by 1 without that mask reports pixel 4 set where
/// the border says it must be clear.
///
/// ---------------------------------------------------------------------------
/// THE FILL IS A PARAMETER, NOT A POLICY -- BECAUSE ERODE AND DILATE DISAGREE
///
/// builds dilate as an OR of shifted copies and erode as an AND of
/// shifted copies. Those two want OPPOSITE fills, and no single choice serves both:
///
/// dilate = OR of shifts -- a pixel outside the image must contribute NOTHING
/// to an OR, so everything outside must read 0.
/// erode = AND of shifts -- a pixel outside the image must contribute NOTHING
/// to an AND, so everything outside must read 1.
///
/// With the wrong one, erode eats a `k`-wide band off every edge of a full frame
/// and dilate grows one. OpenCV reaches the same conclusion and encodes it the
/// same way: cv::erode and cv::dilate default `borderValue` to
/// morphologyDefaultBorderValue, which the implementation resolves to the depth's
/// MAXIMUM for erosion and its MINIMUM for dilation -- ones outside for erode,
/// zeros outside for dilate. tests/test_shift.cpp pins that premise against the
/// real cv::erode / cv::dilate rather than leaving it as a claim in a comment.
///
/// So every entry point here takes `(BorderType borderType, bool borderValue)`,
/// defaulting to BORDER_CONSTANT / false -- the plain zero fill specifies for
/// the bare three-argument call, and what that work’s derivative wants.
///
/// dilate step: shift(src, dst, dx, dy, BORDER_CONSTANT, false)
/// erode step: shift(src, dst, dx, dy, BORDER_CONSTANT, true)
/// OpenCV-exact non-constant borders: pass the BorderType through unchanged.
///
/// ---------------------------------------------------------------------------
/// BORDER SEMANTICS ARE OPENCV'S, EXACTLY
///
/// impl::borderIndex is cv::borderInterpolate: same five types, same -1 for
/// BORDER_CONSTANT, same `len == 1` answer for both reflect flavours. That is what
/// makes the Tier 1 operations built on this file bit-exact rather than merely
/// close -- a border rule that differs by one column at one edge is invisible in
/// the middle of a frame and fatal to the drop-in promise.
///
/// It is written as a closed form (one modulo into the pattern's period) where
/// OpenCV writes a do-while, so that a caller shifting by more than the image
/// width cannot spin. The two are pinned against each other over
/// p in [-3*len, 3*len] for a range of lengths in tests/test_shift.cpp, and the
/// core-only half of that suite compares against an independently written
/// do-while reference so the property survives in the builds that have no OpenCV.
///
/// ---------------------------------------------------------------------------
/// WHAT A KERNEL IN THIS FILE PROMISES
///
/// 1. **Views, never containers**. Strides are read per row and may differ
/// between src and dst.
/// 2. **Padding bits stay zero** in the destination, whatever the source held and
/// whatever the fill is. Every row's trailing word is stored masked.
/// 3. **No allocation, no throw** (the design notes). A 2-D shift is one pass
/// over the destination -- the vertical part is a row-index remap, so it needs
/// no scratch and the caller needs no temporary between the two axes.
/// 4. **PRECONDITION ON `dst`, identical to ops/logic.hpp**: it must span its
/// image's full width or end on a word boundary. The trailing partial word is
/// stored masked, which writes zeros into bits [width, rowWords * WordBits) --
/// padding in the usual case, and a wider image's next 1..WordBits-1 pixels
/// when `dst` is a sub-width window onto one. Nothing can diagnose that: every
/// address written is inside the parent image.
///
/// ---------------------------------------------------------------------------
/// ALIASING: `src` AND `dst` MUST SHARE NO WORD. IN PLACE IS NOT SUPPORTED.
///
/// This is the half of earlier work that applies, and the exact-alias half deliberately
/// does NOT extend here. permits `dst` to be exactly a source because the
/// kernels are POINTWISE IN THE WORD INDEX -- word i of the destination is
/// read from word i of the source and from nothing else, so it is read immediately
/// before it is overwritten. A shift is not pointwise: word i of the destination
/// is built from words i +/- wordShift and i +/- wordShift + 1.
///
/// A direction-aware loop would rescue the purely horizontal case (ascending for
/// shiftLeft, descending for shiftRight, the memmove argument). It does not rescue
/// the vertical one, and that is why the rule is stated once rather than per axis:
/// with dy != 0 the source row for destination row y is not monotonic in y. Take a
/// 10-row image, shiftUp by 5, BORDER_REFLECT_101. Destination row 9 reads source
/// row borderIndex(14) = 4 -- a row an ascending loop overwrote four iterations
/// ago, and a descending loop has not written yet but will need again. There is no
/// row order that works for every (dy, BorderType), and the design rule forbids the temporary
/// that would make one unnecessary.
///
/// So: exactly one supported relationship, checked per row rather than by bounding
/// box ( -- interleaved row bands and left/right column tiles share a buffer
/// without sharing a byte, and rejecting them would reject a pyramid downsample).
/// Callers wanting in-place semantics own a destination and swap.
///
/// Empty views (width or height 0) are a no-op, not an error.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow, impl::viewsShareNoWord, and the
// impl:: word helpers (bitsPerWord, bitMask, minRowWords) the row arithmetic below
// is written in terms of. Shared with ops/logic.hpp so that has one copy.
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief Largest shift magnitude these kernels accept, in pixels.
/// @note An offset is carried as a ptrdiff_t and added to a pixel coordinate, so
/// the two together must not overflow. Half of PTRDIFF_MAX is astronomically
/// more than any image dimension and leaves the sum exact; anything beyond it
/// is a programming error, reported by BINCV_ASSERT like every other
/// precondition here (the design notes).
inline constexpr size_t maxShiftOffset() {
    return static_cast<size_t>(PTRDIFF_MAX) / 2;
}

/// @brief True for the five BorderType values core/types.hpp defines.
/// @note A BorderType is an enum, so a caller can hand over anything an int holds.
/// borderIndex below would treat an unknown value as BORDER_CONSTANT,
/// which is a plausible answer to a question that was never asked -- this
/// turns it into a diagnosed one in debug builds.
inline bool isKnownBorderType(BorderType type) {
    return type == BORDER_CONSTANT || type == BORDER_REPLICATE || type == BORDER_REFLECT ||
           type == BORDER_REFLECT_101 || type == BORDER_WRAP;
}

/// @brief cv::borderInterpolate: the source index a coordinate outside [0, len)
/// extrapolates to, or -1 when the border is a constant.
/// @param p The coordinate, which may be negative or >= len.
/// @param len The extent along that axis, in pixels. Must be > 0.
/// @param type One of the five BorderType values.
///
/// @note **This function is the Tier 1 promise.** Every downstream neighbourhood
/// operation's border behaviour is this mapping and nothing else, so it is
/// compared directly against cv::borderInterpolate in tests/test_shift.cpp
/// rather than only through the images built on it.
/// @note Closed form rather than OpenCV's `do {... } while (p out of range)`. The
/// two agree by construction: both reflect patterns are periodic, with
/// period 2*len for BORDER_REFLECT (fedcba|abcdef|fedcba) and 2*len-2 for
/// BORDER_REFLECT_101 (gfedcb|abcdefg|fedcba), so one signed modulo into the
/// period reaches the same fixed point the loop iterates towards. The
/// difference that matters is termination: shifting a 7-pixel row by 500 --
/// which requires to be correct, not merely defined -- costs the loop
/// ~35 iterations per pixel and costs this one modulo.
/// @note `len == 1` returns 0 for BOTH reflect flavours, matching OpenCV. The
/// REFLECT_101 period is 2*len-2 == 0 there, so the closed form would divide
/// by zero; OpenCV's loop would spin. Neither is reachable, because the case
/// is answered first.
inline ptrdiff_t borderIndex(ptrdiff_t p, size_t lenPixels, BorderType type) {
    const ptrdiff_t len = static_cast<ptrdiff_t>(lenPixels);
    if (p >= 0 && p < len) return p;

    switch (type) {
        case BORDER_REPLICATE:
            return p < 0 ? static_cast<ptrdiff_t>(0) : len - 1;

        case BORDER_REFLECT:
        case BORDER_REFLECT_101: {
            if (len == 1) return 0;
            const ptrdiff_t delta = (type == BORDER_REFLECT_101) ? 1 : 0;
            const ptrdiff_t period = 2 * len - 2 * delta;
            ptrdiff_t q = p % period;
            if (q < 0) q += period;
            return q < len ? q : (2 * len - 1 - delta - q);
        }

        case BORDER_WRAP: {
            ptrdiff_t q = p % len;
            if (q < 0) q += len;
            return q;
        }

        case BORDER_CONSTANT:
        default:
            return -1;
    }
}

/// @brief Word `j` of a row, with everything outside the row's PIXELS reading
/// `fill` -- both whole words past the row and the padding bits of the last.
/// @note The second half is the one that is easy to omit and impossible to see
/// afterwards: see the "out-of-range reads" section at the top of this file.
/// @note `j < 0` is reachable from the shiftRight recurrence and is not an error.
template <typename WordType>
inline WordType extendedRowWord(const WordType* row, ptrdiff_t j, size_t rowWords,
                                WordType tailMask, WordType fill) {
    if (j < 0 || static_cast<size_t>(j) >= rowWords) return fill;

    const WordType w = row[static_cast<size_t>(j)];
    if (static_cast<size_t>(j) + 1 != rowWords) return w;

    const WordType keep = static_cast<WordType>(w & tailMask);
    const WordType outside = static_cast<WordType>(fill & static_cast<WordType>(~tailMask));
    return static_cast<WordType>(keep | outside);
}

/// @brief Writes one destination row as a constant, padding bits left zero.
/// @note rowWords >= 1 here: it is only called for a non-empty row.
template <typename WordType>
inline void fillRowWords(WordType* dstRow, size_t rowWords, WordType value, WordType tailMask) {
    for (size_t i = 0; i + 1 < rowWords; ++i) dstRow[i] = value;
    dstRow[rowWords - 1] = static_cast<WordType>(value & tailMask);
}

/// @brief One row of horizontal shift by `dx` columns, filling with `fill`.
/// @param dx Positive shifts the image LEFT (dst column c takes src column c + dx).
/// @param fill The word written where the source is outside the image. For a
/// non-constant BorderType the caller passes 0 and repairs the affected
/// columns afterwards with fixupHorizontalBorder; those are exactly the
/// columns this function fills, so nothing correct is overwritten.
/// @note The whole point of the packed representation: WordBits pixels move per
/// word operation, and the source's own bit alignment never has to match the
/// destination's.
/// @note Deliberately NOT split into a bounds-check-free interior loop and two
/// edge loops. That is the obvious optimisation and it is Phase 5's business
/// (: "correct scalar first"); the branch inside
/// extendedRowWord is perfectly predictable and the alternative doubles the
/// number of index expressions that can be off by one.
template <typename WordType>
inline void shiftRowHorizontal(const WordType* srcRow, WordType* dstRow, size_t width,
                               ptrdiff_t dx, WordType fill) {
    constexpr size_t wordBits = bitsPerWord<WordType>();
    const size_t rowWords = minRowWords<WordType>(width);
    const WordType tailMask = rowTailMask<WordType>(width);

    // Magnitude in unsigned arithmetic: negating PTRDIFF_MIN is signed overflow,
    // and this is the one place a caller's number reaches a negation.
    const size_t k = (dx < 0) ? (static_cast<size_t>(0) - static_cast<size_t>(dx))
                              : static_cast<size_t>(dx);

    // Every column comes from outside the image. Handled before the recurrence
    // rather than inside it, because `i + wordShift` would otherwise be free to
    // wrap for an absurd k -- and a wrapped index is an in-range one.
    if (k >= width) {
        fillRowWords(dstRow, rowWords, fill, tailMask);
        return;
    }

    const size_t wordShift = k / wordBits;
    const size_t bitShift = k % wordBits;

    if (dx >= 0) {
        // dst[i] gathers from src[i + wordShift] and the word above it.
        if (bitShift == 0) {
            for (size_t i = 0; i < rowWords; ++i) {
                dstRow[i] = extendedRowWord(srcRow, static_cast<ptrdiff_t>(i + wordShift),
                                            rowWords, tailMask, fill);
            }
        } else {
            const size_t carry = wordBits - bitShift;      // in (0, wordBits)
            for (size_t i = 0; i < rowWords; ++i) {
                const WordType lo = extendedRowWord(
                    srcRow, static_cast<ptrdiff_t>(i + wordShift), rowWords, tailMask, fill);
                const WordType hi = extendedRowWord(
                    srcRow, static_cast<ptrdiff_t>(i + wordShift + 1), rowWords, tailMask, fill);
                dstRow[i] = static_cast<WordType>(static_cast<WordType>(lo >> bitShift) |
                                                  static_cast<WordType>(hi << carry));
            }
        }
    } else {
        // The mirror image: dst[i] gathers from src[i - wordShift] and the word
        // below it.
        if (bitShift == 0) {
            for (size_t i = 0; i < rowWords; ++i) {
                dstRow[i] = extendedRowWord(
                    srcRow, static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(wordShift),
                    rowWords, tailMask, fill);
            }
        } else {
            const size_t carry = wordBits - bitShift;      // in (0, wordBits)
            for (size_t i = 0; i < rowWords; ++i) {
                const ptrdiff_t base =
                    static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(wordShift);
                const WordType hi =
                    extendedRowWord(srcRow, base, rowWords, tailMask, fill);
                const WordType lo =
                    extendedRowWord(srcRow, base - 1, rowWords, tailMask, fill);
                dstRow[i] = static_cast<WordType>(static_cast<WordType>(hi << bitShift) |
                                                  static_cast<WordType>(lo >> carry));
            }
        }
    }

    // Property 2: the destination's padding bits are zero however the row was
    // built, and whatever the fill was.
    dstRow[rowWords - 1] = static_cast<WordType>(dstRow[rowWords - 1] & tailMask);
}

/// @brief Rewrites the destination columns whose source column lies outside the
/// row, for the four NON-CONSTANT border types.
///
/// @note Per pixel, and unapologetically so. REPLICATE, REFLECT, REFLECT_101 and
/// WRAP each map a different out-of-range column to a different source
/// column, so there is no word-wide answer -- but there are at most
/// min(|dx|, width) such columns at one edge, and |dx| is a structuring
/// element's radius in every caller the MVP has. The bulk of the row stays
/// word-parallel.
/// @note Writes only columns < width, so it cannot dirty a padding bit.
template <typename WordType>
inline void fixupHorizontalBorder(const WordType* srcRow, WordType* dstRow, size_t width,
                                  ptrdiff_t dx, BorderType type, bool borderValue) {
    constexpr size_t wordBits = bitsPerWord<WordType>();

    size_t first = 0;
    size_t last = 0;
    if (dx > 0) {
        const size_t k = static_cast<size_t>(dx);          // dst[c] = src[c + k]
        first = (k >= width) ? 0 : (width - k);
        last = width;
    } else if (dx < 0) {
        const size_t k = static_cast<size_t>(0) - static_cast<size_t>(dx);
        first = 0;
        last = (k >= width) ? width : k;                   // dst[c] = src[c - k]
    } else {
        return;                                            // nothing leaves the row
    }

    for (size_t c = first; c < last; ++c) {
        const ptrdiff_t sx = borderIndex(static_cast<ptrdiff_t>(c) + dx, width, type);
        bool value;
        if (sx < 0) {
            value = borderValue;                           // unreachable: type != CONSTANT
        } else {
            const size_t sxu = static_cast<size_t>(sx);
            value = (srcRow[sxu / wordBits] & bitMask<WordType>(sxu)) != 0;
        }

        const size_t wordAt = c / wordBits;
        const WordType mask = bitMask<WordType>(c);
        dstRow[wordAt] = value
            ? static_cast<WordType>(dstRow[wordAt] | mask)
            : static_cast<WordType>(dstRow[wordAt] & static_cast<WordType>(~mask));
    }
}

} // namespace impl

// ---------------------------------------------------------------------------
// The kernels ( views, never containers)
// ---------------------------------------------------------------------------

/// @brief dst[y][x] = src[y + dy][x + dx], extrapolating outside the image.
/// **API TIER 3** by name -- OpenCV has no shift function, so nothing is
/// being promised drop-in compatibility -- **with TIER 1 BORDER SEMANTICS**:
/// the extrapolation is bit-exact against cv::borderInterpolate and
/// cv::copyMakeBorder, which is what lets the Tier 1 operations built on it
/// (morphology) be bit-exact in turn.
///
/// @param src Source view.
/// @param dst Destination view; must have src's dimensions, and must share no word
/// with src (see the aliasing section at the top of this file -- in place is
/// NOT supported here, unlike in ops/logic.hpp).
/// @param dx Positive moves the image LEFT: dst column c takes src column c + dx.
/// @param dy Positive moves the image UP: dst row r takes src row r + dy.
/// @param borderType How coordinates outside the image extrapolate. Defaults to
/// BORDER_CONSTANT, i.e. a plain fill.
/// @param borderValue The pixel value outside the image under BORDER_CONSTANT.
/// Ignored by the other four types. **false for a dilate step, true for an
/// erode step** -- see the fill section at the top of this file.
///
/// @note One pass over the destination, no scratch buffer, no allocation: the
/// vertical part is a row-index remap rather than a copy, so a 2-D shift
/// costs the same as a horizontal one and a caller composing the two axes
/// never needs a temporary between them. That is why this entry point exists
/// alongside the four directional ones -- shifts by a structuring
/// element offset, which is 2-D.
/// @note Offsets larger than the image are correct, not merely defined: every
/// destination pixel then comes from the border, which for BORDER_WRAP or
/// either reflect flavour is still a well-defined source pixel.
/// @note The destination's padding bits are zero on return regardless of the fill,
/// and the source's padding bits are never read as pixels even when the
/// source wraps a buffer whose padding is dirty.
/// @note Never throws and never allocates (the design notes). Mismatched
/// dimensions, a stride shorter than a row, an unknown BorderType, an
/// absurd offset, and any overlap between src and dst are programming
/// errors: BINCV_ASSERT reports them in debug builds and they are undefined
/// in release, exactly as an out-of-range at is.
template <typename WordType>
inline void shift(BinMatConstView<WordType> src, BinMatView<WordType> dst, ptrdiff_t dx,
                  ptrdiff_t dy, BorderType borderType = BORDER_CONSTANT,
                  bool borderValue = false) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "shift: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(src.width, src.height, src.stride) &&
                     impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "shift: every view's stride must cover a whole row");
    BINCV_ASSERT(impl::viewsShareNoWord(src, dst),
                 "shift: dst must share no word with src (in place is not supported)");
    BINCV_ASSERT(impl::isKnownBorderType(borderType), "shift: unknown BorderType");
    BINCV_ASSERT((dx >= 0 ? static_cast<size_t>(dx)
                          : static_cast<size_t>(0) - static_cast<size_t>(dx)) <=
                         impl::maxShiftOffset() &&
                     (dy >= 0 ? static_cast<size_t>(dy)
                              : static_cast<size_t>(0) - static_cast<size_t>(dy)) <=
                         impl::maxShiftOffset(),
                 "shift: the offset magnitude must not exceed PTRDIFF_MAX / 2");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "shift: a non-empty view needs a non-null pointer");

    const size_t rowWords = impl::minRowWords<WordType>(dst.width);
    const WordType tailMask = impl::rowTailMask<WordType>(dst.width);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));
    const WordType constantFill = borderValue ? allOnes : static_cast<WordType>(0);

    // Under a non-constant BorderType the word path's fill never survives: the
    // columns it reaches are exactly the ones fixupHorizontalBorder rewrites. Zero
    // is used there so that a missing fixup is a visibly wrong image rather than a
    // plausible one.
    const WordType wordFill =
        (borderType == BORDER_CONSTANT) ? constantFill : static_cast<WordType>(0);

    for (size_t y = 0; y < dst.height; ++y) {
        WordType* dstRow = dst.row(y);
        const ptrdiff_t sy =
            impl::borderIndex(static_cast<ptrdiff_t>(y) + dy, src.height, borderType);

        // Outside the image vertically under BORDER_CONSTANT: the whole row is the
        // border value, corners included. That is also what cv::copyMakeBorder
        // does -- its constant top and bottom bands span the padded width.
        if (sy < 0) {
            impl::fillRowWords(dstRow, rowWords, constantFill, tailMask);
            continue;
        }

        const WordType* srcRow = src.row(static_cast<size_t>(sy));
        impl::shiftRowHorizontal(srcRow, dstRow, dst.width, dx, wordFill);
        if (borderType != BORDER_CONSTANT) {
            impl::fixupHorizontalBorder(srcRow, dstRow, dst.width, dx, borderType, borderValue);
        }
    }
}

/// @brief dst[y][x] = src[y][x + k] -- moves the image LEFT by k columns.
/// **API TIER 3** by name, TIER 1 border semantics. See shift.
/// @param k Shift distance in pixels. 0 is a copy; k >= width leaves every column
/// coming from the border.
/// @note The default border is BORDER_CONSTANT with value false, i.e. the
/// zero-filled right edge specifies for the three-argument
/// call. Pass `true` for the erode step; see the fill section at the top of
/// this file.
template <typename WordType>
inline void shiftLeft(BinMatConstView<WordType> src, BinMatView<WordType> dst, size_t k,
                      BorderType borderType = BORDER_CONSTANT, bool borderValue = false) {
    BINCV_ASSERT(k <= impl::maxShiftOffset(),
                 "shiftLeft: the shift distance must not exceed PTRDIFF_MAX / 2");
    shift(src, dst, static_cast<ptrdiff_t>(k), static_cast<ptrdiff_t>(0), borderType,
          borderValue);
}

/// @brief dst[y][x] = src[y][x - k] -- moves the image RIGHT by k columns.
/// **API TIER 3** by name, TIER 1 border semantics. See shift.
/// @note The mirror image of shiftLeft, and the other half of the derivative's
/// `(src >> 1) & ~(src << 1)` (the design notes).
template <typename WordType>
inline void shiftRight(BinMatConstView<WordType> src, BinMatView<WordType> dst, size_t k,
                       BorderType borderType = BORDER_CONSTANT, bool borderValue = false) {
    BINCV_ASSERT(k <= impl::maxShiftOffset(),
                 "shiftRight: the shift distance must not exceed PTRDIFF_MAX / 2");
    shift(src, dst, static_cast<ptrdiff_t>(0) - static_cast<ptrdiff_t>(k),
          static_cast<ptrdiff_t>(0), borderType, borderValue);
}

/// @brief dst[y][x] = src[y + k][x] -- moves the image UP by k rows.
/// **API TIER 3** by name, TIER 1 border semantics. See shift.
/// @note No bit manipulation at all: a vertical shift is a row-index remap,
/// so each destination row is one masked copy of a source row -- or the
/// border value, when the remapped index is outside the image.
/// @note Correct for k >= height. Under BORDER_CONSTANT every row is then the fill;
/// under the other four types every row still resolves to a real source row.
template <typename WordType>
inline void shiftUp(BinMatConstView<WordType> src, BinMatView<WordType> dst, size_t k,
                    BorderType borderType = BORDER_CONSTANT, bool borderValue = false) {
    BINCV_ASSERT(k <= impl::maxShiftOffset(),
                 "shiftUp: the shift distance must not exceed PTRDIFF_MAX / 2");
    shift(src, dst, static_cast<ptrdiff_t>(0), static_cast<ptrdiff_t>(k), borderType,
          borderValue);
}

/// @brief dst[y][x] = src[y - k][x] -- moves the image DOWN by k rows.
/// **API TIER 3** by name, TIER 1 border semantics. See shift and shiftUp.
template <typename WordType>
inline void shiftDown(BinMatConstView<WordType> src, BinMatView<WordType> dst, size_t k,
                      BorderType borderType = BORDER_CONSTANT, bool borderValue = false) {
    BINCV_ASSERT(k <= impl::maxShiftOffset(),
                 "shiftDown: the shift distance must not exceed PTRDIFF_MAX / 2");
    shift(src, dst, static_cast<ptrdiff_t>(0),
          static_cast<ptrdiff_t>(0) - static_cast<ptrdiff_t>(k), borderType, borderValue);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
