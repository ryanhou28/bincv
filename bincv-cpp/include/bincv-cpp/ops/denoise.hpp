#pragma once

/// @file denoise.hpp
/// @brief The reference pipeline's three-pixel median filter.
/// **API TIER 3** -- see the tier note below.
///
/// The first VIO frontend operation in the project. Everything under ops/ before
/// this file is a primitive; this is the first kernel a real frontend calls by
/// name, and it is defined by what the reference implementation does rather than
/// by what OpenCV offers.
///
/// ---------------------------------------------------------------------------
/// THE NEIGHBOURHOOD, READ OUT OF THE REFERENCE RATHER THAN INFERRED
///
/// the reference frontend's denoiser, three_pix_median_filter:
///
/// cv::Mat right_pixels = cv::Mat::zeros(img.size, img.type);
/// cv::Mat above_pixels = cv::Mat::zeros(img.size, img.type);
/// img.colRange(1, img.cols).copyTo(right_pixels.colRange(0, img.cols - 1));
/// img.rowRange(0, img.rows - 1).copyTo(above_pixels.rowRange(1, img.rows));
///
/// // | | p1 | |
/// // | | p2 | p3 |
/// // p1 is above_pixels, p3 is right_pixels
/// // Median = max(min(p1, p2), min(max(p1, p2), p3))
/// cv::min(above_pixels, img, min_p1_p2);
/// cv::max(above_pixels, img, max_p1_p2);
/// cv::min(max_p1_p2, right_pixels, min_max);
/// cv::max(min_p1_p2, min_max, median_img);
///
/// So, in this file's coordinates (row y, column x):
///
/// p1 = src[y - 1][x] the pixel ABOVE
/// p2 = src[y ][x] the pixel ITSELF
/// p3 = src[y ][x + 1] the pixel to its RIGHT
///
/// It is a three-pixel L, not a 3x3 window and not a 1x3 line, and it is
/// ASYMMETRIC -- there is no left neighbour and no below neighbour. That is the
/// operation the reference pipeline runs, so it is the operation binCV
/// reproduces.
///
/// ---------------------------------------------------------------------------
/// THE BORDER IS ZERO FILL, AND IT FALLS OUT OF THOSE TWO copyTo RANGES
///
/// This is the detail that would silently differ. Both neighbour matrices are
/// `cv::Mat::zeros` and only the INTERIOR is copied over:
///
/// * `right_pixels.colRange(0, cols - 1)` is written; column `cols - 1` is
/// never assigned, so **the last column's right neighbour is 0**.
/// * `above_pixels.rowRange(1, rows)` is written; row 0 is never assigned, so
/// **the first row's above neighbour is 0**.
///
/// Zero fill. NOT replicate, NOT reflect -- and the difference is invisible on
/// interior pixels, so a test that skips the edges cannot see it. In this file's
/// vocabulary that is BORDER_CONSTANT with value `false`, which is also
/// ops/shift.hpp's default, so the composed spelling
///
/// shiftDown(src, above, 1); // p1, zero-filled top row
/// shiftLeft(src, right, 1); // p3, zero-filled last column
/// majority3(above, src, right, dst);
///
/// is exactly this file's kernel with two frame-sized scratch buffers added. See
/// "why this is fused" below for why denoiseMedian3 does not do that.
///
/// ---------------------------------------------------------------------------
/// WHY MEDIAN OF THREE IS maj3, AND WHY THAT IS STILL DEMONSTRATED
///
/// On {0, 1} pixels `cv::min` is `&` and `cv::max` is `|`, so the reference's
/// sorting network is
///
/// max(min(p1,p2), min(max(p1,p2), p3)) = (p1 & p2) | ((p1 | p2) & p3)
/// = (p1 & p2) | (p1 & p3) | (p2 & p3)
/// = maj3(p1, p2, p3)
///
/// which is the design notes's expression and ops/bitslice.hpp's `maj3`. The
/// algebra is why the kernel is one instruction per 64 pixels; it is NOT why the
/// kernel is believed correct. tests/test_denoise.cpp runs the reference's own
/// cv::min / cv::max calls -- ported, not paraphrased, so the border comes from
/// the same `cv::Mat::zeros` construction -- and compares pixel for pixel across
/// the size and fill matrix.
///
/// ---------------------------------------------------------------------------
/// API TIER 3, DELIBERATELY
///
/// OpenCV has no equivalent. `cv::medianBlur` is a median over a SQUARE window
/// (3x3 at ksize 3), not this asymmetric three-pixel L, and its border is
/// BORDER_REPLICATE rather than zero. Borrowing the name would make a Tier 1
/// drop-in promise that this operation cannot keep (the design notes), so the
/// name says what it is: `denoiseMedian3`.
///
/// ---------------------------------------------------------------------------
/// WHY THIS IS ONE FUSED PASS AND NOT shift + shift + majority3
///
/// Both spellings compute the same image. They differ in what they cost:
///
/// composed 3 passes over the frame, and TWO FRAME-SIZED SCRATCH BUFFERS
/// the caller has to own -- "no heap allocation inside kernels"
/// (CLAUDE.md) means a kernel cannot conjure them, so they become
/// part of the pipeline's peak working set. At 640x480 uint32_t
/// that is 2 x 38400 B added to a 38400 B frame: 3x the footprint
/// of the operation.
/// fused 1 pass, NO scratch at all. The above-neighbour is a row index
/// (row y - 1, or zeros at y == 0 -- a vertical shift moves no bits,
///), and the right-neighbour is the current row's words shifted
/// one bit down, computed inline into a register.
///
/// Memory and speed do not conflict here -- the fused form is smaller AND makes
/// one traversal instead of three -- so no experiment is needed to choose
/// (CLAUDE.md, "How performance and footprint decisions get made"). It is
/// measured rather than asserted all the same: benchmark/denoise_benchmark.cpp
/// reports both spellings against the OpenCV denominator, with the scratch
/// footprint alongside, and that is the evidence for this paragraph.
///
/// ---------------------------------------------------------------------------
/// WHAT THE KERNEL PROMISES (the ops/ contract)
///
/// 1. **Views, never containers.** BinMatConstView in, BinMatView out, each
/// stride read per row, so an over-aligned matrix or a wrapped buffer
/// works unchanged.
/// 2. **No scratch and no allocation**, as above.
/// 3. **PADDING BITS STAY ZERO, AND ONE MASK DOES IT.** The source's trailing
/// word is masked BEFORE the right-shift, because the shift would otherwise
/// pull bit `width % WordBits` of that word -- PADDING, which a wrapped
/// buffer's caller owns and may leave dirty -- into pixel `width - 1`. That
/// mask is a BORDER requirement first: without it the last column's
/// neighbour silently stops being the zero fill the reference specifies, on
/// exactly one column of exactly the widths that do not end on a word
/// boundary.
///
/// It also makes the destination's padding zero for free, and that is why
/// the store carries no second mask: `c` and its shift are both zero past
/// `width`, and majority needs two of three. See medianRow3 for the
/// measurement, and for the coupling this creates -- a change to the border
/// is a change to the padding invariant here.
/// 4. **Never throws.** Mismatched dimensions, a stride too short for a row and
/// an overlapping destination are programming errors reported by
/// BINCV_ASSERT in debug builds and undefined in release (the design notes).
///
/// ---------------------------------------------------------------------------
/// ALIASING: `dst` MUST SHARE NO WORD WITH `src` -- IN PLACE IS NOT SUPPORTED
///
/// This is the half of earlier work ops/shift.hpp takes, and for the same reason: the
/// operation is NOT pointwise in the word index. Destination row y reads source
/// row y - 1, so an in-place call would read a row it has already overwritten and
/// the filter would feed on its own output from the second row onwards. The
/// reference has the same property and answers it the same way -- it builds
/// `median_img` separately and assigns `img = median_img` at the end.
///
/// The check is impl::viewsShareNoWord, the exact per-row predicate shared with
/// ops/shift.hpp, not a bounding-box test: two views over one buffer can
/// interleave without sharing a byte, and rejecting those would reject legal
/// arguments.
///
/// ---------------------------------------------------------------------------
/// PRECONDITION ON `dst`, identical to ops/logic.hpp's: it must span its image's
/// full width, or end on a word boundary. The trailing partial word is stored
/// masked, so in a sub-width window onto a WIDER image the bits between `width`
/// and the end of that word -- the neighbours' live pixels -- are zeroed, and
/// nothing diagnoses it.
///
/// Empty views (width or height 0) are a no-op, not an error.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow and impl::viewsShareNoWord -- the
// row-geometry and aliasing vocabulary every kernel under ops/ is written
// in. One copy is the only way the aliasing rule cannot drift.
#include "../impl/kernel_util.hpp"
// impl::maj3, the word primitive exists to provide. the design notes's
// expression lives there and is enumerated exhaustively by tests/test_bitslice.cpp
// -- this file supplies the neighbourhood, not the arithmetic.
#include "bitslice.hpp"
// impl::minRowWords, and BinMat for nothing but the doc examples above.
#include "../quantMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief One destination row of the three-pixel median.
/// @tparam HasAbove False for row 0, where p1 is the zero border for every
/// column. A compile-time parameter rather than a null check, so the
/// inner loop of the common case carries no branch and the y == 0 case
/// collapses to `p2 & p3` -- maj3(0, b, c) is exactly that.
/// @param above Row y - 1 of the source. Unread when HasAbove is false.
/// @param cur Row y of the source: p2, and the row p3 is shifted out of.
/// @param dst Row y of the destination.
/// @param words Whole words a row of `width` pixels occupies -- at least 1.
/// @param tailMask The live bits of the last of those words.
/// @note THE SOURCE'S TRAILING WORD IS MASKED BEFORE THE SHIFT. `cur[words - 1]`
/// may carry the caller's padding bits, and the right-neighbour of pixel
/// `width - 1` is bit `width % WordBits` of that word. Zero fill (see the
/// border section at the top of this file) means that neighbour must read
/// 0, so the mask is not tidiness -- it is the border.
template <typename WordType, bool HasAbove>
inline void medianRow3(const WordType* above, const WordType* cur, WordType* dst, size_t words,
                       WordType tailMask) {
    constexpr size_t wordBits = sizeof(WordType) * 8;

    // Whole words: the right-neighbour word is this word shifted one pixel down
    // (pixel x lives in bit x % WordBits, so "one column left" is `>> 1`) with the
    // next word's first pixel arriving in the top bit. Both casts are deliberate
    // narrowings -- `>>` and `<<` on uint8_t/uint16_t promote to int -- and are
    // what -Wconversion asks a reader to be told about.
    for (size_t i = 0; i + 1 < words; ++i) {
        const WordType c = cur[i];
        // cur[i + 1]'s bit 0 is pixel (i + 1) * WordBits, which is inside `width`
        // for every i here, so it never needs the tail mask.
        const WordType carriedIn = static_cast<WordType>(cur[i + 1] << (wordBits - 1));
        const WordType r = static_cast<WordType>(static_cast<WordType>(c >> 1) | carriedIn);
        dst[i] = HasAbove ? maj3<WordType>(above[i], c, r) : static_cast<WordType>(c & r);
    }

    // The trailing word. `c` is masked first (see the note above), so the shift
    // brings a zero into pixel width - 1 whatever the source's padding held, and
    // there is no next word to take a bit from.
    //
    // THE STORE IS NOT MASKED AGAIN, AND THAT IS MEASURED RATHER THAN ASSUMED.
    // Masking `c` puts zeros in every bit past `width` of BOTH `c` and `r`, and
    // majority needs two of three: maj3(anything, 0, 0) == 0, and `c & r` == 0.
    // So the destination's padding is already zero when it is stored, whatever
    // the ABOVE row's padding holds -- and a second `& tailMask` here would be an
    // operation no test can observe. Measured: the store WAS masked when this
    // kernel was written, and adding that mask back changes nothing -- 9744 of
    // 9744 checks pass either way, in every configuration. Deleting the mask on
    // `c` above fails 1215 of the same 9744.
    //
    // The coupling that creates is stated so it cannot be broken silently: in
    // this kernel the padding invariant (CLAUDE.md;) is carried by the mask
    // on `c`, which is there for a BORDER reason -- the last column's right
    // neighbour must read 0. Anything that changes this filter's border must
    // re-establish the invariant explicitly rather than assume it survived.
    const WordType c = static_cast<WordType>(cur[words - 1] & tailMask);
    const WordType r = static_cast<WordType>(c >> 1);
    dst[words - 1] = HasAbove ? maj3<WordType>(above[words - 1], c, r)
                              : static_cast<WordType>(c & r);
}

} // namespace impl

// ---------------------------------------------------------------------------
// The kernel ( views, never containers)
// ---------------------------------------------------------------------------

/// @brief dst[y][x] = median(src[y-1][x], src[y][x], src[y][x+1]), with the
/// out-of-image neighbours reading 0. **API TIER 3.**
///
/// The reference pipeline's `three_pix_median_filter`, bit-parallel: for binary
/// pixels the median of three IS their majority (the design notes), so each word
/// of the destination costs one `maj3` over 8..64 pixels.
///
/// @param src Source view.
/// @param dst Destination view; must have src's dimensions, and must share no
/// word with src. **In place is NOT supported** -- see the aliasing
/// section at the top of this file.
///
/// @note THE NEIGHBOURHOOD IS ASYMMETRIC AND THAT IS NOT A BUG: above, self, and
/// right. It is read directly out of the reference implementation, quoted
/// at the top of this file, and there is no left or below neighbour.
/// @note THE BORDER IS ZERO FILL, matching the reference's two `cv::Mat::zeros`
/// neighbour matrices: row 0's above-neighbour is 0 and column
/// `width - 1`'s right-neighbour is 0. Not replicate, not reflect. On a
/// {0,1} image a zero neighbour can only pull a pixel DOWN (majority with a
/// 0 needs both remaining pixels set), so the filter erodes the top row and
/// the last column slightly -- which is what the reference does, and
/// therefore what binCV must do.
/// @note One pass, no scratch buffer, no allocation. The above-neighbour is a row
/// index, not a copy; the right-neighbour is computed into a register.
/// @note The destination's padding bits are zero on return, and the source's
/// padding bits are never read as pixels -- they are masked off before the
/// right-shift, which is what keeps the last column's neighbour at 0 for a
/// source whose padding is dirty. That single mask carries both properties;
/// see property 3 at the top of this file.
/// @note PRECONDITION ON `dst`: it must span its image's full width, or end on a
/// word boundary. See the top of this file.
/// @note Empty views (width or height 0) are a no-op, not an error.
/// @note Never throws and never allocates (the design notes). Mismatched
/// dimensions, a stride shorter than a row, and any overlap between src and
/// dst are programming errors: BINCV_ASSERT reports them in debug builds
/// and they are undefined in release, exactly as an out-of-range at is.
/// @note There is deliberately no QuantMat<N> overload, for ops/bitslice.hpp's
/// reason: bit 3 of a median is not the median of three bit 3s, so a
/// per-plane loop would compile, run, and be wrong. The N-bit median is a
/// sorting network over bit-sliced values and nothing in the MVP asks for
/// one -- the reference filters the BINARY frame.
template <typename WordType>
inline void denoiseMedian3(BinMatConstView<WordType> src, BinMatView<WordType> dst) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "denoiseMedian3: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(src.width, src.height, src.stride) &&
                     impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "denoiseMedian3: every view's stride must cover a whole row");
    BINCV_ASSERT(impl::viewsShareNoWord(src, dst),
                 "denoiseMedian3: dst must share no word with src (in place is not supported)");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "denoiseMedian3: a non-empty view needs a non-null pointer");

    const size_t words = impl::minRowWords<WordType>(dst.width);
    const WordType tailMask = impl::rowTailMask<WordType>(dst.width);

    // Row 0: the above-neighbour is the zero border for every column, so the
    // median collapses to p2 & p3. Written as its own call rather than as a
    // branch inside the loop -- see medianRow3's HasAbove.
    impl::medianRow3<WordType, false>(nullptr, src.row(0), dst.row(0), words, tailMask);

    for (size_t y = 1; y < dst.height; ++y) {
        impl::medianRow3<WordType, true>(src.row(y - 1), src.row(y), dst.row(y), words, tailMask);
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
