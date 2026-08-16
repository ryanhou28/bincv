#pragma once

/// @file kernel_util.hpp
/// @brief The vocabulary every kernel under ops/ is written in: the row tail
///        mask, the stride sanity check, and the D-11 overlap predicates.
///
/// Extracted from ops/logic.hpp during T2.3, unchanged, because the second kernel
/// header needed the same helpers and a copy would have been a second place for
/// the aliasing rule to drift from the first. D-11 binds every kernel added under
/// ops/, so the predicate that enforces it belongs where every kernel can reach
/// it rather than inside the first kernel that happened to need it.
///
/// Everything here is `impl::` and internal. It is not part of the public API and
/// carries no stability promise.
///
/// @note The two ALIAS predicates are deliberately separate, because the two
///       kernel families need different halves of D-11:
///
///         viewsShareNoWord()      -- no shared word at all.
///         destinationAliasIsSafe()-- the above, OR exactly the same words.
///
///       ops/logic.hpp uses the second: its operations are pointwise in the word
///       index, so `m &= other` reads each word immediately before overwriting it.
///       ops/shift.hpp uses the first: a shift is NOT pointwise -- word i of the
///       destination is built from words i +/- wordShift of the source -- so the
///       in-place half of D-11 does not extend to it. See the aliasing section at
///       the top of ops/shift.hpp for the case that makes it unrecoverable.

#include <cstddef>
#include <cstdint>

// impl::bitsPerWord / lowBitsMask / minRowWords / bitMask, and the two view types
// the predicates below take. binMat.hpp carries core/view.hpp and core/error.hpp
// with it and includes impl/binMat_impl.hpp (where those word helpers live) at
// its end, so this one include is the whole dependency.
#include "../binMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief Mask of the bits in a row's LAST word that actually hold pixels.
/// @note All ones when the row ends on a word boundary. This is the mask that
///       keeps the padding-bit invariant: `word & rowTailMask(width)` is the only
///       thing standing between a word-wise kernel and phantom pixels.
template <typename WordType>
inline WordType rowTailMask(size_t width) {
    const size_t tail = width % bitsPerWord<WordType>();
    return tail == 0 ? static_cast<WordType>(~static_cast<WordType>(0))
                     : lowBitsMask<WordType>(tail);
}

/// @brief Words a view addresses, from its first word to the last word of its
///        last row -- the span a kernel may touch.
/// @note Not `height * stride`: the final row's padding words past
///       ceil(width / WordBits) are outside the region kernels here write, and
///       counting them would report an overlap that cannot happen.
template <typename WordType>
inline size_t viewSpanWords(size_t width, size_t height, size_t stride) {
    if (width == 0 || height == 0) return 0;
    return (height - 1) * stride + minRowWords<WordType>(width);
}

/// @brief True when two byte ranges do not share a single byte.
/// @note Through uintptr_t rather than pointer comparison: `<` between pointers
///       into unrelated objects is unspecified, and this predicate exists
///       precisely to be handed pointers whose relationship is unknown.
inline bool byteRangesDisjoint(const void* first, size_t firstBytes,
                               const void* second, size_t secondBytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    return (a + firstBytes) <= b || (b + secondBytes) <= a;
}

/// @brief True when two views at the SAME stride cannot share a word, whatever
///        their row counts -- an accept-only shortcut, never a rejection.
///
/// @note This is the interleaving case a bounding-box test gets wrong. With a
///       common stride S every row of either view starts at `base + k * S` for
///       some integer k, so two rows can only meet if the pointer delta is within
///       one row's length of a multiple of S. Let `d = delta mod S`: every row
///       distance is congruent to d, so its magnitude is at least min(d, S - d),
///       and if that already exceeds a row the two views are disjoint no matter
///       how many rows each has. Alternate row bands (d == S/2) and left/right
///       column tiles (d == one row) are exactly the cases this accepts in O(1).
/// @note Returns false when it cannot decide. The exhaustive check below is then
///       the answer, so a false here never rejects anything on its own.
template <typename WordType>
inline bool sameStrideRowsCannotMeet(const BinMatConstView<WordType>& src,
                                     const BinMatView<WordType>& dst) {
    if (src.stride != dst.stride || src.stride == 0) return false;

    const uintptr_t a = reinterpret_cast<uintptr_t>(src.ptr);
    const uintptr_t b = reinterpret_cast<uintptr_t>(dst.ptr);
    const uintptr_t delta = (a < b) ? (b - a) : (a - b);
    const uintptr_t strideBytes =
        static_cast<uintptr_t>(src.stride) * static_cast<uintptr_t>(sizeof(WordType));

    const size_t srcRowWords = minRowWords<WordType>(src.width);
    const size_t dstRowWords = minRowWords<WordType>(dst.width);
    const uintptr_t rowBytes =
        static_cast<uintptr_t>((srcRowWords > dstRowWords ? srcRowWords : dstRowWords)) *
        static_cast<uintptr_t>(sizeof(WordType));

    const uintptr_t offset = delta % strideBytes;
    return offset >= rowBytes && (strideBytes - offset) >= rowBytes;
}

/// @brief True when no row of `src` shares a byte with any row of `dst`.
/// @note Exhaustive and exact: O(src.height * dst.height) byte-range tests. It is
///       reached only from BINCV_ASSERT, and only after the bounding-box and
///       same-stride tests have both failed to separate the two -- i.e. only for
///       views laid over one buffer at different strides. That is rare, small, and
///       precisely the case where being approximate is what caused the bug.
/// @note Rows are addressed by arithmetic rather than through row(), so that the
///       stride precondition row() asserts is reported by the kernel's own stride
///       check rather than from inside another assert's condition.
template <typename WordType>
inline bool everyRowPairIsDisjoint(const BinMatConstView<WordType>& src,
                                   const BinMatView<WordType>& dst) {
    const size_t srcRowBytes = minRowWords<WordType>(src.width) * sizeof(WordType);
    const size_t dstRowBytes = minRowWords<WordType>(dst.width) * sizeof(WordType);
    for (size_t ys = 0; ys < src.height; ++ys) {
        const WordType* srcRow = src.ptr + ys * src.stride;
        for (size_t yd = 0; yd < dst.height; ++yd) {
            const WordType* dstRow = dst.ptr + yd * dst.stride;
            if (!byteRangesDisjoint(srcRow, srcRowBytes, dstRow, dstRowBytes)) return false;
        }
    }
    return true;
}

/// @brief True when `src` and `dst` share no word at all -- half of D-11.
/// @note Three steps, cheapest first, and every one of them can only ACCEPT: the
///       verdict "they share a word" is reached exactly once, by the exhaustive
///       per-row test. A bounding-box test alone rejected interleaved row bands
///       and column tiles that share no byte -- legal views under D-5, correct in
///       release, and an abort in debug.
template <typename WordType>
inline bool viewsShareNoWord(const BinMatConstView<WordType>& src,
                             const BinMatView<WordType>& dst) {
    // 1. Nothing is addressed, so nothing can be shared. (A null pointer on a
    //    non-empty view has its own assert; it is not this predicate's verdict.)
    if (src.ptr == nullptr || dst.ptr == nullptr) return true;
    if (src.width == 0 || src.height == 0 || dst.width == 0 || dst.height == 0) return true;

    // 2. Bounding spans disjoint -- two separate allocations, the common case,
    //    and the only step that is one comparison rather than a loop.
    const size_t srcWords = viewSpanWords<WordType>(src.width, src.height, src.stride);
    const size_t dstWords = viewSpanWords<WordType>(dst.width, dst.height, dst.stride);
    if (byteRangesDisjoint(src.ptr, srcWords * sizeof(WordType),
                           dst.ptr, dstWords * sizeof(WordType))) {
        return true;
    }

    // 3. Overlapping spans, which is not the same as overlapping rows.
    if (sameStrideRowsCannotMeet(src, dst)) return true;
    return everyRowPairIsDisjoint(src, dst);
}

/// @brief D-11 as a predicate, for a kernel that is POINTWISE in the word index.
/// @return true if `dst` is exactly `src` (same first word, same stride -- so
///         word i of one is word i of the other), or shares no memory with it.
/// @note Called only from BINCV_ASSERT, so it is compiled into debug builds and
///       not into release ones. Dimensions are checked separately and are equal
///       by the time this runs, which is what makes "same pointer and stride"
///       sufficient for "the same word index means the same word".
/// @note `stride == dst.stride` is not required when the destination is a single
///       row: stride addresses nothing then (row(0) == ptr), and BinMatView::row
///       already carries the same `height <= 1` exemption for the same reason.
/// @note A kernel that is NOT pointwise in the word index -- ops/shift.hpp -- must
///       use viewsShareNoWord() instead. The exact-alias case is safe only because
///       word i of the destination is read from word i of the source and from
///       nothing else.
template <typename WordType>
inline bool destinationAliasIsSafe(const BinMatConstView<WordType>& src,
                                   const BinMatView<WordType>& dst) {
    // The supported in-place case: exactly the same words in the same order.
    if (static_cast<const void*>(src.ptr) == static_cast<const void*>(dst.ptr) &&
        (src.stride == dst.stride || dst.height <= 1)) {
        return true;
    }
    return viewsShareNoWord(src, dst);
}

/// @brief A view's stride is long enough to hold one of its own rows.
/// @note The one internal inconsistency a view can carry that neither the pixel
///       loop nor the aliasing check would notice: with `stride < ceil(width /
///       WordBits)` consecutive rows overlap, so the kernel reads and writes each
///       row through the previous one's tail. Measured, before this was asserted:
///       bitwiseNot over a 64x3 view at stride 1 produced three rows of
///       `ffff0000` from a source of `0000ffff`, in every build, silently.
/// @note BinMat's wrap constructor rejects the same numbers by name, so this only
///       ever fires for a hand-built view -- which is exactly what a kernel takes
///       (D-5), and therefore where the check has to live.
/// @note `height <= 1` is exempt because stride addresses nothing there, matching
///       BinMatView::row's own precondition.
template <typename WordType>
inline bool strideCoversARow(size_t width, size_t height, size_t stride) {
    return height <= 1 || stride >= minRowWords<WordType>(width);
}

} // namespace impl

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
