#pragma once

/// @file logic.hpp
/// @brief Bitwise logic over bit-packed binary images (T2.2). API TIER 1.
///
/// The first kernels in the project, and the first test of the thesis: a binary
/// image stored one bit per pixel means an AND over a 640x480 frame is 9600
/// 32-bit words rather than 307200 bytes, so the same answer moves an eighth of
/// the memory (ARCHITECTURE 4.6, 6.1).
///
/// Four operations, each bit-exact against its OpenCV counterpart on equivalent
/// content (ARCHITECTURE 5.1, Tier 1) -- proven across the T2.1 size and fill
/// matrix by tests/test_logic.cpp, not asserted here.
///
/// ---------------------------------------------------------------------------
/// WHAT A KERNEL IN THIS FILE PROMISES
///
///  1. **Views, never containers** (D-5). Every kernel takes BinMatConstView /
///     BinMatView, so it compiles once per WordType and does not care how its
///     arguments were allocated or aligned. The QuantMat overloads at the bottom
///     are a plane loop over these kernels, not a second implementation.
///
///  2. **Strides may differ between the arguments, and are read per row.** The
///     single-contiguous-run path is taken only when all three strides equal
///     ceil(width / WordBits) -- i.e. when the rows really are dense and adjacent
///     -- and every other case walks row by row through view.row(y). A kernel
///     that assumed one dense run would be correct on matrices built the same way
///     and silently wrong the moment one argument is over-aligned (D-4 makes that
///     a per-object choice) or wraps a caller's buffer with its own stride.
///
///  3. **PADDING BITS STAY ZERO** (CLAUDE.md, hard rules). Every kernel here
///     writes whole words, so the trailing partial word of each row carries bits
///     past `width` that no pixel comparison can see -- and bitwiseNot SETS every
///     one of them. They are masked off before the word is stored. Measured
///     during T2.1: a word-wise NOT without the mask was bit-exact against
///     cv::bitwise_not on all 240 swept cases at uint64_t and left 826,200
///     phantom set bits behind, which the next word-wise reduction counts as
///     pixels. The mask is applied by all four operations, not only by NOT, so a
///     destination is clean even when an input's padding was dirty (a wrapped
///     buffer's padding belongs to its caller -- see BinMat's wrap constructor).
///
///  4. **No allocation, and no throw** (ARCHITECTURE 5.3). Mismatched dimensions,
///     a stride too short to hold a row, and unsafe aliasing are programming
///     errors, reported by BINCV_ASSERT in debug builds and undefined in release,
///     exactly as at() is.
///
///  5. **A DESTINATION IS WRITTEN A WHOLE WORD AT A TIME.** See the precondition
///     section below -- it is the one thing about these kernels a caller can get
///     wrong without any check catching it.
///
/// ---------------------------------------------------------------------------
/// PRECONDITION ON `dst`: IT MAY NOT BE A SUB-WIDTH WINDOW ONTO A WIDER IMAGE
///
/// Property 3 has a cost, and it is a real precondition rather than a footnote.
/// The trailing partial word of every destination row is stored masked, which
/// writes ZEROS into bits [width, minRowWords * WordBits) of that word. When those
/// bits are padding -- the usual case -- that is the invariant. When `dst` is a
/// window onto a WIDER image, they are the next 1..WordBits-1 pixels of that
/// image, and they are destroyed.
///
/// So a destination view must either span its image's full width, or end on a word
/// boundary (`width % WordBits == 0`). Measured: a 70-pixel-wide destination
/// windowed onto a 640-wide, 2-row image cleared all 52 live pixels in columns
/// 70..95. Nothing diagnoses it -- every address written is inside the parent
/// image, so it is neither an out-of-bounds write nor an aliasing violation, and a
/// pixel comparison against the window sees the right answer.
///
/// This is the same hazard BinMat's wrap constructor already carries for fill()
/// and pad(), stated here because a caller building a windowed VIEW never reads
/// that constructor. Sources are unaffected: nothing past `width` is ever read.
///
/// ---------------------------------------------------------------------------
/// ALIASING: `dst` MAY BE `a` OR `b` EXACTLY, OR SHARE NO MEMORY WITH THEM
///
/// In-place is supported and tested: `bitwiseAnd(m.constView(), other, m.view())`
/// computes m &= other. It is safe because these operations are pointwise in the
/// word index -- word i of the destination is written from word i of each source
/// and from nothing else -- so a destination that maps to the same word for every
/// index reads what it is about to overwrite.
///
/// The two legal cases are therefore "exactly the same words" and "no shared word
/// at all". A destination that overlaps a source at a DIFFERENT offset or stride
/// is undefined: word i of dst may then be word j of the source, and the row loop
/// would read a word it has already written. That case is asserted against in
/// debug builds (destinationAliasIsSafe), because it cannot be diagnosed by any
/// sanitizer -- the memory is valid, the result is simply wrong for some pixels.
///
/// "No shared word at all" is checked EXACTLY, per row, and not by comparing the
/// two views' bounding spans. Two views over one buffer can interleave without
/// sharing a byte -- alternate row bands (the shape a pyramid downsample takes,
/// ARCHITECTURE 7.2) and left/right column tiles both do -- and a bounding-box
/// test rejects every one of them. D-5 says a kernel takes any
/// {ptr, width, height, stride}; rejecting a legal view in debug and accepting it
/// in release is the worst of both.
///
/// Empty views (width or height 0) are a no-op, not an error: a 0-column frame
/// has nothing to compute and every loop below runs zero times.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// QuantMat<N> and BinMat, for the per-plane overloads at the bottom of this file
// -- which are the ONLY reason a kernel header names a container. It also carries
// impl::minRowWords / impl::lowBitsMask, the two word helpers the row arithmetic
// below is written in terms of.
#include "../quantMat.hpp"

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

/// @brief The aliasing contract at the top of this file, as a predicate.
/// @return true if `dst` is exactly `src` (same first word, same stride -- so
///         word i of one is word i of the other), or shares no memory with it.
/// @note Called only from BINCV_ASSERT, so it is compiled into debug builds and
///       not into release ones. Dimensions are checked separately and are equal
///       by the time this runs, which is what makes "same pointer and stride"
///       sufficient for "the same word index means the same word".
/// @note Four steps, cheapest first, and every one of them can only ACCEPT: the
///       verdict "unsafe" is reached exactly once, by the exhaustive per-row test.
///       The earlier steps existed on their own before, and a bounding-box test
///       alone rejected interleaved row bands and column tiles that share no byte
///       -- legal views under D-5, correct in release, and an abort in debug.
/// @note `stride == dst.stride` is not required when the destination is a single
///       row: stride addresses nothing then (row(0) == ptr), and BinMatView::row
///       already carries the same `height <= 1` exemption for the same reason.
template <typename WordType>
inline bool destinationAliasIsSafe(const BinMatConstView<WordType>& src,
                                   const BinMatView<WordType>& dst) {
    // 1. Exactly the same words in the same order: the supported in-place case.
    if (static_cast<const void*>(src.ptr) == static_cast<const void*>(dst.ptr) &&
        (src.stride == dst.stride || dst.height <= 1)) {
        return true;
    }

    // 2. Nothing is addressed, so nothing can be shared. (A null pointer on a
    //    non-empty view has its own assert; it is not this predicate's verdict.)
    if (src.ptr == nullptr || dst.ptr == nullptr) return true;
    if (src.width == 0 || src.height == 0 || dst.width == 0 || dst.height == 0) return true;

    // 3. Bounding spans disjoint -- two separate allocations, the common case,
    //    and the only step that is one comparison rather than a loop.
    const size_t srcWords = viewSpanWords<WordType>(src.width, src.height, src.stride);
    const size_t dstWords = viewSpanWords<WordType>(dst.width, dst.height, dst.stride);
    if (byteRangesDisjoint(src.ptr, srcWords * sizeof(WordType),
                           dst.ptr, dstWords * sizeof(WordType))) {
        return true;
    }

    // 4. Overlapping spans, which is not the same as overlapping rows.
    if (sameStrideRowsCannotMeet(src, dst)) return true;
    return everyRowPairIsDisjoint(src, dst);
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

// The four word operations, as types rather than function pointers so that each
// kernel instantiation inlines to the single machine instruction it is.
//
// The casts are not decoration. Integer promotion runs `&`, `|`, `^` and `~` in
// `int` for uint8_t and uint16_t, so storing the result back into a WordType is a
// narrowing conversion -- which is exactly what -Wconversion reports, and what
// makes a deliberate truncation indistinguishable from an accidental one unless
// it is written down (CLAUDE.md: "a cast is where a reader is told the truncation
// is intended").
struct BitAnd {
    template <typename WordType>
    static WordType apply(WordType a, WordType b) { return static_cast<WordType>(a & b); }
};
struct BitOr {
    template <typename WordType>
    static WordType apply(WordType a, WordType b) { return static_cast<WordType>(a | b); }
};
struct BitXor {
    template <typename WordType>
    static WordType apply(WordType a, WordType b) { return static_cast<WordType>(a ^ b); }
};
struct BitNot {
    template <typename WordType>
    static WordType apply(WordType a) { return static_cast<WordType>(~a); }
};

/// @brief The two-input kernel body: dst = Op(a, b), word-wise, padding cleared.
/// @note Shared by AND, OR and XOR. They differ in one instruction and in nothing
///       else -- not in the stride handling, not in the tail mask, not in the
///       aliasing contract -- so writing them out three times would be three
///       chances for those to drift apart.
template <typename WordType, typename Op>
inline void applyBinary(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                        BinMatView<WordType> dst) {
    BINCV_ASSERT(a.width == dst.width && a.height == dst.height &&
                     b.width == dst.width && b.height == dst.height,
                 "bitwise op: a, b and dst must have the same dimensions");
    BINCV_ASSERT(strideCoversARow<WordType>(a.width, a.height, a.stride) &&
                     strideCoversARow<WordType>(b.width, b.height, b.stride) &&
                     strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "bitwise op: every view's stride must cover a whole row");
    BINCV_ASSERT(destinationAliasIsSafe(a, dst) && destinationAliasIsSafe(b, dst),
                 "bitwise op: dst must alias an input exactly or not overlap it");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && dst.ptr != nullptr,
                 "bitwise op: a non-empty view needs a non-null pointer");

    const size_t words = minRowWords<WordType>(dst.width);
    const WordType tailMask = rowTailMask<WordType>(dst.width);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    // One contiguous run, and ONLY when the arguments earn it: every row dense
    // (stride == the words a row needs) in all three views, and no partial word
    // to mask. Anything else -- an over-aligned matrix (D-4), a wrapped buffer
    // with the caller's own stride, a width that is not a multiple of WordBits --
    // goes through the row loop below.
    if (tailMask == allOnes && a.stride == words && b.stride == words &&
        dst.stride == words) {
        const size_t total = words * dst.height;
        const WordType* pa = a.ptr;
        const WordType* pb = b.ptr;
        WordType* pd = dst.ptr;
        for (size_t i = 0; i < total; ++i) {
            pd[i] = Op::template apply<WordType>(pa[i], pb[i]);
        }
        return;
    }

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* ra = a.row(y);
        const WordType* rb = b.row(y);
        WordType* rd = dst.row(y);

        // Whole words first, so the loop the compiler vectorizes carries no mask,
        // then the trailing word with the padding bits removed. words >= 1 here,
        // since width > 0.
        for (size_t i = 0; i + 1 < words; ++i) {
            rd[i] = Op::template apply<WordType>(ra[i], rb[i]);
        }
        rd[words - 1] = static_cast<WordType>(
            Op::template apply<WordType>(ra[words - 1], rb[words - 1]) & tailMask);
    }
}

/// @brief The one-input kernel body: dst = Op(src), word-wise, padding cleared.
/// @note THE PADDING MASK IS LOAD-BEARING HERE in a way it is not for AND: `~`
///       sets every bit past `width` in the trailing word regardless of what the
///       source held. See property 3 at the top of this file for what that cost
///       when it was missing.
template <typename WordType, typename Op>
inline void applyUnary(BinMatConstView<WordType> src, BinMatView<WordType> dst) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "bitwise op: src and dst must have the same dimensions");
    BINCV_ASSERT(strideCoversARow<WordType>(src.width, src.height, src.stride) &&
                     strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "bitwise op: every view's stride must cover a whole row");
    BINCV_ASSERT(destinationAliasIsSafe(src, dst),
                 "bitwise op: dst must alias an input exactly or not overlap it");

    if (dst.width == 0 || dst.height == 0) return;

    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "bitwise op: a non-empty view needs a non-null pointer");

    const size_t words = minRowWords<WordType>(dst.width);
    const WordType tailMask = rowTailMask<WordType>(dst.width);
    const WordType allOnes = static_cast<WordType>(~static_cast<WordType>(0));

    if (tailMask == allOnes && src.stride == words && dst.stride == words) {
        const size_t total = words * dst.height;
        const WordType* ps = src.ptr;
        WordType* pd = dst.ptr;
        for (size_t i = 0; i < total; ++i) {
            pd[i] = Op::template apply<WordType>(ps[i]);
        }
        return;
    }

    for (size_t y = 0; y < dst.height; ++y) {
        const WordType* rs = src.row(y);
        WordType* rd = dst.row(y);
        for (size_t i = 0; i + 1 < words; ++i) {
            rd[i] = Op::template apply<WordType>(rs[i]);
        }
        rd[words - 1] = static_cast<WordType>(
            Op::template apply<WordType>(rs[words - 1]) & tailMask);
    }
}

} // namespace impl

// ---------------------------------------------------------------------------
// The kernels (D-5: views, never containers)
// ---------------------------------------------------------------------------

/// @brief dst = a & b, pixel for pixel. **API TIER 1** -- bit-exact against
///        cv::bitwise_and on the same binary content stored as CV_8U.
/// @param a First source view.
/// @param b Second source view; must have a's dimensions.
/// @param dst Destination view; must have a's dimensions. May be `a` or `b`
///        exactly (in-place), or share no memory with either -- see the aliasing
///        section at the top of this file.
/// @note Word-wise, one word per WordBits pixels, reading each view's own stride
///       per row. Padding bits past `width` are left zero in the destination.
/// @note PRECONDITION ON `dst`: it must span its image's full width, or end on a
///       word boundary. The trailing partial word is stored masked, so bits
///       [width, minRowWords * WordBits) of each destination row are set to zero;
///       in a sub-width window onto a wider image those bits are that image's next
///       pixels and are destroyed. Nothing diagnoses it -- see the precondition
///       section at the top of this file. Sources are unaffected.
/// @note Never throws and never allocates (ARCHITECTURE 5.3). Mismatched
///       dimensions, a stride shorter than a row, and overlapping-but-not-
///       identical views are programming errors: BINCV_ASSERT reports them in
///       debug builds, and they are undefined in release, exactly as an
///       out-of-range at() is.
template <typename WordType>
inline void bitwiseAnd(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                       BinMatView<WordType> dst) {
    impl::applyBinary<WordType, impl::BitAnd>(a, b, dst);
}

/// @brief dst = a | b, pixel for pixel. **API TIER 1** -- bit-exact against
///        cv::bitwise_or. See bitwiseAnd() for the shared contract.
template <typename WordType>
inline void bitwiseOr(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                      BinMatView<WordType> dst) {
    impl::applyBinary<WordType, impl::BitOr>(a, b, dst);
}

/// @brief dst = a ^ b, pixel for pixel. **API TIER 1** -- bit-exact against
///        cv::bitwise_xor. See bitwiseAnd() for the shared contract.
template <typename WordType>
inline void bitwiseXor(BinMatConstView<WordType> a, BinMatConstView<WordType> b,
                       BinMatView<WordType> dst) {
    impl::applyBinary<WordType, impl::BitXor>(a, b, dst);
}

/// @brief dst = ~src, pixel for pixel. **API TIER 1** -- bit-exact against
///        cv::bitwise_not on the same binary content stored as CV_8U (which maps
///        {0, 255} to {255, 0}).
/// @param src Source view.
/// @param dst Destination view; must have src's dimensions. May be `src` exactly
///        (in-place), or share no memory with it.
/// @note THIS IS THE OPERATION THAT SETS PADDING BITS. Inverting a word sets
///       every bit past `width` in a row's trailing partial word; they are masked
///       off before the word is stored, so a destination this kernel wrote is
///       safe to hand to a word-wise reduction. Nothing about the pixels would
///       have said otherwise -- see property 3 at the top of this file.
/// @note Same precondition on `dst` as bitwiseAnd(): full width, or a width that
///       is a multiple of WordBits. The mask that keeps padding clean is the same
///       write that would clobber a wider image's next pixels.
template <typename WordType>
inline void bitwiseNot(BinMatConstView<WordType> src, BinMatView<WordType> dst) {
    impl::applyUnary<WordType, impl::BitNot>(src, dst);
}

// ---------------------------------------------------------------------------
// QuantMat overloads: the same four operations, applied per plane
// ---------------------------------------------------------------------------
//
// Bit-plane logic is plane-wise by construction (ARCHITECTURE 4.1: "logic
// operations apply per plane and are free"), so these are a loop over plane()
// and nothing else. They are convenience wrappers over the kernels above, NOT
// kernels themselves -- which is why taking a container here does not contradict
// D-5: the compiled inner loop is still the view kernel, and a caller who holds
// views rather than containers never reaches this overload.
//
// N == 1 comes here too: BinMat IS QuantMat<1> (core/types.hpp), so
// bitwiseAnd(binMatA, binMatB, binMatDst) is a one-iteration loop over the same
// kernel, with no plane-loop cost the specialization was written to avoid.
//
// The operands must have the same plane count as well as the same dimensions,
// which the signature enforces at compile time rather than at runtime.
// SignedQuantMat has no overload of its own on purpose: `mag = pos | neg` is
// plane arithmetic, not signed arithmetic, and the right spelling for it is
// bitwiseOr(...) over the named planes or over planes() -- an overload taking the
// interpreted type would suggest these operations respect the sign convention,
// and they do not.

/// @brief Per-plane dst = a & b over all N planes. **API TIER 1** per plane.
/// @note Thin wrapper: N calls to the view kernel, one per plane, in plane order.
/// @note Same aliasing contract as the kernel, applied plane by plane -- `dst`
///       may be `a` or `b` (the planes then alias exactly, plane for plane).
template <size_t N, typename WordType>
inline void bitwiseAnd(const QuantMat<N, WordType>& a, const QuantMat<N, WordType>& b,
                       QuantMat<N, WordType>& dst) {
    for (size_t p = 0; p < N; ++p) {
        bitwiseAnd(a.plane(p), b.plane(p), dst.plane(p));
    }
}

/// @brief Per-plane dst = a | b over all N planes. **API TIER 1** per plane.
template <size_t N, typename WordType>
inline void bitwiseOr(const QuantMat<N, WordType>& a, const QuantMat<N, WordType>& b,
                      QuantMat<N, WordType>& dst) {
    for (size_t p = 0; p < N; ++p) {
        bitwiseOr(a.plane(p), b.plane(p), dst.plane(p));
    }
}

/// @brief Per-plane dst = a ^ b over all N planes. **API TIER 1** per plane.
template <size_t N, typename WordType>
inline void bitwiseXor(const QuantMat<N, WordType>& a, const QuantMat<N, WordType>& b,
                       QuantMat<N, WordType>& dst) {
    for (size_t p = 0; p < N; ++p) {
        bitwiseXor(a.plane(p), b.plane(p), dst.plane(p));
    }
}

/// @brief Per-plane dst = ~src over all N planes. **API TIER 1** per plane.
/// @note At N > 1 this is a bitwise complement of every plane, i.e. the pixel
///       value becomes MaxValue - value. That is what "bitwise not" means for an
///       N-bit unsigned image and what cv::bitwise_not does to a CV_8U one; it is
///       NOT a negation, and it has nothing to say about a sign plane.
template <size_t N, typename WordType>
inline void bitwiseNot(const QuantMat<N, WordType>& src, QuantMat<N, WordType>& dst) {
    for (size_t p = 0; p < N; ++p) {
        bitwiseNot(src.plane(p), dst.plane(p));
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
