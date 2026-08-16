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
/// The predicate itself lives in impl/kernel_util.hpp, shared with ops/shift.hpp
/// (D-11 binds every kernel under ops/, so one copy is the only way the rule
/// cannot drift). Note that shift uses the OTHER half of it: the exact-alias case
/// below is legal only because these operations are pointwise in the word index,
/// which a shift is not.
///
/// Empty views (width or height 0) are a no-op, not an error: a 0-column frame
/// has nothing to compute and every loop below runs zero times.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
// impl::rowTailMask, impl::strideCoversARow and impl::destinationAliasIsSafe --
// the row-geometry and D-11 aliasing vocabulary shared with every other kernel
// under ops/. They lived in this file until T2.3 needed the same three, at which
// point one copy became the only way the aliasing rule cannot drift.
#include "../impl/kernel_util.hpp"
// QuantMat<N> and BinMat, for the per-plane overloads at the bottom of this file
// -- which are the ONLY reason a kernel header names a container. It also carries
// impl::minRowWords / impl::lowBitsMask, the two word helpers the row arithmetic
// below is written in terms of.
#include "../quantMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

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
