#pragma once

#include <cstddef>
#include <type_traits>

// BINCV_ASSERT for the row() precondition, and BINCV_ABI_NAMESPACE. The views
// are the type kernels bind to, so their preconditions are exactly the hot-path
// case the project's error policy is written for -- not a place for a raw
// assert() and its stringified-condition-only diagnostic (ARCHITECTURE 5.3).
#include "error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

template <typename WordType_> struct BinMatConstView;

/// @brief Non-owning, mutable view of a bit-packed matrix: {ptr, size, stride}.
/// @tparam WordType_ The unsigned integral type pixels are packed into.
///
/// @note This is the type kernels bind to (D-5). A kernel compiles once per
///       (WordType, N) and works regardless of how its arguments were allocated
///       or aligned, because everything variable about the layout is here at
///       runtime.
/// @note Views manage no lifetime whatsoever. The referenced memory belongs to
///       something else -- a Storage, a caller's buffer, a sensor DMA region --
///       and must outlive the view.
/// @note Construct with BinMatView<W>{ptr, width, height, stride} -- all four.
template <typename WordType_>
struct BinMatView {
    static_assert(std::is_same<WordType_, typename std::remove_cv<WordType_>::type>::value,
                  "WordType must not be cv-qualified; use BinMatConstView for read-only access (D-9)");
    static_assert(std::is_integral<WordType_>::value && std::is_unsigned<WordType_>::value &&
                      !std::is_same<WordType_, bool>::value &&
                      !std::is_same<WordType_, char>::value,
                  "WordType must be an unsigned integral type, and not bool or char");
    static_assert(sizeof(WordType_) == 1 || sizeof(WordType_) == 2 ||
                      sizeof(WordType_) == 4 || sizeof(WordType_) == 8,
                  "WordType must be 8, 16, 32, or 64 bits wide");

    /// The storage word type, exposed so kernels can name it.
    using WordType = WordType_;

    /// Number of pixels packed into a single word.
    static constexpr size_t WordBits = sizeof(WordType) * 8;

    WordType* ptr = nullptr;  ///< first word of row 0
    size_t width = 0;         ///< row length in PIXELS
    size_t height = 0;        ///< number of rows in PIXELS
    size_t stride = 0;        ///< distance between rows in WORDS

    BinMatView() = default;
    /// @note **FOUR ARGUMENTS, AND THE CONSTRUCTOR EXISTS SO THREE WILL NOT COMPILE.**
    ///       As a bare aggregate this type accepted `{ptr, width, height}` and left
    ///       `stride` zero, which makes every row of a multi-row view alias row 0. The
    ///       `row()` precondition below catches it in a checked build and is a **no-op
    ///       in release**, so a release binary runs on garbage and looks like it works.
    ///
    ///       **That is not hypothetical.** A binCV user hit exactly this on
    ///       `ResponseMap`, whose three-argument construction in
    ///       `examples/vio_frontend.cpp` inflated the detector's NMS survivor count
    ///       threefold; only their checked build caught it. `ResponseMap` is fixed the
    ///       same way, and this type had the identical hole.
    ///
    ///       A debug-only assertion is the right guard for an index; it is the wrong
    ///       guard for a field a caller can silently fail to supply.
    BinMatView(WordType* ptr_, size_t width_, size_t height_, size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}

    /// @brief True if the view addresses no pixels.
    bool empty() const { return ptr == nullptr || width == 0 || height == 0; }

    /// @brief First word of row y.
    /// @note Unchecked: y must be in [0, height). Kernels index rows in their
    ///       inner loops, so this cannot afford a branch.
    /// @note stride is a runtime value in words, and is not assumed to be any
    ///       particular multiple -- kernels must never assume alignment.
    /// @note Precondition, asserted in debug builds only: a multi-row view must
    ///       carry a non-zero stride. This is the one field whose omission from
    ///       the aggregate initializer is not self-announcing -- a missing ptr,
    ///       width or height leaves the view empty(), whereas a missing stride
    ///       yields a plausible-looking view in which every row aliases row 0
    ///       (ARCHITECTURE 5.3: an inconsistent view is a programming error,
    ///       caught by assertion in debug).
    WordType* row(size_t y) {
        BINCV_ASSERT(stride != 0 || height <= 1,
                     "BinMatView: multi-row view needs a non-zero stride");
        return ptr + y * stride;
    }
    const WordType* row(size_t y) const {
        BINCV_ASSERT(stride != 0 || height <= 1,
                     "BinMatView: multi-row view needs a non-zero stride");
        return ptr + y * stride;
    }

    /// @brief Implicitly converts to the read-only view (D-9).
    /// @note Defined out of line below, once BinMatConstView is complete.
    /// @note Template argument deduction does not consider user-defined
    ///       conversions, so a kernel declared as f(BinMatConstView<W>) will not
    ///       deduce W from a BinMatView<W>. Callers either name the argument
    ///       explicitly as f<W>(v), or pass something that already is a
    ///       BinMatConstView -- which is what BinMat::constView() (T1.3) returns.
    ///       The conversion still applies at every non-deduced call, and to any
    ///       concrete BinMatConstView parameter.
    operator BinMatConstView<WordType>() const;
};

/// @brief Non-owning, read-only view of a bit-packed matrix.
/// @tparam WordType_ The unsigned integral type pixels are packed into.
///
/// @note Identical to BinMatView except that the referenced words are const.
///       These are two distinct types rather than BinMatView<const WordType>
///       (D-9): templating on constness fights the unsigned-integral constraint
///       on WordType and produces unreadable diagnostics.
/// @note Construct with BinMatConstView<W>{ptr, width, height, stride} -- all four.
template <typename WordType_>
struct BinMatConstView {
    static_assert(std::is_same<WordType_, typename std::remove_cv<WordType_>::type>::value,
                  "WordType must not be cv-qualified; BinMatConstView already makes the words const (D-9)");
    static_assert(std::is_integral<WordType_>::value && std::is_unsigned<WordType_>::value &&
                      !std::is_same<WordType_, bool>::value &&
                      !std::is_same<WordType_, char>::value,
                  "WordType must be an unsigned integral type, and not bool or char");
    static_assert(sizeof(WordType_) == 1 || sizeof(WordType_) == 2 ||
                      sizeof(WordType_) == 4 || sizeof(WordType_) == 8,
                  "WordType must be 8, 16, 32, or 64 bits wide");

    /// The storage word type, exposed so kernels can name it.
    using WordType = WordType_;

    /// Number of pixels packed into a single word.
    static constexpr size_t WordBits = sizeof(WordType) * 8;

    const WordType* ptr = nullptr;  ///< first word of row 0
    size_t width = 0;               ///< row length in PIXELS
    size_t height = 0;              ///< number of rows in PIXELS
    size_t stride = 0;              ///< distance between rows in WORDS

    BinMatConstView() = default;
    /// @note **FOUR ARGUMENTS, AND THE CONSTRUCTOR EXISTS SO THREE WILL NOT COMPILE.**
    ///       As a bare aggregate this type accepted `{ptr, width, height}` and left
    ///       `stride` zero, which makes every row of a multi-row view alias row 0. The
    ///       `row()` precondition below catches it in a checked build and is a **no-op
    ///       in release**, so a release binary runs on garbage and looks like it works.
    ///
    ///       **That is not hypothetical.** A binCV user hit exactly this on
    ///       `ResponseMap`, whose three-argument construction in
    ///       `examples/vio_frontend.cpp` inflated the detector's NMS survivor count
    ///       threefold; only their checked build caught it. `ResponseMap` is fixed the
    ///       same way, and this type had the identical hole.
    ///
    ///       A debug-only assertion is the right guard for an index; it is the wrong
    ///       guard for a field a caller can silently fail to supply.
    BinMatConstView(const WordType* ptr_, size_t width_, size_t height_, size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}

    /// @brief True if the view addresses no pixels.
    bool empty() const { return ptr == nullptr || width == 0 || height == 0; }

    /// @brief First word of row y.
    /// @note Unchecked: y must be in [0, height). See BinMatView::row, including
    ///       the debug-only non-zero-stride precondition.
    const WordType* row(size_t y) const {
        BINCV_ASSERT(stride != 0 || height <= 1,
                     "BinMatConstView: multi-row view needs a non-zero stride");
        return ptr + y * stride;
    }
};

template <typename WordType_>
inline BinMatView<WordType_>::operator BinMatConstView<WordType_>() const {
    return BinMatConstView<WordType>{ptr, width, height, stride};
}

/// @brief Reads a 64-bit bit-plane as a 32-bit one. **API TIER 3.** No copy.
///
/// **A 64-BIT PLANE ALREADY IS A 32-BIT PLANE WITH TWICE THE STRIDE.** binCV puts pixel
/// `x` at bit `x % WordBits` of word `x / WordBits`, so on a little-endian machine the
/// two layouts are the same bytes: for `b = x % 64 < 32` the 32-bit index is `2w` and
/// the bit is `b`; for `b >= 32` it is `2w + 1` and `b - 32`. Both halves match, and
/// `tests/test_view_narrow.cpp` checks it pixel by pixel rather than asserting it.
///
/// **WHY IT EXISTS.** Every vectorised tracking kernel is gated on 32-bit words
/// ([D-73](../../../../docs/ARCHITECTURE.md)) — the AVX2 keypoint batch and all four NEON
/// residual kernels — because an LK window is 31 pixels and a wider word is more than
/// half idle. A caller who wants 64-bit words for the rest of their pipeline, where
/// they genuinely halve the work, used to face a choice between that and a tracker
/// running 8.6× slow. **This removes the choice**: keep the 64-bit storage, narrow the
/// view at the call, and the vector kernels apply.
///
/// @note Padding stays zero, which is the invariant word-wise reductions depend on: a
///       64-bit word's zero padding bits are the derived 32-bit word's zero padding
///       bits, and a fully-padding high half is simply an all-zero word.
/// @note Alignment carries: an 8-byte-aligned buffer is 4-byte aligned.
template <typename WordType>
inline BinMatConstView<uint32_t> narrowPlane(BinMatConstView<WordType> v) {
    static_assert(sizeof(WordType) == 8,
                  "narrowPlane: only a 64-bit plane can be read as a 32-bit one");
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                  "narrowPlane: the two layouts coincide only on a little-endian machine");
#endif
    return BinMatConstView<uint32_t>{reinterpret_cast<const uint32_t*>(v.ptr), v.width,
                                     v.height, v.stride * 2};
}

/// @brief The same reinterpretation for a WRITABLE plane. **API TIER 3.** No copy.
///
/// **The const form covers reading, which is all the tracker needs. This covers the
/// stage before it.** `edgeThreshold` and `packQuant` write bit-planes and are gated on
/// 32-bit words for the same reason the tracker is — their vector arms use 32-lane
/// move-masks — so a caller with 64-bit storage loses the sensor stage's vector path
/// too, and `narrowPlane`'s const overload cannot help because those kernels write.
///
/// @note **The aliasing rule is the caller's and it is not decorative.** The narrowed
///       view and the original address the same bytes, so writing through one while
///       reading the other in the same kernel call is exactly the overlap D-11's
///       predicates reject. Narrow the DESTINATION, pass the source as it is, and do not
///       hold both spellings of the same plane across a call that writes.
/// @note Everything the const form promises holds here: padding stays zero, alignment
///       carries, and the layouts coincide only on a little-endian machine.
template <typename WordType>
inline BinMatView<uint32_t> narrowPlaneMutable(BinMatView<WordType> v) {
    static_assert(sizeof(WordType) == 8,
                  "narrowPlaneMutable: only a 64-bit plane can be written as a 32-bit one");
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                  "narrowPlaneMutable: the two layouts coincide only on little-endian");
#endif
    return BinMatView<uint32_t>{reinterpret_cast<uint32_t*>(v.ptr), v.width, v.height,
                                v.stride * 2};
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
