#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace bincv {

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
/// @note Aggregate: construct with BinMatView<W>{ptr, width, height, stride}.
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
        assert((stride != 0 || height <= 1) && "BinMatView: multi-row view needs a non-zero stride");
        return ptr + y * stride;
    }
    const WordType* row(size_t y) const {
        assert((stride != 0 || height <= 1) && "BinMatView: multi-row view needs a non-zero stride");
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
/// @note Aggregate: construct with BinMatConstView<W>{ptr, width, height, stride}.
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

    /// @brief True if the view addresses no pixels.
    bool empty() const { return ptr == nullptr || width == 0 || height == 0; }

    /// @brief First word of row y.
    /// @note Unchecked: y must be in [0, height). See BinMatView::row, including
    ///       the debug-only non-zero-stride precondition.
    const WordType* row(size_t y) const {
        assert((stride != 0 || height <= 1) && "BinMatConstView: multi-row view needs a non-zero stride");
        return ptr + y * stride;
    }
};

template <typename WordType_>
inline BinMatView<WordType_>::operator BinMatConstView<WordType_>() const {
    return BinMatConstView<WordType>{ptr, width, height, stride};
}

} // namespace bincv
