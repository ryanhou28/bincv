#pragma once

#include <cstdint>
#include <cstddef>

// BINCV_ABI_NAMESPACE. Every binCV header opens the same versioned inline
// namespace, so that objects compiled in different configurations cannot merge
// silently; see core/error.hpp.
#include "error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Size structure representing width and height
/// @note Replaces cv::Size in core to eliminate OpenCV dependency
struct Size {
    int width;
    int height;

    Size() : width(0), height(0) {}
    Size(int w, int h) : width(w), height(h) {}

    /// @brief Calculate the area (width * height)
    int area() const { return width * height; }

    /// @brief Check if the size is empty (zero width or height)
    bool empty() const { return width <= 0 || height <= 0; }

    /// @brief Equality comparison
    bool operator==(const Size& other) const {
        return width == other.width && height == other.height;
    }

    /// @brief Inequality comparison
    bool operator!=(const Size& other) const {
        return !(*this == other);
    }
};

/// @brief Morphological structuring element shapes
enum MorphShape {
    MORPH_RECT = 0,      ///< Rectangular structuring element
    MORPH_CROSS = 1,     ///< Cross-shaped structuring element
    MORPH_ELLIPSE = 2    ///< Elliptical structuring element
};

/// @brief Morphological operation types
enum MorphOp {
    MORPH_ERODE = 0,     ///< Erosion operation
    MORPH_DILATE = 1,    ///< Dilation operation
    MORPH_OPEN = 2,      ///< Opening operation (erode then dilate)
    MORPH_CLOSE = 3,     ///< Closing operation (dilate then erode)
    MORPH_GRADIENT = 4,  ///< Morphological gradient (dilate - erode)
    MORPH_TOPHAT = 5,    ///< Top hat (original - open)
    MORPH_BLACKHAT = 6   ///< Black hat (close - original)
};

/// @brief Border extrapolation types
enum BorderType {
    BORDER_CONSTANT = 0,   ///< Constant border (iiiiii|abcdefgh|iiiiiii with some specified i)
    BORDER_REPLICATE = 1,  ///< Replicate border (aaaaaa|abcdefgh|hhhhhhh)
    BORDER_REFLECT = 2,    ///< Reflect border (fedcba|abcdefgh|hgfedcb)
    BORDER_WRAP = 3,       ///< Wrap border (cdefgh|abcdefgh|abcdefg)
    BORDER_REFLECT_101 = 4 ///< Reflect 101 border (gfedcb|abcdefgh|gfedcba)
};

/// @brief Forward declaration of the QuantMat template -- the N-bit container.
/// @note The default template argument is declared here (not on the definition)
///       because this header is included first; a default may only be given once.
/// @note The primary template (N planes, quantMat.hpp) and the hand-written N=1
///       partial specialization (binMat.hpp) are defined in different headers, so
///       the declaration they both need lives here, ahead of either.
template <size_t N, typename WordType = uint32_t> class QuantMat;

/// @brief The 1-bit container: an alias for the N=1 specialization of QuantMat.
/// @note ARCHITECTURE 4.4. BinMat is a name, not a separate type -- QuantMat<1>
///       IS the hand-written single-plane container, so a kernel or container
///       written against QuantMat<N> accepts the binary case with no adapter and
///       no plane loop. binMat.hpp defines that specialization.
/// @note One consequence of the alias, and it is the only one: class template
///       argument deduction through an alias template is C++20 (P1814), so under
///       this project's C++17 the empty argument list is required --
///       `BinMat<> m(w, h)`, not `BinMat m(w, h)`. `BinMat<uint32_t>` and the
///       BinMat8/16/32/64 aliases below are unaffected.
template <typename WordType = uint32_t> using BinMat = QuantMat<1, WordType>;

/// @brief Type aliases for convenience
/// @note These provide easy access to BinMat with different word sizes

using BinMat8  = BinMat<uint8_t>;   ///< BinMat with 8-bit words (8 pixels per word)
using BinMat16 = BinMat<uint16_t>;  ///< BinMat with 16-bit words (16 pixels per word)
using BinMat32 = BinMat<uint32_t>;  ///< BinMat with 32-bit words (32 pixels per word) - Default
using BinMat64 = BinMat<uint64_t>;  ///< BinMat with 64-bit words (64 pixels per word)

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
