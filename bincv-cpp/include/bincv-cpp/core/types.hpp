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

/// @brief An axis-aligned rectangle in PIXELS: origin (x, y), extent (width, height).
///
/// @note Field names and order are cv::Rect's, deliberately. A caller porting
/// `cv::countNonZero(img(roi))` writes the same four numbers here, and the
/// region reduction is Tier 1 against exactly that expression.
/// @note The rectangle covers columns [x, x + width) and rows [y, y + height) --
/// half-open, like cv::Rect. `width` and `height` are extents, not
/// coordinates of the far corner.
/// @note **Signed on purpose.** A window centerd on a keypoint near an edge has a
/// negative origin (the design notes: 31x31 windows over the whole frame),
/// and the alternative -- making the caller clamp before it can express the
/// window -- moves the same clipping arithmetic into every call site, where
/// it would be written once per caller instead of once per library. Every
/// consumer in ops/ clips against the image and is documented as doing so.
/// @note No intersection or union arithmetic here. This is the argument type a
/// kernel takes, not a geometry library; ops/reduce.hpp does the one
/// clipping operation the MVP needs, internally.
struct Rect {
    int x;       ///< Column of the left edge; may be negative (clipped by the op)
    int y;       ///< Row of the top edge; may be negative (clipped by the op)
    int width;   ///< Extent in columns; <= 0 means an empty rectangle
    int height;  ///< Extent in rows; <= 0 means an empty rectangle

    Rect() : x(0), y(0), width(0), height(0) {}
    Rect(int x_, int y_, int width_, int height_) : x(x_), y(y_), width(width_), height(height_) {}

    /// @brief True when the rectangle covers no pixel at all, before any clipping.
    /// @note A rectangle that lies wholly outside an image is NOT empty by this
    /// test -- it is empty after clipping, which is the operation's business
    /// and not the rectangle's.
    bool empty() const { return width <= 0 || height <= 0; }

    /// @brief Equality, field for field.
    bool operator==(const Rect& other) const {
        return x == other.x && y == other.y && width == other.width && height == other.height;
    }

    bool operator!=(const Rect& other) const { return !(*this == other); }
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
/// because this header is included first; a default may only be given once.
/// @note The primary template (N planes, quantMat.hpp) and the hand-written N=1
/// partial specialization (binMat.hpp) are defined in different headers, so
/// the declaration they both need lives here, ahead of either.
template <size_t N, typename WordType = uint32_t> class QuantMat;

/// @brief The 1-bit container: an alias for the N=1 specialization of QuantMat.
/// @note the design notes. BinMat is a name, not a separate type -- QuantMat<1>
/// IS the hand-written single-plane container, so a kernel or container
/// written against QuantMat<N> accepts the binary case with no adapter and
/// no plane loop. binMat.hpp defines that specialization.
/// @note One consequence of the alias, and it is the only one: class template
/// argument deduction through an alias template is C++20 (P1814), so under
/// this project's C++17 the empty argument list is required --
/// `BinMat<> m(w, h)`, not `BinMat m(w, h)`. `BinMat<uint32_t>` and the
/// BinMat8/16/32/64 aliases below are unaffected.
template <typename WordType = uint32_t> using BinMat = QuantMat<1, WordType>;

/// @brief Type aliases for convenience
/// @note These provide easy access to BinMat with different word sizes

/// @brief A point with sub-pixel coordinates -- the tracker's and the refiner's.
/// @note **Declared here rather than in ops/opticalFlow.hpp, where it used to live.**
/// `ops/subpix.hpp` needs it and nothing else from the tracker, and a two-float
/// POD is not worth including three thousand lines of Lucas-Kanade to reach.
struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

/// @brief The word type to use unless you have measured a reason not to.
///
/// **PICK THIS ONE.** `BinMat` and `QuantMat` are templated on the word type and
/// every width is correct — but **only `uint32_t` reaches the vectorized tracking
/// kernels**: the AVX2 eight-keypoint batch and all four NEON residual kernels are
/// gated on `sizeof(WordType) == 4`.
///
/// A wider word looks like it should mean fewer operations per row, and
/// a measurement measured exactly that for *reductions*. For
/// *tracking* it opts out of every vector path instead:
/// a measurement measured `uint64_t` at **1.32× slower on `track`**,
/// and an integrator who chose it for a real VIO frontend measured **8.6× slower**
/// keypoint tracking before finding the gate
///.
///
/// The tracker now refuses to compile at a depth that HAS vector kernels with a word
/// that cannot reach them, so this is a recommendation rather than a trap — but
/// spelling it `DefaultWord` is cheaper than reading that diagnostic.
using DefaultWord = uint32_t;

using BinMat8  = BinMat<uint8_t>;   ///< BinMat with 8-bit words (8 pixels per word)
using BinMat16 = BinMat<uint16_t>;  ///< BinMat with 16-bit words (16 pixels per word)
using BinMat32 = BinMat<uint32_t>;  ///< BinMat with 32-bit words (32 pixels per word) - Default
using BinMat64 = BinMat<uint64_t>;  ///< BinMat with 64-bit words (64 pixels per word)

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
