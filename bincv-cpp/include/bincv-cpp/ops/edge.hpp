#pragma once

/// @file edge.hpp
/// @brief Gradient-magnitude edge extraction, 8- or 16-bit in, 1 bit out.
///        **API TIER 3** -- there is no `cv::` equivalent; see the tier note.
///
/// ---------------------------------------------------------------------------
/// READ OUT OF THE REFERENCE, NOT INFERRED
///
/// `SEAL/src/temporal_processing/edge_filter.cpp`, `rl_fast_edge_filter_wide`:
///
///     kernel_x = [-1  0  1]      diff_x = |filter2D(img, kernel_x)|
///     kernel_y = [-1  0  1]^T    diff_y = |filter2D(img, kernel_y)|
///     mask     = (diff_x >= t) OR (diff_y >= t)
///
/// THREE DETAILS THAT ARE EASY TO GET WRONG, AND ALL THREE ARE THE DEFAULTS:
///
///   * the combination is **OR**, not AND;
///   * the relation is **`>=`**, not `>`;
///   * **"wide" is the CENTRAL difference** `[-1, 0, 1]` -- the left neighbour
///     against the RIGHT neighbour, spanning two pixels -- not an adjacent `[-1, 1]`.
///
/// A caller who says nothing gets exactly the reference's operation.
///
/// ---------------------------------------------------------------------------
/// ALL TWELVE COMBINATIONS SHIP
///
/// combine {Or, And} x relation {Ge, Gt} x spatial {Wide, Forward, Backward}. They
/// are compile-time parameters, so a caller pays only for the one instantiated and
/// the comparison folds to a single predicate -- the same requirement ops/pack.hpp
/// puts on its rules, for the same measured reason
/// ([X-72](../../../EXPERIMENTS.md): a runtime flag cost 17% elsewhere).
///
/// The point of the operation is that these choices are cheap. A caller wanting AND
/// instead of OR, or an adjacent difference instead of a central one, should not have
/// to fork the kernel.
///
/// ---------------------------------------------------------------------------
/// WHY `SrcT` AND NOT JUST `uint8_t` (ARCHITECTURE 7.8.1)
///
/// **This operation is why that section exists.** "Downconvert 12->8 yourself, then
/// call binCV" is `v >> 4`, which truncates the OPERANDS before they are differenced:
/// a genuine 12-bit gradient of 15 counts becomes **exactly zero**, and the edge is
/// gone before binCV sees the pixel. Low contrast -- indoors, at night, on untextured
/// walls -- is where a VIO frontend needs every edge it can get, so the workaround
/// fails hardest exactly where it matters.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND THE NAME IS NOT OPENCV'S
///
/// `cv::Sobel` + `cv::threshold` is a DIFFERENT computation: a 3x3 separable kernel
/// with smoothing, against this operation's single-axis difference. CLAUDE.md forbids
/// borrowing an OpenCV name for an operation that does not match it.
///
/// The border rule IS OpenCV's, though: `cv::filter2D` defaults to
/// `BORDER_REFLECT_101`, so index -1 reads index 1 and index `w` reads index `w-2`.
/// That is what makes the shipped defaults reproduce the reference exactly rather
/// than approximately.

#include <cstddef>
#include <cstdint>

#include "../binMat.hpp"
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief How the two axes' results are combined. `Or` is the reference's.
enum class EdgeCombine { Or, And };

/// @brief How a gradient is compared with the threshold. `Ge` is the reference's.
enum class EdgeRelation { Ge, Gt };

/// @brief Which pixels are differenced. `Wide` is the reference's.
enum class EdgeSpatial {
    Wide,      ///< `|v[x+1] - v[x-1]|` -- the central difference, spanning two pixels
    Forward,   ///< `|v[x+1] - v[x]|`
    Backward,  ///< `|v[x] - v[x-1]|`
};

namespace impl {

/// @brief `BORDER_REFLECT_101`: index -1 reads 1, index n reads n-2. **INTERNAL.**
/// @note OpenCV's `filter2D` default, and therefore the reference's. A one-pixel
///       extent reflects to itself.
inline size_t reflect101Edge(long long i, size_t n) {
    if (n <= 1) return 0;
    const long long last = static_cast<long long>(n) - 1;
    while (i < 0 || i > last) {
        if (i < 0) i = -i;
        if (i > last) i = 2 * last - i;
    }
    return static_cast<size_t>(i);
}

/// @brief One axis' absolute difference at `(y, x)`. **INTERNAL.**
/// @tparam Horizontal True for the x axis, false for the y axis.
template <EdgeSpatial S, bool Horizontal, typename SrcT>
inline long long axisDiff(const SrcT* src, size_t width, size_t height, size_t stride,
                          size_t y, size_t x) {
    long long aI, bI;   // the two indices along the axis, before reflection
    const long long here = static_cast<long long>(Horizontal ? x : y);
    if constexpr (S == EdgeSpatial::Wide) {
        aI = here + 1;
        bI = here - 1;
    } else if constexpr (S == EdgeSpatial::Forward) {
        aI = here + 1;
        bI = here;
    } else {
        aI = here;
        bI = here - 1;
    }
    const size_t n = Horizontal ? width : height;
    const size_t a = reflect101Edge(aI, n);
    const size_t b = reflect101Edge(bI, n);
    const long long va =
        static_cast<long long>(Horizontal ? src[y * stride + a] : src[a * stride + x]);
    const long long vb =
        static_cast<long long>(Horizontal ? src[y * stride + b] : src[b * stride + x]);
    const long long d = va - vb;
    return d < 0 ? -d : d;
}

template <EdgeRelation R>
inline bool passesEdge(long long d, long long t) {
    if constexpr (R == EdgeRelation::Ge) return d >= t;
    return d > t;
}

/// @brief Is `(y, x)` an edge? **INTERNAL** -- the whole predicate, in one place.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT>
inline bool isEdge(const SrcT* src, size_t width, size_t height, size_t stride, size_t y,
                   size_t x, long long t) {
    const bool px =
        passesEdge<R>(axisDiff<S, true, SrcT>(src, width, height, stride, y, x), t);
    // SHORT-CIRCUITING IS CORRECT AND WORTH HAVING: on a sparse edge map most pixels
    // fail both tests, so `And` skips the vertical difference for nearly all of them.
    if constexpr (C == EdgeCombine::Or) {
        if (px) return true;
    } else {
        if (!px) return false;
    }
    return passesEdge<R>(axisDiff<S, false, SrcT>(src, width, height, stride, y, x), t);
}

} // namespace impl

/// @brief Gradient-magnitude edge extraction straight into bits. **API TIER 3.**
///
/// @param src Row-major, `srcStride` ELEMENTS between rows (not bytes).
/// @param t Threshold, in source units.
///
/// **THE DEFAULTS ARE THE REFERENCE.** `edgeThreshold(src, w, h, stride, dst, 17)`
/// with no template arguments is `rl_fast_edge_filter_wide(img, 17)`.
///
/// @note 8-bit-in, 1-bit-out, and **the byte never exists**: the comparison yields a
///       boolean per pixel which goes straight into a word. Computing an 8-bit edge
///       image and packing it afterwards would be two passes and an intermediate.
/// @note `dst`'s padding bits are zero on return.
/// @note Never allocates and never throws.
template <EdgeCombine C = EdgeCombine::Or, EdgeRelation R = EdgeRelation::Ge,
          EdgeSpatial S = EdgeSpatial::Wide, typename SrcT = uint8_t,
          typename WordType = uint32_t>
inline void edgeThreshold(const SrcT* src, size_t width, size_t height, size_t srcStride,
                          BinMatView<WordType> dst, SrcT t) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "edgeThreshold: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "edgeThreshold: dst's stride must cover a whole row");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "edgeThreshold: a non-empty image needs non-null pointers");

    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    const long long tt = static_cast<long long>(t);
    for (size_t y = 0; y < height; ++y) {
        WordType* row = dst.row(y);
        for (size_t x = 0; x < width; x += kBits) {
            const size_t n = (width - x < kBits) ? (width - x) : kBits;
            WordType acc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (impl::isEdge<C, R, S, SrcT>(src, width, height, srcStride, y, x + i, tt))
                    acc = static_cast<WordType>(acc | (WordType{1} << i));
            }
            // STORED, not OR-ed: the bits past `width` are never set, which is the
            // padding invariant with no mask needed.
            row[x / kBits] = acc;
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
