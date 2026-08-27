#pragma once

/// @file medianWide.hpp
/// @brief The reference pipeline's median filter, on a WIDE (8- or 16-bit) image,
///        with a caller-chosen neighbourhood. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS EXISTS ALONGSIDE ops/denoise.hpp
///
/// [ops/denoise.hpp](denoise.hpp) implements the same neighbourhood for BINARY
/// input, where median collapses to `maj3` -- one expression, 32 pixels per word.
/// That is the right kernel and it is not going anywhere.
///
/// But the reference filters the **grayscale** image, BEFORE binarisation:
/// `SEALProcessor.cpp` runs `three_pix_median_filter(img)` and only then
/// `rl_fast_edge_filter_wide(img, t)`. A binary-only median cannot sit where the
/// reference puts it, so a frontend that wanted the reference's pipeline had to
/// borrow OpenCV for this one step.
///
/// ---------------------------------------------------------------------------
/// THE NEIGHBOURHOOD IS THE CALLER'S, AND THE REFERENCE HAS TWO OF THEM
///
/// `SEAL/src/temporal_processing/denoise.cpp` carries `three_pix_median_filter` --
/// the asymmetric L, `p1` above / `p2` centre / `p3` right -- **and**
/// `five_pix_median_filter`, the plus. Both ship here as named constants, and an
/// arbitrary offset set is a template argument rather than a fork.
///
/// This is emphatically NOT `cv::medianBlur`, whose neighbourhood is a square and
/// whose border is replicated. Tier 3, and the name is not borrowed
/// ([CLAUDE.md](../../../CLAUDE.md)).
///
/// ---------------------------------------------------------------------------
/// THE BORDER IS ZERO FILL, AND THAT IS THE REFERENCE'S, NOT A CHOICE
///
/// The reference builds its shifted neighbours as `cv::Mat::zeros` and copies the
/// overlapping region in, so the row and column that fall off the edge KEEP THE
/// ZEROS. A pixel at the top row therefore takes its median against a 0, not
/// against a replicated or reflected neighbour. ops/denoise.hpp records the same
/// rule for the same reason.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One sample position, relative to the pixel being written.
struct MedianOffset {
    int dy;
    int dx;
};

/// @brief A neighbourhood: `K` offsets, `K` odd so the median is a single element.
template <size_t K>
struct MedianPattern {
    static_assert(K % 2 == 1, "a median needs an odd number of samples");
    MedianOffset offset[K];
};

/// @brief The reference's `three_pix_median_filter`: above, centre, right.
/// @note An asymmetric **L**, not a line and not a square. It is chosen for what it
///       costs in race logic, not for isotropy, which is why no OpenCV kernel
///       matches it.
inline constexpr MedianPattern<3> kMedianReferenceL{{{-1, 0}, {0, 0}, {0, 1}}};

/// @brief The reference's `five_pix_median_filter`: the plus.
inline constexpr MedianPattern<5> kMedianReferencePlus{
    {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}}};

/// @brief Median filter over a caller-chosen neighbourhood. **API TIER 3.**
/// @param src,dst Row-major, strides in ELEMENTS. **`src` and `dst` must not alias**
///        -- every output reads neighbours that a partial in-place write would have
///        already changed.
/// @note Out-of-range samples read as **zero**; see the border note above.
/// @note Never allocates and never throws.
template <size_t K, typename SrcT>
inline void medianWide(const SrcT* src, size_t width, size_t height, size_t srcStride,
                       SrcT* dst, size_t dstStride, const MedianPattern<K>& pattern) {
    BINCV_ASSERT(src != dst, "medianWide: src and dst must not alias");
    if (width == 0 || height == 0) return;
    BINCV_ASSERT(src != nullptr && dst != nullptr,
                 "medianWide: a non-empty image needs non-null pointers");

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            SrcT v[K];
            for (size_t k = 0; k < K; ++k) {
                const long long sy = static_cast<long long>(y) + pattern.offset[k].dy;
                const long long sx = static_cast<long long>(x) + pattern.offset[k].dx;
                const bool inside = sy >= 0 && sx >= 0 &&
                                    sy < static_cast<long long>(height) &&
                                    sx < static_cast<long long>(width);
                v[k] = inside ? src[static_cast<size_t>(sy) * srcStride +
                                    static_cast<size_t>(sx)]
                              : SrcT{0};
            }
            // Insertion sort. `K` is 3 or 5 in every shipped pattern and odd by
            // static_assert, so this is a handful of compares fully unrolled -- a
            // sorting network would be the same instructions with more source.
            for (size_t i = 1; i < K; ++i) {
                SrcT key = v[i];
                size_t j = i;
                while (j > 0 && v[j - 1] > key) {
                    v[j] = v[j - 1];
                    --j;
                }
                v[j] = key;
            }
            dst[y * dstStride + x] = v[K / 2];
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
