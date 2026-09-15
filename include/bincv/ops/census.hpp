#pragma once

/// @file census.hpp
/// @brief The census transform: a wide image into K comparison bit-planes.
/// **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE OPERATION WHERE THE ALGORITHM AND THE REPRESENTATION COINCIDE
///
/// The census transform (Zabih & Woodfill, 1994) replaces each pixel with a bit
/// string: bit k is the comparison of the pixel against its k-th neighbour. A
/// census image therefore IS binCV's native object -- K one-bit planes -- and the
/// dense-stereo cost built on it is `popcount(censusL ^ censusR-shifted)`, XOR
/// and bulk counts over whole words. Nothing is packed after the fact; the
/// transform WRITES bits.
///
/// This is the first piece of the dense-disparity pipeline, and independently
/// useful: census planes are illumination-monotonic features (any monotone
/// remap of the pixels leaves every comparison unchanged), which is the property
/// stereo costs borrow it for.
///
/// ---------------------------------------------------------------------------
/// SHAPE
///
/// * **Wide in, bits out**, like the sensor stage and the descriptor family: the
///   comparisons carry information exactly because the input still has gray
///   levels.
/// * The output is an ARRAY OF PLANE VIEWS, one per offset, not a `QuantMat`:
///   the planes are independent comparisons, not digits of a number, and K (24
///   at 5x5) exceeds the N <= 8 a QuantMat carries. The same array-of-views
///   contract `keypointOrientation` uses.
/// * The convention is `bit = I(p + offset) > I(p)`, fixed and documented;
///   either convention works so long as both images of a pair use one.
/// * A comparison whose neighbour falls outside the frame writes 0 -- the rim of
///   a disparity map is invalid whatever the fill, and 0 keeps the planes'
///   padding invariant trivially.
/// * Tier 3: OpenCV's mainline has no census (one lives in opencv_contrib's
///   stereo module); this borrows no name and answers to a per-pixel reference.
///
/// The FULL-FRAME spelling here is the convenience form -- 24 planes of 752x480
/// at uint32 are ~1.08 MB. The dense pipeline streams a band of rows instead;
/// the memory rule recorded for it makes the band, not the frame, the resident
/// object.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One census comparison offset, relative to the pixel being written.
struct CensusOffset {
    int8_t dx, dy;
};

/// @brief A census neighbourhood: `K` offsets, none of them (0, 0).
template <size_t K>
struct CensusPattern {
    CensusOffset at[K];
};

/// @brief The 8-neighbour census (3x3 minus center), raster order. **API TIER 3.**
inline constexpr CensusPattern<8> kCensus3x3 = {{{-1, -1},
                                                 {0, -1},
                                                 {1, -1},
                                                 {-1, 0},
                                                 {1, 0},
                                                 {-1, 1},
                                                 {0, 1},
                                                 {1, 1}}};

/// @brief The 24-comparison census (5x5 minus center), raster order -- the
/// neighbourhood the dense-stereo design is written against. **API TIER 3.**
inline constexpr CensusPattern<24> kCensus5x5 = {
    {{-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
     {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
     {-2, 0},  {-1, 0},          {1, 0},  {2, 0},
     {-2, 1},  {-1, 1},  {0, 1}, {1, 1},  {2, 1},
     {-2, 2},  {-1, 2},  {0, 2}, {1, 2},  {2, 2}}};

/// @brief Census transform: plane `k` of `planes` gets
/// `I(p + pattern.at[k]) > I(p)` at every pixel `p`. **API TIER 3.**
/// @param img Row-major, `stride` ELEMENTS between rows, like every wide input.
/// @param planes `K` caller-owned plane views, each `width x height`. Every word
/// of every plane is written; padding bits end zero.
/// @note Never allocates, never throws. The per-pixel loop is the shipped v1 --
/// correct, and priced by its benchmark arm from birth; the word-parallel
/// restructuring is the dense pipeline's optimization to claim once a
/// caller's share is known.
namespace impl {

/// @brief One plane-row of the census: the comparisons of image row `y` against
/// its `(dx, dy)` neighbours, packed into `dst`. **INTERNAL.**
/// @note THE one definition of the census row, shared by the full-frame
/// transform and the dense-disparity pipeline's streaming band -- a second
/// copy is how the two would silently diverge. Writes every word of the
/// row; padding bits end zero.
template <typename SrcT, typename WordType>
inline void censusRow(const SrcT* img, size_t width, size_t height, size_t stride,
                      long long dx, long long dy, size_t y, WordType* dst) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    const size_t words = minRowWords<WordType>(width);
    // Columns whose neighbour stays inside the row: [xLo, xHi).
    const long long xLo = dx < 0 ? -dx : 0;
    const long long xHi = static_cast<long long>(width) - (dx > 0 ? dx : 0);
    const long long yn = static_cast<long long>(y) + dy;
    if (yn < 0 || yn >= static_cast<long long>(height) || xLo >= xHi) {
        for (size_t w = 0; w < words; ++w) dst[w] = 0;
        return;
    }
    const SrcT* rowC = img + y * stride;
    const SrcT* rowN = img + static_cast<size_t>(yn) * stride;
    for (size_t w = 0; w < words; ++w) {
        WordType acc = 0;
        const size_t base = w * kBits;
        const size_t xEnd = base + kBits < width ? base + kBits : width;
        for (size_t x = base; x < xEnd; ++x) {
            const long long xs = static_cast<long long>(x);
            if (xs < xLo || xs >= xHi) continue;
            acc = static_cast<WordType>(
                acc | (static_cast<WordType>(rowN[xs + dx] > rowC[x]) << (x - base)));
        }
        dst[w] = acc;
    }
}

} // namespace impl

template <size_t K, typename SrcT, typename WordType>
inline void censusTransform(const SrcT* img, size_t width, size_t height, size_t stride,
                            const CensusPattern<K>& pattern,
                            const BinMatView<WordType>* planes) {
    static_assert(K >= 1, "censusTransform: an empty pattern transforms nothing");
    BINCV_ASSERT(img != nullptr && planes != nullptr, "censusTransform: null argument");
    BINCV_ASSERT(stride >= width, "censusTransform: stride must cover a row");
    if (width == 0 || height == 0) return;

    for (size_t k = 0; k < K; ++k) {
        BINCV_ASSERT(planes[k].width == width && planes[k].height == height,
                     "censusTransform: plane extent must match the image");
        BINCV_ASSERT(impl::strideCoversARow<WordType>(planes[k].width, planes[k].height,
                                                      planes[k].stride),
                     "censusTransform: a plane's stride must cover a whole row");
        BINCV_ASSERT(pattern.at[k].dx != 0 || pattern.at[k].dy != 0,
                     "censusTransform: a (0, 0) offset compares a pixel with itself"
                     " and is always 0");
        BinMatView<WordType> plane = planes[k];   // a copy: views are three words,
                                                  // and a const array element's
                                                  // row() would be const too
        for (size_t y = 0; y < height; ++y) {
            impl::censusRow<SrcT, WordType>(img, width, height, stride, pattern.at[k].dx,
                                            pattern.at[k].dy, y, plane.row(y));
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
