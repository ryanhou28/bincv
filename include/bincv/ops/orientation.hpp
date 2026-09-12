#pragma once

/// @file orientation.hpp
/// @brief Keypoint orientation from the intensity centroid. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE FIRST HALF OF ORIENTED DESCRIPTORS
///
/// BRIEF's known failure mode is rotation -- the ORB paper measures matching
/// falling off a cliff past about 10 degrees -- and ORB's fix is to give every
/// keypoint an orientation and steer the sampling pattern by it. The orientation
/// is the **intensity centroid**: over a disc around the keypoint,
///
///     m10 = sum of x * I(x, y),   m01 = sum of y * I(x, y),
///     theta = atan2(m01, m10)
///
/// This file computes theta; [ops/descriptor.hpp](descriptor.hpp) steers the
/// pattern by it. They are separate files because they read different things:
/// the descriptor samples pairs, this reduces a whole disc.
///
/// ---------------------------------------------------------------------------
/// TWO SPELLINGS, BECAUSE THE PIPELINE HAS TWO INPUTS
///
/// * **Wide** (`SrcT` pixels) -- the spelling that pairs with `computeBrief`,
///   which reads the wide frame for the same reason (a comparison between two
///   one-bit pixels carries almost nothing; a centroid over them carries plenty,
///   see below). A plain weighted sum.
/// * **Bit-plane** -- for a pipeline that detects on the packed frame, where
///   `detectFast(BinMatConstView, ...)` already runs. Here the moments are
///   population counts, which is the binCV-native shape:
///
///       m01 row term: rowCount * dy               -- one popcount per row
///       m10 row term: sum over bits b of
///                     2^b * popcount(seg & M_b)   -- six masked popcounts
///
///   where `M_b` masks the positions whose bit `b` is set. Six masked counts
///   replace a per-pixel multiply-accumulate at up to 63 pixels per row. An
///   N-bit image weights plane p's counts by 2^p -- the same decomposition every
///   reduction in this library already uses.
///
/// The two spellings compute THE SAME integer moments over THE SAME disc, and
/// tests/test_orientation.cpp holds them to exact agreement -- same numbers,
/// not close numbers.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND THE PER-WORD POPCOUNTS ARE INTERNAL
///
/// OpenCV has no standalone orientation call -- it lives inside `cv::ORB` -- so
/// this takes no OpenCV name. The bit-plane path counts single 64-bit segments,
/// which is exactly the per-word popcount the public API refuses to offer; it
/// stays refused. This kernel amortizes the pattern over a whole disc traversal
/// and keeps the helper internal, which is what the bulk-only rule is for.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"
// impl::popcountWord lives in reduce.hpp and deliberately NOT in kernel_util.hpp --
// see the note there on keeping the per-word count out of every kernel's reach.
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief Half-width of the disc's row at height `dy`: the largest `xm` with
/// `xm^2 + dy^2 <= radius^2`. **INTERNAL.**
/// @note The plain integer disc, not OpenCV's rounded-and-symmetrized `umax`
/// table -- Tier 3 owns its numerics, and this construction is exactly
/// reproducible from the inequality alone.
inline int discHalfWidth(int radius, int dy) {
    const int d2 = radius * radius - dy * dy;
    int xm = radius;
    while (xm * xm > d2) --xm;
    return xm;
}

/// @brief Up to 63 consecutive bits of a bit-plane row, as a uint64_t. **INTERNAL.**
/// @param row First word of the row.
/// @param firstBit Bit index of the first pixel wanted.
/// @param len Number of pixels, at most 63. The caller has already checked the
/// range lies inside the row's width, so every word touched exists.
template <typename WordType>
inline uint64_t rowBitsSegment(const WordType* row, size_t firstBit, unsigned len) {
    const size_t wb = bitsPerWord<WordType>();
    size_t word = firstBit / wb;
    const unsigned off = static_cast<unsigned>(firstBit % wb);
    uint64_t seg = static_cast<uint64_t>(row[word]) >> off;
    unsigned got = static_cast<unsigned>(wb) - off;
    while (got < len) {
        ++word;
        seg |= static_cast<uint64_t>(row[word]) << got;
        got += static_cast<unsigned>(wb);
    }
    return seg & ((uint64_t{1} << len) - 1u);
}

/// @brief `sum over set bits of their position`, positions 0..63. **INTERNAL.**
/// @note The six masks pick the positions whose position-bit `k` is set, so the
/// weighted sum decomposes into six masked popcounts -- the same move the
/// gradient covariance makes, applied to the position index itself.
inline long long bitPositionSum(uint64_t seg) {
    static constexpr uint64_t kMask[6] = {
        0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL};
    long long sum = 0;
    for (int k = 0; k < 6; ++k) {
        sum += static_cast<long long>(popcountWord<uint64_t>(seg & kMask[k])) << k;
    }
    return sum;
}

} // namespace impl

/// @brief Orientation of `count` keypoints on a WIDE image, from the intensity
/// centroid over a disc of `radius`. **API TIER 3.**
/// @param keypointsXY `count` (x, y) pairs, interleaved -- the same raw-array
/// contract as `computeBrief`, and for the same reason: an orientation
/// pass should not drag a point type in.
/// @param angle One float per keypoint, radians in (-pi, pi], `atan2(m01, m10)`.
/// A flat patch (both moments zero) reports 0 -- deterministic, and
/// as good as any other angle for a patch with no structure.
/// @param keep Optional: set to 0 for a keypoint whose bounding square falls
/// outside the image, exactly as `computeBrief` reports it. Its angle
/// is written as 0. The square, not the disc: the descriptor that
/// consumes this angle samples the square, so a keypoint this call
/// rejects is one the descriptor was going to reject anyway.
/// @param radius Disc radius in pixels, in [1, 31]. 15 pairs with the 31-pixel
/// descriptor patch.
/// @note Never allocates. Accumulators are `long long`: a 31x31 disc of
/// uint16_t pixels weighted by +/-15 peaks well inside 63 bits.
template <typename SrcT>
inline void keypointOrientation(const SrcT* img, size_t width, size_t height,
                                size_t stride, const float* keypointsXY, size_t count,
                                float* angle, uint8_t* keep = nullptr, int radius = 15) {
    if (count == 0) return;
    BINCV_ASSERT(img != nullptr && keypointsXY != nullptr && angle != nullptr,
                 "keypointOrientation: null argument");
    BINCV_ASSERT(radius >= 1 && radius <= 31, "keypointOrientation: radius out of [1, 31]");

    int xm[32];
    for (int dy = 0; dy <= radius; ++dy) xm[dy] = impl::discHalfWidth(radius, dy);

    for (size_t k = 0; k < count; ++k) {
        const long long cx = static_cast<long long>(keypointsXY[2 * k]);
        const long long cy = static_cast<long long>(keypointsXY[2 * k + 1]);
        const bool inside = cx - radius >= 0 && cy - radius >= 0 &&
                            cx + radius < static_cast<long long>(width) &&
                            cy + radius < static_cast<long long>(height);
        if (!inside) {
            angle[k] = 0.0f;
            if (keep != nullptr) keep[k] = uint8_t{0};
            continue;
        }
        const SrcT* center = img + static_cast<size_t>(cy) * stride + static_cast<size_t>(cx);
        long long m10 = 0, m01 = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            const int h2 = xm[dy < 0 ? -dy : dy];
            const SrcT* row = center + static_cast<long long>(dy) * static_cast<long long>(stride);
            long long rowSum = 0, rowXSum = 0;
            for (int dx = -h2; dx <= h2; ++dx) {
                const long long v = static_cast<long long>(row[dx]);
                rowSum += v;
                rowXSum += dx * v;
            }
            m10 += rowXSum;
            m01 += dy * rowSum;
        }
        angle[k] = (m10 == 0 && m01 == 0)
                       ? 0.0f
                       : std::atan2(static_cast<float>(m01), static_cast<float>(m10));
        if (keep != nullptr) keep[k] = uint8_t{1};
    }
}

/// @brief The same orientation on a BIT-PLANE image: `planeCount` planes, plane
/// `p` weighted by `2^p`. **API TIER 3.**
/// @param planes One view per bit-plane, all the same extent. A binary image is
/// `planeCount == 1`; an N-bit `QuantMat` names its planes into this
/// array (`constPlane(0) .. constPlane(N-1)`).
/// @note The moments are EXACTLY the wide spelling's on the equivalent pixel
/// values -- the masked-popcount decomposition is an implementation, not an
/// approximation, and the test holds the two to integer equality.
template <typename WordType>
inline void keypointOrientation(const BinMatConstView<WordType>* planes, size_t planeCount,
                                const float* keypointsXY, size_t count, float* angle,
                                uint8_t* keep = nullptr, int radius = 15) {
    if (count == 0) return;
    BINCV_ASSERT(planes != nullptr && keypointsXY != nullptr && angle != nullptr,
                 "keypointOrientation: null argument");
    BINCV_ASSERT(planeCount >= 1 && planeCount <= 32,
                 "keypointOrientation: planeCount out of [1, 32]");
    BINCV_ASSERT(radius >= 1 && radius <= 31, "keypointOrientation: radius out of [1, 31]");
    const size_t width = planes[0].width, height = planes[0].height;
    for (size_t p = 1; p < planeCount; ++p) {
        BINCV_ASSERT(planes[p].width == width && planes[p].height == height,
                     "keypointOrientation: planes disagree on extent");
    }

    int xm[32];
    for (int dy = 0; dy <= radius; ++dy) xm[dy] = impl::discHalfWidth(radius, dy);

    for (size_t k = 0; k < count; ++k) {
        const long long cx = static_cast<long long>(keypointsXY[2 * k]);
        const long long cy = static_cast<long long>(keypointsXY[2 * k + 1]);
        const bool inside = cx - radius >= 0 && cy - radius >= 0 &&
                            cx + radius < static_cast<long long>(width) &&
                            cy + radius < static_cast<long long>(height);
        if (!inside) {
            angle[k] = 0.0f;
            if (keep != nullptr) keep[k] = uint8_t{0};
            continue;
        }
        long long m10 = 0, m01 = 0;
        for (size_t p = 0; p < planeCount; ++p) {
            const long long wp = 1LL << p;
            long long pm10 = 0, pm01 = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const int h2 = xm[dy < 0 ? -dy : dy];
                const unsigned len = static_cast<unsigned>(2 * h2 + 1);
                const WordType* row =
                    planes[p].ptr +
                    static_cast<size_t>(cy + dy) * planes[p].stride;
                const uint64_t seg = impl::rowBitsSegment<WordType>(
                    row, static_cast<size_t>(cx - h2), len);
                const long long cnt =
                    static_cast<long long>(impl::popcountWord<uint64_t>(seg));
                // Positions run 0..2*h2 with position 0 at dx == -h2, so the
                // weighted x-sum re-centers by subtracting h2 per set bit.
                pm10 += impl::bitPositionSum(seg) - h2 * cnt;
                pm01 += dy * cnt;
            }
            m10 += wp * pm10;
            m01 += wp * pm01;
        }
        angle[k] = (m10 == 0 && m01 == 0)
                       ? 0.0f
                       : std::atan2(static_cast<float>(m01), static_cast<float>(m10));
        if (keep != nullptr) keep[k] = uint8_t{1};
    }
}

/// @brief The 1-bit convenience spelling: a single plane, no array to build.
/// **API TIER 3.**
template <typename WordType>
inline void keypointOrientation(const BinMatConstView<WordType>& img,
                                const float* keypointsXY, size_t count, float* angle,
                                uint8_t* keep = nullptr, int radius = 15) {
    keypointOrientation<WordType>(&img, 1, keypointsXY, count, angle, keep, radius);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
