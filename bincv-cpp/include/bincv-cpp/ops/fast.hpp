#pragma once

/// @file fast.hpp
/// @brief FAST corner detection on a wide image. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// WHY binCV HAS THIS WHEN IT ALREADY DETECTS CORNERS
///
/// binCV detects with Shi-Tomasi -- `cornerMinEigenVal` / `goodFeaturesToTrack` --
/// which is what LK wants, because it scores exactly the thing LK needs: a window
/// whose gradient covariance is well conditioned.
///
/// FAST is what the ORB-SLAM family detects with, and it is here because
/// [ops/descriptor.hpp](descriptor.hpp) is. A descriptor pipeline wants corners that
/// are repeatable under rotation and cheap to find, and Shi-Tomasi is neither of
/// those things first. **They ship together or not at all** -- FAST without
/// descriptors would be a detector nobody asked for.
///
/// ---------------------------------------------------------------------------
/// TIER 2, NOT TIER 1, AND THE DIFFERENCE IS THE SCORE
///
/// The DETECTION rule is `cv::FAST`'s exactly: `arcLength` contiguous pixels of the
/// 16-pixel Bresenham ring all brighter than `centre + t`, or all darker than
/// `centre - t`. Same ring, same order, same contiguity-wraps-around rule.
///
/// The SCORE is not. OpenCV scores a corner by binary-searching the largest threshold
/// at which it survives; this sums how far the qualifying arc exceeds the threshold.
/// **Both order corners sensibly and they do not agree**, so non-maximum suppression
/// over them can keep different points. Tier 2 says exactly that: same role, same call
/// shape, different numerics.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One detected corner.
struct FastCorner {
    int x;
    int y;
    long long score;   ///< see the tier note -- NOT `cv::FAST`'s score
};

namespace impl {

/// @brief The 16-pixel Bresenham ring of radius 3, clockwise from straight up.
/// @note The order matters: contiguity is defined ALONG this ring, so a different
///       winding would accept different corners. This is `cv::FAST`'s order.
inline constexpr int kFastRingX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
inline constexpr int kFastRingY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

} // namespace impl

/// @brief Detects FAST corners. **API TIER 2** -- see the tier note.
///
/// @param arcLength Contiguous ring pixels required. 9 is `cv::FAST`'s default and
///        the one ORB uses.
/// @param out,capacity Caller-provided; **no allocation happens here**
///        ([CLAUDE.md](../../../CLAUDE.md)).
/// @param truncated Set when more corners were found than `capacity` held. **A
///        silently truncated detection looks like a sparse image**, which is the kind
///        of failure that gets diagnosed as a tuning problem for weeks.
/// @return How many corners were written.
///
/// @note Pixels within 3 of a border are never candidates: the ring would fall
///       outside, and there is no sensible border rule for "is this a corner" --
///       reflecting the image would invent structure that is not there.
template <typename SrcT>
inline size_t detectFast(const SrcT* img, size_t width, size_t height, size_t stride,
                         long long threshold, FastCorner* out, size_t capacity,
                         bool* truncated = nullptr, int arcLength = 9) {
    BINCV_ASSERT(arcLength >= 1 && arcLength <= 16,
                 "detectFast: arcLength must be within the ring");
    if (truncated != nullptr) *truncated = false;
    if (width < 7 || height < 7 || capacity == 0) return 0;
    BINCV_ASSERT(img != nullptr && out != nullptr, "detectFast: null argument");

    // The largest count of compass points {0, 4, 8, 12} that EVERY window of
    // `arcLength` consecutive ring positions is guaranteed to contain. Computed
    // rather than tabulated so an unusual `arcLength` cannot silently get a bound
    // that belongs to a different one.
    int minCompass = 4;
    for (int s = 0; s < 16; ++s) {
        int c = 0;
        for (int k = 0; k < arcLength; ++k) {
            const int idx = (s + k) & 15;
            if ((idx & 3) == 0) ++c;
        }
        if (c < minCompass) minCompass = c;
    }

    size_t n = 0;
    for (size_t y = 3; y + 3 < height; ++y) {
        for (size_t x = 3; x + 3 < width; ++x) {
            const long long c = static_cast<long long>(img[y * stride + x]);
            const long long hi = c + threshold;
            const long long lo = c - threshold;

            long long ring[16];
            for (int k = 0; k < 16; ++k) {
                ring[k] = static_cast<long long>(
                    img[static_cast<size_t>(static_cast<long long>(y) + impl::kFastRingY[k]) *
                            stride +
                        static_cast<size_t>(static_cast<long long>(x) + impl::kFastRingX[k])]);
            }

            // A cheap reject first: without it every flat pixel walks the full arc
            // scan. THE BOUND MUST BE THE WORST CASE, NOT THE TYPICAL ONE -- an arc of
            // `arcLength` contains AT LEAST `minCompass` of the four compass points,
            // and rejecting on anything stricter throws away real corners. At
            // arcLength 9 that minimum is TWO, not three: the window 1..9 contains
            // only indices 4 and 8. Using three cost every corner of a filled square,
            // whose ring is bright for barely a quarter of its length.
            const int haveHi = (ring[0] > hi) + (ring[4] > hi) + (ring[8] > hi) + (ring[12] > hi);
            const int haveLo = (ring[0] < lo) + (ring[4] < lo) + (ring[8] < lo) + (ring[12] < lo);
            if (haveHi < minCompass && haveLo < minCompass) continue;

            // Contiguity WRAPS, so the scan runs to 16 + arcLength - 1.
            long long bestScore = 0;
            for (int sign = 0; sign < 2; ++sign) {
                int run = 0;
                for (int k = 0; k < 16 + arcLength - 1; ++k) {
                    const long long v = ring[k & 15];
                    const bool pass = (sign == 0) ? (v > hi) : (v < lo);
                    if (pass) {
                        ++run;
                        if (run >= arcLength) {
                            long long s = 0;
                            for (int j = 0; j < 16; ++j) {
                                const long long d = (sign == 0) ? (ring[j] - hi) : (lo - ring[j]);
                                if (d > 0) s += d;
                            }
                            if (s > bestScore) bestScore = s;
                            break;
                        }
                    } else {
                        run = 0;
                    }
                }
            }
            if (bestScore <= 0) continue;
            if (n >= capacity) {
                if (truncated != nullptr) *truncated = true;
                return n;
            }
            out[n].x = static_cast<int>(x);
            out[n].y = static_cast<int>(y);
            out[n].score = bestScore;
            ++n;
        }
    }
    return n;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
