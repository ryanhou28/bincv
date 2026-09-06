#pragma once

/// @file stereo.hpp
/// @brief Sparse rectified stereo matching: descriptor search along the epipolar
/// row, then Hamming window refinement to sub-pixel disparity. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHAT THIS IS
///
/// A stereo frontend turns a left keypoint into a depth by finding it in the
/// right image. On a RECTIFIED pair that search is one-dimensional -- same row,
/// disparity in a known range -- and both stages of the standard method are
/// operations this library is built from:
///
/// 1. **Coarse**: the left keypoint's descriptor against right-image candidates
///    in the same row band -- `popcount(a ^ b)`, the matcher ops/descriptor.hpp
///    already ships, with the row-and-range constraint it does not have.
/// 2. **Fine**: a small window slid along the row around the coarse hit, scored
///    by `popcount((L ^ R) & mask)` per word -- the exact inner loop
///    ops/blockMatch.hpp runs, pointed at one axis -- with the parabola fit that
///    file already carries for the sub-pixel offset.
///
/// The ORB-SLAM family does this shape with SAD on bytes. On packed frames the
/// window score is an XOR and a bulk count, which is the whole thesis.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND WHY THE STAGES ARE SEPARABLE
///
/// `cv::StereoBM` / `cv::StereoSGBM` are dense; a sparse per-keypoint rectified
/// search has no OpenCV equivalent and borrows no OpenCV name.
///
/// The two stages are separate entry points because they have separate callers:
/// a frontend with descriptors runs both; a caller with an initial disparity
/// from anywhere else (a projection of a map point, a previous frame) fills
/// `StereoMatch::disparity` itself and runs refinement alone.
///
/// ---------------------------------------------------------------------------
/// CONTRACTS -- ops/blockMatch.hpp's, unchanged
///
/// * Views, never containers; no heap, no scratch, no throw. The refinement
///   keeps a running minimum, never a cost surface.
/// * **It is a ONE-BIT refinement**, like the block matcher and for the same
///   reason: Hamming distance is defined on bits.
/// * Windows clip at the frame edge; right-frame reads outside the image
///   replicate -- through the same impl machinery, so the border behavior
///   cannot drift from the tracker's.
/// * Rectification is the CALLER's (it turns one wide image into another --
///   the input boundary). `rowTolerance` absorbs the residual error of a real
///   rectification in the DESCRIPTOR stage; the refinement stage trusts the
///   epipolar geometry and scans one axis only.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"
#include "blockMatch.hpp"   // BlockMatchLevel, impl::hammingAt, impl::parabolicOffset
#include "descriptor.hpp"   // hammingDistance
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Search and window parameters for the sparse rectified stereo matcher.
struct StereoMatchParams {
    /// @brief Disparity range, level-0 pixels, `0 <= min < max`. Disparity is
    /// `leftX - rightX`, so the range encodes "how close can a point be".
    int minDisparity = 0;
    int maxDisparity = 64;

    /// @brief Rows the DESCRIPTOR stage will look above and below the left
    /// keypoint's row. A perfect rectification needs 0; real ones do not,
    /// and the caller knows theirs.
    int rowTolerance = 2;

    /// @brief Largest descriptor Hamming distance the coarse stage accepts.
    /// @note An absolute bit count, so it scales with the descriptor length the
    /// caller chose; 100 is the ORB-SLAM family's high threshold at 256
    /// bits. A ratio test is deliberately NOT applied here: along an
    /// epipolar row the second-best candidate is often the true match's
    /// neighbour, and rejecting on that ratio throws away exactly the
    /// matches the refinement stage exists to sharpen.
    unsigned maxHamming = 100;

    /// @brief Refinement window, pixels, > 2 on a side.
    int winWidth = 11;
    int winHeight = 11;

    /// @brief Half-extent of the refinement scan around the coarse disparity.
    int refineRadius = 4;

    /// @brief Parabolic sub-pixel fit on the disparity axis, three window scores.
    /// The integer floor is the block matcher's 0.2887 px per axis; depth
    /// is 1/disparity, so the fit matters more here than anywhere.
    bool subPixel = true;
};

/// @brief One left keypoint's stereo result.
struct StereoMatch {
    float disparity = 0.0f;    ///< `leftX - rightX`, level-0 pixels
    unsigned distance = 0;     ///< descriptor Hamming of the accepted candidate
    size_t rightIndex = 0;     ///< index into the right keypoint set
    uint8_t valid = 0;         ///< 0: no candidate survived; the other fields are void
};

/// @brief COARSE stage: each left descriptor against the right keypoints in its
/// row band and disparity range. **API TIER 3.**
/// @param leftXY / rightXY (x, y) pairs, interleaved, level-0 pixels -- the same
/// raw-array contract as the descriptor family.
/// @param leftDesc / rightDesc `words` words per keypoint, `computeBrief`'s or
/// `computeBriefSteered`'s layout.
/// @param out One `StereoMatch` per LEFT keypoint, every entry written.
/// @note Brute force over the right set per left keypoint -- O(L * R) distance
/// tests, each gated by two float comparisons before any descriptor word is
/// read. At frontend counts (hundreds against hundreds) the gate leaves a
/// few candidates per keypoint; a row-bucketed index would need scratch,
/// and the no-scratch rule outranks a constant factor here until a profile
/// says otherwise.
template <typename WordType>
inline void stereoDescriptorMatch(const float* leftXY, size_t leftCount,
                                  const WordType* leftDesc, const float* rightXY,
                                  size_t rightCount, const WordType* rightDesc,
                                  size_t words, StereoMatch* out,
                                  const StereoMatchParams& params = StereoMatchParams()) {
    if (leftCount == 0) return;
    BINCV_ASSERT(leftXY != nullptr && leftDesc != nullptr && out != nullptr,
                 "stereoDescriptorMatch: null left argument");
    BINCV_ASSERT(rightCount == 0 || (rightXY != nullptr && rightDesc != nullptr),
                 "stereoDescriptorMatch: null right argument");
    BINCV_ASSERT(params.minDisparity >= 0 && params.maxDisparity > params.minDisparity,
                 "stereoDescriptorMatch: need 0 <= minDisparity < maxDisparity");
    BINCV_ASSERT(params.rowTolerance >= 0, "stereoDescriptorMatch: rowTolerance < 0");

    const float rowTol = static_cast<float>(params.rowTolerance);
    const float minD = static_cast<float>(params.minDisparity);
    const float maxD = static_cast<float>(params.maxDisparity);

    for (size_t i = 0; i < leftCount; ++i) {
        const float xL = leftXY[2 * i];
        const float yL = leftXY[2 * i + 1];
        unsigned best = 0xFFFFFFFFu;
        size_t bestIdx = 0;
        for (size_t j = 0; j < rightCount; ++j) {
            const float dy = rightXY[2 * j + 1] - yL;
            if (dy > rowTol || dy < -rowTol) continue;
            const float d = xL - rightXY[2 * j];
            if (d < minD || d > maxD) continue;
            const unsigned dist = hammingDistance<WordType>(leftDesc + i * words,
                                                            rightDesc + j * words, words);
            if (dist < best) {
                best = dist;
                bestIdx = j;
            }
        }
        StereoMatch m;
        if (best <= params.maxHamming) {
            m.disparity = xL - rightXY[2 * bestIdx];
            m.distance = best;
            m.rightIndex = bestIdx;
            m.valid = 1;
        }
        out[i] = m;
    }
}

/// @brief FINE stage: slide a window along the epipolar row around each valid
/// match's disparity, score by Hamming distance on the packed frames, and
/// refine to sub-pixel. **API TIER 3.**
/// @param left / right The RECTIFIED pair, one bit per pixel, same extent.
/// @param leftXY The left keypoints the windows anchor on.
/// @param inout One entry per left keypoint. On the way in, `valid` and
/// `disparity` are read -- from `stereoDescriptorMatch`, or filled by a
/// caller refining an initial disparity of their own. On the way out,
/// `disparity` is refined and `valid` reports whether a window could be
/// scored at all (a keypoint whose window misses the frame entirely
/// reports 0, never a clamped answer).
/// @note The scan is HORIZONTAL only: rectification promised the row, and this
/// stage holds it to that. The result is clamped to the params' disparity
/// range -- a refinement that wandered out of the range the caller stated
/// would be reporting a depth the caller already excluded.
template <typename WordType>
inline void stereoRefineDisparity(const BinMatConstView<WordType>& left,
                                  const BinMatConstView<WordType>& right,
                                  const float* leftXY, size_t count, StereoMatch* inout,
                                  const StereoMatchParams& params = StereoMatchParams()) {
    if (count == 0) return;
    BINCV_ASSERT(leftXY != nullptr && inout != nullptr,
                 "stereoRefineDisparity: null argument");
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "stereoRefineDisparity: the pair must share its extent");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "stereoRefineDisparity: the window must be more than 2 pixels on a side");
    BINCV_ASSERT(params.refineRadius >= 1, "stereoRefineDisparity: refineRadius < 1");

    // The block matcher's level type names the two frames; `prev` anchors the
    // window, `next` is read displaced -- here by MINUS the disparity, because
    // the right camera sees everything shifted left.
    BlockMatchLevel<WordType> lv;
    lv.prev = left;
    lv.next = right;

    const int winW = params.winWidth;
    const int winH = params.winHeight;
    const float halfWinX = static_cast<float>(winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(winH - 1) * 0.5f;
    const long long minD = static_cast<long long>(params.minDisparity);
    const long long maxD = static_cast<long long>(params.maxDisparity);

    for (size_t i = 0; i < count; ++i) {
        if (!inout[i].valid) continue;
        const long long anchorX = impl::floorToLL(leftXY[2 * i] - halfWinX);
        const long long anchorY = impl::floorToLL(leftXY[2 * i + 1] - halfWinY);
        const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), winW, winH);
        const impl::RegionWords<WordType> region =
            impl::clipRegion<WordType>(left.width, left.height, window);
        if (region.isEmpty) {
            inout[i].valid = 0;
            continue;
        }

        // Integer scan around the coarse disparity, clamped to the stated range.
        const long long d0 =
            impl::floorToLL(static_cast<float>(inout[i].disparity) + 0.5f);
        long long lo = d0 - params.refineRadius, hi = d0 + params.refineRadius;
        if (lo < minD) lo = minD;
        if (hi > maxD) hi = maxD;
        if (lo > hi) {
            inout[i].valid = 0;
            continue;
        }
        long long bestCost = -1, bestD = lo;
        for (long long d = lo; d <= hi; ++d) {
            const long long cost = impl::hammingAt<WordType>(lv, region, -d, 0);
            if (bestCost < 0 || cost < bestCost) {
                bestCost = cost;
                bestD = d;
            }
        }

        double sub = 0.0;
        if (params.subPixel) {
            // Neighbours may sit outside [lo, hi]; the fit wants the true local
            // shape, and one extra score per side is cheaper than a biased vertex.
            const long long cm = impl::hammingAt<WordType>(lv, region, -(bestD - 1), 0);
            const long long cp = impl::hammingAt<WordType>(lv, region, -(bestD + 1), 0);
            sub = impl::parabolicOffset(cm, bestCost, cp);
        }
        double refined = static_cast<double>(bestD) + sub;
        if (refined < static_cast<double>(minD)) refined = static_cast<double>(minD);
        if (refined > static_cast<double>(maxD)) refined = static_cast<double>(maxD);
        inout[i].disparity = static_cast<float>(refined);
    }
}

/// @brief Both stages: descriptor search, then window refinement. **API TIER 3.**
/// @note Exactly `stereoDescriptorMatch` into `stereoRefineDisparity`; it exists
/// so the common caller cannot get the hand-off wrong, not because the
/// composition adds anything.
template <typename WordType>
inline void stereoMatchRectified(const BinMatConstView<WordType>& left,
                                 const BinMatConstView<WordType>& right,
                                 const float* leftXY, size_t leftCount,
                                 const WordType* leftDesc, const float* rightXY,
                                 size_t rightCount, const WordType* rightDesc, size_t words,
                                 StereoMatch* out,
                                 const StereoMatchParams& params = StereoMatchParams()) {
    stereoDescriptorMatch<WordType>(leftXY, leftCount, leftDesc, rightXY, rightCount,
                                    rightDesc, words, out, params);
    stereoRefineDisparity<WordType>(left, right, leftXY, leftCount, out, params);
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
