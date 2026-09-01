#pragma once

/// @file blockMatch.hpp
/// @brief Route (a): fully bit-parallel keypoint tracking by Hamming block
///        matching over bit-packed frames (T4.2 / E-6). **API TIER 3** -- no
///        OpenCV equivalent, so it borrows no OpenCV name.
///
/// [ARCHITECTURE 7.9](../../../../docs/ARCHITECTURE.md) names TWO routes for tracking on
/// binary frames and only route (b) had ever been built. This is route (a).
///
/// **SCOPE, BECAUSE [CLAUDE.md](../../../../CLAUDE.md) PUTS TEMPLATE MATCHING OUT OF
/// SCOPE.** It does, as an *operation*: `cv::matchTemplate` is deliberately absent
/// and stays absent. This is not that. It is an internal tracker search named in
/// 7.9 and scheduled as T4.2 since the roadmap was written, and it exposes no
/// template-matching API -- the entry point takes keypoints and returns tracked
/// keypoints, exactly as `calcOpticalFlowPyrLK` does.
///
/// ===========================================================================
/// WHAT IT IS, AND WHAT IT COSTS AND SAVES AGAINST ROUTE (b)
/// ===========================================================================
/// For each keypoint, take the previous frame's window and slide it over the next
/// frame at INTEGER displacements, scoring each by Hamming distance. On bit-packed
/// frames that score is `popcount((prev ^ next) & mask)` per word -- one XOR and
/// one popcount, no derivative, no covariance, no solve, **and not one
/// floating-point operation inside the search**.
///
///  * **IT NEEDS NO DERIVATIVE, AND THAT IS A FOOTPRINT RESULT, NOT A DETAIL.**
///    Route (b) carries two `SignedQuantMat` ladders -- `2(N+1)` planes per level
///    on top of the two frames. Route (a) carries the two frames and nothing else.
///  * **Its cost is `O(R²)` per level where route (b)'s is `O(iterations)`.** The
///    search radius is the whole cost story and is swept in X-26.
///  * **Its accuracy floor is derivable and is stated before any measurement**
///    (X-26): a whole-pixel matcher returns `round(d)`, so on a translation with
///    fractional part `q` its per-axis error is exactly `min(q, 1-q)`, which over
///    `q` uniform is **0.2887 px per axis and 0.408 px over two**. `subPixel`
///    below is what addresses that, and it is the only part of this file that is
///    not integer arithmetic.
///  * **It is a ONE-BIT algorithm.** Hamming distance is defined on bits. Route
///    (b) does better on a `1/2/2/2` ladder than on `1/1/1/1`
///    ([D-23](../../../../docs/ARCHITECTURE.md)), and route (a) cannot enter that
///    comparison without an N-bit cost function. X-26 reports both the same-ladder
///    comparison (the algorithm question) and the best-ladder one (the practical
///    question) rather than picking whichever flatters route (a).
///
/// ===========================================================================
/// CONTRACTS -- the same ones ops/opticalFlow.hpp carries
/// ===========================================================================
///  * **Views, never containers** (D-5).
///  * **No heap, anywhere, and no scratch buffer.** The search keeps a running
///    minimum rather than a cost surface: a `(2R+1)²` surface would be scratch,
///    and the sub-pixel fit re-evaluates its four neighbours instead -- four window
///    scores against the `(2R+1)²` the search already paid.
///  * **Never throws.** A track that fails reports `status[i] == 0`.
///  * **Windows clip at the frame edge** and next-frame reads outside the level
///    **replicate**, through the same `impl::clipRegion` and
///    `impl::ReplicatedShiftedRow` route (b) uses -- so the two routes differ in
///    the SEARCH and in nothing else, which is what makes X-26 a comparison.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"
// Point2f, impl::displacedRow, impl::ReplicatedShiftedRow, impl::floorToLL --
// route (a) reuses route (b)'s tap machinery deliberately. Two implementations of
// "read the next frame displaced by (dx, dy) with a replicate border" is how two
// border behaviours happen, and the comparison would then not be one.
#include "opticalFlow.hpp"
#include "reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief Search and window parameters for `calcOpticalFlowBlockMatch`.
struct BlockMatchParams {
    int winWidth = 31;   ///< window width in pixels, > 2
    int winHeight = 31;  ///< window height in pixels, > 2

    /// @brief Half-extent of the integer search at EACH level, in pixels.
    /// @note The cost is `(2*searchRadius + 1)^2` window scores per point per
    ///       level, so this is the operation's whole cost story. Coarse-to-fine
    ///       makes it cheap in the useful sense: a radius of `R` over `L` levels
    ///       reaches a displacement of `R * (2^L - 1)`, so R = 2 over four levels
    ///       already covers +/- 30 px at 25 scores per level.
    int searchRadius = 2;

    /// @brief Refine the level-0 result to sub-pixel by fitting a parabola to the
    ///        Hamming cost along each axis.
    /// @note This is the ONLY floating-point arithmetic in the operation, it runs
    ///       once per point at level 0, and it exists because the integer floor is
    ///       0.408 px (see the header). Four extra window scores per point.
    bool subPixel = true;
};

/// @brief One pyramid level for route (a): both frames, and **no derivative**.
/// @note Deliberately not `LKLevel`. Route (a) never forms a gradient, and a level
///       type that carried four unused planes would misstate the footprint this
///       operation is partly being measured on.
template <typename WordType>
struct BlockMatchLevel {
    BinMatConstView<WordType> prev;
    BinMatConstView<WordType> next;

    size_t width() const { return prev.width; }
    size_t height() const { return prev.height; }
};

/// @brief Names two frames' level into a BlockMatchLevel. **API TIER 3.**
template <typename WordType>
inline BlockMatchLevel<WordType> blockMatchLevel(const BinMat<WordType>& prev,
                                                 const BinMat<WordType>& next) {
    BlockMatchLevel<WordType> lv;
    lv.prev = prev.constView();
    lv.next = next.constView();
    return lv;
}

namespace impl {

/// @brief Hamming distance between the previous window and the next frame
///        displaced by `(tapX, tapY)`. **INTERNAL.**
/// @return `sum over the window of popcount(prev ^ nextDisplaced)`.
/// @note One XOR and one popcount per word -- the cheapest per-word body in the
///       library, and the entire reason route (a) exists as a candidate.
template <typename WordType>
inline long long hammingAt(const BlockMatchLevel<WordType>& lv, const RegionWords<WordType>& r,
                           long long tapX, long long tapY) {
    long long cost = 0;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const WordType* ip = lv.prev.row(y);
        const ReplicatedShiftedRow<WordType> row =
            displacedRow<WordType>(lv.next, static_cast<long long>(y) + tapY, tapX);
        visitRowWords<WordType>(r, [&](size_t i, WordType mask) {
            const WordType diff = static_cast<WordType>((ip[i] ^ row.word(i)) & mask);
            cost += static_cast<long long>(popcountWord<WordType>(diff));
        });
    }
    return cost;
}

/// @brief The vertex of the parabola through `(-1, cm)`, `(0, c0)`, `(+1, cp)`.
/// @return An offset in `(-0.5, 0.5)`, or 0 when the three points are not convex.
/// @note The standard sub-pixel fit, and the guard is not decoration: on a binary
///       cost surface `cm == c0 == cp` happens on any flat plateau, and the
///       denominator is then exactly zero. A non-convex triple means the integer
///       minimum is not a minimum along this axis, and 0 is the honest answer.
inline double parabolicOffset(long long cm, long long c0, long long cp) {
    const double denom = static_cast<double>(cm - 2 * c0 + cp);
    if (!(denom > 0.0)) return 0.0;
    const double offset = 0.5 * static_cast<double>(cm - cp) / denom;
    if (offset > 0.5) return 0.5;
    if (offset < -0.5) return -0.5;
    return offset;
}

} // namespace impl

/// @brief Pyramidal keypoint tracking by integer Hamming block matching.
///        **API TIER 3.**
/// @param levels `levelCount` levels, **LEVEL 0 FIRST**, one bit per pixel.
/// @param prevPts `pointCount` keypoints in LEVEL-0 coordinates. Read only.
/// @param nextPts Out: tracked positions in LEVEL-0 coordinates, written for every
///        point whether or not it was tracked.
/// @param status Out: 1 if tracked, 0 if lost. Every entry is written.
/// @param pointCount Number of keypoints.
/// @param params Window, search radius and sub-pixel refinement.
///
/// @note Coarse to fine over the same ladder route (b) uses: the displacement found
///       at a level is DOUBLED into the next one and refined there, so a radius of
///       `R` over `L` levels reaches `R * (2^L - 1)` pixels.
/// @note **The previous window sits on the integer grid**, exactly as in route (b)
///       (deviation (i)), and for the same reason -- so that the two routes see the
///       same aperture and the comparison is of the search.
/// @note **No allocation, no scratch, no throw.**
template <typename WordType>
inline void calcOpticalFlowBlockMatch(const BlockMatchLevel<WordType>* levels, size_t levelCount,
                                      const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                                      size_t pointCount,
                                      const BlockMatchParams& params = BlockMatchParams()) {
    if (pointCount == 0) return;
    BINCV_ASSERT(prevPts != nullptr && nextPts != nullptr && status != nullptr,
                 "blockMatch: prevPts, nextPts and status must be non-null");
    BINCV_ASSERT(impl::byteRangesDisjoint(prevPts, pointCount * sizeof(Point2f), nextPts,
                                          pointCount * sizeof(Point2f)),
                 "blockMatch: nextPts must not overlap prevPts");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "blockMatch: the window must be more than 2 pixels on a side");
    BINCV_ASSERT(params.searchRadius >= 1, "blockMatch: searchRadius must be at least 1");

    for (size_t i = 0; i < pointCount; ++i) {
        status[i] = 1;
        nextPts[i] = prevPts[i];
    }
    if (levelCount == 0) {
        for (size_t i = 0; i < pointCount; ++i) status[i] = 0;
        return;
    }
    BINCV_ASSERT(levels != nullptr, "blockMatch: levels must be non-null");

    const int winW = params.winWidth;
    const int winH = params.winHeight;
    const float halfWinX = static_cast<float>(winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(winH - 1) * 0.5f;
    const long long radius = static_cast<long long>(params.searchRadius);

    // The same pyramid cap route (b) applies (deviation (vi)), through the same
    // rule -- a level at or below the window size gives every point nearly the
    // same window and therefore nearly the same answer.
    size_t usableLevels = 1;
    while (usableLevels < levelCount &&
           levels[usableLevels].width() > static_cast<size_t>(winW) &&
           levels[usableLevels].height() > static_cast<size_t>(winH)) {
        ++usableLevels;
    }

    for (size_t p = 0; p < pointCount; ++p) {
        // The displacement estimate, in the CURRENT level's pixels. Integer until
        // the sub-pixel fit at level 0, which is why it can be carried exactly.
        long long estX = 0;
        long long estY = 0;
        double subX = 0.0;
        double subY = 0.0;
        bool lost = false;

        for (size_t li = usableLevels; li-- > 0;) {
            const BlockMatchLevel<WordType>& lv = levels[li];
            BINCV_ASSERT(lv.prev.width == lv.next.width && lv.prev.height == lv.next.height,
                         "blockMatch: a level's two planes must share its dimensions");
            const float scale = 1.0f / static_cast<float>(1u << li);

            // Doubling happens on the way IN to each finer level, so that the
            // estimate is always in the level being searched.
            if (li + 1 != usableLevels) {
                estX *= 2;
                estY *= 2;
            }

            const float prevX = prevPts[p].x * scale - halfWinX;
            const float prevY = prevPts[p].y * scale - halfWinY;
            const long long anchorX = impl::floorToLL(prevX);
            const long long anchorY = impl::floorToLL(prevY);
            const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), winW, winH);
            const impl::RegionWords<WordType> region =
                impl::clipRegion<WordType>(lv.width(), lv.height(), window);
            if (region.isEmpty) {
                lost = true;
                break;
            }

            // THE SEARCH. A running minimum, not a cost surface -- see CONTRACTS.
            // Ties keep the FIRST candidate in scan order, and the scan is centred,
            // so a flat plateau resolves to its top-left rather than drifting with
            // the previous estimate.
            long long bestCost = -1;
            long long bestDx = 0;
            long long bestDy = 0;
            for (long long dy = -radius; dy <= radius; ++dy) {
                for (long long dx = -radius; dx <= radius; ++dx) {
                    const long long cost =
                        impl::hammingAt<WordType>(lv, region, estX + dx, estY + dy);
                    if (bestCost < 0 || cost < bestCost) {
                        bestCost = cost;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
            }
            estX += bestDx;
            estY += bestDy;

            // SUB-PIXEL, level 0 only, four extra window scores. The integer floor
            // is 0.408 px and this is the only thing that addresses it.
            if (li == 0 && params.subPixel) {
                const long long c0 = impl::hammingAt<WordType>(lv, region, estX, estY);
                const long long cxm = impl::hammingAt<WordType>(lv, region, estX - 1, estY);
                const long long cxp = impl::hammingAt<WordType>(lv, region, estX + 1, estY);
                const long long cym = impl::hammingAt<WordType>(lv, region, estX, estY - 1);
                const long long cyp = impl::hammingAt<WordType>(lv, region, estX, estY + 1);
                subX = impl::parabolicOffset(cxm, c0, cxp);
                subY = impl::parabolicOffset(cym, c0, cyp);
            }
        }

        if (lost) {
            status[p] = 0;
            continue;
        }
        nextPts[p].x = prevPts[p].x + static_cast<float>(static_cast<double>(estX) + subX);
        nextPts[p].y = prevPts[p].y + static_cast<float>(static_cast<double>(estY) + subY);
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
