#pragma once

/// @file medianEdgeStream.hpp
/// @brief The sensor stage -- `medianWide` then `edgeThreshold` -- over rows as they
/// arrive, holding a few rows instead of two frames. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS EXISTS
///
/// A camera delivers rows. The whole-frame calls take a frame, so a caller using
/// them holds the 8-bit input frame and the median's 8-bit output beside it: at
/// 752x480 that is 721,920 bytes, more than everything downstream of it in the
/// feature tracker put together, and on a microcontroller with 1 MB of RAM the
/// difference between the tracker fitting and not. ops/pack.hpp's `packRows` is
/// the streaming entry point for an operation that reads one row; this is the one
/// for the two that read their neighbours.
///
/// What it holds is the neighbourhood and nothing more: the input rows the median
/// pattern reaches (two for the reference's L), and the three median rows the edge
/// test reads. 5 x 752 = 3,760 bytes at the reference's shape.
///
/// ---------------------------------------------------------------------------
/// BIT-IDENTICAL TO THE TWO WHOLE-FRAME CALLS, BY CONSTRUCTION
///
/// Every output row is produced by the same row bodies `medianWide` and
/// `edgeThreshold` loop over (`impl::medianRow`, `impl::edgeRow`), handed the same
/// rows with the same borders applied: a median sample row outside the frame is
/// null and reads as zero, and the edge test's row above row 0 is row 1 and its row
/// below the last is the one before it (`BORDER_REFLECT_101`). How the frame is cut
/// into pushes cannot change a bit, which `tests/test_median_edge_stream.cpp` holds
/// it to over every chunking.
///
/// ---------------------------------------------------------------------------
/// WHEN A ROW IS WRITTEN
///
/// Edge row `y` needs median rows `y - 1 .. y + 1`, and median row `r` needs the
/// input rows its pattern reaches, down to `r + dyMax`. So pushing input row `i`
/// completes edge row `i - dyMax - 1`, and the last rows are completed by the push
/// that delivers the frame's last row. `rowsDone()` says how far `dst` is written.
///
/// @note Never allocates and never throws; the scratch is the caller's.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../binMat.hpp"
#include "../core/error.hpp"
#include "edge.hpp"
#include "medianWide.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief `medianWide` then `edgeThreshold`, row by row. **API TIER 3.**
/// @tparam K, SrcT The median's sample count and source type, as `medianWide`.
/// @tparam WordType, C, R, S As `edgeThreshold`; the defaults are the reference's.
template <size_t K, typename SrcT = uint8_t, typename WordType = uint32_t,
          EdgeCombine C = EdgeCombine::Or, EdgeRelation R = EdgeRelation::Ge,
          EdgeSpatial S = EdgeSpatial::Wide>
class MedianEdgeStream {
public:
    /// @brief Scratch elements (of `SrcT`) a stream of this width and pattern needs.
    static constexpr size_t scratchElements(size_t width, const MedianPattern<K>& pattern) {
        return (inputRows(pattern) + kMedianRows) * width;
    }

    /// @param scratch At least `scratchElements(width, pattern)` elements, the
    /// caller's, untouched by anything else until the frame is done.
    /// @param dst The binary frame the edge rows are written into, `width` x `height`.
    MedianEdgeStream(SrcT* scratch, size_t scratchCount, const MedianPattern<K>& pattern,
                     BinMatView<WordType> dst, SrcT threshold)
        : scratch_(scratch), pattern_(pattern), dst_(dst), threshold_(threshold) {
        BINCV_ASSERT(scratchCount >= scratchElements(dst.width, pattern),
                     "MedianEdgeStream: scratch smaller than scratchElements()");
        (void)scratchCount;
        for (size_t k = 0; k < K; ++k) {
            dyMax_ = pattern.offset[k].dy > dyMax_ ? pattern.offset[k].dy : dyMax_;
        }
        inRows_ = inputRows(pattern);
    }

    /// @brief Starts a new frame; `dst` is written again from row 0.
    void reset() {
        pushed_ = 0;
        medians_ = 0;
        edges_ = 0;
    }

    /// @brief Feeds the next `count` input rows, `srcStride` elements apart, and writes
    /// every edge row they complete.
    void push(const SrcT* rows, size_t count, size_t srcStride) {
        const size_t w = dst_.width, h = dst_.height;
        BINCV_ASSERT(pushed_ + count <= h, "MedianEdgeStream: more rows than the frame has");
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(inputSlot(pushed_), rows + i * srcStride, w * sizeof(SrcT));
            ++pushed_;
            // Median rows whose lowest sample row has now arrived -- or, once the
            // frame is complete, every one still owed, since the rows they would
            // wait for are outside the frame and read as zero.
            const long long ready = pushed_ == h ? static_cast<long long>(h)
                                                 : static_cast<long long>(pushed_) - dyMax_;
            while (static_cast<long long>(medians_) < ready) {
                computeMedian(medians_++);
                // Edge row y needs median row y + 1, or y - 1 at the last row.
                while (edges_ < h && (edges_ + 1 < medians_ || medians_ == h)) {
                    computeEdge(edges_++);
                }
            }
        }
    }

    /// @brief Edge rows of `dst` written so far, from row 0.
    size_t rowsDone() const { return edges_; }

private:
    static constexpr size_t kMedianRows = 3;

    static constexpr size_t inputRows(const MedianPattern<K>& pattern) {
        int lo = 0, hi = 0;
        for (size_t k = 0; k < K; ++k) {
            lo = pattern.offset[k].dy < lo ? pattern.offset[k].dy : lo;
            hi = pattern.offset[k].dy > hi ? pattern.offset[k].dy : hi;
        }
        return static_cast<size_t>(hi - lo + 1);
    }

    SrcT* inputSlot(size_t row) const { return scratch_ + (row % inRows_) * dst_.width; }
    SrcT* medianSlot(size_t row) const {
        return scratch_ + (inRows_ + row % kMedianRows) * dst_.width;
    }

    void computeMedian(size_t r) {
        const long long h = static_cast<long long>(dst_.height);
        const SrcT* rows[K];
        for (size_t k = 0; k < K; ++k) {
            const long long sy = static_cast<long long>(r) + pattern_.offset[k].dy;
            rows[k] = (sy >= 0 && sy < h) ? inputSlot(static_cast<size_t>(sy)) : nullptr;
        }
        impl::medianRow<K, SrcT>(rows, dst_.width, medianSlot(r), pattern_);
    }

    void computeEdge(size_t y) {
        const size_t h = dst_.height;
        const long long yy = static_cast<long long>(y);
        const SrcT* up = medianSlot(impl::reflect101Edge(yy - 1, h));
        const SrcT* dn = medianSlot(impl::reflect101Edge(yy + 1, h));
        impl::edgeRow<C, R, S, SrcT, WordType>(up, medianSlot(y), dn, dst_.width, dst_.row(y),
                                               threshold_);
    }

    SrcT* scratch_;
    MedianPattern<K> pattern_;
    BinMatView<WordType> dst_;
    SrcT threshold_;
    int dyMax_ = 0;
    size_t inRows_ = 1;
    size_t pushed_ = 0, medians_ = 0, edges_ = 0;
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
