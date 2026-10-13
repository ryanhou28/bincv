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
/// What it holds is the neighbourhood and nothing more: the input rows a later push's
/// medians still read (one for the reference's L), and the three median rows the
/// edge test reads. 4 x 752 = 3,008 bytes at the reference's shape.
///
/// ---------------------------------------------------------------------------
/// THE CALLER'S ROWS ARE READ WHERE THEY ARE
///
/// A camera's DMA leaves its lines in a buffer the caller owns, so a push reads the
/// rows it is given in place, and copies out only the ones a median of the NEXT push
/// will need -- the last `inputRows - 1` of them, one row in sixteen for the
/// reference's L pushed sixteen at a time. The rows are the caller's again when
/// `push` returns.
///
/// Each held row starts on a 4-byte boundary when the scratch does (its length is
/// rounded up), which is what the edge test's aligned arm needs on a core without a
/// vector unit.
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
        return (inputRows(pattern) - 1 + kMedianRows) * slotElements(width);
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
        held_ = inputRows(pattern) - 1;
        slot_ = slotElements(dst.width);
    }

    /// @brief Starts a new frame; `dst` is written again from row 0.
    void reset() {
        pushed_ = 0;
        medians_ = 0;
        edges_ = 0;
    }

    /// @brief Feeds the next `count` input rows, `srcStride` elements apart, and writes
    /// every edge row they complete.
    /// @note The rows are read in place and only for the duration of the call.
    void push(const SrcT* rows, size_t count, size_t srcStride) {
        const size_t w = dst_.width, h = dst_.height;
        BINCV_ASSERT(pushed_ + count <= h, "MedianEdgeStream: more rows than the frame has");
        if (count == 0) return;
        chunk_ = rows;
        chunkStride_ = srcStride;
        chunkFirst_ = pushed_;
        pushed_ += count;
        // Median rows whose lowest sample row has now arrived -- or, once the frame is
        // complete, every one still owed, since the rows they would wait for are
        // outside the frame and read as zero.
        const long long ready = pushed_ == h ? static_cast<long long>(h)
                                             : static_cast<long long>(pushed_) - dyMax_;
        while (static_cast<long long>(medians_) < ready) {
            computeMedian(medians_++);
            // Edge row y needs median row y + 1, or y - 1 at the last row.
            while (edges_ < h && (edges_ + 1 < medians_ || medians_ == h)) {
                computeEdge(edges_++);
            }
        }
        // The rows the next push's first median reaches back to. Any older than this
        // push are already held.
        const size_t keep = pushed_ > held_ ? pushed_ - held_ : 0;
        for (size_t r = keep > chunkFirst_ ? keep : chunkFirst_; r < pushed_; ++r) {
            std::memcpy(heldSlot(r), rows + (r - chunkFirst_) * srcStride, w * sizeof(SrcT));
        }
        chunk_ = nullptr;
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

    /// A held row's length in elements: `width` rounded up to a whole number of 4-byte
    /// words, so every slot of a 4-byte-aligned scratch starts on a boundary.
    static constexpr size_t slotElements(size_t width) {
        constexpr size_t kPer = sizeof(SrcT) < 4 ? 4 / sizeof(SrcT) : 1;
        return (width + kPer - 1) / kPer * kPer;
    }

    // Never asked for when nothing is held: every row a median reads is then this push's.
    SrcT* heldSlot(size_t row) const { return scratch_ + (held_ ? row % held_ : 0) * slot_; }
    SrcT* medianSlot(size_t row) const {
        return scratch_ + (held_ + row % kMedianRows) * slot_;
    }

    /// Input row `row`, in the caller's chunk if this push delivered it, else held.
    const SrcT* inputRow(size_t row) const {
        if (row >= chunkFirst_) return chunk_ + (row - chunkFirst_) * chunkStride_;
        return heldSlot(row);
    }

    void computeMedian(size_t r) {
        const long long h = static_cast<long long>(dst_.height);
        const SrcT* rows[K];
        for (size_t k = 0; k < K; ++k) {
            const long long sy = static_cast<long long>(r) + pattern_.offset[k].dy;
            rows[k] = (sy >= 0 && sy < h) ? inputRow(static_cast<size_t>(sy)) : nullptr;
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
    size_t held_ = 0;  // input rows kept between pushes
    size_t slot_ = 0;  // elements per held row
    const SrcT* chunk_ = nullptr;
    size_t chunkStride_ = 0, chunkFirst_ = 0;
    size_t pushed_ = 0, medians_ = 0, edges_ = 0;
};

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
