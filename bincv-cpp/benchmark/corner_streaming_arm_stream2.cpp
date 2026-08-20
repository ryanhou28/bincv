// T3.11 / X-23 -- ARM S2, STREAMING, TWO PASSES. NOT SHIPPED.
//
// This is the shape the task estimate describes -- "a three-row ring, for roughly
// 2x the response compute" -- and the arm exists to price it rather than to
// assume it. Pass 1 evaluates every response purely to find the global maximum
// and throws the values away; pass 2 re-evaluates into the ring, thresholds,
// suppresses and ranks.
//
// It is a straight transcription of `selectGoodFeatures` steps 1-4b with the
// frame-sized map replaced by three rows, so it carries no extra state at all --
// its peak is the same ring plus the same candidate array as S1. The two arms
// therefore differ ONLY in the number of evaluations, which is exactly the term
// the estimate was about.
//
// It lives here rather than in ops/corner.hpp because nothing should ship two
// implementations of one answer. If S2 had won the comparison, this file is what
// would have moved into the header.

#include <algorithm>

#include "corner_streaming_arms.hpp"

namespace t311 {

namespace {

/// @brief Pass 1: the global maximum, from a sweep that keeps nothing.
/// @note One row of scratch is still needed, because the row kernel writes a row.
///       It is ring row 0 -- the ring the caller already owns -- so this pass adds
///       no bytes.
template <typename W>
float globalMax(const Planes<W>& p, int blockSize, bincv::ResponseMap ring) {
    const int width = static_cast<int>(p.magX.width);
    const int height = static_cast<int>(p.magX.height);
    float* scratch = ring.row(0);
    bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, blockSize, 0, scratch);
    float maxVal = scratch[0];
    for (int x = 1; x < width; ++x) {
        if (scratch[static_cast<std::size_t>(x)] > maxVal)
            maxVal = scratch[static_cast<std::size_t>(x)];
    }
    for (int y = 1; y < height; ++y) {
        bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, blockSize, y, scratch);
        for (int x = 0; x < width; ++x) {
            if (scratch[static_cast<std::size_t>(x)] > maxVal)
                maxVal = scratch[static_cast<std::size_t>(x)];
        }
    }
    return maxVal;
}

}  // namespace

template <typename W>
float stream2Respond(const Planes<W>& p, int blockSize, bincv::ResponseMap ring) {
    const int width = static_cast<int>(p.magX.width);
    const int height = static_cast<int>(p.magX.height);
    if (width == 0 || height == 0) return 0.0f;

    const float maxVal = globalMax<W>(p, blockSize, ring);
    // Pass 2's sweep, without the suppression -- the response work the second pass
    // costs, which is the whole point of this arm.
    for (int y = 0; y < height; ++y) {
        bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, blockSize, y,
                                       ring.row(static_cast<std::size_t>(y) %
                                                bincv::kResponseRingRows));
    }
    return maxVal;
}

template <typename W>
bincv::CornerResult stream2Detect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                  bincv::ResponseMap ring, bincv::Corner* corners,
                                  std::size_t capacity) {
    bincv::CornerResult out;
    const int width = static_cast<int>(p.magX.width);
    const int height = static_cast<int>(p.magX.height);
    if (width == 0 || height == 0) return out;

    // Steps 1 and 2, exactly as selectGoodFeatures forms them.
    const float maxVal = globalMax<W>(p, params.blockSize, ring);
    const float threshold =
        static_cast<float>(static_cast<double>(maxVal) * params.qualityLevel);

    // Step 3: the second sweep, with the threshold-fused 3x3 NMS over the ring and
    // the same bounded heap. Identical to the frame-map scan except that `prev`,
    // `cur` and `next` come out of three rows instead of out of a frame.
    std::size_t ranked = 0;
    for (int y = 0; y < height; ++y) {
        float* cur = ring.row(static_cast<std::size_t>(y) % bincv::kResponseRingRows);
        bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, params.blockSize, y, cur);
        if (y < 2) continue;

        const int cy = y - 1;
        const float* above =
            ring.row(static_cast<std::size_t>(cy - 1) % bincv::kResponseRingRows);
        const float* mid = ring.row(static_cast<std::size_t>(cy) % bincv::kResponseRingRows);
        const float* below = cur;
        for (int x = 1; x + 1 < width; ++x) {
            const float val = mid[static_cast<std::size_t>(x)];
            if (!(val > threshold)) continue;
            bool isMax = true;
            for (int dx = -1; dx <= 1 && isMax; ++dx) {
                const std::size_t c = static_cast<std::size_t>(x + dx);
                if (above[c] > val || mid[c] > val || below[c] > val) isMax = false;
            }
            if (!isMax) continue;

            bincv::Corner candidate;
            candidate.x = x;
            candidate.y = cy;
            candidate.response = val;
            if (ranked < capacity) {
                corners[ranked++] = candidate;
                std::push_heap(corners, corners + ranked, bincv::impl::CornerStronger());
            } else {
                out.candidatesTruncated = true;
                if (capacity == 0) {
                    out.candidatesRanked = 0;
                    return out;
                }
                if (bincv::impl::CornerStronger()(candidate, corners[0])) {
                    std::pop_heap(corners, corners + ranked, bincv::impl::CornerStronger());
                    corners[ranked - 1] = candidate;
                    std::push_heap(corners, corners + ranked, bincv::impl::CornerStronger());
                }
            }
        }
    }
    out.candidatesRanked = ranked;
    if (ranked == 0) return out;

    // Steps 4a and 4b, unchanged.
    std::sort(corners, corners + ranked, bincv::impl::CornerStronger());
    const std::size_t limit = (params.maxCorners > 0)
                                  ? std::min(capacity, static_cast<std::size_t>(params.maxCorners))
                                  : capacity;
    std::size_t kept = 0;
    if (params.minDistance >= 1.0) {
        const double minDistanceSq = params.minDistance * params.minDistance;
        for (std::size_t i = 0; i < ranked && kept < limit; ++i) {
            const bincv::Corner candidate = corners[i];
            bool good = true;
            for (std::size_t j = 0; j < kept; ++j) {
                const double dx =
                    static_cast<double>(candidate.x) - static_cast<double>(corners[j].x);
                const double dy =
                    static_cast<double>(candidate.y) - static_cast<double>(corners[j].y);
                if (dx * dx + dy * dy < minDistanceSq) {
                    good = false;
                    break;
                }
            }
            if (good) corners[kept++] = candidate;
        }
    } else {
        kept = (ranked < limit) ? ranked : limit;
    }
    out.count = kept;
    return out;
}

template float stream2Respond<uint32_t>(const Planes<uint32_t>&, int, bincv::ResponseMap);
template float stream2Respond<uint64_t>(const Planes<uint64_t>&, int, bincv::ResponseMap);
template bincv::CornerResult stream2Detect<uint32_t>(const Planes<uint32_t>&,
                                                     const bincv::GoodFeaturesParams&,
                                                     bincv::ResponseMap, bincv::Corner*,
                                                     std::size_t);
template bincv::CornerResult stream2Detect<uint64_t>(const Planes<uint64_t>&,
                                                     const bincv::GoodFeaturesParams&,
                                                     bincv::ResponseMap, bincv::Corner*,
                                                     std::size_t);

}  // namespace t311
