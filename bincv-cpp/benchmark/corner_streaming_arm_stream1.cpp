// T3.11 / X-23 -- ARM S1, STREAMING, ONE PASS. This arm is the SHIPPED entry
// point: `bincv::goodFeaturesToTrackStreaming`, called through a separate
// translation unit so its code layout is its own.
//
// One evaluation per pixel into a three-row ring; a running global maximum; a
// top-K over RAW 3x3 maxima pruned against the running threshold; the quality
// threshold, the sort and the spacing filter applied after the last row. Why that
// is EXACTLY the frame-map form's answer rather than an approximation of it is
// the four-item argument in ops/corner.hpp's "STAGE 4" section, and
// `Corner.Streaming_*` proves it element for element.
//
// `stream1Respond` is the sweep with its running maximum and nothing else -- the
// state the form must have before it can suppress, matching the definition F's
// arm is held to.

#include "corner_streaming_arms.hpp"

namespace t311 {

template <typename W>
float stream1Respond(const Planes<W>& p, int blockSize, bincv::ResponseMap ring) {
    const int width = static_cast<int>(p.magX.width);
    const int height = static_cast<int>(p.magX.height);
    if (width == 0 || height == 0) return 0.0f;

    float* first = ring.row(0);
    bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, blockSize, 0, first);
    float runningMax = first[0];
    for (int x = 1; x < width; ++x) {
        if (first[static_cast<std::size_t>(x)] > runningMax)
            runningMax = first[static_cast<std::size_t>(x)];
    }
    for (int y = 1; y < height; ++y) {
        float* cur = ring.row(static_cast<std::size_t>(y) % bincv::kResponseRingRows);
        bincv::cornerMinEigenValRow<W>(p.magX, p.magY, p.signX, p.signY, blockSize, y, cur);
        for (int x = 0; x < width; ++x) {
            if (cur[static_cast<std::size_t>(x)] > runningMax)
                runningMax = cur[static_cast<std::size_t>(x)];
        }
    }
    return runningMax;
}

template <typename W>
bincv::CornerResult stream1Detect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                  bincv::ResponseMap ring, bincv::Corner* corners,
                                  std::size_t capacity) {
    return bincv::goodFeaturesToTrackStreaming<W>(p.magX, p.magY, p.signX, p.signY, params, ring,
                                                  corners, capacity);
}

template float stream1Respond<uint32_t>(const Planes<uint32_t>&, int, bincv::ResponseMap);
template float stream1Respond<uint64_t>(const Planes<uint64_t>&, int, bincv::ResponseMap);
template bincv::CornerResult stream1Detect<uint32_t>(const Planes<uint32_t>&,
                                                     const bincv::GoodFeaturesParams&,
                                                     bincv::ResponseMap, bincv::Corner*,
                                                     std::size_t);
template bincv::CornerResult stream1Detect<uint64_t>(const Planes<uint64_t>&,
                                                     const bincv::GoodFeaturesParams&,
                                                     bincv::ResponseMap, bincv::Corner*,
                                                     std::size_t);

}  // namespace t311
