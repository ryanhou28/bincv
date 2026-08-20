// T3.11 / X-23 -- ARM F, THE CONTROL: the shipped frame-sized response map.
//
// Nothing here is new code. It is `cornerMinEigenVal` into a caller-owned
// 1 228 800 B `float` map (column-major, two slid `SlidingWindowCount`s down each
// column, the cross term recomputed) followed by `selectGoodFeatures`. The arm
// exists so that both sides of X-23's ratio come from ONE build, rather than one
// side being quoted from X-18 or X-19 -- and so that the control's code layout is
// as isolated as the streaming arms' is.
//
// `frameRespond` deliberately includes the `minMaxLoc` pass over the finished
// map. See corner_streaming_arms.hpp for why: the frame-map form cannot threshold
// without it, and charging only the streaming arms for their global bookkeeping
// would be a rigged column.

#include "corner_streaming_arms.hpp"

namespace t311 {

template <typename W>
float frameRespond(const Planes<W>& p, int blockSize, bincv::ResponseMap scratch) {
    bincv::cornerMinEigenVal<W>(p.magX, p.magY, p.signX, p.signY, blockSize, scratch);

    // selectGoodFeatures step 1, spelled here rather than called, because the
    // selection entry point cannot be split. Same region -- the WHOLE map, border
    // included -- and the same seed pixel.
    const bincv::ConstResponseMap map(scratch);
    float maxVal = map.row(0)[0];
    for (std::size_t y = 0; y < map.height; ++y) {
        const float* row = map.row(y);
        for (std::size_t x = 0; x < map.width; ++x) {
            if (row[x] > maxVal) maxVal = row[x];
        }
    }
    return maxVal;
}

template <typename W>
bincv::CornerResult frameDetect(const Planes<W>& p, const bincv::GoodFeaturesParams& params,
                                bincv::ResponseMap scratch, bincv::Corner* corners,
                                std::size_t capacity) {
    bincv::cornerMinEigenVal<W>(p.magX, p.magY, p.signX, p.signY, params.blockSize, scratch);
    return bincv::selectGoodFeatures(bincv::ConstResponseMap(scratch), params, corners, capacity);
}

template float frameRespond<uint32_t>(const Planes<uint32_t>&, int, bincv::ResponseMap);
template float frameRespond<uint64_t>(const Planes<uint64_t>&, int, bincv::ResponseMap);
template bincv::CornerResult frameDetect<uint32_t>(const Planes<uint32_t>&,
                                                   const bincv::GoodFeaturesParams&,
                                                   bincv::ResponseMap, bincv::Corner*,
                                                   std::size_t);
template bincv::CornerResult frameDetect<uint64_t>(const Planes<uint64_t>&,
                                                   const bincv::GoodFeaturesParams&,
                                                   bincv::ResponseMap, bincv::Corner*,
                                                   std::size_t);

}  // namespace t311
