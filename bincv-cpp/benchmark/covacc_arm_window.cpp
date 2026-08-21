// X-29 arm W -- ONE accumulator for the whole window.
//
// The rows accumulate straight into the window total: no per-row struct to zero,
// no per-row add. What it gives up is D-15 item 4's reason for existing -- the
// per-row partials break the serialized dependency chain through popcount latency,
// which X-11b measured worth 1.08x AT N = 1, where the structure is four counters.
// At N = 4 it is 64, and the per-row zero-and-add is ~3N^2+N adds plus 4N^2 words
// of zeroing against 1-2 words of real work per row.
#include "covacc_arms.hpp"

namespace {

template <size_t N>
int64_t windowWideAt(const bincv::BinMatConstView<uint32_t>* magX,
                     const bincv::BinMatConstView<uint32_t>* magY,
                     bincv::BinMatConstView<uint32_t> signX,
                     bincv::BinMatConstView<uint32_t> signY, const bincv::Rect* windows,
                     size_t windowCount) {
    int64_t sum = 0;
    for (size_t k = 0; k < windowCount; ++k) {
        const auto r = bincv::impl::clipRegion<uint32_t>(magX[0].width, magX[0].height, windows[k]);
        if (r.isEmpty) continue;
        bincv::impl::BitSlicedPairCounts<N> total;
        for (size_t y = r.y0; y < r.y1; ++y) {
            const uint32_t* rowX[N];
            const uint32_t* rowY[N];
            for (size_t p = 0; p < N; ++p) {
                rowX[p] = magX[p].row(y);
                rowY[p] = magY[p].row(y);
            }
            // THE ONE DIFFERENCE: accumulate straight into `total`.
            bincv::impl::bitSlicedPairRowRegion<N, uint32_t>(rowX, rowY, signX.row(y),
                                                             signY.row(y), r, total);
        }
        const bincv::GradientCovariance c = bincv::impl::combineBitSlicedPairs<N>(total);
        sum += c.sumXX + c.sumYY + c.sumXY;
    }
    return sum;
}

} // namespace

namespace covacc {
int64_t windowWide(const bincv::BinMatConstView<uint32_t>* magX,
                   const bincv::BinMatConstView<uint32_t>* magY,
                   bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                   size_t n, const bincv::Rect* windows, size_t windowCount) {
    switch (n) {
        case 1: return windowWideAt<1>(magX, magY, signX, signY, windows, windowCount);
        case 2: return windowWideAt<2>(magX, magY, signX, signY, windows, windowCount);
        case 3: return windowWideAt<3>(magX, magY, signX, signY, windows, windowCount);
        default: return windowWideAt<4>(magX, magY, signX, signY, windows, windowCount);
    }
}
} // namespace covacc
