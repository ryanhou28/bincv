// The two per-row arms share this body VERBATIM by inclusion, so that P and P'
// cannot drift and the noise floor stays a measurement of layout rather than of a
// copy-paste divergence. See covacc_arms.hpp.

#include "covacc_arms.hpp"

namespace {

/// The shipped shape: a fresh BitSlicedPairCounts per row, added into the total.
template <size_t N>
int64_t perRowAt(const bincv::BinMatConstView<uint32_t>* magX,
                 const bincv::BinMatConstView<uint32_t>* magY,
                 bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                 const bincv::Rect* windows, size_t windowCount) {
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
            bincv::impl::BitSlicedPairCounts<N> row;
            bincv::impl::bitSlicedPairRowRegion<N, uint32_t>(rowX, rowY, signX.row(y),
                                                             signY.row(y), r, row);
            total.add(row);
        }
        const bincv::GradientCovariance c = bincv::impl::combineBitSlicedPairs<N>(total);
        sum += c.sumXX + c.sumYY + c.sumXY;
    }
    return sum;
}

int64_t dispatch(const bincv::BinMatConstView<uint32_t>* magX,
                 const bincv::BinMatConstView<uint32_t>* magY,
                 bincv::BinMatConstView<uint32_t> signX, bincv::BinMatConstView<uint32_t> signY,
                 size_t n, const bincv::Rect* windows, size_t windowCount) {
    switch (n) {
        case 1: return perRowAt<1>(magX, magY, signX, signY, windows, windowCount);
        case 2: return perRowAt<2>(magX, magY, signX, signY, windows, windowCount);
        case 3: return perRowAt<3>(magX, magY, signX, signY, windows, windowCount);
        default: return perRowAt<4>(magX, magY, signX, signY, windows, windowCount);
    }
}

} // namespace
