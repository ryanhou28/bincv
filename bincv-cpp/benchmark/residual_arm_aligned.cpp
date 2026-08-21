// X-34 arm A -- the window extracted into ONE aligned word per row.
//
// A 31-pixel window at an arbitrary offset spans 1.94 uint32 words on average, so
// the shipped kernel issues twice the popcounts it needs, each covering 15.5 useful
// pixels instead of 31. Extracting the window's bits into bits [0, width) of a
// single word makes every popcount cover the whole window.
//
// THE TAPS COST NOTHING EXTRA. ReplicatedShiftedRow already shifts; asking it for
// `word(0)` with `off = x0 + tapX` returns exactly the 32 source bits starting at
// the window's left edge, aligned to bit 0. Only the previous-frame planes need an
// explicit extraction, and those are entirely inside the frame -- the region is
// already clipped -- so they need no border handling at all.
#include "residual_arms.hpp"

namespace {

constexpr size_t N = 2;
using W = uint32_t;
constexpr size_t kBits = 32;

/// Bits [x0, x0 + kBits) of a row, aligned to bit 0. The region is clipped, so
/// every bit that survives the caller's mask is inside the frame.
/// @note The `s == 0` guard is not decoration: `hi << 32` is undefined.
inline W alignedWord(const W* row, size_t words, size_t x0) {
    const size_t w0 = x0 / kBits;
    const size_t s = x0 % kBits;
    const W lo = (w0 < words) ? row[w0] : 0;
    if (s == 0) return lo;
    const W hi = (w0 + 1 < words) ? row[w0 + 1] : 0;
    return static_cast<W>((lo >> s) | (hi << (kBits - s)));
}

} // namespace

namespace residual {

void aligned(const bincv::LKLevelN<N, W>& lv, const bincv::impl::RegionWords<W>& r, size_t x0,
             size_t x1, long long tapX, long long tapY, bincv::impl::TapSums& sumsX,
             bincv::impl::TapSums& sumsY) {
    const size_t width = x1 - x0;
    if (width == 0) return;
    const size_t words = bincv::impl::minRowWords<W>(lv.prev[0].width);
    // One mask for the whole window, not a head and a tail.
    const W mask = bincv::impl::lowBitsMask<W>(width);

    for (size_t y = r.y0; y < r.y1; ++y) {
        W t00[N], t01[N], t10[N], t11[N], self[N], magX[N], magY[N];
        const long long srcY = static_cast<long long>(y) + tapY;
        for (size_t k = 0; k < N; ++k) {
            // off = x0 + tapX, then word(0): the window's source bits, aligned.
            t00[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY,
                                                  static_cast<long long>(x0) + tapX).word(0);
            t01[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY,
                                                  static_cast<long long>(x0) + tapX + 1).word(0);
            t10[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY + 1,
                                                  static_cast<long long>(x0) + tapX).word(0);
            t11[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY + 1,
                                                  static_cast<long long>(x0) + tapX + 1).word(0);
            self[k] = alignedWord(lv.prev[k].row(y), words, x0);
            magX[k] = static_cast<W>(alignedWord(lv.dxMag[k].row(y), words, x0) & mask);
            magY[k] = static_cast<W>(alignedWord(lv.dyMag[k].row(y), words, x0) & mask);
        }
        const W signX = alignedWord(lv.dxSign.row(y), words, x0);
        const W signY = alignedWord(lv.dySign.row(y), words, x0);

        sumsX.t00 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t00);
        sumsX.t01 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t01);
        sumsX.t10 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t10);
        sumsX.t11 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t11);
        sumsX.self += bincv::impl::slicedSignedSum<N, W>(magX, signX, self);
        sumsY.t00 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t00);
        sumsY.t01 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t01);
        sumsY.t10 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t10);
        sumsY.t11 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t11);
        sumsY.self += bincv::impl::slicedSignedSum<N, W>(magY, signY, self);
    }
}

} // namespace residual
