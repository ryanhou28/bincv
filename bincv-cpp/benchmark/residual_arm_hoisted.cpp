// X-32 arm H -- two tap extractions per word instead of four.
//
// t01 IS t00 shifted one pixel and t11 IS t10 shifted one pixel, so only the two
// unshifted readers need ReplicatedShiftedRow::word(). The shifted pair comes from
// a shift and an or -- but ONLY IF word(i+1) is carried into the next iteration,
// because otherwise deriving t01_i needs t00_{i+1} and the call count is unchanged.
// visitRowWords walks consecutive indices, so the carry is always valid after the
// first word of a row.
#include "residual_arms.hpp"

namespace {

constexpr size_t N = 2;
using W = uint32_t;
constexpr size_t kBits = 32;

} // namespace

namespace residual {

void hoisted(const bincv::LKLevelN<N, W>& lv, const bincv::impl::RegionWords<W>& r,
             long long tapX, long long tapY, bincv::impl::TapSums& sumsX,
             bincv::impl::TapSums& sumsY) {
    for (size_t y = r.y0; y < r.y1; ++y) {
        const W* mx[N];
        const W* my[N];
        const W* ip[N];
        for (size_t k = 0; k < N; ++k) {
            mx[k] = lv.dxMag[k].row(y);
            my[k] = lv.dyMag[k].row(y);
            ip[k] = lv.prev[k].row(y);
        }
        const W* sx = lv.dxSign.row(y);
        const W* sy = lv.dySign.row(y);

        const long long srcY = static_cast<long long>(y) + tapY;
        // TWO readers per plane, not four: the +1 taps are derived.
        bincv::impl::ReplicatedShiftedRow<W> top[N], bot[N];
        for (size_t k = 0; k < N; ++k) {
            top[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY, tapX);
            bot[k] = bincv::impl::displacedRow<W>(lv.next[k], srcY + 1, tapX);
        }

        bincv::impl::TapSums rowX, rowY;
        W carryTop[N], carryBot[N];
        bool primed = false;
        size_t lastIdx = 0;

        bincv::impl::visitRowWords<W>(r, [&](size_t i, W mask) {
            W t00[N], t01[N], t10[N], t11[N], self[N];
            for (size_t k = 0; k < N; ++k) {
                const bool consecutive = primed && i == lastIdx + 1;
                const W cTop = consecutive ? carryTop[k] : top[k].word(i);
                const W cBot = consecutive ? carryBot[k] : bot[k].word(i);
                const W nTop = top[k].word(i + 1);
                const W nBot = bot[k].word(i + 1);
                t00[k] = cTop;
                t10[k] = cBot;
                // The identity: pixel x of the +1 tap is pixel x+1 of this one.
                t01[k] = static_cast<W>((cTop >> 1) | (nTop << (kBits - 1)));
                t11[k] = static_cast<W>((cBot >> 1) | (nBot << (kBits - 1)));
                carryTop[k] = nTop;
                carryBot[k] = nBot;
                self[k] = ip[k][i];
            }
            primed = true;
            lastIdx = i;

            W magX[N], magY[N];
            for (size_t k = 0; k < N; ++k) {
                magX[k] = static_cast<W>(mx[k][i] & mask);
                magY[k] = static_cast<W>(my[k][i] & mask);
            }
            const W signX = sx[i];
            const W signY = sy[i];

            rowX.t00 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t00);
            rowX.t01 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t01);
            rowX.t10 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t10);
            rowX.t11 += bincv::impl::slicedSignedSum<N, W>(magX, signX, t11);
            rowX.self += bincv::impl::slicedSignedSum<N, W>(magX, signX, self);
            rowY.t00 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t00);
            rowY.t01 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t01);
            rowY.t10 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t10);
            rowY.t11 += bincv::impl::slicedSignedSum<N, W>(magY, signY, t11);
            rowY.self += bincv::impl::slicedSignedSum<N, W>(magY, signY, self);
        });
        sumsX.t00 += rowX.t00; sumsX.t01 += rowX.t01; sumsX.t10 += rowX.t10;
        sumsX.t11 += rowX.t11; sumsX.self += rowX.self;
        sumsY.t00 += rowY.t00; sumsY.t01 += rowY.t01; sumsY.t10 += rowY.t10;
        sumsY.t11 += rowY.t11; sumsY.self += rowY.self;
    }
}

} // namespace residual
