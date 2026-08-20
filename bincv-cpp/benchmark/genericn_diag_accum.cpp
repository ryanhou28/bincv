// T3.9 / E-4 (X-21) -- THE ACCUMULATOR DECOMPOSITION POINT, ADDED AT TRIAGE.
//
// X-21's conclusion 3 recorded something the decision rule has no band for: on the
// whole-frame count the LIBRARY is FASTER than the hand-written control -- 0.883x
// and 0.942x. It attributed that to impl::visitRowWords' head/interior/tail
// skeleton. That attribution does not survive reading the two loops side by side:
// genericn_arm_handwritten.cpp's count ALREADY has that skeleton -- it runs the
// interior words unmasked and masks only the trailing word, which is exactly what
// visitRowWords does when the head mask is all ones.
//
// The difference the original run never isolated is WHERE THE SUM LANDS.
// impl::countRowRegion returns a PER-ROW partial that impl::countViewRegion adds,
// so a 640x480 count is 480 independent popcount dependency chains; the
// hand-written arm accumulates into one `total` across the whole frame, so it is
// one 9600-long chain. That is D-15's accumulator split, measured separately by
// X-11b at 1.03-1.09x at LK window sizes -- on a target where the popcount is the
// latency bottleneck (D-6). The same applies to the covariance: the library builds
// a fresh CovarianceCount per row inside impl::covarianceRowRegion, the
// hand-written arm carries xx/yy/both/whenSet across all 31 window rows.
//
// This file measures that one variable and nothing else. It contains FOUR
// functions and no binCV header:
//
//   count      one-chain   an exact copy of the hand-written arm's count
//   count      per-row     the same loop, per-row partial summed at the end
//   covariance one-chain   an exact copy of the hand-written arm's covariance
//   covariance per-row     the same loop, per-row partials summed at the end
//
// ALL FOUR IN ONE TRANSLATION UNIT, deliberately. genericn_arms.hpp explains why
// the three ARMS are in separate objects -- morphology_path_benchmark measured
// ~10% timing movement from code layout alone -- and that cuts both ways here: a
// per-row variant in a different object from its one-chain twin would confound the
// A/B with layout. Keeping the twins together makes the comparison within one
// object, and the one-chain copy doubles as a LAYOUT CONTROL: if it does not
// reproduce the hand-written arm's number, the layout effect is visible rather
// than assumed.
//
// THIS IS NOT AN ARM. The rule comparison is the three arms of genericn_arms.hpp,
// committed and run before this file existed, and nothing here changes it.

#include <cstddef>
#include <cstdint>

#include "genericn_arms.hpp"

namespace {

inline size_t popcount32(uint32_t w) {
    return static_cast<size_t>(__builtin_popcountll(static_cast<unsigned long long>(w)));
}

inline uint32_t headMaskFrom(unsigned k) {
    return static_cast<uint32_t>(~((UINT32_C(1) << k) - UINT32_C(1)));
}

inline uint32_t lowMask(unsigned k) {
    return k >= 32u ? UINT32_C(0xFFFFFFFF)
                    : static_cast<uint32_t>((UINT32_C(1) << k) - UINT32_C(1));
}

/// @brief One word's contribution to the four covariance numbers.
/// @note Character for character genericn_arm_handwritten.cpp's, so the only
///       difference between the two covariance functions below is the accumulator.
inline void covarianceWord(const uint32_t* mx, const uint32_t* my, const uint32_t* sx,
                           const uint32_t* sy, size_t i, uint32_t mask, size_t& xx, size_t& yy,
                           size_t& both, size_t& whenSet) {
    const uint32_t a = static_cast<uint32_t>(mx[i] & mask);
    const uint32_t b = static_cast<uint32_t>(my[i] & mask);
    const uint32_t ab = static_cast<uint32_t>(a & b);
    const uint32_t sel = static_cast<uint32_t>(sx[i] ^ sy[i]);
    xx += popcount32(a);
    yy += popcount32(b);
    both += popcount32(ab);
    whenSet += popcount32(static_cast<uint32_t>(ab & sel));
}

/// @brief Window geometry, resolved once. Shared by both covariance variants so
///        the clipping arithmetic cannot differ between them.
struct Window {
    bool empty = true;
    size_t x0 = 0, x1 = 0, y0 = 0, y1 = 0;
    size_t firstWord = 0, lastWord = 0;
    uint32_t headMask = 0, tailMask = 0;
};

Window clipWindow(int width, int height, int wx, int wy, int wsize) {
    Window win;
    if (width <= 0 || height <= 0 || wsize <= 0) return win;

    const long long lx0 = (wx > 0) ? wx : 0;
    const long long lx1 = (static_cast<long long>(wx) + wsize < width)
                              ? static_cast<long long>(wx) + wsize
                              : width;
    const long long ly0 = (wy > 0) ? wy : 0;
    const long long ly1 = (static_cast<long long>(wy) + wsize < height)
                              ? static_cast<long long>(wy) + wsize
                              : height;
    if (lx0 >= lx1 || ly0 >= ly1) return win;

    win.empty = false;
    win.x0 = static_cast<size_t>(lx0);
    win.x1 = static_cast<size_t>(lx1);
    win.y0 = static_cast<size_t>(ly0);
    win.y1 = static_cast<size_t>(ly1);
    win.firstWord = win.x0 / 32u;
    win.lastWord = (win.x1 - 1u) / 32u;
    win.headMask = headMaskFrom(static_cast<unsigned>(win.x0 % 32u));
    win.tailMask = lowMask(static_cast<unsigned>(((win.x1 - 1u) % 32u) + 1u));
    return win;
}

}  // namespace

namespace t39 {

/// @brief The hand-written count, ONE accumulator across the whole frame.
/// @note An exact copy of genericn_arm_handwritten.cpp's, present as the layout
///       control described in this file's header.
size_t countWholeOneChain(const Word* src, size_t strideWords, int width, int height) {
    if (width <= 0 || height <= 0) return 0;

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t words = (w + 31u) / 32u;
    const uint32_t tailMask = lowMask(static_cast<unsigned>(w % 32u == 0u ? 32u : w % 32u));

    size_t total = 0;
    for (size_t y = 0; y < h; ++y) {
        const uint32_t* r = src + y * strideWords;
        for (size_t i = 0; i + 1u < words; ++i) total += popcount32(r[i]);
        total += popcount32(static_cast<uint32_t>(r[words - 1u] & tailMask));
    }
    return total;
}

/// @brief The same count with D-15's accumulator: a per-row partial, summed after.
/// @note The word loop, the masks and the popcount spelling are identical to
///       countWholeOneChain above. The ONLY difference is that `rowTotal` starts at
///       zero every row, which is what impl::countRowRegion returning a value does.
size_t countWholePerRow(const Word* src, size_t strideWords, int width, int height) {
    if (width <= 0 || height <= 0) return 0;

    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    const size_t words = (w + 31u) / 32u;
    const uint32_t tailMask = lowMask(static_cast<unsigned>(w % 32u == 0u ? 32u : w % 32u));

    size_t total = 0;
    for (size_t y = 0; y < h; ++y) {
        const uint32_t* r = src + y * strideWords;
        size_t rowTotal = 0;
        for (size_t i = 0; i + 1u < words; ++i) rowTotal += popcount32(r[i]);
        rowTotal += popcount32(static_cast<uint32_t>(r[words - 1u] & tailMask));
        total += rowTotal;
    }
    return total;
}

/// @brief The hand-written covariance, ONE chain across all window rows.
/// @note An exact copy of genericn_arm_handwritten.cpp's body; the layout control.
Cov covarianceWindowOneChain(const Word* dx, const Word* dy, size_t strideWords, int width,
                             int height, int wx, int wy, int wsize) {
    Cov out;
    const Window win = clipWindow(width, height, wx, wy, wsize);
    if (win.empty) return out;

    const size_t planeWords = static_cast<size_t>(height) * strideWords;
    const uint32_t* magX = dx;
    const uint32_t* sgnX = dx + planeWords;
    const uint32_t* magY = dy;
    const uint32_t* sgnY = dy + planeWords;

    size_t xx = 0, yy = 0, both = 0, whenSet = 0;

    for (size_t y = win.y0; y < win.y1; ++y) {
        const uint32_t* mx = magX + y * strideWords;
        const uint32_t* my = magY + y * strideWords;
        const uint32_t* sx = sgnX + y * strideWords;
        const uint32_t* sy = sgnY + y * strideWords;

        if (win.firstWord == win.lastWord) {
            covarianceWord(mx, my, sx, sy, win.firstWord,
                           static_cast<uint32_t>(win.headMask & win.tailMask), xx, yy, both,
                           whenSet);
            continue;
        }
        covarianceWord(mx, my, sx, sy, win.firstWord, win.headMask, xx, yy, both, whenSet);
        for (size_t i = win.firstWord + 1u; i < win.lastWord; ++i) {
            covarianceWord(mx, my, sx, sy, i, UINT32_C(0xFFFFFFFF), xx, yy, both, whenSet);
        }
        covarianceWord(mx, my, sx, sy, win.lastWord, win.tailMask, xx, yy, both, whenSet);
    }

    out.xx = xx;
    out.yy = yy;
    out.whenSet = whenSet;
    out.whenClear = both - whenSet;
    return out;
}

/// @brief The same covariance with D-15's accumulator: four partials per ROW.
/// @note impl::covarianceRowRegion's shape. Same words, same masks, same
///       popcounts, same order -- only the dependency chains are cut per row.
Cov covarianceWindowPerRow(const Word* dx, const Word* dy, size_t strideWords, int width,
                           int height, int wx, int wy, int wsize) {
    Cov out;
    const Window win = clipWindow(width, height, wx, wy, wsize);
    if (win.empty) return out;

    const size_t planeWords = static_cast<size_t>(height) * strideWords;
    const uint32_t* magX = dx;
    const uint32_t* sgnX = dx + planeWords;
    const uint32_t* magY = dy;
    const uint32_t* sgnY = dy + planeWords;

    size_t xx = 0, yy = 0, both = 0, whenSet = 0;

    for (size_t y = win.y0; y < win.y1; ++y) {
        const uint32_t* mx = magX + y * strideWords;
        const uint32_t* my = magY + y * strideWords;
        const uint32_t* sx = sgnX + y * strideWords;
        const uint32_t* sy = sgnY + y * strideWords;

        size_t rxx = 0, ryy = 0, rboth = 0, rset = 0;
        if (win.firstWord == win.lastWord) {
            covarianceWord(mx, my, sx, sy, win.firstWord,
                           static_cast<uint32_t>(win.headMask & win.tailMask), rxx, ryy, rboth,
                           rset);
        } else {
            covarianceWord(mx, my, sx, sy, win.firstWord, win.headMask, rxx, ryy, rboth, rset);
            for (size_t i = win.firstWord + 1u; i < win.lastWord; ++i) {
                covarianceWord(mx, my, sx, sy, i, UINT32_C(0xFFFFFFFF), rxx, ryy, rboth, rset);
            }
            covarianceWord(mx, my, sx, sy, win.lastWord, win.tailMask, rxx, ryy, rboth, rset);
        }
        xx += rxx;
        yy += ryy;
        both += rboth;
        whenSet += rset;
    }

    out.xx = xx;
    out.yy = yy;
    out.whenSet = whenSet;
    out.whenClear = both - whenSet;
    return out;
}

}  // namespace t39
