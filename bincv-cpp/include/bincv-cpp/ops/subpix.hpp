#pragma once

/// @file subpix.hpp
/// @brief Sub-pixel corner refinement on bit-planes. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS WORKS AT ALL ON TERNARY DERIVATIVES
///
/// `cv::cornerSubPix` refines a corner by solving, over a window around it,
///
///     G q = b,    G = sum w (grad I)(grad I)^T,    b = sum w (grad I)(grad I)^T p
///
/// for the position `q` that every edge in the window points at. **The solution is
/// invariant to a scale on the image**: scaling `I` by `s` scales `grad I` by `s`, so it
/// scales BOTH `G` and `b` by `s^2`, and `q = G^-1 b` is unchanged.
///
/// That is why this is possible on binCV's data at all. A ternary derivative carries
/// `+/-1` where a byte pipeline carries `+/-255`, and the refined position is the same
/// to within rounding. **A binCV user checked it independently before asking for this
/// operation: refining on a 0/255 image against the same content as 0/1 agreed to
/// 0.00018 px mean over 5924 corners.**
///
/// ---------------------------------------------------------------------------
/// AND WHY IT IS binCV-SHAPED RATHER THAN A PORTED LOOP
///
/// The weights and the positions are per-pixel scalars, so unlike every reduction in
/// this library this one does **not** collapse into popcounts -- a weighted sum is not
/// a bit count. What bit-planes buy here is different and still large:
///
/// > **A PIXEL WITH NO GRADIENT CONTRIBUTES NOTHING TO ANY OF THE SIX SUMS, AND
/// > `|dx| | |dy|` FINDS ALL OF THEM A WORD AT A TIME.**
///
/// On a binarised edge map most of a 31x31 window is flat, so the loop below computes
/// the skip mask word-wise and then visits only set bits. The dense spelling would touch
/// 961 pixels; this touches the edges.
///
/// ---------------------------------------------------------------------------
/// TIER 2, AND THE DIFFERENCE IS THE GRADIENT
///
/// The refinement rule, the Gaussian window, the zero-zone and the termination are
/// `cv::cornerSubPix`'s. **The gradient is not**: OpenCV computes its own from the
/// 8-bit image with a Sobel-like scheme, and this takes binCV's already-computed
/// `SignedQuantMat` derivatives -- which is D-5 (a kernel binds to views, and the
/// frontend has these already) and also the only shape that avoids materialising an
/// 8-bit image the library exists to avoid.

#include <cmath>
#include <cstddef>

#include "../quantMat.hpp"
#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief `cv::cornerSubPix`'s `winSize`, `zeroZone` and `criteria`, in one struct.
struct SubPixParams {
    /// Half-width of the search window; the window is `(2*winHalf + 1)` square.
    /// OpenCV's `winSize(5,5)` -- ORB-SLAM's and HybVIO's choice -- is `winHalf = 5`.
    int winHalf = 5;
    /// Half-width of a central region excluded from the sums, or **-1 for none**.
    /// OpenCV's `zeroZone`. It exists to avoid a singular `G` when the autocorrelation
    /// matrix has a large central peak; -1 (none) is what most callers pass.
    int zeroHalf = -1;
    int maxIterations = 40;      ///< `TermCriteria::MAX_ITER`
    double epsilon = 0.001;      ///< `TermCriteria::EPS`, in pixels, on the step
};

/// @brief What `cornerSubPix` did, per corner. **API TIER 3** -- OpenCV reports nothing.
struct SubPixResult {
    size_t refined = 0;    ///< corners whose position moved and converged
    size_t singular = 0;   ///< corners left untouched because `G` was not invertible
    size_t clamped = 0;    ///< corners whose step left the window and were left in place

    /// @brief Corners whose refined position ended up further than `winHalf` from where
    ///        it started, and were therefore REVERTED to the input.
    ///
    /// **This is `cv::cornerSubPix`'s own rule, not an addition** -- "if new point is too
    /// far from initial, it means poor convergence; leave initial point as the result",
    /// tested on `|dx| > win.width || |dy| > win.height` after the loop. binCV did not
    /// implement it until F-4, and the difference is not subtle: a seed that walks out of
    /// its own window is exactly the case where the two answers diverge by more than the
    /// window is wide.
    size_t diverged = 0;
};

namespace impl {

/// @brief OpenCV's Gaussian window mask, built once per call. **INTERNAL.**
///
/// @note **`exp(-(dx^2 + dy^2) / winHalf^2)`, and the width is not a free parameter.**
///       `cv::cornerSubPix` normalises each offset by the half-window and exponentiates
///       the sum of squares -- `vy = exp(-y*y)` with `y = (i - win.height)/win.height`,
///       times the same in x -- so the weight is exactly 1/e at the edge of the window
///       along either axis.
/// @note **THIS WAS WRONG UNTIL F-4, BY A FACTOR OF TWO IN THE EXPONENT** -- the
///       denominator read `2*(winHalf/2)^2 = winHalf^2/2`, giving `exp(-2r^2/winHalf^2)`,
///       a Gaussian sqrt(2) too narrow. Reported from outside at 4.53 px mean against
///       OpenCV on real frames.
/// @note **THE TEST SAW IT AND PASSED ANYWAY, AND THAT IS THE PART WORTH REMEMBERING.**
///       `SubPix.AgreesWithOpenCVOnTheSameCorner` measured **0.0325 px with the wrong
///       mask and 0.0035 px with the right one** -- it was sensitive, by a factor of
///       nine. It passed because its bound was 0.1, chosen from the number the code
///       happened to produce rather than from what a correct implementation reaches. A
///       tolerance fitted to the observed value cannot fail; it can only record.
inline void subPixMask(int winHalf, int zeroHalf, double* mask) {
    const int side = 2 * winHalf + 1;
    const double denom = static_cast<double>(winHalf) * static_cast<double>(winHalf);
    for (int dy = -winHalf; dy <= winHalf; ++dy) {
        for (int dx = -winHalf; dx <= winHalf; ++dx) {
            const double r = static_cast<double>(dx * dx + dy * dy);
            double w = (denom > 0.0) ? std::exp(-r / denom) : 1.0;
            // The zero zone is a HOLE, not a smaller window: OpenCV keeps the outer
            // ring and drops the middle.
            if (zeroHalf >= 0 && dx >= -zeroHalf && dx <= zeroHalf && dy >= -zeroHalf &&
                dy <= zeroHalf) {
                w = 0.0;
            }
            mask[(dy + winHalf) * side + (dx + winHalf)] = w;
        }
    }
}

/// The largest window this operation will build a mask for, so the mask is a stack
/// buffer and the kernel allocates nothing (CLAUDE.md's hard rule).
inline constexpr int kMaxWinHalf = 15;

}  // namespace impl

/// @brief Refines corner positions to sub-pixel accuracy. **API TIER 2.**
///
/// @param dx,dy The previous frame's signed derivatives -- exactly what
///        `ops/covariance.hpp` and the tracker already consume.
/// @param corners In/out. Positions are refined in place; a corner whose window leaves
///        the image, or whose `G` is singular, is **left exactly where it was**.
/// @param count Number of corners.
///
/// @note **Never allocates and never throws.** The Gaussian mask is a stack buffer
///       bounded by `impl::kMaxWinHalf`.
/// @note A corner is refined **independently of every other**, so this splits over
///       `parallelFor` exactly as tracking does. It is not split here because the
///       operation is a few microseconds per frame at realistic corner counts; measure
///       before adding it.
/// @note The step is computed in `double` and the position stored as `float`, matching
///       the tracker's convention and OpenCV's.
template <size_t N, typename WordType>
inline SubPixResult cornerSubPix(const SignedQuantMat<N, WordType>& dx,
                                 const SignedQuantMat<N, WordType>& dy, Point2f* corners,
                                 size_t count, const SubPixParams& params = SubPixParams()) {
    SubPixResult out;
    BINCV_ASSERT(params.winHalf >= 1 && params.winHalf <= impl::kMaxWinHalf,
                 "cornerSubPix: winHalf outside [1, kMaxWinHalf]");
    BINCV_ASSERT(corners != nullptr || count == 0,
                 "cornerSubPix: a non-zero count needs a corner array");
    if (count == 0 || params.winHalf < 1 || params.winHalf > impl::kMaxWinHalf) return out;

    const int winHalf = params.winHalf;
    const int side = 2 * winHalf + 1;
    double mask[(2 * impl::kMaxWinHalf + 1) * (2 * impl::kMaxWinHalf + 1)];
    impl::subPixMask(winHalf, params.zeroHalf, mask);

    const size_t width = dx.getWidth();
    const size_t height = dx.getHeight();
    const double eps2 = params.epsilon * params.epsilon;

    for (size_t c = 0; c < count; ++c) {
        const double startX = static_cast<double>(corners[c].x);
        const double startY = static_cast<double>(corners[c].y);
        double cx = startX;
        double cy = startY;

        for (int it = 0; it < params.maxIterations; ++it) {
            // THE WINDOW IS ANCHORED ON THE ROUNDED POSITION, AND THAT IS A DEVIATION.
            // A bit-plane derivative cannot be interpolated -- the popcount identity is
            // exact only for {-1, 0, +1} -- so the samples are integer pixels and the
            // refinement is the offset within them. This comment used to claim the
            // rounding was "as OpenCV's is"; it is not, and saying so hid a real
            // difference behind a false reassurance.
            const long long ix0 = static_cast<long long>(std::floor(cx + 0.5));
            const long long iy0 = static_cast<long long>(std::floor(cy + 0.5));
            if (ix0 - winHalf < 0 || iy0 - winHalf < 0 ||
                ix0 + winHalf >= static_cast<long long>(width) ||
                iy0 + winHalf >= static_cast<long long>(height)) {
                ++out.clamped;
                break;
            }

            double gxx = 0.0, gxy = 0.0, gyy = 0.0, bx = 0.0, by = 0.0;
            constexpr size_t kBits = impl::bitsPerWord<WordType>();
            const size_t xLo = static_cast<size_t>(ix0 - winHalf);
            const size_t xHi = static_cast<size_t>(ix0 + winHalf);   // inclusive

            for (int wy = -winHalf; wy <= winHalf; ++wy) {
                const size_t y = static_cast<size_t>(iy0 + wy);
                const double* mrow = mask + (wy + winHalf) * side;

                const WordType* mx[N];
                const WordType* my[N];
                for (size_t j = 0; j < N; ++j) {
                    mx[j] = dx.constMagnitude(j).row(y);
                    my[j] = dy.constMagnitude(j).row(y);
                }
                const WordType* sx = dx.constSign().row(y);
                const WordType* sy = dy.constSign().row(y);

                for (size_t wi = xLo / kBits; wi <= xHi / kBits; ++wi) {
                    // WHICH PIXELS IN THIS WORD CAN CONTRIBUTE AT ALL, A WORD AT A
                    // TIME. A pixel with zero gradient in both axes adds nothing to any
                    // of the five sums, and on a binarised edge map most of a 31x31
                    // window is exactly that. This is what bit-planes buy in an
                    // operation that cannot be reduced to popcounts.
                    WordType nz = 0;
                    for (size_t j = 0; j < N; ++j) {
                        nz = static_cast<WordType>(nz | mx[j][wi] | my[j][wi]);
                    }
                    // Trim to the window's columns; `wi` may span either end.
                    const size_t lo = (wi * kBits > xLo) ? 0 : (xLo - wi * kBits);
                    const size_t hiExcl =
                        (xHi >= (wi + 1) * kBits) ? kBits : (xHi - wi * kBits + 1);
                    if (lo != 0) {
                        nz = static_cast<WordType>(nz & ~impl::lowBitsMask<WordType>(lo));
                    }
                    if (hiExcl < kBits) {
                        nz = static_cast<WordType>(nz & impl::lowBitsMask<WordType>(hiExcl));
                    }

                    while (nz != 0) {
                        const size_t b = static_cast<size_t>(
                            __builtin_ctzll(static_cast<unsigned long long>(nz)));
                        nz = static_cast<WordType>(nz & (nz - 1));
                        const size_t x = wi * kBits + b;
                        const int wx = static_cast<int>(static_cast<long long>(x) - ix0);
                        const double w = mrow[wx + winHalf];
                        if (w == 0.0) continue;   // the zero zone

                        // The signed value, assembled from the planes already loaded --
                        // `at()` would re-derive the row pointers for every pixel.
                        const WordType bit = static_cast<WordType>(WordType{1} << b);
                        long long vx = 0, vy = 0;
                        for (size_t j = 0; j < N; ++j) {
                            if (mx[j][wi] & bit) vx |= (1LL << j);
                            if (my[j][wi] & bit) vy |= (1LL << j);
                        }
                        if (sx[wi] & bit) vx = -vx;
                        if (sy[wi] & bit) vy = -vy;

                        const double gx = static_cast<double>(vx);
                        const double gy = static_cast<double>(vy);
                        const double xx = w * gx * gx;
                        const double xy = w * gx * gy;
                        const double yy = w * gy * gy;
                        // `p` is the sample's offset from the window centre, so the
                        // solve returns an offset and nothing large is accumulated.
                        const double px = static_cast<double>(wx);
                        const double py = static_cast<double>(wy);
                        gxx += xx;
                        gxy += xy;
                        gyy += yy;
                        bx += xx * px + xy * py;
                        by += xy * px + yy * py;
                    }
                }
            }

            const double det = gxx * gyy - gxy * gxy;
            if (det == 0.0) {
                ++out.singular;
                break;
            }
            // q = G^-1 b, as an offset from the window centre.
            const double qx = (gyy * bx - gxy * by) / det;
            const double qy = (gxx * by - gxy * bx) / det;
            // OUTSIDE THE WINDOW IS A DIVERGENCE, NOT AN ANSWER. OpenCV stops when the
            // step leaves the window; a corner that wants to move further than the
            // evidence extends has not been refined, it has been lost.
            if (std::fabs(qx) > static_cast<double>(winHalf) ||
                std::fabs(qy) > static_cast<double>(winHalf)) {
                ++out.clamped;
                break;
            }

            const double nx = static_cast<double>(ix0) + qx;
            const double ny = static_cast<double>(iy0) + qy;
            const double stepX = nx - cx;
            const double stepY = ny - cy;
            cx = nx;
            cy = ny;
            corners[c].x = static_cast<float>(cx);
            corners[c].y = static_cast<float>(cy);
            if (stepX * stepX + stepY * stepY <= eps2) {
                ++out.refined;
                break;
            }
            if (it + 1 == params.maxIterations) ++out.refined;
        }

        // POOR CONVERGENCE -- cv::cornerSubPix's rule, applied after the loop exactly as
        // it applies it. A point that has walked further than its own half-window from
        // the seed has not been refined, it has been captured by different structure, and
        // the seed is the better answer. Measured on asymmetric content: without this the
        // mean disagreement with cv::cornerSubPix is 2.02 px and the worst is 5.54 px,
        // and BOTH numbers come from the handful of seeds this rule rejects.
        if (std::fabs(cx - startX) > static_cast<double>(winHalf) ||
            std::fabs(cy - startY) > static_cast<double>(winHalf)) {
            corners[c].x = static_cast<float>(startX);
            corners[c].y = static_cast<float>(startY);
            ++out.diverged;
        }
    }
    return out;
}

}  // inline namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
