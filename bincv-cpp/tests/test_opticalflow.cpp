// Hybrid Lucas-Kanade keypoint tracking (T3.8) -- ops/opticalFlow.hpp.
//
// WHAT THIS SUITE HAS TO STAND BEHIND, GIVEN THAT NOTHING HERE IS BIT-EXACT
// AGAINST ANYTHING
//
// The operation is API TIER 2 and there is no denominator to be exact against.
// `cv::calcOpticalFlowPyrLK` warps and interpolates BOTH windows over CV_8U
// content; this one anchors the previous window on the integer grid and carries
// the whole subpixel displacement on the next-frame side, over 1-bit content
// (ops/opticalFlow.hpp, deviation (i)). An exact comparison is not available even
// in principle. Four independent things stand in its place.
//
//   1. GROUND TRUTH, NOT ANOTHER ESTIMATOR. Every accuracy case is a SYNTHETIC
//      WARP of a CONTINUOUS field: `frame1(z) = [f(A^-1 z) > 0]` where `frame0(z)
//      = [f(z) > 0]`. The displacement of every point is then EXACTLY `A z - z`,
//      analytically, with no resampling anywhere -- the binarization is applied to
//      the warped continuous field rather than to a resampled binary image, which
//      is the only way to warp binary content without inventing information. The
//      harness was written before the tracker was pointed at it.
//   2. AN INDEPENDENT PER-PIXEL FLOAT IMPLEMENTATION of the same documented
//      algorithm, sharing no code with the kernel: it reads `dx.at()` / `dy.at()`
//      one pixel at a time, bilinearly interpolates the next frame one pixel at a
//      time in `double`, and accumulates `xx/yy/xy` and `b1/b2` with multiplies.
//      That is the formulation the popcount identity of ops/opticalFlow.hpp is
//      asserted to REPLACE, so agreement between the two is the whole
//      bit-parallel claim -- checked both at the level of one residual (tight, at
//      many offsets) and at the level of a whole tracked frame.
//   3. WORD-TYPE INVARIANCE. The same logical frames at uint8_t, uint16_t,
//      uint32_t and uint64_t must give BIT-IDENTICAL flow. Every quantity inside
//      is an exact integer count or a double derived from one, so any difference
//      is a word-boundary bug -- and the cross-word tap shift is where such a bug
//      would live.
//   4. THE LOSS RULES, driven to fire. A blank frame, a window off the edge, an
//      estimate that walks out of range.
//
// ===========================================================================
// THE TOLERANCE, STATED BEFORE ANY ERROR WAS MEASURED, WITH ITS DERIVATION
// ===========================================================================
// A tolerance chosen after seeing the errors is not a tolerance. These four
// numbers come from properties of the REPRESENTATION, not from a run:
//
//   T1. RMS endpoint error <= 0.25 px.
//       A binarized frame locates each edge crossing to within +/-0.5 px -- that
//       is the quantization step of a 1-bit representation, and no estimator
//       reading only binarized content can do better on a SINGLE crossing. A 31x31
//       window on textured content contains many crossings at many orientations,
//       and the least-squares fit averages them, so the aggregate error must beat
//       the single-crossing bound. A factor of two is the modest form of that
//       claim: it asserts an effective count of only four independent crossings
//       per window.
//   T2. Maximum endpoint error <= 1.0 px.
//       ONE WHOLE PIXEL OF THE GRID THE ESTIMATE IS READ OFF. The same +/-0.5 px
//       per-axis localization bound as T1, doubled: T1 says the AGGREGATE must
//       beat the single-crossing bound by a factor of two, T2 says no SINGLE point
//       may miss it by more than a factor of two. A point worse than one whole
//       pixel is not a noisy estimate of the right displacement, it is a different
//       displacement.
//       AN EARLIER VERSION OF THIS FILE JUSTIFIED THE SAME 1.0 px BY ROUTE (a) --
//       "integer-only census/Hamming matching gives 1.0 px, and a subpixel tracker
//       may never be worse". That justification was wrong twice over and is
//       withdrawn: a minimizing integer matcher returns `round(d)`, not
//       `floor(d)`, so its error is `min(q, 1-q) <= 0.5` per axis rather than 1.0;
//       and binCV contains no route (a) implementation to have measured, since
//       route (a) is E-6 / T4.2. The NUMBER is unchanged -- it was not refitted --
//       only the derivation, which now rests on the representation alone.
//   T3. At least 80% of eligible points tracked, AND NO TRACKED POINT MAY BE
//       STUCK. `status == 1` is not evidence of tracking on its own: on the real
//       frame every one of 141 points comes back tracked, including ones that
//       returned EXACTLY their input position while ground truth moved by 1.4 px.
//       A point is STUCK when ground truth moved it by at least 0.5 px -- the
//       1-bit localization bound, i.e. a motion the representation can resolve at
//       all -- and the tracker reports a total displacement no larger than
//       `lk_term_criteria_eps` (0.03 px), the step size at which the iteration
//       declares itself converged. Such a point never moved; counting it as
//       tracked is what made an 80% rule vacuous.
//   T4. THE CRITERION THAT SEPARATES THIS OPERATION FROM AN INTEGER-GRID ONE. A
//       tracker whose displacements are whole pixels can only return the nearest
//       integer, so on a translation with fractional part `q` its endpoint error
//       is `min(q, 1-q)` per axis by construction -- 0.25 at q = 0.25, 0.50 at
//       q = 0.50, 0.25 at q = 0.75. The RMS error here must be strictly below
//       that, otherwise the subpixel machinery -- which is the entire cost of
//       route (b) -- has bought nothing. This is a property of the integer grid,
//       derived, and NOT a measurement of route (a): no census/Hamming matcher is
//       implemented or run here.
//
// AND ONE ALLOWANCE, ALSO DERIVED RATHER THAN FITTED. LK's model is a pure
// translation of the window. Under a rotation `theta` or a scale `s` the true
// displacement varies ACROSS the window, by up to `halfWin * theta` and
// `halfWin * |s - 1|` respectively at the window's corners. That is model error,
// not estimator error, and it is added to T1 and T2 for those two cases only.
// At 31x31 and 1 degree it is 0.26 px; at 31x31 and 1.02x it is 0.30 px.
//
// ELIGIBILITY, ALSO FIXED IN ADVANCE. A point is eligible when its 31x31 window
// AND its ground-truth destination window both lie fully inside the frame. Near
// the border the two frames do not contain the same content -- material enters
// and leaves -- so "ground truth" there is not ground truth. The reference fills
// that region by reflecting the frame into a padded copy, which is not ground
// truth either; ops/opticalFlow.hpp declines the padded copy (deviation (ii)).
//
// ===========================================================================
// THE FOOTPRINT CASE
// ===========================================================================
// `Flow.FrontendFootprint_640x480` is the number Phase 4 needs and the project's
// memory claim rests on: peak working set of denoise -> pyramid -> derivative ->
// corner -> track, at 640x480, broken down by stage, with every allocation
// counted rather than estimated. E-10 predicts the float response map dominates.
// The case prints the table and pins the dominant term, so a future change that
// moves it fails here rather than in a report nobody re-ran.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#ifdef BINCV_WITH_OPENCV
#include <filesystem>

#include <opencv2/opencv.hpp>
#endif

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/core/types.hpp"
#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/covariance.hpp"
#include "bincv-cpp/ops/denoise.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// The allocation counter, in the idiom tests/test_storage.cpp established and
// tests/test_corner.cpp reuses -- the C++17 OVER-ALIGNED forms included, which a
// counter replacing only the plain pair cannot see.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_newCount = 0;
// LIVE BYTES AND THEIR HIGH-WATER MARK, not just a call count. A counter of
// `operator new` CALLS cannot say what a stage's peak is; X-23's frontend table
// used to add up the buffers its author had listed, so a buffer nobody listed --
// including one acquired inside a kernel -- could not move the number. These two
// make the total a READING: every allocation adds its REQUESTED size, every free
// subtracts it, and `g_peakBytes` records the largest live total since it was last
// armed. The requested size is recoverable at free time because every block
// carries a header (below), which is why this replaces the plain `malloc` form.
std::size_t g_liveBytes = 0;
std::size_t g_peakBytes = 0;

// {requested bytes, base pointer} written immediately below the returned address.
// 16 B on a 64-bit target, and every block is therefore allocated with alignment
// at least that, so the header never breaks the alignment the caller asked for.
constexpr std::size_t kAllocHeader = 2 * sizeof(void*);

void* countedAllocateAligned(std::size_t bytes, std::size_t alignment) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    if (alignment < kAllocHeader) alignment = kAllocHeader;
    const std::size_t wanted = (bytes == 0) ? 1 : bytes;
    const std::size_t rounded = ((wanted + alignment - 1) / alignment) * alignment;
    void* base = std::aligned_alloc(alignment, rounded + alignment);
    if (base == nullptr) std::abort();
    char* ret = static_cast<char*>(base) + alignment;
    std::memcpy(ret - sizeof(void*), &base, sizeof(void*));
    std::memcpy(ret - kAllocHeader, &wanted, sizeof(std::size_t));
    g_liveBytes += wanted;
    if (g_liveBytes > g_peakBytes) g_peakBytes = g_liveBytes;
    return ret;
}

void* countedAllocate(std::size_t bytes) {
    return countedAllocateAligned(bytes, alignof(std::max_align_t));
}

void countedFree(void* p) noexcept {
    if (p == nullptr) return;
    char* ret = static_cast<char*>(p);
    void* base = nullptr;
    std::size_t wanted = 0;
    std::memcpy(&base, ret - sizeof(void*), sizeof(void*));
    std::memcpy(&wanted, ret - kAllocHeader, sizeof(std::size_t));
    g_liveBytes -= wanted;
    std::free(base);
}

/// @brief Arm the high-water mark at the current live total.
void armPeak() { g_peakBytes = g_liveBytes; }
} // namespace

void* operator new(std::size_t bytes)   { return countedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return countedAllocate(bytes); }
void operator delete(void* p) noexcept                { countedFree(p); }
void operator delete[](void* p) noexcept              { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept   { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept { countedFree(p); }

void* operator new(std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t bytes, std::align_val_t a) {
    return countedAllocateAligned(bytes, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::align_val_t) noexcept                { countedFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept              { countedFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept   { countedFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { countedFree(p); }

namespace {

using bincv::BinMat;
using bincv::Corner;
using bincv::CornerResult;
using bincv::GoodFeaturesParams;
using bincv::LKLevel;
using bincv::LKParams;
using bincv::Point2f;
using bincv::Rect;
using bincv::ResponseMap;
using bincv::TernaryMat;

// ---------------------------------------------------------------------------
// THE TOLERANCE. Named here, once, so that no case can quietly use a different
// one. See the derivation at the top of the file.
// ---------------------------------------------------------------------------
constexpr double kRmsTolerance = 0.25;     ///< T1, pixels
constexpr double kMaxTolerance = 1.00;     ///< T2, pixels
constexpr double kMinTrackedFraction = 0.80;  ///< T3
/// T3's second half. Ground truth moved by at least this much -- the 1-bit
/// localization bound, the smallest motion the representation resolves at all...
constexpr double kTruthMoved = 0.50;
/// ...and the tracker moved by no more than this: `lk_term_criteria_eps`, the step
/// at which the iteration calls itself converged. Both derived, neither fitted.
constexpr double kStuckFlow = 0.03;

// ---------------------------------------------------------------------------
// THE GROUND-TRUTH WARP HARNESS
//
// A continuous scalar field, sampled through the INVERSE warp and thresholded.
// Nothing is resampled and nothing is interpolated: `frame1` is the binarization
// of the warped field, not a warp of the binarization. Ground truth is therefore
// the warp's own arithmetic, exact to double precision.
// ---------------------------------------------------------------------------

/// @brief The texture. Four incommensurate terms at four orientations, so that
///        the binarized level set has edge crossings at many angles and the 2x2
///        covariance is well conditioned somewhere -- which is what makes a
///        corner detector produce points at all.
double field(double x, double y) {
    return std::sin(x / 7.3 + 0.4) * std::cos(y / 5.1)
         + 0.6 * std::sin((x + y) / 11.7)
         + 0.5 * std::cos((x - 2.0 * y) / 9.3)
         + 0.4 * std::sin(x / 3.1) * std::sin(y / 3.7);
}

/// @brief An affine warp about a centre: `A (z - c) + c + t`.
struct Warp {
    double m00 = 1.0, m01 = 0.0, m10 = 0.0, m11 = 1.0;
    double tx = 0.0, ty = 0.0;
    double cx = 0.0, cy = 0.0;

    void forward(double x, double y, double& ox, double& oy) const {
        const double dx = x - cx, dy = y - cy;
        ox = m00 * dx + m01 * dy + cx + tx;
        oy = m10 * dx + m11 * dy + cy + ty;
    }
    /// @note The inverse is written out rather than solved numerically: a 2x2
    ///       inverse is four lines and a numerical solve would make the ground
    ///       truth depend on a solver's accuracy.
    void inverse(double x, double y, double& ox, double& oy) const {
        const double det = m00 * m11 - m01 * m10;
        const double i00 =  m11 / det, i01 = -m01 / det;
        const double i10 = -m10 / det, i11 =  m00 / det;
        const double dx = x - cx - tx, dy = y - cy - ty;
        ox = i00 * dx + i01 * dy + cx;
        oy = i10 * dx + i11 * dy + cy;
    }
};

Warp translation(double tx, double ty) {
    Warp w; w.tx = tx; w.ty = ty; return w;
}
Warp rotation(double degrees, double cx, double cy) {
    Warp w;
    const double r = degrees * 3.14159265358979323846 / 180.0;
    w.m00 = std::cos(r); w.m01 = -std::sin(r);
    w.m10 = std::sin(r); w.m11 =  std::cos(r);
    w.cx = cx; w.cy = cy;
    return w;
}
Warp scaling(double s, double cx, double cy) {
    Warp w; w.m00 = s; w.m11 = s; w.cx = cx; w.cy = cy; return w;
}

/// @brief `dst(z) = [field(warp^-1 z) > 0]`.
template <typename WordType>
void renderWarped(BinMat<WordType>& dst, const Warp& warp) {
    for (int y = 0; y < dst.rows(); ++y) {
        for (int x = 0; x < dst.cols(); ++x) {
            double sx = 0.0, sy = 0.0;
            warp.inverse(static_cast<double>(x), static_cast<double>(y), sx, sy);
            dst.set(y, x, field(sx, sy) > 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// The pyramid bundle the tracker takes. Every level is ONE bit deep, because the
// popcount covariance is exact only for a ternary derivative
// (ops/opticalFlow.hpp, deviation (v)).
// ---------------------------------------------------------------------------
template <typename WordType>
struct Frontend {
    std::vector<BinMat<WordType>> prev, next;
    std::vector<TernaryMat<WordType>> dx, dy;
    std::vector<LKLevel<WordType>> levels;

    Frontend(int width, int height, int levelCount) {
        int w = width, h = height;
        for (int i = 0; i < levelCount; ++i) {
            prev.emplace_back(w, h);
            next.emplace_back(w, h);
            dx.emplace_back(w, h);
            dy.emplace_back(w, h);
            w = static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w)));
            h = static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h)));
        }
    }

    /// @brief The two KERNEL stages: pyrDown down both ladders, then the ternary
    ///        derivative of every previous level. Allocates nothing.
    void runKernels() {
        for (size_t i = 1; i < prev.size(); ++i) {
            bincv::pyrDown<1, 1, WordType>(prev[i - 1], prev[i]);
            bincv::pyrDown<1, 1, WordType>(next[i - 1], next[i]);
        }
        for (size_t i = 0; i < prev.size(); ++i) {
            bincv::derivativeX(prev[i], dx[i]);
            bincv::derivativeY(prev[i], dy[i]);
        }
    }

    /// @brief Names the planes into the view bundles. The vector is the TEST's,
    ///        not a kernel buffer -- kept out of runKernels() so that the
    ///        allocation count around the kernels is a reading of the kernels.
    void bindLevels() {
        levels.clear();
        levels.reserve(prev.size());
        for (size_t i = 0; i < prev.size(); ++i) {
            levels.push_back(bincv::lkLevel(prev[i], next[i], dx[i], dy[i]));
        }
    }

    /// @brief Everything downstream of level 0 of both frames.
    void build() {
        runKernels();
        bindLevels();
    }
};

// ---------------------------------------------------------------------------
// THE INDEPENDENT PER-PIXEL FLOAT IMPLEMENTATION
//
// The formulation ops/opticalFlow.hpp's popcount identity replaces: one multiply
// and one bilinear interpolation per pixel per iteration, accumulated in double.
// It shares no code with the kernel -- not the covariance, not the residual, not
// the tap extraction. It reproduces the same DOCUMENTED algorithm (integer-anchored
// previous window, clipped window, replicated next-frame taps, the same
// termination rules), because the point is to check the arithmetic, not to
// re-derive the design.
// ---------------------------------------------------------------------------

template <typename WordType>
double refPixel(const BinMat<WordType>& m, long long x, long long y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= m.cols()) x = m.cols() - 1;
    if (y >= m.rows()) y = m.rows() - 1;
    return m.at(static_cast<int>(y), static_cast<int>(x)) ? 1.0 : 0.0;
}

template <typename WordType>
double refBilinear(const BinMat<WordType>& m, double x, double y) {
    const long long ix = static_cast<long long>(std::floor(x));
    const long long iy = static_cast<long long>(std::floor(y));
    const double a = x - static_cast<double>(ix);
    const double b = y - static_cast<double>(iy);
    return (1.0 - a) * (1.0 - b) * refPixel(m, ix, iy)
         + a * (1.0 - b) * refPixel(m, ix + 1, iy)
         + (1.0 - a) * b * refPixel(m, ix, iy + 1)
         + a * b * refPixel(m, ix + 1, iy + 1);
}

/// @brief The window's clipped pixel extent, computed independently of
///        impl::clipRegion.
struct Extent { long long x0, x1, y0, y1; };

Extent refClip(long long ax, long long ay, int winW, int winH, long long width,
               long long height) {
    Extent e;
    e.x0 = std::max<long long>(ax, 0);
    e.y0 = std::max<long long>(ay, 0);
    e.x1 = std::min<long long>(ax + winW, width);
    e.y1 = std::min<long long>(ay + winH, height);
    return e;
}

/// @brief `b1`, `b2` by per-pixel multiply-accumulate.
template <typename WordType>
void refResidual(const BinMat<WordType>& prev, const BinMat<WordType>& next,
                 const TernaryMat<WordType>& dx, const TernaryMat<WordType>& dy,
                 const Extent& e, double offX, double offY, double& b1, double& b2) {
    b1 = 0.0;
    b2 = 0.0;
    for (long long y = e.y0; y < e.y1; ++y) {
        for (long long x = e.x0; x < e.x1; ++x) {
            const double gx = static_cast<double>(dx.at(static_cast<int>(y), static_cast<int>(x)));
            const double gy = static_cast<double>(dy.at(static_cast<int>(y), static_cast<int>(x)));
            if (gx == 0.0 && gy == 0.0) continue;
            const double jv = refBilinear(next, static_cast<double>(x) + offX,
                                          static_cast<double>(y) + offY);
            const double iv = prev.at(static_cast<int>(y), static_cast<int>(x)) ? 1.0 : 0.0;
            const double diff = jv - iv;
            b1 += diff * gx;
            b2 += diff * gy;
        }
    }
}

/// @brief `mean |J(z + off) - I(z)|` over the clipped window, per pixel, in double.
/// @note The quantity `impl::windowMeanAbsDiff` computes from popcounts. Written
///       here as the multiply-free definition -- one bilinear interpolation and
///       one absolute value per pixel -- so that the collapse
///       `|Jinterp - I| = I + (1 - 2I)*Jinterp` is checked rather than asserted.
template <typename WordType>
double refMeanAbsDiff(const BinMat<WordType>& prev, const BinMat<WordType>& next, const Extent& e,
                      double offX, double offY) {
    double sum = 0.0;
    size_t n = 0;
    for (long long y = e.y0; y < e.y1; ++y) {
        for (long long x = e.x0; x < e.x1; ++x) {
            const double jv = refBilinear(next, static_cast<double>(x) + offX,
                                          static_cast<double>(y) + offY);
            const double iv = prev.at(static_cast<int>(y), static_cast<int>(x)) ? 1.0 : 0.0;
            sum += std::fabs(jv - iv);
            ++n;
        }
    }
    return (n > 0) ? sum / static_cast<double>(n) : 0.0;
}

/// @brief How many of `levelCount` levels the tracker will actually use, by the
///        reference's own rule: a level at or below the window size ends the
///        pyramid (ops/opticalFlow.hpp, deviation (vi)). Computed here from the
///        level DIMENSIONS, independently of the kernel's own loop.
template <typename WordType>
size_t usableLevelCount(const Frontend<WordType>& fe, int winW, int winH) {
    size_t n = 1;
    while (n < fe.prev.size() && fe.prev[n].cols() > winW && fe.prev[n].rows() > winH) ++n;
    return n;
}

/// @brief The whole tracker, per pixel, in double.
template <typename WordType>
void refTrack(const Frontend<WordType>& fe, const std::vector<Point2f>& prevPts,
              std::vector<Point2f>& nextPts, std::vector<uint8_t>& status,
              const LKParams& params) {
    const size_t n = prevPts.size();
    nextPts.assign(n, Point2f{});
    status.assign(n, uint8_t{1});
    const int winW = params.winWidth, winH = params.winHeight;
    const float halfX = static_cast<float>(winW - 1) * 0.5f;
    const float halfY = static_cast<float>(winH - 1) * 0.5f;
    const double eps2 = static_cast<double>(params.epsilon) * static_cast<double>(params.epsilon);
    // Deviation (vi), reproduced independently: levels at or below the window are
    // not used. Derived from the level sizes here, not read off the kernel.
    const size_t levelCount = usableLevelCount(fe, winW, winH);

    for (size_t li = levelCount; li-- > 0;) {
        const BinMat<WordType>& P = fe.prev[li];
        const BinMat<WordType>& N = fe.next[li];
        const TernaryMat<WordType>& DX = fe.dx[li];
        const TernaryMat<WordType>& DY = fe.dy[li];
        const float scale = 1.0f / static_cast<float>(1u << li);
        const bool coarsest = (li + 1 == levelCount);
        const bool finest = (li == 0);
        const long long W = static_cast<long long>(P.cols());
        const long long H = static_cast<long long>(P.rows());

        for (size_t p = 0; p < n; ++p) {
            if (coarsest) {
                nextPts[p].x = prevPts[p].x * scale;
                nextPts[p].y = prevPts[p].y * scale;
            } else {
                nextPts[p].x *= 2.0f;
                nextPts[p].y *= 2.0f;
            }
        }

        for (size_t p = 0; p < n; ++p) {
            const float px = prevPts[p].x * scale - halfX;
            const float py = prevPts[p].y * scale - halfY;
            const long long ax = static_cast<long long>(std::floor(px));
            const long long ay = static_cast<long long>(std::floor(py));
            if (ax < -winW || ax >= W || ay < -winH || ay >= H) {
                if (finest) status[p] = 0;
                continue;
            }
            const Extent e = refClip(ax, ay, winW, winH, W, H);
            if (e.x0 >= e.x1 || e.y0 >= e.y1) {
                if (finest) status[p] = 0;
                continue;
            }
            double xx = 0.0, yy = 0.0, xy = 0.0;
            for (long long y = e.y0; y < e.y1; ++y) {
                for (long long x = e.x0; x < e.x1; ++x) {
                    const double gx =
                        static_cast<double>(DX.at(static_cast<int>(y), static_cast<int>(x)));
                    const double gy =
                        static_cast<double>(DY.at(static_cast<int>(y), static_cast<int>(x)));
                    xx += gx * gx;
                    yy += gy * gy;
                    xy += gx * gy;
                }
            }
            const double det = xx * yy - xy * xy;
            const double s = xx + yy;
            const double d = xx - yy;
            const double lambda = 0.5 * (s - std::sqrt(d * d + 4.0 * xy * xy));
            const double refEig = bincv::impl::kReferenceMinEigScale *
                                  static_cast<double>(static_cast<float>(lambda)) /
                                  static_cast<double>(winW * winH);
            if (det <= 0.0 || refEig < static_cast<double>(params.minEigThreshold)) {
                if (finest) status[p] = 0;
                continue;
            }
            float nx = nextPts[p].x - halfX;
            float ny = nextPts[p].y - halfY;
            double pdx = 0.0, pdy = 0.0;
            for (int it = 0; it < params.maxIterations; ++it) {
                const long long ox = static_cast<long long>(std::floor(nx));
                const long long oy = static_cast<long long>(std::floor(ny));
                if (ox < -winW || ox >= W || oy < -winH || oy >= H) {
                    if (finest) status[p] = 0;
                    break;
                }
                // From `px`/`py`, not from the integer anchor -- see the note in
                // ops/opticalFlow.hpp. The two differ by frac(prevPt) at every
                // level above 0.
                const double offX = static_cast<double>(nx) - static_cast<double>(px);
                const double offY = static_cast<double>(ny) - static_cast<double>(py);
                double b1 = 0.0, b2 = 0.0;
                refResidual(P, N, DX, DY, e, offX, offY, b1, b2);
                const double dX = 2.0 * (xy * b2 - yy * b1) / det;
                const double dY = 2.0 * (xy * b1 - xx * b2) / det;
                nx += static_cast<float>(dX);
                ny += static_cast<float>(dY);
                nextPts[p].x = nx + halfX;
                nextPts[p].y = ny + halfY;
                if (dX * dX + dY * dY <= eps2) break;
                if (it > 0 && std::fabs(dX + pdx) < 0.01 && std::fabs(dY + pdy) < 0.01) {
                    nextPts[p].x -= static_cast<float>(dX * 0.5);
                    nextPts[p].y -= static_cast<float>(dY * 0.5);
                    nx = nextPts[p].x - halfX;
                    ny = nextPts[p].y - halfY;
                    break;
                }
                pdx = dX;
                pdy = dY;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Point selection and error accounting
// ---------------------------------------------------------------------------

/// @brief Corners of `dx`/`dy` that are ELIGIBLE under the rule fixed at the top
///        of the file: this window and the ground-truth destination window both
///        fully inside the frame.
template <typename WordType>
std::vector<Point2f> eligiblePoints(const TernaryMat<WordType>& dx,
                                    const TernaryMat<WordType>& dy, int width, int height,
                                    const Warp& warp, int winW, int winH) {
    std::vector<float> mapStorage(static_cast<size_t>(width) * static_cast<size_t>(height), 0.0f);
    ResponseMap map{mapStorage.data(), static_cast<size_t>(width), static_cast<size_t>(height),
                    static_cast<size_t>(width)};
    std::vector<Corner> found(static_cast<size_t>(width) * static_cast<size_t>(height));
    GoodFeaturesParams params;  // seal_params.yaml verbatim
    const CornerResult r =
        bincv::goodFeaturesToTrack(dx, dy, params, map, found.data(), found.size());

    const double margin = 0.5 * static_cast<double>(std::max(winW, winH) - 1) + 2.0;
    std::vector<Point2f> out;
    for (size_t i = 0; i < r.count; ++i) {
        const double x = found[i].x, y = found[i].y;
        double wx = 0.0, wy = 0.0;
        warp.forward(x, y, wx, wy);
        const bool inside = x >= margin && y >= margin &&
                            x <= static_cast<double>(width) - 1.0 - margin &&
                            y <= static_cast<double>(height) - 1.0 - margin &&
                            wx >= margin && wy >= margin &&
                            wx <= static_cast<double>(width) - 1.0 - margin &&
                            wy <= static_cast<double>(height) - 1.0 - margin;
        if (inside) out.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
    }
    return out;
}

/// @brief The smallest `referenceMinEig` over a point set -- the quantity LOSS
///        RULE 2 compares against `minEigThreshold`, in the reference's units.
/// @note Reported rather than asserted: a rejection threshold that never rejects
///       anything on the content the operation is aimed at is a fact about the
///       coverage of the loss rules, and the only honest place to put it is next
///       to the numbers it explains.
template <typename WordType>
double smallestReferenceMinEig(const Frontend<WordType>& fe, const std::vector<Point2f>& pts,
                               const LKParams& params) {
    const double halfX = 0.5 * static_cast<double>(params.winWidth - 1);
    const double halfY = 0.5 * static_cast<double>(params.winHeight - 1);
    double smallest = -1.0;
    for (const Point2f& p : pts) {
        const Rect window(static_cast<int>(std::floor(static_cast<double>(p.x) - halfX)),
                          static_cast<int>(std::floor(static_cast<double>(p.y) - halfY)),
                          params.winWidth, params.winHeight);
        const bincv::GradientCovariance a = bincv::gradientCovariance<WordType>(
            fe.dx[0].constMagnitude(0), fe.dy[0].constMagnitude(0), fe.dx[0].constSign(),
            fe.dy[0].constSign(), window);
        const double eig = static_cast<double>(bincv::impl::minEigenValue(a.sumXX, a.sumYY,
                                                                         a.sumXY));
        const double referenceEig = bincv::impl::kReferenceMinEigScale * eig /
                                    static_cast<double>(params.winWidth * params.winHeight);
        if (smallest < 0.0 || referenceEig < smallest) smallest = referenceEig;
    }
    return smallest;
}

/// @brief The subset of `pts` whose window is fully inside EVERY level of `fe`.
/// @note The control for deviation (ii). binCV clips the window against the level
///       where the reference pads it with a winSize-wide reflected border, and at
///       a coarse level a point that is comfortably interior at level 0 can have
///       most of its window outside. Restricting to points that never clip
///       separates that cost from the cost of the level's BIT DEPTH, which is the
///       distinction E-7 / T4.1 turns on.
template <typename WordType>
std::vector<Point2f> unclippedAtEveryLevel(const Frontend<WordType>& fe,
                                           const std::vector<Point2f>& pts, int winW, int winH) {
    std::vector<Point2f> out;
    const double halfX = 0.5 * static_cast<double>(winW - 1);
    const double halfY = 0.5 * static_cast<double>(winH - 1);
    for (const Point2f& p : pts) {
        bool ok = true;
        for (size_t li = 0; li < fe.prev.size(); ++li) {
            const double scale = 1.0 / static_cast<double>(1u << li);
            const double x = static_cast<double>(p.x) * scale;
            const double y = static_cast<double>(p.y) * scale;
            if (x - halfX < 0.0 || y - halfY < 0.0 ||
                x + halfX >= static_cast<double>(fe.prev[li].cols()) ||
                y + halfY >= static_cast<double>(fe.prev[li].rows())) {
                ok = false;
                break;
            }
        }
        if (ok) out.push_back(p);
    }
    return out;
}

struct FlowStats {
    size_t eligible = 0;
    size_t tracked = 0;
    size_t stuck = 0;     ///< T3: tracked, but never moved while truth did
    size_t truthMoved = 0;  ///< points whose ground truth moved >= kTruthMoved
    double rms = 0.0;
    double maxError = 0.0;
};

FlowStats measure(const std::vector<Point2f>& prevPts, const std::vector<Point2f>& nextPts,
                  const std::vector<uint8_t>& status, const Warp& warp) {
    FlowStats s;
    s.eligible = prevPts.size();
    double sumSq = 0.0;
    for (size_t i = 0; i < prevPts.size(); ++i) {
        if (status[i] == 0) continue;
        double gx = 0.0, gy = 0.0;
        warp.forward(static_cast<double>(prevPts[i].x), static_cast<double>(prevPts[i].y), gx, gy);
        const double ex = static_cast<double>(nextPts[i].x) - gx;
        const double ey = static_cast<double>(nextPts[i].y) - gy;
        const double e = std::sqrt(ex * ex + ey * ey);
        sumSq += e * e;
        if (e > s.maxError) s.maxError = e;
        ++s.tracked;

        // T3's second half. Both quantities are of the POINT, not of the frame:
        // under a rotation or a scale the ground-truth displacement is near zero
        // at the centre of the image and large at its corners, so "did truth
        // move" has to be asked per point.
        const double tx = gx - static_cast<double>(prevPts[i].x);
        const double ty = gy - static_cast<double>(prevPts[i].y);
        const double truth = std::sqrt(tx * tx + ty * ty);
        const double fx = static_cast<double>(nextPts[i].x) - static_cast<double>(prevPts[i].x);
        const double fy = static_cast<double>(nextPts[i].y) - static_cast<double>(prevPts[i].y);
        const double reported = std::sqrt(fx * fx + fy * fy);
        if (truth >= kTruthMoved) {
            ++s.truthMoved;
            if (reported <= kStuckFlow) ++s.stuck;
        }
    }
    s.rms = (s.tracked > 0) ? std::sqrt(sumSq / static_cast<double>(s.tracked)) : 0.0;
    return s;
}

/// @brief Runs one warp case end to end and checks it against the tolerance
///        stated at the top of the file.
/// @param modelError The a-priori allowance for a non-translational warp:
///        `halfWin * theta` or `halfWin * |s-1|`, and exactly 0 for a translation.
template <typename WordType>
FlowStats runPoints(const char* label, Frontend<WordType>& fe, const Warp& warp,
                    const std::vector<Point2f>& pts, double modelError, bool enforce) {
    LKParams params;  // seal_params.yaml verbatim
    std::vector<Point2f> out(pts.size());
    std::vector<uint8_t> status(pts.size());
    std::vector<float> err(pts.size());
    bincv::calcOpticalFlowPyrLK<WordType>(fe.levels.data(), fe.levels.size(), pts.data(),
                                          out.data(), status.data(), err.data(), pts.size(),
                                          params);

    const FlowStats s = measure(pts, out, status, warp);
    const bool withinRms = s.rms <= kRmsTolerance + modelError;
    const bool withinMax = s.maxError <= kMaxTolerance + modelError;
    std::printf("  %-26s eligible=%3zu tracked=%3zu stuck=%3zu/%3zu rms=%.4f max=%.4f "
                "(tol rms<=%.4f max<=%.4f)  %s\n",
                label, s.eligible, s.tracked, s.stuck, s.truthMoved, s.rms, s.maxError,
                kRmsTolerance + modelError, kMaxTolerance + modelError,
                (withinRms && withinMax) ? "WITHIN" : "OVER TOLERANCE");
    BINCV_CHECK(s.eligible >= 16);
    BINCV_CHECK(static_cast<double>(s.tracked) >=
                kMinTrackedFraction * static_cast<double>(s.eligible));
    if (enforce) {
        BINCV_CHECK(withinRms);
        BINCV_CHECK(withinMax);
        // T3's second half. On an enforced case this is implied by T2 wherever
        // ground truth exceeds the max tolerance, and it is NOT implied where
        // truth is between 0.5 px and 1.0 px -- which is exactly the range the
        // real frame fails in. Asserting it here costs nothing on content that
        // works and is the gate that content which does not work must fail.
        BINCV_CHECK_EQ(s.stuck, size_t{0});
    }
    return s;
}

/// @brief Selects the eligible points and runs them.
template <typename WordType>
FlowStats runOnFrames(const char* label, Frontend<WordType>& fe, const Warp& warp,
                      double modelError, bool enforce = true) {
    LKParams params;
    const std::vector<Point2f> pts = eligiblePoints(fe.dx[0], fe.dy[0], fe.prev[0].cols(),
                                                    fe.prev[0].rows(), warp, params.winWidth,
                                                    params.winHeight);
    return runPoints<WordType>(label, fe, warp, pts, modelError, enforce);
}

/// @brief The synthetic-texture spelling: render both frames from the continuous
///        field and run.
template <typename WordType>
FlowStats runCase(const char* label, int width, int height, const Warp& warp, double modelError) {
    const int levelCount = 4;  // seal_params.yaml: lk_max_level 3
    Frontend<WordType> fe(width, height, levelCount);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], warp);
    fe.build();
    return runOnFrames<WordType>(label, fe, warp, modelError);
}


// ---------------------------------------------------------------------------
// X-24 / E-7: THE SAME FRONTEND AT A CHOSEN BIT DEPTH PER LEVEL
//
// `Frontend` above is the shipped 1-bit ladder and stays exactly as it is -- it
// is what every X-20 number was measured on. This is its generic-N counterpart,
// and it exists so that E-7's question can be asked without disturbing the
// baseline it has to be compared against.
//
// **LEVEL 0 IS ONE BIT IN EVERY LADDER AND IS NOT A VARIABLE.** It is the binary
// frame -- the project's premise (ARCHITECTURE 1). Only the levels pyrDown
// PRODUCES have a depth to choose, which is why every ladder below starts `1, ...`.
// ---------------------------------------------------------------------------
namespace {

/// @brief One SignedQuantMat per level, each at that level's depth and extent.
/// @note Recursive for the reason impl::PyramidLevels is: the levels have
///       DIFFERENT types, so a vector of them would need type erasure.
template <typename WordType, size_t... LevelBits>
struct DerivLadder;

template <typename WordType, size_t N0>
struct DerivLadder<WordType, N0> {
    bincv::SignedQuantMat<N0, WordType> mat;
    DerivLadder(int w, int h) : mat(w, h) {}
    template <size_t I>
    bincv::SignedQuantMat<N0, WordType>& get() {
        static_assert(I == 0, "derivative ladder index out of range");
        return mat;
    }
    size_t bytes() const { return mat.sizeInWords() * sizeof(WordType); }
};

template <typename WordType, size_t N0, size_t N1, size_t... Rest>
struct DerivLadder<WordType, N0, N1, Rest...> {
    bincv::SignedQuantMat<N0, WordType> mat;
    DerivLadder<WordType, N1, Rest...> rest;
    DerivLadder(int w, int h)
        : mat(w, h),
          rest(static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w))),
               static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h)))) {}
    template <size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return mat;
        } else {
            return rest.template get<I - 1>();
        }
    }
    size_t bytes() const { return mat.sizeInWords() * sizeof(WordType) + rest.bytes(); }
};

template <typename WordType, size_t... LevelBits>
struct LadderFrontend {
    static constexpr size_t Levels = sizeof...(LevelBits);
    using Pyr = bincv::Pyramid<WordType, LevelBits...>;

    Pyr prev, next;
    DerivLadder<WordType, LevelBits...> dx, dy;
    bincv::LKLevels<WordType, LevelBits...> levels;

    LadderFrontend(int w, int h) : prev(w, h), next(w, h), dx(w, h), dy(w, h) {}

    /// @brief pyrDown down both ladders, then the derivative of every previous
    ///        level, then bind the views. Level 0 of both pyramids is the caller's.
    void build() {
        prev.build();
        next.build();
        buildDeriv<0>();
        bind<0>();
    }

    template <size_t I>
    void buildDeriv() {
        if constexpr (I < Levels) {
            bincv::derivativeX(prev.template level<I>(), dx.template get<I>());
            bincv::derivativeY(prev.template level<I>(), dy.template get<I>());
            buildDeriv<I + 1>();
        }
    }
    template <size_t I>
    void bind() {
        if constexpr (I < Levels) {
            levels.template get<I>() = bincv::lkLevel<Pyr::template levelBits<I>()>(
                prev.template level<I>(), next.template level<I>(), dx.template get<I>(),
                dy.template get<I>());
            bind<I + 1>();
        }
    }

    /// @brief Peak working set of the tracking stage: both pyramids and BOTH
    ///        derivative ladders. They coexist -- the tracker reads all of them --
    ///        so this is a peak, not a per-buffer ratio (CLAUDE.md, benchmarking).
    size_t bytes() const {
        return prev.sizeInBytes() + next.sizeInBytes() + dx.bytes() + dy.bytes();
    }
};

/// @brief Copies a rendered 1-bit frame into a ladder's level 0.
template <typename WordType, size_t... LevelBits>
void seedLevelZero(LadderFrontend<WordType, LevelBits...>& fe, const BinMat<WordType>& prevSrc,
                   const BinMat<WordType>& nextSrc) {
    for (int y = 0; y < prevSrc.rows(); ++y) {
        for (int x = 0; x < prevSrc.cols(); ++x) {
            fe.prev.template level<0>().set(y, x, prevSrc.at(y, x));
            fe.next.template level<0>().set(y, x, nextSrc.at(y, x));
        }
    }
}

/// @brief Runs one ladder over one warp and returns X-20's own FlowStats.
/// @note Uses `measure()` -- the SAME function every X-20 number came out of --
///       so the tolerance and the stuck rule cannot drift between the baseline
///       and the sweep. Two copies of a tolerance is how two tolerances happen.
template <typename WordType, size_t... LevelBits>
FlowStats runLadder(const char* label, const BinMat<WordType>& prevSrc,
                    const BinMat<WordType>& nextSrc, const Warp& warp,
                    const std::vector<Point2f>& pts, double modelError, size_t* bytesOut) {
    LadderFrontend<WordType, LevelBits...> fe(prevSrc.cols(), prevSrc.rows());
    seedLevelZero(fe, prevSrc, nextSrc);
    fe.build();
    if (bytesOut != nullptr) *bytesOut = fe.bytes();

    LKParams params;  // seal_params.yaml verbatim
    std::vector<Point2f> out(pts.size());
    std::vector<uint8_t> status(pts.size());
    bincv::calcOpticalFlowPyrLK(fe.levels, pts.data(), out.data(), status.data(), nullptr,
                                pts.size(), params);
    const FlowStats s = measure(pts, out, status, warp);
    const bool within = s.rms <= kRmsTolerance + modelError && s.maxError <= kMaxTolerance +
                        modelError;
    std::printf("  %-12s tracked=%3zu/%3zu stuck=%2zu/%2zu  rms=%7.4f  max=%7.4f  bytes=%8zu  %s\n",
                label, s.tracked, s.eligible, s.stuck, s.truthMoved, s.rms, s.maxError,
                fe.bytes(), within ? "WITHIN" : "OVER");
    return s;
}

} // namespace

constexpr int kW = 320;
constexpr int kH = 240;

#ifdef BINCV_WITH_OPENCV
// ---------------------------------------------------------------------------
// THE SAME MEASUREMENT ON THE REPO'S REAL TEST IMAGE
//
// Everything above runs on synthetic texture; T3.8's first Done-when bullet asks
// for real content as well, and real content needs a PNG decoder. This half is
// therefore behind BINCV_WITH_OPENCV -- the pattern tests/test_denoise.cpp and
// tests/test_derivative.cpp already use -- rather than in a separate suite, so
// that it shares ONE harness, ONE tolerance and ONE set of eligibility rules with
// the synthetic cases. Two copies of a tolerance is how two tolerances happen.
//
// THE BINARIZATION IS THE REFERENCE PIPELINE'S OWN, NOT A THRESHOLD CHOSEN HERE.
// `rl_fast_edge_filter_wide` (SEAL/src/temporal_processing/edge_filter.cpp) with
// `edge_threshold: 17` is what produces the frames the SEAL frontend actually
// tracks: `|[-1,0,1] * I| >= 17` horizontally OR vertically. Ported below, it
// reproduces the repo's shipped `_bin_normalized.png` to within 0.024% of pixels,
// which is how it is known to be the right function rather than a plausible one.
// An earlier version of this case used a global Otsu threshold instead and
// produced content -- 19% set, 2.7% edge pixels, large smooth regions -- that the
// frontend never sees; it is worth saying so, because the two binarizations give
// materially different tracking numbers and only one of them is the reference's.
//
// THE GROUND TRUTH IS CONSTRUCTED THE SAME WAY IT IS FOR SYNTHETIC TEXTURE: the
// warp is applied to the GRAYSCALE frame -- the continuous-valued thing -- and the
// binarization is applied AFTERWARDS to both frames, by the same function. Warping
// the bits instead would require resampling binary content, which cannot be done
// without inventing information; that is the mistake this harness is shaped to
// avoid. The only inexactness left is the bicubic resampling of the grayscale, and
// it is in the CONTENT, not in the ground truth: the displacement of every point
// is still the warp's own arithmetic.
// ---------------------------------------------------------------------------

/// @brief `rl_fast_edge_filter_wide`, ported call for call.
cv::Mat referenceEdgeFilter(const cv::Mat& gray, int edgeThreshold) {
    const cv::Mat kernelX = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat kernelY = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat diffX, diffY;
    cv::filter2D(gray, diffX, CV_32F, kernelX);
    cv::filter2D(gray, diffY, CV_32F, kernelY);
    diffX = cv::abs(diffX);
    diffY = cv::abs(diffY);
    const cv::Mat mask = (diffX >= edgeThreshold) | (diffY >= edgeThreshold);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8U);
    out.setTo(255, mask);
    return out;
}

/// @brief cv::warpAffine's 2x3 for `p' = A (p - c) + c + t`, i.e. the forward map
///        from frame 0 to frame 1.
cv::Mat affineOf(const Warp& w) {
    cv::Mat m(2, 3, CV_64F);
    m.at<double>(0, 0) = w.m00;
    m.at<double>(0, 1) = w.m01;
    m.at<double>(1, 0) = w.m10;
    m.at<double>(1, 1) = w.m11;
    m.at<double>(0, 2) = w.cx - (w.m00 * w.cx + w.m01 * w.cy) + w.tx;
    m.at<double>(1, 2) = w.cy - (w.m10 * w.cx + w.m11 * w.cy) + w.ty;
    return m;
}

/// @brief Loads the repo's sample frame as grayscale. Empty if it is missing.
cv::Mat loadRealFrame() {
    const std::string path = std::filesystem::path(__FILE__).parent_path().string() +
                             "/images/1403715887284058112.png";
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

/// @brief Builds the two binarized frames for one warp of the real image.
template <typename WordType>
void buildRealFrontend(const cv::Mat& gray, const Warp& warp, int levelCount,
                       Frontend<WordType>& fe) {
    cv::Mat warped;
    cv::warpAffine(gray, warped, affineOf(warp), gray.size(), cv::INTER_CUBIC,
                   cv::BORDER_REFLECT_101);
    const cv::Mat bin0 = referenceEdgeFilter(gray, 17);
    const cv::Mat bin1 = referenceEdgeFilter(warped, 17);
    fe.prev[0].fromCVMat(bin0);
    fe.next[0].fromCVMat(bin1);
    fe.build();
    (void)levelCount;
}

template <typename WordType>
FlowStats runRealFrameCase(const cv::Mat& gray, const char* label, const Warp& warp,
                           double modelError, int levelCount, bool enforce) {
    Frontend<WordType> fe(gray.cols, gray.rows, levelCount);
    buildRealFrontend(gray, warp, levelCount, fe);
    return runOnFrames<WordType>(label, fe, warp, modelError, enforce);
}
#endif // BINCV_WITH_OPENCV

} // namespace

// ===========================================================================
// T1/T2/T3 -- the tolerance, on translations
// ===========================================================================

BINCV_TEST(Flow, SubPixelTranslation_uint32_t) {
    std::printf("\n  sub-pixel translations -- the case a whole-pixel tracker cannot do\n");
    runCase<uint32_t>("shift (0.25, 0.25)", kW, kH, translation(0.25, 0.25), 0.0);
    runCase<uint32_t>("shift (0.50, 0.50)", kW, kH, translation(0.50, 0.50), 0.0);
    runCase<uint32_t>("shift (0.75, 0.75)", kW, kH, translation(0.75, 0.75), 0.0);
    runCase<uint32_t>("shift (0.75, 0.25)", kW, kH, translation(0.75, 0.25), 0.0);
    runCase<uint32_t>("shift (2.25, -1.50)", kW, kH, translation(2.25, -1.50), 0.0);
}

BINCV_TEST(Flow, IntegerTranslation_uint32_t) {
    std::printf("\n  integer translations -- the easy baseline\n");
    runCase<uint32_t>("shift (1, 0)", kW, kH, translation(1.0, 0.0), 0.0);
    runCase<uint32_t>("shift (0, -2)", kW, kH, translation(0.0, -2.0), 0.0);
    runCase<uint32_t>("shift (3, 2)", kW, kH, translation(3.0, 2.0), 0.0);
    runCase<uint32_t>("shift (-5, 4)", kW, kH, translation(-5.0, 4.0), 0.0);
}

// ===========================================================================
// T4 -- THE CLAIM. A tracker whose displacements are whole pixels returns
// round(d), so on a translation with fractional part q its error is min(q, 1-q).
// This must beat that bound. See T4 at the top of the file: it is a property of
// the integer grid, NOT a measurement of route (a), which is E-6 / T4.2 and is
// neither implemented nor run here.
// ===========================================================================

BINCV_TEST(Flow, BeatsTheIntegerGrid_uint32_t) {
    std::printf("\n  T4: RMS must be strictly below min(q, 1-q), the error a\n"
                "      whole-pixel tracker cannot avoid on a translation of q\n");
    const double fractions[3] = {0.25, 0.5, 0.75};
    for (int i = 0; i < 3; ++i) {
        const double q = fractions[i];
        const double bound = std::min(q, 1.0 - q);
        const FlowStats s = runCase<uint32_t>("", kW, kH, translation(q, 0.0), 0.0);
        std::printf("      q=%.2f  rms=%.4f  whole-pixel bound min(q,1-q)=%.4f  -> %s\n", q, s.rms,
                    bound, (s.rms < bound) ? "BEATS IT" : "DOES NOT BEAT IT");
        BINCV_CHECK(s.rms < bound);
    }
}

// ===========================================================================
// Rotation and scale, with the a-priori model-error allowance
// ===========================================================================

BINCV_TEST(Flow, RotationAndScale_uint32_t) {
    LKParams params;
    const double halfWin = 0.5 * static_cast<double>(params.winWidth - 1);
    const double degrees = 1.0;
    const double rotModel = halfWin * degrees * 3.14159265358979323846 / 180.0;
    const double scaleFactor = 1.02;
    const double scaleModel = halfWin * (scaleFactor - 1.0);
    std::printf("\n  rotation %.1f deg  -> model error allowance %.4f px\n"
                "  scale    %.3fx    -> model error allowance %.4f px\n",
                degrees, rotModel, scaleFactor, scaleModel);
    runCase<uint32_t>("rotate 1 deg", kW, kH, rotation(degrees, kW * 0.5, kH * 0.5), rotModel);
    runCase<uint32_t>("rotate -1 deg", kW, kH, rotation(-degrees, kW * 0.5, kH * 0.5), rotModel);
    runCase<uint32_t>("scale 1.02", kW, kH, scaling(scaleFactor, kW * 0.5, kH * 0.5), scaleModel);
    runCase<uint32_t>("scale 0.98", kW, kH, scaling(1.0 / scaleFactor, kW * 0.5, kH * 0.5),
                      scaleModel);
}

// ===========================================================================
// THE BIT-PARALLEL RESIDUAL IDENTITY -- evidence 2, at its sharpest
// ===========================================================================

namespace {

template <typename WordType>
void residualIdentity(const char* typeName) {
    Frontend<WordType> fe(96, 72, 1);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], translation(1.37, -0.62));
    fe.build();

    const int winW = 15, winH = 15;
    double worst = 0.0;
    size_t compared = 0;
    // Origins swept from a full window outside every edge to a full window past
    // it, and offsets covering both signs, both integer parts and every quadrant
    // of the bilinear cell -- the tap shift is the arithmetic most able to be
    // subtly wrong, and it is wrong differently at each of those.
    for (int oy = -winH; oy <= 72; oy += 7) {
        for (int ox = -winW; ox <= 96; ox += 9) {
            const bincv::impl::RegionWords<WordType> region =
                bincv::impl::clipRegion<WordType>(96, 72, Rect(ox, oy, winW, winH));
            if (region.isEmpty) continue;
            const Extent e = refClip(ox, oy, winW, winH, 96, 72);
            const double offsets[6][2] = {{0.0, 0.0},   {0.25, 0.75}, {-1.5, 2.25},
                                          {3.4, -2.9},  {-0.1, -0.1}, {17.0, -13.0}};
            for (int k = 0; k < 6; ++k) {
                const double offX = static_cast<double>(ox) + offsets[k][0];
                const double offY = static_cast<double>(oy) + offsets[k][1];
                const long long tapX = static_cast<long long>(std::floor(offX - ox));
                const long long tapY = static_cast<long long>(std::floor(offY - oy));
                const double fx = (offX - ox) - static_cast<double>(tapX);
                const double fy = (offY - oy) - static_cast<double>(tapY);
                const double w00 = (1.0 - fx) * (1.0 - fy);
                const double w01 = fx * (1.0 - fy);
                const double w10 = (1.0 - fx) * fy;
                const double w11 = fx * fy;

                bincv::impl::TapSums sx, sy;
                bincv::impl::residualSums<WordType>(fe.levels[0], region, tapX, tapY, sx, sy);
                const double gotB1 = sx.combine(w00, w01, w10, w11);
                const double gotB2 = sy.combine(w00, w01, w10, w11);

                double wantB1 = 0.0, wantB2 = 0.0;
                refResidual(fe.prev[0], fe.next[0], fe.dx[0], fe.dy[0], e, offX - ox, offY - oy,
                            wantB1, wantB2);
                worst = std::max(worst, std::fabs(gotB1 - wantB1));
                worst = std::max(worst, std::fabs(gotB2 - wantB2));
                ++compared;
            }
        }
    }
    std::printf("  residual identity %-9s %6zu positions, worst |popcount - per-pixel| = %.3e\n",
                typeName, compared, worst);
    BINCV_CHECK(compared > 400);
    // Both sides sum the same terms in different orders in double; only rounding
    // separates them. 1e-9 on sums of magnitude ~1e2 is ~1e-11 relative, far
    // tighter than any real disagreement could hide under.
    BINCV_CHECK(worst < 1e-9);
}

} // namespace

BINCV_TEST(Flow, ResidualIdentity_uint8_t)  { residualIdentity<uint8_t>("uint8_t"); }
BINCV_TEST(Flow, ResidualIdentity_uint16_t) { residualIdentity<uint16_t>("uint16_t"); }
BINCV_TEST(Flow, ResidualIdentity_uint32_t) { residualIdentity<uint32_t>("uint32_t"); }
BINCV_TEST(Flow, ResidualIdentity_uint64_t) { residualIdentity<uint64_t>("uint64_t"); }

// ===========================================================================
// THE WHOLE TRACKER AGAINST THE PER-PIXEL FLOAT IMPLEMENTATION
// ===========================================================================

BINCV_TEST(Flow, MatchesPerPixelFloatImplementation_uint32_t) {
    Frontend<uint32_t> fe(160, 128, 3);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], translation(1.6, -0.85));
    fe.build();

    LKParams params;
    std::vector<Point2f> pts = eligiblePoints(fe.dx[0], fe.dy[0], 160, 128,
                                              translation(1.6, -0.85), params.winWidth,
                                              params.winHeight);
    BINCV_CHECK(pts.size() >= 4);

    std::vector<Point2f> got(pts.size());
    std::vector<uint8_t> gotStatus(pts.size());
    bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                          got.data(), gotStatus.data(), nullptr, pts.size(),
                                          params);

    std::vector<Point2f> want;
    std::vector<uint8_t> wantStatus;
    refTrack(fe, pts, want, wantStatus, params);

    double worst = 0.0;
    size_t statusMismatches = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (gotStatus[i] != wantStatus[i]) ++statusMismatches;
        const double ex = static_cast<double>(got[i].x) - static_cast<double>(want[i].x);
        const double ey = static_cast<double>(got[i].y) - static_cast<double>(want[i].y);
        worst = std::max(worst, std::sqrt(ex * ex + ey * ey));
    }
    std::printf("  whole tracker vs per-pixel float: %zu points, worst disagreement %.3e px,"
                " %zu status mismatches\n", pts.size(), worst, statusMismatches);
    BINCV_CHECK_EQ(statusMismatches, size_t{0});
    // The two differ only in the ORDER of a sum, and the positions are stored as
    // float, so the two iterations follow the same path. A hundredth of a pixel is
    // three orders of magnitude below the tolerance the operation is held to.
    BINCV_CHECK(worst < 0.01);
}

// ===========================================================================
// WORD-TYPE INVARIANCE -- where a cross-word tap bug would live
// ===========================================================================

BINCV_TEST(Flow, WordTypeInvariance) {
    const Warp warp = translation(1.4, -0.7);
    std::vector<Point2f> reference;
    std::vector<uint8_t> referenceStatus;
    size_t mismatches = 0;

    // A width that is NOT a multiple of 64, so every word type has a different
    // trailing partial word and a different set of cross-word boundaries.
    const int w = 149, h = 101;

    Frontend<uint8_t>  f8(w, h, 3);
    Frontend<uint16_t> f16(w, h, 3);
    Frontend<uint32_t> f32(w, h, 3);
    Frontend<uint64_t> f64(w, h, 3);
    renderWarped(f8.prev[0], Warp{});  renderWarped(f8.next[0], warp);  f8.build();
    renderWarped(f16.prev[0], Warp{}); renderWarped(f16.next[0], warp); f16.build();
    renderWarped(f32.prev[0], Warp{}); renderWarped(f32.next[0], warp); f32.build();
    renderWarped(f64.prev[0], Warp{}); renderWarped(f64.next[0], warp); f64.build();

    LKParams params;
    std::vector<Point2f> pts =
        eligiblePoints(f32.dx[0], f32.dy[0], w, h, warp, params.winWidth, params.winHeight);
    BINCV_CHECK(pts.size() >= 3);

    std::vector<Point2f> out(pts.size());
    std::vector<uint8_t> st(pts.size());

#define BINCV_FLOW_RUN(fe, W)                                                            \
    bincv::calcOpticalFlowPyrLK<W>((fe).levels.data(), (fe).levels.size(), pts.data(),    \
                                   out.data(), st.data(), nullptr, pts.size(), params);   \
    if (reference.empty()) { reference = out; referenceStatus = st; }                     \
    else {                                                                               \
        for (size_t i = 0; i < pts.size(); ++i) {                                        \
            if (out[i].x != reference[i].x || out[i].y != reference[i].y ||               \
                st[i] != referenceStatus[i]) ++mismatches;                                \
        }                                                                                \
    }

    BINCV_FLOW_RUN(f8, uint8_t)
    BINCV_FLOW_RUN(f16, uint16_t)
    BINCV_FLOW_RUN(f32, uint32_t)
    BINCV_FLOW_RUN(f64, uint64_t)
#undef BINCV_FLOW_RUN

    std::printf("  word-type invariance at %dx%d: %zu points, %zu bit-level mismatches\n", w, h,
                pts.size(), mismatches);
    BINCV_CHECK_EQ(mismatches, size_t{0});
}

// ===========================================================================
// `err` IS THE RESIDUAL AT THE POSITION THAT WAS RETURNED
//
// THE BUG THIS PINS. `err` used to be computed from the tap offset and the four
// bilinear weights left over from the START of the last executed iteration --
// i.e. one whole step before the position actually handed back, and one and a
// half steps before it whenever the oscillation rule fired and backed the
// estimate off by half a delta. Measured against an independent per-pixel float
// residual at the RETURNED point: 134% high at maxIterations = 1, 51% at 2, 0.4%
// at the default 20. Nothing in the suite read an `err` value at all, so the
// popcount collapse `|Jinterp - I| = I + (1 - 2I)*Jinterp` -- one of the two
// identities the whole file rests on -- had no coverage while the residual
// identity next to it had 864 positions per word type.
//
// The iteration budget is swept deliberately: at the default the tracker has
// converged and a stale-by-one-iteration error is nearly the right answer, which
// is exactly why a test that only ran the default would not have caught it.
// ===========================================================================

BINCV_TEST(Flow, ErrorIsMeasuredAtTheReturnedPosition_uint32_t) {
    std::printf("\n  err vs an independent per-pixel float residual AT THE RETURNED POINT\n");
    const Warp warp = translation(1.37, -0.62);
    const int iterationBudgets[4] = {1, 2, 3, 20};
    for (int k = 0; k < 4; ++k) {
        Frontend<uint32_t> fe(160, 128, 1);
        renderWarped(fe.prev[0], Warp{});
        renderWarped(fe.next[0], warp);
        fe.build();

        LKParams params;
        params.maxIterations = iterationBudgets[k];
        std::vector<Point2f> pts = eligiblePoints(fe.dx[0], fe.dy[0], 160, 128, warp,
                                                  params.winWidth, params.winHeight);
        BINCV_CHECK(pts.size() >= 4);
        std::vector<Point2f> out(pts.size());
        std::vector<uint8_t> st(pts.size());
        std::vector<float> er(pts.size());
        bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                              out.data(), st.data(), er.data(), pts.size(),
                                              params);

        const double halfX = 0.5 * static_cast<double>(params.winWidth - 1);
        const double halfY = 0.5 * static_cast<double>(params.winHeight - 1);
        double worst = 0.0;
        size_t compared = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (st[i] == 0) continue;
            // The window is the PREVIOUS frame's, anchored on the integer grid
            // (deviation (i)); the displacement is measured from the unrounded
            // previous position, which is what the kernel's `offX`/`offY` are.
            const double px = static_cast<double>(pts[i].x) - halfX;
            const double py = static_cast<double>(pts[i].y) - halfY;
            const Extent e = refClip(static_cast<long long>(std::floor(px)),
                                     static_cast<long long>(std::floor(py)), params.winWidth,
                                     params.winHeight, 160, 128);
            const double offX = (static_cast<double>(out[i].x) - halfX) - px;
            const double offY = (static_cast<double>(out[i].y) - halfY) - py;
            const double want = refMeanAbsDiff(fe.prev[0], fe.next[0], e, offX, offY);
            worst = std::max(worst, std::fabs(static_cast<double>(er[i]) - want));
            ++compared;
        }
        std::printf("    maxIterations=%2d  %2zu points  worst |err - per-pixel float| = %.3e\n",
                    params.maxIterations, compared, worst);
        BINCV_CHECK(compared >= 4);
        // Both sides sum the same terms in a different order and `err` is stored
        // as float, so only rounding separates them: 1e-6 on a quantity of order
        // 0.04 is four orders of magnitude below the smallest staleness the bug
        // produced (0.0001 at the default budget, 0.055 at one iteration).
        BINCV_CHECK(worst < 1e-6);
    }
}

// ===========================================================================
// A LEVEL NO LARGER THAN THE WINDOW IS IGNORED (deviation (vi))
//
// The reference stops BUILDING levels at the first one that is not strictly
// larger than winSize and truncates maxLevel; binCV cannot decline to build one,
// because the caller owns the pyramid, so it declines to USE one. Without the
// rule the repo's own 149x101 word-type case was tracking on a 38x26 level under
// a 31x31 window -- every window clipped, every point getting nearly the same A
// and b, and that estimate multiplied by 4 on the way down.
//
// The check is EQUALITY against the same call with the undersized levels never
// passed, which is the only formulation that cannot be satisfied by a cap that
// fires at the wrong level.
// ===========================================================================

BINCV_TEST(Flow, LevelsAtOrBelowTheWindowAreIgnored_uint32_t) {
    LKParams params;
    const Warp warp = translation(1.4, -0.7);
    const int w = 149, h = 101;

    Frontend<uint32_t> fe(w, h, 4);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], warp);
    fe.build();

    const size_t usable = usableLevelCount(fe, params.winWidth, params.winHeight);
    std::printf("\n  level sizes at %dx%d:", w, h);
    for (size_t i = 0; i < fe.prev.size(); ++i) {
        std::printf(" %dx%d%s", fe.prev[i].cols(), fe.prev[i].rows(),
                    (i + 1 == usable) ? " |" : "");
    }
    std::printf("   window %dx%d -> %zu of %zu levels usable\n", params.winWidth,
                params.winHeight, usable, fe.prev.size());
    BINCV_CHECK(usable < fe.prev.size());  // the case must actually exercise the cap

    std::vector<Point2f> pts =
        eligiblePoints(fe.dx[0], fe.dy[0], w, h, warp, params.winWidth, params.winHeight);
    BINCV_CHECK(pts.size() >= 3);

    std::vector<Point2f> withAll(pts.size()), withUsable(pts.size());
    std::vector<uint8_t> stAll(pts.size()), stUsable(pts.size());
    bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                          withAll.data(), stAll.data(), nullptr, pts.size(),
                                          params);
    bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), usable, pts.data(), withUsable.data(),
                                          stUsable.data(), nullptr, pts.size(), params);

    size_t mismatches = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (withAll[i].x != withUsable[i].x || withAll[i].y != withUsable[i].y ||
            stAll[i] != stUsable[i]) {
            ++mismatches;
        }
    }
    std::printf("  %zu points: passing 4 levels is BIT-IDENTICAL to passing %zu, mismatches=%zu\n",
                pts.size(), usable, mismatches);
    BINCV_CHECK_EQ(mismatches, size_t{0});

    // And the per-pixel float implementation, which derives the same cap from the
    // level sizes rather than from the kernel, agrees with both.
    std::vector<Point2f> want;
    std::vector<uint8_t> wantStatus;
    refTrack(fe, pts, want, wantStatus, params);
    double worst = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const double ex = static_cast<double>(withAll[i].x) - static_cast<double>(want[i].x);
        const double ey = static_cast<double>(withAll[i].y) - static_cast<double>(want[i].y);
        worst = std::max(worst, std::sqrt(ex * ex + ey * ey));
        if (stAll[i] != wantStatus[i]) ++mismatches;
    }
    std::printf("  vs per-pixel float with the same cap: worst %.3e px, %zu status mismatches\n",
                worst, mismatches);
    BINCV_CHECK_EQ(mismatches, size_t{0});
    BINCV_CHECK(worst < 0.01);
}

// ===========================================================================
// THE LOSS RULES, DRIVEN TO FIRE
// ===========================================================================

BINCV_TEST(Flow, LossRules_uint32_t) {
    LKParams params;

    // Rule 2 -- a blank frame has no gradient anywhere, so every window is
    // singular and every point must be lost.
    {
        Frontend<uint32_t> fe(96, 72, 2);
        fe.prev[0].fill(false);
        fe.next[0].fill(false);
        fe.build();
        std::vector<Point2f> pts{{40.0f, 30.0f}, {50.0f, 40.0f}};
        std::vector<Point2f> out(2);
        std::vector<uint8_t> st(2);
        std::vector<float> er(2);
        bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                              out.data(), st.data(), er.data(), 2, params);
        BINCV_CHECK_EQ(st[0], uint8_t{0});
        BINCV_CHECK_EQ(st[1], uint8_t{0});
        BINCV_CHECK_EQ(er[0], 0.0f);
    }

    // Rule 1 -- the window's origin out of range. At level 0 with a 31x31 window
    // the anchor is `p - 15`, so a point at -20 is a full window off the frame.
    {
        Frontend<uint32_t> fe(96, 72, 1);
        renderWarped(fe.prev[0], Warp{});
        renderWarped(fe.next[0], translation(1.0, 0.0));
        fe.build();
        std::vector<Point2f> pts{{-64.0f, 30.0f}, {200.0f, 30.0f}, {40.0f, -64.0f},
                                 {40.0f, 200.0f}, {40.0f, 30.0f}};
        std::vector<Point2f> out(pts.size());
        std::vector<uint8_t> st(pts.size());
        bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                              out.data(), st.data(), nullptr, pts.size(), params);
        BINCV_CHECK_EQ(st[0], uint8_t{0});
        BINCV_CHECK_EQ(st[1], uint8_t{0});
        BINCV_CHECK_EQ(st[2], uint8_t{0});
        BINCV_CHECK_EQ(st[3], uint8_t{0});
        BINCV_CHECK_EQ(st[4], uint8_t{1});
    }

    // Rule 3, ON THE POSITION THAT IS RETURNED. The in-loop test runs at the top
    // of each iteration, so the LAST step can carry a point out of range after it
    // has already passed -- and `status` describes the position the caller gets.
    // With one iteration allowed, the point below starts at a legal origin (46 of
    // a 48-wide level, a window almost entirely clipped away, which is legal
    // under rule 1) and its single step lands the origin at 48, one past the end.
    // The reference makes exactly this test in exactly this place, but only when
    // `err` was requested; here it is unconditional (deviation (vii)), so the
    // case is run BOTH ways and must agree.
    {
        Frontend<uint32_t> fe(48, 48, 1);
        renderWarped(fe.prev[0], Warp{});
        renderWarped(fe.next[0], translation(3.0, 0.0));
        fe.build();
        LKParams one = params;
        one.maxIterations = 1;
        const std::vector<Point2f> pts{{61.0f, 21.0f}};
        const double half = 0.5 * static_cast<double>(one.winWidth - 1);
        const long long anchor = static_cast<long long>(std::floor(61.0 - half));
        BINCV_CHECK(anchor >= -one.winWidth && anchor < 48);  // rule 1 does NOT fire

        for (int withErr = 0; withErr < 2; ++withErr) {
            std::vector<Point2f> out(1);
            std::vector<uint8_t> st(1, uint8_t{1});
            std::vector<float> er(1, -1.0f);
            bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                                  out.data(), st.data(),
                                                  (withErr != 0) ? er.data() : nullptr, 1, one);
            const long long finalOrigin =
                static_cast<long long>(std::floor(static_cast<double>(out[0].x) - half));
            // The fixture must actually produce an out-of-range RETURNED origin,
            // or the case proves nothing.
            BINCV_CHECK(finalOrigin >= 48);
            BINCV_CHECK_EQ(st[0], uint8_t{0});
            if (withErr != 0) BINCV_CHECK_EQ(er[0], 0.0f);
        }
    }

    // Degenerate arguments are values, not errors -- AND EVERY OUT ENTRY IS
    // WRITTEN, which is the documented contract. A zero-level call used to return
    // before the initialisation loop, leaving `status` and `err` holding whatever
    // the caller's buffer held; the poison below is what makes that visible
    // rather than accidentally-correct on a zeroed vector.
    {
        std::vector<Point2f> pts{{1.0f, 1.0f}, {7.0f, 9.0f}};
        std::vector<Point2f> out{{-1.0f, -1.0f}, {-1.0f, -1.0f}};
        std::vector<uint8_t> st{uint8_t{7}, uint8_t{7}};
        std::vector<float> er{-1.0f, -1.0f};
        bincv::calcOpticalFlowPyrLK<uint32_t>(nullptr, 0, pts.data(), out.data(), st.data(),
                                              er.data(), pts.size(), params);
        for (size_t i = 0; i < pts.size(); ++i) {
            BINCV_CHECK_EQ(st[i], uint8_t{0});   // no level -> nothing was tracked
            BINCV_CHECK_EQ(er[i], 0.0f);
            BINCV_CHECK_EQ(out[i].x, pts[i].x);  // last estimate is the point itself
            BINCV_CHECK_EQ(out[i].y, pts[i].y);
        }

        Frontend<uint32_t> fe(64, 64, 1);
        fe.build();
        bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                              out.data(), st.data(), nullptr, 0, params);
        BINCV_CHECK(true);  // zero points is a no-op
    }
}

// ===========================================================================
// NO HEAP
// ===========================================================================

BINCV_TEST(Flow, NoHeapInTheTracker_uint32_t) {
    Frontend<uint32_t> fe(128, 96, 3);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], translation(1.25, 0.5));
    fe.build();

    LKParams params;
    std::vector<Point2f> pts{{40.0f, 40.0f}, {60.0f, 50.0f}, {70.0f, 30.0f}};
    std::vector<Point2f> out(pts.size());
    std::vector<uint8_t> st(pts.size());
    std::vector<float> er(pts.size());

    const std::size_t before = g_newCount;
    bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                          out.data(), st.data(), er.data(), pts.size(), params);
    const std::size_t during = g_newCount - before;
    std::printf("  operator new across calcOpticalFlowPyrLK: %zu\n", during);
    BINCV_CHECK_EQ(during, std::size_t{0});

    // The counter is exercised, so that the zero above is a reading and not a
    // blind spot -- the idiom tests/test_covariance.cpp established.
    const std::size_t probeBefore = g_newCount;
    {
        std::vector<double> probe(8, 1.0);
        BINCV_CHECK(probe.size() == 8);
    }
    struct alignas(64) OverAligned { double v[8]; };
    {
        OverAligned* p = new OverAligned();
        BINCV_CHECK(p != nullptr);
        delete p;
    }
    // Read the counter into a local BEFORE the check: BINCV_CHECK_EQ names its
    // argument twice, once in the condition and once inside std::to_string, and
    // std::to_string allocates -- so the counter would be moved by the very act of
    // reading it, in an order the standard does not fix.
    const std::size_t probeAllocs = g_newCount - probeBefore;
    BINCV_CHECK_EQ(probeAllocs, std::size_t{2});
}

// ===========================================================================
// THE FOOTPRINT OF THE FULL FRONTEND -- the number Phase 4 needs
// ===========================================================================

namespace {
/// @brief FNV-1a over the exact bytes of a ranked corner prefix -- coordinates and
///        the response's exact `float` bits.
/// @note The element-for-element comparison of the two shapes still happens, in
///       full, in the scaffolding block below; this exists so that the two MEASURED
///       peak windows can each re-check the answer without holding the other
///       shape's array live inside the window being measured. A buffer carried
///       across the boundary purely to compare against would land in the peak and
///       corrupt the reading it was carried through.
std::uint64_t cornerDigest(const Corner* c, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, std::size_t bytes) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < bytes; ++i) {
            h ^= b[i];
            h *= 1099511628211ULL;
        }
    };
    mix(&n, sizeof(n));
    for (std::size_t i = 0; i < n; ++i) {
        mix(&c[i].x, sizeof(c[i].x));
        mix(&c[i].y, sizeof(c[i].y));
        mix(&c[i].response, sizeof(c[i].response));
    }
    return h;
}
} // namespace


BINCV_TEST(Flow, FrontendFootprint_640x480) {
    constexpr int W = 640;
    constexpr int H = 480;
    constexpr int LEVELS = 4;   // seal_params.yaml: lk_max_level 3
    constexpr int POINTS = 200; // seal_params.yaml: gftt_max_corners 200
    using Word = uint32_t;      // D-14

    std::printf("\n  PEAK FOOTPRINT OF THE FULL FRONTEND, %dx%d, %d pyramid levels,\n"
                "  1 bit per level, uint32_t words. THE TWO TOTALS ARE READ OFF A LIVE-BYTE\n"
                "  HIGH-WATER MARK, not summed from a list of buffers, and the per-stage rows\n"
                "  are then required to ACCOUNT for that reading exactly.\n", W, H, LEVELS);

    // THE BASELINE. Everything the test harness itself has on the heap before the
    // frontend exists. Every peak below is reported as `high-water - baseline`, so
    // gtest's own allocations are outside the number rather than inside it.
    const std::size_t baseline = g_liveBytes;

    // ---- stage 1: denoise -------------------------------------------------
    // The incoming binarized frame. denoiseMedian3 writes into pyramid level 0,
    // so the only buffer this stage owns is its source.
    BinMat<Word> incoming(W, H);
    BinMat<Word> incomingNext(W, H);
    renderWarped(incoming, translation(0.0, 0.0));
    renderWarped(incomingNext, translation(1.4, -0.7));

    Frontend<Word> fe(W, H, LEVELS);

    const std::size_t beforeDenoise = g_newCount;
    bincv::denoiseMedian3<Word>(incoming.constView(), fe.prev[0].view());
    bincv::denoiseMedian3<Word>(incomingNext.constView(), fe.next[0].view());
    const std::size_t denoiseAllocs = g_newCount - beforeDenoise;

    // ---- stages 2 and 3: pyramid and derivative ---------------------------
    const std::size_t beforeBuild = g_newCount;
    fe.runKernels();
    const std::size_t buildAllocs = g_newCount - beforeBuild;
    fe.bindLevels();

    // ---- stage 5's buffers, allocated NOW ---------------------------------
    // The track stage owns four fixed-size arrays. They are allocated before either
    // peak window so that both windows contain the WHOLE frontend and the two
    // readings differ by exactly one thing: the corner stage's response storage.
    std::vector<Point2f> prevPts(POINTS);
    std::vector<Point2f> nextPts(POINTS);
    std::vector<uint8_t> status(POINTS);
    std::vector<float> errs(POINTS);

    GoodFeaturesParams gftt;

    // ---- SCAFFOLDING: the equality, and the survivor count ----------------
    // THIS BLOCK IS NOT MEASURED AND SAYS SO. It holds both shapes' outputs at once
    // -- which is the only way to compare them element for element -- plus a W*H
    // probe buffer that cannot truncate. All of it is destroyed before either peak
    // window is armed. What survives is three scalars and a digest.
    std::size_t candidateCount = 0;
    std::size_t refCount = 0;
    std::uint64_t refDigest = 0;
    {
        std::vector<float> probeStorage(static_cast<std::size_t>(W) * static_cast<std::size_t>(H),
                                        0.0f);
        ResponseMap probeMap{probeStorage.data(), static_cast<std::size_t>(W),
                             static_cast<std::size_t>(H), static_cast<std::size_t>(W)};
        std::vector<Corner> probe(static_cast<std::size_t>(W) * static_cast<std::size_t>(H));
        const CornerResult probeResult = bincv::goodFeaturesToTrack(fe.dx[0], fe.dy[0], gftt,
                                                                    probeMap, probe.data(),
                                                                    probe.size());
        BINCV_CHECK_EQ(probeResult.candidatesTruncated, false);
        BINCV_CHECK(probeResult.count > 0);
        candidateCount = probeResult.candidatesRanked;
        refCount = probeResult.count;
        refDigest = cornerDigest(probe.data(), probeResult.candidatesRanked);

        // The ring, against the map, on X-20's own frontend content -- the whole
        // ranked prefix, coordinates and exact float bits.
        std::vector<float> ringStorage(bincv::kResponseRingRows * static_cast<std::size_t>(W),
                                       0.0f);
        ResponseMap ring{ringStorage.data(), static_cast<std::size_t>(W),
                         bincv::kResponseRingRows, static_cast<std::size_t>(W)};
        std::vector<Corner> streamCorners(candidateCount);
        const CornerResult streamResult = bincv::goodFeaturesToTrackStreaming(
            fe.dx[0], fe.dy[0], gftt, ring, streamCorners.data(), streamCorners.size());
        BINCV_CHECK_EQ(streamResult.count, refCount);
        BINCV_CHECK_EQ(streamResult.candidatesRanked, candidateCount);
        BINCV_CHECK_EQ(streamResult.candidatesTruncated, false);
        std::size_t differing = 0;
        for (std::size_t i = 0; i < candidateCount; ++i) {
            if (streamCorners[i].x != probe[i].x || streamCorners[i].y != probe[i].y ||
                streamCorners[i].response != probe[i].response) {
                ++differing;
            }
        }
        BINCV_CHECK_EQ(differing, std::size_t{0});
    }

    // ---- READING 1: the frame-map frontend's peak -------------------------
    // Sized to the MEASURED NMS survivor count, which is what T3.7 / X-19 did:
    // the array is also the candidate buffer, so an over-sized one would inflate
    // the footprint and an under-sized one would truncate.
    //
    // NOTHING INSIDE EITHER MEASURED WINDOW MAY ASSERT. `BINCV_CHECK_EQ` builds its
    // message eagerly -- `std::to_string(actual)` runs whether the check passes or
    // not -- so a check inside the window allocates a few hundred bytes and lands
    // in the high-water mark being read. That is not hypothetical: the first
    // version of this rewrite asserted inside the window and the two readings
    // disagreed by 313 B. The windows therefore only RECORD; every assertion is
    // made after the buffers are gone.
    std::size_t framePeak = 0;
    std::size_t cornerAllocs = 0;
    std::size_t trackAllocs = 0;
    std::size_t responseBytes = 0;
    std::size_t candidateBytes = 0;
    std::size_t frameCount = 0;
    std::size_t frameRanked = 0;
    bool frameTruncated = true;
    std::uint64_t frameDigest = 0;
    {
        armPeak();
        std::vector<float> responseStorage(static_cast<std::size_t>(W) *
                                               static_cast<std::size_t>(H), 0.0f);
        ResponseMap response{responseStorage.data(), static_cast<std::size_t>(W),
                             static_cast<std::size_t>(H), static_cast<std::size_t>(W)};
        std::vector<Corner> corners(candidateCount);
        const std::size_t beforeCorner = g_newCount;
        const CornerResult cornerResult = bincv::goodFeaturesToTrack(fe.dx[0], fe.dy[0], gftt,
                                                                    response, corners.data(),
                                                                    corners.size());
        cornerAllocs = g_newCount - beforeCorner;
        frameCount = cornerResult.count;
        frameRanked = cornerResult.candidatesRanked;
        frameTruncated = cornerResult.candidatesTruncated;
        frameDigest = cornerDigest(corners.data(), cornerResult.candidatesRanked);
        responseBytes = responseStorage.size() * sizeof(float);
        candidateBytes = corners.size() * sizeof(Corner);

        // Stage 5 runs here, on the frame-map form's corners, so the track stage's
        // buffers are inside the window that is measured rather than beside it.
        const std::size_t used = std::min<std::size_t>(cornerResult.count, POINTS);
        for (std::size_t i = 0; i < used; ++i) {
            prevPts[i] = Point2f{static_cast<float>(corners[i].x),
                                 static_cast<float>(corners[i].y)};
        }
        const std::size_t beforeTrack = g_newCount;
        bincv::calcOpticalFlowPyrLK<Word>(fe.levels.data(), fe.levels.size(), prevPts.data(),
                                          nextPts.data(), status.data(), errs.data(), used);
        trackAllocs = g_newCount - beforeTrack;
        // READ AT THE HIGH-WATER MARK, before anything in this scope is freed.
        framePeak = g_peakBytes - baseline;
    }
    BINCV_CHECK_EQ(frameTruncated, false);
    BINCV_CHECK_EQ(frameCount, refCount);
    BINCV_CHECK_EQ(frameRanked, candidateCount);
    BINCV_CHECK_EQ(frameDigest, refDigest);

    // ---- READING 2: the streaming frontend's peak -------------------------
    // THE FRAME MAP IS GONE. It was destroyed when the scope above closed, which is
    // the point: a table that prints a streaming peak while a 1 228 800 B float map
    // is still live is not measuring the streaming shape.
    std::size_t streamPeak = 0;
    std::size_t streamAllocs = 0;
    std::size_t streamTrackAllocs = 0;
    std::size_t ringBytes = 0;
    std::size_t streamCount = 0;
    std::size_t streamRanked = 0;
    bool streamTruncated = true;
    std::uint64_t streamDigest = 0;
    {
        armPeak();
        std::vector<float> ringStorage(bincv::kResponseRingRows * static_cast<std::size_t>(W),
                                       0.0f);
        ResponseMap ring{ringStorage.data(), static_cast<std::size_t>(W),
                         bincv::kResponseRingRows, static_cast<std::size_t>(W)};
        std::vector<Corner> streamCorners(candidateCount);
        const std::size_t beforeStream = g_newCount;
        const CornerResult streamResult = bincv::goodFeaturesToTrackStreaming(
            fe.dx[0], fe.dy[0], gftt, ring, streamCorners.data(), streamCorners.size());
        streamAllocs = g_newCount - beforeStream;
        streamCount = streamResult.count;
        streamRanked = streamResult.candidatesRanked;
        streamTruncated = streamResult.candidatesTruncated;
        streamDigest = cornerDigest(streamCorners.data(), streamResult.candidatesRanked);
        ringBytes = ringStorage.size() * sizeof(float);

        const std::size_t used = std::min<std::size_t>(streamResult.count, POINTS);
        for (std::size_t i = 0; i < used; ++i) {
            prevPts[i] = Point2f{static_cast<float>(streamCorners[i].x),
                                 static_cast<float>(streamCorners[i].y)};
        }
        const std::size_t beforeTrack = g_newCount;
        bincv::calcOpticalFlowPyrLK<Word>(fe.levels.data(), fe.levels.size(), prevPts.data(),
                                          nextPts.data(), status.data(), errs.data(), used);
        streamTrackAllocs = g_newCount - beforeTrack;
        streamPeak = g_peakBytes - baseline;
    }
    BINCV_CHECK_EQ(streamTruncated, false);
    BINCV_CHECK_EQ(streamCount, refCount);
    BINCV_CHECK_EQ(streamRanked, candidateCount);
    BINCV_CHECK_EQ(streamDigest, refDigest);

    // ---- the attribution, and the requirement that it ADD UP --------------
    const std::size_t wordBytes = sizeof(Word);
    std::size_t pyramidBytes = 0, derivativeBytes = 0;
    for (int i = 0; i < LEVELS; ++i) {
        pyramidBytes += (fe.prev[static_cast<std::size_t>(i)].sizeInWords() +
                         fe.next[static_cast<std::size_t>(i)].sizeInWords()) * wordBytes;
        derivativeBytes += (fe.dx[static_cast<std::size_t>(i)].sizeInWords() +
                            fe.dy[static_cast<std::size_t>(i)].sizeInWords()) * wordBytes;
    }
    const std::size_t denoiseBytes = (incoming.sizeInWords() + incomingNext.sizeInWords()) *
                                     wordBytes;
    const std::size_t cornerBytes = responseBytes + candidateBytes;
    const std::size_t trackBytes = (prevPts.size() + nextPts.size()) * sizeof(Point2f) +
                                   status.size() * sizeof(uint8_t) +
                                   errs.size() * sizeof(float);
    const std::size_t total = denoiseBytes + pyramidBytes + derivativeBytes + cornerBytes +
                              trackBytes;
    // THE CONTAINER BOOKKEEPING NO STAGE OWNS: the vectors of BinMat/TernaryMat
    // objects inside `Frontend` and its `levels` bundle. Named and printed rather
    // than folded into a stage, because the two readings must differ by the
    // response storage ALONE, and that is only checkable if the residual is stated.
    const std::size_t bookkeeping = framePeak - total;

    auto row = [&](const char* name, const char* what, std::size_t bytes) {
        std::printf("    %-12s %-46s %9zu B  %5.1f%%\n", name, what, bytes,
                    100.0 * static_cast<double>(bytes) / static_cast<double>(total));
    };
    std::printf("    %-12s %-46s %9s  %6s\n", "STAGE", "BUFFERS IT OWNS", "BYTES", "SHARE");
    row("denoise", "2 incoming frames, 1 bit/px (dst is pyramid L0)", denoiseBytes);
    row("pyramid", "2 x 4 levels, 1 bit/px", pyramidBytes);
    row("derivative", "dx+dy ternary, 2 bits/px, prev pyramid only", derivativeBytes);
    row("corner", "float response map + candidate array (see note)", cornerBytes);
    row("track", "prevPts/nextPts/status/err, 200 points", trackBytes);
    std::printf("    %-12s %-46s %9zu B\n", "TOTAL", "", total);
    std::printf("      MEASURED PEAK (live-byte high-water mark, harness baseline removed):"
                " %zu B\n      = the rows above + %zu B of container bookkeeping no stage owns"
                " (vectors of BinMat/TernaryMat and the LKLevel bundle)\n", framePeak,
                bookkeeping);
    std::printf("      of which the float response map alone: %zu B (%.1f%%), %zu candidates"
                " (%zu B)\n", responseBytes,
                100.0 * static_cast<double>(responseBytes) / static_cast<double>(total),
                candidateCount, candidateBytes);
    // THE ONE CONTENT-DEPENDENT TERM, SAID PLAINLY. Every other row is fixed by
    // the frame size; the candidate array is sized by how many NMS survivors this
    // frame happens to have, so the total is a PER-FRAME READING and not an upper
    // bound. A deployed caller cannot know the count in advance and must either
    // provision for it or accept truncation; provisioning the structural maximum
    // (W*H) would be 3 686 400 B and would make the candidate array, not the
    // response map, the dominant term. The real-frame case prints the same count
    // on decoded content, which is where the observed range comes from.
    std::printf("      the candidate array is the ONLY content-dependent row: %zu B here"
                " (%.1f%% of the total); provisioning W*H would be %zu B\n",
                candidateBytes, 100.0 * static_cast<double>(candidateBytes) /
                                    static_cast<double>(total),
                static_cast<std::size_t>(W) * static_cast<std::size_t>(H) * sizeof(Corner));
    std::printf("      operator new inside the kernels: denoise %zu, pyramid+derivative %zu,"
                " corner %zu, corner-streaming %zu, track %zu/%zu\n", denoiseAllocs, buildAllocs,
                cornerAllocs, streamAllocs, trackAllocs, streamTrackAllocs);

    // ---- THE SAME TABLE WITH T3.11's STREAMING CORNER STAGE ---------------
    // Same five stages, same accounting, one row replaced -- so the 71.4% row can
    // be read directly against its replacement rather than against a projection.
    //
    // The 16 B of carry the streaming form keeps for the two GLOBAL properties -- a
    // running maximum, a running retained count and the strongest discarded
    // response -- is NOT on the heap and therefore cannot appear in the reading. It
    // is added to the attributed total explicitly and labelled as the one term here
    // that is counted rather than read, because X-23 said every byte of carry comes
    // off the saving and rounding it away would be the easy lie.
    const std::size_t streamCarryBytes = 2 * sizeof(float) + sizeof(std::size_t);
    const std::size_t streamCornerBytes = ringBytes + candidateBytes + streamCarryBytes;
    const std::size_t streamTotal = denoiseBytes + pyramidBytes + derivativeBytes +
                                    streamCornerBytes + trackBytes;
    std::printf("\n    THE SAME FRONTEND WITH T3.11's STREAMING CORNER STAGE (identical corners,\n"
                "    asserted above -- count, coordinates, order and CornerResult):\n");
    std::printf("    %-12s %-46s %9s  %6s\n", "STAGE", "BUFFERS IT OWNS", "BYTES", "SHARE");
    auto srow = [&](const char* name, const char* what, std::size_t bytes) {
        std::printf("    %-12s %-46s %9zu B  %5.1f%%\n", name, what, bytes,
                    100.0 * static_cast<double>(bytes) / static_cast<double>(streamTotal));
    };
    srow("denoise", "2 incoming frames, 1 bit/px (dst is pyramid L0)", denoiseBytes);
    srow("pyramid", "2 x 4 levels, 1 bit/px", pyramidBytes);
    srow("derivative", "dx+dy ternary, 2 bits/px, prev pyramid only", derivativeBytes);
    srow("corner", "3-row float ring + candidate array + 3 scalars", streamCornerBytes);
    srow("track", "prevPts/nextPts/status/err, 200 points", trackBytes);
    std::printf("    %-12s %-46s %9zu B\n", "TOTAL", "", streamTotal);
    std::printf("      MEASURED PEAK (same high-water mark, frame map DESTROYED first):"
                " %zu B\n      = the rows above - %zu B of stack carry + %zu B of the same"
                " container bookkeeping\n", streamPeak, streamCarryBytes, bookkeeping);
    std::printf("      corner stage %zu B -> %zu B (%.2fx); frontend %zu B -> %zu B (%.2fx)\n",
                cornerBytes, streamCornerBytes,
                static_cast<double>(cornerBytes) / static_cast<double>(streamCornerBytes), total,
                streamTotal, static_cast<double>(total) / static_cast<double>(streamTotal));
    std::printf("      the ring is %zu B and the carry %zu B; the candidate array (%zu B) is now\n"
                "      the corner stage's dominant term, and it is the one content-dependent row\n",
                ringBytes, streamCarryBytes, candidateBytes);

    // THE ATTRIBUTION MUST ACCOUNT FOR THE READING, EXACTLY.
    //
    // This is what makes the two totals measurements rather than sums over a list
    // of buffers someone remembered to write down. `framePeak` and `streamPeak` are
    // high-water marks over LIVE BYTES: any heap buffer anywhere in the frontend --
    // including one acquired inside a kernel, which no enumeration can see -- lands
    // in them. Requiring the stage rows to reproduce them to the byte means a
    // buffer that appears without a row fails here.
    //
    // The residual is the same in both readings BY CHECK, not by assumption, so the
    // saving below is a difference of two measurements of the same thing.
    //
    // WHAT THIS READING STILL CANNOT SEE, SAID PLAINLY: it is a HEAP high-water
    // mark. A kernel that kept a frame-sized `static` array, or one that put a
    // frame-sized array on the stack, would not move it -- measured, by mutating
    // `goodFeaturesToTrackStreaming` to hold a `static float[640*480]`: the BSS of
    // the test binary grows from 648 B to 1 229 464 B and every number here is
    // unchanged (29/29 checks still pass). What
    // covers that is the no-heap rule's companion -- scratch is CALLER-PROVIDED
    // (D-5) -- and reading the header, not this case. A heap buffer anywhere,
    // including one allocated and freed INSIDE a kernel, does move it: the same
    // mutation spelled `new float[640*480] ... delete[]` fails the two checks below
    // (streaming peak 1 730 912 B against 502 112 B).
    BINCV_CHECK_EQ(framePeak, total + bookkeeping);
    BINCV_CHECK_EQ(streamPeak, streamTotal - streamCarryBytes + bookkeeping);
    BINCV_CHECK(bookkeeping < ringBytes);
    // The saving, read: the two windows differ by the response storage alone.
    BINCV_CHECK_EQ(framePeak - streamPeak, responseBytes - ringBytes);

    // X-23's saving gate, evaluated in the place that can actually fail: if a
    // later change puts the streaming frontend back above 750 000 B, D-22's
    // footprint claim has gone and this says so rather than a report nobody re-ran.
    BINCV_CHECK(streamTotal <= 750000);
    BINCV_CHECK(streamAllocs == 0);
    // E-10's sentence, NEGATED -- and negated in a form that can actually fail.
    // `ring < everything else` is near-vacuous at 7 680 B against 492 784 B and
    // tests nothing. These two do: the response storage must be smaller than every
    // other stage in the table, and the corner stage must no longer be the largest
    // stage in the frontend at all. Either is false the moment the ring grows with
    // the frame again.
    BINCV_CHECK(ringBytes < denoiseBytes && ringBytes < pyramidBytes &&
                ringBytes < derivativeBytes);
    BINCV_CHECK(ringBytes < candidateBytes);
    BINCV_CHECK(streamCornerBytes < derivativeBytes);

    // NO HEAP inside any kernel.
    BINCV_CHECK_EQ(denoiseAllocs, std::size_t{0});
    BINCV_CHECK_EQ(buildAllocs, std::size_t{0});
    BINCV_CHECK_EQ(cornerAllocs, std::size_t{0});
    BINCV_CHECK_EQ(trackAllocs, std::size_t{0});
    BINCV_CHECK_EQ(streamTrackAllocs, std::size_t{0});

    // E-10's prediction, pinned. T3.7's float response map is 4 B/pixel where
    // every other plane in the frontend is 1 or 2 BITS per pixel, so it is
    // expected to dominate. If a future change moves the dominant term, this
    // fails here rather than in a report nobody re-ran.
    // The assertion is the expression the line below PRINTS, candidate array
    // included. Spelled `pyramid + derivative + denoise + track` it omitted the
    // candidate row -- the second-largest term and the only one that moves with
    // content -- so it compared 1 228 800 B against a constant and could not fail
    // at this frame size for any content, while the printed sentence could flip.
    BINCV_CHECK(responseBytes > total - responseBytes);
    std::printf("      E-10: the response map %s the rest of the frontend combined.\n",
                (responseBytes > total - responseBytes) ? "EXCEEDS" : "does not exceed");

    // A SCALE REFERENCE, AND EXPLICITLY NOT CLAUDE.md's DENOMINATOR. CLAUDE.md
    // requires the comparison to be "OpenCV doing the same semantic operation on
    // the same binary content stored as CV_8U" -- for this frontend that is two
    // CV_8U frames, two winSize-padded CV_8U pyramids, CV_16S derivatives and the
    // same float response map, and none of it is computed here. Building and
    // measuring that pipeline is E-5 / T4.3. What the ratio below says is only
    // how the binCV frontend's peak compares with a single raw 640x480 frame, so
    // that the table has a familiar unit; it is not a memory win and must not be
    // quoted as one.
    const std::size_t oneByteFrame = static_cast<std::size_t>(W) * static_cast<std::size_t>(H);
    std::printf("      SCALE REFERENCE (not the CV_8U denominator, which is E-5/T4.3):"
                " whole frontend = %.2f x ONE raw CV_8U 640x480 frame (%zu B)\n",
                static_cast<double>(total) / static_cast<double>(oneByteFrame), oneByteFrame);
}

// ===========================================================================
// THE REAL FRAME -- same harness, same tolerance, decoded content
//
// READ THE VERDICT COLUMN. The tolerance stated at the top of this file is MET on
// synthetic texture and is NOT MET here, and this case is written to say so rather
// than to be widened until it passes. What it asserts is what it can assert
// without taking a decision this task is not authorised to take: that a stationary
// frame tracks EXACTLY, that the bit-parallel implementation agrees with the
// per-pixel float one on this content too -- and the direction and the SPLIT of
// the effects that explain the miss.
//
// THERE ARE TWO INDEPENDENT FAILURE MODES HERE, AND AN EARLIER VERSION OF THIS
// FILE ATTRIBUTED BOTH OF THEM TO THE PYRAMID. They are separated below because
// E-7 / T4.1 is scoped by which is which.
//
//   (1) A LEVEL-0 FAILURE MODE, WITH NO PYRAMID INVOLVED AT ALL. On this content
//       a 31x31 window at ONE level tracks an AXIS-ALIGNED 1 px translation to
//       0.002 px RMS -- and the DIAGONAL one, (1,1), to 0.75 px, with 28% of the
//       points returning EXACTLY zero flow while ground truth moved by 1.41 px.
//       Those points are a stationary point of the iteration, not a rejection:
//       `b1 = b2 = 0` at zero displacement, on a one-pixel-wide edge map whose
//       gradients along the two axes are the same pixels. The old banner's "level
//       0 tracks a 1-pixel translation to 0.002 px RMS" was true only of the
//       axis-aligned case it happened to measure. It is why the sub-pixel table
//       misses the tolerance at ONE level, where there is no pyramid to blame.
//
//   (2) A PYRAMID FAILURE MODE, WHICH IS REAL AND IS **PART** BIT DEPTH AND
//       **PART** CLIPPING. Accuracy degrades monotonically as 1-bit levels are
//       added: 0.0017 px RMS at one level to 3.25 px at four for a 1 px shift.
//       Two things move together there and this case measures them apart, by
//       running the SAME warp over the subset of points whose window is fully
//       inside EVERY level. On that subset the 1 px shift gives 1.47 px at four
//       levels instead of 3.25 -- so roughly half of the headline number is the
//       coarse-level window CLIPPING binCV chose over the reference's 1.24x padded
//       levels (deviation (ii)), an accuracy cost of that decision that had never
//       been measured. The other half survives the control: 0.0024 px at one level
//       to 1.47 px at four, on windows that never clip at any level, is still a
//       600x degradation and still six times the tolerance, so a level whose
//       pixels are BITS genuinely cannot localise sub-pixel motion, and E-7 /
//       T4.1 remains a precondition. What it is NOT is the whole story.
//
// AND THE REJECTION THRESHOLD REJECTS NOTHING HERE. `lk_min_eig_threshold: 0.001`
// against a smallest measured `referenceMinEig` of 0.033 on these points -- a
// factor of 33. Every one of 141 points comes back tracked in every case above,
// including the stuck ones, which is why T3 grew its second half (see the top of
// the file): a status byte is not evidence of tracking. Outside the blank-frame
// case in Flow.LossRules, loss rule 2 is untested on real content because it
// never fires on real content.
// ===========================================================================


// ---------------------------------------------------------------------------
// T4.1 / E-7: the N-BIT TRACKER.
//
// ops/opticalFlow.hpp grew a generic-N path so that E-7 can ask its question at
// all -- a pyramid level deeper than one bit was not previously trackable. Two
// things have to be true before any accuracy number measured on it means
// anything, and both are checked here rather than argued:
//
//   1. AT N == 1 THE GENERIC PATH MUST BE THE HAND-WRITTEN PATH. Not close --
//      identical. The 1-bit tracker is the one every result in X-20 was measured
//      on, so if the generic route disagreed with it at the depth both express,
//      every depth comparison would be measuring the rewrite rather than the
//      depth.
//   2. AT N > 1 THE BIT-SLICED RESIDUAL MUST BE EXACT. There is no hand-written
//      path to compare against there, so the control is a per-pixel loop in
//      `long long` that knows nothing about bit-slicing -- the same shape of
//      control `refTrack` is for the 1-bit residual.
// ---------------------------------------------------------------------------
namespace {

/// @brief Per-pixel `sum(V * G)` over a clipped window, in exact integers.
/// @note Deliberately naive and deliberately NOT bit-sliced: it reads pixel
///       VALUES through QuantMat::at and multiplies them. It reproduces
///       displacedRow's BORDER_REPLICATE by clamping, which is the one piece of
///       the kernel's behaviour a per-pixel control still has to model.
template <size_t N, typename WordType>
unsigned replicatedAt(const bincv::QuantMat<N, WordType>& m, long long x, long long y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > m.cols() - 1) x = m.cols() - 1;
    if (y > m.rows() - 1) y = m.rows() - 1;
    return m.at(static_cast<int>(y), static_cast<int>(x));
}

/// @brief The ten sums `impl::residualSums` produces, computed pixel by pixel.
template <size_t N, typename WordType>
void referenceResidualSums(const bincv::QuantMat<N, WordType>& prev,
                           const bincv::QuantMat<N, WordType>& next,
                           const bincv::SignedQuantMat<N, WordType>& dx,
                           const bincv::SignedQuantMat<N, WordType>& dy, const bincv::Rect& window,
                           long long tapX, long long tapY, long long (&outX)[5],
                           long long (&outY)[5]) {
    for (int k = 0; k < 5; ++k) { outX[k] = 0; outY[k] = 0; }
    const int x0 = std::max(0, window.x);
    const int y0 = std::max(0, window.y);
    const int x1 = std::min(prev.cols(), window.x + window.width);
    const int y1 = std::min(prev.rows(), window.y + window.height);
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const long long gx = dx.at(y, x);
            const long long gy = dy.at(y, x);
            const long long taps[5] = {
                replicatedAt(next, x + tapX, y + tapY),
                replicatedAt(next, x + tapX + 1, y + tapY),
                replicatedAt(next, x + tapX, y + tapY + 1),
                replicatedAt(next, x + tapX + 1, y + tapY + 1),
                prev.at(y, x)};
            for (int k = 0; k < 5; ++k) {
                outX[k] += taps[k] * gx;
                outY[k] += taps[k] * gy;
            }
        }
    }
}

/// @brief `2N^2` popcounts per word against `2N` multiplies per pixel, at one N.
template <size_t N, typename WordType>
size_t checkResidualAtDepth(uint64_t seed) {
    const int width = 77, height = 53;
    bincv::QuantMat<N, WordType> prev(width, height), next(width, height);
    const unsigned maxValue = (1u << N) - 1u;
    uint64_t state = seed;
    auto nextRandom = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<unsigned>(state >> 33);
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            prev.set(y, x, nextRandom() % (maxValue + 1u));
            next.set(y, x, nextRandom() % (maxValue + 1u));
        }
    }
    bincv::SignedQuantMat<N, WordType> dx(width, height), dy(width, height);
    bincv::derivativeX(prev, dx);
    bincv::derivativeY(prev, dy);
    const auto level = bincv::lkLevel<N>(prev, next, dx, dy);

    size_t mismatches = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const bincv::Rect window(static_cast<int>(nextRandom() % 60u) - 4,
                                 static_cast<int>(nextRandom() % 40u) - 4,
                                 3 + static_cast<int>(nextRandom() % 20u),
                                 3 + static_cast<int>(nextRandom() % 20u));
        // Taps run negative and past the right edge on purpose: that is the only
        // way the control exercises the replicate border the kernel implements
        // with mask-selects rather than with a clamp.
        const long long tapX = static_cast<long long>(nextRandom() % 11u) - 5;
        const long long tapY = static_cast<long long>(nextRandom() % 11u) - 5;
        const auto region = bincv::impl::clipRegion<WordType>(
            static_cast<size_t>(width), static_cast<size_t>(height), window);
        if (region.isEmpty) continue;

        bincv::impl::TapSums sumsX, sumsY;
        bincv::impl::residualSums(level, region, tapX, tapY, sumsX, sumsY);
        long long wantX[5], wantY[5];
        referenceResidualSums(prev, next, dx, dy, window, tapX, tapY, wantX, wantY);
        const long long gotX[5] = {sumsX.t00, sumsX.t01, sumsX.t10, sumsX.t11, sumsX.self};
        const long long gotY[5] = {sumsY.t00, sumsY.t01, sumsY.t10, sumsY.t11, sumsY.self};
        for (int k = 0; k < 5; ++k) {
            if (gotX[k] != wantX[k] || gotY[k] != wantY[k]) ++mismatches;
        }
    }
    return mismatches;
}

} // namespace

BINCV_TEST(Flow, GenericNAtOneBitIsTheHandWrittenPath_uint32_t) {
    // BinMat<W> IS QuantMat<1, W> and TernaryMat<W> IS SignedQuantMat<1, W>, so
    // both lkLevel overloads accept these arguments. Partial ordering picks the
    // MORE SPECIALIZED 1-bit one for a bare call; `lkLevel<1>` names the generic
    // one, because binding `1` to the 1-bit overload's `typename WordType` is
    // ill-formed and removes it from the set. That is what makes this comparison
    // possible at all, and it is checked here so a future change to either
    // signature fails loudly rather than silently comparing a path with itself.
    Frontend<uint32_t> fe(160, 120, 3);
    renderWarped(fe.prev[0], Warp{});
    renderWarped(fe.next[0], translation(1.3, -0.7));
    fe.build();

    static_assert(std::is_same<decltype(bincv::lkLevel(fe.prev[0], fe.next[0], fe.dx[0], fe.dy[0])),
                               bincv::LKLevel<uint32_t>>::value,
                  "a bare lkLevel call must select the 1-bit overload");
    static_assert(
        std::is_same<decltype(bincv::lkLevel<1>(fe.prev[0], fe.next[0], fe.dx[0], fe.dy[0])),
                     bincv::LKLevelN<1, uint32_t>>::value,
        "lkLevel<1> must select the generic-N overload");

    std::vector<bincv::LKLevelN<1, uint32_t>> generic;
    for (size_t i = 0; i < fe.levels.size(); ++i) {
        generic.push_back(bincv::lkLevel<1>(fe.prev[i], fe.next[i], fe.dx[i], fe.dy[i]));
    }

    std::vector<Point2f> points;
    for (int y = 24; y < 120 - 24; y += 6) {
        for (int x = 24; x < 160 - 24; x += 6) {
            points.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    std::vector<Point2f> handWritten(points.size()), genericOut(points.size());
    std::vector<uint8_t> handStatus(points.size()), genericStatus(points.size());
    std::vector<float> handErr(points.size()), genericErr(points.size());
    LKParams params;

    bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), points.data(),
                                          handWritten.data(), handStatus.data(), handErr.data(),
                                          points.size(), params);
    bincv::calcOpticalFlowPyrLK<1, uint32_t>(generic.data(), generic.size(), points.data(),
                                             genericOut.data(), genericStatus.data(),
                                             genericErr.data(), points.size(), params);

    size_t statusDiff = 0, positionDiff = 0, errDiff = 0, tracked = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (handStatus[i] != genericStatus[i]) ++statusDiff;
        if (handStatus[i] != 0) ++tracked;
        // EXACT equality, deliberately. Both paths do the same double arithmetic
        // in the same order; anything but a bit-for-bit match is a real difference.
        if (handWritten[i].x != genericOut[i].x || handWritten[i].y != genericOut[i].y) {
            ++positionDiff;
        }
        // `err` is the interesting one: the 1-bit path uses the collapsed identity
        // `|J - I| = I + (1 - 2I)*J` and the generic path evaluates |.| per pixel,
        // so this compares two DIFFERENT computations of the same quantity.
        if (handErr[i] != genericErr[i]) ++errDiff;
    }
    std::printf("  generic-N at N=1 vs hand-written: %zu points, %zu tracked,"
                " status/pos/err differences %zu/%zu/%zu\n",
                points.size(), tracked, statusDiff, positionDiff, errDiff);
    BINCV_CHECK(tracked > 100);
    BINCV_CHECK_EQ(statusDiff, size_t{0});
    BINCV_CHECK_EQ(positionDiff, size_t{0});
    BINCV_CHECK_EQ(errDiff, size_t{0});
}

BINCV_TEST(Flow, NBitResidualIsExactAgainstPerPixel_uint32_t) {
    // N = 1 is in the sweep even though the test above already covers that depth:
    // here it is the generic kernel against a per-pixel control rather than
    // against the other kernel, so a fault shared by both bit-sliced paths would
    // still show up.
    const size_t n1 = checkResidualAtDepth<1, uint32_t>(11);
    const size_t n2 = checkResidualAtDepth<2, uint32_t>(22);
    const size_t n3 = checkResidualAtDepth<3, uint32_t>(33);
    const size_t n4 = checkResidualAtDepth<4, uint32_t>(44);
    const size_t n5 = checkResidualAtDepth<5, uint32_t>(55);
    std::printf("  bit-sliced residual vs per-pixel, mismatching sums at N=1..5:"
                " %zu %zu %zu %zu %zu\n", n1, n2, n3, n4, n5);
    BINCV_CHECK_EQ(n1, size_t{0});
    BINCV_CHECK_EQ(n2, size_t{0});
    BINCV_CHECK_EQ(n3, size_t{0});
    BINCV_CHECK_EQ(n4, size_t{0});
    BINCV_CHECK_EQ(n5, size_t{0});
}

BINCV_TEST(Flow, X24_LadderSweep_Synthetic_uint32_t) {
    // X-24's synthetic half. X-20 PASSED its synthetic cases at four 1-bit levels;
    // the miss was on the reference pipeline's own edge maps. So this half is not
    // where the rule is decided -- it is the control that stops a ladder from
    // passing on real content by wrecking synthetic content, which the decision
    // rule requires explicitly.
    const int width = kW, height = kH;
    BinMat<uint32_t> prevSrc(width, height), nextSrc(width, height);
    const Warp warp = translation(1.3, -0.7);
    renderWarped(prevSrc, Warp{});
    renderWarped(nextSrc, warp);

    // The point set is IDENTICAL across ladders by construction: eligiblePoints
    // reads level 0's derivative, and level 0 is 1 bit in every ladder. That is
    // what makes the rows below comparable at all (band D of the rule).
    Frontend<uint32_t> base(width, height, 4);
    renderWarped(base.prev[0], Warp{});
    renderWarped(base.next[0], warp);
    base.build();
    LKParams params;
    const std::vector<Point2f> pts = eligiblePoints(base.dx[0], base.dy[0], width, height, warp,
                                                    params.winWidth, params.winHeight);
    std::printf("\n  X-24 synthetic %dx%d, translation (1.30, -0.70), %zu eligible points\n",
                width, height, pts.size());
    std::printf("  tolerance: rms <= %.4f, max <= %.4f (X-20's, inherited verbatim)\n",
                kRmsTolerance, kMaxTolerance);

    size_t b = 0;
    const FlowStats one   = runLadder<uint32_t, 1>            ("1 (1 level)", prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1111 = runLadder<uint32_t, 1, 1, 1, 1>   ("1/1/1/1",     prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1222 = runLadder<uint32_t, 1, 2, 2, 2>   ("1/2/2/2",     prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1333 = runLadder<uint32_t, 1, 3, 3, 3>   ("1/3/3/3",     prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1344 = runLadder<uint32_t, 1, 3, 4, 4>   ("1/3/4/4",     prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1355 = runLadder<uint32_t, 1, 3, 5, 5>   ("1/3/5/5",     prevSrc, nextSrc, warp, pts, 0.0, &b);
    const FlowStats l1357 = runLadder<uint32_t, 1, 3, 5, 7>   ("1/3/5/7",     prevSrc, nextSrc, warp, pts, 0.0, &b);

    // No tolerance is asserted here: this is a sweep, and X-24's rule is evaluated
    // on the real-frame half. What IS asserted is the precondition that makes the
    // sweep readable -- every ladder saw the same points and tracked enough of them.
    const FlowStats* all[] = {&one, &l1111, &l1222, &l1333, &l1344, &l1355, &l1357};
    for (const FlowStats* s : all) {
        BINCV_CHECK_EQ(s->eligible, pts.size());
        BINCV_CHECK(static_cast<double>(s->tracked) >=
                    kMinTrackedFraction * static_cast<double>(s->eligible));
    }
}

BINCV_TEST(Flow, MixedDepthLadderTracksAndIsNotTheUniformOne_uint32_t) {
    // The mixed-depth ladder is the form E-7's question needs, so it has to run
    // before E-7 can be measured. This checks the PLUMBING -- that every level is
    // visited coarse-to-fine at its own depth and that points come back tracked --
    // not the accuracy, which is X-24's to measure.
    const int width = 160, height = 120;
    bincv::QuantMat<1, uint32_t> p0(width, height), n0(width, height);
    bincv::QuantMat<3, uint32_t> p1(80, 60), n1(80, 60);
    bincv::QuantMat<4, uint32_t> p2(40, 30), n2(40, 30);
    // Level 0 is the frame; the coarse levels here are a decimation of it, which
    // is enough for a plumbing check and is NOT how X-24 will build them.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const bool a = field(static_cast<double>(x), static_cast<double>(y)) > 0.0;
            const bool b = field(static_cast<double>(x) - 1.0, static_cast<double>(y)) > 0.0;
            p0.set(y, x, a ? 1u : 0u);
            n0.set(y, x, b ? 1u : 0u);
        }
    }
    for (int y = 0; y < 60; ++y) {
        for (int x = 0; x < 80; ++x) {
            p1.set(y, x, static_cast<unsigned>(x + y) % 8u);
            n1.set(y, x, static_cast<unsigned>(x + y + 1) % 8u);
        }
    }
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 40; ++x) {
            p2.set(y, x, static_cast<unsigned>(x * 2 + y) % 16u);
            n2.set(y, x, static_cast<unsigned>(x * 2 + y + 1) % 16u);
        }
    }
    bincv::SignedQuantMat<1, uint32_t> dx0(width, height), dy0(width, height);
    bincv::SignedQuantMat<3, uint32_t> dx1(80, 60), dy1(80, 60);
    bincv::SignedQuantMat<4, uint32_t> dx2(40, 30), dy2(40, 30);
    bincv::derivativeX(p0, dx0); bincv::derivativeY(p0, dy0);
    bincv::derivativeX(p1, dx1); bincv::derivativeY(p1, dy1);
    bincv::derivativeX(p2, dx2); bincv::derivativeY(p2, dy2);

    bincv::LKLevels<uint32_t, 1, 3, 4> ladder;
    ladder.get<0>() = bincv::lkLevel<1>(p0, n0, dx0, dy0);
    ladder.get<1>() = bincv::lkLevel<3>(p1, n1, dx1, dy1);
    ladder.get<2>() = bincv::lkLevel<4>(p2, n2, dx2, dy2);
    static_assert(bincv::LKLevels<uint32_t, 1, 3, 4>::Levels == 3, "three levels");

    std::vector<Point2f> points;
    for (int y = 20; y < height - 20; y += 8) {
        for (int x = 20; x < width - 20; x += 8) {
            points.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    std::vector<Point2f> out(points.size());
    std::vector<uint8_t> outStatus(points.size());
    LKParams params;
    params.winWidth = 11;
    params.winHeight = 11;
    bincv::calcOpticalFlowPyrLK(ladder, points.data(), out.data(), outStatus.data(), nullptr,
                                points.size(), params);

    size_t tracked = 0, moved = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (outStatus[i] != 0) ++tracked;
        if (out[i].x != points[i].x || out[i].y != points[i].y) ++moved;
    }
    std::printf("  mixed 1/3/4 ladder: %zu points, %zu tracked, %zu moved\n", points.size(),
                tracked, moved);
    BINCV_CHECK(tracked > 0);
    BINCV_CHECK(moved > 0);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Flow, RealFrameWarps_uint32_t) {
    const cv::Mat gray = loadRealFrame();
    if (gray.empty()) {
        std::printf("  (skipped: sample image not found)\n");
        BINCV_CHECK(true);
        return;
    }
    const cv::Mat bin = referenceEdgeFilter(gray, 17);
    std::printf("\n  the repo's real test image, %dx%d, binarized by the reference\n"
                "  pipeline's own rl_fast_edge_filter_wide(edge_threshold = 17):"
                " %.2f%% set\n", gray.cols, gray.rows,
                100.0 * static_cast<double>(cv::countNonZero(bin)) /
                    static_cast<double>(gray.total()));

    LKParams params;
    const double halfWin = 0.5 * static_cast<double>(params.winWidth - 1);
    const double rotModel = halfWin * 1.0 * 3.14159265358979323846 / 180.0;
    const double scaleModel = halfWin * 0.02;

    // A stationary frame must track EXACTLY, at every level count. This is the
    // one real-frame property that is asserted without qualification, and it is
    // the one that caught the offset-origin bug ops/opticalFlow.hpp records: with
    // the tap displacement measured from the integer anchor instead of from
    // `prevPt`, a stationary point drifted by up to 1.4 px through four levels.
    for (int levels = 1; levels <= 4; ++levels) {
        char label[64];
        std::snprintf(label, sizeof(label), "real: stationary, %d level(s)", levels);
        const FlowStats s =
            runRealFrameCase<uint32_t>(gray, label, translation(0.0, 0.0), 0.0, levels, true);
        BINCV_CHECK_EQ(s.maxError, 0.0);
    }

    // ---- failure mode (1): ONE level, no pyramid -------------------------
    // `enforce` is false: see the banner above.
    std::printf("\n  ONE 1-bit level (level 0 only) -- NO PYRAMID EXISTS IN ANY ROW BELOW:\n");
    const FlowStats one025 = runRealFrameCase<uint32_t>(gray, "real: shift (0.25, 0.25)",
                                                        translation(0.25, 0.25), 0.0, 1, false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0.25, 0)", translation(0.25, 0.0), 0.0, 1,
                               false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0.50, 0.50)", translation(0.50, 0.50), 0.0, 1,
                               false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0.75, 0.75)", translation(0.75, 0.75), 0.0, 1,
                               false);
    // THE AXIS-ALIGNED / DIAGONAL SPLIT, which is the point of this block. The
    // first two rows are the near-exact special case an earlier version of this
    // file reported as if it were the general one; the last two are the same
    // content, the same harness and the same single level.
    const FlowStats oneAxis =
        runRealFrameCase<uint32_t>(gray, "real: shift (1, 0)", translation(1.0, 0.0), 0.0, 1,
                                   false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0, 1)", translation(0.0, 1.0), 0.0, 1, false);
    const FlowStats oneDiag =
        runRealFrameCase<uint32_t>(gray, "real: shift (1, 1)", translation(1.0, 1.0), 0.0, 1,
                                   false);
    const FlowStats oneDiag2 =
        runRealFrameCase<uint32_t>(gray, "real: shift (2, 2)", translation(2.0, 2.0), 0.0, 1,
                                   false);

    // PINNED: the level-0 failure mode exists and is NOT the pyramid. A diagonal
    // integer translation -- the easiest case there is, exactly recoverable by an
    // integer search -- misses both halves of the tolerance at one level, and it
    // does so by leaving points exactly where they started.
    std::printf("\n  LEVEL 0, NO PYRAMID: axis-aligned (1,0) rms %.4f  vs  diagonal (1,1)"
                " rms %.4f, %zu/%zu points stuck at zero flow\n",
                oneAxis.rms, oneDiag.rms, oneDiag.stuck, oneDiag.truthMoved);
    BINCV_CHECK(oneAxis.rms <= kRmsTolerance);      // the special case, still special
    BINCV_CHECK(oneDiag.rms > kRmsTolerance);       // and the general one, still missing
    BINCV_CHECK(oneDiag.maxError > kMaxTolerance);
    BINCV_CHECK(oneDiag.stuck > 0);                 // by not moving at all
    BINCV_CHECK(oneDiag2.stuck > 0);

    // ---- failure mode (2): the pyramid, and the clipping control ---------
    std::printf("\n  FOUR 1-bit levels -- seal_params.yaml's lk_max_level 3:\n");
    const FlowStats four025 = runRealFrameCase<uint32_t>(gray, "real: shift (0.25, 0.25)",
                                                         translation(0.25, 0.25), 0.0, 4, false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0.50, 0.50)", translation(0.50, 0.50), 0.0, 4,
                               false);
    runRealFrameCase<uint32_t>(gray, "real: shift (0.75, 0.75)", translation(0.75, 0.75), 0.0, 4,
                               false);
    const FlowStats four100 =
        runRealFrameCase<uint32_t>(gray, "real: shift (1, 0)", translation(1.0, 0.0), 0.0, 4,
                                   false);
    runRealFrameCase<uint32_t>(gray, "real: shift (2, -3)", translation(2.0, -3.0), 0.0, 4, false);
    runRealFrameCase<uint32_t>(gray, "real: rotate 1 deg",
                               rotation(1.0, gray.cols * 0.5, gray.rows * 0.5), rotModel, 4,
                               false);
    runRealFrameCase<uint32_t>(gray, "real: scale 1.02",
                               scaling(1.02, gray.cols * 0.5, gray.rows * 0.5), scaleModel, 4,
                               false);

    // THE FINDING, PINNED. Adding 1-bit pyramid levels makes this content WORSE,
    // by a margin far outside any run-to-run variation -- there is none, the whole
    // computation is deterministic. If a future change (an N-bit level, E-7's
    // answer) reverses it, this fails and the banner above has to be rewritten,
    // which is the point.
    std::printf("\n  1 level -> 4 levels:  q=0.25 rms %.4f -> %.4f,  1 px rms %.4f -> %.4f\n",
                one025.rms, four025.rms, oneAxis.rms, four100.rms);
    BINCV_CHECK(four025.rms > one025.rms);
    BINCV_CHECK(four100.rms > oneAxis.rms);

    // AND HOW MUCH OF IT IS THE CLIPPED WINDOW RATHER THAN THE BIT DEPTH. One
    // fixed point set -- the windows that are inside every one of the four levels
    // -- evaluated at each level count, against the same rows above. Same content,
    // same warps, same kernels; the only thing removed is deviation (ii).
    std::printf("\n  THE CLIPPING CONTROL: the same warps over the points whose 31x31 window\n"
                "  is inside EVERY level, so no window clips at any level:\n");
    double subsetOne = 0.0, subsetFour = 0.0;
    for (int levels = 1; levels <= 4; ++levels) {
        Frontend<uint32_t> fe(gray.cols, gray.rows, levels);
        buildRealFrontend(gray, translation(1.0, 0.0), levels, fe);
        Frontend<uint32_t> deepest(gray.cols, gray.rows, 4);
        buildRealFrontend(gray, translation(1.0, 0.0), 4, deepest);
        const std::vector<Point2f> all =
            eligiblePoints(fe.dx[0], fe.dy[0], gray.cols, gray.rows, translation(1.0, 0.0),
                           params.winWidth, params.winHeight);
        const std::vector<Point2f> inside =
            unclippedAtEveryLevel(deepest, all, params.winWidth, params.winHeight);
        char label[64];
        // Both columns at every level count, so the comparison is a table in the
        // suite's own output rather than two numbers from two different runs.
        std::snprintf(label, sizeof(label), "real: (1,0), %d level(s), ALL", levels);
        runPoints<uint32_t>(label, fe, translation(1.0, 0.0), all, 0.0, false);
        std::snprintf(label, sizeof(label), "real: (1,0), %d level(s), unclipped", levels);
        const FlowStats s =
            runPoints<uint32_t>(label, fe, translation(1.0, 0.0), inside, 0.0, false);
        if (levels == 1) subsetOne = s.rms;
        if (levels == 4) subsetFour = s.rms;
    }
    std::printf("  1 px shift at FOUR levels: all %zu points rms %.4f, the %s subset rms %.4f\n",
                four100.eligible, four100.rms, "unclipped-at-every-level", subsetFour);

    // PINNED, BOTH DIRECTIONS. Removing the clipping removes a large part of the
    // error -- so the attribution "1-bit levels and nothing else" is wrong -- and
    // what is left still degrades by orders of magnitude with level count and is
    // still nowhere near the tolerance -- so the 1-bit level is a real cause and
    // T4.1 is still a precondition. Both halves have to hold or the diagnosis in
    // TASKS.md / EXPERIMENTS.md X-20 is wrong and has to be rewritten.
    BINCV_CHECK(subsetFour < four100.rms);          // clipping contributes
    BINCV_CHECK(subsetFour > kRmsTolerance);        // and is not the whole cause
    BINCV_CHECK(subsetFour > subsetOne);            // depth degrades unclipped windows too
    BINCV_CHECK(subsetOne <= kRmsTolerance);

    // ---- what the loss rules and the footprint table need said out loud ---
    {
        Frontend<uint32_t> fe(gray.cols, gray.rows, 1);
        buildRealFrontend(gray, translation(0.5, 0.5), 1, fe);
        const std::vector<Point2f> pts =
            eligiblePoints(fe.dx[0], fe.dy[0], gray.cols, gray.rows, translation(0.5, 0.5),
                           params.winWidth, params.winHeight);
        const double smallest = smallestReferenceMinEig(fe, pts, params);
        std::printf("\n  LOSS RULE 2 ON THIS CONTENT: smallest referenceMinEig over %zu points"
                    " = %.4f,\n  threshold = %.4f -> it rejects NOTHING here (factor %.0f of"
                    " headroom)\n", pts.size(), smallest,
                    static_cast<double>(params.minEigThreshold),
                    smallest / static_cast<double>(params.minEigThreshold));
        BINCV_CHECK(smallest > static_cast<double>(params.minEigThreshold));

        // The content-dependent term of the footprint table, on decoded content,
        // so that Flow.FrontendFootprint_640x480's per-frame candidate count has a
        // measured range rather than a single reading.
        std::vector<float> mapStorage(static_cast<std::size_t>(gray.cols) *
                                          static_cast<std::size_t>(gray.rows), 0.0f);
        ResponseMap map{mapStorage.data(), static_cast<std::size_t>(gray.cols),
                        static_cast<std::size_t>(gray.rows), static_cast<std::size_t>(gray.cols)};
        std::vector<Corner> found(static_cast<std::size_t>(gray.cols) *
                                  static_cast<std::size_t>(gray.rows));
        GoodFeaturesParams gftt;
        const CornerResult r =
            bincv::goodFeaturesToTrack(fe.dx[0], fe.dy[0], gftt, map, found.data(), found.size());
        BINCV_CHECK_EQ(r.candidatesTruncated, false);
        std::printf("  FOOTPRINT RANGE: %zu NMS survivors on this %dx%d real frame"
                    " (%zu B of candidate array); the 640x480 table reports its own\n",
                    r.candidatesRanked, gray.cols, gray.rows,
                    r.candidatesRanked * sizeof(Corner));
    }

    // AND THE BIT-PARALLEL IMPLEMENTATION IS NOT WHAT IS LOSING THE ACCURACY. The
    // per-pixel float implementation, on this same content, must agree with it.
    {
        Frontend<uint32_t> fe(gray.cols, gray.rows, 1);
        buildRealFrontend(gray, translation(0.5, 0.5), 1, fe);
        std::vector<Point2f> pts = eligiblePoints(fe.dx[0], fe.dy[0], gray.cols, gray.rows,
                                                  translation(0.5, 0.5), params.winWidth,
                                                  params.winHeight);
        BINCV_CHECK(pts.size() >= 16);
        std::vector<Point2f> got(pts.size());
        std::vector<uint8_t> gotStatus(pts.size());
        bincv::calcOpticalFlowPyrLK<uint32_t>(fe.levels.data(), fe.levels.size(), pts.data(),
                                              got.data(), gotStatus.data(), nullptr, pts.size(),
                                              params);
        std::vector<Point2f> want;
        std::vector<uint8_t> wantStatus;
        refTrack(fe, pts, want, wantStatus, params);
        double worst = 0.0;
        size_t statusMismatches = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (gotStatus[i] != wantStatus[i]) ++statusMismatches;
            const double ex = static_cast<double>(got[i].x) - static_cast<double>(want[i].x);
            const double ey = static_cast<double>(got[i].y) - static_cast<double>(want[i].y);
            worst = std::max(worst, std::sqrt(ex * ex + ey * ey));
        }
        std::printf("  real frame, popcount residual vs per-pixel float: %zu points,"
                    " worst %.3e px, %zu status mismatches\n", pts.size(), worst,
                    statusMismatches);
        BINCV_CHECK_EQ(statusMismatches, size_t{0});
        BINCV_CHECK(worst < 0.01);
    }
}

// ---------------------------------------------------------------------------
// X-24 / E-7 -- THE MEASUREMENT THE RULE IS DECIDED ON.
//
// These are X-20's own failing rows, re-run at every ladder. The tolerance,
// the binarization, the warps, the eligibility rule and the stuck rule are all
// X-20's, reached through the same functions -- nothing here is re-derived.
// ---------------------------------------------------------------------------
namespace {

/// @brief How many DISTINCT values a level actually holds, against how many its
///        declared depth could hold. X-2's question, asked of the real path.
template <size_t N, typename WordType>
void printLevelAlphabet(const bincv::QuantMat<N, WordType>& level, int index) {
    std::vector<size_t> counts(size_t{1} << N, 0);
    for (int y = 0; y < level.rows(); ++y) {
        for (int x = 0; x < level.cols(); ++x) counts[level.at(y, x)]++;
    }
    size_t distinct = 0, set = 0;
    for (size_t v = 0; v < counts.size(); ++v) {
        if (counts[v] != 0) ++distinct;
        if (v != 0) set += counts[v];
    }
    std::printf("    L%d  %4dx%-4d declared N=%zu (%zu values)  ACTUALLY USED %2zu"
                "  non-zero %6zu / %zu (%.2f%%)\n",
                index, level.cols(), level.rows(), N, counts.size(), distinct, set,
                static_cast<size_t>(level.cols()) * static_cast<size_t>(level.rows()),
                100.0 * static_cast<double>(set) /
                    static_cast<double>(level.cols() * level.rows()));
}

template <typename WordType>
void x24RealCase(const cv::Mat& gray, const char* label, const Warp& warp, double modelError,
                 bool unclippedOnly = false) {
    cv::Mat warped;
    cv::warpAffine(gray, warped, affineOf(warp), gray.size(), cv::INTER_CUBIC,
                   cv::BORDER_REFLECT_101);
    const cv::Mat bin0 = referenceEdgeFilter(gray, 17);
    const cv::Mat bin1 = referenceEdgeFilter(warped, 17);

    BinMat<WordType> prevSrc(gray.cols, gray.rows), nextSrc(gray.cols, gray.rows);
    prevSrc.fromCVMat(bin0);
    nextSrc.fromCVMat(bin1);

    // The point set comes from LEVEL 0, which is 1 bit in every ladder, so every
    // row below is measured over the SAME points. Band D of X-24's rule exists
    // because a curve over different point sets is not a curve.
    Frontend<WordType> base(gray.cols, gray.rows, 4);
    base.prev[0].fromCVMat(bin0);
    base.next[0].fromCVMat(bin1);
    base.build();
    LKParams params;
    std::vector<Point2f> pts = eligiblePoints(base.dx[0], base.dy[0], gray.cols, gray.rows,
                                              warp, params.winWidth, params.winHeight);
    // X-20's own control for deviation (ii), applied HERE because it is the one
    // thing that could hide a depth effect: it attributed about half the
    // four-level error to the clipped coarse-level window, and a window that is
    // half outside the level is not measuring that level's ALPHABET.
    if (unclippedOnly) {
        pts = unclippedAtEveryLevel(base, pts, params.winWidth, params.winHeight);
    }

    std::printf("\n  %s  --  %zu eligible points, tol rms<=%.4f max<=%.4f\n", label, pts.size(),
                kRmsTolerance + modelError, kMaxTolerance + modelError);
    size_t b = 0;
    runLadder<WordType, 1>            ("1 (1 level)", prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 1, 1, 1>   ("1/1/1/1",     prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 2, 2, 2>   ("1/2/2/2",     prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 3, 3, 3>   ("1/3/3/3",     prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 3, 4, 4>   ("1/3/4/4",     prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 3, 5, 5>   ("1/3/5/5",     prevSrc, nextSrc, warp, pts, modelError, &b);
    runLadder<WordType, 1, 3, 5, 7>   ("1/3/5/7",     prevSrc, nextSrc, warp, pts, modelError, &b);
}

} // namespace

BINCV_TEST(Flow, X24_LadderSweep_RealFrame_uint32_t) {
    const cv::Mat gray = loadRealFrame();
    if (gray.empty()) {
        std::printf("  (skipped: sample image not found)\n");
        BINCV_CHECK(true);
        return;
    }
    LKParams params;
    const double halfWin = 0.5 * static_cast<double>(params.winWidth - 1);
    const double rotModel = halfWin * 1.0 * 3.14159265358979323846 / 180.0;
    const double scaleModel = halfWin * 0.02;

    std::printf("\n  ===================================================================\n"
                "  X-24 / E-7: pyramid level bit depths, on the reference pipeline's\n"
                "  own edge maps -- THE CONFIGURATION X-20 MISSED ON.\n"
                "  Tolerance is X-20's, inherited verbatim: rms <= %.4f, max <= %.4f.\n"
                "  `bytes` is both pyramids plus both derivative ladders -- a PEAK,\n"
                "  since the tracker reads all of them.\n"
                "  ===================================================================\n",
                kRmsTolerance, kMaxTolerance);

    x24RealCase<uint32_t>(gray, "real: stationary",        translation(0.0, 0.0), 0.0);
    x24RealCase<uint32_t>(gray, "real: shift (1, 0)",      translation(1.0, 0.0), 0.0);
    x24RealCase<uint32_t>(gray, "real: shift (0.25, 0.25)", translation(0.25, 0.25), 0.0);
    x24RealCase<uint32_t>(gray, "real: shift (0.50, 0.50)", translation(0.50, 0.50), 0.0);
    x24RealCase<uint32_t>(gray, "real: shift (0.75, 0.75)", translation(0.75, 0.75), 0.0);
    x24RealCase<uint32_t>(gray, "real: shift (2, -3)",     translation(2.0, -3.0), 0.0);
    x24RealCase<uint32_t>(gray, "real: rotate 1 deg",      rotation(1.0, gray.cols * 0.5,
                                                                    gray.rows * 0.5), rotModel);
    x24RealCase<uint32_t>(gray, "real: scale 1.02",        scaling(1.02, gray.cols * 0.5,
                                                                   gray.rows * 0.5), scaleModel);

    // THE CLIPPING CONTROL. If depth helps anywhere, it should help here: these
    // are the points whose 31x31 window is fully inside EVERY level, so the
    // coarse-level reading is of the level's alphabet and not of its border.
    std::printf("\n  ---- restricted to points that never clip at ANY level ----\n");
    x24RealCase<uint32_t>(gray, "unclipped: shift (1, 0)", translation(1.0, 0.0), 0.0, true);
    x24RealCase<uint32_t>(gray, "unclipped: shift (0.25, 0.25)", translation(0.25, 0.25), 0.0,
                          true);
    x24RealCase<uint32_t>(gray, "unclipped: shift (0.75, 0.75)", translation(0.75, 0.75), 0.0,
                          true);

    // THE DISCRIMINATING CASE. Everything above is a motion a single level can
    // already handle, so a ladder can score well there by CONTRIBUTING NOTHING.
    // These are displacements a one-level tracker cannot follow -- the pyramid
    // has to do real work -- which is the only regime in which "this ladder is
    // better" means "this ladder tracks", rather than "this ladder does least
    // harm".
    std::printf("\n  ---- unclipped, LARGE motion: the pyramid must actually work ----\n");
    x24RealCase<uint32_t>(gray, "unclipped: shift (2, -3)", translation(2.0, -3.0), 0.0, true);
    x24RealCase<uint32_t>(gray, "unclipped: shift (6, 4)", translation(6.0, 4.0), 0.0, true);
    x24RealCase<uint32_t>(gray, "unclipped: shift (12, -8)", translation(12.0, -8.0), 0.0, true);

    // T4.1's other deliverable: RE-RUN X-2 AGAINST THE REAL PYRAMID PATH.
    // X-2 read the natural alphabet as 1/3/4/5 from one 256^2 frame; X-15
    // corrected it to 1/3/5/7 from the representation. This measures what the
    // uncapped ladder ACTUALLY holds on the reference pipeline's own edge map,
    // which is the content the frontend sees, and closes X-2's caveat.
    {
        bincv::Pyramid<uint32_t, 1, 3, 5, 7> deep(gray.cols, gray.rows);
        const cv::Mat bin0 = referenceEdgeFilter(gray, 17);
        deep.level<0>().fromCVMat(bin0);
        deep.build();
        std::printf("\n  ---- X-2 re-run: the UNCAPPED ladder's real alphabet ----\n");
        printLevelAlphabet(deep.level<0>(), 0);
        printLevelAlphabet(deep.level<1>(), 1);
        printLevelAlphabet(deep.level<2>(), 2);
        printLevelAlphabet(deep.level<3>(), 3);
    }

    // No pass/fail asserted here: X-24's rule is evaluated in EXPERIMENTS.md from
    // the whole curve, and asserting a band inside the sweep that produces it
    // would be deciding the experiment from inside the measurement.
    BINCV_CHECK(true);
}

#endif // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_opticalflow")
