#pragma once

/// @file subpix.hpp
/// @brief Sub-pixel corner refinement on bit-planes. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// WHAT IS cv::cornerSubPix's, AND WHY IT CAN RUN ON TERNARY DERIVATIVES
///
/// `cv::cornerSubPix` refines a corner by solving, over a window around it,
///
/// G q = b, G = sum w (grad I)(grad I)^T, b = sum w (grad I)(grad I)^T p
///
/// for the position `q` that every edge in the window points at. Its gradient is the
/// central difference `I(x+1) - I(x-1)`, taken on the image bilinearly resampled at the
/// corner's SUB-PIXEL position (`cv::getRectSubPix`, replicated border).
///
/// Two linear facts make that computable from binCV's derivative planes, with no 8-bit
/// image:
///
/// 1. **Resampling and central differencing commute.** Every sample of the window shares
///    the corner's fractional part `(a, b)`, so the gradient of the resampled image at
///    window offset `p` is the same bilinear blend of the four integer-pixel central
///    differences around it --
///    `gx = (1-a)(1-b) dx[q] + a(1-b) dx[q + (1,0)] + (1-a)b dx[q + (0,1)] + ab dx[q + (1,1)]`
///    with `q` the integer pixel at or before the sample -- and likewise `gy`. On a binary image `dx`, `dy` are exactly the ternary planes
///    ops/derivative.hpp writes.
/// 2. **The solution is invariant to a scale on the image**: scaling `I` by `s` scales
///    `G` and `b` by `s^2`, and `q = G^-1 b` is unchanged. A ternary plane carries `+/-1`
///    where OpenCV's 0/255 image carries `+/-255`; the answer is the same to rounding.
///
/// So the window, its Gaussian weights, the zero zone, the iteration, the termination,
/// the replicated border and the poor-convergence rule are all `cv::cornerSubPix`'s, and
/// the answer is its answer on the same bits.
///
/// **A WINDOW ANCHORED ON THE ROUNDED POSITION IS A DIFFERENT REFINEMENT, NOT AN
/// APPROXIMATION OF THIS ONE.** Re-anchoring every iteration on `round(c)` moves each
/// step's sampling grid by up to half a pixel, and the walks end in different places:
/// on 34,200 corners of 171 binarized EuRoC frames, a median of 0.73 px from
/// `cv::cornerSubPix` at this struct's defaults and 1.08 px at window 21, p99 5.5 and
/// 11.3 px. The bilinear blend is what removes that.
///
/// ---------------------------------------------------------------------------
/// WHY IT IS TIER 2 AND NOT TIER 1
///
/// OpenCV resamples in `float` and sums 0/255 values; this blends exact ternary values
/// in `double`. The two agree to rounding, not to the bit, and a rounding difference at
/// the `epsilon` test or at the poor-convergence threshold occasionally ends a walk one
/// step apart. OpenCV's own versions differ by more than that: 4.5.4's 8-bit resampler
/// clamps the horizontal fraction to at least 1e-4, and 4.8.0's reads column
/// `width - 2` for the part of a window above the image and right of its last column.
/// This computes the exact blend both approximate. On the frames above
/// (benchmark/subpix_agreement.cpp), the share of corners within 1e-3 px:
///
/// | | defaults | window 21, eps 0.03 |
/// |---|---|---|
/// | binCV against 4.8.0 | 99.96% | 99.85% |
/// | 4.8.0 against itself, 0/1 against 0/255 | 99.96% | 99.89% |
/// | binCV against 4.5.4 | 99.28% | 95.67% |
/// | 4.5.4 against 4.8.0 | 99.28% | 95.66% |
///
/// A loose epsilon is what turns 4.5.4's 1e-4 px first step from an integer seed into a
/// walk that stops a step apart.
///
/// ---------------------------------------------------------------------------
/// WHAT BIT-PLANES BUY HERE
///
/// The weights and the positions are per-pixel scalars, so unlike every reduction in
/// this library this one does **not** collapse into popcounts -- a weighted sum is not
/// a bit count. What bit-planes buy is the SKIP: a window sample whose four neighbouring
/// integer pixels have no gradient in either axis contributes nothing to any of the five
/// sums, and `|dx| | |dy|`, OR-ed with itself one column and one row on, finds every
/// such sample a word at a time. How many that removes is the content's: around the
/// corners a detector finds on the committed real frame, 19% of an 11x11 window and 28%
/// of a 21x21 one. The rest of the cost is the samples themselves, and there the planes
/// buy the input -- the derivatives a pipeline already holds, where `cv::cornerSubPix`
/// resamples an 8-bit frame on every iteration.

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../quantMat.hpp"
#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../impl/kernel_util.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief `cv::cornerSubPix`'s `winSize`, `zeroZone` and `criteria`, in one struct.
/// **API TIER 2** -- the same meanings `cv::cornerSubPix` gives them.
struct SubPixParams {
    /// Half-width of the search window; the window is `(2*winHalf + 1)` square.
    /// OpenCV's `winSize(5,5)` -- ORB-SLAM's choice -- is `winHalf = 5`.
    int winHalf = 5;
    /// Half-width of a central region excluded from the sums, or **-1 for none**.
    /// OpenCV's `zeroZone`, applied only when `zeroHalf < winHalf`, as OpenCV applies it.
    int zeroHalf = -1;
    /// `TermCriteria::MAX_ITER`, taken into `[1, 100]` as `cv::cornerSubPix` takes it.
    int maxIterations = 40;
    double epsilon = 0.001;      ///< `TermCriteria::EPS`, in pixels, on the step
};

/// @brief What `cornerSubPix` did, per corner. **API TIER 3** -- OpenCV reports nothing.
/// Every corner is counted in exactly one of the four.
struct SubPixResult {
    /// Corners whose walk converged (a step within `epsilon`) or ran its iterations.
    size_t refined = 0;
    /// Corners whose walk stopped on a `G` with `|det| <= DBL_EPSILON^2` -- OpenCV's
    /// test -- where they had got to. A corner whose first window is flat is untouched.
    size_t singular = 0;
    /// Corners whose walk stepped outside the image and stopped THERE, as
    /// `cv::cornerSubPix`'s does: the position returned is outside the image unless the
    /// poor-convergence rule reverts it. A seed outside the image is counted here too,
    /// and is not moved.
    size_t outside = 0;
    /// @brief Corners that ended further than `winHalf` from their seed in either axis,
    /// and were therefore REVERTED to the seed.
    ///
    /// **This is `cv::cornerSubPix`'s own rule, not an addition** -- "if new point is too
    /// far from initial, it means poor convergence; leave initial point as the result".
    size_t diverged = 0;
};

namespace impl {

/// @brief `cv::cornerSubPix`'s Gaussian window mask, in its own `float` arithmetic.
/// **INTERNAL.**
///
/// @note **`exp(-(dx^2 + dy^2) / winHalf^2)`, and the width is not a free parameter.**
/// OpenCV normalizes each offset by the half-window and exponentiates the sum of
/// squares -- `vy = exp(-y*y)` with `y = (i - win.height)/win.height`, times the same in
/// x -- so the weight is exactly 1/e at the edge of the window along either axis. Built
/// with the same `float` expressions, so the weights are OpenCV's to the bit wherever
/// the C library's `expf` is the same.
/// @note The zero zone is a HOLE, not a smaller window: OpenCV keeps the outer ring and
/// drops the middle -- and drops nothing when the hole would be the whole window.
inline void subPixMask(int winHalf, int zeroHalf, float* mask) {
    const int side = 2 * winHalf + 1;
    for (int i = 0; i < side; ++i) {
        const float y = static_cast<float>(i - winHalf) / static_cast<float>(winHalf);
        const float vy = std::exp(-y * y);
        for (int j = 0; j < side; ++j) {
            const float x = static_cast<float>(j - winHalf) / static_cast<float>(winHalf);
            mask[i * side + j] = vy * std::exp(-x * x);
        }
    }
    if (zeroHalf >= 0 && zeroHalf < winHalf) {
        for (int i = winHalf - zeroHalf; i <= winHalf + zeroHalf; ++i) {
            for (int j = winHalf - zeroHalf; j <= winHalf + zeroHalf; ++j) {
                mask[i * side + j] = 0.0f;
            }
        }
    }
}

/// The largest window this operation will build a mask for, so the mask is a stack
/// buffer -- `float[31 * 31]`, 3,844 B -- and the kernel allocates nothing (no heap
/// in a kernel).
inline constexpr int kMaxWinHalf = 15;

/// `cv::cornerSubPix`'s cap on the iteration count (`MAX_ITERS`).
inline constexpr int kSubPixMaxIterations = 100;

/// @brief Index of the lowest set bit of a NON-ZERO word, without any compiler
/// builtin. **INTERNAL.**
/// @note The MSVC and no-builtin fallback, for the reason ops/reduce.hpp keeps
/// `popcountWordPortable`: the library claims a C++17 compiler and nothing
/// else, and `__builtin_ctzll` alone would make this header GCC/clang-only.
/// A loop, because it runs only where the builtin does not exist and the
/// caller has already tested the word for zero.
inline size_t ctzPortable(unsigned long long v) {
    size_t n = 0;
    while ((v & 1ull) == 0ull) {
        v >>= 1;
        ++n;
    }
    return n;
}

/// @brief Index of the lowest set bit of a NON-ZERO word. **INTERNAL.**
inline size_t ctzWord(unsigned long long v) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<size_t>(__builtin_ctzll(v));
#else
    return ctzPortable(v);
#endif
}

/// @brief The five weighted sums of one window. **INTERNAL.**
struct SubPixSums {
    double gxx = 0.0, gxy = 0.0, gyy = 0.0, bx = 0.0, by = 0.0;

    /// @brief Adds one window sample: gradient `(gx, gy)`, weight `m`, offset `(px, py)`
    /// from the sampling centre. `cv::cornerSubPix`'s expressions, in its order.
    void add(double gx, double gy, double m, double px, double py) {
        const double xx = gx * gx * m;
        const double xy = gx * gy * m;
        const double yy = gy * gy * m;
        gxx += xx;
        gxy += xy;
        gyy += yy;
        bx += xx * px + xy * py;
        by += xy * px + yy * py;
    }
};

/// @brief The bilinear weights of a sampling position's fractional part. **INTERNAL.**
/// `w00` weights the integer pixel at or before the sample, `w10` the one right of it,
/// `w01` the one below, `w11` the diagonal.
struct SubPixBlend {
    double w00, w10, w01, w11;

    SubPixBlend(double a, double b)
        : w00((1.0 - a) * (1.0 - b)), w10(a * (1.0 - b)), w01((1.0 - a) * b), w11(a * b) {}

    /// The blend of four integer-pixel values, row by row and in this order: the order
    /// is the definition the device arm and the tests reproduce.
    double operator()(double g00, double g10, double g01, double g11) const {
        return (w00 * g00 + w10 * g10) + (w01 * g01 + w11 * g11);
    }
};

/// @brief The signed value at `(x, y)` of a derivative held as planes. **INTERNAL.**
template <size_t N, typename WordType>
inline int subPixValue(const SignedQuantMat<N, WordType>& d, size_t x, size_t y) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    const size_t wi = x / kBits;
    const WordType bit = static_cast<WordType>(WordType{1} << (x % kBits));
    int v = 0;
    for (size_t j = 0; j < N; ++j) {
        if (d.constMagnitude(j).row(y)[wi] & bit) v |= (1 << j);
    }
    return (d.constSign().row(y)[wi] & bit) ? -v : v;
}

/// @brief `dx` beyond the image as `cv::getRectSubPix`'s replicated border makes it.
/// **INTERNAL.** Replicating the image makes it flat across the left and right edges,
/// so the x-derivative is zero outside the columns and the top and bottom rows'
/// x-derivative carried outward.
template <size_t N, typename WordType>
inline int subPixReplicatedX(const SignedQuantMat<N, WordType>& dx, long long x, long long y) {
    const long long w = static_cast<long long>(dx.getWidth());
    const long long h = static_cast<long long>(dx.getHeight());
    if (x < 0 || x >= w) return 0;
    const long long yc = y < 0 ? 0 : (y >= h ? h - 1 : y);
    return subPixValue(dx, static_cast<size_t>(x), static_cast<size_t>(yc));
}

/// @brief `dy` beyond the image, the transpose of `subPixReplicatedX`. **INTERNAL.**
template <size_t N, typename WordType>
inline int subPixReplicatedY(const SignedQuantMat<N, WordType>& dy, long long x, long long y) {
    const long long w = static_cast<long long>(dy.getWidth());
    const long long h = static_cast<long long>(dy.getHeight());
    if (y < 0 || y >= h) return 0;
    const long long xc = x < 0 ? 0 : (x >= w ? w - 1 : x);
    return subPixValue(dy, static_cast<size_t>(xc), static_cast<size_t>(y));
}

/// @brief One window's sums, every sample visited, through the replicated border.
/// **INTERNAL.** The path for a window that reaches past the image, and the dense
/// spelling the word-wise path is held to: a sample it skips adds exactly zero.
template <size_t N, typename WordType>
inline SubPixSums subPixWindowDense(const SignedQuantMat<N, WordType>& dx,
                                    const SignedQuantMat<N, WordType>& dy, const float* mask,
                                    int winHalf, long long x0, long long y0,
                                    const SubPixBlend& blend) {
    SubPixSums s;
    const int side = 2 * winHalf + 1;
    for (int i = -winHalf; i <= winHalf; ++i) {
        const long long v = y0 + i;
        for (int j = -winHalf; j <= winHalf; ++j) {
            const long long u = x0 + j;
            const double gx = blend(subPixReplicatedX(dx, u, v), subPixReplicatedX(dx, u + 1, v),
                                    subPixReplicatedX(dx, u, v + 1),
                                    subPixReplicatedX(dx, u + 1, v + 1));
            const double gy = blend(subPixReplicatedY(dy, u, v), subPixReplicatedY(dy, u + 1, v),
                                    subPixReplicatedY(dy, u, v + 1),
                                    subPixReplicatedY(dy, u + 1, v + 1));
            s.add(gx, gy, static_cast<double>(mask[(i + winHalf) * side + (j + winHalf)]),
                  static_cast<double>(j), static_cast<double>(i));
        }
    }
    return s;
}

/// @brief One row of a derivative's planes, staged for one word of the window: each
/// plane's word, and the same word one column on. **INTERNAL.**
template <size_t N, typename WordType>
struct SubPixRowWords {
    WordType mag[N], magNext[N], sign, signNext;

    /// Bit `b` of the staged words is the value at column `wi * kBits + b`, and of the
    /// `Next` words the value one column further.
    void load(const SignedQuantMat<N, WordType>& d, size_t y, size_t wi, bool haveNext) {
        constexpr size_t kBits = bitsPerWord<WordType>();
        for (size_t j = 0; j < N; ++j) {
            const WordType* r = d.constMagnitude(j).row(y);
            mag[j] = r[wi];
            magNext[j] = static_cast<WordType>(
                (r[wi] >> 1) | (haveNext ? static_cast<WordType>(r[wi + 1] << (kBits - 1)) : 0));
        }
        const WordType* r = d.constSign().row(y);
        sign = r[wi];
        signNext = static_cast<WordType>(
            (r[wi] >> 1) | (haveNext ? static_cast<WordType>(r[wi + 1] << (kBits - 1)) : 0));
    }
    WordType any() const {
        WordType m = 0;
        for (size_t j = 0; j < N; ++j) m = static_cast<WordType>(m | mag[j] | magNext[j]);
        return m;
    }
    /// The signed value at bit `b`, without a branch: whether a pixel has a gradient is
    /// exactly what an edge map makes unpredictable. A set sign on a zero magnitude --
    /// which the canonical-zero rule permits -- negates zero, so reads zero.
    static int value(const WordType* m, WordType s, size_t b) {
        int v = 0;
        for (size_t j = 0; j < N; ++j) v |= static_cast<int>((m[j] >> b) & 1u) << j;
        const int neg = static_cast<int>((s >> b) & 1u);
        return (v ^ -neg) + neg;
    }
    int at(size_t b) const { return value(mag, sign, b); }
    int next(size_t b) const { return value(magNext, signNext, b); }
};

/// @brief One window's sums for a window wholly inside the image, visiting only the
/// samples that can contribute. **INTERNAL.**
///
/// A sample at window column `j` blends integer columns `x0 + j` and `x0 + j + 1` of
/// rows `y0 + i` and `y0 + i + 1`. It contributes nothing unless one of those four has
/// a gradient, so the candidate set of a word is the OR of both rows' magnitude planes
/// and of the same planes one column on -- `|dx| | |dy|` dilated by one pixel right and
/// one down, found a word at a time.
/// @tparam Blend false on an integer position (`a == b == 0`), where the blend is the
/// pixel itself: only `x0 + j` of row `y0 + i` is read, and the sums are the same to
/// the bit -- every weight but one is zero and the products are exact.
template <bool Blend, size_t N, typename WordType>
inline SubPixSums subPixWindowSkip(const SignedQuantMat<N, WordType>& dx,
                                   const SignedQuantMat<N, WordType>& dy, const float* mask,
                                   int winHalf, long long x0, long long y0,
                                   const SubPixBlend& blend) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    SubPixSums s;
    const int side = 2 * winHalf + 1;
    const size_t xLo = static_cast<size_t>(x0 - winHalf);
    const size_t xHi = static_cast<size_t>(x0 + winHalf);   // inclusive

    for (int i = -winHalf; i <= winHalf; ++i) {
        const size_t v0 = static_cast<size_t>(y0 + i);
        const float* mrow = mask + (i + winHalf) * side;

        for (size_t wi = xLo / kBits; wi <= xHi / kBits; ++wi) {
            // The next word is read only when a column of this word's part of the window
            // needs its right neighbour from it -- and since the window ends inside the
            // image, that word exists whenever it is needed.
            const bool haveNext = Blend && (wi + 1) * kBits <= xHi + 1;
            SubPixRowWords<N, WordType> rx0, ry0, rx1, ry1;
            rx0.load(dx, v0, wi, haveNext);
            ry0.load(dy, v0, wi, haveNext);
            WordType nz = 0;
            if constexpr (Blend) {
                rx1.load(dx, v0 + 1, wi, haveNext);
                ry1.load(dy, v0 + 1, wi, haveNext);
                nz = static_cast<WordType>(rx0.any() | ry0.any() | rx1.any() | ry1.any());
            } else {
                for (size_t j = 0; j < N; ++j) {
                    nz = static_cast<WordType>(nz | rx0.mag[j] | ry0.mag[j]);
                }
            }
            // Trim to the window's columns; `wi` may span either end.
            const size_t lo = (wi * kBits > xLo) ? 0 : (xLo - wi * kBits);
            const size_t hiExcl = (xHi >= (wi + 1) * kBits) ? kBits : (xHi - wi * kBits + 1);
            if (lo != 0) nz = static_cast<WordType>(nz & ~lowBitsMask<WordType>(lo));
            if (hiExcl < kBits) nz = static_cast<WordType>(nz & lowBitsMask<WordType>(hiExcl));

            while (nz != 0) {
                const size_t bpos = ctzWord(static_cast<unsigned long long>(nz));
                nz = static_cast<WordType>(nz & (nz - 1));
                const int j = static_cast<int>(static_cast<long long>(wi * kBits + bpos) - x0);
                double gx, gy;
                if constexpr (Blend) {
                    gx = blend(rx0.at(bpos), rx0.next(bpos), rx1.at(bpos), rx1.next(bpos));
                    gy = blend(ry0.at(bpos), ry0.next(bpos), ry1.at(bpos), ry1.next(bpos));
                } else {
                    gx = static_cast<double>(rx0.at(bpos));
                    gy = static_cast<double>(ry0.at(bpos));
                }
                s.add(gx, gy, static_cast<double>(mrow[j + winHalf]), static_cast<double>(j),
                      static_cast<double>(i));
            }
        }
    }
    return s;
}

/// @brief Bits `[x, x + len)` of a plane row, `len <= 32`, bit 0 first, read from the
/// image's columns only. **INTERNAL.** A column left of 0 or right of `width - 1` reads
/// 0, or -- `replicate` -- the edge column's bit, which is cv::getRectSubPix's border.
template <typename WordType>
inline uint32_t subPixSpan(const WordType* row, long long x, unsigned len, long long width,
                           bool replicate) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    const long long lo = x < 0 ? 0 : x;
    const long long hi = (x + len < width) ? x + len : width;   // exclusive
    uint64_t acc = 0;
    if (lo < hi) {
        size_t wi = static_cast<size_t>(lo) / kBits;
        const unsigned off = static_cast<unsigned>(static_cast<size_t>(lo) % kBits);
        const unsigned n = static_cast<unsigned>(hi - lo);
        uint64_t v = static_cast<uint64_t>(row[wi]) >> off;
        unsigned got = static_cast<unsigned>(kBits) - off;
        while (got < n) {
            v |= static_cast<uint64_t>(row[++wi]) << got;
            got += static_cast<unsigned>(kBits);
        }
        acc = (v & ((uint64_t{1} << n) - 1u)) << (lo - x);
    }
    if (replicate) {
        if (x < 0 && (row[0] & 1u)) acc |= (uint64_t{1} << (-x)) - 1u;
        if (x + len > width) {
            const size_t last = static_cast<size_t>(width - 1);
            if ((row[last / kBits] >> (last % kBits)) & 1u) {
                acc |= ((uint64_t{1} << len) - 1u) & ~((uint64_t{1} << (width - x)) - 1u);
            }
        }
    }
    return static_cast<uint32_t>(acc);
}

/// @brief Spreads 32 bits into the even bits of 64. **INTERNAL.**
inline uint64_t subPixSpread(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2)) & 0x3333333333333333ull;
    x = (x | (x << 1)) & 0x5555555555555555ull;
    return x;
}

/// @brief The rows one ternary window reads, staged once per integer cell. **INTERNAL.**
///
/// A window spans at most 32 columns, so each image row it reads -- `2 * winHalf + 2` of
/// them, columns `x0 - winHalf .. x0 + winHalf + 1` -- is held as a 32-bit span of each
/// magnitude plane, for the candidate masks, and as both planes interleaved two bits a
/// pixel, magnitude even and sign odd, for the values. Beyond the image the spans are
/// cv::getRectSubPix's replicated border: `dx` reads zero left and right of the image and
/// the edge rows above and below it, `dy` the transpose. A walk spends a few iterations
/// in each cell, and the spans depend on the cell alone, so they are reused until it
/// moves.
struct SubPixTernaryRows {
    long long x0 = 0, y0 = 0;
    bool valid = false;
    uint32_t magX[2 * kMaxWinHalf + 2], magY[2 * kMaxWinHalf + 2];
    uint64_t codeX[2 * kMaxWinHalf + 2], codeY[2 * kMaxWinHalf + 2];

    template <typename WordType>
    void stage(const SignedQuantMat<1, WordType>& dx, const SignedQuantMat<1, WordType>& dy,
               int winHalf, long long cx0, long long cy0) {
        if (valid && cx0 == x0 && cy0 == y0) return;
        const long long width = static_cast<long long>(dx.getWidth());
        const long long height = static_cast<long long>(dx.getHeight());
        const unsigned span = static_cast<unsigned>(2 * winHalf + 2);
        const long long xLo = cx0 - winHalf;
        for (int r = 0; r < 2 * winHalf + 2; ++r) {
            const long long y = cy0 - winHalf + r;
            const size_t yc = static_cast<size_t>(y < 0 ? 0 : (y >= height ? height - 1 : y));
            const uint32_t sx = subPixSpan(dx.constSign().row(yc), xLo, span, width, false);
            magX[r] = subPixSpan(dx.constMagnitude(0).row(yc), xLo, span, width, false);
            codeX[r] = subPixSpread(magX[r]) | (subPixSpread(sx) << 1);
            if (y < 0 || y >= height) {
                magY[r] = 0;
                codeY[r] = 0;
            } else {
                const size_t yr = static_cast<size_t>(y);
                const uint32_t sy = subPixSpan(dy.constSign().row(yr), xLo, span, width, true);
                magY[r] = subPixSpan(dy.constMagnitude(0).row(yr), xLo, span, width, true);
                codeY[r] = subPixSpread(magY[r]) | (subPixSpread(sy) << 1);
            }
        }
        x0 = cx0;
        y0 = cy0;
        valid = true;
    }
};

/// @brief One ternary window's sums, visiting only the samples that can contribute.
/// **INTERNAL.**
///
/// A sample at window column `j` blends columns `j` and `j + 1` of rows `r` and `r + 1`
/// of the staged spans. It contributes nothing unless one of those four has a gradient,
/// so the candidates of a row are the OR of both rows' magnitude spans and of the same
/// spans one column on. A sample's four neighbours are two 4-bit fields of the
/// interleaved spans, and each row's half of the blend is one entry of a 16-entry table
/// built for this iteration's weights. Every product in a table entry is a weight times
/// -1, 0 or +1, which is exact, so a lookup is the blend formula to the bit.
/// @tparam Blend false on an integer position (`a == b == 0`), where the blend is the
/// pixel itself and the sums are the same to the bit: every weight but one is zero.
template <bool Blend>
inline SubPixSums subPixWindowTernary(const SubPixTernaryRows& rows, const float* mask,
                                      int winHalf, const SubPixBlend& blend) {
    // A (magnitude, sign) code as a value: a zero magnitude reads zero whatever its
    // sign -- the canonical-zero rule.
    static constexpr double kValue[4] = {0.0, 1.0, 0.0, -1.0};
    const int side = 2 * winHalf + 1;
    const uint32_t inWindow = static_cast<uint32_t>((uint64_t{1} << side) - 1u);

    double upper[16], lower[16];
    if constexpr (Blend) {
        for (unsigned c = 0; c < 16; ++c) {
            upper[c] = blend.w00 * kValue[c & 3u] + blend.w10 * kValue[c >> 2];
            lower[c] = blend.w01 * kValue[c & 3u] + blend.w11 * kValue[c >> 2];
        }
    }

    SubPixSums s;
    for (int r = 0; r < side; ++r) {
        const float* mrow = mask + r * side;
        const double py = static_cast<double>(r - winHalf);
        const uint64_t cx0 = rows.codeX[r], cy0 = rows.codeY[r];
        uint64_t cx1 = 0, cy1 = 0;
        uint32_t cand;
        if constexpr (Blend) {
            cx1 = rows.codeX[r + 1];
            cy1 = rows.codeY[r + 1];
            const uint32_t m = rows.magX[r] | rows.magY[r] | rows.magX[r + 1] | rows.magY[r + 1];
            cand = (m | (m >> 1)) & inWindow;
        } else {
            cand = (rows.magX[r] | rows.magY[r]) & inWindow;
        }
        while (cand != 0) {
            const unsigned j = static_cast<unsigned>(ctzWord(cand));
            cand &= cand - 1u;
            const unsigned sh = 2 * j;
            double gx, gy;
            if constexpr (Blend) {
                gx = upper[(cx0 >> sh) & 15u] + lower[(cx1 >> sh) & 15u];
                gy = upper[(cy0 >> sh) & 15u] + lower[(cy1 >> sh) & 15u];
            } else {
                gx = kValue[(cx0 >> sh) & 3u];
                gy = kValue[(cy0 >> sh) & 3u];
            }
            s.add(gx, gy, static_cast<double>(mrow[j]),
                  static_cast<double>(static_cast<int>(j) - winHalf), py);
        }
    }
    return s;
}

/// @brief One window's sums, by whichever path reaches the same bits fastest.
/// **INTERNAL.** Ternary planes take the staged spans, inside the image or past it; a
/// deeper plane stack is summed word-wise inside the image and through the replicated
/// border densely.
template <size_t N, typename WordType>
inline SubPixSums subPixWindow(const SignedQuantMat<N, WordType>& dx,
                               const SignedQuantMat<N, WordType>& dy, const float* mask,
                               int winHalf, long long x0, long long y0, double a, double b,
                               SubPixTernaryRows& rows) {
    const SubPixBlend blend(a, b);
    const bool integer = (a == 0.0 && b == 0.0);
    if constexpr (N == 1) {
        rows.stage(dx, dy, winHalf, x0, y0);
        return integer ? subPixWindowTernary<false>(rows, mask, winHalf, blend)
                       : subPixWindowTernary<true>(rows, mask, winHalf, blend);
    } else {
        static_cast<void>(rows);
        const long long width = static_cast<long long>(dx.getWidth());
        const long long height = static_cast<long long>(dx.getHeight());
        if (x0 - winHalf < 0 || y0 - winHalf < 0 || x0 + winHalf + 1 >= width ||
            y0 + winHalf + 1 >= height) {
            return subPixWindowDense(dx, dy, mask, winHalf, x0, y0, blend);
        }
        return integer ? subPixWindowSkip<false>(dx, dy, mask, winHalf, x0, y0, blend)
                       : subPixWindowSkip<true>(dx, dy, mask, winHalf, x0, y0, blend);
    }
}

}  // namespace impl

/// @brief Refines corner positions to sub-pixel accuracy: `cv::cornerSubPix` on the
/// image the derivative planes were taken from. **API TIER 2.**
///
/// @param dx,dy The image's signed derivatives, `I(x+1) - I(x-1)` -- what
/// ops/derivative.hpp writes and `ops/covariance.hpp` and the tracker already consume.
/// @param corners In/out. Positions are refined in place.
/// @param count Number of corners.
///
/// @note **THE BORDER.** A window that reaches past the image is refined over
/// `cv::cornerSubPix`'s replicated border, not skipped. Inside the image the planes are
/// read as given, so the answer is OpenCV's only where they hold OpenCV's border values:
/// derivatives taken with `BORDER_REPLICATE`. ops/derivative.hpp's default,
/// `BORDER_REFLECT_101`, differs in exactly two places -- `dx` in the first and last
/// COLUMN and `dy` in the first and last ROW, which it makes 0 where a replicated border
/// gives the one-sided difference. A window reads integer columns `floor(x) - winHalf`
/// through `floor(x) + winHalf + 1`, so a walk whose every position keeps
/// `winHalf + 1 <= x < width - winHalf - 2`, and the same in y, never reads those columns
/// or rows and gets the same answer from either.
/// @note **A walk can end outside the image**, and when it does it is returned there,
/// as `cv::cornerSubPix` returns it: the caller who needs a position inside the image
/// tests for one (`SubPixResult::outside` counts them).
/// @note **Never allocates and never throws.** The Gaussian mask is a stack buffer of
/// 3,844 B (`float[31 * 31]`, bounded by `impl::kMaxWinHalf`) whatever `winHalf` is.
/// @note A corner is refined **independently of every other**, so this splits over
/// `parallelFor` exactly as tracking does. It is not split here because the
/// operation is a few microseconds per frame at realistic corner counts; measure
/// before adding it.
/// @note The window is summed in `double` and the position kept as `float` between
/// iterations, as `cv::cornerSubPix` keeps it.
template <size_t N, typename WordType>
inline SubPixResult cornerSubPix(const SignedQuantMat<N, WordType>& dx,
                                 const SignedQuantMat<N, WordType>& dy, Point2f* corners,
                                 size_t count, const SubPixParams& params = SubPixParams()) {
    SubPixResult out;
    BINCV_ASSERT(params.winHalf >= 1 && params.winHalf <= impl::kMaxWinHalf,
                 "cornerSubPix: winHalf outside [1, kMaxWinHalf]");
    BINCV_ASSERT(corners != nullptr || count == 0,
                 "cornerSubPix: a non-zero count needs a corner array");
    BINCV_ASSERT(dx.getWidth() == dy.getWidth() && dx.getHeight() == dy.getHeight(),
                 "cornerSubPix: dx and dy differ in size");
    if (count == 0 || params.winHalf < 1 || params.winHalf > impl::kMaxWinHalf) return out;

    const int winHalf = params.winHalf;
    float mask[(2 * impl::kMaxWinHalf + 1) * (2 * impl::kMaxWinHalf + 1)];
    impl::subPixMask(winHalf, params.zeroHalf, mask);

    const long long width = static_cast<long long>(dx.getWidth());
    const long long height = static_cast<long long>(dx.getHeight());
    const float widthF = static_cast<float>(width);
    const float heightF = static_cast<float>(height);
    const int maxIterations =
        params.maxIterations < 1
            ? 1
            : (params.maxIterations > impl::kSubPixMaxIterations ? impl::kSubPixMaxIterations
                                                                 : params.maxIterations);
    const double eps = params.epsilon > 0.0 ? params.epsilon : 0.0;
    const double eps2 = eps * eps;
    const float limit = static_cast<float>(winHalf);
    impl::SubPixTernaryRows rows;

    for (size_t c = 0; c < count; ++c) {
        const Point2f start = corners[c];
        if (!(start.x >= 0.0f && start.x < widthF && start.y >= 0.0f && start.y < heightF)) {
            ++out.outside;
            continue;
        }
        Point2f cur = start;
        size_t* outcome = &out.refined;
        int iter = 0;
        double err = 0.0;
        do {
            // The sampling grid is the position's own: the integer pixel at or before it,
            // and the fractional part `(a, b)` every sample of the window shares.
            const float fx = std::floor(cur.x);
            const float fy = std::floor(cur.y);
            const long long x0 = static_cast<long long>(fx);
            const long long y0 = static_cast<long long>(fy);
            const double a = static_cast<double>(cur.x - fx);
            const double b = static_cast<double>(cur.y - fy);

            const impl::SubPixSums s =
                impl::subPixWindow(dx, dy, mask, winHalf, x0, y0, a, b, rows);

            const double det = s.gxx * s.gyy - s.gxy * s.gxy;
            if (std::fabs(det) <= DBL_EPSILON * DBL_EPSILON) {
                outcome = &out.singular;
                break;
            }
            // q = G^-1 b, as an offset from the sampling centre -- which is the position
            // itself, so the new position is the old one plus q.
            const double scale = 1.0 / det;
            Point2f next;
            next.x = static_cast<float>(static_cast<double>(cur.x) + s.gyy * scale * s.bx -
                                        s.gxy * scale * s.by);
            next.y = static_cast<float>(static_cast<double>(cur.y) - s.gxy * scale * s.bx +
                                        s.gxx * scale * s.by);
            const float ex = next.x - cur.x;
            const float ey = next.y - cur.y;
            err = static_cast<double>(ex * ex + ey * ey);
            cur = next;
            if (cur.x < 0.0f || cur.x >= widthF || cur.y < 0.0f || cur.y >= heightF) {
                outcome = &out.outside;
                break;
            }
        } while (++iter < maxIterations && err > eps2);

        // POOR CONVERGENCE -- cv::cornerSubPix's rule, applied after the loop exactly as
        // it applies it. A point that has walked further than its own half-window from
        // the seed has not been refined, it has been captured by different structure, and
        // the seed is the better answer.
        if (std::fabs(cur.x - start.x) > limit || std::fabs(cur.y - start.y) > limit) {
            cur = start;
            outcome = &out.diverged;
        }
        corners[c] = cur;
        ++*outcome;
    }
    return out;
}

}  // inline namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
