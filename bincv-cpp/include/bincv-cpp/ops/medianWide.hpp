#pragma once

/// @file medianWide.hpp
/// @brief The reference pipeline's median filter, on a WIDE (8- or 16-bit) image,
/// with a caller-chosen neighbourhood. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS EXISTS ALONGSIDE ops/denoise.hpp
///
/// [ops/denoise.hpp](denoise.hpp) implements the same neighbourhood for BINARY
/// input, where median collapses to `maj3` -- one expression, 32 pixels per word.
/// That is the right kernel and it is not going anywhere.
///
/// But the reference filters the **grayscale** image, BEFORE binarisation:
/// `SEALProcessor.cpp` runs `three_pix_median_filter(img)` and only then
/// `rl_fast_edge_filter_wide(img, t)`. A binary-only median cannot sit where the
/// reference puts it, so a frontend that wanted the reference's pipeline had to
/// borrow OpenCV for this one step.
///
/// ---------------------------------------------------------------------------
/// THE NEIGHBOURHOOD IS THE CALLER'S, AND THE REFERENCE HAS TWO OF THEM
///
/// `SEAL/src/temporal_processing/denoise.cpp` carries `three_pix_median_filter` --
/// the asymmetric L, `p1` above / `p2` centre / `p3` right -- **and**
/// `five_pix_median_filter`, the plus. Both ship here as named constants, and an
/// arbitrary offset set is a template argument rather than a fork.
///
/// This is emphatically NOT `cv::medianBlur`, whose neighbourhood is a square and
/// whose border is replicated. Tier 3, and the name is not borrowed
/// ([CLAUDE.md](../../../../CLAUDE.md)).
///
/// ---------------------------------------------------------------------------
/// THE BORDER IS ZERO FILL, AND THAT IS THE REFERENCE'S, NOT A CHOICE
///
/// The reference builds its shifted neighbours as `cv::Mat::zeros` and copies the
/// overlapping region in, so the row and column that fall off the edge KEEP THE
/// ZEROS. A pixel at the top row therefore takes its median against a 0, not
/// against a replicated or reflected neighbour. ops/denoise.hpp records the same
/// rule for the same reason.
// F-5: BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"


#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_MEDIAN_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_MEDIAN_NEON 1
#include <arm_neon.h>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One sample position, relative to the pixel being written.
struct MedianOffset {
    int dy;
    int dx;
};

/// @brief A neighbourhood: `K` offsets, `K` odd so the median is a single element.
template <size_t K>
struct MedianPattern {
    static_assert(K % 2 == 1, "a median needs an odd number of samples");
    MedianOffset offset[K];
};

/// @brief The reference's `three_pix_median_filter`: above, centre, right.
/// @note An asymmetric **L**, not a line and not a square. It is chosen for what it
/// costs in race logic, not for isotropy, which is why no OpenCV kernel
/// matches it.
inline constexpr MedianPattern<3> kMedianReferenceL{{{-1, 0}, {0, 0}, {0, 1}}};

/// @brief The reference's `five_pix_median_filter`: the plus.
inline constexpr MedianPattern<5> kMedianReferencePlus{
    {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}}};

namespace impl {

#if defined(BINCV_MEDIAN_AVX2)
/// @brief Is AVX2 present? Asked once, not once per row.
inline bool hasMedianSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}

/// @brief `med3` for thirty-two pixels: `max(min(a,b), min(max(a,b), c))`.
__attribute__((target("avx2"))) inline void med3Store(const uint8_t* a, const uint8_t* b,
                                                      const uint8_t* c, uint8_t* out) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i vc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c));
    const __m256i lo = _mm256_min_epu8(va, vb);
    const __m256i hi = _mm256_max_epu8(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),
                        _mm256_max_epu8(lo, _mm256_min_epu8(hi, vc)));
}
#elif defined(BINCV_MEDIAN_NEON)
inline bool hasMedianSimd() { return true; }

/// @brief `med3` for sixteen pixels. NEON is baseline on aarch64; nothing to dispatch on.
inline void med3Store(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out) {
    const uint8x16_t va = vld1q_u8(a), vb = vld1q_u8(b), vc = vld1q_u8(c);
    const uint8x16_t lo = vminq_u8(va, vb);
    const uint8x16_t hi = vmaxq_u8(va, vb);
    vst1q_u8(out, vmaxq_u8(lo, vminq_u8(hi, vc)));
}
#endif

/// @brief One pixel by the scalar rule, border included. **INTERNAL.**
/// @note The vector sweep covers the interior and calls this for everything else, so
/// the out-of-range-reads-as-zero rule stays written down exactly once.
template <size_t K, typename SrcT>
inline void med3Scalar(const SrcT* src, size_t width, size_t height, size_t srcStride,
                       SrcT* dst, size_t dstStride, const MedianPattern<K>& pattern,
                       size_t y, size_t x) {
    SrcT v[K];
    for (size_t k = 0; k < K; ++k) {
        const long long sy = static_cast<long long>(y) + pattern.offset[k].dy;
        const long long sx = static_cast<long long>(x) + pattern.offset[k].dx;
        const bool inside = sy >= 0 && sx >= 0 && sy < static_cast<long long>(height) &&
                            sx < static_cast<long long>(width);
        v[k] = inside
                   ? src[static_cast<size_t>(sy) * srcStride + static_cast<size_t>(sx)]
                   : SrcT{0};
    }
    for (size_t i = 1; i < K; ++i) {
        SrcT key = v[i];
        size_t j = i;
        while (j > 0 && v[j - 1] > key) {
            v[j] = v[j - 1];
            --j;
        }
        v[j] = key;
    }
    dst[y * dstStride + x] = v[K / 2];
}

}  // namespace impl

/// @brief Median filter over a caller-chosen neighbourhood. **API TIER 3.**
/// @param src,dst Row-major, strides in ELEMENTS. **`src` and `dst` must not alias**
/// -- every output reads neighbours that a partial in-place write would have
/// already changed.
/// @note Out-of-range samples read as **zero**; see the border note above.
/// @note Never allocates and never throws.
template <size_t K, typename SrcT>
inline void medianWide(const SrcT* src, size_t width, size_t height, size_t srcStride,
                       SrcT* dst, size_t dstStride, const MedianPattern<K>& pattern) {
    BINCV_ASSERT(src != dst, "medianWide: src and dst must not alias");
    if (width == 0 || height == 0) return;
    BINCV_ASSERT(src != nullptr && dst != nullptr,
                 "medianWide: a non-empty image needs non-null pointers");

    // ==================================================================
    // earlier work: THE THREE-SAMPLE MEDIAN IS MIN AND MAX, AND NOTHING ELSE.
    //
    // med3(a, b, c) = max(min(a, b), min(max(a, b), c))
    //
    // Five register operations for as many pixels as fit — thirty-two on AVX2, sixteen
    // on NEON — against a scalar sorting network **and three bounds-checked gathers**
    // per pixel. put this kernel at **78% of the whole frontend** the moment it
    // was wired in, which is what made it worth writing.
    //
    // ONLY THE INTERIOR. A row or column where any offset leaves the image keeps the
    // scalar body below, which is also the oracle: `tests/test_median_wide.cpp` compares
    // them, and the border rule (out-of-range reads as ZERO) lives in one place still.
#if defined(BINCV_MEDIAN_AVX2) || defined(BINCV_MEDIAN_NEON)
    if constexpr (K == 3 && sizeof(SrcT) == 1) {
#if defined(BINCV_MEDIAN_AVX2)
        constexpr size_t kStep = 32;
        const bool haveSimd = impl::hasMedianSimd();
#else
        constexpr size_t kStep = 16;
        const bool haveSimd = true;
#endif
        // The interior in y and x: every offset must land inside the image.
        long long dyLo = 0, dyHi = 0, dxLo = 0, dxHi = 0;
        for (size_t k = 0; k < 3; ++k) {
            dyLo = pattern.offset[k].dy < dyLo ? pattern.offset[k].dy : dyLo;
            dyHi = pattern.offset[k].dy > dyHi ? pattern.offset[k].dy : dyHi;
            dxLo = pattern.offset[k].dx < dxLo ? pattern.offset[k].dx : dxLo;
            dxHi = pattern.offset[k].dx > dxHi ? pattern.offset[k].dx : dxHi;
        }
        const size_t y0 = static_cast<size_t>(-dyLo);
        const size_t x0 = static_cast<size_t>(-dxLo);
        const long long yEnd = static_cast<long long>(height) - dyHi;
        const long long xEnd = static_cast<long long>(width) - dxHi;
        if (haveSimd && yEnd > static_cast<long long>(y0) &&
            xEnd > static_cast<long long>(x0)) {
            const size_t xStop = static_cast<size_t>(xEnd);
            for (size_t y = y0; y < static_cast<size_t>(yEnd); ++y) {
                size_t x = x0;
                for (; x + kStep <= xStop; x += kStep) {
                    const SrcT* p0 = src + (static_cast<long long>(y) + pattern.offset[0].dy) *
                                               static_cast<long long>(srcStride) +
                                     static_cast<long long>(x) + pattern.offset[0].dx;
                    const SrcT* p1 = src + (static_cast<long long>(y) + pattern.offset[1].dy) *
                                               static_cast<long long>(srcStride) +
                                     static_cast<long long>(x) + pattern.offset[1].dx;
                    const SrcT* p2 = src + (static_cast<long long>(y) + pattern.offset[2].dy) *
                                               static_cast<long long>(srcStride) +
                                     static_cast<long long>(x) + pattern.offset[2].dx;
                    impl::med3Store(reinterpret_cast<const uint8_t*>(p0),
                                    reinterpret_cast<const uint8_t*>(p1),
                                    reinterpret_cast<const uint8_t*>(p2),
                                    reinterpret_cast<uint8_t*>(dst + y * dstStride + x));
                }
                // The row's tail, and then the next row; the border rows and columns
                // fall through to the scalar sweep below.
                for (; x < xStop; ++x) {
                    SrcT v[3];
                    for (size_t k = 0; k < 3; ++k) {
                        v[k] = src[(static_cast<long long>(y) + pattern.offset[k].dy) *
                                       static_cast<long long>(srcStride) +
                                   static_cast<long long>(x) + pattern.offset[k].dx];
                    }
                    const SrcT lo = v[0] < v[1] ? v[0] : v[1];
                    const SrcT hi = v[0] < v[1] ? v[1] : v[0];
                    const SrcT m = hi < v[2] ? hi : v[2];
                    dst[y * dstStride + x] = lo > m ? lo : m;
                }
            }
            // The border, VISITED RATHER THAN SKIPPED. Walking the whole image and
            // `continue`-ing over the interior is 360 000 branch-only iterations a
            // frame -- measured at ~0.4 ms, against a border of about 2 000 pixels.
            const auto scalarRow = [&](size_t yy, size_t xa, size_t xb) {
                for (size_t xx = xa; xx < xb; ++xx) {
                    impl::med3Scalar<K, SrcT>(src, width, height, srcStride, dst, dstStride,
                                              pattern, yy, xx);
                }
            };
            for (size_t y = 0; y < y0; ++y) scalarRow(y, 0, width);
            for (size_t y = static_cast<size_t>(yEnd); y < height; ++y) scalarRow(y, 0, width);
            for (size_t y = y0; y < static_cast<size_t>(yEnd); ++y) {
                scalarRow(y, 0, x0);
                scalarRow(y, xStop, width);
            }
            return;
        }
    }
#endif

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            SrcT v[K];
            for (size_t k = 0; k < K; ++k) {
                const long long sy = static_cast<long long>(y) + pattern.offset[k].dy;
                const long long sx = static_cast<long long>(x) + pattern.offset[k].dx;
                const bool inside = sy >= 0 && sx >= 0 &&
                                    sy < static_cast<long long>(height) &&
                                    sx < static_cast<long long>(width);
                v[k] = inside ? src[static_cast<size_t>(sy) * srcStride +
                                    static_cast<size_t>(sx)]
                              : SrcT{0};
            }
            // Insertion sort. `K` is 3 or 5 in every shipped pattern and odd by
            // static_assert, so this is a handful of compares fully unrolled -- a
            // sorting network would be the same instructions with more source.
            for (size_t i = 1; i < K; ++i) {
                SrcT key = v[i];
                size_t j = i;
                while (j > 0 && v[j - 1] > key) {
                    v[j] = v[j - 1];
                    --j;
                }
                v[j] = key;
            }
            dst[y * dstStride + x] = v[K / 2];
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
