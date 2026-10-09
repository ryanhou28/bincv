#pragma once

/// @file medianWide.hpp
/// @brief Median filter on a WIDE (8- or 16-bit) image over a caller-chosen
/// neighbourhood. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS EXISTS ALONGSIDE ops/denoise.hpp
///
/// [ops/denoise.hpp](denoise.hpp) implements the same neighbourhood for BINARY
/// input, where median collapses to `maj3` -- one expression, 32 pixels per word.
/// That is the right kernel and it is not going anywhere.
///
/// But the reference pipeline (the visual-inertial odometry system, not in this
/// repository, that binCV was built to serve stage by stage; see
/// docs/ARCHITECTURE.md) filters the **grayscale** image, BEFORE binarization: its
/// frontend runs a three-sample median and only then its edge threshold. A
/// binary-only median cannot sit where the reference puts it, so a caller that
/// wanted the reference's pipeline would have to borrow OpenCV for this one step.
///
/// ---------------------------------------------------------------------------
/// THE NEIGHBOURHOOD IS THE CALLER'S, AND THE REFERENCE HAS TWO OF THEM
///
/// The reference pipeline's denoiser carries a three-sample median -- the
/// asymmetric L, `p1` above / `p2` center / `p3` right -- **and** a five-sample
/// one, the plus. Both ship here as named constants, and an
/// arbitrary offset set is a template argument rather than a fork.
///
/// This is emphatically NOT `cv::medianBlur`, whose neighbourhood is a square and
/// whose border is replicated. Tier 3, and the name is not borrowed.
///
/// ---------------------------------------------------------------------------
/// THE BORDER IS ZERO FILL, AND THAT IS THE REFERENCE'S, NOT A CHOICE
///
/// The reference builds its shifted neighbours as `cv::Mat::zeros` and copies the
/// overlapping region in, so the row and column that fall off the edge KEEP THE
/// ZEROS. A pixel at the top row therefore takes its median against a 0, not
/// against a replicated or reflected neighbour. ops/denoise.hpp records the same
/// rule for the same reason.

// BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"

// !__CUDACC__: a CUDA translation unit takes the portable arm, matching every
// other x86 arm's gate (impl/binMat_impl.hpp has the full note).
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#define BINCV_MEDIAN_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_MEDIAN_NEON 1
#include <arm_neon.h>
#elif defined(__ARM_FEATURE_SIMD32) && !defined(__CUDACC__)
// The ARMv7E-M / ARMv8-M DSP extension: four byte lanes in a general register, which
// on a core with no vector unit is the widest arithmetic there is.
#define BINCV_MEDIAN_DSP 1
#include <arm_acle.h>
#include <cstring>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One sample position, relative to the pixel being written. **API TIER 3.**
struct MedianOffset {
    int dy;
    int dx;
};

/// @brief A neighbourhood: `K` offsets, `K` odd so the median is a single element.
/// **API TIER 3.**
template <size_t K>
struct MedianPattern {
    static_assert(K % 2 == 1, "a median needs an odd number of samples");
    MedianOffset offset[K];
};

/// @brief The reference pipeline's three-sample median: above, center, right.
/// **API TIER 3.**
/// @note An asymmetric **L**, not a line and not a square. It is chosen for what it
/// costs in race logic, not for isotropy, which is why no OpenCV kernel
/// matches it.
inline constexpr MedianPattern<3> kMedianReferenceL{{{-1, 0}, {0, 0}, {0, 1}}};

/// @brief The reference pipeline's five-sample median: the plus. **API TIER 3.**
inline constexpr MedianPattern<5> kMedianReferencePlus{
    {{0, 0}, {0, 1}, {1, 0}, {0, -1}, {-1, 0}}};

namespace impl {

#if defined(BINCV_MEDIAN_AVX2)
/// @brief Force the portable arm, for the benchmark and the tests. **INTERNAL.**
inline bool& medianSimdEnabled() {
    static bool on = true;
    return on;
}

/// @brief Is AVX2 present? Asked once, not once per row. **INTERNAL.**
inline bool hasMedianSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes && medianSimdEnabled();
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
/// @brief Force the portable arm, for the benchmark and the tests. **INTERNAL.**
inline bool& medianSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasMedianSimd() { return medianSimdEnabled(); }

/// @brief `med3` for sixteen pixels. NEON is baseline on aarch64; nothing to dispatch on.
inline void med3Store(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out) {
    const uint8x16_t va = vld1q_u8(a), vb = vld1q_u8(b), vc = vld1q_u8(c);
    const uint8x16_t lo = vminq_u8(va, vb);
    const uint8x16_t hi = vmaxq_u8(va, vb);
    vst1q_u8(out, vmaxq_u8(lo, vminq_u8(hi, vc)));
}
#elif defined(BINCV_MEDIAN_DSP)
inline bool& medianSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasMedianSimd() { return medianSimdEnabled(); }

/// @brief Sixteen medians of three, four byte lanes at a time. `usub8` leaves each
/// lane's `a >= b` in its GE flag and `sel` picks by it, so a byte min or max is two
/// instructions; the median is three of them sharing one compare.
inline void med3Store(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* out) {
    for (size_t o = 0; o < 16; o += 4) {
        uint32_t va, vb, vc;
        std::memcpy(&va, a + o, 4);
        std::memcpy(&vb, b + o, 4);
        std::memcpy(&vc, c + o, 4);
        (void)__usub8(va, vb);  // GE[i]: a_i >= b_i
        const uint32_t lo = __sel(vb, va);
        const uint32_t hi = __sel(va, vb);
        (void)__usub8(hi, vc);  // GE[i]: hi_i >= c_i
        const uint32_t m = __sel(vc, hi);
        (void)__usub8(lo, m);   // GE[i]: lo_i >= m_i
        const uint32_t med = __sel(lo, m);
        std::memcpy(out + o, &med, 4);
    }
}
#endif

/// @brief One pixel by the scalar rule, border included. **INTERNAL.**
/// @note The vector sweep covers the interior and calls this for everything else, so
/// the out-of-range-reads-as-zero rule stays written down exactly once.
/// @brief The pattern's horizontal reach: every `x` in `[lo, hi)` has all its samples
/// inside a row of `width`. **INTERNAL.** Empty when the row is narrower than the
/// pattern.
template <size_t K>
inline void medianInterior(const MedianPattern<K>& pattern, size_t width, size_t& lo,
                           size_t& hi) {
    int dxLo = 0, dxHi = 0;
    for (size_t k = 0; k < K; ++k) {
        dxLo = pattern.offset[k].dx < dxLo ? pattern.offset[k].dx : dxLo;
        dxHi = pattern.offset[k].dx > dxHi ? pattern.offset[k].dx : dxHi;
    }
    lo = static_cast<size_t>(-dxLo);
    const size_t reach = static_cast<size_t>(dxHi);
    hi = width > reach ? width - reach : 0;
    // A row narrower than the pattern's reach has no interior, and neither bound may
    // pass the row's end: the border sweeps run [0, lo) and [hi, width).
    if (lo > width) lo = width;
    if (hi < lo) hi = lo;
}

/// @brief The median of K samples, in place: `v[K / 2]` after the call. **INTERNAL.**
/// @note For three samples, `max(min(a, b), min(max(a, b), c))` -- the same function
/// as the vector arm's, and as the insertion sort, with no branch. Every larger
/// pattern sorts: `K` is 3 or 5 in every shipped pattern and odd by static_assert,
/// so the sort is a handful of compares fully unrolled.
template <size_t K, typename SrcT>
inline SrcT medianOf(SrcT (&v)[K]) {
    if constexpr (K == 3) {
        const SrcT lo = v[0] < v[1] ? v[0] : v[1];
        const SrcT hi = v[0] < v[1] ? v[1] : v[0];
        const SrcT m = hi < v[2] ? hi : v[2];
        return lo > m ? lo : m;
    } else {
        for (size_t i = 1; i < K; ++i) {
            SrcT key = v[i];
            size_t j = i;
            while (j > 0 && v[j - 1] > key) {
                v[j] = v[j - 1];
                --j;
            }
            v[j] = key;
        }
        return v[K / 2];
    }
}

/// @brief One output row. **INTERNAL** -- the body every row of every entry point
/// goes through, so the whole-frame call and the streaming one
/// (ops/medianEdgeStream.hpp) cannot differ.
/// @param rows `rows[k]` is the source row that `pattern.offset[k]` lands on, or
/// null when that row is outside the image -- the vertical border is the caller's,
/// and a null row reads as zero. The horizontal border is this function's, and
/// reads as zero too.
template <size_t K, typename SrcT>
inline void medianRow(const SrcT* const (&rows)[K], size_t width, SrcT* out,
                      const MedianPattern<K>& pattern) {
    size_t lo = 0, hi = 0;
    medianInterior(pattern, width, lo, hi);
    bool allRows = true;
    for (size_t k = 0; k < K; ++k) allRows = allRows && rows[k] != nullptr;
    if (!allRows) lo = hi = width;  // no interior: every pixel takes the checked path

    // The checked sample: the border rule, in one place.
    const auto checked = [&](size_t x) {
        SrcT v[K];
        for (size_t k = 0; k < K; ++k) {
            const long long sx = static_cast<long long>(x) + pattern.offset[k].dx;
            v[k] = (rows[k] != nullptr && sx >= 0 && sx < static_cast<long long>(width))
                       ? rows[k][static_cast<size_t>(sx)]
                       : SrcT{0};
        }
        out[x] = medianOf<K, SrcT>(v);
    };

    for (size_t x = 0; x < lo; ++x) checked(x);

    size_t x = lo;
#if defined(BINCV_MEDIAN_AVX2) || defined(BINCV_MEDIAN_NEON) || defined(BINCV_MEDIAN_DSP)
    // ==================================================================
    // THE THREE-SAMPLE MEDIAN IS MIN AND MAX, AND NOTHING ELSE.
    //
    // med3(a, b, c) = max(min(a, b), min(max(a, b), c))
    //
    // Five register operations for as many pixels as fit — thirty-two on AVX2, sixteen
    // on NEON — against a scalar sorting network **and three bounds-checked gathers**
    // per pixel. The first end-to-end profile put this kernel at **78% of the
    // whole pipeline** the moment it was wired in, which is what made it worth
    // writing.
    //
    // ONLY THE INTERIOR: every sample of every pixel it covers is inside its row.
    if constexpr (K == 3 && sizeof(SrcT) == 1) {
#if defined(BINCV_MEDIAN_AVX2)
        constexpr size_t kStep = 32;
#else
        constexpr size_t kStep = 16;
#endif
        if (hasMedianSimd()) {
            const ptrdiff_t d0 = pattern.offset[0].dx, d1 = pattern.offset[1].dx,
                            d2 = pattern.offset[2].dx;
            for (; x + kStep <= hi; x += kStep) {
                const ptrdiff_t xi = static_cast<ptrdiff_t>(x);
                med3Store(reinterpret_cast<const uint8_t*>(rows[0] + (xi + d0)),
                          reinterpret_cast<const uint8_t*>(rows[1] + (xi + d1)),
                          reinterpret_cast<const uint8_t*>(rows[2] + (xi + d2)),
                          reinterpret_cast<uint8_t*>(out + x));
            }
        }
    }
#endif
    // The interior, one pixel at a time: no bounds test, because [lo, hi) is where
    // none is needed. The row pointers and column offsets are copied into locals
    // first: `out` is a byte pointer, which may alias anything, so a sample spelled
    // `rows[k][x + pattern.offset[k].dx]` would be reloaded after every store.
    if (x < hi) {
        const SrcT* r[K];
        ptrdiff_t d[K];
        for (size_t k = 0; k < K; ++k) {
            r[k] = rows[k];
            d[k] = pattern.offset[k].dx;
        }
        if constexpr (K == 3) {
            const SrcT* const r0 = r[0];
            const SrcT* const r1 = r[1];
            const SrcT* const r2 = r[2];
            const ptrdiff_t d0 = d[0], d1 = d[1], d2 = d[2];
            for (; x < hi; ++x) {
                const ptrdiff_t xi = static_cast<ptrdiff_t>(x);
                SrcT v[3] = {r0[xi + d0], r1[xi + d1], r2[xi + d2]};
                out[x] = medianOf<3, SrcT>(v);
            }
        } else {
            for (; x < hi; ++x) {
                const ptrdiff_t xi = static_cast<ptrdiff_t>(x);
                SrcT v[K];
                for (size_t k = 0; k < K; ++k) v[k] = r[k][xi + d[k]];
                out[x] = medianOf<K, SrcT>(v);
            }
        }
    }

    for (; x < width; ++x) checked(x);
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

    // Row by row through the one body every entry point shares. The vertical border
    // is applied here: a sample row outside the image is handed over as null.
    for (size_t y = 0; y < height; ++y) {
        const SrcT* rows[K];
        for (size_t k = 0; k < K; ++k) {
            const long long sy = static_cast<long long>(y) + pattern.offset[k].dy;
            rows[k] = (sy >= 0 && sy < static_cast<long long>(height))
                          ? src + static_cast<size_t>(sy) * srcStride
                          : nullptr;
        }
        impl::medianRow<K, SrcT>(rows, width, dst + y * dstStride, pattern);
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
