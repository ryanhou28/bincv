#pragma once

/// @file pack.hpp
/// @brief Turning a plain pixel array into bits. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS FILE IS IN CORE AND NOT BEHIND `BINCV_WITH_OPENCV`
///
/// Before this file, EVERY path that got pixels into binCV took a `cv::Mat`:
/// `QuantMat::fromCVMat` and `bincv::threshold` both did, and `QuantMat::wrap`
/// takes a buffer that is ALREADY bit-packed, which is a different problem. So
/// the core-only build -- the configuration the whole memory argument rests on,
/// and three of `verify.sh`'s four -- HAD NO WAY TO RECEIVE AN IMAGE.
///
/// A sensor hands a driver a buffer. That buffer is what this file takes.
///
/// ---------------------------------------------------------------------------
/// THE INPUT CONTRACT ([ARCHITECTURE §7.8](../../../ARCHITECTURE.md))
///
/// binCV accepts a SINGLE-CHANNEL, INTEGER-TYPED, STRIDED pixel array and turns
/// it into bits. Getting to that array is the caller's job. The Y plane of a
/// YUV420 buffer already IS such an array -- pass its stride, do not convert it.
///
/// `SrcT` is `uint8_t` or `uint16_t`, and the second is not a luxury: 10-, 12-
/// and 16-bit sensors are ordinary, and downconverting to 8 bits first would
/// discard the low bits BEFORE the rule decides. For a plain threshold that is a
/// boundary rounding difference; for a gradient it is a total loss (§7.8.1).
///
/// ---------------------------------------------------------------------------
/// WHY THE RULE IS A TEMPLATE PARAMETER AND NOT A FUNCTION POINTER
///
/// [X-71](../../../EXPERIMENTS.md) measured this loop at **46x** on x86 and
/// **14x** on aarch64 by turning it into a compare and a move-mask. That only
/// works if the comparison is ONE PREDICATE the compiler can see. A runtime
/// callback would put a call in the inner loop and give the whole factor back --
/// [X-72](../../../EXPERIMENTS.md) measured a mere runtime BRANCH costing 17%
/// elsewhere in this library.
///
/// So the shipped rules are an enum, resolved at compile time. `packBitsIf`
/// takes an arbitrary predicate for anything they cannot express, and is
/// honestly slower.

#include <cstddef>
#include <cstdint>

#include "../binMat.hpp"
#include "../impl/kernel_util.hpp"

// T5.9: the N-bit packer's vector path, selected at RUN TIME on x86 so the library's
// baseline ISA is unchanged, and baseline on aarch64 where NEON always exists.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_PACKQUANT_AVX2 1
#define BINCV_PACKQUANT_SIMD 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_PACKQUANT_SIMD 1
#include <arm_neon.h>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief How a source pixel becomes a bit. **Compile-time; see the file header.**
enum class PackRule {
    NonZero,        ///< `v != 0`. `QuantMat<1>::fromCVMat`'s historical rule.
    GreaterThan,    ///< `v > threshold`. `bincv::threshold`'s rule.
    GreaterEqual,   ///< `v >= threshold`. The reference edge filter's relation.
};

namespace impl {

/// @brief `PackRule` mapped onto the internal `PackCmp`. **INTERNAL.**
/// @note Two enums exist because impl/binMat_impl.hpp owns the row packer and cannot
///       include this file -- pack.hpp includes binMat.hpp, which includes that one.
///       The row packer is shared; only the name of the tag differs.
template <PackRule R>
constexpr PackCmp toPackCmp() {
    if constexpr (R == PackRule::NonZero) {
        return PackCmp::NonZero;
    } else if constexpr (R == PackRule::GreaterThan) {
        return PackCmp::GreaterThan;
    } else {
        return PackCmp::GreaterEqual;
    }
}

} // namespace impl

/// @brief Packs `rowCount` rows into `dst` starting at `dstRow`. **API TIER 3.**
///
/// **THE STREAMING ENTRY POINT, AND IT IS NOT A CONVENIENCE.** A microcontroller
/// reading a sensor over DMA gets **rows as they arrive** and may not have RAM for a
/// whole frame -- the footprint argument that justifies binCV is the same argument
/// that says it cannot buffer one. This consumes a line buffer of any height, so a
/// caller can pack as the frame streams in and never materialise it.
///
/// `packBits` is a call to this with `dstRow = 0` and every row at once.
///
/// @param src The first row of the chunk; `srcStride` ELEMENTS between rows.
/// @param dstRow Which row of `dst` this chunk begins at.
/// @note Rows are independent -- nothing here reads a neighbouring row -- so chunk
///       boundaries cannot change the result. That is what makes streaming exact
///       rather than approximate, and it is why the ops that DO read neighbours
///       (ops/edge.hpp, ops/medianWide.hpp) have no such entry point.
template <PackRule R, typename SrcT, typename WordType>
inline void packRows(const SrcT* src, size_t width, size_t rowCount, size_t srcStride,
                     BinMatView<WordType> dst, size_t dstRow, SrcT t = SrcT{0}) {
    BINCV_ASSERT(width == dst.width, "packRows: src and dst must have the same width");
    BINCV_ASSERT(dstRow + rowCount <= dst.height, "packRows: chunk runs past dst");
    if (width == 0 || rowCount == 0) return;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "packRows: a non-empty chunk needs non-null pointers");
    const size_t words = impl::minRowWords<WordType>(dst.width);
    for (size_t r = 0; r < rowCount; ++r) {
        WordType* out = dst.row(dstRow + r);
        for (size_t i = 0; i < words; ++i) out[i] = 0;
        impl::packRowCmp<impl::toPackCmp<R>(), SrcT, WordType>(src + r * srcStride, width, t,
                                                               out);
    }
}

/// @brief Packs a pixel array to one bit per pixel. **API TIER 3.**
/// @tparam R The rule, at compile time.
/// @param src Row-major, `srcStride` elements between rows (NOT bytes).
/// @param t Threshold; ignored by `NonZero`.
/// @note `dst`'s padding bits are zero on return.
/// @note Never allocates and never throws. Mismatched dimensions are a
///       programming error, reported by `BINCV_ASSERT` in debug builds.
template <PackRule R, typename SrcT, typename WordType>
inline void packBits(const SrcT* src, size_t width, size_t height, size_t srcStride,
                     BinMatView<WordType> dst, SrcT t = SrcT{0}) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "packBits: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "packBits: dst's stride must cover a whole row");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "packBits: a non-empty image needs non-null pointers");
    packRows<R, SrcT, WordType>(src, width, height, srcStride, dst, 0, t);
}

// ===========================================================================
// T5.9 -- N BITS PER PIXEL, WITHOUT OpenCV.
//
// Everything above writes ONE bit per pixel. At `N > 1` the only way into binCV was
// `QuantMat<N>::fromCVMat`, which takes a `cv::Mat` -- so **N-bit ingestion required
// linking OpenCV**, which is the one thing the core-only build exists to avoid.
//
// AND THE POLICY WAS HARD-CODED, TWICE, INCONSISTENTLY: `BinMat::fromCVMat` reads any
// nonzero byte as 1, `QuantMat<N>::fromCVMat` scales with `round(v * MaxValue / 255)`.
// A caller wanting a mid-grey split or a non-monotonic map had to convert and then
// re-quantise -- two passes and an 8-bit intermediate that this library exists to
// avoid.
// ===========================================================================

/// @brief How a source pixel becomes an N-bit value. **Compile-time.**
enum class QuantRule {
    Scale,   ///< `round(v * MaxValue / SrcMax)`. `QuantMat<N>::fromCVMat`'s rule, and
             ///< the exact inverse of `toCVMatNormalized` (D-42).
};

namespace impl {

} // namespace impl

namespace impl {

#if defined(BINCV_PACKQUANT_SIMD)

#if defined(BINCV_PACKQUANT_AVX2)
/// @brief Is AVX2 present? Asked once, not once per row.
/// @brief Force the portable path, for the benchmark and the tests. **INTERNAL.**
/// @note Not a tuning knob. It is how the vector path is held to BIT-EXACTNESS and how
///       a benchmark can show it is actually RUNNING -- X-89 shipped a vector block
///       that was compiled out and measured three "improvements" against it before the
///       kernel was timed in isolation and found not to respond to `-mavx2`.
inline bool& packQuantSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasPackQuantSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes && packQuantSimdEnabled();
}

/// @brief Thirty-two pixels quantised and transposed into N plane words. **INTERNAL.**
/// @param bits `out[p]` receives plane `p`'s thirty-two bits, LSB = lowest x, which is
///        `bitMask(x) = 1 << (x % WordBits)` -- the same convention X-71 relies on, so
///        `movemask_epi8`'s result IS the word with no shuffle.
template <size_t N>
__attribute__((target("avx2"))) inline void quantMask32(const uint8_t* src,
                                                        const uint8_t* thresholds,
                                                        unsigned maxValue, uint32_t* bits) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    // q = the number of thresholds this pixel clears. `cmpeq(max(v,t), v)` is `v >= t`
    // on UNSIGNED bytes -- `cmpgt_epi8` is signed and would invert above 127.
    __m256i q = _mm256_setzero_si256();
    for (unsigned k = 0; k < maxValue; ++k) {
        const __m256i tv = _mm256_set1_epi8(static_cast<char>(thresholds[k]));
        const __m256i ge = _mm256_cmpeq_epi8(_mm256_max_epu8(v, tv), v);
        q = _mm256_sub_epi8(q, ge);   // a set lane is -1, so subtracting adds one
    }
    for (size_t p = 0; p < N; ++p) {
        const __m256i sel = _mm256_set1_epi8(static_cast<char>(1u << p));
        const __m256i m = _mm256_cmpeq_epi8(_mm256_and_si256(q, sel), sel);
        bits[p] = static_cast<uint32_t>(_mm256_movemask_epi8(m));
    }
}
#else
/// @brief Force the portable path; see the AVX2 form above.
inline bool& packQuantSimdEnabled() {
    static bool on = true;
    return on;
}

inline bool hasPackQuantSimd() { return packQuantSimdEnabled(); }

/// @brief The same, without a move-mask. **INTERNAL.**
/// @note aarch64 has none, so AND with per-lane bit weights and let pairwise adds fold
///       sixteen byte masks into sixteen bits -- X-71's substitute, unchanged.
template <size_t N>
inline void quantMask32(const uint8_t* src, const uint8_t* thresholds, unsigned maxValue,
                        uint32_t* bits) {
    static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                   1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t weights = vld1q_u8(kW);
    for (size_t p = 0; p < N; ++p) bits[p] = 0;
    for (int half = 0; half < 2; ++half) {
        const uint8x16_t v = vld1q_u8(src + static_cast<size_t>(half) * 16);
        uint8x16_t q = vdupq_n_u8(0);
        for (unsigned k = 0; k < maxValue; ++k) {
            // NEON's compares ARE unsigned, so no bias is needed here.
            q = vsubq_u8(q, vcgeq_u8(v, vdupq_n_u8(thresholds[k])));
        }
        for (size_t p = 0; p < N; ++p) {
            const uint8x16_t sel = vdupq_n_u8(static_cast<uint8_t>(1u << p));
            const uint8x16_t m = vandq_u8(vceqq_u8(vandq_u8(q, sel), sel), weights);
            const uint32_t lo = vaddv_u8(vget_low_u8(m));
            const uint32_t hi = vaddv_u8(vget_high_u8(m));
            bits[p] |= (lo | (hi << 8)) << (half * 16);
        }
    }
}
#endif

#endif  // BINCV_PACKQUANT_SIMD

}  // namespace impl

/// @brief Packs a pixel array to **N bits per pixel**, no OpenCV. **API TIER 3.**
///
/// @tparam R The quantisation rule, at compile time -- [X-72](../../../EXPERIMENTS.md)
///         measured a runtime flag in a hot loop costing 17%, and this one is hotter.
/// @param dst The N destination planes, LSB first, exactly `QuantMat<N>::plane(i)`.
///        **Views, not the container** (CLAUDE.md): a kernel must not care how its
///        arguments were allocated.
///
/// @note `Scale`'s defaults reproduce `QuantMat<N>::fromCVMat` **bit for bit**, which
///       `test_pack.cpp` pins against it. That rule is load-bearing: it is
///       `toCVMatNormalized`'s exact inverse.
/// @note Each destination plane's padding bits are zero on return.
/// @note Never allocates and never throws.
template <QuantRule R, size_t N, typename SrcT, typename WordType>
inline void packQuant(const SrcT* src, size_t width, size_t height, size_t srcStride,
                      BinMatView<WordType> (&dst)[N]) {
    static_assert(N >= 1 && N <= 8, "packQuant: N outside QuantMat's supported range");
    for (size_t p = 0; p < N; ++p) {
        BINCV_ASSERT(width == dst[p].width && height == dst[p].height,
                     "packQuant: src and every plane must have the same dimensions");
        BINCV_ASSERT(impl::strideCoversARow<WordType>(dst[p].width, dst[p].height,
                                                      dst[p].stride),
                     "packQuant: a plane's stride must cover a whole row");
    }
    if (width == 0 || height == 0) return;
    BINCV_ASSERT(src != nullptr, "packQuant: a non-empty image needs a non-null pointer");

    constexpr unsigned kMaxValue = (1u << N) - 1u;
    SrcT thresholds[kMaxValue];
    impl::quantThresholds<SrcT>(kMaxValue, thresholds);

    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    const size_t words = impl::minRowWords<WordType>(width);
    static_assert(R == QuantRule::Scale, "packQuant: unknown rule");

    for (size_t y = 0; y < height; ++y) {
        const SrcT* rowIn = src + y * srcStride;
        WordType* out[N];
        for (size_t p = 0; p < N; ++p) {
            out[p] = dst[p].row(y);
            for (size_t i = 0; i < words; ++i) out[p][i] = 0;
        }
        size_t x = 0;
#if defined(BINCV_PACKQUANT_SIMD)
        // THE SCALE IS A HANDFUL OF COMPARISONS AND THE TRANSPOSE IS A MOVE-MASK.
        // `quantScale` is monotonic, so the value is the number of thresholds a pixel
        // clears -- `MaxValue` byte compares, three at N = 2. Extracting plane p is then
        // one AND, one compare and one move-mask per plane, which is X-71's trick with
        // the comparison replaced. A 256-entry lookup table, which is what
        // `fromCVMat` uses, cannot be done in a vector register at all.
        if constexpr (sizeof(SrcT) == 1 && sizeof(WordType) == 4 && kMaxValue <= 15) {
            if (impl::hasPackQuantSimd()) {
                for (; x + 32 <= width; x += 32) {
                    uint32_t bits[N];
                    impl::quantMask32<N>(reinterpret_cast<const uint8_t*>(rowIn + x),
                                         reinterpret_cast<const uint8_t*>(thresholds),
                                         kMaxValue, bits);
                    for (size_t p = 0; p < N; ++p) out[p][x / kBits] = bits[p];
                }
            }
        }
#endif
        // EIGHT PIXELS AND ALL N PLANES PER TRANSPOSE. `transpose8x8` turns a byte per
        // pixel into a bit per plane in three delta-swaps, so the portable path is ~3
        // operations per pixel for every plane rather than one bit test per (pixel,
        // plane). It is also the tail of the vector path above.
        for (; x < width; x += 8) {
            const size_t n = (width - x < 8) ? (width - x) : size_t{8};
            uint64_t m = 0;
            for (size_t i = 0; i < n; ++i) {
                m |= static_cast<uint64_t>(impl::quantScale<SrcT>(rowIn[x + i], kMaxValue))
                     << (8 * i);
            }
            const uint64_t tr = impl::transpose8x8(m);
            // A group of 8 never straddles a word: every WordType's width is a
            // multiple of 8.
            const size_t wi = x / kBits;
            const size_t shift = x % kBits;
            for (size_t p = 0; p < N; ++p) {
                const uint64_t b = (tr >> (8 * p)) & 0xFFu;
                out[p][wi] = static_cast<WordType>(out[p][wi] |
                                                   static_cast<WordType>(b << shift));
            }
        }
    }
}

/// @brief `packQuant` with an arbitrary per-pixel map. **API TIER 3.**
/// @param map Anything callable as `unsigned(SrcT)`, returning `0..(1 << N) - 1`. A
///        256-entry lookup table is `[&](SrcT v) { return lut[v]; }`.
/// @note **Slower on purpose**, for the reason `packBitsIf` is: a map the compiler
///       cannot see is a map the vector path cannot use. Reach for `packQuant` first.
/// @note Values above `(1 << N) - 1` are a programming error; the extra bits are
///       dropped rather than silently corrupting a neighbouring plane.
template <size_t N, typename SrcT, typename WordType, typename Map>
inline void packQuantWith(const SrcT* src, size_t width, size_t height, size_t srcStride,
                          BinMatView<WordType> (&dst)[N], Map map) {
    static_assert(N >= 1 && N <= 8, "packQuantWith: N outside QuantMat's supported range");
    if (width == 0 || height == 0) return;
    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    constexpr unsigned kMaxValue = (1u << N) - 1u;
    const size_t words = impl::minRowWords<WordType>(width);
    for (size_t y = 0; y < height; ++y) {
        const SrcT* rowIn = src + y * srcStride;
        WordType* out[N];
        for (size_t p = 0; p < N; ++p) {
            out[p] = dst[p].row(y);
            for (size_t i = 0; i < words; ++i) out[p][i] = 0;
        }
        for (size_t x = 0; x < width; ++x) {
            const unsigned q = static_cast<unsigned>(map(rowIn[x])) & kMaxValue;
            if (q == 0) continue;
            const size_t w = x / kBits;
            const WordType bit = static_cast<WordType>(WordType{1} << (x % kBits));
            for (size_t p = 0; p < N; ++p) {
                if ((q >> p) & 1u) out[p][w] = static_cast<WordType>(out[p][w] | bit);
            }
        }
    }
}

/// @brief `packBits` with an arbitrary per-pixel predicate. **API TIER 3.**
/// @note **Slower on purpose.** A predicate the compiler cannot see is a predicate
///       the vector path cannot use, so this is the portable loop always. Reach for
///       `PackRule` first; use this for a lookup table or a non-monotonic rule.
template <typename SrcT, typename WordType, typename Pred>
inline void packBitsIf(const SrcT* src, size_t width, size_t height, size_t srcStride,
                       BinMatView<WordType> dst, Pred pred) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "packBitsIf: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return;
    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    for (size_t y = 0; y < height; ++y) {
        const SrcT* rowIn = src + y * srcStride;
        WordType* rowOut = dst.row(y);
        for (size_t x = 0; x < width; x += kBits) {
            const size_t n = (width - x < kBits) ? (width - x) : kBits;
            WordType acc = 0;
            for (size_t i = 0; i < n; ++i)
                acc |= static_cast<WordType>(
                    static_cast<WordType>(pred(rowIn[x + i]) ? 1 : 0) << i);
            rowOut[x / kBits] = acc;
        }
    }
}

/// @brief The reverse: one bit per pixel out to one byte per pixel. **API TIER 3.**
/// @param onValue What a set bit becomes; `zeroValue` what a clear bit becomes.
/// @note **8 bits out is always enough and that is not a shortcut.** `QuantMat`
///       asserts `N <= 8`, so nothing binCV holds exceeds 255. The asymmetry with
///       the input side -- which needs 16 -- falls straight out of that.
template <typename WordType>
inline void unpackTo8Bit(BinMatConstView<WordType> src, uint8_t* dst, size_t dstStride,
                         uint8_t onValue = 255, uint8_t zeroValue = 0) {
    BINCV_ASSERT(dst != nullptr || src.height == 0,
                 "unpackTo8Bit: a non-empty image needs a non-null destination");
    if (src.width == 0 || src.height == 0) return;
    impl::unpackTo8BitRaw<WordType>(src.ptr, src.stride, src.width, src.height, dst,
                                    dstStride, onValue, zeroValue);
}

/// @brief Writes a binary image as a binary PGM (`P5`) to a caller-supplied buffer.
///        **API TIER 3.**
/// @return Bytes written, or the bytes REQUIRED if `cap` is too small (and nothing is
///         written). Call once with `cap == 0` to size the buffer.
/// @note **This is the only way to look at what binCV produced on a target with no
///       OpenCV**, and debugging a frontend you cannot see is not debugging. PGM is
///       chosen because it is the only image format whose encoder is a `printf` and a
///       `memcpy` -- binCV must not carry a real codec
///       ([ARCHITECTURE §7.8](../../../ARCHITECTURE.md): a PNG decoder is **eight
///       times the size of everything binCV does**, measured).
/// @note Takes a buffer rather than a path: `bincv_core` does no file I/O, has no
///       allocator and builds without exceptions. Where the bytes go is the caller's.
template <typename WordType>
inline size_t writePgm(BinMatConstView<WordType> src, uint8_t* out, size_t cap,
                       uint8_t onValue = 255, uint8_t zeroValue = 0) {
    // "P5\n<w> <h>\n255\n" -- built by hand because <cstdio> into a fixed buffer is
    // the one thing a freestanding target reliably lacks.
    uint8_t header[32];
    size_t h = 0;
    auto put = [&](char c) { if (h < sizeof(header)) header[h++] = static_cast<uint8_t>(c); };
    auto putNum = [&](size_t v) {
        char tmp[24];
        size_t n = 0;
        if (v == 0) tmp[n++] = '0';
        while (v > 0) { tmp[n++] = static_cast<char>('0' + (v % 10)); v /= 10; }
        while (n > 0) put(tmp[--n]);
    };
    put('P'); put('5'); put('\n');
    putNum(src.width); put(' '); putNum(src.height); put('\n');
    putNum(255); put('\n');

    const size_t need = h + src.width * src.height;
    if (out == nullptr || cap < need) return need;
    // `h` is bounded by `header`'s extent by construction (`put` refuses past it),
    // but saying so explicitly is what lets the compiler prove the copy is in range --
    // without it, -Warray-bounds cannot see past the early return and the gate is red.
    const size_t headerBytes = h < sizeof(header) ? h : sizeof(header);
    for (size_t i = 0; i < headerBytes; ++i) out[i] = header[i];
    unpackTo8Bit<WordType>(src, out + headerBytes, src.width, onValue, zeroValue);
    return need;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
