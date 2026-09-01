#pragma once

/// @file edge.hpp
/// @brief Gradient-magnitude edge extraction, 8- or 16-bit in, 1 bit out.
/// **API TIER 3** -- there is no `cv::` equivalent; see the tier note.
///
/// ---------------------------------------------------------------------------
/// READ OUT OF THE REFERENCE, NOT INFERRED
///
/// `SEAL/src/temporal_processing/edge_filter.cpp`, `rl_fast_edge_filter_wide`:
///
/// kernel_x = [-1 0 1] diff_x = |filter2D(img, kernel_x)|
/// kernel_y = [-1 0 1]^T diff_y = |filter2D(img, kernel_y)|
/// mask = (diff_x >= t) OR (diff_y >= t)
///
/// THREE DETAILS THAT ARE EASY TO GET WRONG, AND ALL THREE ARE THE DEFAULTS:
///
/// * the combination is **OR**, not AND;
/// * the relation is **`>=`**, not `>`;
/// * **"wide" is the CENTRAL difference** `[-1, 0, 1]` -- the left neighbour
/// against the RIGHT neighbour, spanning two pixels -- not an adjacent `[-1, 1]`.
///
/// A caller who says nothing gets exactly the reference's operation.
///
/// ---------------------------------------------------------------------------
/// ALL TWELVE COMBINATIONS SHIP
///
/// combine {Or, And} x relation {Ge, Gt} x spatial {Wide, Forward, Backward}. They
/// are compile-time parameters, so a caller pays only for the one instantiated and
/// the comparison folds to a single predicate -- the same requirement ops/pack.hpp
/// puts on its rules, for the same measured reason
/// ( a runtime flag cost 17% elsewhere).
///
/// The point of the operation is that these choices are cheap. A caller wanting AND
/// instead of OR, or an adjacent difference instead of a central one, should not have
/// to fork the kernel.
///
/// ---------------------------------------------------------------------------
/// WHY `SrcT` AND NOT JUST `uint8_t` (the design notes)
///
/// **This operation is why that section exists.** "Downconvert 12->8 yourself, then
/// call binCV" is `v >> 4`, which truncates the OPERANDS before they are differenced:
/// a genuine 12-bit gradient of 15 counts becomes **exactly zero**, and the edge is
/// gone before binCV sees the pixel. Low contrast -- indoors, at night, on untextured
/// walls -- is where a VIO frontend needs every edge it can get, so the workaround
/// fails hardest exactly where it matters.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND THE NAME IS NOT OPENCV'S
///
/// `cv::Sobel` + `cv::threshold` is a DIFFERENT computation: a 3x3 separable kernel
/// with smoothing, against this operation's single-axis difference. CLAUDE.md forbids
/// borrowing an OpenCV name for an operation that does not match it.
///
/// The border rule IS OpenCV's, though: `cv::filter2D` defaults to
/// `BORDER_REFLECT_101`, so index -1 reads index 1 and index `w` reads index `w-2`.
/// That is what makes the shipped defaults reproduce the reference exactly rather
/// than approximately.
// F-5: BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"


#include <cstddef>
#include <cstdint>

#include "../binMat.hpp"
#include "../impl/kernel_util.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_EDGE_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_EDGE_NEON 1
#include <arm_neon.h>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief How the two axes' results are combined. `Or` is the reference's.
enum class EdgeCombine { Or, And };

/// @brief How a gradient is compared with the threshold. `Ge` is the reference's.
enum class EdgeRelation { Ge, Gt };

/// @brief Which pixels are differenced. `Wide` is the reference's.
enum class EdgeSpatial {
    Wide,      ///< `|v[x+1] - v[x-1]|` -- the central difference, spanning two pixels
    Forward,   ///< `|v[x+1] - v[x]|`
    Backward,  ///< `|v[x] - v[x-1]|`
};

namespace impl {

/// @brief `BORDER_REFLECT_101`: index -1 reads 1, index n reads n-2. **INTERNAL.**
/// @note OpenCV's `filter2D` default, and therefore the reference's. A one-pixel
/// extent reflects to itself.
inline size_t reflect101Edge(long long i, size_t n) {
    if (n <= 1) return 0;
    const long long last = static_cast<long long>(n) - 1;
    while (i < 0 || i > last) {
        if (i < 0) i = -i;
        if (i > last) i = 2 * last - i;
    }
    return static_cast<size_t>(i);
}

/// @brief One axis' absolute difference at `(y, x)`. **INTERNAL.**
/// @tparam Horizontal True for the x axis, false for the y axis.
template <EdgeSpatial S, bool Horizontal, typename SrcT>
inline long long axisDiff(const SrcT* src, size_t width, size_t height, size_t stride,
                          size_t y, size_t x) {
    long long aI, bI;   // the two indices along the axis, before reflection
    const long long here = static_cast<long long>(Horizontal ? x : y);
    if constexpr (S == EdgeSpatial::Wide) {
        aI = here + 1;
        bI = here - 1;
    } else if constexpr (S == EdgeSpatial::Forward) {
        aI = here + 1;
        bI = here;
    } else {
        aI = here;
        bI = here - 1;
    }
    const size_t n = Horizontal ? width : height;
    const size_t a = reflect101Edge(aI, n);
    const size_t b = reflect101Edge(bI, n);
    const long long va =
        static_cast<long long>(Horizontal ? src[y * stride + a] : src[a * stride + x]);
    const long long vb =
        static_cast<long long>(Horizontal ? src[y * stride + b] : src[b * stride + x]);
    const long long d = va - vb;
    return d < 0 ? -d : d;
}

template <EdgeRelation R>
inline bool passesEdge(long long d, long long t) {
    if constexpr (R == EdgeRelation::Ge) return d >= t;
    return d > t;
}

/// @brief Is `(y, x)` an edge? **INTERNAL** -- the whole predicate, in one place.
template <EdgeCombine C, EdgeRelation R, EdgeSpatial S, typename SrcT>
inline bool isEdge(const SrcT* src, size_t width, size_t height, size_t stride, size_t y,
                   size_t x, long long t) {
    const bool px =
        passesEdge<R>(axisDiff<S, true, SrcT>(src, width, height, stride, y, x), t);
    // SHORT-CIRCUITING IS CORRECT AND WORTH HAVING: on a sparse edge map most pixels
    // fail both tests, so `And` skips the vertical difference for nearly all of them.
    if constexpr (C == EdgeCombine::Or) {
        if (px) return true;
    } else {
        if (!px) return false;
    }
    return passesEdge<R>(axisDiff<S, false, SrcT>(src, width, height, stride, y, x), t);
}


// ==================================================================
// earlier work: THE EDGE PREDICATE, THIRTY-TWO PIXELS AT A TIME.
//
// `|a - b| >= t` on unsigned bytes is `subs_epu8(a,b) | subs_epu8(b,a)` for the absolute
// difference and `subs_epu8(t, d) == 0` for the comparison — **no widening, no sign, no
// branch.** The result is a byte mask, and `movemask_epi8` turns thirty-two of those into
// the thirty-two bits of an output word directly, LSB first, which is exactly binCV's bit
// order. **The 8-bit intermediate this operation exists to avoid never appears
// even inside the kernel.**
//
// aarch64 has no move-mask, so bit weights and three pairwise adds fold sixteen byte
// masks into sixteen bits — the same substitute a measurement measured for the row packer.
//
// The scalar body above stays: it is the border rule, the general `SrcT`, the general
// `EdgeSpatial`, and the oracle `tests/test_edge.cpp` compares against.

#if defined(BINCV_EDGE_AVX2)
inline bool hasEdgeSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}

/// @brief `|a - b| >= tp` for thirty-two unsigned bytes, as a byte mask.
__attribute__((target("avx2"))) inline __m256i edgePass32(__m256i a, __m256i b, int tp) {
    const __m256i d = _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
    return _mm256_cmpeq_epi8(_mm256_subs_epu8(_mm256_set1_epi8(static_cast<char>(tp)), d),
                             _mm256_setzero_si256());
}

/// @brief Thirty-two pixels of the edge predicate, as thirty-two bits.
/// @param tp The threshold ALREADY adjusted for the relation: `t` for `Ge`, `t + 1` for
/// `Gt`, so the comparison itself has only one spelling.
template <EdgeCombine C>
__attribute__((target("avx2"))) inline uint32_t edgeMask32(const uint8_t* rowUp,
                                                           const uint8_t* row,
                                                           const uint8_t* rowDn,
                                                           const uint8_t* colA,
                                                           const uint8_t* colB, int tp) {
    const __m256i px = edgePass32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(colA)),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(colB)),
                                  tp);
    const __m256i py = edgePass32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowDn)),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rowUp)),
                                  tp);
    (void)row;
    const __m256i m = (C == EdgeCombine::Or) ? _mm256_or_si256(px, py)
                                             : _mm256_and_si256(px, py);
    return static_cast<uint32_t>(_mm256_movemask_epi8(m));
}
#elif defined(BINCV_EDGE_NEON)
inline bool hasEdgeSimd() { return true; }

inline uint8x16_t edgePass16(uint8x16_t a, uint8x16_t b, int tp) {
    const uint8x16_t d = vorrq_u8(vqsubq_u8(a, b), vqsubq_u8(b, a));
    return vceqq_u8(vqsubq_u8(vdupq_n_u8(static_cast<uint8_t>(tp)), d), vdupq_n_u8(0));
}

/// @brief Sixteen byte masks into sixteen bits, LSB first. No move-mask on aarch64.
inline uint32_t edgeFold16(uint8x16_t m) {
    static const uint8_t kW[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                   1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t w = vandq_u8(m, vld1q_u8(kW));
    const uint8x8_t lo = vget_low_u8(w), hi = vget_high_u8(w);
    const uint8_t a = vaddv_u8(lo), b = vaddv_u8(hi);
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8);
}

template <EdgeCombine C>
inline uint32_t edgeMask32(const uint8_t* rowUp, const uint8_t* row, const uint8_t* rowDn,
                           const uint8_t* colA, const uint8_t* colB, int tp) {
    (void)row;
    uint32_t out = 0;
    for (int half = 0; half < 2; ++half) {
        const size_t o = static_cast<size_t>(half) * 16;
        const uint8x16_t px = edgePass16(vld1q_u8(colA + o), vld1q_u8(colB + o), tp);
        const uint8x16_t py = edgePass16(vld1q_u8(rowDn + o), vld1q_u8(rowUp + o), tp);
        const uint8x16_t m = (C == EdgeCombine::Or) ? vorrq_u8(px, py) : vandq_u8(px, py);
        out |= edgeFold16(m) << (half * 16);
    }
    return out;
}
#endif

} // namespace impl

/// @brief Gradient-magnitude edge extraction straight into bits. **API TIER 3.**
///
/// @param src Row-major, `srcStride` ELEMENTS between rows (not bytes).
/// @param t Threshold, in source units.
///
/// **THE DEFAULTS ARE THE REFERENCE.** `edgeThreshold(src, w, h, stride, dst, 17)`
/// with no template arguments is `rl_fast_edge_filter_wide(img, 17)`.
///
/// @note 8-bit-in, 1-bit-out, and **the byte never exists**: the comparison yields a
/// boolean per pixel which goes straight into a word. Computing an 8-bit edge
/// image and packing it afterwards would be two passes and an intermediate.
/// @note `dst`'s padding bits are zero on return.
/// @note Never allocates and never throws.
template <EdgeCombine C = EdgeCombine::Or, EdgeRelation R = EdgeRelation::Ge,
          EdgeSpatial S = EdgeSpatial::Wide, typename SrcT = uint8_t,
          typename WordType = uint32_t>
inline void edgeThreshold(const SrcT* src, size_t width, size_t height, size_t srcStride,
                          BinMatView<WordType> dst, SrcT t) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "edgeThreshold: src and dst must have the same dimensions");
    BINCV_ASSERT(impl::strideCoversARow<WordType>(dst.width, dst.height, dst.stride),
                 "edgeThreshold: dst's stride must cover a whole row");
    if (dst.width == 0 || dst.height == 0) return;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "edgeThreshold: a non-empty image needs non-null pointers");

    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    const long long tt = static_cast<long long>(t);

#if defined(BINCV_EDGE_AVX2) || defined(BINCV_EDGE_NEON)
    // earlier work: the vector interior. Only the shipped shape — 8-bit source, 32-bit
    // words, the WIDE central difference — because that is what the sensor stage runs
    // and every other combination still has the scalar body, which is also the oracle.
    //
    // A word is taken whole or not at all: the mask covers pixels [32w, 32w+32), so the
    // block qualifies only when all of them, AND their +/-1 neighbours, are interior.
    // At 752 pixels that is 22 of 24 words; the other two and the first and last rows
    // take the scalar path below.
    if constexpr (S == EdgeSpatial::Wide && sizeof(SrcT) == 1 && sizeof(WordType) == 4) {
        const long long tp = (R == EdgeRelation::Ge) ? tt : tt + 1;
        if (impl::hasEdgeSimd() && height >= 3 && width >= kBits + 1 && tp <= 255) {
            const uint8_t* s8 = reinterpret_cast<const uint8_t*>(src);
            const size_t words = (width + kBits - 1) / kBits;
            const auto mask32 = [&](size_t y, size_t xs) {
                return impl::edgeMask32<C>(s8 + (y - 1) * srcStride + xs,
                                           s8 + y * srcStride + xs,
                                           s8 + (y + 1) * srcStride + xs,
                                           s8 + y * srcStride + xs + 1,
                                           s8 + y * srcStride + xs - 1,
                                           static_cast<int>(tp));
            };
            for (size_t y = 1; y + 1 < height; ++y) {
                WordType* row = dst.row(y);
                for (size_t w = 0; w < words; ++w) {
                    const size_t x = w * kBits;
                    if (x + kBits <= width) {
                        // Reads `x - 1` through `x + 32`. At `x == 0` the low read lands
                        // on the previous row's last byte -- in bounds because this
                        // sweep skips row 0 -- and produces a wrong bit 0, fixed below.
                        row[w] = static_cast<WordType>(mask32(y, x));
                    } else {
                        // THE TAIL WORD, BY AN OVERLAPPING WINDOW rather than by falling
                        // back to the scalar predicate for sixteen pixels a row. Anchor
                        // the vector at `width - 32`, which is fully in bounds, and shift
                        // the bits into place; the padding bits shift in as zero, which
                        // is the invariant that would otherwise need a mask.
                        const size_t xs = width - kBits;
                        row[w] = static_cast<WordType>(mask32(y, xs) >> (x - xs));
                    }
                }
                // The two columns whose neighbour is outside the image. Two pixels a
                // row, against the sixty-four that skipping whole words used to cost.
                for (const size_t bx : {size_t{0}, width - 1}) {
                    const bool on =
                        impl::isEdge<C, R, S, SrcT>(src, width, height, srcStride, y, bx, tt);
                    WordType& wd = row[bx / kBits];
                    const WordType bit = static_cast<WordType>(WordType{1} << (bx % kBits));
                    wd = on ? static_cast<WordType>(wd | bit)
                            : static_cast<WordType>(wd & ~bit);
                }
            }
            // The first and last rows, whose vertical neighbour is outside.
            for (const size_t by : {size_t{0}, height - 1}) {
                WordType* row = dst.row(by);
                for (size_t x = 0; x < width; x += kBits) {
                    const size_t n = (width - x < kBits) ? (width - x) : kBits;
                    WordType acc = 0;
                    for (size_t i = 0; i < n; ++i) {
                        if (impl::isEdge<C, R, S, SrcT>(src, width, height, srcStride, by,
                                                        x + i, tt)) {
                            acc = static_cast<WordType>(acc | (WordType{1} << i));
                        }
                    }
                    row[x / kBits] = acc;
                }
            }
            return;
        }
    }
#endif

    for (size_t y = 0; y < height; ++y) {
        WordType* row = dst.row(y);
        for (size_t x = 0; x < width; x += kBits) {
            const size_t n = (width - x < kBits) ? (width - x) : kBits;
            WordType acc = 0;
            for (size_t i = 0; i < n; ++i) {
                if (impl::isEdge<C, R, S, SrcT>(src, width, height, srcStride, y, x + i, tt))
                    acc = static_cast<WordType>(acc | (WordType{1} << i));
            }
            // STORED, not OR-ed: the bits past `width` are never set, which is the
            // padding invariant with no mask needed.
            row[x / kBits] = acc;
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
