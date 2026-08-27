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
    const size_t words = impl::minRowWords<WordType>(dst.width);
    for (size_t y = 0; y < height; ++y) {
        WordType* out = dst.row(y);
        // The shared packer ORs, so the row starts clean. A row is tens of words:
        // noise beside the packing, and it makes the padding invariant unconditional.
        for (size_t i = 0; i < words; ++i) out[i] = 0;
        impl::packRowCmp<impl::toPackCmp<R>(), SrcT, WordType>(src + y * srcStride, width, t,
                                                               out);
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
    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    for (size_t y = 0; y < src.height; ++y) {
        const WordType* rowIn = src.row(y);
        uint8_t* rowOut = dst + y * dstStride;
        for (size_t x = 0; x < src.width; ++x) {
            const WordType w = rowIn[x / kBits];
            rowOut[x] = ((w >> (x % kBits)) & WordType{1}) ? onValue : zeroValue;
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
