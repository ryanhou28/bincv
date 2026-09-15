#pragma once

/// @file denseDisparity.hpp
/// @brief Dense disparity from census matching: winner-take-all over a
/// window-aggregated Hamming cost, streamed so the cost volume never
/// exists. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// THE MEMORY RULE IS THE DESIGN
///
/// A dense stereo cost volume is `W x H x D` of something -- 23 MB at the
/// reference frame and 64 disparities, the largest allocation this library
/// would ever have touched. This kernel is built so it NEVER exists:
///
///   * ROW-STREAMED: only a band of census rows (window height, both images)
///     is resident -- the ring in the caller's scratch.
///   * DISPARITY-INNER: for each output row, every candidate disparity's
///     aggregated cost is computed and folded into a running per-pixel best.
///     Nothing outlives the row but the answer.
///
/// Peak scratch at 752x480, K = 24, 9x9, uint32: ~43 KB of census band + a few
/// hundred bytes of accumulator planes + three uint16 rows -- INDEPENDENT OF THE
/// DISPARITY RANGE. The price is recomputation: each output row re-evaluates its
/// window's raw cost per disparity instead of sliding it; whether a sliding
/// accumulator buys that back is a measured question the benchmark arm exists to
/// answer, not a foregone one.
///
/// ---------------------------------------------------------------------------
/// WHAT IS BIT-SLICED AND WHAT IS NOT, IN v1
///
/// The raw cost is fully word-parallel: per word of pixels, K census XORs and a
/// `bitSlicedSum` fold to a 5-bit lane-wise Hamming distance, then a ripple
/// carry adds the window's rows into an accumulator ladder -- W pixels per gate
/// throughout. The HORIZONTAL aggregation and the winner-take-all then run per
/// pixel from extracted integers. That scalarization is v1's known cost, priced
/// by the benchmark from birth; the bit-sliced borrow-compare WTA is the
/// restructuring the profile will justify or decline, exactly as the census
/// transform's own per-pixel v1 records its successor's target.
///
/// ---------------------------------------------------------------------------
/// CONTRACTS
///
/// * Rectified pair, left reference, disparity `d = xL - xR >= 0`, one byte per
///   output pixel. `kDenseDisparityInvalid` (255) marks pixels no candidate
///   could serve: the window rim, columns left of the disparity range's right
///   support, degenerate geometry.
/// * Ties keep the SMALLEST disparity -- the farthest surface, the conservative
///   reading of an ambiguous cost -- deterministically.
/// * Views on nothing: wide inputs and a byte map out; caller-provided scratch
///   (`denseDisparityScratchWords` / `denseDisparityScratchRows` size it); no
///   allocation, no throw.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../impl/kernel_util.hpp"
#include "bitslice.hpp"
#include "census.hpp"
#include "shift.hpp"   // impl::extendedRowWord, the bounded row read the shift owns

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief The disparity byte written where no candidate could be evaluated.
inline constexpr uint8_t kDenseDisparityInvalid = 255;

/// @brief Search and aggregation parameters for `denseDisparity`.
struct DenseDisparityParams {
    /// @brief Disparity range, `0 <= min <= max <= 254` (255 is the invalid
    /// marker). The range encodes "how close can a surface be".
    int minDisparity = 0;
    int maxDisparity = 64;

    /// @brief Aggregation window, odd, >= 3 on a side. Larger windows smooth
    /// the map and blur depth edges -- the standard trade, the caller's.
    int winWidth = 9;
    int winHeight = 9;
};

/// @brief WordType units of scratch `denseDisparity` needs: the two census
/// bands and the accumulator ladder. **API TIER 3.**
template <size_t K, typename WordType>
inline size_t denseDisparityScratchWords(size_t width, const DenseDisparityParams& p) {
    const size_t rowWords = impl::minRowWords<WordType>(width);
    const size_t winH = static_cast<size_t>(p.winHeight);
    const size_t accPlanes = bitSlicedSumPlanes(K * winH);
    return (2 * K * winH + accPlanes) * rowWords;
}

/// @brief uint16_t units of scratch `denseDisparity` needs: the extraction row
/// and the two running-best rows. **API TIER 3.**
inline size_t denseDisparityScratchRows(size_t width) { return 3 * width; }

namespace impl {

/// @brief Word `i` of a census row shifted RIGHT by `d` pixels -- the read that
/// aligns right-image column `x - d` under left column `x`. Out-of-row
/// bits read 0, so a left column with no right support accumulates a
/// meaningless cost that the caller's column gate never lets win.
/// **INTERNAL.**
template <typename WordType>
inline WordType censusWordShiftedRight(const WordType* row, size_t rowWords,
                                       WordType tailMask, size_t i, int d) {
    const size_t wb = bitsPerWord<WordType>();
    const size_t s = static_cast<size_t>(d) / wb;
    const unsigned r = static_cast<unsigned>(static_cast<size_t>(d) % wb);
    const ptrdiff_t j = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(s);
    const WordType hi = extendedRowWord<WordType>(row, j, rowWords, tailMask, WordType{0});
    if (r == 0) return hi;
    const WordType lo =
        extendedRowWord<WordType>(row, j - 1, rowWords, tailMask, WordType{0});
    return static_cast<WordType>(static_cast<WordType>(hi << r) |
                                 static_cast<WordType>(lo >> (wb - r)));
}

} // namespace impl

/// @brief Dense disparity over a rectified pair: census cost, box aggregation,
/// winner-take-all, one byte per pixel. **API TIER 3.**
/// @param left / right The rectified pair, row-major, strides in ELEMENTS.
/// @param pattern The census neighbourhood both images are transformed with.
/// @param scratchWords / scratchRows Caller-owned, sized by the two functions
/// above; contents are unspecified on return.
/// @param disparity `height * dispStride` bytes, every pixel written: a
/// disparity in `[minDisparity, maxDisparity]` or `kDenseDisparityInvalid`.
/// @note `cv::StereoBM` prices the same ROLE with different numerics (SAD on
/// prefiltered bytes, its own validity rules); the test suite closes the
/// triangle on synthetic ground truth rather than claiming equality.
template <size_t K, typename SrcT, typename WordType>
inline void denseDisparity(const SrcT* left, const SrcT* right, size_t width,
                           size_t height, size_t strideL, size_t strideR,
                           const CensusPattern<K>& pattern,
                           const DenseDisparityParams& params, WordType* scratchWords,
                           size_t scratchWordCount, uint16_t* scratchRows,
                           size_t scratchRowCount, uint8_t* disparity,
                           size_t dispStride) {
    static_assert(K >= 1 && K <= 32,
                  "denseDisparity: the per-word gather stages K census words on the"
                  " stack; 32 is the supported ceiling");
    BINCV_ASSERT(left != nullptr && right != nullptr && disparity != nullptr,
                 "denseDisparity: null image or output");
    BINCV_ASSERT(strideL >= width && strideR >= width && dispStride >= width,
                 "denseDisparity: strides must cover a row");
    BINCV_ASSERT(params.winWidth >= 3 && params.winHeight >= 3 &&
                     (params.winWidth & 1) == 1 && (params.winHeight & 1) == 1,
                 "denseDisparity: the window must be odd and at least 3 on a side");
    BINCV_ASSERT(params.minDisparity >= 0 && params.maxDisparity >= params.minDisparity &&
                     params.maxDisparity <= 254,
                 "denseDisparity: need 0 <= min <= max <= 254 (255 marks invalid)");
    BINCV_ASSERT((scratchWords != nullptr &&
                  scratchWordCount >=
                      denseDisparityScratchWords<K, WordType>(width, params)),
                 "denseDisparity: word scratch too small -- size it with"
                 " denseDisparityScratchWords");
    BINCV_ASSERT(scratchRows != nullptr &&
                     scratchRowCount >= denseDisparityScratchRows(width),
                 "denseDisparity: row scratch too small -- size it with"
                 " denseDisparityScratchRows");
    // The size parameters exist for the asserts, which release builds compile
    // away; the cast keeps the signature honest there too.
    static_cast<void>(scratchWordCount);
    static_cast<void>(scratchRowCount);
    if (width == 0 || height == 0) return;

    const size_t winH = static_cast<size_t>(params.winHeight);
    const int hw = params.winWidth / 2;
    const size_t hh = winH / 2;
    const size_t rowWords = impl::minRowWords<WordType>(width);
    const size_t costPlanes = bitSlicedSumPlanes(K);
    const size_t accPlanes = bitSlicedSumPlanes(K * winH);
    const WordType tailMask = impl::rowTailMask<WordType>(width);

    WordType* cenL = scratchWords;
    WordType* cenR = cenL + K * winH * rowWords;
    WordType* acc = cenR + K * winH * rowWords;
    uint16_t* ext = scratchRows;
    uint16_t* bestC = ext + width;
    uint16_t* bestD = bestC + width;

    const auto bandL = [&](size_t k, size_t r) {
        return cenL + (k * winH + r % winH) * rowWords;
    };
    const auto bandR = [&](size_t k, size_t r) {
        return cenR + (k * winH + r % winH) * rowWords;
    };
    const auto fillBandRow = [&](size_t r) {
        for (size_t k = 0; k < K; ++k) {
            impl::censusRow<SrcT, WordType>(left, width, height, strideL, pattern.at[k].dx,
                                            pattern.at[k].dy, r, bandL(k, r));
            impl::censusRow<SrcT, WordType>(right, width, height, strideR,
                                            pattern.at[k].dx, pattern.at[k].dy, r,
                                            bandR(k, r));
        }
    };
    const auto invalidRow = [&](size_t y) {
        uint8_t* out = disparity + y * dispStride;
        for (size_t x = 0; x < width; ++x) out[x] = kDenseDisparityInvalid;
    };

    // The largest disparity any column can support with a full window.
    const long long dMaxSupported =
        static_cast<long long>(width) - static_cast<long long>(params.winWidth);
    const int dEnd = params.maxDisparity <= dMaxSupported
                         ? params.maxDisparity
                         : static_cast<int>(dMaxSupported);
    if (height < winH || static_cast<size_t>(params.winWidth) > width ||
        dEnd < params.minDisparity) {
        for (size_t y = 0; y < height; ++y) invalidRow(y);
        return;
    }

    for (size_t y = 0; y < hh; ++y) invalidRow(y);
    for (size_t y = height - hh; y < height; ++y) invalidRow(y);

    for (size_t r = 0; r < winH; ++r) fillBandRow(r);

    for (size_t y = hh; y + hh < height; ++y) {
        if (y > hh) fillBandRow(y + hh);   // the band slides one row

        for (size_t x = 0; x < width; ++x) {
            bestC[x] = 0xFFFFu;
            bestD[x] = 0xFFFFu;
        }

        for (int d = params.minDisparity; d <= dEnd; ++d) {
            for (size_t w = 0; w < accPlanes * rowWords; ++w) acc[w] = 0;

            for (size_t wr = y - hh; wr <= y + hh; ++wr) {
                for (size_t i = 0; i < rowWords; ++i) {
                    WordType xr[K];
                    for (size_t k = 0; k < K; ++k) {
                        xr[k] = static_cast<WordType>(
                            bandL(k, wr)[i] ^
                            impl::censusWordShiftedRight<WordType>(bandR(k, wr), rowWords,
                                                                   tailMask, i, d));
                    }
                    WordType cost[8];
                    bitSlicedSum<WordType>(xr, K, cost);
                    // Ripple the 5-bit lane costs into the accumulator ladder.
                    // The plane budget bounds the value at K * winH, so the
                    // final carry is zero by construction.
                    WordType carry = 0;
                    for (size_t p = 0; p < accPlanes; ++p) {
                        const WordType v = p < costPlanes ? cost[p] : WordType{0};
                        WordType* a = acc + p * rowWords + i;
                        const WordType sum = static_cast<WordType>(*a ^ v ^ carry);
                        carry = maj3<WordType>(*a, v, carry);
                        *a = sum;
                    }
                }
            }

            // Extraction and the per-pixel half: window-column sum by rolling,
            // then the running best. v1's scalar stage, and the profile's
            // named target.
            constexpr size_t wb = impl::bitsPerWord<WordType>();
            for (size_t x = 0; x < width; ++x) {
                unsigned v = 0;
                const size_t wi = x / wb, b = x % wb;
                for (size_t p = 0; p < accPlanes; ++p) {
                    v |= static_cast<unsigned>((acc[p * rowWords + wi] >> b) & 1u) << p;
                }
                ext[x] = static_cast<uint16_t>(v);
            }
            const size_t xLo = static_cast<size_t>(hw) > static_cast<size_t>(d) + static_cast<size_t>(hw)
                                   ? static_cast<size_t>(hw)
                                   : static_cast<size_t>(d) + static_cast<size_t>(hw);
            if (xLo + static_cast<size_t>(hw) >= width) continue;
            int sum = 0;
            for (size_t u = xLo - static_cast<size_t>(hw);
                 u <= xLo + static_cast<size_t>(hw); ++u)
                sum += ext[u];
            for (size_t x = xLo; x + static_cast<size_t>(hw) < width; ++x) {
                if (x > xLo)
                    sum += static_cast<int>(ext[x + static_cast<size_t>(hw)]) -
                           static_cast<int>(ext[x - static_cast<size_t>(hw) - 1]);
                if (static_cast<unsigned>(sum) < bestC[x]) {
                    bestC[x] = static_cast<uint16_t>(sum);
                    bestD[x] = static_cast<uint16_t>(d);
                }
            }
        }

        uint8_t* out = disparity + y * dispStride;
        for (size_t x = 0; x < width; ++x) {
            out[x] = bestD[x] <= 254u ? static_cast<uint8_t>(bestD[x])
                                      : kDenseDisparityInvalid;
        }
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
