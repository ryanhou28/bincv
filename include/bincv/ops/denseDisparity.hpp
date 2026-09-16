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
/// Two vertical-aggregation arms, both under the rule, chosen by
/// `DenseDisparityParams::recomputeVertical`:
///
///   * SLIDING (the default): a per-disparity accumulator ring -- each row
///     advance costs one bit-sliced add and one subtract per disparity instead
///     of re-evaluating the window's rows. Scratch grows LINEAR in D at ~0.8 KB
///     per disparity (u64): ~92 KB total at the reference configuration --
///     still an order of magnitude under StereoBM's output alone.
///   * RECOMPUTE: v1's shape, scratch INDEPENDENT of D (~47 KB), for the target
///     where that property outranks ~4x of the band work. The equality of the
///     two arms' output maps is a test, not a hope.
///
/// ---------------------------------------------------------------------------
/// WHAT IS BIT-SLICED AND WHAT IS NOT
///
/// The raw cost is fully word-parallel: per word of pixels, K census XORs and a
/// `bitSlicedSum` fold to a lane-wise Hamming distance, rippled into the
/// vertical accumulator -- W pixels per gate throughout. Extraction to integers
/// runs eight pixels at a time through the 8x8 bit transpose whenever the
/// accumulator fits eight planes (K * winHeight <= 255 -- the shipped 5x5/9x9
/// configuration is 216), with the per-pixel loop as the general fallback. The
/// horizontal rolling sum and the winner-take-all stay scalar: two adds and a
/// compare per pixel, measured cheap enough to leave.
///
/// WORD TYPE, measured rather than defaulted: this kernel is pure word
/// arithmetic with no narrow-guarded vector paths, and its scratch is a band,
/// not a frame -- so the library's uint32 default (argued from row-stride waste)
/// buys nothing here, and **uint64 measured 1.63x faster on the reference
/// device**. The binary path's NEON arm exists at uint64 only, for the same
/// reason it is the recommended type. Callers should instantiate this kernel at
/// uint64 unless they have measured a reason otherwise.
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
#include <type_traits>

#ifdef BINCV_DENSE_STAGE_TIMING
#include <chrono>
#endif

// The winner-take-all's vector arm. F-5: BEFORE THE GATE, NOT AFTER -- simd.hpp
// defines BINCV_HAVE_NEON from the compiler's own macros on aarch64, so an
// include-only integration still gets the NEON kernels, and BINCV_NO_NEON
// remains the deliberate off-switch a benchmark times the scalar arm through.
#include "../core/simd.hpp"
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

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

    /// @brief Take the RECOMPUTE arm: re-evaluate the vertical window per row
    /// per disparity, keeping scratch INDEPENDENT of the disparity range
    /// (v1's shape). Off by default -- the sliding accumulator is ~4x the
    /// band work back for ~0.8 KB of scratch per disparity -- and switchable
    /// so the benchmark can time both and the tests can hold them to
    /// IDENTICAL output maps.
    bool recomputeVertical = false;
};

/// @brief WordType units of scratch `denseDisparity` needs: the two census
/// bands and the accumulator ladder. **API TIER 3.**
template <size_t K, typename WordType>
inline size_t denseDisparityScratchWords(size_t width, const DenseDisparityParams& p) {
    const size_t rowWords = impl::minRowWords<WordType>(width);
    const size_t winH = static_cast<size_t>(p.winHeight);
    const size_t accPlanes = bitSlicedSumPlanes(K * winH);
    // The sliding arm keeps one accumulator per candidate disparity; the
    // recompute arm keeps one, full stop. Both sit beside the census band.
    const size_t range = p.recomputeVertical
                             ? 1
                             : static_cast<size_t>(p.maxDisparity - p.minDisparity) + 1;
    return (2 * K * winH + range * accPlanes) * rowWords;
}

/// @brief uint16_t units of scratch `denseDisparity` needs: the extraction row
/// and the two running-best rows. **API TIER 3.**
inline size_t denseDisparityScratchRows(size_t width) { return 3 * width; }

namespace impl {

#ifdef BINCV_DENSE_STAGE_TIMING
/// @brief Where `denseDisparityBinary`'s time goes, by stage. **DIAGNOSTIC ONLY —
/// off by default and compiled out otherwise.** Exists so a vector pass lands on
/// the stage that owns the time instead of the one that looks expensive; the LK
/// stage profile earned its keep the same way after three guessed optimizations
/// in a row measured under 2%. Nanoseconds, accumulated; the clock reads sit in
/// the per-disparity loop, a few percent of overhead that is compared only
/// against itself.
struct DenseStageTiming {
    uint64_t ringNs = 0;     ///< the vertical accumulator slide, all disparities
    uint64_t treeNs = 0;     ///< the horizontal doubling tree, all disparities
    uint64_t wtaNs = 0;      ///< compare, range mask and the masked selects
    uint64_t extractNs = 0;  ///< best-disparity planes to output bytes
};
inline DenseStageTiming& denseStageTiming() {
    static DenseStageTiming t;
    return t;
}
inline uint64_t denseStageNow() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}
#endif

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

#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
/// @brief Two uint64 words of `laneShiftDown` -- words `i` and `i+1` of the
/// result, whose reads (`i+s` .. `i+s+2`) the CALLER has bounds-checked; the
/// scalar body keeps the words it cannot prove in-row. `j = 64*s + r`.
/// **INTERNAL.**
/// @note `r == 0` needs no branch: a USHL by 64 reads as zero, so the high
/// word contributes nothing and the low word passes through unshifted.
inline uint64x2_t laneShiftDownPair(const uint64_t* row, size_t i, size_t s,
                                    unsigned r) {
    const uint64x2_t lo = vld1q_u64(row + i + s);
    const uint64x2_t hi = vld1q_u64(row + i + s + 1);
    const int64x2_t shr = vdupq_n_s64(-static_cast<int64_t>(r));
    const int64x2_t shl = vdupq_n_s64(static_cast<int64_t>(64u - r));
    return vorrq_u64(vshlq_u64(lo, shr), vshlq_u64(hi, shl));
}

/// @brief Two uint64 words of `censusWordShiftedRight`, same caller-checked
/// bounds contract as `laneShiftDownPair` (reads `i-s-1` .. `i-s+1`, all
/// strictly inside the row, so the tail mask never applies). `d = 64*s + r`.
/// **INTERNAL.**
inline uint64x2_t censusShiftRightPair(const uint64_t* row, size_t i, size_t s,
                                       unsigned r) {
    const uint64x2_t hi = vld1q_u64(row + i - s);
    const uint64x2_t lo = vld1q_u64(row + i - s - 1);
    const int64x2_t shl = vdupq_n_s64(static_cast<int64_t>(r));
    const int64x2_t shr = vdupq_n_s64(-static_cast<int64_t>(64u - r));
    return vorrq_u64(vshlq_u64(hi, shl), vshlq_u64(lo, shr));
}
#endif  // BINCV_HAVE_NEON && __aarch64__

/// @brief `acc += v`, bit-sliced: `v` carries `vPlanes` planes, `acc` carries
/// `accPlanes`, at word column `i` of rows `rowWords` wide. The plane budget
/// bounds the value, so the final carry is zero by construction. **INTERNAL.**
template <typename WordType>
inline void accAddWord(WordType* acc, size_t accPlanes, size_t rowWords, size_t i,
                       const WordType* v, size_t vPlanes) {
    WordType carry = 0;
    for (size_t p = 0; p < accPlanes; ++p) {
        const WordType vp = p < vPlanes ? v[p] : WordType{0};
        WordType* a = acc + p * rowWords + i;
        const WordType sum = static_cast<WordType>(*a ^ vp ^ carry);
        carry = maj3<WordType>(*a, vp, carry);
        *a = sum;
    }
}

/// @brief `acc -= v`, bit-sliced borrow ripple. The accumulator always holds at
/// least the row being removed (it was added by the same arithmetic), so
/// the final borrow is zero by construction. **INTERNAL.**
template <typename WordType>
inline void accSubWord(WordType* acc, size_t accPlanes, size_t rowWords, size_t i,
                       const WordType* v, size_t vPlanes) {
    WordType borrow = 0;
    for (size_t p = 0; p < accPlanes; ++p) {
        const WordType vp = p < vPlanes ? v[p] : WordType{0};
        WordType* a = acc + p * rowWords + i;
        const WordType diff = static_cast<WordType>(*a ^ vp ^ borrow);
        borrow = static_cast<WordType>(
            (static_cast<WordType>(~*a) & (vp | borrow)) | (vp & borrow));
        *a = diff;
    }
}

/// @brief Word `i` of a plane row with LANES SHIFTED DOWN by `j` (lane x reads
/// lane x + j); out-of-row lanes read 0. **INTERNAL.**
template <typename WordType>
inline WordType laneShiftDown(const WordType* row, size_t rowWords, size_t i, unsigned j) {
    const size_t wb = bitsPerWord<WordType>();
    const size_t sWords = j / wb;
    const unsigned r = static_cast<unsigned>(j % wb);
    const size_t lo = i + sWords;
    const WordType a = lo < rowWords ? row[lo] : WordType{0};
    if (r == 0) return a;
    const WordType b = lo + 1 < rowWords ? row[lo + 1] : WordType{0};
    return static_cast<WordType>(static_cast<WordType>(a >> r) |
                                 static_cast<WordType>(b << (wb - r)));
}

/// @brief `dst += shift(src, j)` over `planes` plane rows, bit-sliced ripple,
/// all arrays `planeCap` planes wide. **INTERNAL.**
template <typename WordType>
inline void planesAddShifted(WordType* dst, const WordType* src, size_t planeCap,
                             size_t rowWords, unsigned j) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    // The tree owned two thirds of the binary path's time on the reference
    // device before this arm existed, so it is the one that pays for lanes.
    // Word-pair outer, planes inner, the carry in a REGISTER: the plane loop's
    // serial carry chain that forced the scalar body plane-outer is two lanes
    // wide here and never touches memory. In-place doubling (`dst == src`) is
    // safe the same way it is in scalar: every read of a pair's words happens
    // before its store, and reads never reach below the writing index.
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        const size_t s = j / 64u;
        const unsigned r = j % 64u;
        size_t i = 0;
        for (; i + 2 <= rowWords && i + s + 3 <= rowWords; i += 2) {
            uint64x2_t carry = vdupq_n_u64(0);
            for (size_t p = 0; p < planeCap; ++p) {
                uint64_t* a = dst + p * rowWords + i;
                const uint64x2_t av = vld1q_u64(a);
                const uint64x2_t vv = laneShiftDownPair(src + p * rowWords, i, s, r);
                const uint64x2_t t = veorq_u64(av, vv);
                vst1q_u64(a, veorq_u64(t, carry));
                carry = vorrq_u64(vandq_u64(av, vv), vandq_u64(carry, t));
            }
        }
        // The words whose shifted reads the pair loop cannot prove in-row:
        // the original word-inner ripple, whose bounds-checked read is the
        // contract here.
        for (; i < rowWords; ++i) {
            uint64_t carry = 0;
            for (size_t p = 0; p < planeCap; ++p) {
                const uint64_t v =
                    laneShiftDown<uint64_t>(src + p * rowWords, rowWords, i, j);
                uint64_t* a = dst + p * rowWords + i;
                const uint64_t sum = *a ^ v ^ carry;
                carry = maj3<uint64_t>(*a, v, carry);
                *a = sum;
            }
        }
        return;
    }
#endif
    // Plane-outer with a carry ROW: each pass is a straight word loop with no
    // cross-iteration dependence, which is the shape the vectorizer takes. The
    // arrays here are a handful of L1-resident rows, so this is NOT the
    // row-staging that measured slower on the band -- that one paid strided
    // gathers across 24 staged rows; this one re-walks at most eight.
    constexpr size_t kCarryCap = 128;   // 1 KB at uint64; a row wider than this
                                        // (a >8K-pixel frame at uint64, narrower
                                        // sooner) takes the word-inner path.
    if (rowWords <= kCarryCap) {
        WordType carryRow[kCarryCap];
        for (size_t i = 0; i < rowWords; ++i) carryRow[i] = 0;
        for (size_t p = 0; p < planeCap; ++p) {
            WordType* a = dst + p * rowWords;
            const WordType* sp = src + p * rowWords;
            for (size_t i = 0; i < rowWords; ++i) {
                const WordType v = laneShiftDown<WordType>(sp, rowWords, i, j);
                const WordType sum = static_cast<WordType>(a[i] ^ v ^ carryRow[i]);
                carryRow[i] = maj3<WordType>(a[i], v, carryRow[i]);
                a[i] = sum;
            }
        }
        return;
    }
    for (size_t i = 0; i < rowWords; ++i) {
        WordType carry = 0;
        for (size_t p = 0; p < planeCap; ++p) {
            const WordType v = laneShiftDown<WordType>(src + p * rowWords, rowWords, i, j);
            WordType* a = dst + p * rowWords + i;
            const WordType sum = static_cast<WordType>(*a ^ v ^ carry);
            carry = maj3<WordType>(*a, v, carry);
            *a = sum;
        }
    }
}

/// @brief The lanes where `h < best`, as a mask word per row word: the borrow
/// out of the lane-wise subtraction `h - best`. **INTERNAL.**
template <typename WordType>
inline void planesLess(const WordType* h, const WordType* best, size_t planeCap,
                       size_t rowWords, WordType* maskOut) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    // Same shape as the adder's arm: the per-word borrow chain that keeps the
    // scalar body word-inner rides in a register, two lanes at a time. No
    // shifted reads, so only an odd last word falls to scalar.
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        size_t i = 0;
        for (; i + 2 <= rowWords; i += 2) {
            uint64x2_t borrow = vdupq_n_u64(0);
            for (size_t p = 0; p < planeCap; ++p) {
                const uint64x2_t hp = vld1q_u64(h + p * rowWords + i);
                const uint64x2_t bp = vld1q_u64(best + p * rowWords + i);
                borrow = vorrq_u64(vbicq_u64(vorrq_u64(bp, borrow), hp),
                                   vandq_u64(bp, borrow));
            }
            vst1q_u64(maskOut + i, borrow);
        }
        for (; i < rowWords; ++i) {
            uint64_t borrow = 0;
            for (size_t p = 0; p < planeCap; ++p) {
                const uint64_t hp = h[p * rowWords + i];
                const uint64_t bp = best[p * rowWords + i];
                borrow = (~hp & (bp | borrow)) | (bp & borrow);
            }
            maskOut[i] = borrow;
        }
        return;
    }
#endif
    for (size_t i = 0; i < rowWords; ++i) {
        WordType borrow = 0;
        for (size_t p = 0; p < planeCap; ++p) {
            const WordType hp = h[p * rowWords + i];
            const WordType bp = best[p * rowWords + i];
            borrow = static_cast<WordType>(
                (static_cast<WordType>(~hp) & (bp | borrow)) | (bp & borrow));
        }
        maskOut[i] = borrow;
    }
}

/// @brief `best = mask ? h : best`, per plane -- the winner-take-all's masked
/// cost update. **INTERNAL.**
template <typename WordType>
inline void planesSelect(WordType* best, const WordType* h, const WordType* mask,
                         size_t planeCap, size_t rowWords) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        for (size_t p = 0; p < planeCap; ++p) {
            uint64_t* b = best + p * rowWords;
            const uint64_t* hp = h + p * rowWords;
            size_t i = 0;
            for (; i + 2 <= rowWords; i += 2)
                vst1q_u64(b + i, vbslq_u64(vld1q_u64(mask + i), vld1q_u64(hp + i),
                                           vld1q_u64(b + i)));
            for (; i < rowWords; ++i)
                b[i] = (b[i] & ~mask[i]) | (hp[i] & mask[i]);
        }
        return;
    }
#endif
    for (size_t p = 0; p < planeCap; ++p) {
        for (size_t i = 0; i < rowWords; ++i) {
            WordType* b = best + p * rowWords + i;
            const WordType hp = h[p * rowWords + i];
            *b = static_cast<WordType>((*b & static_cast<WordType>(~mask[i])) |
                                       (hp & mask[i]));
        }
    }
}

/// @brief One row-slide of one disparity's vertical accumulator: subtract the
/// leaving cost row, add the entering one, both computed in place from the
/// packed inputs. **INTERNAL.**
///
/// The vector arm fuses the two ripples per plane -- the borrow reads the
/// PRE-subtract word and the carry the post-subtract one, which is exactly the
/// sequential sub-then-add composed column-wise, so the scalar arm below and
/// this one produce identical words.
template <typename WordType>
inline void accSlideRow(WordType* acc, size_t accPlanes, size_t rowWords,
                        const WordType* leftLeave, const WordType* rightLeave,
                        const WordType* leftEnter, const WordType* rightEnter,
                        WordType tailMask, int d) {
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    if constexpr (std::is_same_v<WordType, uint64_t>) {
        const size_t s = static_cast<size_t>(d) / 64u;
        const unsigned r = static_cast<unsigned>(static_cast<size_t>(d) % 64u);
        const auto scalarWord = [&](size_t i) {
            const uint64_t cl = leftLeave[i] ^ censusWordShiftedRight<uint64_t>(
                                                   rightLeave, rowWords, tailMask, i, d);
            accSubWord<uint64_t>(acc, accPlanes, rowWords, i, &cl, 1);
            const uint64_t ce = leftEnter[i] ^ censusWordShiftedRight<uint64_t>(
                                                   rightEnter, rowWords, tailMask, i, d);
            accAddWord<uint64_t>(acc, accPlanes, rowWords, i, &ce, 1);
        };
        size_t i = 0;
        for (; i < rowWords && i < s + 1; ++i) scalarWord(i);
        for (; i + 2 <= rowWords && i + 3 <= rowWords + s; i += 2) {
            uint64x2_t borrow = veorq_u64(vld1q_u64(leftLeave + i),
                                          censusShiftRightPair(rightLeave, i, s, r));
            uint64x2_t carry = veorq_u64(vld1q_u64(leftEnter + i),
                                         censusShiftRightPair(rightEnter, i, s, r));
            for (size_t p = 0; p < accPlanes; ++p) {
                uint64_t* a = acc + p * rowWords + i;
                const uint64x2_t av = vld1q_u64(a);
                const uint64x2_t t = veorq_u64(av, borrow);
                borrow = vbicq_u64(borrow, av);
                vst1q_u64(a, veorq_u64(t, carry));
                carry = vandq_u64(t, carry);
            }
        }
        for (; i < rowWords; ++i) scalarWord(i);
        return;
    }
#endif
    for (size_t i = 0; i < rowWords; ++i) {
        const WordType cl = static_cast<WordType>(
            leftLeave[i] ^
            censusWordShiftedRight<WordType>(rightLeave, rowWords, tailMask, i, d));
        accSubWord<WordType>(acc, accPlanes, rowWords, i, &cl, 1);
        const WordType ce = static_cast<WordType>(
            leftEnter[i] ^
            censusWordShiftedRight<WordType>(rightEnter, rowWords, tailMask, i, d));
        accAddWord<WordType>(acc, accPlanes, rowWords, i, &ce, 1);
    }
}

/// @brief The word mask of lanes in `[lo, hi]` (inclusive), for word `i`.
/// **INTERNAL.**
template <typename WordType>
inline WordType laneRangeMask(size_t i, long long lo, long long hi) {
    const long long wb = static_cast<long long>(bitsPerWord<WordType>());
    const long long base = static_cast<long long>(i) * wb;
    long long a = lo - base, b = hi - base;
    if (b < 0 || a >= wb) return 0;
    if (a < 0) a = 0;
    if (b >= wb) b = wb - 1;
    const WordType full = static_cast<WordType>(~WordType{0});
    const WordType upTo =
        b + 1 >= wb ? full
                    : static_cast<WordType>(
                          static_cast<WordType>(WordType{1} << (b + 1)) - WordType{1});
    return static_cast<WordType>(upTo &
                                 static_cast<WordType>(full << a));
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
    WordType* accBase = cenR + K * winH * rowWords;
    const size_t range = params.recomputeVertical
                             ? 1
                             : static_cast<size_t>(params.maxDisparity -
                                                   params.minDisparity) + 1;
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
    // The raw census cost of word `i` of band row `wr` at disparity `d`:
    // K XORs against the shifted right census, folded lane-wise. Deliberately
    // PER WORD, computed in registers: a row-staged spelling (shift and XOR
    // whole rows into scratch, then gather columns for the fold) measured 12%
    // SLOWER on the reference device -- the strided column gather across K
    // staged rows costs more cache traffic than the autovectorizer's straight
    // loops give back. Recorded so it is not retried on the same shape.
    const auto rawCostWord = [&](size_t wr, int d, size_t i, WordType* cost) {
        WordType xr[K];
        for (size_t k = 0; k < K; ++k) {
            xr[k] = static_cast<WordType>(
                bandL(k, wr)[i] ^ impl::censusWordShiftedRight<WordType>(
                                      bandR(k, wr), rowWords, tailMask, i, d));
        }
        bitSlicedSum<WordType>(xr, K, cost);
    };
    const auto accFor = [&](int d) {
        return accBase + (params.recomputeVertical
                              ? size_t{0}
                              : static_cast<size_t>(d - params.minDisparity) *
                                    accPlanes * rowWords);
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

    // The sliding arm's accumulators, seeded from the first window.
    if (!params.recomputeVertical) {
        for (size_t w = 0; w < range * accPlanes * rowWords; ++w) accBase[w] = 0;
        for (int d = params.minDisparity; d <= dEnd; ++d) {
            WordType* acc = accFor(d);
            for (size_t wr = 0; wr < winH; ++wr) {
                for (size_t i = 0; i < rowWords; ++i) {
                    WordType cost[8];
                    rawCostWord(wr, d, i, cost);
                    impl::accAddWord<WordType>(acc, accPlanes, rowWords, i, cost,
                                               costPlanes);
                }
            }
        }
    }

    constexpr size_t wb = impl::bitsPerWord<WordType>();
    for (size_t y = hh; y + hh < height; ++y) {
        if (y > hh) {
            if (!params.recomputeVertical) {
                // The leaving row still occupies its ring slot -- the entering
                // row lands in the SAME slot (they are winHeight rows apart) --
                // so the order is: subtract, refill, add.
                const size_t leave = y - 1 - hh;
                for (int d = params.minDisparity; d <= dEnd; ++d) {
                    WordType* acc = accFor(d);
                    for (size_t i = 0; i < rowWords; ++i) {
                        WordType cost[8];
                        rawCostWord(leave, d, i, cost);
                        impl::accSubWord<WordType>(acc, accPlanes, rowWords, i, cost,
                                                   costPlanes);
                    }
                }
                fillBandRow(y + hh);
                const size_t enter = y + hh;
                for (int d = params.minDisparity; d <= dEnd; ++d) {
                    WordType* acc = accFor(d);
                    for (size_t i = 0; i < rowWords; ++i) {
                        WordType cost[8];
                        rawCostWord(enter, d, i, cost);
                        impl::accAddWord<WordType>(acc, accPlanes, rowWords, i, cost,
                                                   costPlanes);
                    }
                }
            } else {
                fillBandRow(y + hh);
            }
        }

        for (size_t x = 0; x < width; ++x) {
            bestC[x] = 0xFFFFu;
            bestD[x] = 0xFFFFu;
        }

        for (int d = params.minDisparity; d <= dEnd; ++d) {
            WordType* acc = accFor(d);
            if (params.recomputeVertical) {
                for (size_t w = 0; w < accPlanes * rowWords; ++w) acc[w] = 0;
                for (size_t wr = y - hh; wr <= y + hh; ++wr) {
                    for (size_t i = 0; i < rowWords; ++i) {
                        WordType cost[8];
                        rawCostWord(wr, d, i, cost);
                        impl::accAddWord<WordType>(acc, accPlanes, rowWords, i, cost,
                                                   costPlanes);
                    }
                }
            }

            // Extraction. Eight pixels per 8x8 bit transpose when the ladder
            // fits eight planes (K * winHeight <= 255); the per-pixel loop is
            // the general fallback. Both produce the same integers.
            if (accPlanes <= 8) {
                for (size_t x0 = 0; x0 < width; x0 += 8) {
                    const size_t wi = x0 / wb;
                    const unsigned b8 = static_cast<unsigned>(((x0 % wb) / 8) * 8);
                    uint64_t gather = 0;
                    for (size_t pp = 0; pp < accPlanes; ++pp) {
                        gather |= static_cast<uint64_t>(
                                      (acc[pp * rowWords + wi] >> b8) &
                                      static_cast<WordType>(0xFF))
                                  << (8 * pp);
                    }
                    const uint64_t t = impl::transpose8x8(gather);
                    const size_t lim = width - x0 < 8 ? width - x0 : 8;
                    for (size_t j = 0; j < lim; ++j)
                        ext[x0 + j] = static_cast<uint16_t>((t >> (8 * j)) & 0xFFu);
                }
            } else {
                for (size_t x = 0; x < width; ++x) {
                    unsigned v = 0;
                    const size_t wi = x / wb, b = x % wb;
                    for (size_t pp = 0; pp < accPlanes; ++pp) {
                        v |= static_cast<unsigned>((acc[pp * rowWords + wi] >> b) & 1u)
                             << pp;
                    }
                    ext[x] = static_cast<uint16_t>(v);
                }
            }

            const size_t xLo =
                static_cast<size_t>(hw) > static_cast<size_t>(d) + static_cast<size_t>(hw)
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

/// @brief WordType units of scratch `denseDisparityBinary` needs. **API TIER 3.**
/// @note No census band exists on this path -- the packed frames ARE the band --
/// so the whole scratch is the accumulator ladder: per-disparity when
/// sliding, one ladder when recomputing.
template <typename WordType>
inline size_t denseDisparityBinaryScratchWords(size_t width,
                                               const DenseDisparityParams& p) {
    const size_t rowWords = impl::minRowWords<WordType>(width);
    const size_t accPlanes = bitSlicedSumPlanes(static_cast<size_t>(p.winHeight));
    const size_t range = p.recomputeVertical
                             ? 1
                             : static_cast<size_t>(p.maxDisparity - p.minDisparity) + 1;
    // The winner-take-all's plane arrays: window sums (hPlanes), a doubling
    // block of the same width, best cost (hPlanes), best disparity (8), and a
    // compare-mask row.
    const size_t hPlanes = bitSlicedSumPlanes(static_cast<size_t>(p.winWidth) *
                                              static_cast<size_t>(p.winHeight));
    return (range * accPlanes + 2 * hPlanes + hPlanes + 8) * rowWords + rowWords;
}

/// @brief Dense disparity over an ALREADY-BINARY rectified pair: the cost is
/// `popcount((L ^ shift(R, d)) over window)` -- one XOR per word of 64
/// pixels, no census, no wide image anywhere. **API TIER 3.**
///
/// THIS IS THE PREMISE-NATIVE PATH. The census spelling above exists for
/// callers holding WIDE frames; a binCV pipeline already holds bits, and on
/// bits the dense cost is the same window Hamming the block-match tracker
/// runs -- with an arithmetic budget an order of magnitude BELOW a byte SAD's,
/// which is what being custom-built for this representation is supposed to
/// buy. Same output contract, same arms, same invalid marker as the census
/// spelling; the aggregation window doubles as the matching support, so
/// `winWidth * winHeight <= 255` keeps the extraction in bytes (asserted).
///
/// On aarch64 at uint64 the plane loops run a NEON arm -- two words per lane
/// with the ripple carries in registers -- switchable off with BINCV_NO_NEON
/// and bit-exact against the scalar arm, which every test pins to the same
/// output maps on both architectures.
///
/// @note A window with no texture (all-equal bits against all-equal bits) has
/// an ambiguous cost everywhere and resolves to the smallest disparity by
/// the tie rule. A texture-validity gate is a caller-side filter today --
/// `countAnd`/windowed counts price one cheaply -- and a recorded
/// follow-up, not a silent promise.
template <typename WordType>
inline void denseDisparityBinary(BinMatConstView<WordType> left,
                                 BinMatConstView<WordType> right,
                                 const DenseDisparityParams& params,
                                 WordType* scratchWords, size_t scratchWordCount,
                                 uint16_t* scratchRows, size_t scratchRowCount,
                                 uint8_t* disparity, size_t dispStride) {
    BINCV_ASSERT(disparity != nullptr, "denseDisparityBinary: null output");
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "denseDisparityBinary: the pair must share its extent");
    const size_t width = left.width, height = left.height;
    BINCV_ASSERT(dispStride >= width, "denseDisparityBinary: dispStride must cover a row");
    BINCV_ASSERT(params.winWidth >= 3 && params.winHeight >= 3 &&
                     (params.winWidth & 1) == 1 && (params.winHeight & 1) == 1,
                 "denseDisparityBinary: the window must be odd and at least 3 on a side");
    BINCV_ASSERT(params.winWidth * params.winHeight <= 255,
                 "denseDisparityBinary: winWidth * winHeight must fit a byte");
    BINCV_ASSERT(params.minDisparity >= 0 && params.maxDisparity >= params.minDisparity &&
                     params.maxDisparity <= 254,
                 "denseDisparityBinary: need 0 <= min <= max <= 254 (255 marks invalid)");
    BINCV_ASSERT((scratchWords != nullptr &&
                  scratchWordCount >=
                      denseDisparityBinaryScratchWords<WordType>(width, params)),
                 "denseDisparityBinary: word scratch too small");
    BINCV_ASSERT((scratchRows != nullptr &&
                  scratchRowCount >= denseDisparityScratchRows(width)),
                 "denseDisparityBinary: row scratch too small");
    static_cast<void>(scratchWordCount);
    static_cast<void>(scratchRowCount);
    if (width == 0 || height == 0) return;

    const size_t winH = static_cast<size_t>(params.winHeight);
    const int hw = params.winWidth / 2;
    const size_t hh = winH / 2;
    const size_t rowWords = impl::minRowWords<WordType>(width);
    const size_t accPlanes = bitSlicedSumPlanes(winH);
    const WordType tailMask = impl::rowTailMask<WordType>(width);
    const size_t range = params.recomputeVertical
                             ? 1
                             : static_cast<size_t>(params.maxDisparity -
                                                   params.minDisparity) + 1;

    const size_t hPlanes = bitSlicedSumPlanes(static_cast<size_t>(params.winWidth) *
                                              winH);
    WordType* accBase = scratchWords;
    WordType* wtaH = accBase + range * accPlanes * rowWords;
    WordType* wtaBlock = wtaH + hPlanes * rowWords;
    WordType* wtaBestC = wtaBlock + hPlanes * rowWords;
    WordType* wtaBestD = wtaBestC + hPlanes * rowWords;
    WordType* wtaMask = wtaBestD + 8 * rowWords;
    static_cast<void>(scratchRows);   // the bit-sliced stage needs no integer rows

    const auto invalidRow = [&](size_t y) {
        uint8_t* out = disparity + y * dispStride;
        for (size_t x = 0; x < width; ++x) out[x] = kDenseDisparityInvalid;
    };
    // The 1-bit raw cost of word `i` of image row `r` at disparity `d`.
    const auto costWord = [&](size_t r, int d, size_t i) {
        return static_cast<WordType>(
            left.row(r)[i] ^ impl::censusWordShiftedRight<WordType>(
                                 right.row(r), rowWords, tailMask, i, d));
    };
    const auto accFor = [&](int d) {
        return accBase + (params.recomputeVertical
                              ? size_t{0}
                              : static_cast<size_t>(d - params.minDisparity) *
                                    accPlanes * rowWords);
    };

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
    static_cast<void>(range);

    for (size_t y = 0; y < hh; ++y) invalidRow(y);
    for (size_t y = height - hh; y < height; ++y) invalidRow(y);

    if (!params.recomputeVertical) {
        for (size_t w = 0;
             w < (static_cast<size_t>(dEnd - params.minDisparity) + 1) * accPlanes *
                     rowWords;
             ++w)
            accBase[w] = 0;
        for (int d = params.minDisparity; d <= dEnd; ++d) {
            WordType* acc = accFor(d);
            for (size_t wr = 0; wr < winH; ++wr) {
                for (size_t i = 0; i < rowWords; ++i) {
                    const WordType c = costWord(wr, d, i);
                    impl::accAddWord<WordType>(acc, accPlanes, rowWords, i, &c, 1);
                }
            }
        }
    }

    constexpr size_t wb = impl::bitsPerWord<WordType>();
    for (size_t y = hh; y + hh < height; ++y) {
        if (y > hh && !params.recomputeVertical) {
#ifdef BINCV_DENSE_STAGE_TIMING
            const uint64_t t0 = impl::denseStageNow();
#endif
            const size_t leave = y - 1 - hh;
            const size_t enter = y + hh;
            for (int d = params.minDisparity; d <= dEnd; ++d) {
                impl::accSlideRow<WordType>(accFor(d), accPlanes, rowWords,
                                            left.row(leave), right.row(leave),
                                            left.row(enter), right.row(enter),
                                            tailMask, d);
            }
#ifdef BINCV_DENSE_STAGE_TIMING
            impl::denseStageTiming().ringNs += impl::denseStageNow() - t0;
#endif
        }

        // ---- the bit-sliced winner-take-all ----
        // Everything stays word-parallel until one extraction per ROW: the
        // horizontal window is a doubling tree of lane-shifted adds, the
        // compare is a lane-wise borrow, the update is a masked select, and
        // the disparity itself lives as eight bit-planes until the row is
        // written. Anchor convention: lane x carries the window COVERING
        // lanes [x, x + winWidth), i.e. the center x + hw -- the output loop
        // shifts back by hw.
        for (size_t w = 0; w < hPlanes * rowWords; ++w) wtaBestC[w] = static_cast<WordType>(~WordType{0});
        for (size_t w = 0; w < 8 * rowWords; ++w) wtaBestD[w] = static_cast<WordType>(~WordType{0});

        for (int d = params.minDisparity; d <= dEnd; ++d) {
            WordType* acc = accFor(d);
            if (params.recomputeVertical) {
                for (size_t w = 0; w < accPlanes * rowWords; ++w) acc[w] = 0;
                for (size_t wr = y - hh; wr <= y + hh; ++wr) {
                    for (size_t i = 0; i < rowWords; ++i) {
                        const WordType c = costWord(wr, d, i);
                        impl::accAddWord<WordType>(acc, accPlanes, rowWords, i, &c, 1);
                    }
                }
            }

#ifdef BINCV_DENSE_STAGE_TIMING
            const uint64_t tTree = impl::denseStageNow();
#endif
            // H = sum over j in [0, winWidth) of laneShift(acc, j), by binary
            // decomposition: `block` holds the sum of a power-of-two run.
            for (size_t w = 0; w < hPlanes * rowWords; ++w) wtaH[w] = 0;
            for (size_t p = 0; p < accPlanes; ++p)
                for (size_t i = 0; i < rowWords; ++i)
                    wtaBlock[p * rowWords + i] = acc[p * rowWords + i];
            for (size_t p = accPlanes; p < hPlanes; ++p)
                for (size_t i = 0; i < rowWords; ++i) wtaBlock[p * rowWords + i] = 0;
            unsigned runWidth = 1, offset = 0;
            unsigned remaining = static_cast<unsigned>(params.winWidth);
            while (true) {
                if (remaining & 1u) {
                    impl::planesAddShifted<WordType>(wtaH, wtaBlock, hPlanes, rowWords,
                                                     offset);
                    offset += runWidth;
                }
                remaining >>= 1u;
                if (remaining == 0) break;
                // block <- block + shift(block, runWidth): a run twice as wide.
                impl::planesAddShifted<WordType>(wtaBlock, wtaBlock, hPlanes, rowWords,
                                                 runWidth);
                runWidth *= 2u;
            }

#ifdef BINCV_DENSE_STAGE_TIMING
            const uint64_t tWta = impl::denseStageNow();
            impl::denseStageTiming().treeNs += tWta - tTree;
#endif
            // Lanes this disparity may claim: anchors in [d, width - winWidth].
            impl::planesLess<WordType>(wtaH, wtaBestC, hPlanes, rowWords, wtaMask);
            const long long anchorHi =
                static_cast<long long>(width) - params.winWidth;
            for (size_t i = 0; i < rowWords; ++i) {
                wtaMask[i] = static_cast<WordType>(
                    wtaMask[i] & impl::laneRangeMask<WordType>(i, d, anchorHi));
            }
            impl::planesSelect<WordType>(wtaBestC, wtaH, wtaMask, hPlanes, rowWords);
            for (size_t p = 0; p < 8; ++p) {
                const bool bit = ((static_cast<unsigned>(d) >> p) & 1u) != 0;
                for (size_t i = 0; i < rowWords; ++i) {
                    WordType* b = wtaBestD + p * rowWords + i;
                    const WordType dp = bit ? wtaMask[i] : WordType{0};
                    *b = static_cast<WordType>((*b & static_cast<WordType>(~wtaMask[i])) |
                                               dp);
                }
            }
#ifdef BINCV_DENSE_STAGE_TIMING
            impl::denseStageTiming().wtaNs += impl::denseStageNow() - tWta;
#endif
        }

#ifdef BINCV_DENSE_STAGE_TIMING
        const uint64_t tExtract = impl::denseStageNow();
#endif
        // One extraction per row: best-disparity planes to bytes, shifted from
        // anchors back to centers. Unclaimed lanes hold all-ones = 255 = the
        // invalid marker, so the sentinel semantics fall out of the init.
        uint8_t* out = disparity + y * dispStride;
        for (size_t x = 0; x < width; ++x) {
            const size_t xa = x - static_cast<size_t>(hw);
            uint8_t v = kDenseDisparityInvalid;
            if (x >= static_cast<size_t>(hw) && x + static_cast<size_t>(hw) < width) {
                unsigned acc8 = 0;
                const size_t wi = xa / wb, b = xa % wb;
                for (size_t p = 0; p < 8; ++p)
                    acc8 |= static_cast<unsigned>((wtaBestD[p * rowWords + wi] >> b) & 1u)
                            << p;
                v = static_cast<uint8_t>(acc8);
            }
            out[x] = v;
        }
#ifdef BINCV_DENSE_STAGE_TIMING
        impl::denseStageTiming().extractNs += impl::denseStageNow() - tExtract;
#endif
    }
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
