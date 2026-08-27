#pragma once

/// @file fast.hpp
/// @brief FAST corner detection on a wide image. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// WHY binCV HAS THIS WHEN IT ALREADY DETECTS CORNERS
///
/// binCV detects with Shi-Tomasi -- `cornerMinEigenVal` / `goodFeaturesToTrack` --
/// which is what LK wants, because it scores exactly the thing LK needs: a window
/// whose gradient covariance is well conditioned.
///
/// FAST is what the ORB-SLAM family detects with, and it is here because
/// [ops/descriptor.hpp](descriptor.hpp) is. A descriptor pipeline wants corners that
/// are repeatable under rotation and cheap to find, and Shi-Tomasi is neither of
/// those things first. **They ship together or not at all** -- FAST without
/// descriptors would be a detector nobody asked for.
///
/// ---------------------------------------------------------------------------
/// TIER 2, NOT TIER 1, AND THE DIFFERENCE IS THE SCORE
///
/// The DETECTION rule is `cv::FAST`'s exactly: `arcLength` contiguous pixels of the
/// 16-pixel Bresenham ring all brighter than `centre + t`, or all darker than
/// `centre - t`. Same ring, same order, same contiguity-wraps-around rule.
///
/// The SCORE is not. OpenCV scores a corner by binary-searching the largest threshold
/// at which it survives; this sums how far the qualifying arc exceeds the threshold.
/// **Both order corners sensibly and they do not agree**, so non-maximum suppression
/// over them can keep different points. Tier 2 says exactly that: same role, same call
/// shape, different numerics.

#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_FAST_RUNTIME_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_FAST_NEON 1
#include <arm_neon.h>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

/// @brief One detected corner.
struct FastCorner {
    int x;
    int y;
    long long score;   ///< see the tier note -- NOT `cv::FAST`'s score
};

namespace impl {

/// @brief The 16-pixel Bresenham ring of radius 3, clockwise from straight up.
/// @note The order matters: contiguity is defined ALONG this ring, so a different
///       winding would accept different corners. This is `cv::FAST`'s order.
inline constexpr int kFastRingX[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
inline constexpr int kFastRingY[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

} // namespace impl

namespace impl {

#if defined(BINCV_FAST_RUNTIME_AVX2)
/// @brief Which of 32 consecutive pixels are FAST corners, as 32 bits. **INTERNAL.**
///
/// **THE RING LOADS ARE CONTIGUOUS AND THAT IS THE WHOLE TRICK.** For a horizontal run
/// of 32 pixels, ring position `k` is 32 consecutive bytes at a fixed offset -- so the
/// sixteen neighbourhoods cost **sixteen vector loads per 32 pixels** where the scalar
/// loop paid sixteen scalar loads per pixel. That is a 32x reduction in loads before
/// any comparison happens.
///
/// The contiguity test then runs VERTICALLY across the sixteen masks:
/// `run = (run + 1) & mask` resets to zero wherever the ring pixel fails, and a lane
/// whose run reaches `arcLength` is a corner. No transpose, no per-lane branch.
///
/// @note The comparisons are UNSIGNED via saturating arithmetic: `v > hi` is
///       `subs_epu8(v, hi) != 0`. SSE/AVX byte compares are signed, and the bias trick
///       ops/pack.hpp uses would cost two extra ops per ring position here.
///       Saturation also gives the clamp for free -- `c + t` saturating at 255 means
///       "nothing is brighter", which is exactly right.
__attribute__((target("avx2"))) inline uint32_t fastMask32(const uint8_t* p, long long stride,
                                                           const long long* ringOff,
                                                           int threshold, int arcLength,
                                                           int minCompass) {
    const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i t = _mm256_set1_epi8(static_cast<char>(threshold));
    const __m256i hi = _mm256_adds_epu8(c, t);
    const __m256i lo = _mm256_subs_epu8(c, t);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i limit = _mm256_set1_epi8(static_cast<char>(arcLength - 1));
    (void)stride;

    const __m256i allOnes = _mm256_set1_epi8(-1);
    __m256i mHi[16], mLo[16];
    // FOUR COMPASS POSITIONS BEFORE THE OTHER TWELVE. About 1% of pixels on a real
    // frame are corners, so nearly every 32-pixel group can be dismissed from four
    // loads -- and dismissing it here skips twelve loads AND the whole run-length
    // loop, which is most of this function. The scalar path has had this reject since
    // it was written; the vector path was paying full price on every group.
    for (int c4 = 0; c4 < 4; ++c4) {
        const int k = c4 * 4;
        const __m256i v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + ringOff[k]));
        mHi[k] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(v, hi), zero), allOnes);
        mLo[k] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(lo, v), zero), allOnes);
    }
    {
        // Per lane, count how many of the four compass points pass. A mask byte is
        // 0xFF, so `sub` of the mask ADDS one -- four subtracts give the count.
        __m256i cntHi = zero, cntLo = zero;
        for (int c4 = 0; c4 < 4; ++c4) {
            cntHi = _mm256_sub_epi8(cntHi, mHi[c4 * 4]);
            cntLo = _mm256_sub_epi8(cntLo, mLo[c4 * 4]);
        }
        const __m256i need = _mm256_set1_epi8(static_cast<char>(minCompass - 1));
        const __m256i any = _mm256_or_si256(_mm256_cmpgt_epi8(cntHi, need),
                                            _mm256_cmpgt_epi8(cntLo, need));
        if (_mm256_movemask_epi8(any) == 0) return 0;
    }
    for (int k = 0; k < 16; ++k) {
        if ((k & 3) == 0) continue;   // already loaded above
        const __m256i v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + ringOff[k]));
        // v > hi  <=>  subs_epu8(v, hi) != 0
        mHi[k] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(v, hi), zero), allOnes);
        // v < lo  <=>  subs_epu8(lo, v) != 0
        mLo[k] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(lo, v), zero), allOnes);
    }

    __m256i runH = zero, runL = zero, found = zero;
    const int steps = 16 + arcLength - 1;
    for (int k = 0; k < steps; ++k) {
        const int idx = k & 15;
        runH = _mm256_and_si256(_mm256_add_epi8(runH, ones), mHi[idx]);
        runL = _mm256_and_si256(_mm256_add_epi8(runL, ones), mLo[idx]);
        found = _mm256_or_si256(found, _mm256_cmpgt_epi8(runH, limit));
        found = _mm256_or_si256(found, _mm256_cmpgt_epi8(runL, limit));
    }
    return static_cast<uint32_t>(_mm256_movemask_epi8(found));
}

inline bool hasFastAvx2() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}
/// @brief Pixels per vector iteration. AVX2 is 32 bytes wide.
constexpr size_t kFastLanes = 32;
#elif defined(BINCV_FAST_NEON)
/// @brief Which of 16 consecutive pixels are FAST corners, as 16 bits. **INTERNAL.**
///
/// **THE SAME STRUCTURE AS THE AVX2 PATH, AND NEON MAKES TWO PARTS OF IT CHEAPER.**
/// The ring loads are contiguous the same way -- ring position `k` is 16 consecutive
/// bytes at a fixed offset -- so sixteen `vld1q_u8` replace sixteen scalar loads *per
/// pixel*.
///
/// Where x86 needs `subs_epu8(v, hi) != 0` to get an unsigned compare out of signed
/// instructions, **NEON compares unsigned natively**: `vcgtq_u8` and `vcltq_u8`, one
/// instruction each and no bias. Saturation is still used for the thresholds
/// (`vqaddq_u8` / `vqsubq_u8`), because `c + t` stopping at 255 is exactly the rule
/// "nothing is brighter than a saturated centre".
///
/// The only thing NEON lacks is a move-mask, and ops/pack.hpp already carries the
/// answer: AND per-lane bit weights, then three pairwise adds fold sixteen bytes into
/// a sixteen-bit mask.
inline uint32_t fastMask32(const uint8_t* p, long long stride, const long long* ringOff,
                           int threshold, int arcLength, int minCompass) {
    (void)stride;
    const uint8x16_t c = vld1q_u8(p);
    const uint8x16_t t = vdupq_n_u8(static_cast<uint8_t>(threshold));
    const uint8x16_t hi = vqaddq_u8(c, t);
    const uint8x16_t lo = vqsubq_u8(c, t);
    const uint8x16_t one = vdupq_n_u8(1);
    const uint8x16_t limit = vdupq_n_u8(static_cast<uint8_t>(arcLength - 1));

    uint8x16_t mHi[16], mLo[16];
    // FOUR COMPASS POSITIONS BEFORE THE OTHER TWELVE -- see the AVX2 path. On a real
    // frame about 1% of pixels are corners, so nearly every group is dismissed here
    // and skips twelve loads plus the whole run-length loop.
    for (int c4 = 0; c4 < 4; ++c4) {
        const int k = c4 * 4;
        const uint8x16_t v = vld1q_u8(p + ringOff[k]);
        mHi[k] = vcgtq_u8(v, hi);
        mLo[k] = vcltq_u8(v, lo);
    }
    {
        uint8x16_t cntHi = vdupq_n_u8(0), cntLo = vdupq_n_u8(0);
        const uint8x16_t one8 = vdupq_n_u8(1);
        for (int c4 = 0; c4 < 4; ++c4) {
            cntHi = vaddq_u8(cntHi, vandq_u8(mHi[c4 * 4], one8));
            cntLo = vaddq_u8(cntLo, vandq_u8(mLo[c4 * 4], one8));
        }
        const uint8x16_t need = vdupq_n_u8(static_cast<uint8_t>(minCompass));
        const uint8x16_t any = vorrq_u8(vcgeq_u8(cntHi, need), vcgeq_u8(cntLo, need));
        if (vmaxvq_u8(any) == 0) return 0;
    }
    for (int k = 0; k < 16; ++k) {
        if ((k & 3) == 0) continue;   // already loaded above
        const uint8x16_t v = vld1q_u8(p + ringOff[k]);
        mHi[k] = vcgtq_u8(v, hi);   // native unsigned compare -- no bias, no saturate
        mLo[k] = vcltq_u8(v, lo);
    }

    uint8x16_t runH = vdupq_n_u8(0), runL = vdupq_n_u8(0), found = vdupq_n_u8(0);
    const int steps = 16 + arcLength - 1;
    for (int k = 0; k < steps; ++k) {
        const int idx = k & 15;
        runH = vandq_u8(vaddq_u8(runH, one), mHi[idx]);
        runL = vandq_u8(vaddq_u8(runL, one), mLo[idx]);
        found = vorrq_u8(found, vcgtq_u8(runH, limit));
        found = vorrq_u8(found, vcgtq_u8(runL, limit));
    }

    // No move-mask on aarch64: AND bit weights, then fold sixteen bytes to sixteen bits.
    const uint8x16_t weights = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const uint8x16_t w = vandq_u8(found, weights);
    uint8x8_t f = vpadd_u8(vget_low_u8(w), vget_high_u8(w));
    f = vpadd_u8(f, f);
    f = vpadd_u8(f, f);
    return static_cast<uint32_t>(vget_lane_u16(vreinterpret_u16_u8(f), 0));
}

/// @brief NEON is baseline on aarch64, so there is nothing to dispatch on.
inline bool hasFastAvx2() { return true; }
/// @brief Pixels per vector iteration. NEON is 16 bytes wide.
constexpr size_t kFastLanes = 16;
#endif

} // namespace impl

/// @brief Detects FAST corners. **API TIER 2** -- see the tier note.
///
/// @param arcLength Contiguous ring pixels required. 9 is `cv::FAST`'s default and
///        the one ORB uses.
/// @param out,capacity Caller-provided; **no allocation happens here**
///        ([CLAUDE.md](../../../CLAUDE.md)).
/// @param truncated Set when more corners were found than `capacity` held. **A
///        silently truncated detection looks like a sparse image**, which is the kind
///        of failure that gets diagnosed as a tuning problem for weeks.
/// @return How many corners were written.
///
/// @note Pixels within 3 of a border are never candidates: the ring would fall
///       outside, and there is no sensible border rule for "is this a corner" --
///       reflecting the image would invent structure that is not there.
template <typename SrcT>
inline size_t detectFast(const SrcT* img, size_t width, size_t height, size_t stride,
                         long long threshold, FastCorner* out, size_t capacity,
                         bool* truncated = nullptr, int arcLength = 9) {
    BINCV_ASSERT(arcLength >= 1 && arcLength <= 16,
                 "detectFast: arcLength must be within the ring");
    if (truncated != nullptr) *truncated = false;
    if (width < 7 || height < 7 || capacity == 0) return 0;
    BINCV_ASSERT(img != nullptr && out != nullptr, "detectFast: null argument");

    // The largest count of compass points {0, 4, 8, 12} that EVERY window of
    // `arcLength` consecutive ring positions is guaranteed to contain. Computed
    // rather than tabulated so an unusual `arcLength` cannot silently get a bound
    // that belongs to a different one.
    int minCompass = 4;
    for (int s = 0; s < 16; ++s) {
        int c = 0;
        for (int k = 0; k < arcLength; ++k) {
            const int idx = (s + k) & 15;
            if ((idx & 3) == 0) ++c;
        }
        if (c < minCompass) minCompass = c;
    }

    // THE RING OFFSETS AS FLAT INDICES, COMPUTED ONCE PER IMAGE.
    // Recomputing `dy * stride + dx` for sixteen neighbours of every pixel was most
    // of this function: 360 000 pixels x 16 address computations, all of them the
    // same sixteen numbers.
    long long ringOff[16];
    for (int k = 0; k < 16; ++k)
        ringOff[k] = static_cast<long long>(impl::kFastRingY[k]) * static_cast<long long>(stride) +
                     impl::kFastRingX[k];

    size_t n = 0;
    for (size_t y = 3; y + 3 < height; ++y) {
        const SrcT* row = img + y * stride;
        size_t x = 3;
#if defined(BINCV_FAST_RUNTIME_AVX2) || defined(BINCV_FAST_NEON)
        // The vector path answers "is this a corner" for 32 pixels at a time; the
        // SCORE is still scalar, and that is the right split -- about 1% of pixels are
        // corners on a real frame, so scoring is rare and rejection is everything.
        if constexpr (sizeof(SrcT) == 1) {
            if (impl::hasFastAvx2() && threshold > 0 && threshold < 256) {
                for (; x + impl::kFastLanes + 3 <= width; x += impl::kFastLanes) {
                    uint32_t m = impl::fastMask32(
                        reinterpret_cast<const uint8_t*>(row + x), static_cast<long long>(stride),
                        ringOff, static_cast<int>(threshold), arcLength, minCompass);
                    while (m != 0) {
                        const unsigned b = static_cast<unsigned>(__builtin_ctz(m));
                        m &= m - 1;
                        const size_t px = x + b;
                        const SrcT* p = row + px;
                        const long long c = static_cast<long long>(*p);
                        const long long hi = c + threshold, lo = c - threshold;
                        long long sc = 0, scLo = 0;
                        for (int j = 0; j < 16; ++j) {
                            const long long v = static_cast<long long>(p[ringOff[j]]);
                            if (v - hi > 0) sc += v - hi;
                            if (lo - v > 0) scLo += lo - v;
                        }
                        const long long best = sc > scLo ? sc : scLo;
                        if (n >= capacity) {
                            if (truncated != nullptr) *truncated = true;
                            return n;
                        }
                        out[n].x = static_cast<int>(px);
                        out[n].y = static_cast<int>(y);
                        out[n].score = best;
                        ++n;
                    }
                }
            }
        }
#endif
        for (; x + 3 < width; ++x) {
            const SrcT* p = row + x;
            const long long c = static_cast<long long>(*p);
            const long long hi = c + threshold;
            const long long lo = c - threshold;

            // FOUR LOADS BEFORE SIXTEEN. Most pixels are not corners and this is what
            // makes that cheap: the compass points alone decide it, and only a
            // survivor pays for the full ring. Reading all sixteen first -- which this
            // function did -- cost 16x the loads on every rejected pixel and made
            // binCV 16x SLOWER than cv::FAST (measured, before this).
            const long long c0 = static_cast<long long>(p[ringOff[0]]);
            const long long c4 = static_cast<long long>(p[ringOff[4]]);
            const long long c8 = static_cast<long long>(p[ringOff[8]]);
            const long long c12 = static_cast<long long>(p[ringOff[12]]);
            const int haveHi = (c0 > hi) + (c4 > hi) + (c8 > hi) + (c12 > hi);
            const int haveLo = (c0 < lo) + (c4 < lo) + (c8 < lo) + (c12 < lo);
            // THE BOUND MUST BE THE WORST CASE, NOT THE TYPICAL ONE: an arc of
            // `arcLength` contains at least `minCompass` compass points, and anything
            // stricter discards real corners. At arcLength 9 that is TWO, not three --
            // the window 1..9 holds only indices 4 and 8.
            if (haveHi < minCompass && haveLo < minCompass) continue;

            long long ring[16];
            uint32_t maskHi = 0, maskLo = 0;
            for (int k = 0; k < 16; ++k) {
                const long long v = static_cast<long long>(p[ringOff[k]]);
                ring[k] = v;
                if (v > hi) maskHi |= (1u << k);
                if (v < lo) maskLo |= (1u << k);
            }

            // CONTIGUITY AS A BIT TRICK, NOT A SCAN. Doubling the 16-bit mask makes
            // the wrap explicit, and `x &= x >> 1` repeated `arcLength - 1` times
            // leaves a bit set exactly where a run of that length STARTS. Eight
            // shift-ands replace a 24-iteration loop with a data-dependent branch in
            // it -- and the loop was most of what made this function 11x slower than
            // cv::FAST on a corner-dense image.
            const auto hasRun = [arcLength](uint32_t m) {
                uint32_t acc = m | (m << 16);
                for (int i = 1; i < arcLength; ++i) acc &= acc >> 1;
                return (acc & 0xFFFFu) != 0u;
            };

            long long bestScore = 0;
            const bool runHi = haveHi >= minCompass && hasRun(maskHi);
            const bool runLo = !runHi && haveLo >= minCompass && hasRun(maskLo);
            if (runHi || runLo) {
                for (int j = 0; j < 16; ++j) {
                    const long long d = runHi ? (ring[j] - hi) : (lo - ring[j]);
                    if (d > 0) bestScore += d;
                }
            }
            if (bestScore <= 0) continue;
            if (n >= capacity) {
                if (truncated != nullptr) *truncated = true;
                return n;
            }
            out[n].x = static_cast<int>(x);
            out[n].y = static_cast<int>(y);
            out[n].score = bestScore;
            ++n;
        }
    }
    return n;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
