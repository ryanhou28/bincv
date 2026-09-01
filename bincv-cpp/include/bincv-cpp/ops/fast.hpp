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
// F-5: BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"


#include <cstddef>
#include <cstdint>

#include "../core/error.hpp"
#include "../core/view.hpp"
#include "../impl/kernel_util.hpp"

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
/// winding would accept different corners. This is `cv::FAST`'s order.
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
/// `subs_epu8(v, hi) != 0`. SSE/AVX byte compares are signed, and the bias trick
/// ops/pack.hpp uses would cost two extra ops per ring position here.
/// Saturation also gives the clamp for free -- `c + t` saturating at 255 means
/// "nothing is brighter", which is exactly right.
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
    // STAGE 0: TWO RING POSITIONS, NOT FOUR.
    // Compass points 0 and 8 are opposite, and ANY window of nine consecutive ring
    // positions contains at least one of them -- 1..9 holds 8, and 9..1 (wrapping)
    // holds 0. So a group where neither passes cannot contain a 9-arc, and two loads
    // settle it where four were being paid.
    //
    // This is where the time actually is. A threshold sweep showed binCV
    // WINNING 1.67-1.86x at high corner density -- the contiguity test is good -- and
    // LOSING 2.4x at zero density, where nothing runs but the reject. At a realistic
    // 1.1% the two cancelled to 1.09x. The reject path was the whole gap.
    {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + ringOff[0]));
        const __m256i v8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + ringOff[8]));
        mHi[0] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(v0, hi), zero), allOnes);
        mHi[8] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(v8, hi), zero), allOnes);
        mLo[0] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(lo, v0), zero), allOnes);
        mLo[8] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(lo, v8), zero), allOnes);
        const __m256i any = _mm256_or_si256(_mm256_or_si256(mHi[0], mHi[8]),
                                            _mm256_or_si256(mLo[0], mLo[8]));
        if (_mm256_movemask_epi8(any) == 0) return 0;
    }
    // FOUR COMPASS POSITIONS BEFORE THE OTHER TWELVE. About 1% of pixels on a real
    // frame are corners, so nearly every 32-pixel group can be dismissed from four
    // loads -- and dismissing it here skips twelve loads AND the whole run-length
    // loop, which is most of this function. The scalar path has had this reject since
    // it was written; the vector path was paying full price on every group.
    for (int c4 = 1; c4 < 4; c4 += 2) {   // 4 and 12; 0 and 8 came from stage 0
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
        // v > hi <=> subs_epu8(v, hi) != 0
        mHi[k] = _mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_subs_epu8(v, hi), zero), allOnes);
        // v < lo <=> subs_epu8(lo, v) != 0
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
    // STAGE 0: TWO RING POSITIONS, NOT FOUR.
    // Compass points 0 and 8 are opposite, and ANY window of nine consecutive ring
    // positions contains at least one of them -- 1..9 holds 8, and 9..1 (wrapping)
    // holds 0. So a group where neither passes cannot contain a 9-arc, and two loads
    // settle it where four were being paid.
    //
    // This is where the time actually is. A threshold sweep showed binCV
    // WINNING 1.67-1.86x at high corner density -- the contiguity test is good -- and
    // LOSING 2.4x at zero density, where nothing runs but the reject. At a realistic
    // 1.1% the two cancelled to 1.09x. The reject path was the whole gap.
    {
        const uint8x16_t v0 = vld1q_u8(p + ringOff[0]);
        const uint8x16_t v8 = vld1q_u8(p + ringOff[8]);
        mHi[0] = vcgtq_u8(v0, hi);
        mHi[8] = vcgtq_u8(v8, hi);
        mLo[0] = vcltq_u8(v0, lo);
        mLo[8] = vcltq_u8(v8, lo);
        const uint8x16_t any = vorrq_u8(vorrq_u8(mHi[0], mHi[8]), vorrq_u8(mLo[0], mLo[8]));
        if (vmaxvq_u8(any) == 0) return 0;
    }
    // FOUR COMPASS POSITIONS BEFORE THE OTHER TWELVE -- see the AVX2 path. On a real
    // frame about 1% of pixels are corners, so nearly every group is dismissed here
    // and skips twelve loads plus the whole run-length loop.
    for (int c4 = 1; c4 < 4; c4 += 2) {   // 4 and 12; 0 and 8 came from stage 0
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
/// the one ORB uses.
/// @param out,capacity Caller-provided; **no allocation happens here**
/// ([CLAUDE.md](../../../../CLAUDE.md)).
/// @param truncated Set when more corners were found than `capacity` held. **A
/// silently truncated detection looks like a sparse image**, which is the kind
/// of failure that gets diagnosed as a tuning problem for weeks.
/// @return How many corners were written.
///
/// @note Pixels within 3 of a border are never candidates: the ring would fall
/// outside, and there is no sensible border rule for "is this a corner" --
/// reflecting the image would invent structure that is not there.
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

// ===========================================================================
// earlier work: THE SAME DETECTOR, ON binCV'S OWN TYPE.
//
// Everything above this line takes `const SrcT*` and a byte stride -- a WIDE image.
// That is why it can only match `cv::FAST`: both sides load the same bytes into the
// same registers. **It is a property of the signature, not of FAST.**
//
// ON A BIT-PLANE FRAME THE DETECTOR COLLAPSES TO BOOLEAN ALGEBRA. Pixels are {0, 1},
// so `p_ring > p_centre + t` can only hold for `t = 0` with `ring = 1, centre = 0`,
// and `p_ring < p_centre - t` only for `ring = 0, centre = 1`. There is exactly ONE
// meaningful threshold, and the whole test becomes
//
// corner = arc9( ring & ~centre ) | arc9( ~ring & centre )
//
// -- sixteen AND-NOTs and an AND-tree, over WHOLE WORDS. **The same instruction that
// decided one pixel now decides thirty-two**, or two hundred and fifty-six in a vector
// register, and that is binCV's entire thesis applied to a detector.
// ===========================================================================

namespace impl {

/// @brief Bits `[w*B + dx, w*B + dx + B)` of a row. **INTERNAL**.
/// @note `dx` is a PIXEL offset in `[-(B-1), B-1]`; FAST needs `|dx| <= 3`. Words
/// outside the row read as zero, which is correct here because every pixel whose
/// ring reaches outside the image is a border pixel and is masked off anyway.
template <typename WordType>
inline WordType fastShiftedWord(const WordType* row, size_t words, size_t w, int dx) {
    constexpr size_t kBits = bitsPerWord<WordType>();
    if (dx == 0) return row[w];
    if (dx > 0) {
        const size_t s = static_cast<size_t>(dx);
        const WordType lo = static_cast<WordType>(row[w] >> s);
        const WordType hi = (w + 1 < words)
                                ? static_cast<WordType>(row[w + 1] << (kBits - s))
                                : static_cast<WordType>(0);
        return static_cast<WordType>(lo | hi);
    }
    const size_t s = static_cast<size_t>(-dx);
    const WordType lo = static_cast<WordType>(row[w] << s);
    const WordType hi = (w > 0) ? static_cast<WordType>(row[w - 1] >> (kBits - s))
                                : static_cast<WordType>(0);
    return static_cast<WordType>(lo | hi);
}

/// @brief `OR over all 16 starts of (AND of `arcLength` consecutive)`. **INTERNAL.**
///
/// **THE WHOLE TEST IS FOUR PASSES OVER SIXTEEN WORDS AND IT NEEDS NO SECOND ARRAY.**
/// `v[k] &= v[k + s]` turns "a run of `L` starts at `k`" into "a run of `L + s` starts
/// at `k`" for any `s <= L`, so doubling 1 -> 2 -> 4 -> 8 and finishing with `s = 1`
/// reaches nine. Written in place, saving only the `s` entries the wrap-around
/// consumes — which is what keeps the AVX2 form of this inside sixteen registers.
/// a measurement measured a four-array version and it ran at **0.7
/// operations per cycle**, three times worse than its own operation count.
///
/// The schedule is derived from `arcLength` rather than tabulated, so an unusual
/// `arcLength` is composed exactly rather than getting a rule that belongs to another.
template <typename WordType>
inline WordType fastArcAny(const WordType src[16], int arcLength) {
    WordType v[16];
    for (int k = 0; k < 16; ++k) v[k] = src[k];
    int have = 1;
    while (have < arcLength) {
        const int step = have < arcLength - have ? have : arcLength - have;
        WordType save[8];
        for (int i = 0; i < step; ++i) save[i] = v[i];
        for (int k = 0; k + step < 16; ++k) v[k] = static_cast<WordType>(v[k] & v[k + step]);
        for (int k = 16 - step; k < 16; ++k) {
            v[k] = static_cast<WordType>(v[k] & save[k + step - 16]);
        }
        have += step;
    }
    WordType any = v[0];
    for (int k = 1; k < 16; ++k) any = static_cast<WordType>(any | v[k]);
    return any;
}

/// @brief The longest run of set bits at ONE pixel, around the cyclic ring.
/// **INTERNAL** -- the Tier 2 score, computed only for detected corners.
inline int fastLongestRun(unsigned ring, int arcLength) {
    // The ring doubled end to end, so a run that wraps is a run in the middle. Each
    // `x &= x >> 1` shortens every run by one, so the number of passes before nothing
    // survives IS the longest run.
    //
    // **BRANCHLESS, AND THAT IS THE POINT.** The obvious `while (x) { x &= x >> 1; }`
    // is a data-dependent loop with an unpredictable exit, run once per corner --
    // and on a real binarized frame corners are 2% of pixels, which made SCORING a
    // third of this operation's time. The caller only asks about pixels that
    // already passed the arc test, so the first `arcLength - 1` passes are known to
    // survive and need no test at all; the remaining ones are counted, not branched on.
    unsigned x = (ring & 0xFFFFu) | ((ring & 0xFFFFu) << 16);
    for (int i = 1; i < arcLength; ++i) x &= x >> 1;
    int len = arcLength;
    for (int i = arcLength; i < 16; ++i) {
        x &= x >> 1;
        len += (x != 0) ? 1 : 0;
    }
    return len;
}

#if defined(BINCV_FAST_RUNTIME_AVX2)

#define BINCV_FASTBIT_FN __attribute__((target("avx2"), always_inline)) inline

/// @brief 256 consecutive pixels of a bit-plane row, starting `dx` pixels along.
/// **INTERNAL**.
///
/// **A BIT-PLANE ROW IS AN LSB-FIRST BIT STREAM AND THAT IS WHY THIS IS FIVE
/// INSTRUCTIONS.** `bitMask(x)` is `1 << (x % WordBits)` and the words are
/// little-endian, so pixel `i` is bit `i % 8` of byte `i / 8` — a flat stream with no
/// reordering anywhere in it. Displacing by `dx` pixels is therefore a byte offset and
/// a shift of at most seven.
///
/// AVX2 has no 256-bit-wide shift, only per-64-bit-lane ones, so the bits that would
/// cross a lane boundary are supplied by a **second load one byte along**: within a
/// lane, `a >> s` and `b << (8 - s)` name the same stream bit wherever both are
/// defined, and between them they cover the lane.
BINCV_FASTBIT_FN __m256i fastRing256(const uint8_t* p, int dx) {
    const int byteOff = dx >= 0 ? 0 : -1;
    const int s = dx - 8 * byteOff;
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + byteOff));
    if (s == 0) return a;
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + byteOff + 1));
    return _mm256_or_si256(_mm256_srli_epi64(a, s), _mm256_slli_epi64(b, 8 - s));
}

/// @brief `OR over all 16 starts of (AND of `arcLength` consecutive)`, for 256 pixels.
/// **INTERNAL**. `fastArcAny`'s schedule, in place, for the same reason.
/// @note **In place is not a tidiness choice, it is the whole difference.** The first
/// version of this held `r2`, `r4`, `r8` and an accumulator — sixty-four live
/// `__m256i` against a register file of sixteen — and ran at 0.7 operations per
/// cycle. Reusing one array of sixteen plus the wrap saves fits.
/// @brief One doubling step, with the step a COMPILE-TIME constant. **INTERNAL.**
/// @note **This is a template and not a parameter for a measured reason.** With a
/// runtime step every index into `v` is variable, so the compiler cannot unroll
/// and `v` has to live in memory — turning each `vpand` into load-load-and-store.
/// a measurement measured that costing **1.6×** on its own.
template <int Step>
BINCV_FASTBIT_FN void fastArcStep256(__m256i* v) {
    __m256i save[static_cast<size_t>(Step)];
    for (int i = 0; i < Step; ++i) save[i] = v[i];
    for (int k = 0; k + Step < 16; ++k) v[k] = _mm256_and_si256(v[k], v[k + Step]);
    for (int k = 16 - Step; k < 16; ++k) {
        v[k] = _mm256_and_si256(v[k], save[k + Step - 16]);
    }
}

BINCV_FASTBIT_FN __m256i fastArcAny256(__m256i* v, int arcLength) {
    // NINE IS `cv::FAST`'s DEFAULT AND ORB'S, so it gets the constant schedule
    // 1 -> 2 -> 4 -> 8, and then the LAST step is folded into the reduction: once
    // `v[k]` is a run of eight, `v[k] & v[k+1]` is a run of nine, and it is wanted only
    // to be ORed. Writing it back to `v` first is sixteen stores per chunk for nothing.
    if (arcLength == 9) {
        fastArcStep256<1>(v);
        fastArcStep256<2>(v);
        fastArcStep256<4>(v);
        // Two accumulators, so the sixteen ORs are two chains of eight and not one of
        // sixteen at the end of every chunk.
        __m256i a0 = _mm256_and_si256(v[0], v[1]);
        __m256i a1 = _mm256_and_si256(v[1], v[2]);
        for (int k = 2; k < 16; k += 2) {
            a0 = _mm256_or_si256(a0, _mm256_and_si256(v[k], v[(k + 1) & 15]));
            a1 = _mm256_or_si256(a1, _mm256_and_si256(v[k + 1], v[(k + 2) & 15]));
        }
        return _mm256_or_si256(a0, a1);
    }
    if (arcLength == 12) {
        fastArcStep256<1>(v);
        fastArcStep256<2>(v);
        fastArcStep256<4>(v);
        fastArcStep256<4>(v);
    } else {
        int have = 1;
        while (have < arcLength) {
            const int step = have < arcLength - have ? have : arcLength - have;
            __m256i save[8];
            for (int i = 0; i < step; ++i) save[i] = v[i];
            for (int k = 0; k + step < 16; ++k) v[k] = _mm256_and_si256(v[k], v[k + step]);
            for (int k = 16 - step; k < 16; ++k) {
                v[k] = _mm256_and_si256(v[k], save[k + step - 16]);
            }
            have += step;
        }
    }
    __m256i any = v[0];
    for (int k = 1; k < 16; ++k) any = _mm256_or_si256(any, v[k]);
    return any;
}

/// @brief The arc-length masks `L = 9..16`, for 256 pixels. **INTERNAL**.
///
/// **THE SCORE IS ALREADY HALF-COMPUTED BY THE DETECTION AND THIS COLLECTS IT.** After
/// the three doublings `v[k]` is `AND(diff[k.. k+7])` — a run of eight — and for any
/// `L` in 9..16,
///
/// AND(diff[k.. k+L-1]) == v[k] & v[(k + L - 8) & 15]
///
/// because two overlapping runs of eight cover any `L <= 16`. So every arc length costs
/// one pass of sixteen ANDs and an OR-reduce, and `L = 9` is the corner mask the
/// detector wanted anyway.
///
/// @param first,last Inclusive range of `L` to produce, written to `out[L - 9]`.
template <int L>
BINCV_FASTBIT_FN void fastArcMask256(const __m256i* v, uint32_t* out) {
    constexpr int kOff = L - 8;
    __m256i a0 = _mm256_and_si256(v[0], v[kOff & 15]);
    __m256i a1 = _mm256_and_si256(v[1], v[(1 + kOff) & 15]);
    for (int k = 2; k < 16; k += 2) {
        a0 = _mm256_or_si256(a0, _mm256_and_si256(v[k], v[(k + kOff) & 15]));
        a1 = _mm256_or_si256(a1, _mm256_and_si256(v[k + 1], v[(k + 1 + kOff) & 15]));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + (L - 9) * 8),
                        _mm256_or_si256(a0, a1));
}

/// @brief The seven masks `L = 10..16`. **INTERNAL**.
/// @note **Unrolled with `L` a template parameter, and a measurement measured why.** Written as a
/// loop over a runtime `L`, the index `v[(k + L - 8) & 15]` is variable — so `v`
/// cannot stay in registers and the whole array goes to memory. On x86 that is
/// nearly free because sixteen `__m256i` were spilling anyway; **on aarch64,
/// where they FIT in the thirty-two registers, it cost 2.36× → 1.98× on the
/// reference device.** The same mistake `fastArcStep256` was already fixed for.
BINCV_FASTBIT_FN void fastArcMasksRest256(const __m256i* v, uint32_t* out) {
    fastArcMask256<10>(v, out);
    fastArcMask256<11>(v, out);
    fastArcMask256<12>(v, out);
    fastArcMask256<13>(v, out);
    fastArcMask256<14>(v, out);
    fastArcMask256<15>(v, out);
    fastArcMask256<16>(v, out);
}

/// @brief One 256-pixel chunk: the corner mask, and the score inputs. **INTERNAL,
/// and the one function carrying `target`**.
///
/// **THIS IS WHERE THE THESIS PAYS.** The scalar word form of this loop does the same
/// sixteen XORs and the same AND-tree — and a `uint32_t` holds thirty-two pixels, which
/// is exactly what a vector register of BYTES holds. **So the bit packing buys nothing
/// until the boolean algebra itself moves into a vector register**, where one `vpand`
/// decides two hundred and fifty-six pixels instead of thirty-two.
///
/// @return True when `masks[1..7]` were filled, so the caller scores from them; false
/// when it should transpose each corner's ring instead.
///
/// **THE CHOICE IS PER CHUNK AND IT IS A MEASURED CROSSOVER, NOT A PREFERENCE.**
/// a measurement measured both scoring arms across corner densities:
/// the seven extra mask passes cost ~217 vector operations per chunk **whatever the
/// density**, and a per-corner transpose costs ~78 scalar operations **per corner** —
/// so the masks win above about three corners in a chunk and lose by up to **1.5×**
/// below it. A library that picked one would be 1.4× slow on half its inputs.
__attribute__((target("avx2"))) inline bool fastBitChunk256(const uint8_t* const* ringRow,
                                                            const uint8_t* centreRow,
                                                            size_t chunkByte, int arcLength,
                                                            int maskThreshold,
                                                            uint32_t* masks,
                                                            uint32_t* diffOut) {
    const __m256i centre =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(centreRow + chunkByte));
    __m256i v[16];
    for (int k = 0; k < 16; ++k) {
        v[k] = _mm256_xor_si256(
            centre, fastRing256(ringRow[kFastRingY[k] + 3] + chunkByte, kFastRingX[k]));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(diffOut + k * 8), v[k]);
    }
    if (arcLength != 9) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(masks),
                            fastArcAny256(v, arcLength));
        return false;
    }
    fastArcStep256<1>(v);
    fastArcStep256<2>(v);
    fastArcStep256<4>(v);
    fastArcMask256<9>(v, masks);
    int corners = 0;
    for (int i = 0; i < 8; ++i) corners += __builtin_popcount(masks[i]);
    if (corners < maskThreshold) return false;
    fastArcMasksRest256(v, masks);
    return true;
}

/// @brief Is AVX2 present? Asked once. (`hasFastAvx2` above serves the byte path.)
inline bool hasFastBitAvx2() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}

#endif  // BINCV_FAST_RUNTIME_AVX2

#if defined(BINCV_FAST_NEON)

#define BINCV_FASTBIT_NEON __attribute__((always_inline)) inline

/// @brief 128 consecutive pixels of a bit-plane row, `Dx` pixels along. **INTERNAL.**
/// @note The AVX2 identity unchanged: a bit-plane row is an LSB-first bit stream, so a
/// pixel displacement is a byte offset and a shift of at most seven, and the bits
/// that would cross a 64-bit lane are supplied by a second load one byte along.
/// @note `Dx` is a TEMPLATE parameter because `vshrq_n_u64` takes an immediate. The
/// ring is a compile-time constant, so nothing is lost by saying so.
template <int Dx>
BINCV_FASTBIT_NEON uint8x16_t fastRingNeon(const uint8_t* p) {
    constexpr int kByteOff = Dx >= 0 ? 0 : -1;
    constexpr int kShift = Dx - 8 * kByteOff;
    const uint8x16_t a = vld1q_u8(p + kByteOff);
    if constexpr (kShift == 0) {
        return a;
    } else {
        const uint8x16_t b = vld1q_u8(p + kByteOff + 1);
        return vorrq_u8(
            vreinterpretq_u8_u64(vshrq_n_u64(vreinterpretq_u64_u8(a), kShift)),
            vreinterpretq_u8_u64(vshlq_n_u64(vreinterpretq_u64_u8(b), 8 - kShift)));
    }
}

/// @brief One doubling step of the arc test, the step a compile-time constant.
template <int Step>
BINCV_FASTBIT_NEON void fastArcStepNeon(uint8x16_t* v) {
    uint8x16_t save[static_cast<size_t>(Step)];
    for (int i = 0; i < Step; ++i) save[i] = v[i];
    for (int k = 0; k + Step < 16; ++k) v[k] = vandq_u8(v[k], v[k + Step]);
    for (int k = 16 - Step; k < 16; ++k) v[k] = vandq_u8(v[k], save[k + Step - 16]);
}

BINCV_FASTBIT_NEON uint8x16_t fastArcAnyNeon(uint8x16_t* v, int arcLength) {
    // aarch64 has THIRTY-TWO vector registers, so the sixteen the tree needs fit with
    // room to spare -- the spilling that dominated the first AVX2 version does
    // not arise here at all.
    if (arcLength == 9) {
        fastArcStepNeon<1>(v);
        fastArcStepNeon<2>(v);
        fastArcStepNeon<4>(v);
        uint8x16_t a0 = vandq_u8(v[0], v[1]);
        uint8x16_t a1 = vandq_u8(v[1], v[2]);
        for (int k = 2; k < 16; k += 2) {
            a0 = vorrq_u8(a0, vandq_u8(v[k], v[(k + 1) & 15]));
            a1 = vorrq_u8(a1, vandq_u8(v[k + 1], v[(k + 2) & 15]));
        }
        return vorrq_u8(a0, a1);
    }
    int have = 1;
    while (have < arcLength) {
        const int step = have < arcLength - have ? have : arcLength - have;
        uint8x16_t save[8];
        for (int i = 0; i < step; ++i) save[i] = v[i];
        for (int k = 0; k + step < 16; ++k) v[k] = vandq_u8(v[k], v[k + step]);
        for (int k = 16 - step; k < 16; ++k) v[k] = vandq_u8(v[k], save[k + step - 16]);
        have += step;
    }
    uint8x16_t any = v[0];
    for (int k = 1; k < 16; ++k) any = vorrq_u8(any, v[k]);
    return any;
}

/// @brief The sixteen ring reads, unrolled so every displacement is an immediate.
template <int K>
BINCV_FASTBIT_NEON void fastRingLoadNeon(const uint8_t* const* ringRow, size_t chunkByte,
                                         uint8x16_t centre, uint8x16_t* v, uint8_t* diffOut) {
    if constexpr (K < 16) {
        v[K] = veorq_u8(centre, fastRingNeon<kFastRingX[K]>(
                                    ringRow[kFastRingY[K] + 3] + chunkByte));
        vst1q_u8(diffOut + K * 16, v[K]);
        fastRingLoadNeon<K + 1>(ringRow, chunkByte, centre, v, diffOut);
    } else {
        (void)ringRow; (void)chunkByte; (void)centre; (void)v; (void)diffOut;
    }
}

/// @brief Corner mask for 128 consecutive pixels. **INTERNAL**.
///
/// **NO ADAPTIVE SCORING HERE, AND MEASURED WHY.** The
/// AVX2 path chooses per chunk between transposing each corner's ring and reading the
/// score off eight nested arc-length masks, and that choice is worth **1.39× → 1.71×**
/// there. Ported to NEON it made the reference device **SLOWER — 2.36× → 2.10×** — and
/// the sweep showed the loss even at a threshold that never takes the mask path, so it
/// is the RESTRUCTURING and not the masks.
///
/// The reason is that the mask form must keep all sixteen `v` vectors live across up to
/// eight passes, where this fold **consumes them in place** and they are dead after.
/// x86 was spilling those sixteen anyway; aarch64's thirty-two registers were holding
/// them, and that is exactly what that measurement’s 2.36× was made of.
///
/// **A cross-architecture win is not a win. Both were measured, and the two backends
/// keep different code because the measurement said to.**
/// @note NEON is baseline on aarch64, so unlike the AVX2 form there is nothing to
/// dispatch on.
inline void fastBitMask128(const uint8_t* const* ringRow, const uint8_t* centreRow,
                           size_t chunkByte, int arcLength, uint8_t* out, uint8_t* diffOut) {
    const uint8x16_t centre = vld1q_u8(centreRow + chunkByte);
    uint8x16_t v[16];
    fastRingLoadNeon<0>(ringRow, chunkByte, centre, v, diffOut);
    vst1q_u8(out, fastArcAnyNeon(v, arcLength));
}

#endif  // BINCV_FAST_NEON

/// @brief Corners in a chunk above which scoring switches to the ARC-LENGTH MASKS.
/// **INTERNAL**.
///
/// **A MEASURED CROSSOVER, AND THE REASON THERE IS A KNOB AT ALL.** The two ways to
/// score cost differently in the density: the seven extra mask passes are ~217 vector
/// operations per chunk **whatever the density**, and transposing a corner's ring is
/// ~78 scalar operations **per corner**. a measurement measured both
/// across a density sweep and they cross at about three corners per chunk — below it
/// the masks lose by up to 1.5×, above it they win by up to 1.42×.
///
/// Three is where the arithmetic says, and the sweep agrees. **Settable so the two arms
/// can be forced and compared on identical input**: `0` always uses the masks,
/// a large value never does. A shipped build leaves it alone.
inline int& fastScoreMaskThreshold() {
    static int threshold = 3;
    return threshold;
}

}  // namespace impl

/// @brief Detects FAST corners on a **bit-plane frame**. **API TIER 2** — see below.
///
/// **THE DETECTION IS `cv::FAST`'s, EXACTLY, ON THE SAME CONTENT.** For a binary image
/// stored as `CV_8U` in `{0, 255}`, `cv::FAST` at any threshold in `[1, 254]` accepts
/// precisely the corners this accepts: `255 > 0 + t` for every such `t`, and
/// `0 < 255 - t` likewise, so the brighter arc requires a zero centre and the darker
/// arc a set one. That equivalence is what `tests/test_fast.cpp` checks, corner for
/// corner and in order.
///
/// **THE SCORE IS NOT `cv::FAST`'s, WHICH IS WHY THIS IS TIER 2** — the same reason the
/// wide-image overload is. On binary content OpenCV's "largest surviving threshold"
/// score is the SAME NUMBER for every corner and orders nothing, so this reports the
/// **longest qualifying arc**, 9 to 16, which at least distinguishes a sharp corner
/// from a marginal one. A caller doing non-maximum suppression over these will keep
/// different points than one doing it over OpenCV's.
///
/// @param img The frame, one bit per pixel.
/// @param out,capacity Caller-provided; **no allocation happens here**.
/// @param truncated Set when more corners were found than `capacity` held.
/// @param arcLength Contiguous ring pixels required; 9 is `cv::FAST`'s default.
///
/// @note There is **no threshold parameter and that is not an omission**. A one-bit
/// alphabet admits exactly one, and offering a knob with one legal value would
/// be worse than not offering it.
/// @note Pixels within 3 of a border are never candidates, matching the wide-image
/// overload and `cv::FAST`.
template <typename WordType>
inline size_t detectFast(const BinMatConstView<WordType>& img, FastCorner* out,
                         size_t capacity, bool* truncated = nullptr, int arcLength = 9) {
    BINCV_ASSERT(arcLength >= 1 && arcLength <= 16,
                 "detectFast: arcLength must be within the ring");
    if (truncated != nullptr) *truncated = false;
    if (capacity == 0) return 0;
    BINCV_ASSERT(out != nullptr, "detectFast: null argument");
    const size_t width = img.width, height = img.height;
    if (width < 7 || height < 7) return 0;

    constexpr size_t kBits = impl::bitsPerWord<WordType>();
    const size_t words = impl::minRowWords<WordType>(width);
    size_t n = 0;

    // The sixteen `ring XOR centre` words for ONE output word. Named once because the
    // scalar path, the vector path's scoring and the border fallback all need it.
    const auto wordDiff = [&](const WordType* const* ringRow, WordType centre, size_t w,
                              WordType* diff) {
        for (int k = 0; k < 16; ++k) {
            const WordType r = impl::fastShiftedWord<WordType>(
                ringRow[impl::kFastRingY[k] + 3], words, w, impl::kFastRingX[k]);
            diff[k] = static_cast<WordType>(r ^ centre);
        }
    };

    bool overflow = false;
    const auto emit = [&](const WordType* diff, size_t x0, size_t y, WordType mask) {
        // The border columns, removed from the WORD rather than tested per pixel.
        for (size_t b = 0; b < kBits; ++b) {
            const size_t x = x0 + b;
            if (x >= 3 && x + 3 < width) continue;
            mask = static_cast<WordType>(mask & ~(static_cast<WordType>(1) << b));
        }
        while (mask != 0) {
            const size_t b =
                static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(mask)));
            mask = static_cast<WordType>(mask & (mask - 1));
            if (n >= capacity) {
                overflow = true;
                return;
            }
            unsigned ring = 0;
            for (int k = 0; k < 16; ++k) {
                ring |= static_cast<unsigned>((diff[k] >> b) & 1u) << k;
            }
            out[n].x = static_cast<int>(x0 + b);
            out[n].y = static_cast<int>(y);
            out[n].score = impl::fastLongestRun(ring, arcLength);
            ++n;
        }
    };

    // that measurement’s arm B: the score read straight off the nested arc-length masks. A pixel's
    // score is `8 + the number of masks holding its bit`, because a run of L implies a
    // run of every shorter length -- so the count IS the maximum, with no transpose and
    // no run loop.
    const auto emitScored = [&](const WordType* lens, size_t x0, size_t y, WordType mask) {
        for (size_t b = 0; b < kBits; ++b) {
            const size_t x = x0 + b;
            if (x >= 3 && x + 3 < width) continue;
            mask = static_cast<WordType>(mask & ~(static_cast<WordType>(1) << b));
        }
        while (mask != 0) {
            const size_t b =
                static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(mask)));
            mask = static_cast<WordType>(mask & (mask - 1));
            if (n >= capacity) {
                overflow = true;
                return;
            }
            int score = 8;
            for (int L = 0; L < 8; ++L) score += static_cast<int>((lens[L] >> b) & 1u);
            out[n].x = static_cast<int>(x0 + b);
            out[n].y = static_cast<int>(y);
            out[n].score = score;
            ++n;
        }
    };

    for (size_t y = 3; y + 3 < height && !overflow; ++y) {
        const WordType* centreRow = img.row(y);
        const WordType* ringRow[7];
        for (int d = -3; d <= 3; ++d) {
            ringRow[d + 3] = img.row(static_cast<size_t>(static_cast<long long>(y) + d));
        }
        size_t w = 0;
#if defined(BINCV_FAST_RUNTIME_AVX2) || defined(BINCV_FAST_NEON)
        // WHY THE FIRST AND LAST ROWS OF THE SWEEP DECLINE THE VECTOR PATH: a bit-plane
        // row's stride carries NO padding (752 px is exactly 96 bytes), and the ring
        // read touches one byte either side of its window. On `y == 3` the top ring row
        // is row 0 and the read before it is outside the allocation; on the last row the
        // read after it is. Two rows of ~474 take the scalar path, which is correct and
        // costs nothing measurable -- declining is cheaper than proving the padding.
        //
        // THE CHUNK IS THE VECTOR WIDTH IN PIXELS: 256 on AVX2, 128 on NEON. That
        // number IS the result -- a `uint32_t` holds 32 pixels, which is exactly what a
        // vector register of BYTES holds, so the bit packing buys nothing until the
        // boolean algebra moves into a vector register.
#if defined(BINCV_FAST_RUNTIME_AVX2)
        constexpr size_t kChunkBytes = 32;
        const bool vectorReady = impl::hasFastBitAvx2();
#else
        constexpr size_t kChunkBytes = 16;
        const bool vectorReady = true;
#endif
        constexpr size_t kChunkWords = kChunkBytes / sizeof(WordType);
        if (vectorReady && kChunkWords >= 1 && y >= 4 && y + 4 < height) {
            const uint8_t* ringBytes[7];
            for (int d = 0; d < 7; ++d) {
                ringBytes[d] = reinterpret_cast<const uint8_t*>(ringRow[d]);
            }
            const uint8_t* centreBytes = reinterpret_cast<const uint8_t*>(centreRow);
            // EIGHT masks, not one: `masks[0]` is the corner mask (a run of nine) and
            // `masks[1..7]` are runs of ten to sixteen, from which a score is a
            // population count rather than a bit transpose.
            alignas(32) uint8_t maskBuf[8 * kChunkBytes];
            alignas(32) uint8_t diffBuf[16 * kChunkBytes];
            for (; w + kChunkWords <= words; w += kChunkWords) {
#if defined(BINCV_FAST_RUNTIME_AVX2)
                const bool scored = impl::fastBitChunk256(
                    ringBytes, centreBytes, w * sizeof(WordType), arcLength,
                    impl::fastScoreMaskThreshold(),
                    reinterpret_cast<uint32_t*>(maskBuf),
                    reinterpret_cast<uint32_t*>(diffBuf));
#else
                impl::fastBitMask128(ringBytes, centreBytes, w * sizeof(WordType), arcLength,
                                     maskBuf, diffBuf);
                constexpr bool scored = false;   // measured a LOSS on NEON
#endif
                const WordType* chunk = reinterpret_cast<const WordType*>(maskBuf);
                for (size_t j = 0; j < kChunkWords && !overflow; ++j) {
                    if (chunk[j] == 0) continue;
                    if (scored) {
                        WordType lens[8];
                        for (int L = 0; L < 8; ++L) {
                            lens[L] = reinterpret_cast<const WordType*>(
                                maskBuf + static_cast<size_t>(L) * kChunkBytes)[j];
                        }
                        emitScored(lens, (w + j) * kBits, y, chunk[j]);
                        continue;
                    }
                    WordType diff[16];
                    for (int k = 0; k < 16; ++k) {
                        diff[k] = reinterpret_cast<const WordType*>(
                            diffBuf + static_cast<size_t>(k) * kChunkBytes)[j];
                    }
                    emit(diff, (w + j) * kBits, y, chunk[j]);
                }
                if (overflow) break;
            }
        }
#endif
        for (; w < words && !overflow; ++w) {
            WordType diff[16];
            wordDiff(ringRow, centreRow[w], w, diff);
            const WordType mask = impl::fastArcAny<WordType>(diff, arcLength);
            if (mask == 0) continue;
            emit(diff, w * kBits, y, mask);
        }
    }
    if (overflow && truncated != nullptr) *truncated = true;
    return n;
}

} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
