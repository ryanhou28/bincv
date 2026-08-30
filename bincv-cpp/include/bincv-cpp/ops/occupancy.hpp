#pragma once

/// @file occupancy.hpp
/// @brief Spacing NEW detections against the points already being tracked (T5.20).
///        **API TIER 3** -- no `cv::` equivalent, so no OpenCV name is borrowed.
///
/// ---------------------------------------------------------------------------
/// WHY THIS FILE EXISTS
///
/// A VIO frontend detects to TOP UP. Tracks die, the live count sags, the detector
/// runs, and every corner it returns has to be rejected if it lands on a point the
/// tracker is already following -- otherwise two tracks chase one feature and the
/// estimator sees a correlation it is told is independent.
///
/// `ops/corner.hpp`'s spacing filter does NOT do this. It spaces the corners of one
/// detection against **each other**, which is `cv::goodFeaturesToTrack`'s job and all
/// of it; the previous frame's tracks are not among its inputs and cannot be. The
/// reference keeps the two apart the same way and supplies the second half
/// separately -- `MaskedFeatureDetector::applyMinDistance(corners, prevCorners,
/// maskRadius)`, called on every detect.
///
/// binCV had no second half, so `examples/vio_frontend.cpp` grew a private copy of
/// one. That is the evidence this file is missing rather than an argument that it is:
/// the library's own example could not do a normal frontend's normal thing with the
/// library's own operations.
///
/// ---------------------------------------------------------------------------
/// TWO ARMS, AND WHICH ONE TO CALL IS A MEASUREMENT (E-46 / X-93)
///
///  - `spaceCandidates` -- exhaustive. Every candidate against every live point.
///    `O(new x live)` distance tests, vectorised eight-wide on AVX2 and four-wide on
///    NEON. **Costs no memory at all.**
///
///  - `spaceCandidatesMasked` -- a 1-bit occupancy frame. Stamp the disc of `radius`
///    around each live point once, then each candidate is **one bit test**.
///    `O(live x radius^2 / WordBits + new)`. **Costs one 1-bit frame** -- 38 400 B at
///    640x480 with `uint32_t` words.
///
/// **CALL `spaceCandidates`. THE BIT-PLANE ARM LOST, AND IT LOST BY AN ORDER OF
/// MAGNITUDE.** X-93 fixed the rule before either arm existed and then measured both at
/// the frontend's own operating point -- 640x480, radius 32, 120 live tracks, 300
/// candidates, 80 free slots:
///
/// | | exhaustive | mask |
/// |---|---|---|
/// | x86-64, AVX2 | **3 333 ns**, 0 B | 88 767 ns, 38 400 B |
/// | Cortex-A72, NEON | **49 617 ns**, 0 B | 380 629 ns, 38 400 B |
///
/// 26.6x and 7.7x, against a memory rule that would have needed the mask merely to
/// TIE. The mask does not overtake until roughly **2 000 candidates on aarch64 and
/// 5 000 on x86** -- seven to seventeen times more than a frontend ever detects.
///
/// **The reason is structural, not an unoptimised inner loop.** Stamping a disc touches
/// pi*r^2 = 3 217 pixels to encode what the exhaustive arm consumes in ONE distance
/// test per candidate -- and that test is vectorised eight-wide on AVX2 and four-wide
/// on NEON, while a disc stamp is scalar per row. The pre-registration predicted a
/// crossover near 100 candidates and was **wrong by a factor of about fifty**, because
/// it counted the disc's word-writes and neither the per-row bound arithmetic nor the
/// vector width on the other side.
///
/// The mask stays because the crossover is real and measured, not because it might be
/// useful: a caller filtering thousands of candidates -- masking a response map before
/// selection, rather than a detection's output after it -- is on the other side of it.
///
/// ---------------------------------------------------------------------------
/// THE TWO ARMS RETURN THE SAME POINTS, AND THAT IS A TEST
///
/// A candidate at an INTEGER position is rejected by the mask if and only if the float
/// distance test rejects it. That is not approximately true and it is not asserted
/// here: the disc's row bounds are found by **exact squared comparison** -- integer x
/// against `(x - cx)^2 < radius^2 - (y - cy)^2` in `double` -- rather than by rounding
/// a `sqrt`, so no boundary pixel can fall on different sides of the two tests.
/// `tests/test_occupancy.cpp` sweeps sub-pixel centres and radii against the float
/// oracle for exactly this.
///
/// Live points may sit anywhere; only the CANDIDATE has to be integral, which is what
/// a detector produces. A caller testing sub-pixel candidates against the mask is
/// asking about the pixel the candidate rounds to, and the exhaustive arm is the one
/// that answers about the point.

#include <cmath>
#include <cstddef>
#include <cstdint>

// binMat.hpp carries core/view.hpp and core/error.hpp and pulls in the word helpers
// (impl::wordIndex, impl::bitMask) at its end; impl/binMat_impl.hpp is not a header a
// consumer includes on its own.
// F-5: BEFORE THE GATE, NOT AFTER. This header defines BINCV_HAVE_NEON from the
// compiler's own macros on aarch64, so an include-only integration still gets the
// NEON kernels. Relying on transitive inclusion would not do -- this file evaluates
// its gate before its first core include.
#include "../core/simd.hpp"

#include "../binMat.hpp"
#include "../core/error.hpp"
#include "../core/types.hpp"
#include "../core/view.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_OCCUPANCY_AVX2 1
#include <immintrin.h>
#elif defined(BINCV_HAVE_NEON) && defined(__aarch64__)
#define BINCV_OCCUPANCY_NEON 1
#include <arm_neon.h>
#endif

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {

namespace impl {

/// @brief Force the portable distance loop, for the benchmark and the tests. **INTERNAL.**
/// @note Not a tuning knob. It is how the vector arm is held to giving the SAME answer
///       as the scalar one, and how a benchmark can show the vector arm is actually
///       running -- X-89 shipped a vector block that a mis-attached `#define` had
///       compiled out and measured three "improvements" against it.
inline bool& spacingSimdEnabled() {
    static bool on = true;
    return on;
}

#if defined(BINCV_OCCUPANCY_AVX2)
inline bool hasSpacingSimd() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes && spacingSimdEnabled();
}
#elif defined(BINCV_OCCUPANCY_NEON)
inline bool hasSpacingSimd() { return spacingSimdEnabled(); }
#else
inline bool hasSpacingSimd() { return false; }
#endif

/// @brief Is any of `pts[0, count)` strictly within `sqrt(r2)` of `(cx, cy)`?
///        **INTERNAL** -- the scalar reference arm.
/// @note Squared distances throughout: the comparison `d2 < r2` answers `d < r`
///       exactly for non-negative reals, and a `sqrt` per pair would be both slower
///       and a source of boundary disagreement between the arms.
inline bool anyWithinScalar(const Point2f* pts, size_t count, float cx, float cy, float r2) {
    for (size_t i = 0; i < count; ++i) {
        const float dx = pts[i].x - cx;
        const float dy = pts[i].y - cy;
        if (dx * dx + dy * dy < r2) return true;
    }
    return false;
}

#if defined(BINCV_OCCUPANCY_AVX2)
/// @brief Eight live points per register. **INTERNAL.**
///
/// `Point2f` is interleaved, so two loads give `{x0 y0 x1 y1 ...}` and a pair of
/// `shuffle_ps` splits them. The shuffle is WITHIN each 128-bit lane, so the eight x
/// values come out permuted -- which costs nothing, because the only thing asked of
/// them is "is any lane within the radius", and that reduction is order-blind.
__attribute__((target("avx2")))
inline bool anyWithinAvx2(const Point2f* pts, size_t count, float cx, float cy, float r2) {
    const __m256 vcx = _mm256_set1_ps(cx);
    const __m256 vcy = _mm256_set1_ps(cy);
    const __m256 vr2 = _mm256_set1_ps(r2);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const float* p = &pts[i].x;
        const __m256 a = _mm256_loadu_ps(p);       // x0 y0 x1 y1 | x2 y2 x3 y3
        const __m256 b = _mm256_loadu_ps(p + 8);   // x4 y4 x5 y5 | x6 y6 x7 y7
        const __m256 xs = _mm256_shuffle_ps(a, b, 0x88);   // (2,0,2,0)
        const __m256 ys = _mm256_shuffle_ps(a, b, 0xDD);   // (3,1,3,1)
        const __m256 dx = _mm256_sub_ps(xs, vcx);
        const __m256 dy = _mm256_sub_ps(ys, vcy);
        const __m256 d2 = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
        // _CMP_LT_OQ, matching the scalar `<` including its treatment of NaN as false.
        const __m256 lt = _mm256_cmp_ps(d2, vr2, _CMP_LT_OQ);
        if (_mm256_movemask_ps(lt) != 0) return true;
    }
    return anyWithinScalar(pts + i, count - i, cx, cy, r2);
}
#endif

#if defined(BINCV_OCCUPANCY_NEON)
/// @brief Four live points per register. **INTERNAL.**
/// @note `vld2q_f32` deinterleaves in the load, so NEON needs no shuffle at all --
///       the one place in this library where the aarch64 spelling is the shorter one.
inline bool anyWithinNeon(const Point2f* pts, size_t count, float cx, float cy, float r2) {
    const float32x4_t vcx = vdupq_n_f32(cx);
    const float32x4_t vcy = vdupq_n_f32(cy);
    const float32x4_t vr2 = vdupq_n_f32(r2);
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4x2_t v = vld2q_f32(&pts[i].x);
        const float32x4_t dx = vsubq_f32(v.val[0], vcx);
        const float32x4_t dy = vsubq_f32(v.val[1], vcy);
        const float32x4_t d2 = vfmaq_f32(vmulq_f32(dx, dx), dy, dy);
        // vmaxvq over the lane mask: any set lane leaves 0xFFFFFFFF as the maximum.
        if (vmaxvq_u32(vcltq_f32(d2, vr2)) != 0) return true;
    }
    return anyWithinScalar(pts + i, count - i, cx, cy, r2);
}
#endif

/// @brief Dispatch. **INTERNAL.**
inline bool anyWithin(const Point2f* pts, size_t count, float cx, float cy, float r2) {
#if defined(BINCV_OCCUPANCY_AVX2)
    if (hasSpacingSimd()) return anyWithinAvx2(pts, count, cx, cy, r2);
#elif defined(BINCV_OCCUPANCY_NEON)
    if (hasSpacingSimd()) return anyWithinNeon(pts, count, cx, cy, r2);
#endif
    return anyWithinScalar(pts, count, cx, cy, r2);
}

/// @brief Sets bits `[x0, x1]` INCLUSIVE of one packed row. **INTERNAL.**
/// @note The caller has already clamped both ends inside `[0, width)`, which is what
///       keeps padding bits zero (CLAUDE.md's hard rule) without a masking pass: a
///       disc never reaches past the last pixel of a row, so no word past `width` is
///       ever touched.
template <typename WordType>
inline void setBitRange(WordType* row, size_t x0, size_t x1) {
    constexpr size_t kBits = sizeof(WordType) * 8;
    const WordType all = static_cast<WordType>(~static_cast<WordType>(0));
    const size_t w0 = x0 / kBits;
    const size_t w1 = x1 / kBits;
    const WordType lo = static_cast<WordType>(all << (x0 % kBits));
    const size_t hb = x1 % kBits;
    const WordType hi = (hb + 1 == kBits)
                            ? all
                            : static_cast<WordType>((static_cast<WordType>(1) << (hb + 1)) -
                                                    static_cast<WordType>(1));
    if (w0 == w1) {
        row[w0] = static_cast<WordType>(row[w0] | static_cast<WordType>(lo & hi));
        return;
    }
    row[w0] = static_cast<WordType>(row[w0] | lo);
    for (size_t w = w0 + 1; w < w1; ++w) row[w] = all;
    row[w1] = static_cast<WordType>(row[w1] | hi);
}

}  // namespace impl

// ---------------------------------------------------------------------------
// ARM (a) -- EXHAUSTIVE
// ---------------------------------------------------------------------------

/// @brief Keeps the candidates that are at least `radius` from every live point and
///        from every candidate already kept. **API TIER 3.**
/// @param candidates In/out. Survivors are compacted to the front, **in the order they
///        arrived** -- so a caller that passes the detector's output untouched keeps
///        its strongest-first ranking, and the greedy filter accepts strong corners
///        before weak ones.
/// @param count Entries in `candidates`.
/// @param live The points already being tracked. May be `nullptr` when `liveCount` is 0.
/// @param liveCount Entries in `live`.
/// @param radius Minimum separation, in pixels. **Rejection is `distance < radius`,
///        strictly** -- the reference's comparison and `ops/corner.hpp`'s.
/// @param limit Stop after keeping this many. Pass the number of free track slots.
/// @return How many candidates were kept; they are `candidates[0, return)`.
///
/// @note **`radius < 1` disables the filter entirely** and keeps the first `limit`
///       candidates. That is `gftt.cpp`'s rule, reproduced because
///       `GoodFeaturesParams::minDistance` documents the same one and a caller
///       forwarding that field to this function must not get a different answer at
///       0.5 than `goodFeaturesToTrack` would.
/// @note Never throws, allocates nothing, and the compaction is in place -- the write
///       index never passes the read index.
/// @note The distance loop is vectorised where the ISA allows and gives the same
///       answer as the scalar arm bit for bit; `impl::spacingSimdEnabled()` forces the
///       scalar one, which is how the tests check that.
inline size_t spaceCandidates(Point2f* candidates, size_t count, const Point2f* live,
                              size_t liveCount, float radius, size_t limit) {
    BINCV_ASSERT(candidates != nullptr || count == 0,
                 "occupancy: a non-zero count needs a non-null candidate array");
    BINCV_ASSERT(live != nullptr || liveCount == 0,
                 "occupancy: a non-zero liveCount needs a non-null live array");

    if (radius < 1.0f) return (count < limit) ? count : limit;

    const float r2 = radius * radius;
    size_t kept = 0;
    for (size_t i = 0; i < count && kept < limit; ++i) {
        const Point2f c = candidates[i];
        if (impl::anyWithin(live, liveCount, c.x, c.y, r2)) continue;
        if (impl::anyWithin(candidates, kept, c.x, c.y, r2)) continue;
        candidates[kept++] = c;
    }
    return kept;
}

// ---------------------------------------------------------------------------
// ARM (b) -- THE 1-BIT OCCUPANCY MASK
// ---------------------------------------------------------------------------

/// @brief Zeroes an occupancy mask. **API TIER 3.**
/// @note Whole words including padding, so the frame is clean for `countRegion` and
///       friends as well as for `occupied`.
template <typename WordType>
inline void clearOccupancy(BinMatView<WordType> mask) {
    if (mask.empty()) return;
    const size_t words = impl::wordIndex<WordType>(mask.width - 1) + 1;
    for (size_t y = 0; y < mask.height; ++y) {
        WordType* row = mask.row(y);
        for (size_t w = 0; w < words; ++w) row[w] = static_cast<WordType>(0);
    }
}

/// @brief Sets every pixel strictly within `radius` of `(cx, cy)`. **API TIER 3.**
///
/// @note **NO `sqrt` IS TAKEN, and that is not micro-optimisation** -- it is what makes
///       the mask agree with the float distance test exactly. Rows are walked outward
///       from the centre, where the half-width only ever shrinks, so each row's bound
///       is reached by decrementing the previous one under the exact test
///       `(x - cx)^2 < radius^2 - (y - cy)^2`. The total decrementing over a whole disc
///       is `O(radius)`, not `O(radius)` per row, so the bounds are free next to the
///       writing and no rounded square root can put a boundary pixel on the wrong side.
/// @note The disc is CLIPPED to the mask, not wrapped, and never writes a word past
///       `width` -- so padding bits stay zero with no masking pass.
template <typename WordType>
inline void markDisc(BinMatView<WordType> mask, float cx, float cy, float radius) {
    if (mask.empty() || radius <= 0.0f) return;

    // FLOAT, AND IN THE SAME ORDER `spaceCandidates` USES IT. The predicate below is
    // `dx*dx + dy*dy < r2` with `dy*dy` hoisted out of the column loop -- which is
    // exactly `impl::anyWithinScalar`'s expression, operation for operation. Computing
    // it in `double` instead would be MORE accurate and therefore WRONG for this job:
    // the two arms have to round the same way, or a pixel at the boundary lands on
    // different sides of the same nominal comparison and the arms stop agreeing.
    const float r2 = radius * radius;
    const long long height = static_cast<long long>(mask.height);
    const long long width = static_cast<long long>(mask.width);

    const auto fits = [cx, r2](long long x, float dy2) {
        const float dx = static_cast<float>(x) - cx;
        return dx * dx + dy2 < r2;
    };

    const auto stampRow = [&](long long y, long long xlo, long long xhi) {
        if (y < 0 || y >= height) return;
        const long long a = xlo < 0 ? 0 : xlo;
        const long long b = xhi >= width ? width - 1 : xhi;
        if (a > b) return;
        impl::setBitRange<WordType>(mask.row(static_cast<size_t>(y)), static_cast<size_t>(a),
                                    static_cast<size_t>(b));
    };

    // A sweep from the widest row outward. `xlo` rises and `xhi` falls monotonically as
    // `dy2` grows, so both are carried rather than recomputed: the total narrowing over
    // a whole disc is O(radius), not O(radius) per row, and no square root is taken --
    // which is the second half of why the two arms agree exactly.
    const auto sweep = [&](long long yStart, long long step) {
        bool seeded = false;
        long long xlo = 0, xhi = -1;
        for (long long y = yStart;; y += step) {
            if (step > 0 && y >= height) return;
            if (step < 0 && y < 0) return;
            const float dy = static_cast<float>(y) - cy;
            const float dy2 = dy * dy;
            if (!(dy2 < r2)) return;              // past the disc; every later row too
            if (!seeded) {
                // The widest row of this sweep, walked out from the centre column.
                // O(radius) once; every later row only narrows what this found.
                // FLOOR, not a truncating cast: for a centre left of the frame
                // `(long long)(-5.3)` is -5 and the column containing cx is -6. The
                // seed has to be the containing column, because the two probes below
                // are `c` and `c+1` -- the only two integers that can be nearest to cx,
                // and one of them is in the interval whenever any integer is.
                const long long c = static_cast<long long>(std::floor(cx));
                xlo = c + 1;
                xhi = c;
                while (fits(xlo - 1, dy2)) --xlo;
                while (fits(xhi + 1, dy2)) ++xhi;
                seeded = true;
            } else {
                while (xlo <= xhi && !fits(xlo, dy2)) ++xlo;
                while (xlo <= xhi && !fits(xhi, dy2)) --xhi;
            }
            if (xlo > xhi) return;                // narrower still further out
            stampRow(y, xlo, xhi);
        }
    };

    // ROUND, NOT FLOOR, AND THE INVARIANT DEPENDS ON IT. Both sweeps require |dy| to
    // GROW with every step, or a later row could be wider than the one before it and
    // the carried bounds would be too narrow. Seeding at the NEAREST row makes
    // |yMid - cy| <= 0.5, so the first step in either direction lands at |dy| >= 0.5
    // and every step after that adds a whole pixel. Seeding at `floor` does not: at
    // cy = 5.9 the rows are dy = -0.9 then +0.1, and the second row is WIDER.
    const long long yMid = static_cast<long long>(std::lround(cy));
    sweep(yMid, 1);           // yMid and downward
    sweep(yMid - 1, -1);      // upward
}

/// @brief Stamps `markDisc` for every point. **API TIER 3.**
template <typename WordType>
inline void markOccupied(BinMatView<WordType> mask, const Point2f* pts, size_t count,
                         float radius) {
    BINCV_ASSERT(pts != nullptr || count == 0,
                 "occupancy: a non-zero count needs a non-null point array");
    for (size_t i = 0; i < count; ++i) markDisc<WordType>(mask, pts[i].x, pts[i].y, radius);
}

/// @brief Is the pixel `(x, y)` claimed? **API TIER 3.**
/// @note Out of bounds is NOT occupied. A candidate outside the frame is the caller's
///       to reject, and answering `true` would silently do it for them.
template <typename WordType>
inline bool occupied(BinMatConstView<WordType> mask, long long x, long long y) {
    if (x < 0 || y < 0) return false;
    if (static_cast<size_t>(x) >= mask.width || static_cast<size_t>(y) >= mask.height) return false;
    const WordType* row = mask.row(static_cast<size_t>(y));
    const size_t ux = static_cast<size_t>(x);
    return (row[impl::wordIndex<WordType>(ux)] & impl::bitMask<WordType>(ux)) != 0;
}

/// @brief `spaceCandidates` through an occupancy mask: test one bit, and stamp the
///        disc of every candidate kept. **API TIER 3.**
/// @param mask A mask the live points have already been stamped into --
///        `clearOccupancy` then `markOccupied`. **It is modified**: each accepted
///        candidate is stamped, which is what spaces the candidates against each other.
/// @param candidates In/out, compacted to the front in arrival order, as arm (a).
/// @param radius Must be the radius the live points were stamped with.
/// @param limit Stop after keeping this many.
/// @return How many were kept.
///
/// @note **Identical output to `spaceCandidates` for integer candidates**, pinned by
///       `Occupancy.ArmsAgreeExactly`. A candidate is rounded to the pixel it sits in;
///       for the integer positions a detector produces that is the point itself.
/// @note **THIS IS NOT THE ARM TO CALL FOR A VIO TOP-UP.** At the frontend's operating
///       point `spaceCandidates` is 26.6x faster on x86 and 7.7x on the reference
///       device, and costs no memory against this arm's 38 400 B. The crossover is
///       around 2 000 candidates on aarch64 and 5 000 on x86 (X-93), so this arm is
///       for a caller filtering THOUSANDS -- which a detection top-up is not.
///       @see EXPERIMENTS.md X-93 for the sweep and the memory cost.
template <typename WordType>
inline size_t spaceCandidatesMasked(BinMatView<WordType> mask, Point2f* candidates,
                                    size_t count, float radius, size_t limit) {
    BINCV_ASSERT(candidates != nullptr || count == 0,
                 "occupancy: a non-zero count needs a non-null candidate array");

    if (radius < 1.0f) return (count < limit) ? count : limit;

    BinMatConstView<WordType> ro{mask.ptr, mask.width, mask.height, mask.stride};
    size_t kept = 0;
    for (size_t i = 0; i < count && kept < limit; ++i) {
        const Point2f c = candidates[i];
        const long long x = static_cast<long long>(std::lround(c.x));
        const long long y = static_cast<long long>(std::lround(c.y));
        if (occupied<WordType>(ro, x, y)) continue;
        markDisc<WordType>(mask, c.x, c.y, radius);
        candidates[kept++] = c;
    }
    return kept;
}

}  // namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
