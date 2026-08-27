#pragma once

#include <cstddef>
#include <cstdint>

// X-79 / E-36: the keypoint batch is selected at RUN TIME, so the library's baseline
// ISA is unchanged and no `-mavx2` build is required of a consumer. Guarded on the
// compiler supporting both the target attribute and the cpu probe, exactly as
// impl/binMat_impl.hpp's row packer is.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define BINCV_X86_LK_BATCH 1
#include <immintrin.h>
#endif

#include "../core/error.hpp"   // BINCV_ABI_NAMESPACE
#include "../ops/reduce.hpp"       // popcountWord, for the scalar oracle only

/// @file
/// @brief EIGHT KEYPOINTS' RESIDUAL SUMS IN AVX2 LANES. **INTERNAL** (E-36, X-79).
///
/// **WHY THIS IS A SEPARATE FILE AND NOT PART OF ops/opticalFlow.hpp.** Everything
/// here is arithmetic over `uint32_t` arrays with no notion of a level, a window or a
/// keypoint — which makes it testable against a scalar oracle on synthetic input, and
/// keeps the tracker's geometry (staging, refill, clipping, convergence) out of a
/// translation unit full of intrinsics.
///
/// **THE LAYOUT IS `[row][plane][lane]` AND THAT IS THE WHOLE TRICK.** A window row is
/// one `uint32_t` per plane per keypoint, so eight keypoints' words at the same row and
/// plane are eight adjacent `uint32_t` — one `__m256i` load, no gather.
/// [X-61](../../../EXPERIMENTS.md) lost precisely because it gathered; the fix was
/// never a better gather, it was arranging not to need one.
///
/// **AND THE MEASURED REASON THE ENTRY POINT IS COARSE.** `target("avx2")` on a leaf
/// helper blocks inlining — GCC and Clang refuse to inline a callee whose target
/// features are not a subset of the caller's — and X-60 measured that costing **1.9×**
/// by turning `slicedSignedSum` into 310 real calls per window. One function carries
/// the attribute and covers the whole window; the helpers it calls carry the same
/// attribute plus `always_inline`, so they fold into it.

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace impl {

/// Lanes per batch: eight `uint32_t` in a `__m256i`, and the reason the window row is
/// one word (a 31-pixel window at `uint32_t`).
constexpr size_t kLkBatchLanes = 8;

/// Rows a batch can stage, matching `kStagedMaxRows` so a window the scalar path can
/// stage is one the batch can hold.
constexpr size_t kLkBatchMaxRows = 64;

#if defined(BINCV_X86_LK_BATCH)

#define BINCV_LKB_FN __attribute__((target("avx2"), always_inline)) inline

/// @brief Per-BYTE popcount of a 256-bit vector. **INTERNAL.**
/// @note AVX2 has no popcount instruction at all — `VPOPCNTDQ` is AVX-512. The nibble
///       table through `vpshufb` is the standard substitute and costs six operations
///       for thirty-two bytes, against `POPCNT`'s one instruction for eight. That
///       looks like a loss and is not: this covers **eight keypoints at once**, so it
///       is six operations where the scalar path issues eight.
/// @note The counts stay in BYTES on purpose. Folding them to per-lane integers costs
///       two more operations, and the caller can weight and fold FOUR plane pairs
///       first — which is why the reduction happens once per (row, value, component)
///       rather than once per popcount.
BINCV_LKB_FN __m256i lkbByteCounts(__m256i v, __m256i lut, __m256i lowMask) {
    const __m256i lo = _mm256_and_si256(v, lowMask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

/// @brief Byte counts within each 32-bit lane, summed to that lane. **INTERNAL.**
/// @note `maddubs` folds adjacent byte pairs into 16-bit, `madd` folds those pairs
///       into 32-bit — and a 32-bit lane's four bytes are exactly two `maddubs` pairs,
///       so two instructions place each lane's total in its own lane and nowhere else.
BINCV_LKB_FN __m256i lkbFoldToLanes(__m256i bytes, __m256i ones8, __m256i ones16) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(bytes, ones8), ones16);
}

/// @brief `sum over the window of V(z) * G(z)` for EIGHT keypoints. **INTERNAL.**
///
/// The vector form of `slicedSignedSum` accumulated over a whole window, which is
/// what makes it worth vectorising: `slicedSignedSum` on its own is `2N^2` popcounts
/// and nothing to hide the reduction behind.
///
/// @param val  `[row][plane][lane]`, the value's planes — a tap or the previous frame.
/// @param magP `[row][plane][lane]`, gradient magnitude where the gradient is
///        POSITIVE, already masked to each lane's region.
/// @param magN the same where it is negative. **Splitting the sign out of the inner
///        loop is why nothing here has to go negative in the byte domain**: both
///        halves are counted as unsigned bytes and subtracted only once they are
///        32-bit lane sums.
/// @param rows Rows in the batch — the tallest lane's. Shorter lanes are padded with
///        ZERO magnitude, which contributes exactly zero and needs no masking.
/// @return One `int32` per lane: `sum_{i,j} 2^(i+j) (popcount(V_i & P_j) -
///         popcount(V_i & N_j))`.
///
/// @note **The byte accumulator would overflow and does not.** A weighted byte is at
///       most `8 + 2*16 + 4*8 = 72` at `N = 2`, so the four plane pairs are folded in
///       the byte domain and only THEN widened — one widening per row rather than
///       four. Accumulating the bytes ACROSS rows would overflow at 255 and is not
///       attempted.
template <size_t N>
BINCV_LKB_FN __m256i lkbWindowSum(const uint32_t* val, const uint32_t* magP,
                                  const uint32_t* magN, size_t rows) {
    static_assert(N == 1 || N == 2, "lkbWindowSum: only the shipped ladder's depths");
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lowMask = _mm256_set1_epi8(0x0F);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();

    for (size_t i = 0; i < rows; ++i) {
        const uint32_t* v = val + i * N * kLkBatchLanes;
        const uint32_t* p = magP + i * N * kLkBatchLanes;
        const uint32_t* n = magN + i * N * kLkBatchLanes;
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v));
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i n0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n));

        __m256i wP = lkbByteCounts(_mm256_and_si256(v0, p0), lut, lowMask);
        __m256i wN = lkbByteCounts(_mm256_and_si256(v0, n0), lut, lowMask);

        if constexpr (N == 2) {
            const __m256i v1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + kLkBatchLanes));
            const __m256i p1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + kLkBatchLanes));
            const __m256i n1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n + kLkBatchLanes));
            // (0,1) and (1,0) share the weight 2, so they share one doubling.
            const __m256i midP =
                _mm256_add_epi8(lkbByteCounts(_mm256_and_si256(v0, p1), lut, lowMask),
                                lkbByteCounts(_mm256_and_si256(v1, p0), lut, lowMask));
            const __m256i midN =
                _mm256_add_epi8(lkbByteCounts(_mm256_and_si256(v0, n1), lut, lowMask),
                                lkbByteCounts(_mm256_and_si256(v1, n0), lut, lowMask));
            const __m256i hiP = lkbByteCounts(_mm256_and_si256(v1, p1), lut, lowMask);
            const __m256i hiN = lkbByteCounts(_mm256_and_si256(v1, n1), lut, lowMask);
            // x2 and x4 as byte adds; there is no byte multiply, and none is needed.
            const __m256i hiP2 = _mm256_add_epi8(hiP, hiP);
            const __m256i hiN2 = _mm256_add_epi8(hiN, hiN);
            wP = _mm256_add_epi8(_mm256_add_epi8(wP, _mm256_add_epi8(midP, midP)),
                                 _mm256_add_epi8(hiP2, hiP2));
            wN = _mm256_add_epi8(_mm256_add_epi8(wN, _mm256_add_epi8(midN, midN)),
                                 _mm256_add_epi8(hiN2, hiN2));
        }

        acc = _mm256_add_epi32(acc, _mm256_sub_epi32(lkbFoldToLanes(wP, ones8, ones16),
                                                     lkbFoldToLanes(wN, ones8, ones16)));
    }
    return acc;
}

/// @brief The whole batched residual: **ten window sums for eight keypoints**.
///        **INTERNAL** (E-36, X-79). This is the one function carrying `target`.
///
/// @param magX,magY `[row][plane][lane]`, masked to each lane's region.
/// @param signX,signY `[row][lane]`, a set bit meaning NEGATIVE (D-3).
/// @param outX,outY `[5][kLkBatchLanes]`, in the order `t00, t01, t10, t11, self` —
///        `TapSums`' own field order, so the caller copies straight across.
///
/// @note **Ten separate passes over the staged rows, and that is deliberate.** Doing
///       all ten sums inside one row loop needs ten accumulators plus both components'
///       split magnitudes plus the four constants — past sixteen `ymm` registers, and
///       the spill costs more than the extra loads do. Each pass here keeps ONE
///       accumulator live and reads `[row][plane][lane]` rows that are already in L1.
/// @note The sign split is recomputed per pass rather than staged. Two operations per
///       row against a 32 KB staged array it would otherwise double.
template <size_t N>
__attribute__((target("avx2"))) inline void lkBatchResidual(
    const uint32_t* self, const uint32_t* t00, const uint32_t* t01, const uint32_t* t10,
    const uint32_t* t11, const uint32_t* magX, const uint32_t* signX, const uint32_t* magY,
    const uint32_t* signY, size_t rows, uint32_t* splitP, uint32_t* splitN, int32_t* outX,
    int32_t* outY) {
    const uint32_t* const values[5] = {t00, t01, t10, t11, self};

    for (int comp = 0; comp < 2; ++comp) {
        const uint32_t* mag = comp == 0 ? magX : magY;
        const uint32_t* sgn = comp == 0 ? signX : signY;
        // The sign split, once per component per batch rather than once per pass:
        // P = mag & ~sign, N = mag & sign. Both stay UNSIGNED, which is what lets the
        // byte-domain folding above work at all.
        for (size_t i = 0; i < rows; ++i) {
            const __m256i s = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(sgn + i * kLkBatchLanes));
            for (size_t k = 0; k < N; ++k) {
                const size_t off = (i * N + k) * kLkBatchLanes;
                const __m256i m =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + off));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(splitP + off),
                                    _mm256_andnot_si256(s, m));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(splitN + off),
                                    _mm256_and_si256(s, m));
            }
        }
        int32_t* out = comp == 0 ? outX : outY;
        for (int v = 0; v < 5; ++v) {
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(out + v * static_cast<int>(kLkBatchLanes)),
                lkbWindowSum<N>(values[static_cast<size_t>(v)], splitP, splitN, rows));
        }
    }
}

/// @brief Is the batch usable on this machine? Asked once, not once per batch.
inline bool hasLkBatch() {
    static const bool kYes = __builtin_cpu_supports("avx2");
    return kYes;
}

#else

/// No AVX2 target: the batch does not exist and the tracker stays on its own path.
inline bool hasLkBatch() { return false; }

#endif  // BINCV_X86_LK_BATCH

/// @brief The scalar oracle the AVX2 kernel is held to. **INTERNAL, TEST-FACING.**
///
/// Not a fallback — nothing calls it in a shipped build. It exists because
/// [CLAUDE.md](../../../CLAUDE.md) makes bit-exactness a proven property, and a vector
/// kernel with no independent spelling of the same arithmetic can only be checked
/// against itself. Same layout, same arguments, plain C++.
template <size_t N>
inline void lkBatchResidualScalar(const uint32_t* self, const uint32_t* t00,
                                  const uint32_t* t01, const uint32_t* t10,
                                  const uint32_t* t11, const uint32_t* magX,
                                  const uint32_t* signX, const uint32_t* magY,
                                  const uint32_t* signY, size_t rows, int32_t* outX,
                                  int32_t* outY) {
    const uint32_t* const values[5] = {t00, t01, t10, t11, self};
    for (int comp = 0; comp < 2; ++comp) {
        const uint32_t* mag = comp == 0 ? magX : magY;
        const uint32_t* sgn = comp == 0 ? signX : signY;
        int32_t* out = comp == 0 ? outX : outY;
        for (size_t v = 0; v < 5; ++v) {
            for (size_t lane = 0; lane < kLkBatchLanes; ++lane) {
                int32_t acc = 0;
                for (size_t i = 0; i < rows; ++i) {
                    const uint32_t s = sgn[i * kLkBatchLanes + lane];
                    for (size_t j = 0; j < N; ++j) {
                        const uint32_t m = mag[(i * N + j) * kLkBatchLanes + lane];
                        for (size_t k = 0; k < N; ++k) {
                            const uint32_t x =
                                values[v][(i * N + k) * kLkBatchLanes + lane] & m;
                            const int32_t total = static_cast<int32_t>(popcountWord(x));
                            const int32_t opp = static_cast<int32_t>(popcountWord(x & s));
                            acc += (total - 2 * opp) * (1 << (j + k));
                        }
                    }
                }
                out[v * kLkBatchLanes + lane] = acc;
            }
        }
    }
}

}  // namespace impl
}  // inline namespace BINCV_ABI_NAMESPACE
}  // namespace bincv
