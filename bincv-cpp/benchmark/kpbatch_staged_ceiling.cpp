// X-66's CEILING for E-36: staging instead of gathering, and CSA instead of popcount.
//
// X-61 batched eight keypoints into AVX2 lanes and measured PARITY. Its own entry
// records why, and the why is not what D-52 originally said: the VECTOR ARITHMETIC
// WON -- about 48 vector ops against 100 scalar -- and five
// `_mm256_i32gather_epi32` per row at ~15 cycles each gave it straight back.
// That is a DATA-MOVEMENT result, not a packing one. Eight keypoints fill a 256-bit
// register by construction; nothing about a 31-pixel row constrains them (D-55).
//
// So this prices the two changes X-61 did not make. NEITHER IS A PORT OF THE NEON
// PATH -- that path exists because aarch64 has no scalar popcount (D-6), which is a
// problem x86 does not have.
//
//   1. STAGE, DO NOT GATHER. Of the twelve words `alignedResidualSums` reads per
//      row, EIGHT belong to the PREVIOUS frame -- `self`, `magX`, `magY`, `signX`,
//      `signY` -- and LK linearises about the previous frame, so they never move
//      across iterations. Transposed once into [row][plane][lane], the inner loop
//      issues `loadu`, not `gather`.
//
//   2. CSA, DO NOT POPCOUNT PER ROW. AVX2 has no vector popcount and this class of
//      core has no AVX512-VPOPCNTDQ, which is exactly what made X-60's emulation
//      lose. But the kernel does not need each row's count -- it needs THE SUM OVER
//      31 ROWS. A carry-save adder tree computes that with AND/XOR alone: 31
//      one-bit-per-lane values compress into five bit-sliced planes, and only those
//      five are popcounted, with weights 1,2,4,8,16. The reduction stops being a
//      popcount problem and becomes a boolean-logic problem -- which is what AVX2 is
//      good at and what binCV's representation is made of.
//
// ARMS -- all three produce THE SAME TEN INTEGERS per keypoint, and are checked for
// EQUALITY before they are timed. Unlike X-62 this is an identity, not an
// approximation: CSA reassociates a sum of integers, which is exact.
//
//   A  shipped: scalar, one keypoint at a time, `slicedSignedSum` with POPCNT
//   B  staged + AVX2, per-row emulated popcount (Mula's pshufb harvest)
//   C  staged + AVX2, Harley-Seal CSA tree, five popcounts per chain
//
// B EXISTS TO SPLIT THE CREDIT. A alone against C cannot say whether the win is the
// contiguity or the CSA, and "which half did it" decides what gets written.
//
// The STAGING TRANSPOSE is timed separately rather than folded in, because its cost
// is amortised over however many LK iterations reuse it -- up to twenty (D-55). A
// ceiling that buried it would be answering at one iteration and reporting at
// twenty.
//
// Rule (EXPERIMENTS.md X-66, committed before this file): ceiling >=2.0x AND the
// whole-frontend arm >=1.3x -> write it; >=2.0x but frontend <1.3x -> report where
// it went; 1.2-2.0x -> report the gap between the op model and the measurement;
// <1.2x -> fourth refutation, close E-36. Footprint above +5% declines it whatever
// the speed.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bincv-cpp/ops/opticalFlow.hpp"
#include "measure_util.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#define KPS_AVX2 1
#else
#define KPS_AVX2 0
#endif

namespace {

constexpr size_t kRows = 31;    ///< the shipped 31x31 window, one word per row (D-31)
constexpr size_t kKp = 8;       ///< eight keypoints fill a 256-bit register exactly
constexpr size_t kN = 2;        ///< three of four levels of the 1/2/2/2 ladder (D-23)
constexpr size_t kVals = 5;     ///< t00 t01 t10 t11 self
constexpr size_t kComp = 2;     ///< the two gradient components

/// One keypoint's window, laid out as the SCALAR kernel reads it.
struct Window {
    uint32_t val[kRows][kVals][kN];   ///< four taps then `self`
    uint32_t mag[kComp][kRows][kN];   ///< already masked to 31 columns
    uint32_t sgn[kComp][kRows];
};

/// The ten integers a window contributes, per keypoint.
struct Sums { long long v[kComp][kVals]; };

// ---------------------------------------------------------------------------
// ARM A -- the shipped shape, one keypoint at a time.
// ---------------------------------------------------------------------------
void armScalar(const Window* w, Sums* out) {
    for (size_t k = 0; k < kKp; ++k) {
        for (size_t c = 0; c < kComp; ++c)
            for (size_t v = 0; v < kVals; ++v) out[k].v[c][v] = 0;
        for (size_t y = 0; y < kRows; ++y) {
            for (size_t c = 0; c < kComp; ++c) {
                for (size_t v = 0; v < kVals; ++v) {
                    out[k].v[c][v] += bincv::impl::slicedSignedSum<kN, uint32_t, false>(
                        w[k].mag[c][y], w[k].sgn[c][y], w[k].val[y][v]);
                }
            }
        }
    }
}

#if KPS_AVX2

// ---------------------------------------------------------------------------
// THE STAGED LAYOUT. [row][...][lane] -- eight keypoints' words for one row and one
// plane are CONTIGUOUS, so the inner loop loads instead of gathering.
// ---------------------------------------------------------------------------
struct Staged {
    __m256i val[kRows][kVals][kN];
    __m256i mag[kComp][kRows][kN];
    __m256i sgn[kComp][kRows];
};

/// The transpose. Paid ONCE per keypoint batch per level; the inner loop below is
/// paid once per LK ITERATION, of which there are up to twenty.
void stage(const Window* w, Staged& s) {
    alignas(32) uint32_t lane[kKp];
    for (size_t y = 0; y < kRows; ++y) {
        for (size_t v = 0; v < kVals; ++v)
            for (size_t p = 0; p < kN; ++p) {
                for (size_t k = 0; k < kKp; ++k) lane[k] = w[k].val[y][v][p];
                s.val[y][v][p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(lane));
            }
        for (size_t c = 0; c < kComp; ++c) {
            for (size_t p = 0; p < kN; ++p) {
                for (size_t k = 0; k < kKp; ++k) lane[k] = w[k].mag[c][y][p];
                s.mag[c][y][p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(lane));
            }
            for (size_t k = 0; k < kKp; ++k) lane[k] = w[k].sgn[c][y];
            s.sgn[c][y] = _mm256_load_si256(reinterpret_cast<const __m256i*>(lane));
        }
    }
}

/// Population count per 32-BIT LANE -- one keypoint per lane, so the count must not
/// cross lanes. Mula's nibble table, then two horizontal folds inside the lane.
inline __m256i popcnt32(__m256i v) {
    const __m256i lut =
        _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i lowMask = _mm256_set1_epi8(0x0f);
    const __m256i lo = _mm256_and_si256(v, lowMask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), lowMask);
    const __m256i bytes = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                                          _mm256_shuffle_epi8(lut, hi));
    const __m256i w16 = _mm256_maddubs_epi16(bytes, _mm256_set1_epi8(1));
    return _mm256_madd_epi16(w16, _mm256_set1_epi16(1));
}

/// A full adder over 256 bits: `l` is the sum bit, `h` the carry. FIVE boolean ops,
/// and this is the whole trick -- it replaces an operation AVX2 cannot do with
/// operations it does at four per cycle.
inline void csa(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c) {
    const __m256i u = _mm256_xor_si256(a, b);
    h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    l = _mm256_xor_si256(u, c);
}

/// Sum of `popcount` over 32 words, per 32-bit lane, by Harley-Seal.
/// @note FIFTEEN CSAs per sixteen inputs -- about one per word, five ops each --
///       against sixteen emulated popcounts at ~8 ops each. Only the five weight
///       planes are ever popcounted.
inline __m256i harleySeal32(const __m256i* d) {
    const __m256i zero = _mm256_setzero_si256();
    __m256i ones = zero, twos = zero, fours = zero, eights = zero;
    __m256i total = zero;
    for (size_t b = 0; b < 2; ++b) {
        const __m256i* q = d + b * 16;
        __m256i twosA, twosB, foursA, foursB, eightsA, eightsB, sixteens;
        csa(twosA, ones, ones, q[0], q[1]);
        csa(twosB, ones, ones, q[2], q[3]);
        csa(foursA, twos, twos, twosA, twosB);
        csa(twosA, ones, ones, q[4], q[5]);
        csa(twosB, ones, ones, q[6], q[7]);
        csa(foursB, twos, twos, twosA, twosB);
        csa(eightsA, fours, fours, foursA, foursB);
        csa(twosA, ones, ones, q[8], q[9]);
        csa(twosB, ones, ones, q[10], q[11]);
        csa(foursA, twos, twos, twosA, twosB);
        csa(twosA, ones, ones, q[12], q[13]);
        csa(twosB, ones, ones, q[14], q[15]);
        csa(foursB, twos, twos, twosA, twosB);
        csa(eightsB, fours, fours, foursA, foursB);
        csa(sixteens, eights, eights, eightsA, eightsB);
        total = _mm256_add_epi32(total, _mm256_slli_epi32(popcnt32(sixteens), 4));
    }
    total = _mm256_add_epi32(total, _mm256_slli_epi32(popcnt32(eights), 3));
    total = _mm256_add_epi32(total, _mm256_slli_epi32(popcnt32(fours), 2));
    total = _mm256_add_epi32(total, _mm256_slli_epi32(popcnt32(twos), 1));
    total = _mm256_add_epi32(total, popcnt32(ones));
    return total;
}

void scatterLanes(__m256i acc, size_t c, size_t v, Sums* out) {
    alignas(32) int32_t lane[kKp];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lane), acc);
    for (size_t k = 0; k < kKp; ++k) out[k].v[c][v] = lane[k];
}

// ---------------------------------------------------------------------------
// ARM B -- staged, but popcounting every row. Isolates what CONTIGUITY alone buys.
// ---------------------------------------------------------------------------
void armStagedPopcount(const Staged& s, Sums* out) {
    for (size_t c = 0; c < kComp; ++c) {
        for (size_t v = 0; v < kVals; ++v) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t j = 0; j < kN; ++j) {
                for (size_t i = 0; i < kN; ++i) {
                    __m256i sub = _mm256_setzero_si256();
                    for (size_t y = 0; y < kRows; ++y) {
                        const __m256i both =
                            _mm256_and_si256(s.val[y][v][i], s.mag[c][y][j]);
                        const __m256i opp = _mm256_and_si256(both, s.sgn[c][y]);
                        // total - 2*opposing, per lane, exactly as the scalar path.
                        sub = _mm256_add_epi32(
                            sub, _mm256_sub_epi32(popcnt32(both),
                                                  _mm256_slli_epi32(popcnt32(opp), 1)));
                    }
                    acc = _mm256_add_epi32(
                        acc, _mm256_slli_epi32(sub, static_cast<int>(i + j)));
                }
            }
            scatterLanes(acc, c, v, out);
        }
    }
}

// ---------------------------------------------------------------------------
// ARM C -- staged, and the rows folded by a CSA tree before anything is counted.
// ---------------------------------------------------------------------------
void armStagedCsa(const Staged& s, Sums* out) {
    __m256i dTot[32], dOpp[32];
    dTot[31] = _mm256_setzero_si256();
    dOpp[31] = _mm256_setzero_si256();
    for (size_t c = 0; c < kComp; ++c) {
        for (size_t v = 0; v < kVals; ++v) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t j = 0; j < kN; ++j) {
                for (size_t i = 0; i < kN; ++i) {
                    for (size_t y = 0; y < kRows; ++y) {
                        const __m256i both =
                            _mm256_and_si256(s.val[y][v][i], s.mag[c][y][j]);
                        dTot[y] = both;
                        dOpp[y] = _mm256_and_si256(both, s.sgn[c][y]);
                    }
                    const __m256i sub =
                        _mm256_sub_epi32(harleySeal32(dTot),
                                         _mm256_slli_epi32(harleySeal32(dOpp), 1));
                    acc = _mm256_add_epi32(
                        acc, _mm256_slli_epi32(sub, static_cast<int>(i + j)));
                }
            }
            scatterLanes(acc, c, v, out);
        }
    }
}

// ---------------------------------------------------------------------------
// ARM D -- THE CONFIGURATION THAT IS ACTUALLY IMPLEMENTABLE, AND THE REASON THIS
// ARM EXISTS AT ALL.
//
// Arms B and C stage EVERYTHING, taps included. A real kernel cannot: the four tap
// words depend on each keypoint's own integer displacement `(tapX, tapY)`, which
// DIFFERS ACROSS LANES and MOVES BETWEEN ITERATIONS. Only the eight
// previous-frame words -- `self`, `magX`, `magY`, `signX`, `signY` -- are genuinely
// invariant (D-55).
//
// So this arm stages those eight and GATHERS the four taps, which is the honest
// upper bound on the shipped path. D-49 records five ceilings that overstated;
// arm C would have been the sixth if this one had not been written.
// ---------------------------------------------------------------------------
void armStagedTapsGathered(const Staged& s, const Window* w, Sums* out) {
    // Lane strides into the Window array, in 32-bit units -- the index vector a
    // gather needs. Constant across the batch, hoisted out of the row loop.
    alignas(32) int32_t idx[kKp];
    for (size_t k = 0; k < kKp; ++k)
        idx[k] = static_cast<int32_t>(k * (sizeof(Window) / sizeof(uint32_t)));
    const __m256i vidx = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx));
    const int32_t* base = reinterpret_cast<const int32_t*>(w);

    __m256i tap[kRows][4][kN];
    for (size_t y = 0; y < kRows; ++y) {
        for (size_t v = 0; v < 4; ++v) {
            for (size_t p = 0; p < kN; ++p) {
                const size_t off = static_cast<size_t>(
                    reinterpret_cast<const uint32_t*>(&w[0].val[y][v][p]) -
                    reinterpret_cast<const uint32_t*>(w));
                tap[y][v][p] = _mm256_i32gather_epi32(base + off, vidx, 4);
            }
        }
    }
    for (size_t c = 0; c < kComp; ++c) {
        for (size_t v = 0; v < kVals; ++v) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t j = 0; j < kN; ++j) {
                for (size_t i = 0; i < kN; ++i) {
                    __m256i sub = _mm256_setzero_si256();
                    for (size_t y = 0; y < kRows; ++y) {
                        // `self` (v == 4) is invariant and staged; the four taps are
                        // gathered. That split is the whole point of this arm.
                        const __m256i val = (v == 4) ? s.val[y][v][i] : tap[y][v][i];
                        const __m256i both = _mm256_and_si256(val, s.mag[c][y][j]);
                        const __m256i opp = _mm256_and_si256(both, s.sgn[c][y]);
                        sub = _mm256_add_epi32(
                            sub, _mm256_sub_epi32(popcnt32(both),
                                                  _mm256_slli_epi32(popcnt32(opp), 1)));
                    }
                    acc = _mm256_add_epi32(
                        acc, _mm256_slli_epi32(sub, static_cast<int>(i + j)));
                }
            }
            scatterLanes(acc, c, v, out);
        }
    }
}

bool same(const Sums* a, const Sums* b) {
    for (size_t k = 0; k < kKp; ++k)
        for (size_t c = 0; c < kComp; ++c)
            for (size_t v = 0; v < kVals; ++v)
                if (a[k].v[c][v] != b[k].v[c][v]) return false;
    return true;
}
#endif

} // namespace

int main() {
    std::printf("=== X-66 CEILING for E-36: staged loads + CSA, 8 keypoints ===\n");
    std::printf("  N=%zu  window=%zux%zu  batch=%zu keypoints  word=uint32_t\n\n", kN, kRows,
                kRows, kKp);
#if !KPS_AVX2
    std::printf("  built without -mavx2 -- nothing to measure here.\n");
    return 0;
#else
    constexpr size_t kBatches = 20;
    std::vector<Window> ws(kBatches * kKp);
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(st >> 33);
    };
    for (Window& w : ws) {
        for (size_t y = 0; y < kRows; ++y) {
            for (size_t v = 0; v < kVals; ++v)
                for (size_t p = 0; p < kN; ++p) w.val[y][v][p] = next();
            for (size_t c = 0; c < kComp; ++c) {
                // Sparse, like a real edge map's gradient magnitude, and masked to
                // 31 columns as D-31's aligned path leaves it.
                for (size_t p = 0; p < kN; ++p) w.mag[c][y][p] = (next() & next()) & 0x7FFFFFFFu;
                w.sgn[c][y] = next();
            }
        }
    }
    std::vector<Staged> sts(kBatches);
    for (size_t b = 0; b < kBatches; ++b) stage(&ws[b * kKp], sts[b]);

    // EQUALITY BEFORE TIMING. This is an identity, not an approximation.
    std::vector<Sums> a(kKp), bb(kKp), cc(kKp), dd(kKp);
    bool okB = true, okC = true, okD = true;
    for (size_t b = 0; b < kBatches; ++b) {
        armScalar(&ws[b * kKp], a.data());
        armStagedPopcount(sts[b], bb.data());
        armStagedCsa(sts[b], cc.data());
        armStagedTapsGathered(sts[b], &ws[b * kKp], dd.data());
        if (!same(a.data(), bb.data())) okB = false;
        if (!same(a.data(), cc.data())) okC = false;
        if (!same(a.data(), dd.data())) okD = false;
    }
    std::printf("  EQUALITY over %zu batches: B %s   C %s   D %s\n", kBatches,
                okB ? "exact" : "MISMATCH", okC ? "exact" : "MISMATCH",
                okD ? "exact" : "MISMATCH");
    if (!okB || !okC || !okD) {
        std::printf("    A: %lld %lld   B: %lld %lld   C: %lld %lld\n", a[0].v[0][0],
                    a[0].v[1][4], bb[0].v[0][0], bb[0].v[1][4], cc[0].v[0][0], cc[0].v[1][4]);
        return 1;
    }

    std::vector<measure::Bench> bs = {
        {"A  shipped scalar, POPCNT", [&](int) {
             Sums o[kKp];
             for (size_t b = 0; b < kBatches; ++b) {
                 armScalar(&ws[b * kKp], o);
                 measure::g_sink += static_cast<size_t>(o[0].v[0][0]);
             }
         }},
        {"B  staged + per-row popcount", [&](int) {
             Sums o[kKp];
             for (size_t b = 0; b < kBatches; ++b) {
                 armStagedPopcount(sts[b], o);
                 measure::g_sink += static_cast<size_t>(o[0].v[0][0]);
             }
         }},
        {"C  staged + CSA tree", [&](int) {
             Sums o[kKp];
             for (size_t b = 0; b < kBatches; ++b) {
                 armStagedCsa(sts[b], o);
                 measure::g_sink += static_cast<size_t>(o[0].v[0][0]);
             }
         }},
        {"D  staged invariants + GATHERED taps", [&](int) {
             Sums o[kKp];
             for (size_t b = 0; b < kBatches; ++b) {
                 armStagedTapsGathered(sts[b], &ws[b * kKp], o);
                 measure::g_sink += static_cast<size_t>(o[0].v[0][0]);
             }
         }},
        {"   (the staging transpose alone)", [&](int) {
             for (size_t b = 0; b < kBatches; ++b) {
                 stage(&ws[b * kKp], sts[b]);
                 measure::g_sink += 1;
             }
         }},
    };
    const auto tt = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-34s %14s %10s\n", "arm", "ns/20 batches", "vs A");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-34s %14.1f %9.3fx\n", bs[i].name.c_str(), tt[i].medianNs,
                    tt[0].medianNs / tt[i].medianNs);

    // WHAT THE STAGING COSTS, AMORTISED. The transpose is paid once per batch per
    // level; the inner loop once per LK ITERATION.
    // Amortise arm D, not arm C. D is what a kernel can actually do, and only the
    // eight INVARIANT words are staged -- so only their transpose is amortised.
    const double stageNs = tt[4].medianNs, aNs = tt[0].medianNs, cNs = tt[3].medianNs;
    std::printf("\n  amortised over LK iterations (transpose once, inner loop per iteration):\n");
    std::printf("    %-12s %14s %14s %9s\n", "iterations", "A", "D + staging", "vs A");
    for (int it : {1, 2, 3, 5, 10, 20}) {
        const double an = aNs * it, cn = stageNs + cNs * it;
        std::printf("    %-12d %14.1f %14.1f %8.3fx\n", it, an, cn, an / cn);
    }
    std::printf("\n  footprint: staging buffer is %zu B per %zu-keypoint batch\n",
                sizeof(Staged), kKp);
    std::printf("  X-66's rule: >=2.0x here AND >=1.3x on the frontend -> write it;\n"
                "  1.2-2.0x -> report the model/measurement gap; <1.2x -> fourth\n"
                "  refutation, close E-36. Footprint >+5%% declines it whatever the speed.\n");
    std::printf("\n  sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
#endif
}
