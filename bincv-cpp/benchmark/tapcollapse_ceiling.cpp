// X-62's CEILING for E-34: can the four tap correlations become one?
//
// D-51's account of why binCV's LK costs 2.9x OpenCV has three multipliers --
// x5 taps, xN^2 plane pairs, x2 sign-magnitude -- and D-52 closed every widening
// avenue, leaving the OPERATION COUNT as the only lever. Of the three multipliers
// only the x5 is a design choice (D-20), so it is the one this prices.
//
// Correlation is linear in the taps, so
//
//     w00*S(T00) + w01*S(T01) + w10*S(T10) + w11*S(T11)
//
// may be computed as ONE correlation of a weighted patch, S(w00*T00 + ...), IF
// the weighted patch can be formed in bit-sliced arithmetic -- which needs the
// bilinear weights QUANTISED to k bits. That is ops/pyramid.hpp's `addShifted`
// machinery pointed at taps instead of filter positions.
//
//   A  shipped: four tap correlations per gradient component, ten
//      `slicedSignedSum` calls per row, combined with float weights per window.
//   B  proposed: ONE weighted patch per row -- formed once and SHARED by both
//      components, where A pays its four correlations per component -- then four
//      calls per row instead of ten.
//
// THREE THINGS THIS CEILING DELIBERATELY GIVES ARM B, AND WHY THAT IS STILL A
// CEILING RATHER THAN A CHEAT:
//
//   1. COMPILE-TIME WEIGHTS. A real kernel's subpixel offset moves every
//      iteration, so B would dispatch over the (k+1)^2 quantised offsets. Here
//      the weights are template parameters, so the set-bit decomposition costs
//      nothing at run time. A ceiling is an upper bound on the idea's speed and
//      the dispatch can only subtract from it.
//   2. A FREE DIVIDE. The weights sum to 2^k, so requantising the patch back to N
//      bits is `add 2^(k-1)` and then SELECT PLANES k..k+N-1 -- no restoring
//      division, unlike `requantizeWeighted`. That is not a favour; it is what a
//      real implementation would also get, and it is the reason a power-of-two
//      weight sum is the right quantisation to try.
//   3. IDENTICAL LOADS. Both arms read the same four taps from the same struct.
//      The measured difference is the arithmetic and nothing else.
//
// The arms are NOT compared for equality -- B is an approximation of A, which is
// the opposite of every other ceiling in this project. So closeness is reported
// alongside speed, and the rule below makes the accuracy band mandatory.
//
// Rule (EXPERIMENTS.md X-62, committed before this file):
//   Band A  >= 1.6x AND weight quantisation costs < 0.02 px at k = 3: write the
//           arm, then measure accuracy on the sequence before it ships.
//   Band B  >= 1.6x but accuracy costs more: the trade is the caller's. Report
//           both axes; do not change the default.
//   Band C  < 1.6x: the weighted-sum construction eats the tap saving. Do NOT
//           write it, and record where the ops went.
//   Band D  SLOWER: `weightedAxis` at this shape costs more than four popcount
//           correlations, which contradicts D-51's arithmetic. Report the MODEL
//           failure, not the timing.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

namespace {

using Word = uint32_t;

constexpr size_t kRows = 31;   ///< the shipped 31x31 window, one word per row (D-31)
constexpr size_t kTaps = 4;    ///< t00 t01 t10 t11; `self` is not a tap
constexpr size_t kN = 2;       ///< three of the four levels of the 1/2/2/2 ladder (D-23)

/// One window's rows, laid out as `alignedResidualSums` sees them.
struct Window {
    Word tap[kRows][kTaps][kN];
    Word self[kRows][kN];
    Word mag[2][kRows][kN];    ///< [component][row][plane], already masked to 31 columns
    Word sgn[2][kRows];
};

/// The two numbers a window's residual actually contributes: `b = sum(diff*grad)`
/// for each gradient component. Both arms produce these, by different routes.
struct Residual { double bx, by; };

// ---------------------------------------------------------------------------
// ARM A -- the shipped shape.
// ---------------------------------------------------------------------------

/// Ten `slicedSignedSum` calls per row; the bilinear weights are applied ONCE per
/// window, in floating point, exactly as `TapSums::combine` does.
Residual armFourCorrelations(const Window& w, const double (&fw)[kTaps]) {
    bincv::impl::TapSums sx, sy;
    for (size_t y = 0; y < kRows; ++y) {
        sx.t00 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.tap[y][0]);
        sx.t01 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.tap[y][1]);
        sx.t10 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.tap[y][2]);
        sx.t11 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.tap[y][3]);
        sx.self += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.self[y]);
        sy.t00 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.tap[y][0]);
        sy.t01 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.tap[y][1]);
        sy.t10 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.tap[y][2]);
        sy.t11 += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.tap[y][3]);
        sy.self += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.self[y]);
    }
    return {sx.combine(fw[0], fw[1], fw[2], fw[3]), sy.combine(fw[0], fw[1], fw[2], fw[3])};
}

// ---------------------------------------------------------------------------
// ARM B -- one weighted patch, formed once per row and shared by both components.
// ---------------------------------------------------------------------------

/// `acc += tap * Q`, decomposed over Q's set bits at compile time.
/// @note Deliberately the same shape as `weightedAxisStage`: C++17 has no
///       constexpr for-loop that can instantiate `addShifted` per iteration, and
///       `addShifted`'s shift MUST stay a template parameter -- X-42 measured
///       2.48x for exactly that.
template <size_t SumPlanes, unsigned Q, size_t Bit>
inline void accumulateWeighted(Word* acc, const Word* tap) {
    if constexpr (Bit < 5) {
        if constexpr (((Q >> Bit) & 1u) != 0u) {
            bincv::impl::addShifted<SumPlanes, kN, Bit, Word>(acc, tap);
        }
        accumulateWeighted<SumPlanes, Q, Bit + 1>(acc, tap);
    }
}

/// One row's weighted patch, requantised back to N bits.
/// @note The four weights sum to `2^K`, so the divide is `add 2^(K-1)` and then
///       take planes `K .. K+N-1`. Plane selection is free; that is the whole
///       reason a power-of-two weight sum is the quantisation worth trying.
template <unsigned K, unsigned Q00, unsigned Q01, unsigned Q10, unsigned Q11>
inline void weightedPatch(const Word (&tap)[kTaps][kN], Word (&out)[kN]) {
    constexpr size_t kSumPlanes = kN + K;
    static_assert(Q00 + Q01 + Q10 + Q11 == (1u << K), "the weights must sum to 2^K");
    Word acc[kSumPlanes];
    for (size_t p = 0; p < kSumPlanes; ++p) acc[p] = 0;
    accumulateWeighted<kSumPlanes, Q00, 0>(acc, tap[0]);
    accumulateWeighted<kSumPlanes, Q01, 0>(acc, tap[1]);
    accumulateWeighted<kSumPlanes, Q10, 0>(acc, tap[2]);
    accumulateWeighted<kSumPlanes, Q11, 0>(acc, tap[3]);
    // Round to nearest: +2^(K-1) for every pixel, i.e. an all-ones addend at
    // plane K-1. `addShifted` with a one-plane operand is the increment chain.
    const Word ones[1] = {static_cast<Word>(~static_cast<Word>(0))};
    bincv::impl::addShifted<kSumPlanes, 1, K - 1, Word>(acc, ones);
    for (size_t p = 0; p < kN; ++p) out[p] = acc[p + K];
}

template <unsigned K, unsigned Q00, unsigned Q01, unsigned Q10, unsigned Q11>
Residual armOneCorrelation(const Window& w) {
    long long bx = 0, by = 0;
    for (size_t y = 0; y < kRows; ++y) {
        Word patch[kN];
        weightedPatch<K, Q00, Q01, Q10, Q11>(w.tap[y], patch);
        bx += bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], patch);
        bx -= bincv::impl::slicedSignedSum<kN, Word>(w.mag[0][y], w.sgn[0][y], w.self[y]);
        by += bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], patch);
        by -= bincv::impl::slicedSignedSum<kN, Word>(w.mag[1][y], w.sgn[1][y], w.self[y]);
    }
    return {static_cast<double>(bx), static_cast<double>(by)};
}

// ---------------------------------------------------------------------------

/// Quantise the four bilinear weights to /2^K, forcing the sum to 2^K exactly by
/// giving the rounding residue to the largest. Only used to CHOOSE the constants
/// below; the timed arms take them as template parameters.
void quantiseWeights(double a, double b, unsigned K, unsigned (&q)[kTaps]) {
    const double w[kTaps] = {(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b};
    const double scale = static_cast<double>(1u << K);
    unsigned total = 0, big = 0;
    for (size_t i = 0; i < kTaps; ++i) {
        q[i] = static_cast<unsigned>(std::lround(w[i] * scale));
        total += q[i];
        if (w[i] > w[big]) big = static_cast<unsigned>(i);
    }
    q[big] = static_cast<unsigned>(static_cast<int>(q[big]) + (static_cast<int>(1u << K) - static_cast<int>(total)));
}

unsigned setBits(unsigned v) { unsigned c = 0; while (v) { c += v & 1u; v >>= 1; } return c; }

} // namespace

int main() {
    std::printf("=== X-62 CEILING for E-34: four tap correlations -> one ===\n");
    std::printf("  N=%zu  window=%zux%zu  word=uint32_t\n\n", kN, kRows, kRows);

    constexpr size_t kWindows = 20;
    std::vector<Window> ws(kWindows);
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<Word>(st >> 33);
    };
    for (Window& w : ws) {
        for (size_t y = 0; y < kRows; ++y) {
            for (size_t t = 0; t < kTaps; ++t)
                for (size_t p = 0; p < kN; ++p) w.tap[y][t][p] = next();
            for (size_t p = 0; p < kN; ++p) w.self[y][p] = next();
            for (size_t c = 0; c < 2; ++c) {
                // Sparse, like a real edge map's gradient magnitude, and masked to
                // 31 columns as D-31's aligned path leaves it.
                for (size_t p = 0; p < kN; ++p) w.mag[c][y][p] = (next() & next()) & 0x7FFFFFFFu;
                w.sgn[c][y] = next();
            }
        }
    }

    // ------------------------------------------------------------------
    // THREE OFFSETS, BECAUSE B's COST IS NOT CONSTANT IN THE OFFSET.
    //
    // B's work is one `addShifted` per SET BIT of each quantised weight, so the
    // half-pixel centre (2,2,2,2 -- four single-bit weights) is B's best case and
    // a generic offset is not. A ceiling reported at the best case only would be
    // the fifth overstated ceiling in this project (D-49).
    // ------------------------------------------------------------------
    struct Offset { const char* name; double a, b; };
    const Offset offs[3] = {
        {"a=b=0.50  (B's best: 2,2,2,2)", 0.50, 0.50},
        {"a=0.375 b=0.625  (generic)", 0.375, 0.625},
        {"a=b=0.25  (rounds to 3,2,2,1)", 0.25, 0.25},
    };
    for (const Offset& o : offs) {
        unsigned q[kTaps];
        quantiseWeights(o.a, o.b, 3, q);
        std::printf("  %-32s k=3 -> %u,%u,%u,%u   %u addShifted/row\n", o.name, q[0], q[1], q[2],
                    q[3], setBits(q[0]) + setBits(q[1]) + setBits(q[2]) + setBits(q[3]));
    }

    // ------------------------------------------------------------------
    // CLOSENESS. B is an approximation, so this is not an equality check.
    //
    // NOT reported relative to `b`. `b` is a residual: near convergence it is the
    // small difference of large cancelling sums, so a ratio against it measures
    // the cancellation, not the approximation. On this random data |b| is ~0 and
    // the ratio reads 0.47 -- a number about the denominator.
    //
    // The transferable quantity is the PER-PIXEL PATCH ERROR in LSB, which is
    // where B's loss actually happens and is independent of the data's statistics:
    // A interpolates at FULL PRECISION by keeping five integer sums (D-20), while
    // B rounds each pixel's weighted value into the SAME N-bit alphabet the taps
    // came from. The absolute error in `b` is reported next to the scale of a tap
    // sum, so the two can be compared without either being a ratio against zero.
    //
    // The PIXEL error still needs the covariance matrix and therefore the sequence
    // harness -- X-62's rule says so, and X-51 is the standing reason a proxy does
    // not settle a shipped default.
    // ------------------------------------------------------------------
    const double a = 0.375, b = 0.625;
    const double fw[kTaps] = {(1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b};
    {
        constexpr unsigned kQ[kTaps] = {2, 1, 3, 2};   // quantiseWeights(0.375, 0.625, 3)
        double sumAbs = 0.0, maxAbs = 0.0;
        size_t pixels = 0;
        for (const Window& w : ws) {
            for (size_t y = 0; y < kRows; ++y) {
                Word patch[kN];
                weightedPatch<3, 2, 1, 3, 2>(w.tap[y], patch);
                for (size_t bit = 0; bit < 31; ++bit) {
                    double truth = 0.0;
                    for (size_t t = 0; t < kTaps; ++t) {
                        unsigned v = 0;
                        for (size_t p = 0; p < kN; ++p)
                            v |= static_cast<unsigned>((w.tap[y][t][p] >> bit) & 1u) << p;
                        truth += fw[t] * v;
                    }
                    unsigned got = 0;
                    for (size_t p = 0; p < kN; ++p)
                        got |= static_cast<unsigned>((patch[p] >> bit) & 1u) << p;
                    const double e = std::abs(static_cast<double>(got) - truth);
                    sumAbs += e;
                    maxAbs = std::max(maxAbs, e);
                    ++pixels;
                }
            }
        }
        std::printf("\n  CLOSENESS at k=3, a=0.375 b=0.625 (NOT equality -- B approximates A):\n");
        std::printf("    per-pixel patch error   mean %.3f LSB   max %.3f LSB   (alphabet is 0..%u)\n",
                    sumAbs / static_cast<double>(pixels), maxAbs, (1u << kN) - 1u);
        std::printf("    weights %u,%u,%u,%u /8 against %.4f,%.4f,%.4f,%.4f\n", kQ[0], kQ[1], kQ[2],
                    kQ[3], fw[0], fw[1], fw[2], fw[3]);
        double sumDb = 0.0, maxDb = 0.0, scale = 0.0;
        for (const Window& w : ws) {
            const Residual ra = armFourCorrelations(w, fw);
            const Residual rb = armOneCorrelation<3, 2, 1, 3, 2>(w);
            const double d = std::abs(rb.bx - ra.bx) + std::abs(rb.by - ra.by);
            sumDb += d;
            maxDb = std::max(maxDb, d);
            scale += std::abs(ra.bx) + std::abs(ra.by);
        }
        std::printf("    |b_B - b_A|             mean %.1f          max %.1f      (|b_A| mean %.1f)\n",
                    sumDb / static_cast<double>(kWindows), maxDb, scale / static_cast<double>(kWindows));
        std::printf("    the PIXEL error needs the covariance matrix: sequence harness, per X-62.\n");
    }

    std::vector<measure::Bench> bs = {
        {"A  shipped: 4 correlations/component", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armFourCorrelations(w, fw);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
        {"B  k=3 best   (2,2,2,2)", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armOneCorrelation<3, 2, 2, 2, 2>(w);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
        {"B  k=3 generic(2,1,3,2)", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armOneCorrelation<3, 2, 1, 3, 2>(w);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
        {"B  k=3 worst  (3,3,1,1)", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armOneCorrelation<3, 3, 3, 1, 1>(w);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
        {"B  k=2 generic(1,1,1,1)", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armOneCorrelation<2, 1, 1, 1, 1>(w);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
        {"B  k=4 generic(4,2,6,4)", [&](int) {
             for (const Window& w : ws) {
                 const Residual r = armOneCorrelation<4, 4, 2, 6, 4>(w);
                 measure::g_sink += static_cast<size_t>(r.bx + r.by);
             }
         }},
    };
    const auto tt = measure::measureInterleaved(bs, 9, 50.0);
    std::printf("\n  %-38s %14s %10s\n", "arm", "ns/20 windows", "vs A");
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf("  %-38s %14.1f %9.3fx\n", bs[i].name.c_str(), tt[i].medianNs,
                    tt[0].medianNs / tt[i].medianNs);

    std::printf("\n  X-62's rule: >=1.6x AND <0.02px at k=3 -> write it, then measure accuracy\n"
                "  on the sequence; >=1.6x but costlier -> the caller's trade, default unchanged;\n"
                "  <1.6x -> do NOT write it, record where the ops went; SLOWER -> D-51's\n"
                "  op-count model is wrong and that, not the timing, is the finding.\n");
    std::printf("\n  sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
