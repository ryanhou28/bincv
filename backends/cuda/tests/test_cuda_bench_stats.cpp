// The BENCHMARK HARNESS'S OWN ARITHMETIC, against values computed by hand.
//
// WHY THIS SUITE EXISTS. Every ratio the CUDA backend publishes is summarized
// by backends/cuda/benchmark/paired_stats.hpp, and the decision "is this a
// result" is a line of arithmetic in that header. Until this file there was
// nothing standing behind it -- the statistics were exercised only by
// benchmarks, which nothing checks, and a summary statistic that is quietly
// wrong does not fail a build or a map comparison. It publishes a number.
//
// WHAT IT IS CHECKING AGAINST. Not the implementation re-spelled: every
// expected value below was computed independently and is written as a literal,
// with the arithmetic that produced it stated in the case. Several are exact in
// binary floating point on purpose (2^-14, 1/8, 0.0390625), so those cases
// compare exactly rather than to a tolerance.
//
// THE CASE THAT MATTERS MOST is RangeTestAndProjectRuleDisagree. The harness
// used to call a ratio a result only when the two arms' sample RANGES were
// disjoint. That is not this project's rule -- benchmark/measure_util.hpp tests
// a DIFFERENCE AGAINST A SPREAD, on medians -- and the two disagree in BOTH
// directions:
//
//   * fifteen rounds that all favour the same arm, with one round slow in BOTH
//     arms, overlap the ranges while leaving every per-round ratio untouched.
//     The range test calls that "not a result"; it is one.
//   * three rounds whose arms are cleanly separated but whose per-round ratio
//     scatters from 1.01x to 1.60x pass the range test while saying almost
//     nothing about the ratio. The range test calls that a result; it is a null
//     one.
//
// Both are pinned below, so neither rule can be swapped back in without a
// failure saying which one is running.
//
// NO GPU IS NEEDED and none is asked for -- this is arithmetic over two vectors
// of samples, which is exactly why it lives in its own header away from the
// CUDA timing code. The suite therefore does NOT exit 77 on a device-less
// machine the way its neighbours do: there is nothing here a missing device
// could make unanswerable.

#include <cmath>
#include <cstdio>
#include <vector>

#include "paired_stats.hpp"
#include "test_util.hpp"

namespace {

using cudabench::PairedTiming;
using cudabench::kScatterNotMeasured;
using cudabench::signTestTwoSidedP;
using cudabench::summarizePaired;

/// Comparison to a stated tolerance, so a failure prints both values.
bool near(double actual, double expected, double tol) {
    const bool ok = std::fabs(actual - expected) <= tol;
    if (!ok) {
        std::printf("      got %.12g, expected %.12g (tolerance %.3g)\n", actual, expected,
                    tol);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// THE CASE THE RANGE TEST GOT WRONG, built to be read.
//
// Fifteen paired rounds. Arm A runs at 1.0 ms in fourteen of them; in the last,
// something takes the machine and BOTH arms are slow together -- A reads 4.0 ms
// and B reads 5.5 ms. That is drift, which is the exact thing pairing cancels:
// the per-round ratio in that round is 5.5 / 4.0 = 1.375, right in the middle
// of the other fourteen.
//
// Arm B is 1.25 ms in the first seven rounds and 1.5 ms in the next seven, so
// the ratios are seven 1.25s, seven 1.5s and one 1.375 -- median 1.375, and
// every single round favours A.
//
// A's samples span 1.0 to 4.0 and B's span 1.25 to 5.5, so the RANGES OVERLAP.
// ---------------------------------------------------------------------------
PairedTiming oneOutlierFifteenRounds() {
    std::vector<double> a(15, 1.0);
    a[14] = 4.0;
    std::vector<double> b;
    for (int i = 0; i < 7; ++i) b.push_back(1.25);
    for (int i = 0; i < 7; ++i) b.push_back(1.5);
    b.push_back(5.5);
    return summarizePaired(a, b);
}

}  // namespace

BINCV_TEST(PairedStats, RangeTestAndProjectRuleDisagree) {
    const PairedTiming p = oneOutlierFifteenRounds();

    // The per-round ratios, by hand: 1.25 (x7), 1.5 (x7), 1.375. Sorted, the
    // eighth of fifteen is 1.375.
    BINCV_CHECK(near(p.ratioMedian, 1.375, 1e-12));
    BINCV_CHECK(near(p.ratioMin, 1.25, 1e-12));
    BINCV_CHECK(near(p.ratioMax, 1.5, 1e-12));
    BINCV_CHECK_EQ(p.rounds, 15);

    // Every round had B slower than A, so the sign count is 15-0 and the
    // two-sided sign-test p is 2^-14 -- exact in binary, so compared exactly.
    BINCV_CHECK_EQ(p.roundsFavouringA, 15);
    BINCV_CHECK_EQ(p.roundsFavouringB, 0);
    BINCV_CHECK_EQ(p.roundsTied, 0);
    BINCV_CHECK(p.signTestP() == 6.103515625e-05);
    BINCV_CHECK(p.unanimous());

    // THE OLD RULE: A spans 1.0-4.0 and B spans 1.25-5.5, so neither range is
    // wholly below the other. separated() is FALSE, and it is still computed
    // and still reported -- as a fact.
    BINCV_CHECK(!p.separated());

    // THE PROJECT'S RULE: the difference is |1.375 - 1| = 37.5%. The within-run
    // spread is the PER-ROUND RATIO's own, (1.5 - 1.25) / 1.375 = 18.18%. 37.5
    // clears 18.18, so this is a result -- which the range test denied.
    BINCV_CHECK(near(p.differencePct(), 37.5, 1e-9));
    BINCV_CHECK(near(p.ratioSpreadPct(), 18.1818181818, 1e-8));
    BINCV_CHECK(p.differenceClearsNoise(kScatterNotMeasured));

    // And the two arms' own spreads, which the range test is really reading,
    // are enormous -- 4.0/1.0 and 5.5/1.25 -- precisely because the drift is
    // still in them. That is the double count: charging the ratio for noise the
    // pairing already removed.
    BINCV_CHECK(p.a.spreadPct() > 200.0);
    BINCV_CHECK(p.b.spreadPct() > 200.0);
}

BINCV_TEST(PairedStats, SeparatedRangesAreNotSufficientEither) {
    // Three rounds. A is 1.0 ms every time; B is 1.01, 1.60 and 1.02. A's range
    // is a single point at 1.0 and B's starts at 1.01, so the two ranges ARE
    // disjoint and the old rule calls this a result.
    //
    // The per-round ratios are 1.01, 1.60 and 1.02: median 1.02, so the
    // difference is 2%, against a per-round spread of (1.60 - 1.01) / 1.02 =
    // 57.8%. The ratio is scattered far wider than it is displaced. Under the
    // project's rule that is a NULL result, and a null result is a result.
    const PairedTiming p = summarizePaired({1.0, 1.0, 1.0}, {1.01, 1.60, 1.02});

    BINCV_CHECK(p.separated());
    BINCV_CHECK(near(p.ratioMedian, 1.02, 1e-12));
    BINCV_CHECK(near(p.differencePct(), 2.0, 1e-9));
    BINCV_CHECK(near(p.ratioSpreadPct(), 57.8431372549, 1e-8));
    BINCV_CHECK(!p.differenceClearsNoise(kScatterNotMeasured));
}

BINCV_TEST(PairedStats, GeometricMeanIsOrderIndependentAndArithmeticIsNot) {
    // Three rounds with ratios 0.5, 2.0 and 4.0 -- A = 1.0 throughout.
    const PairedTiming fwd = summarizePaired({1.0, 1.0, 1.0}, {0.5, 2.0, 4.0});
    // The same three rounds read from the other arm: ratios 2.0, 0.5, 0.25.
    const PairedTiming rev = summarizePaired({0.5, 2.0, 4.0}, {1.0, 1.0, 1.0});

    // (0.5 * 2 * 4)^(1/3) = 4^(1/3) = 1.587401...
    BINCV_CHECK(near(fwd.ratioGeoMean, 1.5874010519681994, 1e-12));
    // (2 * 0.5 * 0.25)^(1/3) = 0.25^(1/3) = 0.629960...
    BINCV_CHECK(near(rev.ratioGeoMean, 0.6299605249474366, 1e-12));

    // THE PROPERTY THAT MAKES IT THE RIGHT CENTRAL VALUE: reading the same
    // rounds from the other arm gives exactly the reciprocal, so the summary
    // does not depend on which arm was put in the denominator.
    BINCV_CHECK(near(fwd.ratioGeoMean * rev.ratioGeoMean, 1.0, 1e-12));

    // The arithmetic mean does not have it: (0.5+2+4)/3 = 2.1667 and
    // (2+0.5+0.25)/3 = 0.9167, whose product is 1.986, not 1. Stated here as a
    // check rather than a comment so the claim is exercised.
    const double arithFwd = (0.5 + 2.0 + 4.0) / 3.0;
    const double arithRev = (2.0 + 0.5 + 0.25) / 3.0;
    BINCV_CHECK(near(arithFwd * arithRev, 1.9861111111, 1e-8));
    BINCV_CHECK(std::fabs(arithFwd * arithRev - 1.0) > 0.9);
}

BINCV_TEST(PairedStats, MedianIsTheRobustOneAndTheGeometricMeanIsNot) {
    // Five rounds at ratio 1.2 with one round at 10.0 -- one descheduling event
    // in arm B alone. The median is unmoved; the geometric mean is dragged to
    // (1.2^4 * 10)^(1/5) = 1.8338. That is why the median is what gets quoted
    // and the geometric mean is printed beside it rather than instead of it.
    const PairedTiming p =
        summarizePaired({1.0, 1.0, 1.0, 1.0, 1.0}, {1.2, 1.2, 1.2, 1.2, 10.0});

    BINCV_CHECK(near(p.ratioMedian, 1.2, 1e-12));
    BINCV_CHECK(near(p.ratioGeoMean, 1.8337705629789587, 1e-12));
    BINCV_CHECK(near(p.ratioMin, 1.2, 1e-12));
    BINCV_CHECK(near(p.ratioMax, 10.0, 1e-12));
    BINCV_CHECK_EQ(p.roundsFavouringA, 5);
}

BINCV_TEST(PairedStats, SignTestValuesAreTheBinomialTail) {
    // Exact binary values, compared exactly.
    //
    //   15-0: 2 * C(15,0) / 2^15 = 2^-14                     = 6.103515625e-05
    //    6-1: 2 * (C(7,0) + C(7,1)) / 2^7 = 2 * 8/128        = 0.125
    //    8-1: 2 * (C(9,0) + C(9,1)) / 2^9 = 2 * 10/512       = 0.0390625
    //    9-0: 2 * C(9,0) / 2^9 = 2/512                       = 0.00390625
    //    5-4: 2 * sum(C(9,0..4)) / 2^9 = 2 * 256/512 = 1     (clamped)
    BINCV_CHECK(signTestTwoSidedP(15, 0) == 6.103515625e-05);
    BINCV_CHECK(signTestTwoSidedP(0, 15) == 6.103515625e-05);  // symmetric
    BINCV_CHECK(signTestTwoSidedP(6, 1) == 0.125);
    BINCV_CHECK(signTestTwoSidedP(8, 1) == 0.0390625);
    BINCV_CHECK(signTestTwoSidedP(9, 0) == 0.00390625);
    BINCV_CHECK(signTestTwoSidedP(5, 4) == 1.0);

    // No usable rounds says nothing, and one round says nothing either: a
    // single coin flip lands somewhere.
    BINCV_CHECK(signTestTwoSidedP(0, 0) == 1.0);
    BINCV_CHECK(signTestTwoSidedP(1, 0) == 1.0);
}

BINCV_TEST(PairedStats, UnanimityNeedsMoreThanOneUsableRound) {
    BINCV_CHECK(summarizePaired({1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}).unanimous());
    BINCV_CHECK(!summarizePaired({1.0}, {2.0}).unanimous());
    // Split rounds are not unanimous whatever the medians say.
    BINCV_CHECK(!summarizePaired({1.0, 2.0}, {2.0, 1.0}).unanimous());
    // No usable rounds at all is not unanimity.
    BINCV_CHECK(!summarizePaired({}, {}).unanimous());
}

BINCV_TEST(PairedStats, TheRuleTakesTheLargerOfTheTwoNoises) {
    // Three rounds with ratios 1.4, 1.5, 1.6: median 1.5, difference 50%,
    // per-round spread (1.6 - 1.4) / 1.5 = 13.33%.
    const PairedTiming p = summarizePaired({1.0, 1.0, 1.0}, {1.4, 1.5, 1.6});
    BINCV_CHECK(near(p.differencePct(), 50.0, 1e-9));
    BINCV_CHECK(near(p.ratioSpreadPct(), 13.3333333333, 1e-8));

    // Nobody has measured the run-to-run scatter: the within-run half is all
    // there is to clear, and 50% clears 13.33%.
    BINCV_CHECK(near(p.noiseToClearPct(kScatterNotMeasured), 13.3333333333, 1e-8));
    BINCV_CHECK(p.differenceClearsNoise(kScatterNotMeasured));

    // A measured scatter SMALLER than the within-run spread does not lower the
    // bar -- the rule takes the larger.
    BINCV_CHECK(near(p.noiseToClearPct(5.0), 13.3333333333, 1e-8));
    BINCV_CHECK(p.differenceClearsNoise(5.0));

    // A measured scatter LARGER than the within-run spread raises it, and
    // 50% against 60% does not clear. This is the half of the rule that a
    // single process cannot see, and the half a benchmark with the figure for
    // its host would supply.
    BINCV_CHECK(near(p.noiseToClearPct(60.0), 60.0, 1e-12));
    BINCV_CHECK(!p.differenceClearsNoise(60.0));

    // At exactly the bar it does not clear: "smaller than the spread is a null
    // result" reads a tie as a null, which is the conservative direction.
    BINCV_CHECK(!p.differenceClearsNoise(50.0));
}

BINCV_TEST(PairedStats, NoDifferenceIsANullResultAndSaysSo) {
    // Ratios 0.9, 1.0, 1.1: median 1.0, difference 0%, spread 20%.
    const PairedTiming p = summarizePaired({1.0, 1.0, 1.0}, {0.9, 1.0, 1.1});
    BINCV_CHECK(near(p.ratioMedian, 1.0, 1e-12));
    BINCV_CHECK(near(p.differencePct(), 0.0, 1e-9));
    BINCV_CHECK(near(p.ratioSpreadPct(), 20.0, 1e-8));
    BINCV_CHECK(!p.differenceClearsNoise(kScatterNotMeasured));

    // One round each way plus one exact tie -- the sign test has nothing to
    // say, which is the correct thing for it to say.
    BINCV_CHECK_EQ(p.roundsFavouringA, 1);
    BINCV_CHECK_EQ(p.roundsFavouringB, 1);
    BINCV_CHECK_EQ(p.roundsTied, 1);
    BINCV_CHECK(p.signTestP() == 1.0);
}

BINCV_TEST(PairedStats, DegenerateInputsDoNotProduceANumber) {
    // No rounds at all: nothing is a result.
    const PairedTiming empty = summarizePaired({}, {});
    BINCV_CHECK_EQ(empty.rounds, 0);
    BINCV_CHECK(near(empty.ratioMedian, 0.0, 0.0));
    BINCV_CHECK(near(empty.ratioGeoMean, 0.0, 0.0));
    BINCV_CHECK(!empty.differenceClearsNoise(kScatterNotMeasured));
    BINCV_CHECK(!empty.separated());

    // A round whose arm-A reading is not positive is not a measurement of a
    // ratio. It is counted as a tie and left out of the ratio statistics
    // entirely, rather than contributing a 0.0 that would drag the range to
    // zero and the geometric mean with it.
    const PairedTiming zeroed = summarizePaired({1.0, 0.0, 1.0}, {2.0, 2.0, 2.0});
    BINCV_CHECK_EQ(zeroed.rounds, 3);
    BINCV_CHECK_EQ(zeroed.roundsTied, 1);
    BINCV_CHECK_EQ(zeroed.roundsFavouringA, 2);
    BINCV_CHECK(near(zeroed.ratioMedian, 2.0, 1e-12));
    BINCV_CHECK(near(zeroed.ratioMin, 2.0, 1e-12));
    BINCV_CHECK(near(zeroed.ratioGeoMean, 2.0, 1e-12));

    // Vectors of different lengths are truncated to the shorter, because an
    // unpaired round is not a paired observation.
    const PairedTiming ragged = summarizePaired({1.0, 1.0, 1.0}, {2.0, 2.0});
    BINCV_CHECK_EQ(ragged.rounds, 2);
    BINCV_CHECK(near(ragged.a.maxMs, 1.0, 1e-12));
    BINCV_CHECK(near(ragged.b.maxMs, 2.0, 1e-12));
}

BINCV_TEST(PairedStats, SeparationIsStillComputedWhenItIsTrue) {
    // Kept because disjoint ranges remain a strong thing to be able to say, and
    // several published claims are stronger for it. What changed is that it no
    // longer decides.
    const PairedTiming apart = summarizePaired({1.0, 1.1, 1.05}, {2.0, 2.1, 2.05});
    BINCV_CHECK(apart.separated());
    const PairedTiming swapped = summarizePaired({2.0, 2.1, 2.05}, {1.0, 1.1, 1.05});
    BINCV_CHECK(swapped.separated());  // disjoint the other way round
    const PairedTiming touching = summarizePaired({1.0, 2.0}, {2.0, 3.0});
    BINCV_CHECK(!touching.separated());  // ranges that meet are not disjoint
}

BINCV_TEST(PairedStats, ArmSummariesAreTheProjectsMedianAndSpread) {
    // The Timing summary itself, so that "median" and "spread" mean the same
    // here as in benchmark/measure_util.hpp: median of an odd count is the
    // middle sample, of an even count the midpoint of the two central ones,
    // and spread is (max - min) / median as a percentage.
    const cudabench::Timing odd = cudabench::summarize({3.0, 1.0, 2.0});
    BINCV_CHECK(near(odd.minMs, 1.0, 1e-12));
    BINCV_CHECK(near(odd.medianMs, 2.0, 1e-12));
    BINCV_CHECK(near(odd.maxMs, 3.0, 1e-12));
    BINCV_CHECK(near(odd.spreadPct(), 100.0, 1e-9));

    const cudabench::Timing even = cudabench::summarize({4.0, 1.0, 3.0, 2.0});
    BINCV_CHECK(near(even.medianMs, 2.5, 1e-12));
    BINCV_CHECK(near(even.spreadPct(), 120.0, 1e-9));

    const cudabench::Timing none = cudabench::summarize({});
    BINCV_CHECK(near(none.medianMs, 0.0, 0.0));
    BINCV_CHECK(near(none.spreadPct(), 0.0, 0.0));
}

BINCV_TEST(PairedStats, TheRunToRunScatterIsNotDefaultedToANumber) {
    // A figure invented here would make the stronger half of the project's rule
    // look applied when nobody had measured it. The slot starts empty and says
    // so, and an empty slot leaves the within-run spread as the bar.
    BINCV_CHECK(cudabench::runToRunScatterPct() == kScatterNotMeasured);
    BINCV_CHECK(kScatterNotMeasured < 0.0);

    const PairedTiming p = summarizePaired({1.0, 1.0, 1.0}, {1.4, 1.5, 1.6});
    BINCV_CHECK(near(p.noiseToClearPct(cudabench::runToRunScatterPct()),
                     p.ratioSpreadPct(), 1e-12));
}

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA paired-statistics tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    return ::bincv::test::runAll("CUDA paired-statistics tests", argc, argv);
}
#endif
