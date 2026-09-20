#pragma once

// What a PAIRED two-arm comparison is summarized by, and the rule that decides
// whether its difference is real.
//
// SEPARATE FROM cuda_bench_util.hpp, AND CUDA-FREE, for one reason: everything
// here is arithmetic over two vectors of samples, so it can be checked against
// hand-computed values by a test that needs no GPU. Producing those vectors
// needs a device; deciding what they mean does not, and the part that decides
// is the part worth pinning.
//
// THE STANDARD IS benchmark/measure_util.hpp'S. This file carries it across to
// a paired design and adds nothing to it. That header states it in three
// sentences:
//
//   * "medianNs is what ratios are taken on -- it is not moved by a single
//     descheduling event the way a mean is, and unlike the minimum it does not
//     report the luckiest batch as though it were typical."
//   * "A difference smaller than the spread is a null result, and a null result
//     is a result."
//   * "The spread bounds WITHIN-run noise only; run-to-run scatter is a
//     separate and sometimes larger number, and an entry that calls a
//     difference real should clear the larger of the two."
//
// So the project's test is a DIFFERENCE AGAINST SPREAD, taken on medians. It is
// not a test for disjoint sample ranges. Requiring max(A) < min(B) is a
// strictly stronger bar that a single unlucky round can veto, and it was never
// what this project does: a pair whose per-round ratio is 1.24x in every one of
// fifteen rounds is a result under the rule above, and fails a range test the
// moment one round is slow in BOTH arms -- which is drift, the exact thing
// pairing exists to cancel.
//
// WHAT A PAIRED DESIGN GIVES THAT THE HOST HARNESS DOES NOT. timeKernelPaired
// brackets both arms inside every round and alternates their order, so each
// round yields ONE observation of the ratio with the drift already divided out.
// Four things are worth saying about a set of those:
//
//   MEDIAN per-round ratio -- the value to quote, for measure_util.hpp's own
//   reason: one bad round cannot move it.
//
//   GEOMETRIC MEAN of the per-round ratios -- the central value of a ratio, and
//   the reason it is not the arithmetic one is not a preference. A ratio and
//   its reciprocal are the same observation read from either arm, so a central
//   value has to satisfy centre(B/A) = 1 / centre(A/B). The geometric mean
//   does. The arithmetic mean does not: average 0.5 and 2.0 and it reads 1.25,
//   so the SAME two rounds report a 25% advantage to whichever arm is put in
//   the denominator. A summary statistic that depends on argument order is not
//   one.
//
//   RANGE -- printed always, because the scatter is part of the answer rather
//   than a blemish on it.
//
//   SIGN COUNT -- how many rounds each arm won. The rounds are PAIRED, so this
//   is a sign test, and it is exactly the statistic that separates
//   "consistently 1.2x" from "noisy around 1.0". Under no difference each round
//   is a coin flip, so fifteen rounds falling the same way has a two-sided
//   probability of 2^-14. Two arms whose ranges overlap because of one slow
//   round still split 15-0, which is why the sign count sees what the range
//   test cannot.
//
// AND THE DECISION. differenceClearsNoise() is measure_util.hpp's rule and
// nothing else: the difference must exceed the LARGER of the within-run spread
// and the run-to-run scatter. Both sides of that comparison are measured AS
// FACTORS, and that is not a presentation choice -- see below.
//
//   * The DIFFERENCE is differenceFactor(): how many times apart the two arms
//     are, max(median, 1 / median), which is 1.00x at parity and rises in
//     either direction.
//   * The WITHIN-RUN SPREAD is ratioSwingFactor(): how many times the PER-ROUND
//     RATIO itself swung across the rounds, max / min. The ratio's swing, not
//     the arms': the ratio is the quantity being decided and the paired round
//     is what measures it, so charging the difference for the arms' spreads
//     would charge it for the drift the pairing already removed. That double
//     count is what the range test does.
//   * The RUN-TO-RUN SCATTER cannot be observed from inside one process -- it
//     takes several. It is therefore a parameter here, in the same units: the
//     largest median this benchmark produced across processes over the
//     smallest. While nobody has measured it for a host the predicate can test
//     only the within-run half, and the printer says so at the verdict rather
//     than letting a reader take half the rule for the whole of it.
//
// WHY FACTORS AND NOT PERCENTAGES, which is the same argument the geometric
// mean is here for, applied to the predicate instead of to the centre.
//
// The obvious spelling is |median - 1| as a percentage against
// (max - min) / median as a percentage. It does not work, and the way it fails
// is not subtle: IT DEPENDS ON WHICH ARM IS THE DENOMINATOR. |median - 1| can
// never exceed 100% for an arm that is FASTER, however much faster it is --
// 0.083x reads 91.7% and 0.0083x still reads 99.2% -- while (max - min) /
// median has no ceiling at all. So a fast arm's claim is capped at 100 while
// the bar it must clear is not, and the two swap places the moment the ratio
// is inverted. Measured on this backend's own headline row, fifteen paired
// rounds of dense disparity:
//
//   as binCV/OpenCV : difference   91.7%  against spread 109.9%  -> NULL
//   as OpenCV/binCV : difference 1103.4%  against spread  66.6%  -> RESULT
//
// Same rounds, same arms, opposite verdicts, decided by nothing but argument
// order -- a 12x lead with a 15-0 sign split reported as "the two arms are the
// same speed as far as this run can tell". A predicate that does that is not
// measuring the arms.
//
// In factors both quantities are invariant under the swap: inverting the ratio
// replaces median by 1 / median, which leaves max(median, 1/median) alone, and
// replaces (min, max) by (1/max, 1/min), which leaves max / min alone. Both
// rows above then read 12.03x against 2.30x and the answer is A RESULT either
// way. This is exactly the property the geometric mean was chosen for at the
// top of this file -- centre(B/A) = 1 / centre(A/B) -- and a predicate that
// lacks it is no more a comparison than an arithmetic mean of ratios is a
// centre.
//
// differencePct() and ratioSpreadPct() are KEPT and still printed, because near
// parity a percentage is the readable form and it is what measure_util.hpp's
// own tables show. They are reported, not decisive, and they are labelled that
// way at the number.
//
// SEPARATION IS STILL REPORTED. PairedTiming::separated() stays, because
// disjoint ranges are a strong fact and several claims read better for it. It
// is a FACT printed beside the verdict; it is not the verdict.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cudabench {

struct Timing {
    double minMs = 0.0;
    double medianMs = 0.0;
    double maxMs = 0.0;

    /// @brief Full scatter as a percentage of the median: (max - min) / median.
    /// The same definition measure::Timing::spreadPct uses, so the two
    /// harnesses' "spread" columns mean one thing.
    double spreadPct() const {
        return medianMs > 0.0 ? (maxMs - minMs) / medianMs * 100.0 : 0.0;
    }
};

/// @brief min / median / max of a sample set, sorted in place.
/// @note The median of an even count is the arithmetic midpoint of the two
/// central samples, which is what measure::Timing does -- so "median" means
/// the same thing in both harnesses. For a RATIO the multiplicative
/// midpoint would be marginally more principled, and it is not worth a
/// second definition of the word: two adjacent samples 10% apart differ by
/// 0.1% between the two midpoints.
inline Timing summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    Timing t;
    if (samples.empty()) return t;
    t.minMs = samples.front();
    t.maxMs = samples.back();
    const size_t m = samples.size();
    t.medianMs = (m % 2 == 1) ? samples[m / 2]
                              : 0.5 * (samples[m / 2 - 1] + samples[m / 2]);
    return t;
}

// ---------------------------------------------------------------------------
// Run-to-run scatter: the half of the rule one process cannot see
// ---------------------------------------------------------------------------

/// @brief The value that means "nobody has measured the run-to-run scatter on
/// this host". Negative on purpose: a scatter FACTOR is at least 1.00x, so a
/// negative can never be mistaken for one, and taking the larger of it and
/// the within-run swing yields the within-run swing with no special case --
/// which lets a printer detect it and say which half of the rule was
/// actually applied.
inline constexpr double kScatterNotMeasured = -1.0;

/// @brief The host's run-to-run scatter as a FACTOR -- the largest median this
/// benchmark produced across independent processes over the smallest -- or
/// kScatterNotMeasured.
/// @note THE SAME UNITS AS ratioSwingFactor, which is the point: the rule takes
/// the larger of the two, and a percentage cannot be compared with a factor.
/// @note NOT DEFAULTED TO A NUMBER. A figure invented here would make the
/// stronger half of measure_util.hpp's rule look applied when it was not,
/// and this project does not fill in a bar nobody has measured. Measuring
/// it means running a benchmark several times and looking at the scatter of
/// its medians ACROSS processes; a benchmark that has that figure for its
/// own host sets this once at startup and every verdict it prints then
/// clears the whole rule instead of half of it.
inline double& runToRunScatterFactor() {
    static double factor = kScatterNotMeasured;
    return factor;
}

// ---------------------------------------------------------------------------
// The sign test
// ---------------------------------------------------------------------------

/// @brief Two-sided sign-test p-value for `winsA` rounds won by one arm and
/// `winsB` by the other.
/// @note Exact, not an approximation: `2 * sum(C(n,i), i = 0..min(winsA,winsB))
/// / 2^n`, clamped at 1. Ties are excluded from n, which is what a sign test
/// does with them.
/// @note REPORTED, NEVER GATED ON. Turning "p below some number" into a merge
/// criterion would be inventing a project-wide threshold, which is the one
/// thing CLAUDE.md says not to do. What this value is for is telling a
/// reader whether fifteen rounds on one side is the kind of thing noise
/// does.
inline double signTestTwoSidedP(int winsA, int winsB) {
    if (winsA < 0 || winsB < 0) return 1.0;
    const int n = winsA + winsB;
    if (n <= 0) return 1.0;
    const int k = std::min(winsA, winsB);
    double term = std::pow(0.5, static_cast<double>(n));  // C(n,0) / 2^n
    double tail = term;
    for (int i = 1; i <= k; ++i) {
        term *= static_cast<double>(n - i + 1) / static_cast<double>(i);
        tail += term;
    }
    return std::min(1.0, 2.0 * tail);
}

// ---------------------------------------------------------------------------
// The paired summary
// ---------------------------------------------------------------------------

/// @brief Two arms measured against each other round by round, summarized.
/// @note `ratio*` are the distribution of B/A computed WITHIN each round, not a
/// ratio of the two medians. The distinction is the whole point: a per-round
/// ratio cancels drift that moved both arms, whereas a ratio of
/// separately-measured medians carries the drift between them.
struct PairedTiming {
    Timing a;                   ///< arm A, its own min/median/max
    Timing b;                   ///< arm B, same
    double ratioMin = 0.0;      ///< smallest per-round B/A
    double ratioMedian = 0.0;   ///< median per-round B/A -- the value to quote
    double ratioMax = 0.0;      ///< largest per-round B/A
    double ratioGeoMean = 0.0;  ///< geometric mean of the per-round B/A
    int roundsFavouringA = 0;   ///< rounds where B was SLOWER (ratio above 1)
    int roundsFavouringB = 0;   ///< rounds where B was FASTER (ratio below 1)
    int roundsTied = 0;         ///< rounds that were exactly equal or unusable
    int rounds = 0;             ///< paired rounds measured (ties included)

    /// @brief Whether the two arms' sample RANGES are disjoint.
    /// @note A REPORTED FACT, NOT A VERDICT. Disjoint ranges are a strong thing
    /// to be able to say and a claim is better for it. Overlapping ranges are
    /// NOT "not a result": one round that is slow in both arms overlaps the
    /// ranges while leaving every per-round ratio untouched, and the ratio is
    /// what is being decided. What decides is differenceClearsNoise(), which
    /// is benchmark/measure_util.hpp's own rule.
    bool separated() const { return a.maxMs < b.minMs || b.maxMs < a.minMs; }

    /// @brief The per-round ratio's own spread, (max - min) / median as a
    /// percentage: the WITHIN-RUN noise on the difference, measured on the
    /// paired observation rather than on either arm.
    /// @note REPORTED, NOT DECISIVE. Readable near parity, and not comparable
    /// with differencePct() away from it -- see the file header. What
    /// decides is ratioSwingFactor().
    double ratioSpreadPct() const {
        return ratioMedian > 0.0 ? (ratioMax - ratioMin) / ratioMedian * 100.0 : 0.0;
    }

    /// @brief How far the median per-round ratio sits from 1.00x, in percent.
    /// @note REPORTED, NOT DECISIVE, and it is the one that fails: an arm that
    /// is faster can never read above 100% here however far ahead it is.
    /// differenceFactor() is what the rule uses.
    double differencePct() const { return std::fabs(ratioMedian - 1.0) * 100.0; }

    /// @brief How many times the PER-ROUND RATIO swung across the rounds,
    /// max / min: the within-run noise on the difference, in the units the
    /// rule compares in. 1.00x is a ratio that read the same every round.
    double ratioSwingFactor() const {
        return ratioMin > 0.0 && ratioMax > 0.0 ? ratioMax / ratioMin : 0.0;
    }

    /// @brief How many times apart the two arms are: max(median, 1 / median),
    /// so 1.00x is parity and the value rises whichever arm is ahead.
    /// This is "the difference" in measure_util.hpp's sentence.
    double differenceFactor() const {
        if (!(ratioMedian > 0.0)) return 0.0;
        return ratioMedian >= 1.0 ? ratioMedian : 1.0 / ratioMedian;
    }

    /// @brief What the difference has to beat: the LARGER of the within-run
    /// swing and the run-to-run scatter, both as factors.
    /// @param runToRunFactor The host's run-to-run scatter as a factor, or
    /// kScatterNotMeasured -- which is negative, so an unmeasured scatter
    /// simply leaves the within-run swing as the bar.
    double noiseToClearFactor(double runToRunFactor) const {
        const double within = ratioSwingFactor();
        return runToRunFactor > within ? runToRunFactor : within;
    }

    /// @brief measure_util.hpp's rule, and nothing else: is the difference
    /// larger than the noise it was measured against?
    /// @note A false here is a NULL RESULT, which that header is explicit is
    /// itself a result -- the two arms are the same speed as far as this run
    /// can tell. It is not "no data".
    /// @note INVARIANT UNDER SWAPPING THE ARMS, which the percentage spelling
    /// of this predicate was not. Both quantities it compares are unchanged
    /// when the ratio is inverted, so the same rounds cannot be a result one
    /// way round and a null the other.
    bool differenceClearsNoise(double runToRunFactor) const {
        return rounds > 0 && ratioMedian > 0.0 &&
               differenceFactor() > noiseToClearFactor(runToRunFactor);
    }

    /// @brief Two-sided sign-test p over the rounds that were not ties.
    double signTestP() const { return signTestTwoSidedP(roundsFavouringA, roundsFavouringB); }

    /// @brief True when every usable round fell the same way. For a pair that is
    /// supposed to be the SAME code (a gate-excluded control), this is a red
    /// flag no median can raise: identical arms scatter both ways.
    bool unanimous() const {
        const int n = roundsFavouringA + roundsFavouringB;
        return n > 1 && (roundsFavouringA == 0 || roundsFavouringB == 0);
    }
};

/// @brief Builds the paired summary from two ALIGNED sample vectors: `sa[r]`
/// and `sb[r]` are arm A's and arm B's readings from the SAME round.
/// @note Pure arithmetic, so a test can hand it invented rounds and check every
/// statistic against a value computed by hand. The timing code is what needs
/// a device; this is what needs proving.
/// @note A round whose arm-A reading is not positive yields no usable ratio --
/// a clock that read zero is not a measurement of anything. Such a round is
/// counted as a tie and left out of the geometric mean, rather than
/// contributing a 0.0 that would drag both the mean and the range.
inline PairedTiming summarizePaired(const std::vector<double>& sa,
                                    const std::vector<double>& sb) {
    PairedTiming p;
    const size_t n = std::min(sa.size(), sb.size());
    const auto cut = static_cast<std::ptrdiff_t>(n);

    std::vector<double> ratios;
    ratios.reserve(n);
    double logSum = 0.0;
    int logCount = 0;
    for (size_t r = 0; r < n; ++r) {
        const double ta = sa[r];
        const double tb = sb[r];
        const double ratio = ta > 0.0 ? tb / ta : 0.0;
        if (ratio > 0.0 && std::isfinite(ratio)) {
            ratios.push_back(ratio);
            logSum += std::log(ratio);
            ++logCount;
        }
        if (!(ta > 0.0) || !(tb > 0.0) || ta == tb) {
            ++p.roundsTied;
        } else if (tb > ta) {
            ++p.roundsFavouringA;
        } else {
            ++p.roundsFavouringB;
        }
    }

    p.a = summarize(std::vector<double>(sa.begin(), sa.begin() + cut));
    p.b = summarize(std::vector<double>(sb.begin(), sb.begin() + cut));
    const Timing rt = summarize(ratios);
    p.ratioMin = rt.minMs;
    p.ratioMedian = rt.medianMs;
    p.ratioMax = rt.maxMs;
    p.ratioGeoMean = logCount > 0 ? std::exp(logSum / static_cast<double>(logCount)) : 0.0;
    p.rounds = static_cast<int>(n);
    return p;
}

}  // namespace cudabench
