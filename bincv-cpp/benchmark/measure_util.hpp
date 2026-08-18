#pragma once

// Timing harness shared by the three Phase 2 experiment benchmarks
// (alignment_benchmark.cpp / E-1, wordwidth_benchmark.cpp / E-2,
// window_benchmark.cpp / E-3).
//
// It is a separate header from bench_util.hpp on purpose: bench_util.hpp is the
// binCV-versus-OpenCV harness and pulls in <opencv2/opencv.hpp>, whereas every one
// of these three experiments compares binCV against binCV. No OpenCV denominator
// applies to "which of our own alternatives is faster", and dragging the
// dependency into these binaries would only make them harder to reason about.
//
// WHAT THE PROTOCOL REQUIRES OF A NUMBER PRINTED HERE (EXPERIMENTS.md "Rules"):
//
//   1. A volatile sink consumes every result, so no measured loop is dead code.
//   2. Inputs rotate through several distinct random images, so nothing constant
//      folds.
//   3. Batches are calibrated to a time budget, then repeated -- and the SPREAD
//      across repeats is reported next to the central value. A difference smaller
//      than the spread is a null result, and a null result is a result.
//   4. Whatever is being compared must AGREE before it is timed. That check lives
//      in the individual benchmarks, since only they know what agreement means.
//
// WHY VARIANTS ARE INTERLEAVED RATHER THAN RUN ONE AFTER THE OTHER
//
// Measuring variant A to completion and then variant B assigns any slow drift
// over the run -- clock ramp, a migrating background process, cache state left by
// the previous variant -- entirely to whichever ran later. These experiments are
// deciding 5% and 15% questions, which is the same order as that drift. So
// measureInterleaved() calibrates each variant once and then runs ONE batch of
// each per round, round-robin, so a drift moves every variant's samples together
// instead of moving the comparison. The reported spread is then an honest bound
// on what a difference has to exceed to be real.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace measure {

using Clock = std::chrono::steady_clock;

/// @brief The volatile sink. Every benchmark result is folded into it and it is
///        printed at exit, so no timed loop can be deleted as dead code.
inline volatile size_t g_sink = 0;

/// @brief splitmix64, so each benchmark generates its own inputs deterministically
///        -- a re-run is the same run, and four seeds give four distinct images.
inline uint64_t nextRandom(uint64_t& state) {
    state += UINT64_C(0x9E3779B97F4A7C15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/// @brief One measurement: the central value AND what the run-to-run scatter was.
/// @note `medianNs` is what ratios are taken on -- it is not moved by a single
///       descheduling event the way a mean is, and unlike the minimum it does not
///       report the luckiest batch as though it were typical. `minNs` and `maxNs`
///       are printed with it so a reader can see whether a reported difference is
///       larger than the noise it was measured against.
struct Timing {
    double minNs = 0.0;     ///< fastest batch, ns per body() call
    double medianNs = 0.0;  ///< median batch -- the value ratios use
    double maxNs = 0.0;     ///< slowest batch
    int iterations = 0;     ///< body() calls per batch
    int repeats = 0;        ///< batches

    /// @brief Full scatter as a percentage of the median: (max - min) / median.
    double spreadPct() const {
        return medianNs > 0.0 ? (maxNs - minNs) / medianNs * 100.0 : 0.0;
    }
};

/// @brief A named variant to time. The body takes the iteration index so that it
///        can rotate over several inputs (validity hazard 2).
struct Bench {
    std::string name;
    std::function<void(int)> body;
};

/// @brief Chooses a batch size that runs for about `targetMs`, so that a clock
///        tick and the call overhead are both negligible against the batch.
inline int calibrate(const std::function<void(int)>& body, double targetMs) {
    int iterations = 1;
    for (int attempt = 0; attempt < 24; ++attempt) {
        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) body(i);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (ms >= targetMs || iterations >= (1 << 22)) return iterations;
        const double scale = (ms > 0.0) ? (targetMs / ms) : 8.0;
        iterations = static_cast<int>(static_cast<double>(iterations) *
                                      std::min(scale * 1.3, 16.0)) + 1;
    }
    return iterations;
}

/// @brief Times every variant against every other, round-robin. See the header
///        comment for why the interleaving is the point.
/// @param benches The variants being compared. All of them, in one call.
/// @param repeats Batches per variant. Every variant gets the same number.
/// @param targetMs Time budget for one batch.
inline std::vector<Timing> measureInterleaved(const std::vector<Bench>& benches, int repeats,
                                              double targetMs) {
    const size_t n = benches.size();
    std::vector<int> iterations(n, 0);
    std::vector<std::vector<double>> samples(n);

    for (size_t b = 0; b < n; ++b) {
        iterations[b] = calibrate(benches[b].body, targetMs);
        samples[b].reserve(static_cast<size_t>(repeats));
    }

    // One discarded warm-up round, so that the first recorded batch of the first
    // variant is not the only one paying for a cold cache and a cold branch
    // predictor.
    for (size_t b = 0; b < n; ++b) {
        for (int i = 0; i < iterations[b]; ++i) benches[b].body(i);
    }

    for (int r = 0; r < repeats; ++r) {
        for (size_t b = 0; b < n; ++b) {
            const auto start = Clock::now();
            for (int i = 0; i < iterations[b]; ++i) benches[b].body(i);
            const double ns =
                std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
                static_cast<double>(iterations[b]);
            samples[b].push_back(ns);
        }
    }

    std::vector<Timing> out(n);
    for (size_t b = 0; b < n; ++b) {
        std::sort(samples[b].begin(), samples[b].end());
        Timing t;
        t.iterations = iterations[b];
        t.repeats = repeats;
        t.minNs = samples[b].front();
        t.maxNs = samples[b].back();
        const size_t m = samples[b].size();
        t.medianNs = (m % 2 == 1) ? samples[b][m / 2]
                                  : 0.5 * (samples[b][m / 2 - 1] + samples[b][m / 2]);
        out[b] = t;
    }
    return out;
}

}  // namespace measure
