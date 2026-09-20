#pragma once

// GPU timing for the CUDA backend's benchmarks, under the same protocol as
// benchmark/measure_util.hpp: batches against a time budget, median reported
// with its spread, a warm-up batch discarded. Two clocks on purpose:
//
//   * KERNEL-RESIDENT time comes from CUDA events around the enqueued work --
//     what a resident pipeline pays per frame once the data lives on device.
//   * END-TO-END time is the host steady clock around enqueue + synchronize --
//     what a caller pays when the transfers are on the measured path.
//
// The two answer different questions and every table here says which one it
// prints. WSL2 inflates LAUNCH overhead specifically, so kernel-resident
// numbers travel better than end-to-end ones; both are honest on this host,
// and the spread is printed so a reader can see what a difference must clear.
//
// THREE MECHANISMS BEYOND THE SINGLE-ARM TIMER, each closing a gap that has
// already produced a misreadable number here:
//
//   * THE LAUNCH FLOOR -- measureLaunchFloor / printLaunchFloor. An empty
//     kernel, timed by this same protocol. A device op can be individually so
//     cheap that a microbenchmark ratio is measuring launch overhead rather
//     than the kernel: the binary matcher runs at 0.069 ms with 24-60% spread
//     on this host for exactly that reason. The floor is not a footnote beside
//     such a number, it is what makes it readable -- so print it.
//
//   * THE INTERLEAVED TWO-ARM TIMER -- timeKernelPaired. timeKernel runs one
//     arm to completion and then the other, which assigns any drift over the
//     run -- clock ramp, a thermal step, another process taking the GPU --
//     entirely to whichever ran second. That is the hazard measure_util.hpp
//     documents for the host, and a GPU under WSL2 has more of it, not less.
//     timeKernelPaired brackets BOTH arms inside every round and alternates
//     their order round to round, so a drift moves both arms' samples together.
//     Each round then yields one PAIRED observation of the ratio, which is a
//     better dataset than a pair of separately-measured medians. What is done
//     with those observations -- median, geometric mean, range, sign count,
//     and the difference-against-spread rule that decides -- lives in
//     paired_stats.hpp, next to the argument for each.
//
//   * THE NAMED MEMORY REPORTER -- DeviceMemMeter and the printMemory* family.
//     Three meters answer three different questions and they do not mix
//     (docs/reports/cuda.md, "one meter per comparison, named at the number"):
//     the allocation sum is what the arrays ask for, the cudaMemGetInfo delta
//     is what the driver reserves, and the read-back pitch is what a row
//     actually occupies. Every printer here stamps the meter's name onto the
//     line, and the driver meter prints its own MEASURED step next to its
//     reading: this driver reserves in 2 MB units, so a 442 KB working set
//     reads 0.00 MB or 2.00 MB depending on nothing but where the previous
//     allocation left the current unit. Both readings are the meter, neither
//     is the footprint, and dividing either by an allocation sum produces a
//     number that answers no question.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <vector>

#include <cuda_runtime.h>

// Timing, summarize, PairedTiming and the rule that decides a paired ratio.
// They are next door rather than here because they need no CUDA, which is what
// lets a test check them against hand-computed values on any machine.
#include "paired_stats.hpp"

namespace cudabench {

/// @brief Times `body` (which ENQUEUES device work on the default stream) with
/// CUDA events: per batch, one event pair brackets `iters` enqueues.
/// @note Single-arm. When the number that matters is a RATIO between two arms,
/// use timeKernelPaired instead -- this one runs its arm to completion, so a
/// second call to it is not drift-comparable with the first.
/// @param stream The stream the EVENTS are recorded on. Pass the same stream
/// the body enqueues onto. Defaulted to the legacy default stream, which is
/// what every caller that enqueues there wants and is a no-op change for
/// them -- but see the note, because the default is NOT safe for a body that
/// enqueues somewhere else.
/// @note AN EVENT RECORDED ON THE WRONG STREAM DOES NOT TIME THE WORK. Events
/// are ordered within their own stream. Recorded on the legacy default
/// stream while the body enqueues on stream S, the pair brackets the
/// default stream's implicit synchronization with S rather than S's work,
/// and the reading acquires the legacy stream's cross-blocking semantics --
/// measured here as 32-second outliers on a 5 ms kernel. When the body
/// takes a stream, pass it.
inline Timing timeKernel(const std::function<void()>& body, int iters = 20,
                         int repeats = 9, cudaStream_t stream = nullptr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    // Warm-up: first launches pay module load and clock ramp.
    for (int i = 0; i < iters; ++i) body();
    cudaDeviceSynchronize();
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(start, stream);
        for (int i = 0; i < iters; ++i) body();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(static_cast<double>(ms) / iters);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return summarize(std::move(samples));
}

// ---------------------------------------------------------------------------
// The interleaved two-arm timer
// ---------------------------------------------------------------------------

/// @brief Times two enqueueing arms with both bracketed inside every round.
/// @param bodyA,bodyB The arms. Each ENQUEUES on the default stream.
/// @param itersA,itersB Enqueues per batch, per arm. They are separate because
/// a reference arm can be 25x slower than the arm it prices, and forcing one
/// batch size on both either wastes a minute or times the fast arm against
/// the clock's resolution.
/// @param repeats Rounds. Each round contributes one sample to each arm and one
/// ratio.
/// @note ROUND ORDER ALTERNATES (A,B then B,A then A,B...). Bracketing alone
/// still hands whichever arm runs second half a round of drift every single
/// round; alternating cancels that to first order, which bracketing on its
/// own does not.
/// @param stream The stream the EVENTS are recorded on -- see timeKernel's note.
/// BOTH arms must enqueue onto this same stream, or the pair is not
/// comparable: one arm on the legacy default stream and the other on an
/// explicit stream do not merely run on different queues, they implicitly
/// serialize against each other, and each arm's bracket then contains part
/// of the other's work.
inline PairedTiming timeKernelPaired(const std::function<void()>& bodyA,
                                     const std::function<void()>& bodyB,
                                     int itersA = 20, int itersB = 20,
                                     int repeats = 9, cudaStream_t stream = nullptr) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const auto batch = [&](const std::function<void()>& body, int iters) {
        cudaEventRecord(start, stream);
        for (int i = 0; i < iters; ++i) body();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return static_cast<double>(ms) / iters;
    };

    // One discarded warm-up of each arm: module load, clock ramp, and -- for an
    // arm selected by a runtime switch -- the first pass through that branch.
    for (int i = 0; i < itersA; ++i) bodyA();
    for (int i = 0; i < itersB; ++i) bodyB();
    cudaDeviceSynchronize();

    // The two vectors stay ALIGNED BY ROUND -- sa[r] and sb[r] are the same
    // round's readings -- because that alignment is what makes the ratio a
    // paired observation. summarizePaired forms the ratios from it.
    std::vector<double> sa, sb;
    sa.reserve(static_cast<size_t>(repeats));
    sb.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        double ta = 0.0, tb = 0.0;
        if (r % 2 == 0) {
            ta = batch(bodyA, itersA);
            tb = batch(bodyB, itersB);
        } else {
            tb = batch(bodyB, itersB);
            ta = batch(bodyA, itersA);
        }
        sa.push_back(ta);
        sb.push_back(tb);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return summarizePaired(sa, sb);
}

/// @brief The same paired protocol on the HOST'S WALL CLOCK, for a pair where
/// one arm is not all device work.
/// @note CUDA events cannot time a round trip. An arm that downloads, refines
/// on the CPU and uploads again spends most of itself outside any stream, so
/// an event bracket around it measures the copies and nothing else. The
/// inequality such an arm exists to settle -- "is doing it on the device
/// cheaper than shipping it home and back" -- is a wall-clock question on
/// both sides, and this times it as one.
/// @param bodyA,bodyB The arms. Each must be SELF-CONTAINED: whatever it
/// enqueues, it also waits for, because the wall clock stops when the
/// calling thread returns and not when the device is finished.
/// @note Order alternates round to round, exactly as timeKernelPaired does and
/// for the same reason: bracketing alone still hands the arm that runs
/// second half a round of drift every round.
inline PairedTiming timeHostPaired(const std::function<void()>& bodyA,
                                   const std::function<void()>& bodyB, int itersA = 20,
                                   int itersB = 20, int repeats = 9) {
    const auto batch = [](const std::function<void()>& body, int iters) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) body();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    };
    for (int i = 0; i < itersA; ++i) bodyA();
    for (int i = 0; i < itersB; ++i) bodyB();
    cudaDeviceSynchronize();

    std::vector<double> sa, sb;
    sa.reserve(static_cast<size_t>(repeats));
    sb.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        double ta = 0.0, tb = 0.0;
        if (r % 2 == 0) {
            ta = batch(bodyA, itersA);
            tb = batch(bodyB, itersB);
        } else {
            tb = batch(bodyB, itersB);
            ta = batch(bodyA, itersA);
        }
        sa.push_back(ta);
        sb.push_back(tb);
    }
    return summarizePaired(sa, sb);
}

inline void printArm(const char* name, const Timing& t, const char* clock) {
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [%s]\n", name, t.medianMs,
                t.spreadPct(), clock);
}

/// @brief An arm with the launch floor's share of it stated on the same line.
/// @note An arm at or under half the floor gets a WORD rather than a
/// percentage. The two are measured in the same batched way but not in the
/// same batch, so at that point their difference is noise and a printed
/// "157%" invites a reader to take the ordering seriously. What is true and
/// worth saying is that the number is launch overhead, not the kernel.
inline void printArmVsFloor(const char* name, const Timing& t, const Timing& floor,
                            const char* clock) {
    const double share = t.medianMs > 0.0 ? floor.medianMs / t.medianMs : 0.0;
    if (share >= 0.5) {
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [%s]  AT THE LAUNCH FLOOR\n", name,
                    t.medianMs, t.spreadPct(), clock);
    } else {
        std::printf(" %-44s %9.3f ms  spread %4.0f%%  [%s]  (launch floor %.0f%% of it)\n",
                    name, t.medianMs, t.spreadPct(), clock, share * 100.0);
    }
}

/// @brief The three statistic lines a paired ratio is reported by, each one
/// labelled at its own value.
/// @note THE LABELS ARE THE POINT. A median, a geometric mean and a range are
/// three different numbers about the same rounds, and a reader who takes one
/// for another has been misled by the printer rather than by the data -- so
/// each says what it is, and the separation fact says that it is a fact
/// rather than the verdict.
inline void printPairedSignAndSeparation(const PairedTiming& p) {
    std::printf("   sign test over those paired rounds: %d favour A, %d favour B,"
                " %d tied -- two-sided p = %.3g\n",
                p.roundsFavouringA, p.roundsFavouringB, p.roundsTied, p.signTestP());
    std::printf("   separation (a FACT, not the verdict): the two arms' sample ranges"
                " %s\n",
                p.separated() ? "are DISJOINT" : "OVERLAP");
}

inline void printPairedStats(const PairedTiming& p) {
    std::printf("   ratio B/A, per-round paired:  median %5.2fx <- QUOTE THIS"
                "   geomean %5.2fx   range %.2f-%.2fx  (%d rounds)\n",
                p.ratioMedian, p.ratioGeoMean, p.ratioMin, p.ratioMax, p.rounds);
    printPairedSignAndSeparation(p);
}

/// @brief The verdict, in the three values paired_stats.hpp defines: whether
/// the DIRECTION is established by the rounds, whether the MAGNITUDE clears
/// the larger of the within-run spread and the run-to-run scatter, or
/// neither.
/// @note BOTH STATEMENTS ARE PRINTED WHENEVER BOTH HOLD, on their own lines.
/// They answer different questions -- "is it in doubt which arm is ahead"
/// and "is the distance bigger than the noise" -- and a row given only the
/// louder one has lost the other. The owner's 2026-09-19 ruling is the
/// reason the first exists: a spread that lies wholly on one side of 1.00x
/// bounds how much an arm wins by, not whether it wins.
/// @note A NULL RESULT IS PRINTED AS A RESULT, in those words, because
/// measure_util.hpp says so: "A difference smaller than the spread is a
/// null result, and a null result is a result." Two arms this run cannot
/// tell apart is a finding about them, not a missing measurement.
/// @note The lines state what decided each half, so that neither the
/// separation fact above nor the sign test beside it can be read as having
/// done so -- the p is printed precisely BECAUSE nothing gates on it, and a
/// reader needs it to tell two unanimous rounds from a hundred.
inline void printPairedVerdict(const PairedTiming& p) {
    const double scatter = runToRunScatterFactor();
    const bool measured = scatter >= 1.0;
    const bool result = p.differenceClearsNoise(scatter);
    const Verdict v = judgePaired(p, scatter);
    char scatterText[48];
    if (measured) {
        std::snprintf(scatterText, sizeof(scatterText), "%.2fx", scatter);
    } else {
        std::snprintf(scatterText, sizeof(scatterText), "NOT MEASURED on this host");
    }
    std::printf("   verdict: %s\n", verdictName(v));

    // THE DIRECTION HALF FIRST, because when it holds it is what the row is
    // quoted as: a range with a sign count, not one number.
    if (p.directionEstablished()) {
        const int wins = p.roundsFavouringA + p.roundsFavouringB;
        std::printf("            %s faster in %d of %d paired rounds -- NO ROUND CROSSED"
                    " 1.00x, so the\n"
                    "            direction is established by the rounds themselves"
                    " (two-sided sign-test\n"
                    "            p = %.3g). Magnitude %.2fx to %.2fx, median %.2fx: the"
                    " spread bounds\n"
                    "            HOW MUCH it wins by, not WHETHER it does (owner's"
                    " ruling, 2026-09-19).\n",
                    p.establishedDirection() == Direction::A ? "ARM A is" : "ARM B is",
                    wins, p.rounds, p.signTestP(), p.magnitudeLoFactor(),
                    p.magnitudeHiFactor(), p.differenceFactor());
    } else if (p.unanimous() && p.roundsTied > 0) {
        // Worth saying explicitly: this row is NOT a plain null, it is a row
        // that would have established a direction but for rounds in which
        // neither arm won. A reader who is not told that reads "null" as
        // "the arms split", which these rounds did not do.
        std::printf("            every USABLE round fell the same way (%d-%d, p = %.3g),"
                    " but %d of %d rounds\n"
                    "            favoured neither%s -- so NOT every round fell that way,"
                    " and the direction\n"
                    "            is not established. A tie is a round the winning arm did"
                    " not win.\n",
                    p.roundsFavouringA, p.roundsFavouringB, p.signTestP(), p.roundsTied,
                    p.rounds,
                    p.roundsUnusable > 0 ? " (some were not measurements at all)" : "");
    }

    // THE NOISE HALF, unchanged in every particular -- same predicate, same
    // quantities, same words. It is printed for every row, including the ones
    // the direction verdict already carried, because the two say different
    // things and the stronger-sounding one does not contain the other.
    std::printf("            %s the two arms are %.2fx apart on the MEDIAN per-round"
                " ratio,\n"
                "            %s the %.2fx it has to beat (the larger of: per-round swing"
                " %.2fx,\n"
                "            run-to-run scatter %s). Decided by measure_util.hpp's\n"
                "            difference-against-spread rule, in FACTORS because a"
                " percentage of it\n"
                "            depends on which arm is the denominator; range separation is"
                " not what\n"
                "            decided it.\n",
                result ? "A RESULT:" : "NULL on the magnitude:", p.differenceFactor(),
                result ? "clearing" : "short of", p.noiseToClearFactor(scatter),
                p.ratioSwingFactor(), scatterText);
    if (result && !measured) {
        std::printf("            That clears the WITHIN-RUN half of the rule only --"
                    " nobody has measured\n"
                    "            this host's run-to-run scatter, and the rule wants the"
                    " larger of the two.\n");
    }
}

/// @brief Both arms, then the per-round ratio's statistics and the verdict.
/// @param expect1x Set when this pair is a GATE-EXCLUDED control -- a case the
/// fast path's own gate rejects, which must therefore read ~1.00x. The
/// verdict then checks for 1.00x instead, because here "the two arms are
/// indistinguishable" is the PASS. Two things that would contradict it get
/// flagged: disjoint ranges, and a unanimous sign count. Identical code
/// scatters both ways, so fifteen rounds falling the same way is a finding
/// about the control even when its median reads 1.00x -- which is a check
/// the range test could not express at all.
/// @brief One machine-readable line per paired comparison, keyed by the two arm
/// names, carrying everything the rule needs.
/// @note WHY IT IS HERE AND NOT AT THE CALL SITES. The run-to-run half of
/// measure_util.hpp's rule cannot be seen from inside one process: it is the
/// scatter of a benchmark's MEDIANS across several. Reading it means
/// aggregating many runs, and aggregating means parsing -- which until now
/// only the two benchmarks that hand-rolled a ROW line could support, while
/// the other families printed prose that nothing could total. Emitting from
/// printPaired gives every pair that goes through it the same row for free,
/// so the stronger half of the rule is reachable for all of them rather than
/// for two.
/// @note The FACTORS are emitted, not the percentages, because they are what
/// decides -- see paired_stats.hpp on why a percentage of a ratio depends on
/// which arm is the denominator.
/// @brief A label the emitted rows carry, naming what the CURRENT pairs are
/// being measured on -- normally the frame geometry.
/// @note IT IS NOT DECORATION. A benchmark that sweeps two frame sizes prints
/// the same two arm names at each of them, so without this the aggregation
/// pools 752x480 with 3840x2160 under one key and reads the difference
/// between the geometries as run-to-run scatter. Measured: pooling the
/// packer's three geometries put the run-to-run factor at 3.18x where each
/// geometry on its own is between 1.16x and 1.34x, which turned every row
/// in that family into a null. A sweep sets this at the top of each pass;
/// a benchmark with one geometry can leave it empty.
inline const char*& pairedScope() {
    static const char* scope = "";
    return scope;
}

inline void emitPairedRow(const char* nameA, const char* nameB, const PairedTiming& p) {
    // FIELDS ARE APPENDED, NEVER REORDERED. scripts/aggregate_cuda_runs.py
    // indexes this line positionally and reads run files older than the
    // column it is looking for, so a row that grew is readable by both the
    // old parser and the new one while a row that moved is readable by
    // neither. The four at the end carry the direction verdict across
    // processes: unanimity over a whole sweep is 105 rounds from seven runs,
    // and no single process can see it.
    std::printf("PAIRED|%s|%s|%s|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f"
                "|%d|%d|%d|%d|%.6f|%.6f|%.4g|%d|%d|%d|%.6f|%.6f\n",
                pairedScope(), nameA, nameB, p.a.minMs, p.a.medianMs, p.a.maxMs,
                p.b.minMs, p.b.medianMs, p.b.maxMs, p.ratioMin, p.ratioMedian,
                p.ratioMax, p.ratioGeoMean,
                p.rounds, p.roundsFavouringA, p.roundsFavouringB, p.roundsTied,
                p.differenceFactor(), p.ratioSwingFactor(), p.signTestP(),
                p.separated() ? 1 : 0,
                p.roundsUnusable, p.directionEstablished() ? 1 : 0,
                p.magnitudeLoFactor(), p.magnitudeHiFactor());
}

inline void printPaired(const char* nameA, const char* nameB, const PairedTiming& p,
                        const char* clock, bool expect1x = false) {
    printArm(nameA, p.a, clock);
    printArm(nameB, p.b, clock);
    printPairedStats(p);
    emitPairedRow(nameA, nameB, p);
    if (expect1x) {
        const bool ok = p.ratioMedian > 0.95 && p.ratioMedian < 1.05;
        std::printf("   verdict: %s\n",
                    ok ? "~1.00x as required -- the switch selects nothing here, which is"
                         " what the gate promises"
                       : "NOT ~1.00x -- the gate is not excluding what it claims to");
        if (p.separated()) {
            std::printf("            (but the two ranges are DISJOINT, which identical code"
                        " should not be -- look again)\n");
        }
        if (p.unanimous()) {
            std::printf("            (and every usable round fell the same way, p = %.3g."
                        " Identical code\n"
                        "             scatters both ways, so look again even though the"
                        " median reads 1.00x)\n",
                        p.signTestP());
        }
    } else {
        printPairedVerdict(p);
    }
}

// ---------------------------------------------------------------------------
// The launch floor
// ---------------------------------------------------------------------------

/// @brief Enqueues a kernel with an empty body. **Defined in
/// cuda_bench_null.cu**, because a `__global__` needs nvcc and these benchmark
/// translation units are host C++.
/// @param grid,block The launch shape. The default is the smallest one that
/// exists -- one block of one warp -- so the number is launch and teardown
/// with no work and no scheduling behind it.
void launchNullKernel(dim3 grid = dim3(1), dim3 block = dim3(32),
                      cudaStream_t stream = nullptr);

/// @brief The floor every kernel-resident number on this machine stands on.
/// @note A LONGER WARM-UP than timeKernel's single batch, and for a reason worth
/// stating. This is normally the first thing a benchmark runs, on an idle
/// device, so its samples otherwise straddle the driver's and the device's
/// ramp: measured here, a single-batch warm-up gave the floor a 150-190%
/// spread against ~30% once the ramp finishes before the first timed batch --
/// and the arms that get compared against the floor are all measured after
/// the device is already warm, so a cold floor is not even the same quantity.
/// @note `iters` defaults to a batch size of the same order the arms here use,
/// not to the largest batch that would minimize the number: back-to-back
/// launches pipeline, so a floor measured over 5,000 enqueues is a different
/// quantity from an arm measured over 100 (swept on this host: 0.0076 ms at
/// 100, 0.0081 at 1,000, 0.0085 at 5,000).
/// @param stream The stream the floor is measured on. A floor compared against
/// arms timed on an explicit stream has to be measured on that same stream,
/// for timeKernel's reason.
inline Timing measureLaunchFloor(dim3 grid = dim3(1), dim3 block = dim3(32),
                                 int iters = 100, int repeats = 25,
                                 double warmupMs = 250.0,
                                 cudaStream_t stream = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        for (int i = 0; i < 500; ++i) launchNullKernel(grid, block, stream);
        cudaDeviceSynchronize();
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= warmupMs) break;
    }
    return timeKernel([&] { launchNullKernel(grid, block, stream); }, iters, repeats,
                      stream);
}

inline void printLaunchFloor(const Timing& t) {
    std::printf(" %-44s %9.4f ms  spread %4.0f%%  [kernel]\n",
                "LAUNCH FLOOR (empty kernel, same protocol)", t.medianMs,
                t.spreadPct());
    std::printf("   Every kernel-resident number below sits on top of this. An arm within\n"
                "   a few multiples of it is reporting launch overhead, not its own work,\n"
                "   and a RATIO between two such arms is reporting almost none of either.\n"
                "   The spread is part of the answer, not a blemish on it: under WSL2 a\n"
                "   launch costs what it costs plus whatever the host was doing.\n");
}

// ---------------------------------------------------------------------------
// The memory meters -- three of them, and they do not mix
// ---------------------------------------------------------------------------

/// @brief METER 2: the driver-side reading, a `cudaMemGetInfo` free-bytes delta
/// across a scope. It is the only meter readable on BOTH sides of a
/// cross-library comparison, which is why every figure here that crosses to
/// OpenCV uses it -- and why nothing that stays inside binCV should, since it
/// has a granularity far larger than a small working set (see
/// measureDriverMeterStep).
class DeviceMemMeter {
public:
    DeviceMemMeter() { reset(); }

    /// @brief Takes the baseline reading. Synchronizes first, because a pending
    /// free has not happened yet as far as this meter is concerned.
    void reset() {
        cudaDeviceSynchronize();
        base_ = freeBytes();
    }

    /// @brief Bytes the driver has taken since `reset()`, clamped at zero: a
    /// concurrent process releasing memory can make the reading go the other
    /// way, and a negative footprint is not a thing.
    size_t deltaBytes() const {
        cudaDeviceSynchronize();
        const size_t f = freeBytes();
        return base_ > f ? base_ - f : 0;
    }

    static size_t freeBytes() {
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
        return f;
    }

private:
    size_t base_ = 0;
};

/// @brief The driver meter's STEP: the size of the unit it reserves in,
/// measured here by allocating one byte at a time until the reading moves.
/// @return The size of the first jump, or 0 if it could not be provoked.
/// @note On this backend's reference driver the answer is 2 MB, and that single
/// fact is what makes every small reading unreadable on its own. A ONE-BYTE
/// allocation reads 2.00 MB when it happens to start a fresh unit and 0.00
/// MB when it fits the unit the last allocation was using -- SAME allocation,
/// two readings, neither of them a footprint. Measuring the step rather than
/// a single probe is what makes the caveat independent of where in the run it
/// is taken, which a one-byte probe is not: run late in a benchmark it reads
/// zero and would look like a meter with no granularity at all.
inline size_t measureDriverMeterStep(int maxProbes = 8192) {
    cudaDeviceSynchronize();
    const size_t base = DeviceMemMeter::freeBytes();
    std::vector<void*> blocks;
    size_t step = 0;
    for (int i = 0; i < maxProbes; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, 1) != cudaSuccess) break;
        blocks.push_back(p);
        const size_t f = DeviceMemMeter::freeBytes();
        if (f < base) {
            step = base - f;
            break;
        }
    }
    for (void* p : blocks) cudaFree(p);
    return step;
}

inline void printMemoryHeader(const char* what) {
    std::printf(" MEMORY, %s -- three meters, each named at its own number.\n"
                " They answer different questions and a ratio across two of them answers\n"
                " none (docs/reports/cuda.md: one meter per comparison).\n",
                what);
}

/// @brief METER 1: the allocation sum -- what the arrays ask for. The meter for
/// binCV against binCV, and against the cost volume the design refuses.
inline void printAllocSum(const char* what, size_t bytes) {
    std::printf("   [meter 1: allocation sum ] %-32s %9.1f KB\n", what,
                static_cast<double>(bytes) / 1024.0);
}

/// @brief METER 2's line, printed with the meter's own measured step beside it
/// so that its granularity cannot be read as a footprint.
inline void printDriverDelta(const char* what, size_t deltaBytes, size_t stepBytes) {
    const double stepMB = static_cast<double>(stepBytes) / (1024.0 * 1024.0);
    const char* pad = "                             ";
    std::printf("   [meter 2: cudaMemGetInfo] %-32s %9.2f MB\n", what,
                static_cast<double>(deltaBytes) / (1024.0 * 1024.0));
    if (stepBytes == 0) {
        std::printf("   [meter 2: its own STEP  ] could not be provoked here;\n"
                    "%streat the reading above as approximate.\n", pad);
        return;
    }
    std::printf("   [meter 2: its own STEP  ] this driver reserves in %.2f MB units,\n"
                "%smeasured right here by allocating one byte at a\n"
                "%stime until the reading moved.\n",
                stepMB, pad, pad);
    if (deltaBytes <= stepBytes) {
        std::printf("%sThis working set fits inside ONE unit, so the\n"
                    "%sreading above is the meter's RESOLUTION and not\n"
                    "%sa footprint: the identical arrays read 0.00 MB\n"
                    "%sor %.2f MB depending only on how much of the\n"
                    "%scurrent unit the previous allocation left over.\n"
                    "%sMeter 1 is what measures this working set.\n"
                    "%sNever divide one meter by the other.\n",
                    pad, pad, pad, pad, stepMB, pad, pad, pad);
    } else {
        std::printf("%sThat is %.0f units, so the reading is the working\n"
                    "%sset ROUNDED UP to a unit -- an upper bound, and\n"
                    "%sstill not divisible by meter 1.\n",
                    pad, static_cast<double>(deltaBytes) / static_cast<double>(stepBytes),
                    pad, pad);
    }
}

/// @brief METER 3: the row pitch READ BACK from the container that owns the
/// array, times its rows. This is what separates "width x height x element"
/// from what the allocation actually lays out, and it is the only one of the
/// three that can show a padded row.
inline void printPitch(const char* array, size_t pitchBytes, size_t rows,
                       size_t naturalRowBytes) {
    std::printf("   [meter 3: read-back pitch] %-18s %5zu B/row x %4zu = %8.1f KB",
                array, pitchBytes, rows,
                static_cast<double>(pitchBytes * rows) / 1024.0);
    if (pitchBytes > naturalRowBytes)
        std::printf("  (+%zu B/row padding)", pitchBytes - naturalRowBytes);
    std::printf("\n");
}

inline void printDevice() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    std::printf(" device: %s (sm_%d%d, %d SMs, %.0f GB/s peak)\n", p.name, p.major,
                p.minor, p.multiProcessorCount,
                2.0 * p.memoryClockRate * (p.memoryBusWidth / 8.0) / 1.0e6);
}

} // namespace cudabench
