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
//     It reports each arm's own median AND the per-round RATIO distribution,
//     because a ratio whose two sample ranges overlap is not a result -- the
//     printer says which it is rather than leaving it to the reader.
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

namespace cudabench {

struct Timing {
    double minMs = 0.0;
    double medianMs = 0.0;
    double maxMs = 0.0;
    double spreadPct() const {
        return medianMs > 0.0 ? (maxMs - minMs) / medianMs * 100.0 : 0.0;
    }
};

/// @brief min / median / max of a sample set, sorted in place.
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

/// @brief Times `body` (which ENQUEUES device work on the default stream) with
/// CUDA events: per batch, one event pair brackets `iters` enqueues.
/// @note Single-arm. When the number that matters is a RATIO between two arms,
/// use timeKernelPaired instead -- this one runs its arm to completion, so a
/// second call to it is not drift-comparable with the first.
inline Timing timeKernel(const std::function<void()>& body, int iters = 20,
                         int repeats = 9) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    // Warm-up: first launches pay module load and clock ramp.
    for (int i = 0; i < iters; ++i) body();
    cudaDeviceSynchronize();
    for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) body();
        cudaEventRecord(stop);
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

/// @brief Two arms measured against each other, plus the ratio's own scatter.
/// @note `ratio*` are the distribution of B/A computed WITHIN each round, not
/// a ratio of the two medians. The distinction is the whole point: a
/// per-round ratio cancels drift that moved both arms, whereas a ratio of
/// separately-measured medians carries the drift between them.
struct PairedTiming {
    Timing a;                  ///< arm A, its own min/median/max
    Timing b;                  ///< arm B, same
    double ratioMin = 0.0;     ///< smallest per-round B/A
    double ratioMedian = 0.0;  ///< median per-round B/A -- the value to quote
    double ratioMax = 0.0;     ///< largest per-round B/A
    int rounds = 0;

    /// @brief Whether the two arms' sample RANGES are disjoint. When they are
    /// not, the arms are not distinguishable at this sample size and the
    /// ratio is not a result, however far from 1.00x its median sits.
    bool separated() const { return a.maxMs < b.minMs || b.maxMs < a.minMs; }
};

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
inline PairedTiming timeKernelPaired(const std::function<void()>& bodyA,
                                     const std::function<void()>& bodyB,
                                     int itersA = 20, int itersB = 20,
                                     int repeats = 9) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const auto batch = [&](const std::function<void()>& body, int iters) {
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) body();
        cudaEventRecord(stop);
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

    std::vector<double> sa, sb, ratios;
    sa.reserve(static_cast<size_t>(repeats));
    sb.reserve(static_cast<size_t>(repeats));
    ratios.reserve(static_cast<size_t>(repeats));
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
        ratios.push_back(ta > 0.0 ? tb / ta : 0.0);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    PairedTiming p;
    p.a = summarize(std::move(sa));
    p.b = summarize(std::move(sb));
    const Timing rt = summarize(std::move(ratios));
    p.ratioMin = rt.minMs;
    p.ratioMedian = rt.medianMs;
    p.ratioMax = rt.maxMs;
    p.rounds = repeats;
    return p;
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

/// @brief Both arms, then the per-round ratio with its range and its verdict.
/// @param expect1x Set when this pair is a GATE-EXCLUDED control -- a case the
/// fast path's own gate rejects, which must therefore read ~1.00x. The
/// verdict then checks for 1.00x instead of for separation, because here
/// overlapping ranges are the PASS.
inline void printPaired(const char* nameA, const char* nameB, const PairedTiming& p,
                        const char* clock, bool expect1x = false) {
    printArm(nameA, p.a, clock);
    printArm(nameB, p.b, clock);
    std::printf("   ratio B/A, interleaved rounds: %5.2fx   per-round range %5.2f-%5.2fx"
                " (%d rounds)\n",
                p.ratioMedian, p.ratioMin, p.ratioMax, p.rounds);
    if (expect1x) {
        const bool ok = p.ratioMedian > 0.95 && p.ratioMedian < 1.05;
        std::printf("   verdict: %s%s\n",
                    ok ? "~1.00x as required -- the switch selects nothing here, which is"
                         " what the gate promises"
                       : "NOT ~1.00x -- the gate is not excluding what it claims to",
                    p.separated()
                        ? "\n            (but the two ranges are DISJOINT, which identical"
                          " code should not be -- look again)"
                        : "");
    } else {
        std::printf("   verdict: %s\n",
                    p.separated()
                        ? "sample ranges are DISJOINT -- the ratio is a result"
                        : "sample ranges OVERLAP -- not a result at this sample size");
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
inline Timing measureLaunchFloor(dim3 grid = dim3(1), dim3 block = dim3(32),
                                 int iters = 100, int repeats = 25,
                                 double warmupMs = 250.0) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        for (int i = 0; i < 500; ++i) launchNullKernel(grid, block);
        cudaDeviceSynchronize();
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= warmupMs) break;
    }
    return timeKernel([&] { launchNullKernel(grid, block); }, iters, repeats);
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
