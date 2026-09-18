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

#include <algorithm>
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

/// @brief Times `body` (which ENQUEUES device work on the default stream) with
/// CUDA events: per batch, one event pair brackets `iters` enqueues.
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
    std::sort(samples.begin(), samples.end());
    Timing t;
    t.minMs = samples.front();
    t.maxMs = samples.back();
    const size_t m = samples.size();
    t.medianMs = (m % 2 == 1) ? samples[m / 2]
                              : 0.5 * (samples[m / 2 - 1] + samples[m / 2]);
    return t;
}

inline void printArm(const char* name, const Timing& t, const char* clock) {
    std::printf(" %-44s %9.3f ms  spread %4.0f%%  [%s]\n", name, t.medianMs,
                t.spreadPct(), clock);
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
