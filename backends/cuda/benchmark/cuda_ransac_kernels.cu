// The device side of the RANSAC-geometry question. See cuda_ransac_kernels.hpp
// for what this is and, more importantly, what it is not.
//
// THREE THINGS ARE DELIBERATE HERE.
//
// * The solver is the HOST library's, compiled for the device. Every hazard a
//   hand-written device elimination would carry -- the in-place Gauss-Seidel
//   Aberth iterate, the pivot tie rule that takes the lowest row on a tie, the
//   phase-5 compaction that makes model index m a running count of ACCEPTED
//   roots rather than a root index -- is not reproduced, so none of them can
//   drift. The reduction key's tie rule depends on that model ordering, so a
//   fork would have had to get it right and keep it right.
//
// * The traversal is forked, which is the only part that should be. Lane
//   mapping over correspondences is where a warp and a row loop genuinely
//   disagree.
//
// * The packed key carries no model. `(support << 24) | (0xFFFFFF - index)`
//   under atomicMax is max-support-then-lowest-index, which is exactly what the
//   host driver's `if (support <= bestInliers) continue` means, and it is
//   deterministic under any block schedule. A (key, model) pair would need a
//   lock; re-solving the one winning sample afterwards costs 1/H of a round.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/error.hpp"
#include "bincv/ops/essential.hpp"
#include "bincv/ops/ransac.hpp"

#include "cuda_ransac_kernels.hpp"

namespace cudabench {
namespace ransacprobe {
namespace {

using bincv::EssentialMatrix;
using bincv::EssentialModel;
using bincv::Point2f;

constexpr unsigned kBlock = 128;
constexpr unsigned kWarp = 32;
constexpr size_t kMaxModels = EssentialModel::kMaxModels;
constexpr size_t kMinimalSet = EssentialModel::kMinimalSetSize;

/// @brief The pool iteration `iter` samples from, which is the host driver's own
/// rule: the whole set, or a prefix that starts at the minimal set and grows
/// by one per iteration.
__device__ inline size_t samplingPool(size_t count, uint64_t iter, bool ordered) {
    if (!ordered) return count;
    const size_t grown = kMinimalSet + static_cast<size_t>(iter);
    return grown < count ? grown : count;
}

/// @brief The packed reduction key. Both halves are widened explicitly: the
/// shift would otherwise happen at `unsigned` and lose every support above
/// 255, and -Wconversion is the diagnostic this project leans on hardest.
__device__ inline unsigned long long packKey(unsigned support, size_t globalModelIndex) {
    return (static_cast<unsigned long long>(support) << 24) |
           (0xFFFFFFull - static_cast<unsigned long long>(globalModelIndex));
}

/// @brief Support of one model, every correspondence tested by this thread.
__device__ inline unsigned scoreSerial(const EssentialMatrix& e, const Point2f* from,
                                       const Point2f* to, size_t count, float threshold) {
    unsigned support = 0;
    for (size_t i = 0; i < count; ++i) {
        if (EssentialModel::residual(e, from[i], to[i]) < threshold) ++support;
    }
    return support;
}

// ---------------------------------------------------------------------------
// The round kernels
// ---------------------------------------------------------------------------

/// @brief One hypothesis per THREAD, scored serially within it.
__global__ void kRoundSerial(const Point2f* from, const Point2f* to, size_t count,
                             uint64_t seed, uint64_t iterBase, unsigned hypotheses,
                             float threshold, unsigned long long* bestKey) {
    const unsigned h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hypotheses) return;

    const uint64_t iter = iterBase + static_cast<uint64_t>(h);
    const uint64_t counter = seed + iter * UINT64_C(0x9E3779B9);
    size_t idx[kMinimalSet];
    if (!bincv::impl::ransacSample(count, kMinimalSet, counter, idx)) return;

    EssentialMatrix models[kMaxModels];
    const size_t produced = EssentialModel::estimate(from, to, idx, models);
    for (size_t m = 0; m < produced; ++m) {
        const unsigned support = scoreSerial(models[m], from, to, count, threshold);
        // Zero support never writes, so a key left at 0 means "no model found" --
        // the host's bestInliers starting at 0 and requiring strictly more.
        if (support == 0) continue;
        atomicMax(bestKey, packKey(support, static_cast<size_t>(h) * kMaxModels + m));
    }
}

/// @brief One hypothesis per thread, scored by the WARP: lane `l` tests
/// correspondences `l, l+32, ...` and `__ballot_sync` packs 32 flags into one
/// word of the host's own flag layout.
/// @note The models differ per lane, so the ballot cannot be shared -- each lane
/// needs the support of ITS OWN model. What the warp shares is the sweep over
/// `from`/`to`, which every lane reads identically. This is the mapping the
/// arm switch selects, and it is the honest version of "the ballot is what a
/// warp is for": here the ballot is a broadcast-read optimization, not a
/// packing win, because nothing in this family is a pixel.
__global__ void kRoundWarpBallot(const Point2f* from, const Point2f* to, size_t count,
                                 uint64_t seed, uint64_t iterBase, unsigned hypotheses,
                                 float threshold, unsigned long long* bestKey) {
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const unsigned lane = threadIdx.x & (kWarp - 1u);
    if (warp >= hypotheses) return;

    // EssentialMatrix carries default member initializers, which __shared__
    // forbids; the storage is raw doubles and the model is read back through one.
    __shared__ double sModels[kBlock / kWarp][kMaxModels][9];
    __shared__ unsigned sProduced[kBlock / kWarp];
    const unsigned w = threadIdx.x / kWarp;

    if (lane == 0) {
        const uint64_t iter = iterBase + static_cast<uint64_t>(warp);
        const uint64_t counter = seed + iter * UINT64_C(0x9E3779B9);
        size_t idx[kMinimalSet];
        if (!bincv::impl::ransacSample(count, kMinimalSet, counter, idx)) {
            sProduced[w] = 0u;
        } else {
            EssentialMatrix local[kMaxModels];
            const size_t n = EssentialModel::estimate(from, to, idx, local);
            for (size_t m = 0; m < n; ++m) {
                for (int j = 0; j < 9; ++j) sModels[w][m][j] = local[m].m[j];
            }
            sProduced[w] = static_cast<unsigned>(n);
        }
    }
    __syncwarp();

    const unsigned produced = sProduced[w];
    for (unsigned m = 0; m < produced; ++m) {
        EssentialMatrix e;
        for (int j = 0; j < 9; ++j) e.m[j] = sModels[w][m][j];
        unsigned support = 0;
        for (size_t base = 0; base < count; base += kWarp) {
            const size_t i = base + lane;
            // The i < count term is what keeps the flag word's bits past count
            // zero -- this library's padding rule, applied to the flag scratch.
            const bool in =
                (i < count) && (EssentialModel::residual(e, from[i], to[i]) < threshold);
            support += static_cast<unsigned>(__popc(__ballot_sync(0xFFFFFFFFu, in)));
        }
        if (lane == 0 && support != 0u) {
            atomicMax(bestKey,
                      packKey(support, static_cast<size_t>(warp) * kMaxModels + m));
        }
    }
}

/// @brief One sample per WARP, solver serial in lane 0, 31 lanes idle. The
/// ceiling probe -- see the header.
__global__ void kRoundOneSamplePerWarp(const Point2f* from, const Point2f* to,
                                       size_t count, uint64_t seed, uint64_t iterBase,
                                       unsigned hypotheses, float threshold,
                                       unsigned long long* bestKey) {
    const unsigned warp = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const unsigned lane = threadIdx.x & (kWarp - 1u);
    if (warp >= hypotheses || lane != 0u) return;

    const uint64_t iter = iterBase + static_cast<uint64_t>(warp);
    const uint64_t counter = seed + iter * UINT64_C(0x9E3779B9);
    size_t idx[kMinimalSet];
    if (!bincv::impl::ransacSample(count, kMinimalSet, counter, idx)) return;

    EssentialMatrix models[kMaxModels];
    const size_t produced = EssentialModel::estimate(from, to, idx, models);
    for (size_t m = 0; m < produced; ++m) {
        const unsigned support = scoreSerial(models[m], from, to, count, threshold);
        if (support == 0) continue;
        atomicMax(bestKey, packKey(support, static_cast<size_t>(warp) * kMaxModels + m));
    }
}

/// @brief Everything one hypothesis produces, written out per sample.
__global__ void kProbeSamples(const Point2f* from, const Point2f* to, size_t count,
                              uint64_t seed, uint64_t iterBase, unsigned samples,
                              float threshold, bool ordered, size_t* outIdx,
                              uint32_t* outDrew, EssentialMatrix* outModels,
                              uint32_t* outProduced, uint32_t* outSupport,
                              uint32_t* outFlags) {
    const unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= samples) return;

    const size_t words = bincv::ransacScratchWords(count);
    const size_t flagBase = static_cast<size_t>(s) * kMaxModels * words;
    for (size_t k = 0; k < kMaxModels * words; ++k) outFlags[flagBase + k] = 0u;
    for (size_t m = 0; m < kMaxModels; ++m) {
        outSupport[static_cast<size_t>(s) * kMaxModels + m] = 0u;
    }
    outProduced[s] = 0u;

    const uint64_t iter = iterBase + static_cast<uint64_t>(s);
    const uint64_t counter = seed + iter * UINT64_C(0x9E3779B9);
    size_t* idx = outIdx + static_cast<size_t>(s) * kMinimalSet;
    for (size_t k = 0; k < kMinimalSet; ++k) idx[k] = 0;
    const size_t pool = samplingPool(count, iter, ordered);
    if (!bincv::impl::ransacSample(pool, kMinimalSet, counter, idx)) {
        outDrew[s] = 0u;
        return;
    }
    outDrew[s] = 1u;

    EssentialMatrix* models = outModels + static_cast<size_t>(s) * kMaxModels;
    const size_t produced = EssentialModel::estimate(from, to, idx, models);
    outProduced[s] = static_cast<uint32_t>(produced);

    for (size_t m = 0; m < produced; ++m) {
        uint32_t* flags = outFlags + flagBase + m * words;
        uint32_t support = 0;
        for (size_t i = 0; i < count; ++i) {
            if (EssentialModel::residual(models[m], from[i], to[i]) < threshold) {
                flags[i >> 5] |= static_cast<uint32_t>(1u) << (i & 31u);
                ++support;
            }
        }
        outSupport[static_cast<size_t>(s) * kMaxModels + m] = support;
    }
}

unsigned gridForThreads(unsigned threads) {
    return (threads + kBlock - 1u) / kBlock;
}

/// @brief What every launcher here checks before it launches.
/// @note The key packs the global model index into 24 bits and shifts support
/// above it, so both have a domain. A device op that narrows its domain has
/// to NAME it, assert it and refuse outside it rather than wrap silently.
cudaError_t validate(const Point2f* dFrom, const Point2f* dTo, size_t count,
                     unsigned hypotheses, float threshold,
                     const unsigned long long* dBestKey) {
    BINCV_ASSERT(count < kMaxCount,
                 "ransac probe: the packed key carries support in the bits above 24, "
                 "so count must stay under 2^24");
    BINCV_ASSERT(static_cast<size_t>(hypotheses) * kMaxModels < kMaxGlobalModelIndex,
                 "ransac probe: the packed key carries the model index in 24 bits, "
                 "so hypotheses * kMaxModels must stay under 2^24");
    if (dFrom == nullptr || dTo == nullptr || dBestKey == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (count < kMinimalSet || hypotheses == 0u) return cudaErrorInvalidValue;
    if (count >= kMaxCount) return cudaErrorInvalidValue;
    if (static_cast<size_t>(hypotheses) * kMaxModels >= kMaxGlobalModelIndex) {
        return cudaErrorInvalidValue;
    }
    if (!(threshold > 0.0f)) return cudaErrorInvalidValue;
    return cudaSuccess;
}

ScoreArm gLastArm = ScoreArm::Serial;

} // namespace

bool& warpScoreArmEnabled() {
    static bool enabled = true;
    return enabled;
}

ScoreArm lastScoreArm() { return gLastArm; }

cudaError_t roundAsync(const Point2f* dFrom, const Point2f* dTo, size_t count,
                       uint64_t seed, uint64_t iterBase, unsigned hypotheses,
                       float threshold, unsigned long long* dBestKey,
                       cudaStream_t stream) {
    const cudaError_t bad = validate(dFrom, dTo, count, hypotheses, threshold, dBestKey);
    if (bad != cudaSuccess) return bad;

    // The arm's OWN gate. Below one full ballot there is nothing for the warp
    // mapping to pack, so the launcher takes the serial path whatever the switch
    // says -- and that case is the benchmark's ~1.00x control.
    const bool warpArm = warpScoreArmEnabled() && count >= kWarpScoreMinCount;
    gLastArm = warpArm ? ScoreArm::WarpBallot : ScoreArm::Serial;

    if (warpArm) {
        kRoundWarpBallot<<<gridForThreads(hypotheses * kWarp), kBlock, 0, stream>>>(
            dFrom, dTo, count, seed, iterBase, hypotheses, threshold, dBestKey);
    } else {
        kRoundSerial<<<gridForThreads(hypotheses), kBlock, 0, stream>>>(
            dFrom, dTo, count, seed, iterBase, hypotheses, threshold, dBestKey);
    }
    return cudaGetLastError();
}

cudaError_t roundOneSamplePerWarpAsync(const Point2f* dFrom, const Point2f* dTo,
                                       size_t count, uint64_t seed, uint64_t iterBase,
                                       unsigned hypotheses, float threshold,
                                       unsigned long long* dBestKey,
                                       cudaStream_t stream) {
    const cudaError_t bad = validate(dFrom, dTo, count, hypotheses, threshold, dBestKey);
    if (bad != cudaSuccess) return bad;
    kRoundOneSamplePerWarp<<<gridForThreads(hypotheses * kWarp), kBlock, 0, stream>>>(
        dFrom, dTo, count, seed, iterBase, hypotheses, threshold, dBestKey);
    return cudaGetLastError();
}

cudaError_t probeSamplesAsync(const Point2f* dFrom, const Point2f* dTo, size_t count,
                              uint64_t seed, uint64_t iterBase, unsigned samples,
                              float threshold, bool orderedSampling, size_t* dIndices,
                              uint32_t* dDrew, EssentialMatrix* dModels,
                              uint32_t* dProduced, uint32_t* dSupport,
                              uint32_t* dFlagWords, cudaStream_t stream) {
    if (dIndices == nullptr || dDrew == nullptr || dModels == nullptr ||
        dProduced == nullptr || dSupport == nullptr || dFlagWords == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (dFrom == nullptr || dTo == nullptr || samples == 0u) return cudaErrorInvalidValue;
    if (count < kMinimalSet || count >= kMaxCount) return cudaErrorInvalidValue;
    if (!(threshold > 0.0f)) return cudaErrorInvalidValue;

    kProbeSamples<<<gridForThreads(samples), kBlock, 0, stream>>>(
        dFrom, dTo, count, seed, iterBase, samples, threshold, orderedSampling, dIndices,
        dDrew, dModels, dProduced, dSupport, dFlagWords);
    return cudaGetLastError();
}

} // namespace ransacprobe
} // namespace cudabench
