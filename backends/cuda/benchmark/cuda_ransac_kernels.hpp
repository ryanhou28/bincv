#pragma once

/// @file cuda_ransac_kernels.hpp
/// @brief The device apparatus for one question: can the frontend's RANSAC
/// geometry stage move to the GPU?
///
/// **THIS IS NOT AN OPERATION AND MUST NOT BECOME ONE BY ACCIDENT.** Nothing
/// here is declared in `include/bincv/cuda/`, nothing is compiled into
/// `bincv_cuda`, and no caller outside this directory and the suite that pins
/// the shared solver can reach it. It exists so that the measurement which
/// answered the question is reproducible, in the same way
/// `cuda_bench_null.cu` carries an empty kernel that the shipped library has no
/// business holding.
///
/// The kernels call `bincv::fivePointEssential`, `bincv::impl::ransacSample` and
/// `bincv::EssentialModel::residual` -- the HOST library's own functions,
/// compiled for the device through `BINCV_HOST_DEVICE`. That is the whole point:
/// the model ordering, the Gauss-Seidel Aberth iterate, the pivot tie rule and
/// the compaction that makes model index `m` a running count of accepted roots
/// are not reproduced here, they are the same code. A hand-written device
/// elimination would have had to keep all four in agreement forever.
///
/// What IS forked, because it is a traversal: the loop over correspondences.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/essential.hpp"

namespace cudabench {
namespace ransacprobe {

/// @brief Correspondences the warp-ballot scoring arm needs before its own gate
/// lets it run: one full ballot's worth.
/// @note Below this the arm has nothing to pack and the launcher takes the
/// serial path whatever the switch says -- which is the case the benchmark
/// prints as its ~1.00x control.
constexpr size_t kWarpScoreMinCount = 32;

/// @brief Hypotheses per launch and correspondences the packed reduction key can
/// carry. The key is `(support << 24) | (0xFFFFFF - globalModelIndex)`, so
/// support must fit 40 bits (it never exceeds `count`) and the model index
/// must fit 24.
constexpr size_t kMaxCount = size_t{1} << 24;
constexpr size_t kMaxGlobalModelIndex = size_t{1} << 24;

/// @brief Scoring arm switch. Selects ONLY the mapping of correspondences to
/// lanes; the solver stays one hypothesis per thread either way.
/// @note One switch, one change. A switch that also moved the solver would
/// leave the benchmark unable to say which half a ratio came from.
bool& warpScoreArmEnabled();

/// @brief Which scoring arm the last `roundAsync` actually launched.
enum class ScoreArm { Serial, WarpBallot };
ScoreArm lastScoreArm();

/// @brief One RANSAC round on the device: `hypotheses` minimal samples drawn
/// from the host's own counter, solved, scored, reduced into `dBestKey`.
/// @param dBestKey Single `unsigned long long`, **initialized to 0 by the
/// caller**. Zero-support models never write, so a key that is still 0
/// means no model was found -- the host driver's `support > bestInliers`
/// with `bestInliers` starting at 0.
/// @note Deterministic regardless of block scheduling: max on the packed key is
/// max-support-then-lowest-index, which is the host driver's tie rule.
cudaError_t roundAsync(const bincv::Point2f* dFrom, const bincv::Point2f* dTo,
                       size_t count, uint64_t seed, uint64_t iterBase,
                       unsigned hypotheses, float threshold,
                       unsigned long long* dBestKey, cudaStream_t stream);

/// @brief The same round with ONE sample per warp and the other 31 lanes idle.
/// @note Not a shipping shape and not proposed as one. It is the CEILING that
/// any warp-per-sample solver must be measured against: it already removes
/// both penalties such a design targets -- the max over 32 divergent trip
/// counts, and 32 lanes' worth of solver frame thrashing one L1 -- while
/// adding none of the internal parallelism. Whatever a real warp-per-sample
/// arm reaches, it is this number divided by at most the solver's internal
/// width.
cudaError_t roundOneSamplePerWarpAsync(const bincv::Point2f* dFrom,
                                       const bincv::Point2f* dTo, size_t count,
                                       uint64_t seed, uint64_t iterBase,
                                       unsigned hypotheses, float threshold,
                                       unsigned long long* dBestKey,
                                       cudaStream_t stream);

/// @brief Everything one hypothesis produces, written out per sample, so the
/// device compilation of the shared solver can be compared against the host's
/// term by term rather than only through a winner.
/// @param dIndices `samples * 5` -- the sampler's output.
/// @param dDrew `samples` -- 1 when the sampler succeeded, 0 when it refused.
/// @param dModels `samples * EssentialModel::kMaxModels`.
/// @param dProduced `samples` -- models the solver returned.
/// @param dSupport `samples * kMaxModels`.
/// @param dFlagWords `samples * kMaxModels * ransacScratchWords(count)` words, in
/// the HOST's inlier-flag BIT ORDER: correspondence `i` at bit `i & 31` of
/// word `i >> 5`. That is the order `RansacScratch` uses for each of its two
/// flag sets, so one set downloads without conversion. Bits past `count` in
/// the last word are ZERO.
cudaError_t probeSamplesAsync(const bincv::Point2f* dFrom, const bincv::Point2f* dTo,
                              size_t count, uint64_t seed, uint64_t iterBase,
                              unsigned samples, float threshold, bool orderedSampling,
                              size_t* dIndices, uint32_t* dDrew,
                              bincv::EssentialMatrix* dModels, uint32_t* dProduced,
                              uint32_t* dSupport, uint32_t* dFlagWords,
                              cudaStream_t stream);

/// @brief Stack the device compilation of `fivePointEssential` uses per thread.
/// @note **MEASURED BY THE COMPILER, not added up** -- the same discipline
/// `bincv::essentialSolverStackBytes()` records for the host, and the reason
/// the two are comparable at all:
///
/// nvcc -ccbin g++-9 -std=c++17 -O3 -DNDEBUG -arch=sm_86 -Xptxas -v
///
/// reports "6800 bytes stack frame, 1296 bytes spill stores, 2552 bytes spill
/// loads" and "Used 255 registers" for the round kernel. On a GPU that frame
/// is LOCAL memory -- global-memory backed, L1-cached -- not a register file,
/// and it is per THREAD, so a round of H hypotheses has H of them live.
/// Re-measure it when the solver changes.
constexpr size_t deviceSolverFrameBytes() { return 6800; }

} // namespace ransacprobe
} // namespace cudabench
