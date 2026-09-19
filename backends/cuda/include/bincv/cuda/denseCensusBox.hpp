#pragma once

/// @file denseCensusBox.hpp
/// @brief The packed census matcher's WARP-COOPERATIVE SEPARABLE-BOX arm, and
/// the runtime switch that turns it off. **INTERNAL** -- the public entry is
/// `denseDisparityCensusPacked` in denseDisparity.hpp and its signature does
/// not change. This header exists so the tests and the benchmark can select
/// the arm, which is what the project's switchable-arm rule requires of a fast
/// path.
///
/// WHAT IS DIFFERENT ABOUT THIS ARM. The shipped packed kernel gives a thread
/// one output pixel and re-reads the whole `winWidth`-wide window for every
/// window row, so a pixel-disparity costs `2 * winWidth` popcounts and about as
/// many loads. The window sum is separable, though, and the horizontal half can
/// be shared -- the obstacle is that on a GPU the columns that share it are
/// ADJACENT, and adjacent columns must be adjacent LANES or the loads stop
/// coalescing. So the sharing is lane-to-lane: one lane owns one descriptor
/// column and keeps that column's vertical window sum in a register, and the
/// `winWidth`-wide horizontal aggregation is a binary-decomposition doubling
/// over `__shfl_down_sync`. No shared memory, no `__syncthreads`, no scratch.
///
/// That distinction is not a preference, it is the recorded negative: staging
/// the same per-pixel costs in SHARED memory measured 1.16 ms against the
/// shipped kernel's 0.91 -- the redundant loads were already L1 hits, so the
/// staging bought nothing and cost 624 barriers per block plus byte-wide bank
/// conflicts. Shuffles have neither failure mode.
///
/// NO STRUCTURAL ADVANTAGE LIVES HERE, and nothing in this file should be read
/// as claiming one. Census EXPANDS data -- 8 bits per pixel in, 24 to 32 out --
/// and the layout that made this path fast is the conventional one-word-per-
/// pixel descriptor, not binCV's bit-planes. This is a standard separable-box
/// technique on a standard descriptor layout. It is here for completeness, on
/// the entry a caller holding ordinary 8-bit frames meets first.

#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/denseDisparity.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {

/// @brief Turn the warp-box arm off, leaving the shipped per-pixel sliding
/// kernel as the reference arm. **INTERNAL.**
///
/// The backend's spelling of the project rule that a fast arm is switchable
/// off, held to bit-exactness against the arm it replaces in ONE binary, and
/// shown by the benchmark to be the arm it timed. Off => the kernel that
/// shipped at 0.91 ms on the reference frame runs, unchanged.
bool& densePackedBoxEnabled();

/// @brief Whether the warp-box arm accepts this parameter shape.
///
/// Two conditions, both derived rather than chosen:
///
///   * `3 <= winWidth <= 17`, odd. A warp has 32 lanes and the horizontal
///     aggregation needs `winWidth - 1` of them as a halo, so only
///     `33 - winWidth` lanes produce output. At 17 that is 16 -- half the warp
///     -- and above it the halo tax exceeds 2x, at which point the shipped
///     per-pixel kernel is genuinely the better arm.
///   * `winWidth * winHeight * 32 <= 0xFFFFFE`. The kernel folds its running
///     best as `(cost << 8) | disparity` in one 32-bit register, which is exact
///     only while the cost fits 24 bits. A descriptor carries at most 32
///     comparisons, so the window cost is at most `winWidth * winHeight * 32`.
///     This is the GENERAL invariant, not the reference frame's `9*9*24 = 1944`.
///
/// Outside either condition the launcher falls back rather than failing: the
/// public op's accepted domain is unchanged by this arm's existence.
bool densePackedBoxAccepts(const DenseDisparityParams& params);

/// @brief Launch the warp-box arm. **INTERNAL**, called only by
/// `denseDisparityCensusPacked`.
/// @pre `densePackedBoxAccepts(params)`, the rim is already filled with
/// `kDenseDisparityInvalid`, `dEnd` is the width-clamped last disparity and
/// `outRows` is `height - 2 * (winHeight / 2)`.
/// @return `cudaErrorInvalidValue` if the precondition does not hold, so a
/// caller that skips the gate gets an error rather than a wrong map.
cudaError_t launchDensePackedWarpBox(DeviceImageConstView<uint32_t> left,
                                     DeviceImageConstView<uint32_t> right, int minD,
                                     int dEnd, const DenseDisparityParams& params,
                                     DeviceImageView<uint8_t> disparity, size_t outRows,
                                     cudaStream_t stream);

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
