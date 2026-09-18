#pragma once

/// @file denseDisparity.hpp
/// @brief Dense disparity on the device: winner-take-all over a
/// window-aggregated Hamming cost, one byte per pixel, resident end to end.
///
/// SAME ANSWER, FORKED TRAVERSAL. The output contract is ops/denseDisparity.hpp's,
/// bit for bit: left reference, `d = xL - xR >= 0`, ties keep the SMALLEST
/// disparity, `kDenseDisparityInvalid` (255) on the window rim and wherever no
/// candidate has full support. The tests hold this map equal to the host's on
/// the same frames. The traversal shares nothing: the host streams rows
/// through bit-sliced accumulator ladders because it must never hold a cost
/// volume; here every output pixel is a thread that folds its candidates in
/// registers, so no cost volume exists EITHER -- the running best never leaves
/// the register file, which is this backend's spelling of the same memory
/// rule. Device scratch: none.
///
/// `DenseDisparityParams` is the host's own struct. `recomputeVertical` picks
/// between two host arms that are proven output-identical, so it selects
/// nothing here and is ignored.

#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/denseDisparity.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {
/// @brief Force the straightforward reference kernel, for the benchmark and
/// the tests. **INTERNAL.** The backend's spelling of the project rule that a
/// fast arm is switchable off, held to bit-exactness in one binary, and shown
/// by the benchmark to be the arm it timed.
bool& denseFastArmEnabled();
} // namespace impl

/// @brief Dense disparity over an ALREADY-BINARY rectified pair in device
/// memory -- the premise-native path, cost = windowed popcount(L ^ shift(R)).
/// Device twin of the host denseDisparityBinary; output map equal by test.
/// @param disparity One byte per pixel, every pixel written.
cudaError_t denseDisparityBinary(DeviceBinMatConstView left,
                                 DeviceBinMatConstView right,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream = nullptr);

/// @brief Dense disparity over two PACKED census descriptor images
/// (census.hpp's `censusTransformPacked` layout: one uint32 per pixel).
/// **THE FAST CENSUS PATH, and the one the wide-input entry should use.**
///
/// The plane-block form below reads a pixel's K comparisons from K different
/// arrays: K loads and K popcounts per pixel pair, each popcount counting one
/// useful bit. Here a pixel's whole descriptor is one word, so the cost of a
/// pixel pair is `__popc(a ^ b)` -- one load each, one xor, one popcount, with
/// K of 32 bits doing useful work. Measured, that is the difference between
/// 7.7 ms and 2.6 ms on the reference frame.
///
/// Identical output to the plane form and to the host: Hamming distance is
/// invariant under a permutation of the descriptor's bits, and the tests hold
/// this path's map byte-equal to the host's wide path.
cudaError_t denseDisparityCensusPacked(DeviceImageConstView<uint32_t> leftDesc,
                                       DeviceImageConstView<uint32_t> rightDesc,
                                       const DenseDisparityParams& params,
                                       DeviceImageView<uint8_t> disparity,
                                       cudaStream_t stream = nullptr);

/// @brief Dense disparity over two census PLANE BLOCKS (census.hpp's layout:
/// K planes of imageHeight rows each). The census entry for wide-input
/// callers: censusTransform both frames on device, then match here.
/// Output map equal to the host census path by test.
cudaError_t denseDisparityCensus(DeviceBinMatConstView leftPlanes,
                                 DeviceBinMatConstView rightPlanes, size_t planes,
                                 size_t imageHeight,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
