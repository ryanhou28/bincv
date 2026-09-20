#pragma once

/// @file keypoints.hpp
/// @brief The link between what a DETECTOR writes and what the KEYPOINT
/// families read, on the device. **API TIER 3** (no OpenCV counterpart at any
/// API level -- see the note at the bottom).
///
/// ---------------------------------------------------------------------------
/// WHY THIS EXISTS, STATED PLAINLY
///
/// `goodFeaturesToTrackAsync` and `detectFastAsync` write records --
/// `DeviceCorner{int x; int y; float response;}` and
/// `DeviceFastCorner{int x; int y; int score;}`. `keypointOrientation`,
/// `computeBrief` and `cornerSubPixAsync` read the host family's own keypoint
/// contract: `count` interleaved `(x, y)` FLOAT pairs (features.hpp). Those are
/// different bytes, and nothing converted one to the other -- so the chain
/// "detect, then orient, then describe" could not be written without taking the
/// keypoints off the device, converting them on the CPU and putting them back.
///
/// That round trip is not expensive in BYTES. It is expensive because the
/// download forces a `cudaStreamSynchronize` in the MIDDLE of the frame, which
/// drains the pipeline at the one point where the rest of the frame's work is
/// already enqueued behind it. This kernel is what removes that stall, and it
/// is the whole reason it is in the library rather than in a caller's loop.
///
/// ---------------------------------------------------------------------------
/// THE COUNT STAYS ON THE DEVICE, AND THAT IS THE POINT
///
/// The corner count is produced by a device atomic and is not on the host when
/// the next launch is enqueued. So this op takes the count as a DEVICE POINTER
/// and reads it in the kernel. A caller who passes a host count has, by
/// definition, already synchronized, and would not need this op.
///
/// `capacity` -- not the count -- sizes the launch, and slots from the count to
/// `capacity` are written **(0, 0)**. That is what lets the consuming launches
/// also be sized by `capacity` with no host round trip: a keypoint at (0, 0)
/// fails every consumer's bounding-box test (`keypointOrientation`'s square,
/// `computeBrief`'s patch reach), so it produces `keep = 0`, an angle of 0 and
/// a zero descriptor, and reads nothing outside the image. Without the zero
/// fill those slots would hold the PREVIOUS frame's keypoints, which is a
/// plausible wrong answer rather than a visible one.
///
/// ---------------------------------------------------------------------------
/// TWO SCALAR STORES, NOT ONE `float2`, AND THAT IS A DECISION
///
/// A `float2` store would halve the store count, at the price of an 8-byte
/// alignment precondition on `xy` -- which would stop a caller slicing one
/// keypoint array into per-octave ranges. The op moves `16 * capacity` bytes
/// (4,096 B at capacity 256) in one launch, so it sits on this platform's
/// launch floor at every capacity a pipeline uses and the store count cannot
/// move the number. The measurement is in the resident pipeline example, which
/// prints this stage against the cheapest launch the library can make.
///
/// ---------------------------------------------------------------------------
/// THE DOMAIN, NAMED
///
/// * `capacity` is `uint32_t`, compaction.hpp's count domain.
/// * The count is read from device memory and **clamped to `capacity`**: an
///   append counter is deliberately unclamped (it reports the TRUE count so a
///   re-run can be sized), so a truncated detection would otherwise index past
///   the corner array.
/// * `xy` must hold `2 * capacity` floats. Null pointers with a non-zero
///   capacity are refused with `cudaErrorInvalidValue`, not launched.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "corner.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief The corner count inside a device `DeviceCornerResult`, as the
/// `const uint32_t*` this family takes.
/// @note A cast rather than `&result->count`, because `result` addresses device
/// memory: forming a member lvalue of an object the host cannot access is
/// not something to rely on, and the offset is pinned by the assertion
/// below instead.
inline const uint32_t* deviceCornerCount(const DeviceCornerResult* result) {
    static_assert(offsetof(DeviceCornerResult, count) == 0,
                  "deviceCornerCount assumes `count` is the first field");
    return reinterpret_cast<const uint32_t*>(result);
}

/// @brief Rewrites a device corner array as the interleaved `(x, y)` float
/// pairs the keypoint families read. **API TIER 3.**
///
/// @param corners `capacity` corner records in device memory -- what
/// `goodFeaturesToTrackAsync` wrote.
/// @param dCount ONE `uint32_t` in DEVICE memory holding the corner count:
/// `deviceCornerCount(result)` for the selection, or
/// `DeviceAppendCounter::devicePtr()` for an append buffer. Values above
/// `capacity` are clamped, because an append counter is unclamped by
/// contract.
/// @param xy Caller-owned, `2 * capacity` floats in device memory. Entries
/// `[0, count)` become `(float(x), float(y))`; entries `[count, capacity)`
/// are written `(0, 0)`.
/// @param capacity Entries `corners` and `xy` hold. **Sizes the launch**, so a
/// consumer launch may be sized by `capacity` too and the frame needs no
/// mid-pipeline synchronize.
/// @return `cudaErrorInvalidValue` on a null pointer with a non-zero capacity,
/// else the launch's own status. Asynchronous.
///
/// @note The integer-to-float conversion is exact: a corner's `x` and `y` are
/// image coordinates, far inside the 24-bit exactly-representable range,
/// so this op loses nothing and a later sub-pixel refinement starts from
/// the integer position the detector actually found.
/// @note Never allocates, takes no scratch, and reads `corners` only up to the
/// clamped count.
cudaError_t keypointsFromCorners(const DeviceCorner* corners, const uint32_t* dCount,
                                 float* xy, uint32_t capacity,
                                 cudaStream_t stream = nullptr);

/// @brief The FAST spelling. **API TIER 3.** Identical contract; `detectFastAsync`
/// appends `DeviceFastCorner`, and its counter is
/// `DeviceAppendCounter::devicePtr()`.
cudaError_t keypointsFromCorners(const DeviceFastCorner* corners, const uint32_t* dCount,
                                 float* xy, uint32_t capacity,
                                 cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// TIER 3, and why there is no OpenCV bar
//
// The conversion has no `cv::cuda` counterpart because OpenCV does not have the
// problem: `cv::cuda::GoodFeaturesToTrackDetector::detect` hands back a
// `GpuMat` of `float2` and `cv::cuda::FastFeatureDetector` a `GpuMat` whose
// first row IS the x coordinates -- the detectors and the consumers were
// designed against one container. binCV's device detectors emit the HOST
// library's record types so a device result can be compared against a host run
// byte for byte, which is the bit-exactness claim the whole backend rests on,
// and this op is the price of keeping it. Under ruling R2 its speed verdict
// against a GPU alternative is therefore recorded OUTSTANDING; what it is
// measured against is the round trip it replaces, which is the only existing
// way to get the same bytes.
// ---------------------------------------------------------------------------

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
