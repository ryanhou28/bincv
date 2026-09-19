#pragma once

/// @file orientation.hpp
/// @brief Keypoint orientation from the intensity centroid, on the device:
/// the twin of ops/orientation.hpp, in both of that header's spellings --
/// a WIDE image and a BIT-PLANE block. **API TIER 3** (OpenCV has no
/// standalone orientation call; it lives inside `cv::ORB`).
///
/// ---------------------------------------------------------------------------
/// ONE LAUNCH FOR THE WHOLE KEYPOINT SET
///
/// Every entry point here takes a `DeviceKeypointSetConstView` and runs ONE
/// launch over all of it. A launch per keypoint is not a GPU implementation of
/// anything: the backend's own recorded precedent is 0.008 ms batched against
/// 3.667 ms for 200 single-region launches, 467x, purely the signature.
///
/// ---------------------------------------------------------------------------
/// THREE ARMS ON THE WIDE SPELLING, AND WHY THERE ARE THREE
///
/// * **W0**, the reference: one thread per keypoint, the host row loop
///   transliterated. Correct at every radius and both pixel types.
/// * **W1**: a warp per keypoint, a lane per disc COLUMN. Isolates what WARP
///   PARALLELISM buys over W0.
/// * **W2**: a warp per keypoint, four pixels per lane -- an aligned 4-byte
///   load and two `__dp4a` (u8 dot-product-accumulate) in place of four byte
///   loads and four multiply-adds. Isolates what VECTORIZATION buys over W1.
///
/// Three, rather than two, because this family's only format claim -- the
/// bit-plane arm against the wide arm -- is decided by which wide arm is the
/// denominator. A scalar denominator would have made the format claim a
/// consequence of leaving the wide arm unoptimized. `load width and format
/// word width are independent` (ARCHITECTURE, backends) cuts both ways, and
/// this is the side of it that costs binCV a number rather than earning one.
///
/// The bit-plane spelling has two arms, its own reference and its own warp
/// arm, and its own switch: a byte gather over a wide disc and a
/// popcount-extract over bit-planes are structurally unrelated kernels, and
/// one switch across both would make a regression in either un-isolatable.
///
/// ---------------------------------------------------------------------------
/// THE ANGLE IS NOT BIT-EXACT, AND THE MOMENTS ARE
///
/// The integer moments `m10` and `m01` are identical to the host's: integer
/// multiply and add in a different grouping is the same integer, and the
/// accumulator bounds below prove no intermediate overflows. `atan2f` on the
/// device and `std::atan2` on the host are two different implementations of a
/// transcendental, and neither is required to be correctly rounded -- so the
/// angle is reported as "same moments, angle within N ULP", with N measured by
/// the test suite, and NOT as bit-exact. `dMoments` exists so a caller who
/// needs an exactly reproducible downstream decision can take the integers
/// and make it themselves.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/orientation.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Force the WIDE reference arm (one thread per keypoint). **INTERNAL.**
/// Same contract as `censusTiledEnabled` / `denseFastArmEnabled`: a fast arm
/// must be switchable off, both arms must be held to one output in one binary,
/// and the benchmark must show which arm it timed.
bool& orientationWideWarpEnabled();

/// @brief Select the `__dp4a` quad form over the warp arm's scalar
/// lane-per-column form. **INTERNAL, AND DEFAULT OFF.** One level below
/// `orientationWideWarpEnabled`, the way `denseBitSlicedEnabled` sits below
/// `denseFastArmEnabled`: with the warp arm off this switch selects nothing.
/// @note Off by default because it LOST its own pre-written bar at the size
/// callers use -- 1.05x at 470 keypoints and 0.89x at 1000, ranges
/// overlapping in every run, on a kernel that sits on the launch floor
/// there. It wins 1.46x at 100,000 keypoints. The reason it is kept rather
/// than deleted is that second number and the SASS behind it; the reason it
/// is not the default is the first.
bool& orientationWideQuadEnabled();

/// @brief Force the BIT-PLANE reference arm. **INTERNAL.** Its own switch,
/// because it is its own kernel: a regression in the popcount-extract arm must
/// not be reachable only through the wide arm's switch.
bool& orientationBitPlaneWarpEnabled();

/// @brief The integer disc as plain kernel-argument data. **INTERNAL.**
/// @note Filled host-side from `bincv::impl::discHalfWidth` -- the host's own
/// disc, not a device re-derivation of the inequality. 44 bytes by value,
/// well inside the 4 KB parameter space, so it is not a device allocation.
/// @note TWO SPELLINGS OF THE SAME TABLE, and the second one is not a
/// duplicate for its own sake. `halfWidth[|dy|]` is an array indexed by a
/// RUNTIME value, and nvcc answers that by copying the kernel parameter into
/// LOCAL memory -- measured: 14 `LDL` in the quad arm, whose whole purpose is
/// to remove instructions from the disc loop. At radius <= 15 every
/// half-width fits four bits and the whole table fits two registers, so the
/// warp arms read it with a shift and a mask and touch no memory at all. The
/// reference arms keep the array: they are the reference, and the array is
/// the readable form. `makeDiscPod` fills both from one loop, so they cannot
/// disagree.
struct DiscPod {
    int8_t halfWidth[32];  ///< indexed by |dy|
    uint32_t packLo;       ///< |dy| = 0..7, four bits each (radius <= 15 only)
    uint32_t packHi;       ///< |dy| = 8..15
    int radius;
};

/// @brief The host's disc, as a POD. **INTERNAL.**
inline DiscPod makeDiscPod(int radius) {
    BINCV_ASSERT(radius >= 1 && radius <= 31,
                 "cuda keypointOrientation: radius out of [1, 31]");
    DiscPod d{};
    d.radius = radius;
    for (int dy = 0; dy <= radius; ++dy) {
        const int h2 = bincv::impl::discHalfWidth(radius, dy);
        d.halfWidth[dy] = static_cast<int8_t>(h2);
        if (radius <= 15) {
            uint32_t& pack = dy < 8 ? d.packLo : d.packHi;
            pack |= static_cast<uint32_t>(h2) << ((dy & 7) * 4);
        }
    }
    return d;
}

cudaError_t keypointOrientationImpl(DeviceImageConstView<uint8_t> img,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream);
cudaError_t keypointOrientationImpl(DeviceImageConstView<uint16_t> img,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream);
cudaError_t keypointOrientationImpl(DevicePlaneBlockConstView planes,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream);

/// @brief Whether the `__dp4a` quad arm's addressing domain holds for `img`.
/// **INTERNAL**, and named here rather than buried in the launcher because the
/// benchmark needs a case the gate EXCLUDES and the test needs to know which
/// arm ran.
/// @note THE DOMAIN, and why it is exactly this: the quad arm reads the
/// 4-aligned word containing each disc pixel. With a 4-aligned base and a
/// stride that is a multiple of four, every row starts 4-aligned, so the
/// aligned word holding the row's LAST disc pixel begins at or before
/// `stride - 4` and the read cannot leave the row -- no allocation slack is
/// required and none is assumed. Drop either condition and the same read can
/// run three bytes past the buffer on the image's last row.
template <typename SrcT>
inline bool quadArmApplies(DeviceImageConstView<SrcT> img, int radius) {
    return sizeof(SrcT) == 1 && radius <= 15 &&
           (reinterpret_cast<uintptr_t>(img.ptr) & 3u) == 0u && (img.stride & 3u) == 0u;
}

} // namespace impl

/// @brief Orientation of a whole keypoint set on a WIDE device image, from the
/// intensity centroid over a disc of `radius`. **API TIER 3.**
/// @param keypoints Device-resident `(x, y)` pairs -- the host family's own
/// interleaved-float contract, so an upload is a raw copy.
/// @param dAngles `keypoints.count` floats in device memory: radians in
/// (-pi, pi], `atan2(m01, m10)`. A flat patch (both moments zero) reports 0,
/// exactly as the host does.
/// @param dKeep Optional, `count` bytes: 0 for a keypoint whose bounding SQUARE
/// falls outside the image, whose angle is then written as 0. The square,
/// not the disc, for the host's reason -- the descriptor that consumes this
/// angle samples the square.
/// @param radius Disc radius in pixels, in [1, 31]. 15 pairs with the 31-pixel
/// descriptor patch.
/// @param dMoments Optional, `2 * count` `long long` in device memory: `m10`
/// then `m01` per keypoint. These ARE bit-exact against the host where the
/// angle cannot be; a caller needing a reproducible downstream decision
/// takes them instead of the float.
/// @note Never allocates, and needs no caller scratch: every per-keypoint
/// intermediate lives in registers and the warp arms reduce through
/// `__shfl_down_sync`. `scratchBytes(N) = 0`.
inline cudaError_t keypointOrientation(DeviceImageConstView<uint8_t> img,
                                       DeviceKeypointSetConstView keypoints, float* dAngles,
                                       uint8_t* dKeep = nullptr, int radius = 15,
                                       long long* dMoments = nullptr,
                                       cudaStream_t stream = nullptr) {
    return impl::keypointOrientationImpl(img, keypoints, dAngles, dKeep, dMoments,
                                         impl::makeDiscPod(radius), stream);
}

/// @brief The `uint16_t` spelling. **API TIER 3.**
/// @note NO `__dp4a` ARM, deliberately. The quad arm's case rests on packing
/// four pixels into one 32-bit lane register, which a 16-bit pixel halves
/// before it starts, and `__dp2a` would need a second alignment-and-mask
/// construction for a type this family's format claim does not rest on. The
/// absence is a stated decision, not an oversight, and the benchmark uses it
/// as a gate-excluded control that must read ~1.00x.
inline cudaError_t keypointOrientation(DeviceImageConstView<uint16_t> img,
                                       DeviceKeypointSetConstView keypoints, float* dAngles,
                                       uint8_t* dKeep = nullptr, int radius = 15,
                                       long long* dMoments = nullptr,
                                       cudaStream_t stream = nullptr) {
    return impl::keypointOrientationImpl(img, keypoints, dAngles, dKeep, dMoments,
                                         impl::makeDiscPod(radius), stream);
}

/// @brief The same orientation on a BIT-PLANE block: `planes.planes` planes,
/// plane `p` weighted by `2^p`. **API TIER 3.**
/// @param planes The host's own `QuantMat` layout -- plane `p` at rows
/// `[p * H, (p + 1) * H)` of one matrix -- so a `QuantMat<N>` uploads as one
/// copy and a `BinMat` is `planes == 1`.
/// @note The moments are EXACTLY the wide spelling's on the equivalent pixel
/// values: the masked-popcount decomposition is an implementation, not an
/// approximation, and the suite holds the two to integer equality.
/// @note THE PER-WORD POPCOUNT STAYS INTERNAL, as the host's does. What is
/// public here is a whole-disc, whole-batch reduction; the six `__popc` per
/// disc row are inside the kernel, which is what the bulk-only rule
/// protects.
inline cudaError_t keypointOrientation(DevicePlaneBlockConstView planes,
                                       DeviceKeypointSetConstView keypoints, float* dAngles,
                                       uint8_t* dKeep = nullptr, int radius = 15,
                                       long long* dMoments = nullptr,
                                       cudaStream_t stream = nullptr) {
    BINCV_ASSERT(planes.planes >= 1 && planes.planes <= 32,
                 "cuda keypointOrientation: planeCount out of [1, 32]");
    return impl::keypointOrientationImpl(planes, keypoints, dAngles, dKeep, dMoments,
                                         impl::makeDiscPod(radius), stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
