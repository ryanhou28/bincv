#pragma once

/// @file threshold.hpp
/// @brief The device arm of ops/threshold.hpp: a wide frame or an N-bit frame
/// already in GPU memory becomes one bit per pixel.
///
/// Two entry points, two tiers, exactly as the host header has them.
///
/// threshold(DeviceImageConstView<uint8_t>, dst, thresh) **TIER 1.**
/// binarize(DevicePlaneBlockConstView, dst, thresh) **TIER 3.**
///
/// ---------------------------------------------------------------------------
/// `threshold` ADDS NO KERNEL, AND THAT IS THE DESIGN
///
/// The host entry point reduces `double thresh` to one integer cutoff and hands
/// the buffer to `packBits` -- its own note says why ("ONE IMPLEMENTATION ...
/// before that split this loop was a second copy of the same word assembly, and
/// only one of the two was ever optimized"). The device arm is the same
/// composition at the same altitude: `impl::thresholdCutoff` -- the HOST's
/// function, not a restatement of it, because it is BINCV_HOST_DEVICE and
/// core-visible -- and then `cuda::packBits`, which is already proven bit-exact
/// against the host packer and already benchmarked.
///
/// That is what makes the TIER 1 claim provable in the gate that verifies it.
/// scripts/verify_cuda.sh configures `-DBINCV_USE_OPENCV=OFF`, so
/// `bincv::threshold(const cv::Mat&, ...)` does not exist there; what the gate
/// compares is the device composition against `impl::thresholdCutoff` +
/// `bincv::packBits`, the host's OWN reduction and the host's OWN packer, rather
/// than against a test-local copy of either. The chain closes:
///
/// device == host (this backend's suite)
/// host == cv::threshold (tests/test_threshold.cpp, wherever OpenCV is on)
///
/// A test-local copy of the cutoff rule would agree on 127.5 and disagree on
/// NaN, and nothing would say so.
///
/// ---------------------------------------------------------------------------
/// THE COMPARISON IS STRICTLY GREATER THAN
///
/// Inherited from the host header, where the full derivation lives: `thresh = 0`
/// sets every NON-ZERO pixel, `thresh = 255` sets nothing, `NaN` sets nothing,
/// `-inf` sets everything. Every `double` is defined, including the ones outside
/// cv::threshold's own domain -- there binCV answers the arithmetic and
/// cv::threshold does not have an answer to match.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NAMED because it is NARROWER than the host's
///
/// * `threshold` takes a **uint8 source only**. The host's Tier 1 entry point
/// takes CV_8UC1 and has no 16-bit spelling, so a device uint16 overload would
/// be a Tier 1 claim with no host counterpart to be bit-exact against -- the
/// backend's own contract forbids that. A 16-bit caller has
/// `cuda::packBits(src, dst, PackRule::GreaterThan, t)`, which is the same
/// operation with the reduction already performed.
/// * `binarize` takes **1 to 32 planes**, the host's own bound: above 32 the
/// cutoff a caller may need (2^32) is not representable in an `unsigned`
/// and the answer stops being monotone in `thresh`. The host rejects that at
/// compile time; a runtime plane count cannot, so it is asserted and returns
/// `cudaErrorInvalidValue`.
/// * The destination word type is `uint32_t`, the backend's only device word.
///
/// ---------------------------------------------------------------------------
/// ALIASING is the host's contract, unchanged and if anything stricter here.
/// `binarize`'s source planes and destination MUST NOT OVERLAP, and unlike
/// ops/logic.hpp the exact-alias case is illegal too: a grid-stride loop gives
/// no ordering between the thread that writes word i and the thread that reads
/// plane p's word i, so even the host's read-before-write reasoning does not
/// apply. `threshold`'s source is a wide image and its destination is a bit
/// matrix; they are different objects by construction.
///
/// Empty views are a no-op, not an error -- but that rule is about WIDTH and
/// HEIGHT. A plane block with ZERO PLANES is not an empty image, it is a plane
/// count outside the domain, and it is rejected: the host spells that bound
/// `static_assert(N >= 1)`, and the runtime equivalent of a compile error is an
/// error return, not a silent success.

#include <cuda_runtime.h>

#include "bincv/ops/threshold.hpp"  // impl::thresholdCutoff -- the host's own
#include "core.hpp"
#include "pack.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief dst = (src > thresh), packing a device CV_8U-equivalent frame into one
/// bit per pixel. **API TIER 1** -- bit-exact against
/// `cv::threshold(src, tmp, thresh, 255, cv::THRESH_BINARY)` on the same
/// content for every `thresh` with `|thresh| < 2^31`, through the host arm
/// (device == host by this backend's suite, host == cv::threshold by
/// tests/test_threshold.cpp). Beyond that range cv::threshold is itself
/// undefined and binCV answers the arithmetic; see the host header.
/// @param src Wide source in DEVICE memory, one `uint8_t` per pixel.
/// @param dst Bit destination in DEVICE memory; must have `src`'s dimensions.
/// @param thresh The constant each pixel is compared against, **STRICTLY GREATER
/// THAN**: a pixel equal to `thresh` is 0.
/// @return The launch's `cudaError_t`. Asynchronous: a kernel failure surfaces at
/// the next synchronization, not here.
///
/// @note NO KERNEL OF ITS OWN. One cutoff and a dispatch -- see the file note.
/// @note THREE OUTCOMES, and only the middle one launches a packer:
/// `cutoff > 255` is "nothing passes", a `cudaMemset2DAsync` of the
/// destination and no kernel at all; `cutoff <= 0` is "everything passes";
/// otherwise `packBits` with `PackRule::GreaterEqual`.
/// @note THE `cutoff <= 0` CASE IS FREE IN CODE AND NOT IN TRAFFIC, and this
/// family's whole thesis is that traffic is the metric, so it is said
/// rather than sold. It goes through `packBits`, which reads all
/// `height * width` source bytes to emit a constant, where the host writes
/// whole words and reads nothing. The alternative is a second kernel --
/// a plain 0xFF memset would set the padding bits past `width`, which the
/// padding invariant forbids. A threshold below zero is not on any
/// pipeline's path, and one more hand-written kernel to keep bit-exact
/// forever is the wrong price for it.
/// @note Padding bits are zero on return in all three outcomes: `packBits`'
/// lanes past `width` contribute 0 to the ballot, and the memset clears
/// the whole trailing word.
/// @note Never allocates. Device scratch: none. Shared memory: none.
inline cudaError_t threshold(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                             double thresh, cudaStream_t stream = nullptr) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda threshold: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda threshold: a non-empty image needs non-null pointers");
    BINCV_ASSERT(dst.stride >= rowWords(dst.width),
                 "cuda threshold: dst's stride must cover a whole row");

    // The host's own reduction, called. `cutoff` is an `int` precisely so the two
    // degenerate ends survive it: 0 is "everything passes", 256 is "nothing
    // passes", and neither fits a uint8_t.
    const int cutoff = bincv::impl::thresholdCutoff(thresh);
    if (cutoff > 255) {
        return cudaMemset2DAsync(dst.ptr, dst.stride * sizeof(uint32_t), 0,
                                 rowWords(dst.width) * sizeof(uint32_t), dst.height,
                                 stream);
    }
    // `p >= cutoff` IS `p > thresh`. Getting this boundary wrong by one moves the
    // pixels equal to `thresh` from 0 to 1 -- a fraction of a percent on a natural
    // image, and the whole image at thresh 0.
    return packBits(src, dst, PackRule::GreaterEqual, static_cast<uint8_t>(cutoff),
                    stream);
}

/// @brief dst = (src > thresh), pixel for pixel, over an N-plane bit-sliced
/// device source. **API TIER 3** -- OpenCV has no N-bit image type on host
/// or device, so there is nothing to be bit-exact against and the name is
/// binCV's own. Bit-exact against the host `bincv::binarize` by test.
/// @param planes The source plane block: N planes in ONE allocation, plane 0 the
/// LEAST significant bit, plane `p` occupying rows `[p*height, (p+1)*height)`
/// -- `packQuant`'s and `censusTransform`'s layout, so a caller who ingested
/// through `cuda::packQuant` hands its output straight here with no repacking.
/// @param dst Destination; must have the planes' dimensions and share no word
/// with the block.
/// @param thresh The constant each pixel is compared against, **STRICTLY GREATER
/// THAN**. A value at or above the largest an N-bit pixel can hold selects
/// nothing, which is reached by arithmetic rather than by choice.
/// @return The launch's `cudaError_t`, or `cudaErrorInvalidValue` when the plane
/// count is outside 1..32 (the domain note at the top of this file).
///
/// @note THE ARITHMETIC IS THE HOST'S OWN `bincv::thresholdGE`, CALLED, not a
/// device restatement of it. That function is BINCV_HOST_DEVICE and its
/// whole (value, threshold) input space is enumerated on the host by
/// tests/test_bitslice.cpp; the traversal of the image is what is forked,
/// which is the backend's rule exactly. One thread owns one OUTPUT WORD,
/// gathers the N plane words at that index and resolves 32 pixels in ~2N
/// word operations -- no ballot, no shared memory, no per-word popcount.
/// @note THE KERNEL IS TEMPLATED ON THE PLANE COUNT, dispatched once per launch.
/// That is not a performance specialization, it is what keeps the gathered
/// plane words in REGISTERS: a runtime-bounded loop over a local array is
/// indexed dynamically, which puts the array in local memory, and a spill is
/// the failure that regressed this backend's dense matcher to 0.6x. With the
/// count a compile-time constant the loop unrolls and the array is statically
/// indexed. `-Xptxas -v` reports 0 spills, which is the check that pins it.
/// @note `thresh >= maxValue` is a whole-image answer and stays HOST-side, where
/// it costs one memset instead of N loads per word -- and where it also stops
/// `thresh + 1` from wrapping when a caller passes the largest `unsigned`.
/// `maxValue` is computed at 64-bit width so `1 << 32` at 32 planes is not
/// undefined behaviour.
/// @note Padding bits are zero on return, and here that is LOAD-BEARING rather
/// than automatic: `thresholdGE` answers every lane including the ones past
/// `width`, and at `threshold == 0` it returns all ones whatever the planes
/// hold. The trailing word is stored masked.
/// @note Never allocates. Device scratch: none. Shared memory: none.
cudaError_t binarize(DevicePlaneBlockConstView planes, DeviceBinMatView dst,
                     unsigned thresh, cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
