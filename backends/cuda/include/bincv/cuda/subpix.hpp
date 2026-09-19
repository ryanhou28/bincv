#pragma once

/// @file subpix.hpp
/// @brief The device arm of ops/subpix.hpp: sub-pixel corner refinement over
/// device bit-planes. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// THERE IS NO `cv::cuda` COUNTERPART, AND THE CLAIM IS SHAPED ACCORDINGLY
///
/// OpenCV's CUDA modules ship corner DETECTORS (`cudaimgproc`) and feature
/// detectors (`cudafeatures2d`); there is no GPU `cornerSubPix` in
/// cudaimgproc, cudafeatures2d, cudaarithm, cudawarping, cudafilters,
/// cudaoptflow or cudastereo. Under ruling R2 this op therefore ships on
/// correctness, memory and the host comparison, with its speed verdict recorded
/// OUTSTANDING -- and **no GPU-vs-GPU speedup is claimed or implied anywhere**.
///
/// **THE CLAIM IS A RESIDENCY ONE.** The four derivative planes are already on
/// the device for the corner op. The alternative to this kernel is downloading
/// them -- 180 KB at 752x480 -- to refine a couple of hundred corners on the
/// CPU and uploading the answers back. The benchmark prices this arm against
/// that specific round trip, and against the narrower alternative of
/// downloading only the bands around the corners, because "pick the right
/// baseline" applies to the thing being replaced too.
///
/// ---------------------------------------------------------------------------
/// WHAT BIT-PLANES BUY HERE, MEASURED RATHER THAN ASSERTED
///
/// A weighted sum is not a bit count -- the host header says so itself -- so
/// nothing collapses into popcounts. What the representation buys is the SKIP:
/// a pixel with no gradient in either axis contributes nothing to any of the
/// five accumulators, and `|dx| | |dy|` finds all of them a WORD at a time, so
/// the loop visits set bits through `__ffs` instead of every window pixel. The
/// host header states the quantity: "the dense spelling would touch 961 pixels;
/// this touches the edges."
///
/// `impl::subPixSkipEnabled()` turns the skip off and runs the dense window
/// loop instead. Both arms are held to byte-identical positions and identical
/// counters in one binary, and the benchmark prints the ratio beside a window in
/// which every bit of both magnitude planes is set -- where the skip can
/// eliminate nothing and the pair must read ~1.00x.
///
/// ---------------------------------------------------------------------------
/// ONE THREAD PER CORNER, AND THAT IS THE BIT-EXACTNESS REQUIREMENT
///
/// Double addition is not associative, so any warp- or block-level reduction
/// sums the five accumulators in an order the host did not use and changes their
/// last bits -- which changes `q`, which can change the converged position and
/// can flip a corner between `refined`, `clamped` and `diverged`. A serial
/// per-thread walk in the host's row-major, LSB-first order is the only shape
/// that is bit-exact by construction. Two hundred corners is about seven warps;
/// the work is microseconds, and the decision rule prices this op against the
/// transfer it replaces rather than against a device it cannot saturate.
///
/// ---------------------------------------------------------------------------
/// TWO PIECES OF FLOATING-POINT CARE, BOTH LOAD-BEARING
///
/// * **`src/subpix.cu` is compiled `-fmad=false`.** `bx += xx*px + xy*py` has two
/// multiplies and an add; `xx*px` is a weight times a small integer and
/// therefore ROUNDS, so a fused multiply-add gives a different result from the
/// host's separate mul-and-add. `det = gxx*gyy - gxy*gxy` is the same shape.
/// By contrast `gxx += w*gx*gx` with `gx` in {-1, 0, +1} is immune -- the
/// product is exactly `+-w` or zero and a fused add of an exact product rounds
/// identically -- which is worth stating so the flag reads as covering
/// `bx`, `by` and `det` rather than as superstition.
/// * **The Gaussian mask is host-built and uploaded, never computed on device.**
/// `impl::subPixMask` calls `std::exp`; CUDA's `exp(double)` is documented at
/// 1-2 ulp where glibc's is about 0.5, so the two are not required to agree,
/// and a one-ulp weight difference propagates straight into the accumulators.
/// `DeviceSubPixMask` runs the HOST's own mask builder and uploads the result.
/// A future "optimisation" that computes the mask on device reintroduces
/// exactly that.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NAMED because it is narrower than the host's (ruling R4)
///
/// * **TERNARY planes only** -- `N == 1`, which is pyramid level 0 and what
/// corner.hpp's promise 4 already restricts corners to. The host's
/// `SignedQuantMat<N, W>` spelling has no device twin, and the four
/// `DeviceBinMatConstView` parameters make an N-bit level unspellable rather
/// than silently wrong.
/// * Word type `uint32_t`. `winHalf` in `[1, impl::kMaxWinHalf]`, as on the host.
/// * Corner counts are `uint32_t` (compaction.hpp's domain).
/// * A violation returns `cudaErrorInvalidValue`; it does not truncate.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/subpix.hpp"
#include "core.hpp"
#include "deviceBinMat.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief What `cornerSubPixAsync` did, per corner. Byte-comparable to the
/// host's `SubPixResult` after `toHost()`.
/// @note The four counters are per-corner and independent, so `atomicAdd` into
/// them is exact regardless of the order threads finish in.
struct DeviceSubPixResult {
    uint32_t refined = 0;   ///< corners whose position moved and converged
    uint32_t singular = 0;  ///< corners left untouched because `G` was not invertible
    uint32_t clamped = 0;   ///< corners whose step left the window
    uint32_t diverged = 0;  ///< corners reverted for walking outside their own window

    SubPixResult toHost() const {
        SubPixResult r;
        r.refined = refined;
        r.singular = singular;
        r.clamped = clamped;
        r.diverged = diverged;
        return r;
    }
};

/// @brief OpenCV's Gaussian window mask, built by the HOST and uploaded once.
/// @note An owning container, because allocation lives in containers here
/// exactly as it does on the host -- and because the whole point is that the
/// weights come from `bincv::impl::subPixMask`, whose `std::exp` the device's
/// is not required to agree with. Build it once per pipeline, not per frame.
/// @note `(2*winHalf + 1)^2` doubles: 968 B at the shipped `winHalf` 5, 7,688 B
/// at the `kMaxWinHalf` cap. **No frame-sized allocation at any winHalf**,
/// which is a checkable binary property rather than a ratio.
class DeviceSubPixMask {
public:
    /// @brief Builds and uploads the mask for `params`.
    /// @note Throws (or aborts, where exceptions are off) on an out-of-domain
    /// `winHalf`, through the project's error policy -- this is a SETUP path,
    /// and a setup path's failure is not a kernel's `cudaError_t`.
    explicit DeviceSubPixMask(const SubPixParams& params);

    const double* devicePtr() const { return mask_.data(); }
    int winHalf() const { return winHalf_; }
    int zeroHalf() const { return zeroHalf_; }
    /// @brief Doubles the mask holds -- `(2*winHalf + 1)^2`.
    size_t size() const { return mask_.size(); }

private:
    DeviceArray<double> mask_;
    int winHalf_ = 0;
    int zeroHalf_ = -1;
};

/// @brief Refines corner positions to sub-pixel accuracy, on the device.
/// **API TIER 2.** Bit-exact against `bincv::cornerSubPix` over a
/// `TernaryMat<uint32_t>` pair: byte-identical positions (exact `float`
/// compare, no tolerance) and identical `{refined, singular, clamped,
/// diverged}`.
/// @param magX,magY Magnitude planes of the x- and y-derivatives.
/// @param signX,signY Sign planes; a SET bit is NEGATIVE.
/// @param cornersXY In/out, in DEVICE memory: `count` interleaved `(x, y)`
/// float pairs, which is the byte layout of the host's `Point2f*`.
/// Positions are refined in place; a corner whose window leaves the image,
/// or whose `G` is singular, is left exactly where it was.
/// @param count Number of corners.
/// @param params `winHalf`, `zeroHalf`, `maxIterations`, `epsilon`. **Must be
/// the params `mask` was built from** -- asserted, and refused in release.
/// @param mask The uploaded Gaussian window: `DeviceSubPixMask::devicePtr()`.
/// @param result One `DeviceSubPixResult` in device memory. **Must be zeroed
/// before the launch**; the kernel only adds to it.
/// @return `cudaErrorInvalidValue` on a domain violation, else the launch's code.
/// @note **The parameter order is the family's**: magnitudes then signs, as
/// `cornerMinEigenValAsync` and `goodFeaturesToTrackAsync` take them. All four
/// are the same type, so a transposed pair compiles silently and produces a
/// plausible wrong answer; one order across the family is the only defence a
/// signature can offer.
/// @note Never allocates. Device scratch: none.
cudaError_t cornerSubPixAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                              DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                              float* cornersXY, uint32_t count, const SubPixParams& params,
                              const double* mask, DeviceSubPixResult* result,
                              cudaStream_t stream = nullptr);

namespace impl {

/// @brief Turns the set-bit skip off and runs the dense window loop instead.
/// **INTERNAL.** Default true.
/// @note This op's only optimisation, and therefore the one a switch has to
/// reach: without it there is nothing proving the skip is answer-preserving in
/// one binary, and no gate-excluded case that must read ~1.00x. The natural
/// control is a window in which every bit of both magnitude planes is set.
bool& subPixSkipEnabled();

} // namespace impl

// The host's Point2f is two floats and nothing else, which is what lets a device
// keypoint array be `float*` and an upload be a raw copy. A layout change here
// would not fail to compile anywhere -- it would refine the wrong coordinates.
static_assert(sizeof(Point2f) == 2 * sizeof(float),
              "a device corner array is Point2f's bytes: two floats, interleaved");
static_assert(offsetof(Point2f, x) == 0,
              "a device corner array is Point2f's bytes: x first");
static_assert(offsetof(Point2f, y) == sizeof(float),
              "a device corner array is Point2f's bytes: y second");

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
