#pragma once

/// @file covariance.hpp
/// @brief The device arm of ops/covariance.hpp: the Lucas-Kanade 2x2 gradient
/// covariance `[SumIx^2, SumIxIy; SumIxIy, SumIy^2]` over a window, from
/// N-BIT sign-magnitude derivative blocks. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHY THIS IS THE BACKEND'S BEST CASE, AND WHERE THE ADVANTAGE MOVES
///
/// ARCHITECTURE §1's identity turns every entry of the 2x2 into a POPULATION
/// COUNT OVER A MASK -- no multiply, no per-pixel accumulator. `__popc` is ONE
/// instruction here where on aarch64 it costs two register-domain crossings per
/// word. reduce.hpp already says that for the ternary case; this file is the
/// N-bit one.
///
/// The cost per word is `3N^2 + N` popcounts against `2N + 2` word loads: the
/// arithmetic grows quadratically in N and the traffic linearly. At N = 1 that
/// is 4 popcounts on 4 loads and the kernel is TRAFFIC-shaped; by N = 3 it is
/// 30 popcounts on 8 loads and it is ALU-shaped. Both are real advantages and
/// they are not the same advantage, so a report that quotes one at the other's
/// operating point is quoting the wrong number. The shipped ladder is 1/2/2/2,
/// so N = 2 (14 popcounts, 6 loads) is the operating point that decides.
///
/// **There is no cv::cuda counterpart and no cv:: counterpart at any API
/// level.** OpenCV exposes no 2x2 gradient covariance: `cv::cuda::createHarrisCorner`
/// and `cv::cuda::createMinEigenValCorner` compute a DENSE FLOAT CORNER RESPONSE
/// *through* a covariance, which makes them the honest comparison for a composed
/// corner operation and not for this one. So this operation's speed verdict is
/// recorded OUTSTANDING rather than measured against a substitute bar.
///
/// ---------------------------------------------------------------------------
/// TWO SHAPES, AND THE BATCHED ONE IS THE POINT
///
/// A launch is ~5-14 us on this host; a 31x31 window is ~62 word visits, i.e.
/// nanoseconds of work. One launch per window is latency, not compute -- so the
/// entry point a tracker uses takes the WHOLE keypoint set and issues ONE
/// launch. The single-region form stays for a caller with one region (which may
/// be a whole frame) and because the batch is held to it by test. This is the
/// same division reduce.hpp makes, for the same measured reason.
///
/// ---------------------------------------------------------------------------
/// THE BATCH'S TWO ARMS, AND THE THIRD ONE THE MEASUREMENT REMOVED
///
/// The shape is one BLOCK per window, 64 threads, warp-shuffled then folded
/// through shared memory -- reduce.cu's `batchKernel` shape, so the two batched
/// reductions in this backend traverse alike. What the optimized arm changes is
/// not the shape but **what unit the block's threads divide up**:
///
/// **BOTH OPTIMIZED ARMS WERE MEASURED AND NEITHER EARNED ITS PLACE.** They are
/// recorded here with their numbers because a rejected alternative is the part
/// of a performance decision a later reader most needs and least often gets.
///
/// 1. ONE THREAD PER WINDOW -- removes the cross-lane reduction entirely, which
/// at 31x31 is a fold of `N(N+1) + 2N^2` counters behind ~62 word visits of
/// real work. It LOST 14 readings out of 14 (1.23x slower at N = 1, 1.76x at
/// N = 2). The reason is occupancy and it is arithmetic: 200 windows is 200
/// THREADS, two blocks on a 48-SM part, leaving the machine idle, where one
/// block per window is 12,800 threads and fills it even after paying the
/// reduction. **Deleted.**
///
/// 2. THE FUNNEL-SHIFT RUN EXTRACTION -- for a clipped run at most 32 pixels
/// wide, `__funnelshift_r(lo, hi, x0 & 31)` puts the whole row into one
/// register, so a thread takes a ROW instead of a (row, word) pair and the
/// 62 units at 31x31 become 31. At the tracker's 200 x 31x31 it read
/// 0.94x-0.97x -- NOT a result, because there both arms fit one pass of a
/// 64-thread block and the batch sits on the launch floor regardless. Priced
/// where the comparison is decidable (4000 windows of 31x240, ~30 MB, the
/// formula's ceiling 2.00x from 8 passes against 4) it read **1.09x-1.15x
/// SLOWER**, 7 processes out of 7. **Kept behind
/// `impl::covarianceAlignedRunEnabled()`, which DEFAULTS OFF.**
///
/// **WHY IT LOSES, AS INSTRUCTIONS.** The funnel saves POPCOUNTS, not LOADS: a
/// run straddling two words must still read both, so the per-row cost stays
/// `2N + 2` loads and gains `2N + 2` funnel shifts to halve a popcount count
/// that was never the bottleneck -- at N = 1 this kernel is traffic-shaped, 4
/// popcounts on 4 loads. The word-VISIT count fell and the BYTE count did not.
///
/// **THE ARM IS NOT GATED ON N, and that is a reading rather than a choice.**
/// The obvious failure mode is a register spill, and `nvcc -Xptxas -v` at sm_86
/// reports 40/56/91/96 registers at N = 1..4 with ZERO bytes spilled anywhere in
/// the supported domain. So no cutoff is derivable from register pressure, and a
/// cutoff picked without one would be a number from nowhere. The aligned-run
/// arm's gate is arithmetic instead of a threshold: one funnel shift reaches 32
/// bits, so a clipped run wider than 32 pixels takes the per-word loop whatever
/// the switch says -- which is the case the benchmark uses as its gate-excluded
/// control.
///
/// **THE SPREAD, SAID AT THE NUMBER.** Every ratio above was taken on a WSL2
/// host under concurrent load, where an empty kernel measures 103-5854% spread,
/// and they are INDICATIVE. The arms' sample ranges overlap within most
/// individual runs; what is reported is the consistency of the per-process
/// medians across 7 processes, which is weaker evidence and is labelled as such.
/// That is also why the losing arm is switched off rather than deleted: a
/// serial pass has to be able to re-take the number that rejected it, and a
/// deleted arm cannot be re-measured.
///
/// ---------------------------------------------------------------------------
/// NO SCRATCH, AND THE BUS CARRIES THE ANSWER RATHER THAN THE WORKINGS
///
/// The selector `sign_x ^ sign_y` is XORed inside the word loop and never
/// materialized as a plane -- the host's no-scratch trade, preserved, and
/// CLAUDE.md's memory tiebreak with it. `~(s_x ^ s_y)` is never formed:
/// `whenClear` is `total - set`, which is also what keeps a trailing word's
/// padding bits out of the count.
///
/// The per-pair counts stay in the block and never cross the bus. Each window
/// costs **24 bytes** of result, against `4N^2` counters if the raw pair counts
/// were downloaded (416 B/window at N = 4 -- 83 KB for 200 keypoints against
/// 4.8 KB). The weighting and the one signed subtraction happen in the
/// epilogue, by calling the HOST's own `impl::combineBitSlicedPairs`, which
/// carries BINCV_HOST_DEVICE: the weighting, the doubled off-diagonal and
/// `SplitCount::crossTerm` therefore have a single implementation across both
/// backends.
///
/// ---------------------------------------------------------------------------
/// THE CLIP IS THE HOST'S OWN FUNCTION
///
/// `bincv::impl::clipRegion` carries BINCV_HOST_DEVICE and the batch kernel
/// CALLS IT on each of its device-resident Rects, exactly as reduce.cu's batch
/// does. Half-open, clamped to the view, negative origins legal, a wholly
/// outside window yields `{0, 0, 0}` -- decided once, in ops/reduce.hpp, for
/// both backends. A clip rule is an invariant, not a kernel, so the
/// fork-the-kernels rule does not license a copy of it.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/covariance.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief `GradientCovariance`'s device-side counterpart: the 2x2, as 24 bytes.
/// @note A distinct POD for the reason `DeviceCovarianceCount` is one -- it is
/// written by device code and must be trivially copyable with no member
/// functions. `toHost` converts at the boundary, so the host type's
/// meaning is not restated here.
/// @note `long long` rather than `int64_t` so that `atomicAdd` on the
/// single-region path names the type it actually takes.
struct DeviceGradientCovariance {
    long long sumXX;  ///< SumIx^2 over the window
    long long sumYY;  ///< SumIy^2 over the window
    long long sumXY;  ///< SumIxIy over the window, SIGNED
};

static_assert(sizeof(DeviceGradientCovariance) == 24,
              "a window's result is 24 bytes -- the figure the memory plan quotes");

/// @brief The host spelling of a device result.
inline GradientCovariance toHost(const DeviceGradientCovariance& d) {
    GradientCovariance out;
    out.sumXX = static_cast<int64_t>(d.sumXX);
    out.sumYY = static_cast<int64_t>(d.sumYY);
    out.sumXY = static_cast<int64_t>(d.sumXY);
    return out;
}

namespace impl {

/// @brief Runtime switch for the FUNNEL-SHIFT run extraction in the batch.
/// **INTERNAL. DEFAULTS TO FALSE** -- the arm is a measured regression (see
/// the file header); the shipped path walks the run word by word. It stays
/// reachable so a serial pass can re-take the indicative number that
/// rejected it.
/// @note Same contract as `shiftFunnelEnabled` and `censusTiledEnabled`: the
/// optimized arm stays reachable, the suite holds both arms to one output
/// in ONE binary, and the benchmark prints the on/off ratio plus a case the
/// arm's own gate excludes.
/// @note It does NOT reach `gradientCovarianceAsync`, whose grid-stride
/// traversal is deliberately left word-by-word -- see there. Timing that
/// entry point with this switch flipped is therefore a second gate-excluded
/// control and must read ~1.00x.
bool& covarianceAlignedRunEnabled();

} // namespace impl

/// @brief The largest magnitude-plane count these kernels are instantiated for.
/// @return 4. A deliberate narrowing of the host contract (which admits 8 in
/// the view form), named here, asserted in the launchers and reported as
/// `cudaErrorInvalidValue` outside. N is compile-time so that the
/// `N(N+1) + 2N^2` per-thread counters are register-resident and the
/// `2^(i+j)` weighting unrolls; a runtime N would index a register array
/// dynamically and spill.
constexpr size_t covarianceMaxPlanes() { return 4; }

// ---------------------------------------------------------------------------
// The batch -- the whole window set, ONE launch
// ---------------------------------------------------------------------------

/// @brief One 2x2 gradient covariance per window, all in a single launch.
/// **THE ENTRY POINT A TRACKER USES. API TIER 3.**
/// @param dxSigned,dySigned The two derivative blocks, `planes == N + 1`:
/// magnitude 0..N-1 then the SIGN plane at index N. Exactly what this
/// backend's `derivativeX` / `derivativeY` write, and byte-identical to the
/// host `SignedQuantMat<N>`. 1 <= N <= covarianceMaxPlanes().
/// @param dWindows `count` Rects in DEVICE memory. Each is clipped by the
/// host's own `impl::clipRegion`: half-open, clamped, negative origins
/// legal, wholly outside yields `{0, 0, 0}` -- a value, not an error.
/// @param dResults `count` results in DEVICE memory, written in window order.
/// @return `cudaSuccess`, the launch's error, or `cudaErrorInvalidValue` for a
/// shape or plane count outside the named domain.
///
/// @note **N == 1 is not special-cased.** This kernel runs its generic loop at
/// N = 1 and must produce, bit for bit, what the host's five-argument
/// ternary `gradientCovariance` produces AND what this backend's existing
/// `countCovarianceBatchAsync` produces recombined through
/// `CovarianceCount::crossTerm`. Delegating at N == 1 would make those
/// checks vacuous, which is why it does not -- the same refusal the host
/// makes, for the same reason.
/// @note **Allocates nothing.** Zero scratch beyond the two blocks read, the
/// window array and the result array; no allocation inside any kernel; 24 B
/// per window on the bus. The sign planes are read only where both
/// magnitudes are set, so a dirty sign plane over a zero magnitude cannot
/// move any of the three numbers -- the host's promise 5, unchanged.
/// @note A bit at or past `width` is never counted, whatever it holds.
cudaError_t gradientCovarianceBatchAsync(DevicePlaneBlockConstView dxSigned,
                                         DevicePlaneBlockConstView dySigned,
                                         const Rect* dWindows, size_t count,
                                         DeviceGradientCovariance* dResults,
                                         cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// The single region -- for a caller with one, and as the batch's oracle
// ---------------------------------------------------------------------------

/// @brief The 2x2 over ONE region, which may be a whole frame. **API TIER 3.**
/// @param dResult One result in DEVICE memory; zeroed by this call before the
/// launch, so a wholly-clipped region reads `{0, 0, 0}`.
/// @note A grid-stride traversal of the clipped region's `(row, word)` pairs,
/// not the batch's per-window shape, because this region may be the frame.
/// Each warp WEIGHTS its own partial counts through the host's
/// `impl::combineBitSlicedPairs` and issues three 64-bit atomic adds. That
/// is exact rather than approximate: the weighting is a LINEAR map over
/// exact integer counts, so weighting per warp and summing is the same
/// integer as summing and weighting once. No per-pair count is ever stored
/// or transferred.
/// @note Identical numbers to the batch -- integer addition, so the
/// combination order cannot change the answer. The suite holds the two to
/// each other as well as to the host.
/// @note **Its traversal stays word-by-word on purpose, and the aligned-run
/// switch does not reach it.** Two reasons, and neither is inertia: this
/// region may be a whole frame, where a 32-pixel funnel gate almost never
/// fires; and keeping its traversal genuinely DIFFERENT from the batch's is
/// what makes it a useful oracle. Two arms that share a traversal agree for
/// reasons that have nothing to do with being right.
cudaError_t gradientCovarianceAsync(DevicePlaneBlockConstView dxSigned,
                                    DevicePlaneBlockConstView dySigned, Rect window,
                                    DeviceGradientCovariance* dResult,
                                    cudaStream_t stream = nullptr);

/// @brief Synchronous convenience: allocate a 24-byte device scalar, launch the
/// single-region form, download, free. **API TIER 3.**
/// @note ONE STREAM SYNCHRONIZE. For tests and callers off the hot path --
/// **a benchmark must use the async forms**, exactly as reduce.hpp's sync
/// conveniences say.
GradientCovariance gradientCovariance(DevicePlaneBlockConstView dxSigned,
                                      DevicePlaneBlockConstView dySigned, Rect window);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
