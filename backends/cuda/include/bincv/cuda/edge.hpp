#pragma once

/// @file edge.hpp
/// @brief The device arm of ops/edge.hpp: gradient-magnitude edge extraction
/// straight into bits, **API TIER 3**.
///
/// ---------------------------------------------------------------------------
/// TIER 3, AND THE NAME IS NOT OPENCV'S
///
/// There is no `cv::cuda` operation that emits one bit per pixel, and none that
/// performs this computation. `cv::Sobel`/`cv::cuda::createSobelFilter` is a
/// 3x3 separable kernel with smoothing against this operation's single-axis
/// difference, so the host header forbids borrowing the name and the device arm
/// keeps binCV's.
///
/// The BORDER rule is OpenCV's, though, and that is what makes the shipped
/// defaults reproduce the reference exactly rather than approximately:
/// `BORDER_REFLECT_101`, index -1 reads 1 and index n reads n-2. The device
/// computes it with the host's own `impl::reflect101Edge` -- one definition,
/// not a device twin of it.
///
/// ---------------------------------------------------------------------------
/// THE HOST'S THREE TEMPLATE PARAMETERS BECOME RUNTIME PARAMETERS
///
/// `EdgeCombine` and `EdgeSpatial` are resolved once per LAUNCH into a templated
/// kernel -- pack.hpp's precedent, "the same specialization at a different
/// altitude". `EdgeRelation` is not a kernel parameter at all: it folds into the
/// threshold host-side exactly as the host's own vector arm folds it,
/// `tp = (relation == Ge) ? t : t + 1`, carried as `unsigned` so `tp == 256` at
/// `t == 255` with `Gt` is simply "nothing passes" and needs no special case.
///
/// ---------------------------------------------------------------------------
/// TWO DELIBERATE FORKS FROM THE HOST BODY. Same answer, different shape, and
/// both are stated here so neither reads as a dropped optimization.
///
/// 1. **No border code path.** The host pays a scalar sweep over the first and
/// last rows and two columns per row, because a 32-byte vector load cannot
/// straddle the frame edge. The device reference arm computes reflect-101 per
/// lane in index arithmetic, so the whole image goes through one arm and there
/// is no second body to keep correct. "How much hand-written code has to stay
/// bit-exact forever" is a metric CLAUDE.md names, and this is the one place
/// the device is structurally SIMPLER than the host.
///
/// 2. **Both axes are computed unconditionally.** The host short-circuits on
/// purpose -- on a sparse edge map most pixels fail both tests, so `And` skips
/// the vertical difference for nearly all of them. Inside a warp that saving is
/// illusory: the lanes diverge and the warp pays both paths anyway, plus the
/// branch. The outputs are identical and the suite holds them so.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NAMED because it is narrower than the host's
///
/// * Source `uint8_t` or `uint16_t` -- the input contract's integer types, and
/// the two the host's `SrcT` is exercised at.
/// * Destination word type `uint32_t`, the backend's only device word. The host
/// compiles at 8, 16, 32 and 64.
/// * `width` and `height` must match between source and destination.
///
/// Shared memory: **0 bytes per block, in every kernel here.** The obvious
/// alternative -- stage an apron tile the way `censusKernelTiled` does -- is not
/// proposed, and that is a decision rather than an omission: this backend has
/// already measured that idea losing on a kernel with far more reuse to harvest
/// (the dense matcher's overlapping windows, 1.16 ms staged against 0.91
/// unstaged). Edge's redundancy is 3x per axis against census's 24x, so there is
/// less to win and the same `__syncthreads()` per tile to pay.
///
/// Device scratch: **none.** No signature here carries a scratch pointer,
/// because nothing here needs state wider than a thread's registers.

#include <cuda_runtime.h>

#include "bincv/ops/edge.hpp"  // EdgeCombine, EdgeRelation, EdgeSpatial, reflect101Edge
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief Gradient-magnitude edge extraction straight into bits, on the device.
/// **API TIER 3.** Bit-exact against the host `bincv::edgeThreshold` for every
/// combination of its three parameters, by test.
/// @param src Wide source in DEVICE memory, `src.stride` ELEMENTS between rows.
/// @param dst Bit destination in DEVICE memory; must have `src`'s dimensions.
/// @param t Threshold, in source units.
/// @param combine How the two axes combine. `Or` is the reference's.
/// @param relation How a gradient is compared with `t`. `Ge` is the reference's.
/// @param spatial Which pixels are differenced. `Wide` -- the central difference
/// `|v[x+1] - v[x-1]|` -- is the reference's.
/// @return The launch's `cudaError_t`. Asynchronous.
///
/// **THE DEFAULTS ARE THE REFERENCE**, as they are on the host: `edgeThreshold(src,
/// dst, 17)` is the tracking pipeline's sensor-stage edge filter.
///
/// @note 8- or 16-bit in, 1 bit out, and **the wide intermediate never exists**,
/// not even inside the kernel: the per-pixel predicate becomes a bit of a
/// ballot without ever being a byte. That is the same structure the host's
/// AVX2 arm gets from `movemask_epi8`, on the format's own 32-bit granule.
/// @note `dst`'s padding bits are zero on return, by construction rather than by
/// masking: a lane past `width` contributes 0 to the ballot.
/// @note Never allocates. Device scratch: none. Shared memory: none.
cudaError_t edgeThreshold(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                          uint8_t t, EdgeCombine combine = EdgeCombine::Or,
                          EdgeRelation relation = EdgeRelation::Ge,
                          EdgeSpatial spatial = EdgeSpatial::Wide,
                          cudaStream_t stream = nullptr);

cudaError_t edgeThreshold(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                          uint16_t t, EdgeCombine combine = EdgeCombine::Or,
                          EdgeRelation relation = EdgeRelation::Ge,
                          EdgeSpatial spatial = EdgeSpatial::Wide,
                          cudaStream_t stream = nullptr);

namespace impl {

/// @brief Forces the reference arm, for the benchmark and the tests.
/// **INTERNAL.** Same contract as `censusTiledEnabled()`.
/// @note A vector arm must be switchable off and the benchmark must show it is
/// on. Both arms are held to the same map in one binary by the suite, and
/// the benchmark prints the on/off ratio beside a case the fast arm's own
/// gate excludes -- which must read ~1.00x, or the switch is not switching.
bool& edgeVectorEnabled();

/// @brief Whether the byte-lane arm would run for this shape. **INTERNAL** --
/// the benchmark and the tests need to name a gate-excluded case without
/// restating the gate, and a restated gate is a gate that can drift.
/// @note The gate, and every clause of it has a reason:
/// * `uint8` source -- the byte-lane instructions are byte-lane.
/// * `EdgeSpatial::Wide` -- the shipped shape, exactly the one the host's
/// own AVX2 arm covers, and the only one whose neighbour quads come from
/// two `__byte_perm`s of one loaded word.
/// * `tp <= 255` -- a byte-lane comparison cannot express a cutoff of 256,
/// which is what `t == 255` with `Gt` folds to. The host's vector arm
/// gates on the same value for the same reason.
/// * 4-byte-aligned row base AND `stride % 4 == 0` -- the arm reads the row
/// as `uint32_t`. The stride clause is not only alignment: it is also
/// what makes the last word-load of a row land inside that row rather
/// than past the allocation, because a stride that is a multiple of 4 and
/// at least `width` is at least `4 * ceil(width / 4)`.
/// * `width >= 128` -- one warp of this arm covers 128 pixels.
bool edgeVectorApplies(size_t width, size_t stride, const void* base, size_t srcElemSize,
                       EdgeSpatial spatial, unsigned tp);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
