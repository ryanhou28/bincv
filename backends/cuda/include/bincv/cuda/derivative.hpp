#pragma once

/// @file derivative.hpp
/// @brief The device arm of ops/derivative.hpp: the binarized spatial
/// derivative `[-1, 0, 1]` on both axes, from an N-bit plane block into a
/// sign-magnitude plane block. **API TIER 3**, with **TIER 1 BORDER
/// SEMANTICS**.
///
/// derivativeX dst(x, y) = src(x + 1, y) - src(x - 1, y)
/// derivativeY dst(x, y) = src(x, y + 1) - src(x, y - 1)
/// derivativeXY both, from one traversal -- the forked-kernel arm
///
/// ---------------------------------------------------------------------------
/// THE TYPES CARRY N, SO NOTHING ELSE HAS TO
///
/// Both blocks are `DevicePlaneBlock*View` (core.hpp) -- the backend's shared
/// plane-block vocabulary, the same layout `packQuant` writes and the same
/// layout `QuantMat<N>` / `SignedQuantMat<N>` hold on the host, byte for byte.
/// The source carries `planes == N`; the destination carries
/// `planes == N + 1`, which is `SignedQuantMat<N>`'s own plane count: N
/// magnitude planes, then the SIGN plane at index N.
///
/// **There is no separate `planes` or `imageHeight` parameter, deliberately.**
/// A block view already carries its plane count and its per-plane height, so a
/// second spelling of either would be a third source of truth that nothing can
/// check and that can silently disagree. The launchers assert the two blocks'
/// extents against each other instead, which is census.hpp's idiom.
///
/// ---------------------------------------------------------------------------
/// THE BORDER RULE IS THE HOST'S, AND IT NEVER ENTERS THE KERNEL
///
/// A 3-tap kernel's only out-of-image columns are `-1` and `width`, and its
/// only out-of-image rows are `-1` and `height`. Each is resolved ONCE on the
/// host by `bincv::impl::borderIndex` -- which IS `cv::borderInterpolate`, and
/// is what makes these edges Tier 1 -- and arrives as a `ptrdiff_t` kernel
/// argument. So there is no device border rule at all, not even a shared one:
/// the class of drift that `rowWords` / `rowTailMask` need a twin sweep for
/// cannot arise here.
///
/// A negative resolved index means BORDER_CONSTANT, and the kernel substitutes
/// `borderValue` -- which reads as the maximum representable N-bit value when
/// true, the host's N-bit reading of the `bool`.
///
/// **This is stated here so that a later "optimization" cannot move the border
/// rule into the kernel unnoticed.** It costs the boundary, not the row.
///
/// ---------------------------------------------------------------------------
/// THE ARITHMETIC IS THE HOST'S OWN FUNCTION
///
/// `bincv::impl::signedDifference<N, uint32_t, false>` carries
/// BINCV_HOST_DEVICE and the kernels CALL IT: the ternary three-op spelling at
/// N == 1, the ripple-borrow subtract plus conditional two's-complement negate
/// above it. Words in, words out, no traversal. The canonical-zero rule (a set
/// sign over a zero magnitude cannot happen, because the sign IS the borrow
/// out) is a property of those expressions, so the device inherits it rather
/// than re-establishing it.
///
/// What IS forked is the traversal. The host carries the left neighbour in a
/// register across the row; a parallel thread cannot, so it LOADS the
/// neighbouring word instead. Adjacent lanes read adjacent words, so the three
/// loads per plane coalesce into one or two cache lines. Staging the row in
/// shared memory is not attempted: the dense matcher already measured that
/// trade on this device at 1.28x SLOWER, because the redundant loads were
/// already L1 hits.
///
/// `__ballot_sync` appears nowhere in this file and its absence is deliberate,
/// not an oversight: this kernel is word-in / word-out and already does 32
/// pixels per instruction. At N == 1 one thread produces 32 pixels of BOTH
/// destination planes from three integer ops (`pos = a & ~b`, `neg = b & ~a`,
/// `mag = pos | neg`). A ballot would pack per-lane bits that are already
/// packed.
///
/// ---------------------------------------------------------------------------
/// THE DOMAIN IS NARROWER THAN THE HOST'S, AND IT IS NAMED
///
/// The host admits N up to 8 in the view form (SignedQuantMat caps at 7). These
/// kernels are instantiated for **N in [1, 4]** -- `derivativeMaxPlanes()` --
/// and RETURN `cudaErrorInvalidValue` outside it, with a BINCV_ASSERT naming
/// the domain in a debug build. N is a compile-time parameter here because the
/// per-plane `a[N]` / `b[N]` / `mag[N]` arrays must stay in registers; a
/// runtime N would index a register array dynamically and spill to local
/// memory, which is the failure this backend has already measured once on the
/// dense matcher's cached ring. Instantiating 1..7 would multiply the code size
/// of five kernels by a ladder nothing ships (the shipped depths are 1/2/2/2).
///
/// ---------------------------------------------------------------------------
/// WHAT THESE KERNELS PROMISE
///
/// 1. **Views, never containers.** Strides are read per view.
/// 2. **PADDING BITS STAY ZERO** in every destination plane, SIGN INCLUDED.
/// The trailing word of each is stored masked, and the sign plane needs it
/// as much as the magnitude planes do: a set padding bit there is a
/// "negative zero" and therefore a canonical-zero violation.
/// The SOURCE's trailing word is deliberately NOT masked, and the device
/// inherits the host's measured coupling verbatim: the dirty padding bit
/// that `cur >> 1` moves lands on exactly the bit the right-border fixup
/// overwrites. Anything that narrows that fixup must re-establish both the
/// last column's right tap and the padding invariant explicitly.
/// 3. **No allocation, no scratch**, in the kernels and in the launchers.
/// 4. **NO IN-PLACE FORM EXISTS.** Neither axis is pointwise in the word
/// index -- destination word i of derivativeX reads source words i-1, i and
/// i+1, and destination row y of derivativeY reads source rows y-1 and y+1
/// -- so no destination plane may share a word with any source plane, and
/// in the fused form the two destination blocks must be distinct from each
/// other. On the device an aliased block is a silent cross-block race,
/// which is strictly harder to see than the host's version of the same bug;
/// the launchers assert it.
/// 5. Empty views are a no-op returning `cudaSuccess`, not an error.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/derivative.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Runtime switch for the FUSED `derivativeXY` arm. **INTERNAL.**
/// @note Same contract as `shiftFunnelEnabled` and `censusTiledEnabled`: the
/// optimized arm must be switchable off, the test holds both arms to one
/// output in ONE binary, and the benchmark prints the on/off ratio. With
/// the switch off, `derivativeXY` issues the two single-axis launches.
bool& derivativeFusedArmEnabled();

} // namespace impl

/// @brief The largest magnitude-plane count these kernels are instantiated for.
/// @return 4. See the DOMAIN section in the file header -- this is a deliberate
/// narrowing of the host contract, and calls above it fail with
/// `cudaErrorInvalidValue` rather than doing something approximate.
constexpr size_t derivativeMaxPlanes() { return 4; }

/// @brief Horizontal binarized derivative, N-bit source to sign-magnitude.
/// **API TIER 3**, TIER 1 border semantics.
/// @param src N magnitude planes (`planes == N`, 1 <= N <= derivativeMaxPlanes).
/// @param dst N+1 planes: magnitude 0..N-1 then the SIGN plane at index N,
/// which is `SignedQuantMat<N>`'s own layout. **A set sign bit means
/// NEGATIVE.** Must have `src`'s width and per-plane height.
/// @param borderType Defaults to BORDER_REFLECT_101 -- `cv::filter2D`'s default
/// and therefore the reference's, and the one that makes the derivative
/// exactly zero on the first and last column instead of manufacturing a
/// full-strength edge around the frame.
/// @param borderValue The pixel outside the image under BORDER_CONSTANT;
/// `true` reads as the maximum representable N-bit value.
/// @return `cudaSuccess`, the launch's error, or `cudaErrorInvalidValue` when
/// the shape or the plane count is outside the named domain.
cudaError_t derivativeX(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                        BorderType borderType = BORDER_REFLECT_101,
                        bool borderValue = false, cudaStream_t stream = nullptr);

/// @brief Vertical binarized derivative. **API TIER 3**, TIER 1 border
/// semantics. Everything derivativeX documents applies with the axis
/// exchanged: the `+1` tap is the row BELOW (larger y).
/// @note NO BIT MANIPULATION AT ALL -- a vertical tap is a row index, so this
/// kernel reads two source rows word for word and never shifts. Its
/// per-word cost is the sign-magnitude subtraction alone.
cudaError_t derivativeY(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                        BorderType borderType = BORDER_REFLECT_101,
                        bool borderValue = false, cudaStream_t stream = nullptr);

/// @brief BOTH axes from one traversal: the fused arm. **API TIER 3**, TIER 1
/// border semantics.
/// @param dxDst,dyDst Two N+1 plane blocks, distinct from each other and from
/// `src`.
/// @note **No single host function computes this.** Its correctness is defined
/// as "equals host derivativeX and host derivativeY run separately, plane
/// for plane" -- the right definition, but a definition rather than an
/// inherited one, which is why the suite checks both axes against both host
/// kernels rather than checking the fused arm against itself.
/// @note One thread owns one destination word index `(y, i)` and emits both
/// axes' `2(N+1)` words from `5N` loads. That is the same load COUNT as the
/// two single-axis kernels combined and ONE launch instead of two. The win
/// is therefore bounded by this platform's launch overhead -- a SIGNATURE
/// result in the class of reduce.hpp's batched form, decaying toward 1.00x
/// as the frame grows into traffic-bound territory -- and the benchmark
/// says so at the number. (There is no intra-thread reuse to collect
/// between the axes: d/dy at row y reads rows y-1 and y+1 and never row y.)
/// @note `impl::derivativeFusedArmEnabled() == false` makes this call the two
/// single-axis kernels instead, with byte-identical output.
cudaError_t derivativeXY(DevicePlaneBlockConstView src, DevicePlaneBlockView dxDst,
                         DevicePlaneBlockView dyDst,
                         BorderType borderType = BORDER_REFLECT_101,
                         bool borderValue = false, cudaStream_t stream = nullptr);

/// @brief The BINARY-level spellings: a 1-bit source gives a ternary result.
/// @note `BinMat` is `QuantMat<1>` on the host and a device bit matrix is the
/// same bytes, so pyramid level 0 reaches these with no adapter. `dst` is a
/// 2-plane block: magnitude, then sign.
inline cudaError_t derivativeX(DeviceBinMatConstView src, DevicePlaneBlockView dst,
                               BorderType borderType = BORDER_REFLECT_101,
                               bool borderValue = false, cudaStream_t stream = nullptr) {
    return derivativeX(DevicePlaneBlockConstView{src.ptr, src.width, src.height, src.stride, 1},
                       dst, borderType, borderValue, stream);
}

inline cudaError_t derivativeY(DeviceBinMatConstView src, DevicePlaneBlockView dst,
                               BorderType borderType = BORDER_REFLECT_101,
                               bool borderValue = false, cudaStream_t stream = nullptr) {
    return derivativeY(DevicePlaneBlockConstView{src.ptr, src.width, src.height, src.stride, 1},
                       dst, borderType, borderValue, stream);
}

inline cudaError_t derivativeXY(DeviceBinMatConstView src, DevicePlaneBlockView dxDst,
                                DevicePlaneBlockView dyDst,
                                BorderType borderType = BORDER_REFLECT_101,
                                bool borderValue = false, cudaStream_t stream = nullptr) {
    return derivativeXY(DevicePlaneBlockConstView{src.ptr, src.width, src.height, src.stride, 1},
                        dxDst, dyDst, borderType, borderValue, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
