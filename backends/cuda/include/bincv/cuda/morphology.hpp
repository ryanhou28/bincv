#pragma once

/// @file morphology.hpp
/// @brief erode / dilate / morphologyEx on bit-packed binary frames, on the
/// device. The host header's algebra, the host header's border rule, one
/// thread per destination WORD.
///
/// **API TIER 1**, over the domain named below. The host twin
/// (`ops/morphology.hpp`) is bit-exact against `cv::erode`, `cv::dilate` and
/// `cv::morphologyEx`; these kernels' own claim is bit-exactness against that
/// host twin, proven directly by `test_cuda_morphology`, and the OpenCV claim
/// is inherited transitively over the inputs the sweep covers.
///
/// ---------------------------------------------------------------------------
/// WHY THIS FAMILY IS ON THE DEVICE AT ALL
///
/// On a CPU binCV's morphology is a DEAD HEAT: erode 3x3 measures 1.04x on x86
/// and 1.00x on aarch64 against `cv::erode`, and the 5x5 ellipse LOSES at
/// 0.32x. One fact explains both -- an AVX2 register holds 32 bytes, so a
/// vectorised BYTE kernel gets the same 32 pixels per instruction a `uint32`
/// word gets, and the packed side's advantage is footprint alone.
///
/// A CUDA thread has no such register. The widest byte-lane primitive on
/// sm_86 is four pixels, and `__vminu4` / `__vmaxu4` -- the two a byte
/// morphology would be built on -- are NOT hardware: they expand to SIX
/// instructions each (three LOP3, SHF, IADD3, PRMT), measured from SASS on
/// this machine. A packed word is still 32 pixels per instruction. The
/// competitor's denominator narrows by 8x and binCV's numerator does not move,
/// so the CPU's dead heat is expected to invert here. That prediction is what
/// `cuda_window_benchmark` exists to confirm or refute; it is not asserted in
/// this header.
///
/// ---------------------------------------------------------------------------
/// THE THREE THINGS THIS KERNEL DOES THAT THE HOST ONE CANNOT
///
/// 1. `__funnelshift_r/l` IS THE SHIFTED-WORD CONSTRUCTION. The host spells
/// `(cur >> d) | (next << (32 - d))`, which nvcc compiles to FOUR
/// instructions -- it does not recognise the idiom. The intrinsic is ONE
/// `SHF`. Every shifted read in this file goes through it, and the
/// word-parallel border below is built on it too.
///
/// 2. `__brev` MAKES THE FOUR NON-CONSTANT BORDERS WORD-PARALLEL. On the host
/// they are the one place packing buys nothing: each out-of-range column
/// maps to a DIFFERENT source column, so the host recomputes a band of
/// `reachX` columns per edge one pixel at a time (and records 1.04x
/// collapsing to 0.64x for exactly that). x86 has no bit-reverse
/// instruction; the device has one. A reflection of 32 consecutive columns
/// IS a bit reversal, so the virtual word one word to the LEFT of a row
/// under BORDER_REFLECT_101 is `__brev(__funnelshift_r(w0, w1, 1))` -- two
/// instructions for 32 pixels, against 32 per-pixel recomputations. Feed
/// those virtual words to the same fold and the per-pixel fixup disappears
/// entirely, along with its warp divergence and its lost-update hazard.
/// Gated (see `morphWordBorderEnabled`), with the banded per-pixel path
/// kept as the reference arm the gate falls back to.
///
/// 3. THE COMPOUND SUBTRACTION IS ONE KERNEL. `cv::morphologyEx` computes
/// GRADIENT / TOPHAT / BLACKHAT with a saturating `cv::subtract` on CV_8U,
/// which on content in {0,255} is exactly `a & ~b`; the host composes that
/// from `bitwiseNot` then `bitwiseAnd`. On the device each of those is a
/// LAUNCH, and at this frame size a launch is a real fraction of the op --
/// so there is one `andNot` kernel, and `morphAndNotFusedEnabled` switches
/// it back to the two-launch spelling so the saving is a measured number
/// rather than an asserted one.
///
/// ---------------------------------------------------------------------------
/// THE ACCEPTED DOMAIN -- NARROWER THAN THE HOST'S, NAMED, ASSERTED, ENFORCED
///
/// The structuring element is a BY-VALUE kernel argument, so its spans must be
/// a fixed-size array. That caps what the device accepts:
///
/// * `element.rows` in [1, 32]. The host accepts any height.
/// * `element.cols` in [1, 512] for a parametric shape, [1, 32] for a
/// `mask`. The host accepts any width (its own suite uses a 129-wide
/// element over a 3-wide frame).
/// * the anchor inside the element and at least one set cell -- the host's
/// own `StructuringElement::valid()`.
///
/// Outside that domain `toDeviceElement` produces an element whose `valid` is
/// false and every launcher **returns `cudaErrorInvalidValue`**, launching
/// nothing. `BINCV_ASSERT` reports it in debug builds as well, and the CUDA
/// gate's Debug configuration is where those assertions are actually compiled.
/// The Tier 1 claim above is a claim over THIS domain, not over the host's.
///
/// @note The element POD is ~288 bytes of kernel parameter space. sm_70 and
/// later give 4 KB; the pre-Volta limit was 256 B and would not hold it.
/// The backend's reference architectures (86, and 87 for Jetson Orin) are
/// both past that line, and the `static_assert` below is what would catch
/// a future one that is not.
///
/// ---------------------------------------------------------------------------
/// WHAT IS NOT HERE, AND WHY
///
/// `morphologyExNeedsScratch` is NOT re-exported into `bincv::cuda`. It is a
/// pure function of a `MorphOp` with nothing device-typed about it, and a
/// second definition of a shared quantity can only drift. Callers sizing a
/// scratch view call `bincv::morphologyExNeedsScratch(op)` -- header-only, no
/// device memory involved.
///
/// A FUSED SINGLE-KERNEL OPEN/CLOSE. It would save one launch and cover
/// BORDER_CONSTANT only, which means a second hand-written morphology kernel
/// to keep bit-exact forever -- a cost CLAUDE.md names as a decision metric
/// and which nobody has priced. It is also where a two-stage shared tile gets
/// its apron arithmetic wrong in a way that is invisible at the frame edge and
/// wrong at every block seam. Not built; `MORPH_OPEN` is two launches.
///
/// A SHARED-MEMORY TILED ARM. The fold is not invertible -- an erosion is an
/// AND over rows, and unlike the dense matcher's Hamming SUM there is nothing
/// to subtract as the window slides -- so a tile buys only the reuse a census
/// tile buys, and the closest precedent in this backend (the packed matcher's
/// shared staging) measured 1.28x SLOWER because the loads it removed were
/// already L1 hits. Here the reuse is even more local: thread `i` reads words
/// `i-1, i, i+1`, which its own warp neighbours are reading in the same
/// instruction. Not built. Recorded as a design decision, not as a measured
/// negative.
///
/// `iterations`, and a custom `borderValue` on `morphologyEx`: absent for the
/// host header's reasons, unchanged.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/morphology.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief The largest element height the device arm accepts. See the domain
/// note above: the spans are a by-value array, so this is a real bound and
/// not a defensive guess.
inline constexpr int kMaxDeviceElementRows = 32;
/// @brief The largest element width the device arm accepts for a PARAMETRIC
/// shape.
inline constexpr int kMaxDeviceElementCols = 512;
/// @brief The largest element width the device arm accepts for a MASK. A
/// masked row is carried as one `uint32` of cell bits, which is what makes
/// the mask travel inside the kernel argument instead of as a second device
/// allocation the caller would have to own and outlive the call.
inline constexpr int kMaxDeviceMaskCols = 32;

/// @brief A `StructuringElement` resolved into the form a kernel can take by
/// value: anchors resolved, the per-row span evaluated ONCE, the mask packed
/// into cell bits, the horizontal reach computed by the host's own helper.
///
/// @note THE SPANS ARE EVALUATED ON THE HOST, ONCE PER CALL. The host header
/// records the measurement that makes this non-negotiable: querying
/// MORPH_ELLIPSE's `sqrt` inside the word loop made a 5x5 erosion 17x
/// SLOWER than `cv::erode`. A device kernel would pay it once per output
/// word -- 11,520 times per reference frame, per element row.
///
/// @note MASK SPANS ARE TIGHT: `[first set cell, last set cell + 1)`, and an
/// empty mask row gets `first == last`. `StructuringElement::spanOfRow`
/// returns `[0, cols)` for a mask, which is NOT tight, and the difference
/// is observable: under BORDER_CONSTANT with a non-default `borderValue`,
/// an element row with no set cell must contribute NOTHING, and a loose
/// span would fold the constant in at every edge. The host's
/// `morphRowGeneric` scans `activeAt` to decide that; this resolves it on
/// the host instead, which is the same answer computed once.
///
/// @note `reachX` comes from `impl::morphMaxOffsetX`, the host's own function,
/// rather than being recomputed here -- the band arithmetic on the two
/// sides must not be able to drift.
struct DeviceStructuringElement {
    int rows = 0;      ///< element height in cells, 1..kMaxDeviceElementRows
    int cols = 0;      ///< element width in cells
    int anchorX = 0;   ///< resolved: OpenCV's -1 already replaced by cols / 2
    int anchorY = 0;   ///< resolved
    int reachX = 0;    ///< max |cell - anchor| over set cells, the host's value
    int masked = 0;    ///< 1 when `cellBits` decides membership inside the span
    int cells3x3 = 0;  ///< the nine cells as bits, row-major; 3x3 gate only
    int valid = 0;     ///< 0 when the element is outside the accepted domain

    int16_t first[kMaxDeviceElementRows] = {};  ///< tight span start, per row
    int16_t last[kMaxDeviceElementRows] = {};   ///< tight span end, per row
    /// Cell membership of row `ey`, bit `ex`. Only read when `masked`.
    uint32_t cellBits[kMaxDeviceElementRows] = {};
};

// A by-value kernel argument. 4 KB is the sm_70+ parameter space; the pre-Volta
// 256 B limit would not hold this and the architecture note in the file header
// says so. This is the check that would catch a future target that is not.
static_assert(sizeof(DeviceStructuringElement) <= 4096,
              "the element must fit sm_70+ kernel parameter space");

/// @brief True when `se` is inside the device arm's accepted domain.
/// @note Exposed so a caller can ask before building anything. The launchers
/// ask too, and return `cudaErrorInvalidValue` when the answer is no.
bool deviceElementDomainOk(const StructuringElement& se);

/// @brief Resolve a host `StructuringElement` into the device form. **HOST
/// function** -- it evaluates the ellipse's `sqrt` per element row and reads
/// the host-owned `mask` bytes, neither of which a kernel may do.
/// @return An element with `valid == 0` when `se` is outside the accepted
/// domain; every launcher refuses such an element rather than launching.
/// @note Not a kernel and not an operation: it carries no tier of its own and
/// is proven only through the ops that consume it.
DeviceStructuringElement toDeviceElement(const StructuringElement& se);

// ---------------------------------------------------------------------------
// The kernels
// ---------------------------------------------------------------------------

/// @brief Morphological erosion: `dst(x,y) = AND over the element of
/// src(x+dx, y+dy)`. **API TIER 1** over the domain named at the top of this
/// file -- bit-exact against `bincv::erode<uint32_t>`, which is itself
/// bit-exact against `cv::erode`.
///
/// @param src Source view, device memory.
/// @param dst Destination, src's dimensions, sharing no word with src. In place
/// is NOT supported: a destination word is built from several source words.
/// @param element From `toDeviceElement`. An invalid one returns
/// `cudaErrorInvalidValue` and launches nothing.
/// @param borderType OpenCV's five, with the host's exact mapping
/// (`impl::borderIndex`, shared through BINCV_HOST_DEVICE rather than
/// transcribed).
/// @param borderValue The pixel outside the image under BORDER_CONSTANT.
/// **Defaults to `true`**, which is `morphologyDefaultBorderValue` for an
/// erosion -- the maximum, so the frame's edge is not eaten away.
///
/// @note The destination's padding bits are zero on return, and the source's
/// padding bits are never read as pixels even when the source wraps a
/// buffer whose padding is dirty.
/// @note Launches on `stream` and returns the launch's error without
/// synchronizing, as every launcher in this backend does.
cudaError_t erode(DeviceBinMatConstView src, DeviceBinMatView dst,
                  const DeviceStructuringElement& element,
                  BorderType borderType = BORDER_CONSTANT, bool borderValue = true,
                  cudaStream_t stream = nullptr);

/// @brief Morphological dilation: `dst(x,y) = OR over the element of
/// src(x+dx, y+dy)`. **API TIER 1**, same basis as `erode`.
/// @param borderValue Defaults to `false` -- `morphologyDefaultBorderValue` for
/// a dilation, the minimum, so the frame does not grow a border.
/// @note One kernel with `erode`, instantiated on the opposite fold, exactly as
/// the host's `MorphFold` is templated: an identity that does not match its
/// fold gives an image right in the interior and wrong at the edge.
cudaError_t dilate(DeviceBinMatConstView src, DeviceBinMatView dst,
                   const DeviceStructuringElement& element,
                   BorderType borderType = BORDER_CONSTANT, bool borderValue = false,
                   cudaStream_t stream = nullptr);

/// @brief The seven `MorphOp` compositions. **API TIER 1** over the accepted
/// domain -- bit-exact against `bincv::morphologyEx<uint32_t>` for every op.
///
/// @param scratch **CALLER-PROVIDED intermediate**, src's dimensions, sharing
/// no word with src or dst. Contents on entry are irrelevant, on return
/// unspecified. Required for every op except MORPH_ERODE and MORPH_DILATE.
/// Ask `bincv::morphologyExNeedsScratch(op)` -- the HOST predicate; it is
/// deliberately not duplicated into this namespace.
/// @note Each step uses the morphological default fill for ITS OWN operation,
/// which is what `cv::morphologyEx` does with its default `borderValue`.
/// @note Launches per op, both switches on: ERODE 1, DILATE 1, OPEN 2,
/// CLOSE 2, GRADIENT 3, TOPHAT 3, BLACKHAT 3. With
/// `morphAndNotFusedEnabled` off, the last three become 4 each.
/// @note `dst` AND `scratch` each have their trailing partial word's bits past
/// `width` cleared, for the reason `erode` states.
cudaError_t morphologyEx(DeviceBinMatConstView src, DeviceBinMatView dst, MorphOp op,
                         const DeviceStructuringElement& element, DeviceBinMatView scratch,
                         BorderType borderType = BORDER_CONSTANT,
                         cudaStream_t stream = nullptr);

namespace impl {

/// @brief Force the GENERAL element kernel instead of the 3x3 centre-anchored
/// specialization. **INTERNAL**, for the benchmark and the tests; same
/// contract as `censusTiledEnabled` and `denseFastArmEnabled`.
/// @note The arm it switches off is the one the HOST has already priced: the
/// host's `morphRow3x3` beats `MorphPath::Generic` by 2.1x-3.7x, and what
/// it removes -- a runtime-trip-count loop over element cells and a
/// data-dependent shift count per cell -- is worth more to a warp than to
/// a core, not less.
bool& morphFastArmEnabled();

/// @brief Force the per-pixel BANDED border fixup instead of the word-parallel
/// virtual border words. **INTERNAL.**
/// @note Its gate is `borderType != BORDER_CONSTANT && reachX < 32 &&
/// width >= 64`. The width bound is real arithmetic, not caution: the
/// virtual word one word past the right edge of a REFLECT_101 row reads
/// source columns down to `width - 64`, so a narrower frame would reflect
/// more than once within one word and the closed form would not hold.
bool& morphWordBorderEnabled();

/// @brief Force the two-launch `bitwiseNot` + `bitwiseAnd` spelling of the
/// compound ops' subtraction instead of the single `dst = a & ~b` kernel.
/// **INTERNAL.**
bool& morphAndNotFusedEnabled();

/// @brief `dst = a & ~b`, padding bits cleared. **INTERNAL** -- the compound
/// ops' subtraction, exposed here only so the benchmark can time it against
/// the two-launch spelling and the suite can hold them equal.
cudaError_t andNot(DeviceBinMatConstView a, DeviceBinMatConstView b, DeviceBinMatView dst,
                   cudaStream_t stream = nullptr);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
