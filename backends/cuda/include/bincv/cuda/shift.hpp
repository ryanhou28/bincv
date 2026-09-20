#pragma once

/// @file shift.hpp
/// @brief The device arm of ops/shift.hpp: a 2-D translation with OpenCV border
/// semantics, one bit per pixel.
///
/// shift dst[y][x] = src[y + dy][x + dx], extrapolating outside.
/// shiftLeft/Right/Up/Down the four axis-aligned spellings.
///
/// ---------------------------------------------------------------------------
/// THE BORDER RULE IS THE HOST'S OWN FUNCTION, NOT A COPY OF IT
///
/// `bincv::impl::borderIndex` -- which is `cv::borderInterpolate`, exactly, and
/// the Tier 1 promise every morphology operation inherits -- carries
/// `BINCV_HOST_DEVICE`, so the kernel below CALLS IT. There is no device twin
/// of the border rule and there is nothing for a twin to drift from. That is
/// the shared-format rule applied where it matters most: a border that differs
/// by one column at one edge is invisible in the middle of a frame and fatal to
/// the drop-in promise.
///
/// Nothing is restated here. `bincv::impl::extendedRowWord` -- the row-edge
/// blend that keeps padding bits reading as the fill -- now carries
/// `BINCV_HOST_DEVICE` too, so the kernel calls the host's own function and
/// there is no second copy to sweep against. It was briefly restated as
/// `impl::extendedRowWordDevice`; annotating the original deleted that copy
/// and morphology's independent one together, and the suite's sweep became
/// what it should always have been -- the host's function called ON THE
/// DEVICE and compared against itself on the host.
///
/// ---------------------------------------------------------------------------
/// WHAT IS STRUCTURAL HERE, AND WHAT IS NOT
///
/// **The memory is structural and the instruction count is not.** shift moves
/// one bit per pixel where a byte path moves one byte: at 752x480 that is
/// 480 * 24 * 4 = 46,080 bytes against 480 * 752 = 360,960, i.e. 7.8333x, and
/// at 3840x2160 exactly 8.0000x. That is a formula, and the claim is agreement
/// with it.
///
/// The instruction claim does NOT hold up and is not made. One
/// `__funnelshift_r` produces 32 shifted pixels in one instruction -- but the
/// byte-side alternative to an integer-pixel translation is not a kernel at
/// all. It is a pitched 2-D DMA (`cudaMemcpy2D`, what a GpuMat ROI copy lowers
/// to), which spends ZERO ALU instructions per pixel. binCV's one instruction
/// is not being compared against thirty-two, it is being compared against none.
/// And at the reference frame size both 46 KB and 361 KB sit inside this
/// device's 4 MB L2, so neither side pays the DRAM traffic the ratio describes.
/// The honest statement at 752x480 is: **8x smaller, and on time a wash against
/// a DMA engine.** The traffic argument becomes a speed argument only where the
/// frames exceed L2.
///
/// `__funnelshift_*` earns its place for a second reason that is not about
/// speed: it shifts by `(count & 31)` and is DEFINED at a count of zero, so the
/// `bitShift == 0` branch the host keeps purely to avoid `x << 32` being
/// undefined behaviour has no device counterpart at all. The reference arm
/// spells the two-shift-or WITH that branch, so a reader can see what the
/// funnel shift replaced, and both arms are held to the same output.
///
/// ---------------------------------------------------------------------------
/// WHAT THIS KERNEL PROMISES -- the host's list, unchanged
///
/// 1. Views, never containers. Strides are read per view and may differ.
/// 2. **Padding bits stay zero** in the destination, whatever the source held
/// and whatever the fill is. Every row's trailing word is stored masked.
/// 3. No allocation, no scratch: the vertical half is a row-index remap.
/// 4. **PRECONDITION ON `dst`**: it must span its image's full width or end on
/// a word boundary. The trailing partial word is stored masked, which writes
/// zeros into bits [width, rowWords * 32) -- padding in the usual case, and a
/// wider image's next 1..31 pixels when `dst` is a sub-width window onto one.
/// Nothing can diagnose that: every address written is inside the parent.
/// 5. `src` and `dst` must share no word. In place is NOT supported, for the
/// host's reason: word i of the destination is built from words i +/- k and
/// i +/- k + 1, and with `dy != 0` the source row for destination row y is
/// not monotonic in y, so no traversal order rescues it.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/shift.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Runtime switch for the `__funnelshift` arm. **INTERNAL.**
/// @note Off selects the two-shift-or reference arm, which carries the host's
/// `bitShift == 0` branch. Both arms are held to the same output in one
/// binary by the suite, and the benchmark prints the ratio.
bool& shiftFunnelEnabled();

} // namespace impl

/// @brief `dst[y][x] = src[y + dy][x + dx]`, extrapolating outside the image.
/// **API TIER 3** -- OpenCV has no shift function on either side of the bus,
/// so nothing here is promised drop-in. The extrapolation is
/// `bincv::impl::borderIndex`, the shared host-and-device function that IS
/// `cv::borderInterpolate`; the device arm calls it rather than restating it.
///
/// @param src Source view.
/// @param dst Destination view, `src`'s dimensions, sharing no word with `src`.
/// @param dx Positive moves the image LEFT: dst column c takes src column c+dx.
/// @param dy Positive moves the image UP: dst row r takes src row r+dy.
/// @param borderType How coordinates outside the image extrapolate.
/// @param borderValue The pixel value outside under BORDER_CONSTANT. Ignored by
/// the other four. **false for a dilate step, true for an erode step** --
/// an OR wants nothing outside and an AND wants everything outside, and no
/// single choice serves both.
/// @param stream The stream to enqueue on.
/// @return `cudaErrorInvalidValue` when the domain below is violated, else the
/// launch's own status.
///
/// @note **THE DOMAIN, NAMED.** `src` and `dst` the same extent; every stride
/// covering a whole row; a known `BorderType`; `|dx|` and `|dy|` at most
/// `bincv::impl::maxShiftOffset()`; non-null pointers when non-empty. An
/// empty view is a no-op, not an error. Asserted in debug and reported in
/// every build.
/// @note Offsets larger than the image are correct, not merely defined.
/// @note One pass over the destination, one thread per destination word, no
/// scratch. The non-constant-border rim is repaired per pixel inside the
/// same kernel: the affected columns are a contiguous run of at most
/// `min(|dx|, width)` at ONE edge, so at 752 pixels a |dx| of 100 puts 5 of
/// a row's 24 words on the divergent path -- a real fraction for the shifts
/// morphology issues, which is why the benchmark prices it as its own row
/// rather than calling it a rim.
cudaError_t shift(DeviceBinMatConstView src, DeviceBinMatView dst, ptrdiff_t dx,
                  ptrdiff_t dy, BorderType borderType = BORDER_CONSTANT,
                  bool borderValue = false, cudaStream_t stream = nullptr);

/// @brief `dst[y][x] = src[y][x + k]` -- moves the image LEFT by k columns.
/// **API TIER 3.** See `shift`.
cudaError_t shiftLeft(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                      BorderType borderType = BORDER_CONSTANT, bool borderValue = false,
                      cudaStream_t stream = nullptr);

/// @brief `dst[y][x] = src[y][x - k]` -- moves the image RIGHT by k columns.
/// **API TIER 3.** See `shift`.
cudaError_t shiftRight(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                       BorderType borderType = BORDER_CONSTANT, bool borderValue = false,
                       cudaStream_t stream = nullptr);

/// @brief `dst[y][x] = src[y + k][x]` -- moves the image UP by k rows.
/// **API TIER 3.** No bit manipulation at all: a vertical shift is a row-index
/// remap, so each destination row is one masked copy of a source row.
cudaError_t shiftUp(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                    BorderType borderType = BORDER_CONSTANT, bool borderValue = false,
                    cudaStream_t stream = nullptr);

/// @brief `dst[y][x] = src[y - k][x]` -- moves the image DOWN by k rows.
/// **API TIER 3.** See `shift` and `shiftUp`.
cudaError_t shiftDown(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                      BorderType borderType = BORDER_CONSTANT, bool borderValue = false,
                      cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
