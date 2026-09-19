#pragma once

/// @file pack.hpp
/// @brief The sensor stage on the device: a wide pixel array already in GPU
/// memory becomes bits without touching the host.
///
/// The rule enum is the HOST's own `bincv::PackRule` -- one definition. It is a
/// runtime parameter here where the host makes it a template parameter: the
/// host needs the predicate visible to the autovectorizer, while this launcher
/// resolves the rule once per LAUNCH into a templated kernel, which is the same
/// specialization at a different altitude.
///
/// The kernel is the hardware's own packer: 32 lanes evaluate one pixel each
/// and `__ballot_sync` returns the packed word -- one bit per lane, LSB =
/// lowest x, exactly the format's bit order. Lanes past `width` contribute 0,
/// so padding bits are zero by construction rather than by masking.
///
/// ---------------------------------------------------------------------------
/// THREE ARMS BEHIND ONE ANSWER, and the two switches that select among them.
///
/// Every entry point here produces the same matrix; which kernel produces it is
/// a performance decision and nothing else, and the suite holds all three to one
/// output in one binary. The arms, fastest first:
///
/// * BYTE-LANE -- one lane, four pixels, one 32-bit load where the warp was
/// issuing thirty-two one-byte loads. uint8 sources on a 4-byte-aligned base
/// and stride, and a folded cutoff a byte can express.
/// * ROW GRID -- one warp, one word, with the image row carried in
/// `blockIdx.y` so the (row, word) pair costs no division. Every source type
/// and rule; needs `height <= 65535`, the hardware's own `gridDim.y` cap.
/// * GRID-STRIDE -- one flat index, `/` and `%` to recover the pair. No bound
/// at all, which is why it stays: it is the arm above the `gridDim.y` cap,
/// and it is the ORACLE the other two are proven against.
///
/// THE SHAPE OF THE INDEX ARITHMETIC WAS THE WHOLE COST. `cuda::threshold` is
/// this packer with a cutoff in front of it, and it lost its role comparison --
/// 2.22x slower than `cv::cuda::threshold` at 3840x2160 while moving 1.78x LESS
/// traffic. `cuobjdump -sass` on `packKernel<uint8_t, GreaterEqual>` showed 184
/// instructions around one load and one store, two of them software divides, on
/// a machine whose integer datapath has no divide instruction. The row grid
/// deletes both; the byte lane then cuts the load instructions fourfold.
///
/// The arms are SHARED, which is this file's standing hazard: `packBits`,
/// `packRows`, `packQuant` and `cuda::threshold` all launch through them, so a
/// change here is a change to four operations' output and to four operations'
/// cost. All four are held to the host library in one suite.

#include <cuda_runtime.h>

#include "bincv/ops/pack.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Runtime switch for the ROW-GRID arm; `true` by default.
/// @note **INTERNAL.** Same contract as `censusTiledEnabled` and
/// `denseFastArmEnabled`: a fast arm must be switchable off so one binary can
/// time both and hold both to the same output. Not thread-safe and not part of
/// the public API -- it exists for the benchmark and the tests.
bool& packRowGridEnabled();

/// @brief Runtime switch for the BYTE-LANE arm; `true` by default. One level
/// below `packRowGridEnabled`, as `denseBitSlicedEnabled` sits below
/// `denseFastArmEnabled`: with the row grid off, this selects nothing.
bool& packByteLaneEnabled();

/// @brief Whether the row-grid arm can express this launch at all.
/// @note The image row is `blockIdx.y`, and that dimension is capped at 65535.
/// Above the cap the grid-stride arm runs -- the one shape with no bound.
bool packRowGridApplies(size_t height);

/// @brief Whether the byte-lane arm admits this source.
/// @note THE GATE IS THE CONTROL. A case this returns `false` for must read
/// ~1.00x when `packByteLaneEnabled()` is toggled, and the benchmark prints
/// exactly that: a source whose stride is not a multiple of 4, and a uint16
/// source. If either moves, the switch is not selecting what it claims to.
/// @param stride Source stride in ELEMENTS.
/// @param base Source base pointer.
/// @param srcElemSize `sizeof` the source element; only 1 is admitted.
/// @param cutoff The rule folded to `v >= cutoff`; only 0..255 fits a byte lane.
bool packByteLaneApplies(size_t stride, const void* base, size_t srcElemSize,
                         unsigned cutoff);

} // namespace impl

/// @brief Packs a device wide image to one bit per pixel under `rule`.
/// Device twin of the host packBits; bit-identical to it by test.
cudaError_t packBits(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     PackRule rule, uint8_t threshold = 0,
                     cudaStream_t stream = nullptr);
cudaError_t packBits(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     PackRule rule, uint16_t threshold = 0,
                     cudaStream_t stream = nullptr);

/// @brief Packs `src`'s rows into `dst` starting at row `dstRow`. Device twin
/// of the host packRows.
/// @note The host's entry point is about STREAMING -- a microcontroller packing
/// sensor rows as they arrive, never holding a frame. That motivation does
/// not transfer (a device frame is resident by construction), but the
/// operation does: this is how a caller packs a band of a larger
/// destination, and how a pyramid level or a tile is filled without a
/// second allocation. Rows are independent, so a chunked result is
/// identical to a whole-frame one -- the same property that makes the host
/// form exact rather than approximate.
cudaError_t packRows(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint8_t threshold = 0,
                     cudaStream_t stream = nullptr);
cudaError_t packRows(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint16_t threshold = 0,
                     cudaStream_t stream = nullptr);

/// @brief Packs a device wide image to **N bits per pixel** -- the N-bit
/// ingestion path. **API TIER 3.** Device twin of the host packQuant.
/// @param planeBlock N planes in ONE matrix, plane `p` occupying rows
/// `[p * height, (p + 1) * height)`, LSB first -- the layout censusTransform
/// already uses, so a caller holds one allocation and one stride rather than
/// N of each.
/// @param n Planes, 1 to 8 (QuantMat's supported range).
/// @note The rule is `QuantRule::Scale`, the host's only rule, and the device
/// computes the SAME integer expression -- `(v * maxValue + srcMax/2) /
/// srcMax` -- rather than a threshold ladder. The ladder exists on the host
/// because it vectorizes; a warp evaluates the arithmetic directly. Equal
/// to the host bit for bit by test, which is what pins that claim.
/// @note Padding bits are zero on return: lanes past `width` contribute 0 to
/// every plane's ballot.
cudaError_t packQuant(DeviceImageConstView<uint8_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream = nullptr);
cudaError_t packQuant(DeviceImageConstView<uint16_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream = nullptr);

/// @brief The reverse: one bit per pixel out to one byte per pixel.
/// **API TIER 3.** Device twin of the host unpackTo8Bit.
cudaError_t unpackTo8Bit(DeviceBinMatConstView src, DeviceImageView<uint8_t> dst,
                         uint8_t onValue = 255, uint8_t zeroValue = 0,
                         cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
