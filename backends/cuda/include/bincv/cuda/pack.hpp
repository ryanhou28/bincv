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

#include <cuda_runtime.h>

#include "bincv/ops/pack.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

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
