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

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
