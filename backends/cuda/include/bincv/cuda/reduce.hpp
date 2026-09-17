#pragma once

/// @file reduce.hpp
/// @brief Bulk population counts over device bit matrices. Reductions stay
/// BULK-ONLY on this backend too -- `__popc` is one instruction here, but a
/// per-word entry point invites per-word round trips over the bus, which is
/// this backend's version of the register-domain crossing the host rule
/// exists to prevent.
///
/// The async forms write a 64-bit count into caller-provided DEVICE memory and
/// return without synchronizing, so a resident pipeline can consume the count
/// on-device or batch the download. The sync forms are conveniences for tests
/// and callers off the hot path; each costs a stream synchronize and says so.

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief Sets *dResult (device memory) to the number of set pixels in `src`.
/// Device twin of the host Tier 1 countNonZero, equal to it by test.
cudaError_t countNonZeroAsync(DeviceBinMatConstView src, unsigned long long* dResult,
                              cudaStream_t stream = nullptr);

/// @brief Sets *dResult to the set pixels of `src` inside `region`, clipped to
/// the view exactly as the host clips it (negative origins legal).
cudaError_t countNonZeroAsync(DeviceBinMatConstView src, Rect region,
                              unsigned long long* dResult,
                              cudaStream_t stream = nullptr);

/// @brief Synchronous conveniences: allocate an 8-byte device scalar, count,
/// download. One stream synchronize each -- benchmark the async forms.
size_t countNonZero(DeviceBinMatConstView src);
size_t countNonZero(DeviceBinMatConstView src, Rect region);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
