#pragma once

/// @file logic.hpp
/// @brief Bitwise logic over device bit matrices -- the smoke test of the
/// shared representation on this backend. Same contract as ops/logic.hpp:
/// word-wise, strides read per row, padding bits zero on return, in-place
/// legal when dst aliases an input EXACTLY.
///
/// Launchers validate on the host (BINCV_ASSERT, debug builds), launch on
/// `stream`, and return the launch's error code without synchronizing --
/// composing a resident pipeline is the caller's, and so is the stream.

#include <cuda_runtime.h>

#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief dst = a & b. Device twin of the host Tier 1 kernel; bit-exact against
/// it by test, not assertion.
cudaError_t bitwiseAnd(DeviceBinMatConstView a, DeviceBinMatConstView b,
                       DeviceBinMatView dst, cudaStream_t stream = nullptr);

/// @brief dst = a | b.
cudaError_t bitwiseOr(DeviceBinMatConstView a, DeviceBinMatConstView b,
                      DeviceBinMatView dst, cudaStream_t stream = nullptr);

/// @brief dst = a ^ b.
cudaError_t bitwiseXor(DeviceBinMatConstView a, DeviceBinMatConstView b,
                       DeviceBinMatView dst, cudaStream_t stream = nullptr);

/// @brief dst = ~src. The operation that would set padding bits; they are
/// masked before the store, as the host kernel masks them.
cudaError_t bitwiseNot(DeviceBinMatConstView src, DeviceBinMatView dst,
                       cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
