#pragma once

/// @file core.hpp
/// @brief The CUDA backend's view vocabulary: device-typed views over the SAME
/// byte layout as the host library, plus the word helpers device code needs.
///
/// ---------------------------------------------------------------------------
/// WHAT IS SHARED AND WHAT IS FORKED (ARCHITECTURE, backends)
///
/// The FORMAT is shared: pixel `x` of a row lives at bit `x % 32` of word
/// `x / 32`, rows are `stride` words apart, padding bits past `width` are zero.
/// A device plane and a host plane are the same bytes, so upload and download
/// are raw pitched copies and cross-device bit-exactness is a testable claim.
///
/// The TYPES are forked on purpose: a `DeviceBinMatView`'s pointer addresses
/// GPU memory, and making that a distinct type is what turns "passed a host
/// view to a device kernel" into a compile error instead of a runtime
/// corruption. No call can hide where its memory lives (CLAUDE.md); that rule
/// is applied here at the view level, not only at the container level.
///
/// THE DEVICE WORD TYPE IS uint32_t, ONLY. A CUDA core is a 32-bit machine:
/// there is no 64-bit integer datapath, so a uint64 op is two uint32 ops and a
/// register pair -- the M7 result (u64 1.30x SLOWER at 32-bit width), not the
/// Pi's. The 32-bit granule is also what the hardware primitives speak:
/// `__popc` counts a 32-bit register and `__ballot_sync` packs one bit per
/// lane of a 32-lane warp -- a packed pixel word in one instruction. Wider
/// MEMORY access is a kernel detail (uint4 loads move four words), never a
/// format change. No caller is restricted: on little-endian, a plane at any
/// host word width is byte-identical to a u32 plane (core/view.hpp,
/// narrowPlane), so upload accepts every host word type as a byte copy.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/error.hpp"
#include "bincv/core/view.hpp"

// __host__ __device__ for the few functions shared verbatim by host-side setup
// code and device kernels. cuda_runtime.h defines the annotations away when the
// including translation unit is not compiled by nvcc, so this expands safely
// everywhere the backend's headers are legal.
#define BINCV_CUDA_HD __host__ __device__

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief Words a device row needs: ceil(width / 32).
/// @note The device twin of impl::minRowWords<uint32_t>. The CUDA test suite
/// asserts the two agree across widths rather than trusting the copy.
BINCV_CUDA_HD constexpr size_t rowWords(size_t width) {
    return (width + 31u) / 32u;
}

/// @brief Mask of the bits in a row's LAST word that hold pixels; all ones when
/// the row ends on a word boundary. The device twin of impl::rowTailMask.
BINCV_CUDA_HD constexpr uint32_t rowTailMask(size_t width) {
    return (width % 32u) == 0u ? 0xFFFFFFFFu
                               : ((uint32_t{1} << (width % 32u)) - 1u);
}

/// @brief Non-owning, mutable view of a bit-packed matrix in DEVICE memory.
/// @note Same four fields, same meaning, same byte layout as the host
/// BinMatView<uint32_t> -- asserted below, not assumed -- but a distinct
/// type, so the location of the memory is visible in every signature.
/// @note Construct with all four arguments, exactly as the host views require
/// and for the same recorded reason: a defaulted stride aliases every row
/// onto row 0 and looks correct.
struct DeviceBinMatView {
    uint32_t* ptr = nullptr;  ///< first word of row 0, in device memory
    size_t width = 0;         ///< row length in PIXELS
    size_t height = 0;        ///< number of rows in PIXELS
    size_t stride = 0;        ///< distance between rows in WORDS

    DeviceBinMatView() = default;
    BINCV_CUDA_HD DeviceBinMatView(uint32_t* ptr_, size_t width_, size_t height_,
                                   size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0;
    }

    /// @brief First word of row y. Unchecked, as the host view's row is.
    BINCV_CUDA_HD uint32_t* row(size_t y) const { return ptr + y * stride; }
};

/// @brief Non-owning, read-only view of a bit-packed matrix in DEVICE memory.
struct DeviceBinMatConstView {
    const uint32_t* ptr = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t stride = 0;

    DeviceBinMatConstView() = default;
    BINCV_CUDA_HD DeviceBinMatConstView(const uint32_t* ptr_, size_t width_,
                                        size_t height_, size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}
    /// @brief A mutable device view converts, as the host pair does.
    BINCV_CUDA_HD DeviceBinMatConstView(const DeviceBinMatView& v)
        : ptr(v.ptr), width(v.width), height(v.height), stride(v.stride) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0;
    }
    BINCV_CUDA_HD const uint32_t* row(size_t y) const { return ptr + y * stride; }
};

// The shared format is a claim about bytes, and the twin types must not be able
// to drift from it silently: same field sizes, same field order, same total
// size as the host view they mirror.
static_assert(sizeof(DeviceBinMatView) == sizeof(BinMatView<uint32_t>),
              "device and host views must have identical layout");
static_assert(sizeof(DeviceBinMatConstView) == sizeof(BinMatConstView<uint32_t>),
              "device and host views must have identical layout");

/// @brief Non-owning view of a WIDE (one value per pixel) image in device
/// memory: the sensor-stage input and the disparity output live here.
/// @tparam T uint8_t or uint16_t for inputs (the input contract's integer
/// types), uint8_t for the disparity map.
template <typename T>
struct DeviceImageView {
    T* ptr = nullptr;
    size_t width = 0;   ///< PIXELS per row
    size_t height = 0;  ///< rows
    size_t stride = 0;  ///< distance between rows in ELEMENTS, as host wide
                        ///< inputs count it

    DeviceImageView() = default;
    BINCV_CUDA_HD DeviceImageView(T* ptr_, size_t width_, size_t height_, size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0;
    }
    BINCV_CUDA_HD T* row(size_t y) const { return ptr + y * stride; }
};

template <typename T>
struct DeviceImageConstView {
    const T* ptr = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t stride = 0;

    DeviceImageConstView() = default;
    BINCV_CUDA_HD DeviceImageConstView(const T* ptr_, size_t width_, size_t height_,
                                       size_t stride_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_) {}
    BINCV_CUDA_HD DeviceImageConstView(const DeviceImageView<T>& v)
        : ptr(v.ptr), width(v.width), height(v.height), stride(v.stride) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0;
    }
    BINCV_CUDA_HD const T* row(size_t y) const { return ptr + y * stride; }
};

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv

/// @def BINCV_CUDA_CHECK
/// @brief Reports a failed CUDA runtime call on a SETUP path -- allocation,
/// copy, synchronize -- through the project's error policy: throws
/// std::runtime_error, or prints and aborts where exceptions are off.
/// @note Setup paths only, matching BINCV_THROW's contract. Kernel launchers
/// return cudaError_t instead: a launch is asynchronous, so its failure is
/// a value the caller owns, not a validation error this library can settle
/// at the call site.
#define BINCV_CUDA_CHECK(call)                                            \
    do {                                                                  \
        const cudaError_t bincvCudaErr_ = (call);                         \
        if (bincvCudaErr_ != cudaSuccess) {                               \
            BINCV_THROW(std::runtime_error,                               \
                        cudaGetErrorString(bincvCudaErr_));               \
        }                                                                 \
    } while (0)
