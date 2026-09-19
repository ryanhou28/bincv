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
// code and device kernels.
//
// KEPT AS A NAME, DEFINED AS AN ALIAS. core/error.hpp now carries
// BINCV_HOST_DEVICE for the same expansion, and two independent `#define`s of one
// annotation are exactly the kind of copy this project refuses: the day one of
// them grows a `__forceinline__` or a clang-CUDA branch, the other silently does
// not. So there is one definition. The SPELLING stays, because the two names
// record different intent and a reader can grep for either: BINCV_HOST_DEVICE
// marks a HOST header's scalar helper that the device is allowed to share, and
// every one of them is a decision about the host library; BINCV_CUDA_HD marks
// the backend's OWN device-typed code, which has no host-only life to protect.
// Deleting the local name would put those two populations under one grep.
#define BINCV_CUDA_HD BINCV_HOST_DEVICE

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

/// @brief Non-owning, mutable view of an **N-bit** image in DEVICE memory: N
/// bit-planes in ONE allocation, plane 0 the least significant bit.
///
/// @note THE LAYOUT IS THE HOST'S, NOT A NEW ONE. `QuantMat<N, WordType>` is a
/// single BinMat of `N * height` rows, so plane `p` is rows
/// `[p * height, (p + 1) * height)` of it and `plane(p)` hands back
/// `data() + p * planeWords()` with `planeWords() == height * alignedWidth`
/// (quantMat.hpp). That is the layout `packQuant`'s `dst` parameter names --
/// "exactly `QuantMat<N>::plane(i)`" -- and the layout this backend's
/// `packQuant` launcher already writes into its `planeBlock` argument. This
/// view addresses those same words. The equality is SWEPT against a real
/// host `QuantMat` over N, width, height and stride in the test suite
/// rather than asserted in prose: a plane offset that drifts returns a
/// fully-formed view of the wrong plane, which reads as a correct answer.
/// @note `height` is ONE PLANE's height, exactly as `QuantMat::getHeight()` is
/// -- not the block's row count. `block()` is the whole stack when a kernel
/// or a transfer wants it as one matrix.
/// @note Every plane shares one stride, as the host's do: the planes are one
/// allocation, not N.
struct DevicePlaneBlockView {
    uint32_t* ptr = nullptr;  ///< first word of plane 0, in device memory
    size_t width = 0;         ///< row length in PIXELS
    size_t height = 0;        ///< rows in ONE plane
    size_t stride = 0;        ///< distance between rows in WORDS, every plane
    size_t planes = 0;        ///< N, the bits per pixel

    DevicePlaneBlockView() = default;
    /// @note Five arguments, none defaulted, for the reason the bit views take
    /// four: a defaulted `planes` aliases every plane onto plane 0.
    BINCV_CUDA_HD DevicePlaneBlockView(uint32_t* ptr_, size_t width_, size_t height_,
                                       size_t stride_, size_t planes_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_), planes(planes_) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0 || planes == 0;
    }

    /// @brief Words in ONE plane -- the host's `QuantMat::planeWords()`.
    BINCV_CUDA_HD size_t planeWords() const { return height * stride; }

    /// @brief First word of plane `p`: the host's `data() + p * planeWords()`.
    BINCV_CUDA_HD uint32_t* planeData(size_t p) const { return ptr + p * planeWords(); }

    /// @brief Plane `p` as a bit matrix -- the type every 1-bit kernel already
    /// takes, so an N-bit caller reaches the binary kernels with no adapter.
    BINCV_CUDA_HD DeviceBinMatView plane(size_t p) const {
        return DeviceBinMatView{planeData(p), width, height, stride};
    }

    /// @brief Row `y` of plane `p`. Identical to `plane(p).row(y)`, spelled for
    /// kernels that walk (plane, row) without materializing a plane view.
    BINCV_CUDA_HD uint32_t* row(size_t p, size_t y) const {
        return ptr + (p * height + y) * stride;
    }

    /// @brief The whole stack as ONE matrix of `planes * height` rows: the form
    /// the packQuant launcher takes and the form a raw transfer copies.
    BINCV_CUDA_HD DeviceBinMatView block() const {
        return DeviceBinMatView{ptr, width, height * planes, stride};
    }
};

/// @brief Non-owning, read-only view of an N-bit image in DEVICE memory.
struct DevicePlaneBlockConstView {
    const uint32_t* ptr = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t stride = 0;
    size_t planes = 0;

    DevicePlaneBlockConstView() = default;
    BINCV_CUDA_HD DevicePlaneBlockConstView(const uint32_t* ptr_, size_t width_,
                                            size_t height_, size_t stride_, size_t planes_)
        : ptr(ptr_), width(width_), height(height_), stride(stride_), planes(planes_) {}
    /// @brief A mutable plane block converts, as the bit views do.
    BINCV_CUDA_HD DevicePlaneBlockConstView(const DevicePlaneBlockView& v)
        : ptr(v.ptr), width(v.width), height(v.height), stride(v.stride), planes(v.planes) {}

    BINCV_CUDA_HD bool empty() const {
        return ptr == nullptr || width == 0 || height == 0 || planes == 0;
    }
    BINCV_CUDA_HD size_t planeWords() const { return height * stride; }
    BINCV_CUDA_HD const uint32_t* planeData(size_t p) const {
        return ptr + p * planeWords();
    }
    BINCV_CUDA_HD DeviceBinMatConstView plane(size_t p) const {
        return DeviceBinMatConstView{planeData(p), width, height, stride};
    }
    BINCV_CUDA_HD const uint32_t* row(size_t p, size_t y) const {
        return ptr + (p * height + y) * stride;
    }
    BINCV_CUDA_HD DeviceBinMatConstView block() const {
        return DeviceBinMatConstView{ptr, width, height * planes, stride};
    }
};

// A plane block is ONE bit matrix plus a plane count, and `plane()` / `block()`
// hand back the bit-matrix view whose layout is already pinned above. Asserting
// the field placement is what keeps that true: a reordered or re-typed field
// here would change which words `plane(p)` names while every signature in the
// backend still compiled.
static_assert(sizeof(DevicePlaneBlockView) == sizeof(DeviceBinMatView) + sizeof(size_t),
              "a plane block is a bit matrix plus a plane count");
static_assert(sizeof(DevicePlaneBlockConstView) ==
                  sizeof(DeviceBinMatConstView) + sizeof(size_t),
              "a plane block is a bit matrix plus a plane count");
static_assert(offsetof(DevicePlaneBlockView, ptr) == offsetof(DeviceBinMatView, ptr),
              "plane block and bit matrix must share their field placement");
static_assert(offsetof(DevicePlaneBlockView, width) == offsetof(DeviceBinMatView, width),
              "plane block and bit matrix must share their field placement");
static_assert(offsetof(DevicePlaneBlockView, height) == offsetof(DeviceBinMatView, height),
              "plane block and bit matrix must share their field placement");
static_assert(offsetof(DevicePlaneBlockView, stride) == offsetof(DeviceBinMatView, stride),
              "plane block and bit matrix must share their field placement");
static_assert(offsetof(DevicePlaneBlockConstView, ptr) ==
                  offsetof(DeviceBinMatConstView, ptr),
              "plane block and bit matrix must share their field placement");
static_assert(offsetof(DevicePlaneBlockConstView, stride) ==
                  offsetof(DeviceBinMatConstView, stride),
              "plane block and bit matrix must share their field placement");

/// @brief Names `planes` bit-planes inside a device matrix allocated as
/// `width x (planes * height)`.
/// @note Host-side setup, not a kernel helper: it is where the block's row count
/// is divided by N, and the one place that division happens.
inline DevicePlaneBlockView planeBlock(DeviceBinMatView block, size_t planes) {
    BINCV_ASSERT(planes >= 1 && planes <= 8,
                 "planeBlock: N outside QuantMat's supported range");
    BINCV_ASSERT(block.height % planes == 0,
                 "planeBlock: the block's row count must be N * the plane height");
    return DevicePlaneBlockView{block.ptr, block.width, block.height / planes, block.stride,
                                planes};
}

/// @brief The read-only spelling of `planeBlock`.
inline DevicePlaneBlockConstView planeBlock(DeviceBinMatConstView block, size_t planes) {
    BINCV_ASSERT(planes >= 1 && planes <= 8,
                 "planeBlock: N outside QuantMat's supported range");
    BINCV_ASSERT(block.height % planes == 0,
                 "planeBlock: the block's row count must be N * the plane height");
    return DevicePlaneBlockConstView{block.ptr, block.width, block.height / planes,
                                     block.stride, planes};
}

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
