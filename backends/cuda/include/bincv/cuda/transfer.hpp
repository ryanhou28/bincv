#pragma once

/// @file transfer.hpp
/// @brief The only operations in this backend that name host memory: pitched
/// copies between host views and device views.
///
/// A transfer is a RAW BYTE COPY, which is the shared representation paying
/// off: no conversion, no repacking, no per-pixel work on either side. For a
/// bit matrix the copied bytes per row are ceil(width / 8) -- exactly the bytes
/// that hold pixels. On little-endian those bytes are identical for every host
/// word width (core/view.hpp, narrowPlane), so upload takes any of the four,
/// and the trailing bits of the last copied byte are zero on both sides by the
/// padding invariant. Words past the copied bytes are never written: the device
/// container zero-fills at allocation and every kernel preserves padding, so
/// the invariant holds without re-clearing per upload; on download, the host
/// destination's own padding is untouched and must already be clean, which a
/// host-allocated matrix guarantees.
///
/// Asynchronous with respect to `stream` when the host memory is pinned;
/// effectively synchronous from pageable memory, which is the common case and
/// is measured as such in the benchmarks. Cost is visible in the signature
/// either way -- nothing else in this backend touches a host pointer.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/view.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief Host bit matrix to device bit matrix. Any host word width.
/// @return The copy call's error code; cudaSuccess on the happy path.
template <typename WordType>
inline cudaError_t upload(BinMatConstView<WordType> src, DeviceBinMatView dst,
                          cudaStream_t stream = nullptr) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "upload: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "upload: a non-empty transfer needs non-null pointers");
    const size_t pixelBytes = (src.width + 7) / 8;
    constexpr size_t wb = sizeof(WordType) * 8;
    // Tight on both sides -- the default stride everywhere -- means the plane
    // is ONE contiguous byte run: equal pitches with no words beyond the row's
    // own. The pitched copy degenerates to a DMA per row otherwise (measured
    // 0.64 ms for 45 KB, ~10x the wide frame's rate); one linear copy is the
    // shape the shared format promises. The extra bytes inside the tail word
    // are the padding bits, zero on both sides by the invariant.
    if (src.stride * sizeof(WordType) == dst.stride * sizeof(uint32_t) &&
        src.stride == (src.width + wb - 1) / wb && dst.stride == rowWords(dst.width)) {
        return cudaMemcpyAsync(dst.ptr, src.ptr,
                               src.stride * sizeof(WordType) * src.height,
                               cudaMemcpyHostToDevice, stream);
    }
    return cudaMemcpy2DAsync(dst.ptr, dst.stride * sizeof(uint32_t), src.ptr,
                             src.stride * sizeof(WordType), pixelBytes, src.height,
                             cudaMemcpyHostToDevice, stream);
}

/// @brief Device bit matrix to host bit matrix. Any host word width.
/// @note Writes only the pixel bytes of each destination row; the destination's
/// padding must already be clean, as a host-allocated matrix's is.
template <typename WordType>
inline cudaError_t download(DeviceBinMatConstView src, BinMatView<WordType> dst,
                            cudaStream_t stream = nullptr) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "download: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "download: a non-empty transfer needs non-null pointers");
    const size_t pixelBytes = (src.width + 7) / 8;
    constexpr size_t wb = sizeof(WordType) * 8;
    // The upload fast path, mirrored; see there.
    if (dst.stride * sizeof(WordType) == src.stride * sizeof(uint32_t) &&
        dst.stride == (dst.width + wb - 1) / wb && src.stride == rowWords(src.width)) {
        return cudaMemcpyAsync(dst.ptr, src.ptr,
                               dst.stride * sizeof(WordType) * dst.height,
                               cudaMemcpyDeviceToHost, stream);
    }
    return cudaMemcpy2DAsync(dst.ptr, dst.stride * sizeof(WordType), src.ptr,
                             src.stride * sizeof(uint32_t), pixelBytes, src.height,
                             cudaMemcpyDeviceToHost, stream);
}

/// @brief Host wide image (the input contract's strided pixel array) to device.
template <typename T>
inline cudaError_t uploadImage(const T* src, size_t width, size_t height,
                               size_t srcStrideElems, DeviceImageView<T> dst,
                               cudaStream_t stream = nullptr) {
    BINCV_ASSERT(width == dst.width && height == dst.height,
                 "uploadImage: src and dst must have the same dimensions");
    if (width == 0 || height == 0) return cudaSuccess;
    BINCV_ASSERT(src != nullptr && dst.ptr != nullptr,
                 "uploadImage: a non-empty transfer needs non-null pointers");
    return cudaMemcpy2DAsync(dst.ptr, dst.stride * sizeof(T), src,
                             srcStrideElems * sizeof(T), width * sizeof(T), height,
                             cudaMemcpyHostToDevice, stream);
}

/// @brief Device wide image to a host strided pixel array.
template <typename T>
inline cudaError_t downloadImage(DeviceImageConstView<T> src, T* dst,
                                 size_t dstStrideElems,
                                 cudaStream_t stream = nullptr) {
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst != nullptr,
                 "downloadImage: a non-empty transfer needs non-null pointers");
    return cudaMemcpy2DAsync(dst, dstStrideElems * sizeof(T), src.ptr,
                             src.stride * sizeof(T), src.width * sizeof(T), src.height,
                             cudaMemcpyDeviceToHost, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
