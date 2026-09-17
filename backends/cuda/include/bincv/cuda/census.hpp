#pragma once

/// @file census.hpp
/// @brief The census transform on the device: a wide image in GPU memory into
/// K comparison bit-planes, semantics identical to ops/census.hpp --
/// `bit = I(p + offset) > I(p)`, out-of-frame comparisons write 0, padding
/// bits end zero. The pattern types and tables are the HOST's own.
///
/// PLANE BLOCK LAYOUT: the K planes live in ONE device matrix of height
/// K * imageHeight, plane k occupying rows [k * H, (k + 1) * H) at the common
/// stride. One allocation, one stride, and the dense-disparity census entry
/// consumes the same shape.

#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/census.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief The pattern as plain kernel-argument data. **INTERNAL.**
struct CensusOffsetsPod {
    int8_t dx[32];
    int8_t dy[32];
    int planes;
};

cudaError_t censusTransformImpl(DeviceImageConstView<uint8_t> img,
                                const CensusOffsetsPod& pattern,
                                DeviceBinMatView planeBlock, cudaStream_t stream);
cudaError_t censusTransformImpl(DeviceImageConstView<uint16_t> img,
                                const CensusOffsetsPod& pattern,
                                DeviceBinMatView planeBlock, cudaStream_t stream);

template <size_t K, typename SrcT>
inline cudaError_t censusTransformDispatch(DeviceImageConstView<SrcT> img,
                                           const CensusPattern<K>& pattern,
                                           DeviceBinMatView planeBlock,
                                           cudaStream_t stream) {
    static_assert(K >= 1 && K <= 32, "cuda censusTransform: 1 to 32 offsets");
    BINCV_ASSERT(planeBlock.width == img.width &&
                     planeBlock.height == K * img.height,
                 "cuda censusTransform: plane block must be width x (K * height)");
    CensusOffsetsPod pod{};
    pod.planes = static_cast<int>(K);
    for (size_t k = 0; k < K; ++k) {
        pod.dx[k] = pattern.at[k].dx;
        pod.dy[k] = pattern.at[k].dy;
    }
    return censusTransformImpl(img, pod, planeBlock, stream);
}

} // namespace impl

/// @brief Census transform into a K-plane block. Device twin of the host
/// censusTransform, bit-identical to it by test.
template <size_t K>
inline cudaError_t censusTransform(DeviceImageConstView<uint8_t> img,
                                   const CensusPattern<K>& pattern,
                                   DeviceBinMatView planeBlock,
                                   cudaStream_t stream = nullptr) {
    return impl::censusTransformDispatch<K, uint8_t>(img, pattern, planeBlock, stream);
}
template <size_t K>
inline cudaError_t censusTransform(DeviceImageConstView<uint16_t> img,
                                   const CensusPattern<K>& pattern,
                                   DeviceBinMatView planeBlock,
                                   cudaStream_t stream = nullptr) {
    return impl::censusTransformDispatch<K, uint16_t>(img, pattern, planeBlock, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
