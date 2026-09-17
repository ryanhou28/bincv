// The device census transform: pack.cu's ballot packer with the predicate
// swapped for the neighbour comparison. Plane k of the block is written by
// grid slice z = k, so all K planes of a launch proceed concurrently.

#include "bincv/cuda/census.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {
namespace {

template <typename SrcT>
__global__ void censusKernel(DeviceImageConstView<SrcT> img, CensusOffsetsPod pattern,
                             DeviceBinMatView planeBlock, size_t words) {
    const unsigned lane = threadIdx.x;
    const int k = static_cast<int>(blockIdx.z);
    const long long dx = pattern.dx[k];
    const long long dy = pattern.dy[k];
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * img.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        bool pred = false;
        const long long yn = static_cast<long long>(y) + dy;
        if (x < img.width && yn >= 0 && yn < static_cast<long long>(img.height)) {
            const long long xn = static_cast<long long>(x) + dx;
            if (xn >= 0 && xn < static_cast<long long>(img.width)) {
                pred = img.row(static_cast<size_t>(yn))[xn] > img.row(y)[x];
            }
        }
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == 0)
            planeBlock.row(static_cast<size_t>(k) * img.height + y)[i] = word;
    }
}

template <typename SrcT>
cudaError_t launchCensus(DeviceImageConstView<SrcT> img, const CensusOffsetsPod& pattern,
                         DeviceBinMatView planeBlock, cudaStream_t stream) {
    if (img.width == 0 || img.height == 0) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr && planeBlock.ptr != nullptr,
                 "cuda censusTransform: a non-empty image needs non-null pointers");
    const size_t words = rowWords(img.width);
    const dim3 block(32, 8);
    const size_t warps = (words * img.height + block.y - 1) / block.y;
    const dim3 grid(static_cast<unsigned>(warps < 2048 ? (warps ? warps : 1) : 2048),
                    1, static_cast<unsigned>(pattern.planes));
    censusKernel<SrcT><<<grid, block, 0, stream>>>(img, pattern, planeBlock, words);
    return cudaGetLastError();
}

} // namespace

cudaError_t censusTransformImpl(DeviceImageConstView<uint8_t> img,
                                const CensusOffsetsPod& pattern,
                                DeviceBinMatView planeBlock, cudaStream_t stream) {
    return launchCensus<uint8_t>(img, pattern, planeBlock, stream);
}

cudaError_t censusTransformImpl(DeviceImageConstView<uint16_t> img,
                                const CensusOffsetsPod& pattern,
                                DeviceBinMatView planeBlock, cudaStream_t stream) {
    return launchCensus<uint16_t>(img, pattern, planeBlock, stream);
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
