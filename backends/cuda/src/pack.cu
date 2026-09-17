// The device sensor stage. One warp produces one packed word: lane x of the
// warp reads pixel 32*i + x, evaluates the rule, and __ballot_sync IS the
// packed word -- the format's 32-bit granule and the warp width coinciding.

#include "bincv/cuda/pack.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

template <typename SrcT, PackRule R>
__global__ void packKernel(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                           SrcT t, size_t words) {
    const unsigned lane = threadIdx.x;  // 0..31
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * dst.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        bool pred = false;
        if (x < src.width) {
            const SrcT v = src.row(y)[x];
            if (R == PackRule::NonZero) pred = v != SrcT{0};
            if (R == PackRule::GreaterThan) pred = v > t;
            if (R == PackRule::GreaterEqual) pred = v >= t;
        }
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == 0) dst.row(y)[i] = word;
    }
}

template <typename SrcT>
cudaError_t launchPack(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                       PackRule rule, SrcT t, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda packBits: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda packBits: a non-empty image needs non-null pointers");
    const size_t words = rowWords(dst.width);
    const dim3 block(32, 8);  // eight warps, eight words per block iteration
    const size_t warps = (words * dst.height + block.y - 1) / block.y;
    const unsigned grid = static_cast<unsigned>(warps < 4096 ? (warps ? warps : 1)
                                                             : 4096);
    switch (rule) {
        case PackRule::NonZero:
            packKernel<SrcT, PackRule::NonZero>
                <<<grid, block, 0, stream>>>(src, dst, t, words);
            break;
        case PackRule::GreaterThan:
            packKernel<SrcT, PackRule::GreaterThan>
                <<<grid, block, 0, stream>>>(src, dst, t, words);
            break;
        case PackRule::GreaterEqual:
            packKernel<SrcT, PackRule::GreaterEqual>
                <<<grid, block, 0, stream>>>(src, dst, t, words);
            break;
    }
    return cudaGetLastError();
}

} // namespace

cudaError_t packBits(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     PackRule rule, uint8_t threshold, cudaStream_t stream) {
    return launchPack<uint8_t>(src, dst, rule, threshold, stream);
}

cudaError_t packBits(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     PackRule rule, uint16_t threshold, cudaStream_t stream) {
    return launchPack<uint16_t>(src, dst, rule, threshold, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
