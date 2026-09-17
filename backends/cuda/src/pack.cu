// The device sensor stage. One warp produces one packed word: lane x of the
// warp reads pixel 32*i + x, evaluates the rule, and __ballot_sync IS the
// packed word -- the format's 32-bit granule and the warp width coinciding.
// The N-bit packer is the same shape, N ballots deep.

#include "bincv/cuda/pack.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

template <typename SrcT, PackRule R>
__global__ void packKernel(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                           size_t dstRow, SrcT t, size_t words) {
    const unsigned lane = threadIdx.x;  // 0..31
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
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
        if (lane == 0) dst.row(dstRow + y)[i] = word;
    }
}

/// @brief The host's `quantScale`, restated for device code: the SAME integer
/// expression, so the two produce the same level for every input.
template <typename SrcT>
__device__ __forceinline__ unsigned quantScaleDevice(SrcT v, unsigned maxValue) {
    const unsigned long long m = (sizeof(SrcT) >= 8)
                                     ? ~0ULL
                                     : ((1ULL << (sizeof(SrcT) * 8)) - 1ULL);
    return static_cast<unsigned>(
        (static_cast<unsigned long long>(v) * maxValue + m / 2ULL) / m);
}

template <typename SrcT>
__global__ void packQuantKernel(DeviceImageConstView<SrcT> src,
                                DeviceBinMatView planeBlock, unsigned n,
                                unsigned maxValue, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        // A lane past the row contributes 0 to every plane, which is the
        // padding invariant holding by construction rather than by masking.
        const unsigned value =
            (x < src.width) ? quantScaleDevice<SrcT>(src.row(y)[x], maxValue) : 0u;
        for (unsigned p = 0; p < n; ++p) {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((value >> p) & 1u) != 0u);
            if (lane == 0)
                planeBlock.row(static_cast<size_t>(p) * src.height + y)[i] = word;
        }
    }
}

__global__ void unpackKernel(DeviceBinMatConstView src, DeviceImageView<uint8_t> dst,
                             uint8_t onValue, uint8_t zeroValue) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y;
    if (x >= src.width || y >= src.height) return;
    const uint32_t word = src.row(y)[x >> 5];
    dst.row(y)[x] = ((word >> (x & 31u)) & 1u) ? onValue : zeroValue;
}

/// @brief Warp-per-word launch geometry, shared by the packers.
dim3 warpGrid(size_t words, size_t height, const dim3& block) {
    const size_t warps = (words * height + block.y - 1) / block.y;
    return dim3(static_cast<unsigned>(warps < 4096 ? (warps ? warps : 1) : 4096));
}

template <typename SrcT>
cudaError_t launchPack(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                       size_t dstRow, PackRule rule, SrcT t, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width, "cuda packRows: src and dst must share a width");
    BINCV_ASSERT(dstRow + src.height <= dst.height, "cuda packRows: chunk runs past dst");
    if (dst.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda packRows: a non-empty image needs non-null pointers");
    const size_t words = rowWords(dst.width);
    const dim3 block(32, 8);  // eight warps, eight words per block iteration
    const dim3 grid = warpGrid(words, src.height, block);
    switch (rule) {
        case PackRule::NonZero:
            packKernel<SrcT, PackRule::NonZero>
                <<<grid, block, 0, stream>>>(src, dst, dstRow, t, words);
            break;
        case PackRule::GreaterThan:
            packKernel<SrcT, PackRule::GreaterThan>
                <<<grid, block, 0, stream>>>(src, dst, dstRow, t, words);
            break;
        case PackRule::GreaterEqual:
            packKernel<SrcT, PackRule::GreaterEqual>
                <<<grid, block, 0, stream>>>(src, dst, dstRow, t, words);
            break;
    }
    return cudaGetLastError();
}

template <typename SrcT>
cudaError_t launchPackQuant(DeviceImageConstView<SrcT> src, DeviceBinMatView planeBlock,
                            size_t n, cudaStream_t stream) {
    BINCV_ASSERT(n >= 1 && n <= 8, "cuda packQuant: N outside QuantMat's supported range");
    BINCV_ASSERT(src.width == planeBlock.width &&
                     planeBlock.height == n * src.height,
                 "cuda packQuant: plane block must be width x (N * height)");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && planeBlock.ptr != nullptr,
                 "cuda packQuant: a non-empty image needs non-null pointers");
    const size_t words = rowWords(planeBlock.width);
    const dim3 block(32, 8);
    const dim3 grid = warpGrid(words, src.height, block);
    const unsigned maxValue = (1u << n) - 1u;
    packQuantKernel<SrcT><<<grid, block, 0, stream>>>(src, planeBlock,
                                                      static_cast<unsigned>(n), maxValue,
                                                      words);
    return cudaGetLastError();
}

} // namespace

cudaError_t packBits(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     PackRule rule, uint8_t threshold, cudaStream_t stream) {
    BINCV_ASSERT(src.height == dst.height,
                 "cuda packBits: src and dst must have the same dimensions");
    return launchPack<uint8_t>(src, dst, 0, rule, threshold, stream);
}

cudaError_t packBits(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     PackRule rule, uint16_t threshold, cudaStream_t stream) {
    BINCV_ASSERT(src.height == dst.height,
                 "cuda packBits: src and dst must have the same dimensions");
    return launchPack<uint16_t>(src, dst, 0, rule, threshold, stream);
}

cudaError_t packRows(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint8_t threshold,
                     cudaStream_t stream) {
    return launchPack<uint8_t>(src, dst, dstRow, rule, threshold, stream);
}

cudaError_t packRows(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint16_t threshold,
                     cudaStream_t stream) {
    return launchPack<uint16_t>(src, dst, dstRow, rule, threshold, stream);
}

cudaError_t packQuant(DeviceImageConstView<uint8_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream) {
    return launchPackQuant<uint8_t>(src, planeBlock, n, stream);
}

cudaError_t packQuant(DeviceImageConstView<uint16_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream) {
    return launchPackQuant<uint16_t>(src, planeBlock, n, stream);
}

cudaError_t unpackTo8Bit(DeviceBinMatConstView src, DeviceImageView<uint8_t> dst,
                         uint8_t onValue, uint8_t zeroValue, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda unpackTo8Bit: src and dst must have the same dimensions");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda unpackTo8Bit: a non-empty image needs non-null pointers");
    const dim3 block(256, 1);
    const dim3 grid(static_cast<unsigned>((src.width + block.x - 1) / block.x),
                    static_cast<unsigned>(src.height));
    unpackKernel<<<grid, block, 0, stream>>>(src, dst, onValue, zeroValue);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
