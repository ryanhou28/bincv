// The device arm of ops/logic.hpp. The kernel is a grid-stride loop over
// (row, word) with the row's stride read per view -- the same geometry the host
// row loop walks, one thread per word instead of one loop iteration.

#include "bincv/cuda/logic.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

struct OpAnd {
    __device__ static uint32_t apply(uint32_t a, uint32_t b) { return a & b; }
};
struct OpOr {
    __device__ static uint32_t apply(uint32_t a, uint32_t b) { return a | b; }
};
struct OpXor {
    __device__ static uint32_t apply(uint32_t a, uint32_t b) { return a ^ b; }
};

template <typename Op>
__global__ void binaryKernel(DeviceBinMatConstView a, DeviceBinMatConstView b,
                             DeviceBinMatView dst, size_t words, uint32_t tailMask) {
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;
        uint32_t v = Op::apply(a.row(y)[i], b.row(y)[i]);
        // The trailing word is stored masked, as the host kernel stores it, so
        // the destination's padding is clean even off a dirty wrapped input.
        if (i == words - 1) v &= tailMask;
        dst.row(y)[i] = v;
    }
}

__global__ void notKernel(DeviceBinMatConstView src, DeviceBinMatView dst,
                          size_t words, uint32_t tailMask) {
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;
        uint32_t v = ~src.row(y)[i];
        if (i == words - 1) v &= tailMask;
        dst.row(y)[i] = v;
    }
}

inline unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    // Grid-stride kernels cap the grid; 4096 blocks saturate this class of
    // device and the loop covers the rest.
    return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

template <typename Op>
cudaError_t launchBinary(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatView dst, cudaStream_t stream) {
    BINCV_ASSERT(a.width == dst.width && a.height == dst.height &&
                     b.width == dst.width && b.height == dst.height,
                 "cuda bitwise op: a, b and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && dst.ptr != nullptr,
                 "cuda bitwise op: a non-empty view needs a non-null pointer");
    const size_t words = rowWords(dst.width);
    const uint32_t tail = rowTailMask(dst.width);
    constexpr unsigned kBlock = 256;
    binaryKernel<Op><<<gridFor(words * dst.height, kBlock), kBlock, 0, stream>>>(
        a, b, dst, words, tail);
    return cudaGetLastError();
}

} // namespace

cudaError_t bitwiseAnd(DeviceBinMatConstView a, DeviceBinMatConstView b,
                       DeviceBinMatView dst, cudaStream_t stream) {
    return launchBinary<OpAnd>(a, b, dst, stream);
}

cudaError_t bitwiseOr(DeviceBinMatConstView a, DeviceBinMatConstView b,
                      DeviceBinMatView dst, cudaStream_t stream) {
    return launchBinary<OpOr>(a, b, dst, stream);
}

cudaError_t bitwiseXor(DeviceBinMatConstView a, DeviceBinMatConstView b,
                       DeviceBinMatView dst, cudaStream_t stream) {
    return launchBinary<OpXor>(a, b, dst, stream);
}

cudaError_t bitwiseNot(DeviceBinMatConstView src, DeviceBinMatView dst,
                       cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda bitwiseNot: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda bitwiseNot: a non-empty view needs a non-null pointer");
    const size_t words = rowWords(dst.width);
    constexpr unsigned kBlock = 256;
    notKernel<<<gridFor(words * dst.height, kBlock), kBlock, 0, stream>>>(
        src, dst, words, rowTailMask(dst.width));
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
