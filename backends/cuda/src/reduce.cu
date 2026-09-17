// The device arm of the bulk reductions. The clip geometry is the HOST's own
// impl::clipRegion -- included, not copied -- so the two backends cannot drift
// on what a Rect means; only the traversal is forked.

#include "bincv/cuda/reduce.hpp"

#include "bincv/ops/reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

__global__ void countKernel(DeviceBinMatConstView src, size_t y0, size_t y1,
                            size_t firstWord, size_t lastWord, uint32_t headMask,
                            uint32_t tailMask, unsigned long long* out) {
    const size_t rowSpan = lastWord - firstWord + 1;
    const size_t total = (y1 - y0) * rowSpan;
    unsigned long long local = 0;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t ry = idx / rowSpan;
        const size_t w = firstWord + (idx - ry * rowSpan);
        uint32_t mask = 0xFFFFFFFFu;
        if (w == firstWord) mask &= headMask;
        if (w == lastWord) mask &= tailMask;
        local += static_cast<unsigned>(__popc(src.row(y0 + ry)[w] & mask));
    }
    // Warp-reduce, then one atomic per warp: exact integer arithmetic, so the
    // combination order cannot change the answer.
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xFFFFFFFFu, local, off);
    if ((threadIdx.x & 31u) == 0 && local != 0) atomicAdd(out, local);
}

cudaError_t launchCount(DeviceBinMatConstView src,
                        const impl::RegionWords<uint32_t>& r,
                        unsigned long long* dResult, cudaStream_t stream) {
    BINCV_ASSERT(dResult != nullptr, "cuda countNonZero: null result pointer");
    cudaError_t err = cudaMemsetAsync(dResult, 0, sizeof(unsigned long long), stream);
    if (err != cudaSuccess) return err;
    if (r.isEmpty) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr, "cuda countNonZero: non-empty view, null pointer");
    const size_t total = (r.y1 - r.y0) * (r.lastWord - r.firstWord + 1);
    constexpr unsigned kBlock = 256;
    const size_t blocks = (total + kBlock - 1) / kBlock;
    const unsigned grid = static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
    countKernel<<<grid, kBlock, 0, stream>>>(src, r.y0, r.y1, r.firstWord, r.lastWord,
                                             r.headMask, r.tailMask, dResult);
    return cudaGetLastError();
}

size_t countSync(DeviceBinMatConstView src, const impl::RegionWords<uint32_t>& r) {
    unsigned long long* dResult = nullptr;
    BINCV_CUDA_CHECK(cudaMalloc(&dResult, sizeof(unsigned long long)));
    const cudaError_t err = launchCount(src, r, dResult, nullptr);
    unsigned long long host = 0;
    if (err == cudaSuccess) {
        const cudaError_t copyErr = cudaMemcpy(&host, dResult, sizeof host,
                                               cudaMemcpyDeviceToHost);
        cudaFree(dResult);
        BINCV_CUDA_CHECK(copyErr);
    } else {
        cudaFree(dResult);
        BINCV_CUDA_CHECK(err);
    }
    return static_cast<size_t>(host);
}

} // namespace

cudaError_t countNonZeroAsync(DeviceBinMatConstView src, unsigned long long* dResult,
                              cudaStream_t stream) {
    return launchCount(src, impl::wholeViewWords<uint32_t>(src.width, src.height),
                       dResult, stream);
}

cudaError_t countNonZeroAsync(DeviceBinMatConstView src, Rect region,
                              unsigned long long* dResult, cudaStream_t stream) {
    return launchCount(src, impl::clipRegion<uint32_t>(src.width, src.height, region),
                       dResult, stream);
}

size_t countNonZero(DeviceBinMatConstView src) {
    return countSync(src, impl::wholeViewWords<uint32_t>(src.width, src.height));
}

size_t countNonZero(DeviceBinMatConstView src, Rect region) {
    return countSync(src, impl::clipRegion<uint32_t>(src.width, src.height, region));
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
