#pragma once

/// @file packCustom.cuh
/// @brief `packBitsIf` and `packQuantWith` for the device: packing under a rule
/// the shipped enums cannot express.
///
/// ---------------------------------------------------------------------------
/// THIS HEADER REQUIRES AN nvcc-COMPILED CALLER, AND THAT IS WHY IT IS `.cuh`
///
/// Everything else in this backend presents plain `.hpp` headers that an
/// ordinary C++ translation unit includes; the kernels live in the compiled
/// library. That cannot work for an arbitrary caller predicate: the kernel must
/// be instantiated with the caller's own functor, so it must be compiled where
/// that functor is visible -- by nvcc. The file extension says so before the
/// first compiler error does.
///
/// The host's equivalents carry the same trade in a different currency:
/// `packBits`' shipped rules are template parameters so the comparison inlines
/// (measured 46x on x86 / 14x on aarch64), and `packBitsIf` is "honestly
/// slower" for taking a predicate. Here the cost is not speed -- a device
/// functor inlines fine -- it is that the caller must be a CUDA translation
/// unit. A caller who can live with the three shipped rules should use
/// `pack.hpp` and stay in plain C++.
///
/// The predicate is called on the DEVICE, once per pixel, and must be
/// `__device__`-callable: a functor with a `__device__ operator()`, or a lambda
/// compiled with `--extended-lambda`.

#include <cstdint>

#include <cuda_runtime.h>

#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

template <typename SrcT, typename Pred>
__global__ void packBitsIfKernel(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                                 Pred pred, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        const bool bit = (x < src.width) && pred(src.row(y)[x]);
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, bit);
        if (lane == 0) dst.row(y)[i] = word;
    }
}

template <typename SrcT, typename Map>
__global__ void packQuantWithKernel(DeviceImageConstView<SrcT> src,
                                    DeviceBinMatView planeBlock, Map map, unsigned n,
                                    size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        const unsigned value = (x < src.width) ? map(src.row(y)[x]) : 0u;
        for (unsigned p = 0; p < n; ++p) {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((value >> p) & 1u) != 0u);
            if (lane == 0)
                planeBlock.row(static_cast<size_t>(p) * src.height + y)[i] = word;
        }
    }
}

inline dim3 warpGridCustom(size_t words, size_t height, const dim3& block) {
    const size_t warps = (words * height + block.y - 1) / block.y;
    return dim3(static_cast<unsigned>(warps < 4096 ? (warps ? warps : 1) : 4096));
}

} // namespace impl

/// @brief `packBits` with an arbitrary per-pixel predicate. **API TIER 3.**
/// @param pred `__device__`-callable, `bool(SrcT)`; a set bit is where it
/// returns true.
/// @note Padding bits are zero on return: lanes past `width` never consult the
/// predicate and contribute 0 to the ballot.
template <typename SrcT, typename Pred>
inline cudaError_t packBitsIf(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                              Pred pred, cudaStream_t stream = nullptr) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda packBitsIf: src and dst must have the same dimensions");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda packBitsIf: a non-empty image needs non-null pointers");
    const size_t words = rowWords(dst.width);
    const dim3 block(32, 8);
    impl::packBitsIfKernel<SrcT, Pred>
        <<<impl::warpGridCustom(words, src.height, block), block, 0, stream>>>(src, dst,
                                                                               pred,
                                                                               words);
    return cudaGetLastError();
}

/// @brief `packQuant` with an arbitrary per-pixel map. **API TIER 3.**
/// @param map `__device__`-callable, `unsigned(SrcT)`, returning a value in
/// `[0, 2^n)`. Values above that are truncated to the planes that exist,
/// exactly as the host form truncates.
/// @param planeBlock N planes in one matrix; see `pack.hpp`'s packQuant.
template <typename SrcT, typename Map>
inline cudaError_t packQuantWith(DeviceImageConstView<SrcT> src,
                                 DeviceBinMatView planeBlock, size_t n, Map map,
                                 cudaStream_t stream = nullptr) {
    BINCV_ASSERT(n >= 1 && n <= 8,
                 "cuda packQuantWith: N outside QuantMat's supported range");
    BINCV_ASSERT(src.width == planeBlock.width && planeBlock.height == n * src.height,
                 "cuda packQuantWith: plane block must be width x (N * height)");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && planeBlock.ptr != nullptr,
                 "cuda packQuantWith: a non-empty image needs non-null pointers");
    const size_t words = rowWords(planeBlock.width);
    const dim3 block(32, 8);
    impl::packQuantWithKernel<SrcT, Map>
        <<<impl::warpGridCustom(words, src.height, block), block, 0, stream>>>(
            src, planeBlock, map, static_cast<unsigned>(n), words);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
