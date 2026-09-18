#pragma once

/// @file census.hpp
/// @brief The census transform on the device: a wide image in GPU memory into
/// K comparisons per pixel, semantics identical to ops/census.hpp --
/// `bit = I(p + offset) > I(p)`, out-of-frame comparisons write 0, padding
/// bits end zero. The pattern types and tables are the HOST's own.
///
/// TWO OUTPUT LAYOUTS, and which to use:
///
/// * `censusTransform` writes a PLANE BLOCK -- the K planes in one device
/// matrix of height `K * imageHeight`, plane k occupying rows
/// `[k * H, (k + 1) * H)` at the common stride. This is binCV's own
/// representation, so it compares plane-for-plane against the host.
/// * `censusTransformPacked` writes ONE WORD PER PIXEL, a pixel's whole
/// descriptor in a `uint32`. **This is what the dense matcher should
/// consume** -- see denseDisparity.hpp for the 8.5x and why.
///
/// Both come from the same pattern and the same comparison, and both are held
/// to the host's answer by test.

#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/census.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace impl {

/// @brief Force the PLANE transform's reference kernel, for the benchmark and
/// the tests. **INTERNAL.** Same contract as `denseFastArmEnabled`: a fast arm
/// must be switchable off, held to bit-exactness in one binary, and shown by
/// the benchmark to be the arm it timed.
/// @note The packed transform has a single implementation and so no switch;
/// what pins it is that its descriptors must drive the dense matcher to
/// the host's own disparity map, which the test suite checks.
bool& censusTiledEnabled();

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

cudaError_t censusTransformPackedImpl(DeviceImageConstView<uint8_t> img,
                                      const CensusOffsetsPod& pattern,
                                      DeviceImageView<uint32_t> dst, cudaStream_t stream);
cudaError_t censusTransformPackedImpl(DeviceImageConstView<uint16_t> img,
                                      const CensusOffsetsPod& pattern,
                                      DeviceImageView<uint32_t> dst, cudaStream_t stream);

/// @brief The pattern as a POD, shared by both layouts' dispatchers.
template <size_t K>
inline CensusOffsetsPod toPod(const CensusPattern<K>& pattern) {
    CensusOffsetsPod pod{};
    pod.planes = static_cast<int>(K);
    for (size_t k = 0; k < K; ++k) {
        pod.dx[k] = pattern.at[k].dx;
        pod.dy[k] = pattern.at[k].dy;
    }
    return pod;
}

template <size_t K, typename SrcT>
inline cudaError_t censusTransformPackedDispatch(DeviceImageConstView<SrcT> img,
                                                 const CensusPattern<K>& pattern,
                                                 DeviceImageView<uint32_t> dst,
                                                 cudaStream_t stream) {
    BINCV_ASSERT(dst.width == img.width && dst.height == img.height,
                 "censusTransformPacked: dst extent must match the image");
    return censusTransformPackedImpl(img, toPod(pattern), dst, stream);
}

template <size_t K, typename SrcT>
inline cudaError_t censusTransformDispatch(DeviceImageConstView<SrcT> img,
                                           const CensusPattern<K>& pattern,
                                           DeviceBinMatView planeBlock,
                                           cudaStream_t stream) {
    static_assert(K >= 1 && K <= 32, "cuda censusTransform: 1 to 32 offsets");
    BINCV_ASSERT(planeBlock.width == img.width &&
                     planeBlock.height == K * img.height,
                 "cuda censusTransform: plane block must be width x (K * height)");
    return censusTransformImpl(img, toPod(pattern), planeBlock, stream);
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

// ---------------------------------------------------------------------------
// THE PACKED DESCRIPTOR: a second layout, for the matcher that consumes it
// ---------------------------------------------------------------------------
//
// The plane block above is binCV's representation and stays the default: it is
// the host's layout, so plane-for-plane equality against the host census is a
// word comparison. But the dense matcher reads a census descriptor ONE PIXEL
// AT A TIME, across all K planes at once -- and in the plane block those K bits
// live in K different arrays, so a single pixel's descriptor costs K loads and
// its Hamming distance costs K popcounts, each counting one useful bit.
//
// Packed, a pixel's whole descriptor is one uint32. Hamming distance becomes
// `__popc(a ^ b)` -- ONE load, ONE xor, ONE popcount for all K comparisons,
// with K of 32 bits doing useful work instead of one.
//
// This is issue #34's anticipated case, and its answer: "add a second
// DOCUMENTED layout to the shared core rather than fork, because two
// independent definitions means neither can be checked against the other."
// Both layouts are produced from the same pattern by the same comparison rule,
// the tests hold the packed matcher's output map byte-equal to the host's, and
// the bit ORDER does not affect any answer -- Hamming distance is invariant
// under a permutation of the bits, as long as both images use one order.

/// @brief Census transform into ONE WORD PER PIXEL: bit `k` of `dst(x, y)` is
/// `I(p + pattern.at[k]) > I(p)`. **API TIER 3.** K <= 32.
/// @param dst `width x height` of uint32, the matcher's input layout.
/// @note Same comparisons, same out-of-frame rule (a neighbour outside the
/// frame contributes 0) and same pattern as the plane form; only the
/// layout differs. `denseDisparityCensusPacked` consumes this.
template <size_t K>
inline cudaError_t censusTransformPacked(DeviceImageConstView<uint8_t> img,
                                         const CensusPattern<K>& pattern,
                                         DeviceImageView<uint32_t> dst,
                                         cudaStream_t stream = nullptr) {
    static_assert(K >= 1 && K <= 32,
                  "censusTransformPacked: a descriptor must fit one uint32");
    return impl::censusTransformPackedDispatch<K, uint8_t>(img, pattern, dst, stream);
}
template <size_t K>
inline cudaError_t censusTransformPacked(DeviceImageConstView<uint16_t> img,
                                         const CensusPattern<K>& pattern,
                                         DeviceImageView<uint32_t> dst,
                                         cudaStream_t stream = nullptr) {
    static_assert(K >= 1 && K <= 32,
                  "censusTransformPacked: a descriptor must fit one uint32");
    return impl::censusTransformPackedDispatch<K, uint16_t>(img, pattern, dst, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
