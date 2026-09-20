// The device arm of ops/threshold.hpp. `threshold` is header-only -- one cutoff
// and a dispatch to the packer -- so what is here is `binarize`: a grid-stride
// loop over (row, word), one thread per OUTPUT word, with the host's own
// thresholdGE doing the comparison.
//
// The source is already bit-sliced, so there is no packing to do and no ballot
// to run: a thread reads N plane words at one index and resolves 32 pixels in
// ~2N word operations. That is this operation's structural advantage over a
// byte-per-pixel threshold, and it is why the op exists.

#include "bincv/cuda/threshold.hpp"

#include "bincv/ops/bitslice.hpp"  // thresholdGE -- BINCV_HOST_DEVICE, called

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief The largest plane count `binarize` accepts, and the host's own bound.
constexpr unsigned kMaxBinarizePlanes = 32;

/// @brief One thread, one output word. `kPlanes` is a compile-time constant so
/// `gathered` is statically indexed and lives in registers.
template <unsigned kPlanes>
__global__ void binarizeKernel(DevicePlaneBlockConstView planes, DeviceBinMatView dst,
                               unsigned geThreshold, size_t words, uint32_t tailMask) {
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;

        uint32_t gathered[kPlanes];
#pragma unroll
        for (unsigned p = 0; p < kPlanes; ++p) gathered[p] = planes.row(p, y)[i];

        // The HOST's comparison, not a device copy of it. Its per-plane branch is
        // on a bit of the threshold -- a kernel argument -- so it is uniform
        // across the warp and costs no divergence.
        uint32_t v = thresholdGE<uint32_t>(gathered, kPlanes, geThreshold);

        // Load-bearing, not decorative: thresholdGE answers every lane in the
        // word including the ones past `width`, and returns all ones at
        // threshold 0 whatever the planes hold.
        if (i == words - 1) v &= tailMask;
        dst.row(y)[i] = v;
    }
}

unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

/// @brief Dispatches the plane count to the templated kernel.
/// @note A switch over 32 instantiations of a ~15-line kernel, written out by a
/// recursive template rather than by hand so the list cannot lose a case.
template <unsigned kPlanes>
struct BinarizeDispatch {
    static void launch(unsigned n, DevicePlaneBlockConstView planes, DeviceBinMatView dst,
                       unsigned geThreshold, size_t words, uint32_t tailMask,
                       unsigned grid, unsigned block, cudaStream_t stream) {
        if (n == kPlanes) {
            binarizeKernel<kPlanes><<<grid, block, 0, stream>>>(planes, dst, geThreshold,
                                                                words, tailMask);
            return;
        }
        BinarizeDispatch<kPlanes - 1>::launch(n, planes, dst, geThreshold, words, tailMask,
                                              grid, block, stream);
    }
};

template <>
struct BinarizeDispatch<1> {
    static void launch(unsigned n, DevicePlaneBlockConstView planes, DeviceBinMatView dst,
                       unsigned geThreshold, size_t words, uint32_t tailMask,
                       unsigned grid, unsigned block, cudaStream_t stream) {
        (void)n;
        binarizeKernel<1><<<grid, block, 0, stream>>>(planes, dst, geThreshold, words,
                                                      tailMask);
    }
};

} // namespace

cudaError_t binarize(DevicePlaneBlockConstView planes, DeviceBinMatView dst,
                     unsigned thresh, cudaStream_t stream) {
    // The device domain, asserted where the host's template bound cannot reach:
    // a runtime plane count. Outside it there is no correct answer to give, so
    // it is an error rather than a clamp.
    BINCV_ASSERT(planes.planes >= 1 && planes.planes <= kMaxBinarizePlanes,
                 "cuda binarize: plane count outside 1..32");
    if (planes.planes < 1 || planes.planes > kMaxBinarizePlanes)
        return cudaErrorInvalidValue;
    BINCV_ASSERT(planes.width == dst.width && planes.height == dst.height,
                 "cuda binarize: every source plane must have dst's dimensions");
    BINCV_ASSERT(dst.stride >= rowWords(dst.width) && planes.stride >= rowWords(dst.width),
                 "cuda binarize: every view's stride must cover a whole row");

    if (dst.width == 0 || dst.height == 0) return cudaSuccess;

    BINCV_ASSERT(planes.ptr != nullptr && dst.ptr != nullptr,
                 "cuda binarize: a non-empty view needs a non-null pointer");
    // The host's aliasing contract, restated: dst must share no word with any
    // plane. A grid-stride loop gives no ordering between the thread writing
    // word i and the thread reading plane p's word i, so even the host's
    // read-before-write reasoning -- which lets ops/logic.hpp permit an exact
    // alias -- does not apply here.
    BINCV_ASSERT(dst.ptr + dst.height * dst.stride <= planes.ptr ||
                     planes.ptr + planes.planes * planes.planeWords() <= dst.ptr,
                 "cuda binarize: dst must share no word with the plane block");

    const size_t words = rowWords(dst.width);
    const unsigned n = static_cast<unsigned>(planes.planes);

    // A threshold no N-bit pixel can exceed selects nothing. Before the launch so
    // that `thresh + 1` below cannot wrap, and so the common all-zero answer costs
    // one memset instead of N loads per word. Computed at 64-bit width: at n == 32
    // `1u << 32` is undefined behaviour, which is the trap the host guards with a
    // constexpr branch and a runtime count cannot.
    const unsigned maxValue = static_cast<unsigned>((1ull << n) - 1ull);
    if (thresh >= maxValue) {
        return cudaMemset2DAsync(dst.ptr, dst.stride * sizeof(uint32_t), 0,
                                 words * sizeof(uint32_t), dst.height, stream);
    }

    // `>` against `thresh` is `>=` against `thresh + 1`; thresholdGE answers the
    // latter.
    const unsigned geThreshold = thresh + 1u;
    constexpr unsigned kBlock = 256;
    BinarizeDispatch<kMaxBinarizePlanes>::launch(
        n, planes, dst, geThreshold, words, rowTailMask(dst.width),
        gridFor(words * dst.height, kBlock), kBlock, stream);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
