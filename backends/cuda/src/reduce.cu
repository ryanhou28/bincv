// The device arm of the bulk reductions. The clip geometry is the HOST's own
// impl::clipRegion -- included, not copied -- so the two backends cannot drift
// on what a Rect means; only the traversal is forked.
//
// Two traversals, each right for its shape (see reduce.hpp):
//   * grid-stride + one atomic per warp, for a SINGLE region that may be a
//     whole frame;
//   * one block per region, no atomics at all, for a BATCH of windows.
// Both are integer addition over the same masked words, so they agree exactly.

#include "bincv/cuda/reduce.hpp"

#include "bincv/ops/reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief The region geometry a kernel needs, flattened out of the host's
/// RegionWords so it can be a kernel argument.
struct RegionArg {
    size_t y0, y1, firstWord, lastWord;
    uint32_t headMask, tailMask;
};

RegionArg toArg(const impl::RegionWords<uint32_t>& r) {
    RegionArg a{};
    a.y0 = r.y0;
    a.y1 = r.y1;
    a.firstWord = r.firstWord;
    a.lastWord = r.lastWord;
    a.headMask = r.headMask;
    a.tailMask = r.tailMask;
    return a;
}

/// @brief The bits of word `w` that lie inside the region.
__device__ __forceinline__ uint32_t wordMask(const RegionArg& r, size_t w) {
    uint32_t m = 0xFFFFFFFFu;
    if (w == r.firstWord) m &= r.headMask;
    if (w == r.lastWord) m &= r.tailMask;
    return m;
}

/// @brief The four covariance counters, accumulated per thread.
struct Quad {
    unsigned long long xx, yy, whenClear, whenSet;
};

__device__ __forceinline__ void warpReduce(Quad& q) {
    for (int off = 16; off > 0; off >>= 1) {
        q.xx += __shfl_down_sync(0xFFFFFFFFu, q.xx, off);
        q.yy += __shfl_down_sync(0xFFFFFFFFu, q.yy, off);
        q.whenClear += __shfl_down_sync(0xFFFFFFFFu, q.whenClear, off);
        q.whenSet += __shfl_down_sync(0xFFFFFFFFu, q.whenSet, off);
    }
}

// ---------------------------------------------------------------------------
// The result is written through `emit`, overloaded on the caller's OUTPUT TYPE.
// A plain count writes one counter, a split writes two, a covariance four --
// each into exactly the object the caller allocated, so no kernel can write
// past a smaller result type. (An earlier draft aliased a two-counter result
// onto a four-counter struct and relied on the other two staying zero; that is
// the shape of bug this project keeps finding, so it is typed instead.)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void emit(unsigned long long* out, const Quad& q) {
    if (q.xx) atomicAdd(out, q.xx);
}
__device__ __forceinline__ void emit(DeviceSplitCount* out, const Quad& q) {
    if (q.whenClear) atomicAdd(&out->whenClear, q.whenClear);
    if (q.whenSet) atomicAdd(&out->whenSet, q.whenSet);
}
__device__ __forceinline__ void emit(DeviceCovarianceCount* out, const Quad& q) {
    if (q.xx) atomicAdd(&out->xx, q.xx);
    if (q.yy) atomicAdd(&out->yy, q.yy);
    if (q.whenClear) atomicAdd(&out->whenClear, q.whenClear);
    if (q.whenSet) atomicAdd(&out->whenSet, q.whenSet);
}

// ---------------------------------------------------------------------------
// What one word contributes. `Mode` selects how much of the quad is live, so a
// plain count does not pay for the covariance's loads: the branches are on a
// compile-time constant and vanish.
// ---------------------------------------------------------------------------
enum class Mode { Count, And, Split, SplitXor, Cov, CovXor };

template <Mode M>
__device__ __forceinline__ void accumulateWord(Quad& q, const uint32_t* ra,
                                               const uint32_t* rb, const uint32_t* rc0,
                                               const uint32_t* rc1, size_t w,
                                               uint32_t mask) {
    const uint32_t av = ra[w] & mask;
    if (M == Mode::Count) {
        q.xx += static_cast<unsigned>(__popc(av));
        return;
    }
    const uint32_t bv = rb[w];
    const uint32_t ab = av & bv;
    if (M == Mode::And) {
        q.xx += static_cast<unsigned>(__popc(ab));
        return;
    }
    if (M == Mode::Cov || M == Mode::CovXor) {
        q.xx += static_cast<unsigned>(__popc(av));
        q.yy += static_cast<unsigned>(__popc(bv & mask));
    }
    // The selector, never complemented: whenClear is the difference of two
    // popcounts, which is what keeps a trailing word's padding out of the
    // count (the host kernel's note).
    const uint32_t sel =
        (M == Mode::SplitXor || M == Mode::CovXor) ? (rc0[w] ^ rc1[w]) : rc0[w];
    const unsigned both = static_cast<unsigned>(__popc(ab));
    const unsigned set = static_cast<unsigned>(__popc(ab & sel));
    q.whenSet += set;
    q.whenClear += both - set;
}

// ---------------------------------------------------------------------------
// Single region: grid-stride over (row, word), one atomic per warp.
// ---------------------------------------------------------------------------
template <Mode M, typename OutT>
__global__ void regionKernel(DeviceBinMatConstView a, DeviceBinMatConstView b,
                             DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                             RegionArg r, OutT* out) {
    const size_t rowSpan = r.lastWord - r.firstWord + 1;
    const size_t total = (r.y1 - r.y0) * rowSpan;
    Quad q{0, 0, 0, 0};
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t ry = idx / rowSpan;
        const size_t w = r.firstWord + (idx - ry * rowSpan);
        const size_t y = r.y0 + ry;
        accumulateWord<M>(q, a.row(y), b.row(y), c0.row(y), c1.row(y), w, wordMask(r, w));
    }
    warpReduce(q);
    if ((threadIdx.x & 31u) == 0) emit(out, q);
}

// ---------------------------------------------------------------------------
// Batch: one block per region, block-reduced, no atomics. Each block owns its
// output, so nothing contends and nothing needs zeroing beforehand.
// ---------------------------------------------------------------------------
template <Mode M>
__global__ void batchKernel(DeviceBinMatConstView a, DeviceBinMatConstView b,
                            DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                            const Rect* regions, size_t count, size_t width,
                            size_t height, DeviceCovarianceCount* out) {
    const size_t idx = blockIdx.x;
    if (idx >= count) return;
    // The clip restated for device code, from the host's own rules: half-open,
    // clamped to the view, negative origins legal, empty yields zeros.
    const Rect rect = regions[idx];
    Quad q{0, 0, 0, 0};
    long long x0 = rect.x, x1 = rect.x + static_cast<long long>(rect.width);
    long long y0 = rect.y, y1 = rect.y + static_cast<long long>(rect.height);
    if (rect.width > 0 && rect.height > 0 && x1 > 0 && y1 > 0 &&
        x0 < static_cast<long long>(width) && y0 < static_cast<long long>(height)) {
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > static_cast<long long>(width)) x1 = static_cast<long long>(width);
        if (y1 > static_cast<long long>(height)) y1 = static_cast<long long>(height);
        if (x0 < x1 && y0 < y1) {
            RegionArg r{};
            r.y0 = static_cast<size_t>(y0);
            r.y1 = static_cast<size_t>(y1);
            r.firstWord = static_cast<size_t>(x0) >> 5;
            r.lastWord = static_cast<size_t>(x1 - 1) >> 5;
            r.headMask = 0xFFFFFFFFu << (static_cast<size_t>(x0) & 31u);
            const unsigned hiBit = static_cast<unsigned>((x1 - 1) & 31);
            r.tailMask = (hiBit == 31u) ? 0xFFFFFFFFu : ((1u << (hiBit + 1)) - 1u);
            const size_t rowSpan = r.lastWord - r.firstWord + 1;
            const size_t total = (r.y1 - r.y0) * rowSpan;
            for (size_t i = threadIdx.x; i < total; i += blockDim.x) {
                const size_t ry = i / rowSpan;
                const size_t w = r.firstWord + (i - ry * rowSpan);
                const size_t y = r.y0 + ry;
                accumulateWord<M>(q, a.row(y), b.row(y), c0.row(y), c1.row(y), w,
                                  wordMask(r, w));
            }
        }
    }
    warpReduce(q);
    // Block reduce over warps: at most 32 of them, folded through shared.
    __shared__ Quad warpSums[32];
    const unsigned warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0) warpSums[warp] = q;
    __syncthreads();
    if (threadIdx.x == 0) {
        Quad t{0, 0, 0, 0};
        const unsigned warps = (blockDim.x + 31u) / 32u;
        for (unsigned i = 0; i < warps; ++i) {
            t.xx += warpSums[i].xx;
            t.yy += warpSums[i].yy;
            t.whenClear += warpSums[i].whenClear;
            t.whenSet += warpSums[i].whenSet;
        }
        out[idx].xx = t.xx;
        out[idx].yy = t.yy;
        out[idx].whenClear = t.whenClear;
        out[idx].whenSet = t.whenSet;
    }
}

constexpr unsigned kBlock = 256;
constexpr unsigned kBatchBlock = 64;  // a 31x31 window is ~60 word triples

unsigned gridFor(size_t total) {
    const size_t blocks = (total + kBlock - 1) / kBlock;
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

/// @brief Zero the result, then launch the single-region traversal.
template <Mode M, typename OutT>
cudaError_t launchRegion(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                         const impl::RegionWords<uint32_t>& r, OutT* dOut,
                         cudaStream_t stream) {
    cudaError_t err = cudaMemsetAsync(dOut, 0, sizeof(OutT), stream);
    if (err != cudaSuccess) return err;
    if (r.isEmpty) return cudaSuccess;
    BINCV_ASSERT(a.ptr != nullptr, "cuda reduce: non-empty view, null pointer");
    const size_t total = (r.y1 - r.y0) * (r.lastWord - r.firstWord + 1);
    regionKernel<M, OutT><<<gridFor(total), kBlock, 0, stream>>>(a, b, c0, c1, toArg(r),
                                                                 dOut);
    return cudaGetLastError();
}

/// @brief One region, synchronously, into all four counters. The sync path
/// always allocates the widest result so there is ONE of it; which fields are
/// meaningful is the mode's business.
template <Mode M>
DeviceCovarianceCount regionSync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                 DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                 const impl::RegionWords<uint32_t>& r) {
    DeviceCovarianceCount* dOut = nullptr;
    BINCV_CUDA_CHECK(cudaMalloc(&dOut, sizeof(DeviceCovarianceCount)));
    cudaError_t err = launchRegion<M, DeviceCovarianceCount>(a, b, c0, c1, r, dOut,
                                                             nullptr);
    DeviceCovarianceCount host{0, 0, 0, 0};
    if (err == cudaSuccess)
        err = cudaMemcpy(&host, dOut, sizeof host, cudaMemcpyDeviceToHost);
    cudaFree(dOut);
    BINCV_CUDA_CHECK(err);
    return host;
}

impl::RegionWords<uint32_t> wholeOf(DeviceBinMatConstView v) {
    return impl::wholeViewWords<uint32_t>(v.width, v.height);
}
impl::RegionWords<uint32_t> clipOf(DeviceBinMatConstView v, Rect region) {
    return impl::clipRegion<uint32_t>(v.width, v.height, region);
}

/// @brief The shared precondition. The casts keep the parameters used in a
/// release build, where BINCV_ASSERT compiles away entirely.
void assertSameExtent(DeviceBinMatConstView a, DeviceBinMatConstView b) {
    BINCV_ASSERT(a.width == b.width && a.height == b.height,
                 "cuda reduce: masked reductions need views of the same size");
    static_cast<void>(a);
    static_cast<void>(b);
}

} // namespace

// ---------------------------------------------------------------------------
// countNonZero
// ---------------------------------------------------------------------------

cudaError_t countNonZeroAsync(DeviceBinMatConstView src, unsigned long long* dResult,
                              cudaStream_t stream) {
    BINCV_ASSERT(dResult != nullptr, "cuda countNonZero: null result pointer");
    return launchRegion<Mode::Count>(src, src, src, src, wholeOf(src), dResult, stream);
}

cudaError_t countNonZeroAsync(DeviceBinMatConstView src, Rect region,
                              unsigned long long* dResult, cudaStream_t stream) {
    BINCV_ASSERT(dResult != nullptr, "cuda countNonZero: null result pointer");
    return launchRegion<Mode::Count>(src, src, src, src, clipOf(src, region), dResult,
                                     stream);
}

size_t countNonZero(DeviceBinMatConstView src) {
    return static_cast<size_t>(regionSync<Mode::Count>(src, src, src, src, wholeOf(src)).xx);
}

size_t countNonZero(DeviceBinMatConstView src, Rect region) {
    return static_cast<size_t>(
        regionSync<Mode::Count>(src, src, src, src, clipOf(src, region)).xx);
}

// ---------------------------------------------------------------------------
// countAnd
// ---------------------------------------------------------------------------

cudaError_t countAndAsync(DeviceBinMatConstView a, DeviceBinMatConstView b, Rect region,
                          unsigned long long* dResult, cudaStream_t stream) {
    assertSameExtent(a, b);
    BINCV_ASSERT(dResult != nullptr, "cuda countAnd: null result pointer");
    return launchRegion<Mode::And>(a, b, a, a, clipOf(a, region), dResult, stream);
}

size_t countAnd(DeviceBinMatConstView a, DeviceBinMatConstView b, Rect region) {
    assertSameExtent(a, b);
    return static_cast<size_t>(regionSync<Mode::And>(a, b, a, a, clipOf(a, region)).xx);
}

// ---------------------------------------------------------------------------
// countAndSplit
// ---------------------------------------------------------------------------

cudaError_t countAndSplitAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                               DeviceBinMatConstView c, Rect region,
                               DeviceSplitCount* dResult, cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c);
    BINCV_ASSERT(dResult != nullptr, "cuda countAndSplit: null result pointer");
    return launchRegion<Mode::Split>(a, b, c, c, clipOf(a, region), dResult, stream);
}

cudaError_t countAndSplitAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                               DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                               Rect region, DeviceSplitCount* dResult,
                               cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c0);
    assertSameExtent(a, c1);
    BINCV_ASSERT(dResult != nullptr, "cuda countAndSplit: null result pointer");
    return launchRegion<Mode::SplitXor>(a, b, c0, c1, clipOf(a, region), dResult, stream);
}

SplitCount countAndSplit(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatConstView c, Rect region) {
    assertSameExtent(a, b);
    assertSameExtent(a, c);
    const DeviceCovarianceCount q = regionSync<Mode::Split>(a, b, c, c, clipOf(a, region));
    return toHost(DeviceSplitCount{q.whenClear, q.whenSet});
}

SplitCount countAndSplit(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                         Rect region) {
    assertSameExtent(a, b);
    assertSameExtent(a, c0);
    assertSameExtent(a, c1);
    const DeviceCovarianceCount q =
        regionSync<Mode::SplitXor>(a, b, c0, c1, clipOf(a, region));
    return toHost(DeviceSplitCount{q.whenClear, q.whenSet});
}

// ---------------------------------------------------------------------------
// countCovariance
// ---------------------------------------------------------------------------

cudaError_t countCovarianceAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                 DeviceBinMatConstView c, Rect region,
                                 DeviceCovarianceCount* dResult, cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c);
    BINCV_ASSERT(dResult != nullptr, "cuda countCovariance: null result pointer");
    return launchRegion<Mode::Cov>(a, b, c, c, clipOf(a, region), dResult, stream);
}

cudaError_t countCovarianceAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                 DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                 Rect region, DeviceCovarianceCount* dResult,
                                 cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c0);
    assertSameExtent(a, c1);
    BINCV_ASSERT(dResult != nullptr, "cuda countCovariance: null result pointer");
    return launchRegion<Mode::CovXor>(a, b, c0, c1, clipOf(a, region), dResult, stream);
}

CovarianceCount countCovariance(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                DeviceBinMatConstView c, Rect region) {
    assertSameExtent(a, b);
    assertSameExtent(a, c);
    return toHost(regionSync<Mode::Cov>(a, b, c, c, clipOf(a, region)));
}

CovarianceCount countCovariance(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                Rect region) {
    assertSameExtent(a, b);
    assertSameExtent(a, c0);
    assertSameExtent(a, c1);
    return toHost(regionSync<Mode::CovXor>(a, b, c0, c1, clipOf(a, region)));
}

// ---------------------------------------------------------------------------
// The batch
// ---------------------------------------------------------------------------

cudaError_t countCovarianceBatchAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                      DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                      const Rect* dRegions, size_t count,
                                      DeviceCovarianceCount* dResults,
                                      cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c0);
    assertSameExtent(a, c1);
    if (count == 0) return cudaSuccess;
    BINCV_ASSERT(dRegions != nullptr && dResults != nullptr,
                 "cuda countCovarianceBatch: null region or result array");
    batchKernel<Mode::CovXor><<<static_cast<unsigned>(count), kBatchBlock, 0, stream>>>(
        a, b, c0, c1, dRegions, count, a.width, a.height, dResults);
    return cudaGetLastError();
}

cudaError_t countCovarianceBatchAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                      DeviceBinMatConstView c, const Rect* dRegions,
                                      size_t count, DeviceCovarianceCount* dResults,
                                      cudaStream_t stream) {
    assertSameExtent(a, b);
    assertSameExtent(a, c);
    if (count == 0) return cudaSuccess;
    BINCV_ASSERT(dRegions != nullptr && dResults != nullptr,
                 "cuda countCovarianceBatch: null region or result array");
    batchKernel<Mode::Cov><<<static_cast<unsigned>(count), kBatchBlock, 0, stream>>>(
        a, b, c, c, dRegions, count, a.width, a.height, dResults);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
