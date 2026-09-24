// The device arm of ops/covariance.hpp: the N-bit bit-sliced gradient
// covariance.
//
// WHAT IS SHARED AND WHAT IS FORKED:
//
//   SHARED, by call --
//     bincv::impl::clipRegion            the half-open, clamped, negative-origin
//                                        window rule, called ON THE DEVICE for
//                                        each Rect of a batch, exactly as
//                                        reduce.cu's batch does. A clip is an
//                                        invariant, not a kernel.
//     bincv::impl::combineBitSlicedPairs the epilogue: the 2^(i+j) weighting,
//                                        the doubled off-diagonal, and -- via
//                                        SplitCount::crossTerm -- the ONE signed
//                                        subtraction in the whole operation.
//     bincv::impl::lowBitsMask           the run mask for the aligned-window arm.
//
//   FORKED -- the traversal and the accumulator. The host walks rows and holds
//   size_t counters; a thread here holds 32-bit counters in registers across
//   its share of the window and never materializes a selector plane.
//
// THE COUNTER BOUND, STATED. Each counter rises by at most 32 per word visited,
// so `unsigned` overflows only after 2^32 / 32 = 134,217,728 word visits IN ONE
// THREAD -- 4.3e9 pixels of one window walked serially. No addressable image
// reaches it. The counters widen to size_t exactly once, in the epilogue, where
// the host's own weighting takes them.

#include "bincv/cuda/covariance.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// The per-thread plane-pair counters. 32-bit and register-resident; the host's
/// BitSlicedPairCounts is size_t and is built from these only in the epilogue.
/// @note `xx` and `yy` are written on the UPPER TRIANGLE only (j >= i):
/// `m_x[i] & m_x[j]` is symmetric, so the lower half would be the same
/// numbers counted twice, and the epilogue doubles the off-diagonal
/// instead. `xyT` / `xyS` are the full N x N, because x and y are different
/// images and that product is not symmetric. The untouched lower halves are
/// never read, so they cost no register.
template <size_t N>
struct PairAcc {
    unsigned xx[N][N];
    unsigned yy[N][N];
    unsigned xyT[N][N];
    unsigned xyS[N][N];
};

template <size_t N>
__device__ __forceinline__ void clearAcc(PairAcc<N>& a) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
#pragma unroll
        for (size_t j = 0; j < N; ++j) {
            a.xx[i][j] = 0u;
            a.yy[i][j] = 0u;
            a.xyT[i][j] = 0u;
            a.xyS[i][j] = 0u;
        }
    }
}

/// `3N^2 + N` popcounts over one word index -- 4 at N = 1, 14 at N = 2, 30 at
/// N = 3, 52 at N = 4, which is the host header's own count.
/// @note At i == j the diagonal term is `__popc(ax[i] & ax[i])`, i.e.
/// `__popc(ax[i])`, which is what makes N == 1 issue countCovariance's FOUR
/// popcounts and not five.
/// @note `~(s_x ^ s_y)` is never formed. `whenClear` is `total - set`, computed
/// in the epilogue -- one popcount cheaper, and it is also what keeps a
/// trailing word's padding bits out of the count. The selector is therefore
/// never masked and does not need to be: it is only ever ANDed with `both`,
/// which is.
template <size_t N>
__device__ __forceinline__ void accumulateWord(PairAcc<N>& acc, const uint32_t (&ax)[N],
                                               const uint32_t (&ay)[N], uint32_t sel) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
#pragma unroll
        for (size_t j = i; j < N; ++j) {
            acc.xx[i][j] += static_cast<unsigned>(__popc(static_cast<int>(ax[i] & ax[j])));
            acc.yy[i][j] += static_cast<unsigned>(__popc(static_cast<int>(ay[i] & ay[j])));
        }
#pragma unroll
        for (size_t j = 0; j < N; ++j) {
            const uint32_t both = ax[i] & ay[j];
            acc.xyT[i][j] += static_cast<unsigned>(__popc(static_cast<int>(both)));
            acc.xyS[i][j] += static_cast<unsigned>(__popc(static_cast<int>(both & sel)));
        }
    }
}

/// The bits of word `w` that lie inside the region.
__device__ __forceinline__ uint32_t wordMask(const bincv::impl::RegionWords<uint32_t>& r,
                                             size_t w) {
    uint32_t m = 0xFFFFFFFFu;
    if (w == r.firstWord) m &= r.headMask;
    if (w == r.lastWord) m &= r.tailMask;
    return m;
}

/// One word index of one row: `2N + 2` loads, the selector XORed in the loop
/// and never materialized as a fifth frame-sized plane.
template <size_t N>
__device__ __forceinline__ void loadWord(const DevicePlaneBlockConstView& dx,
                                         const DevicePlaneBlockConstView& dy, size_t y,
                                         size_t w, uint32_t mask, uint32_t (&ax)[N],
                                         uint32_t (&ay)[N], uint32_t& sel) {
#pragma unroll
    for (size_t q = 0; q < N; ++q) {
        ax[q] = dx.row(q, y)[w] & mask;
        ay[q] = dy.row(q, y)[w] & mask;
    }
    sel = dx.row(N, y)[w] ^ dy.row(N, y)[w];
}

/// THE WINDOW ALIGNED INTO ONE WORD. A run at most 32 pixels wide still
/// straddles two words on every row, so each masked `__popc` sees ~15.5 useful
/// bits of 32. One funnel shift puts the whole run in the low bits of one
/// register, and the row's word VISITS go from two to one -- at N = 1, eight
/// popcounts per row become four; at N = 2, twenty-eight become fourteen.
///
/// **IT SAVES POPCOUNTS AND NOT LOADS, WHICH IS WHY IT LOSES HERE.** Extracting
/// a run that straddles two words still has to READ both: `runWord` loads `lo`
/// and `hi`, so a 31-pixel row costs the same 2N + 2 loads either way and buys
/// half the popcounts for the price of 2N + 2 extra funnel shifts. At N = 1
/// this kernel is traffic-shaped -- 4 popcounts on 4 loads -- so the half it
/// saves was never the cost. See the measurement at the switch below.
/// @note `lastWord` is `firstWord` or `firstWord + 1` here: a run of at most 32
/// pixels starting in `firstWord` cannot reach a third word. When it is
/// `firstWord`, the high operand contributes nothing to the low `span` bits
/// and a zero is passed rather than a load past the run.
__device__ __forceinline__ uint32_t runWord(const uint32_t* row, size_t firstWord,
                                            size_t lastWord, unsigned shift) {
    const uint32_t lo = row[firstWord];
    const uint32_t hi = (lastWord > firstWord) ? row[lastWord] : 0u;
    return __funnelshift_r(lo, hi, shift);
}

template <size_t N>
__device__ __forceinline__ void loadRun(const DevicePlaneBlockConstView& dx,
                                        const DevicePlaneBlockConstView& dy, size_t y,
                                        const bincv::impl::RegionWords<uint32_t>& r,
                                        unsigned shift, uint32_t mask, uint32_t (&ax)[N],
                                        uint32_t (&ay)[N], uint32_t& sel) {
#pragma unroll
    for (size_t q = 0; q < N; ++q) {
        ax[q] = runWord(dx.row(q, y), r.firstWord, r.lastWord, shift) & mask;
        ay[q] = runWord(dy.row(q, y), r.firstWord, r.lastWord, shift) & mask;
    }
    sel = runWord(dx.row(N, y), r.firstWord, r.lastWord, shift) ^
          runWord(dy.row(N, y), r.firstWord, r.lastWord, shift);
}

template <size_t N>
__device__ __forceinline__ void warpReduceAcc(PairAcc<N>& a) {
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (size_t i = 0; i < N; ++i) {
#pragma unroll
            for (size_t j = i; j < N; ++j) {
                a.xx[i][j] += __shfl_down_sync(0xFFFFFFFFu, a.xx[i][j], off);
                a.yy[i][j] += __shfl_down_sync(0xFFFFFFFFu, a.yy[i][j], off);
            }
#pragma unroll
            for (size_t j = 0; j < N; ++j) {
                a.xyT[i][j] += __shfl_down_sync(0xFFFFFFFFu, a.xyT[i][j], off);
                a.xyS[i][j] += __shfl_down_sync(0xFFFFFFFFu, a.xyS[i][j], off);
            }
        }
    }
}

/// THE EPILOGUE IS THE HOST'S OWN FUNCTION. The counters widen here, and
/// `impl::combineBitSlicedPairs` applies the 2^(i+j) weighting, doubles the
/// off-diagonal and takes `SplitCount::crossTerm` for the signed subtraction.
/// Nothing about the weighting is restated in this file.
template <size_t N>
__device__ __forceinline__ DeviceGradientCovariance combine(const PairAcc<N>& a) {
    bincv::impl::BitSlicedPairCounts<N> c;
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
#pragma unroll
        for (size_t j = i; j < N; ++j) {
            c.xx[i][j] = a.xx[i][j];
            c.yy[i][j] = a.yy[i][j];
        }
#pragma unroll
        for (size_t j = 0; j < N; ++j) {
            c.xyTotal[i][j] = a.xyT[i][j];
            c.xySet[i][j] = a.xyS[i][j];
        }
    }
    const GradientCovariance g = bincv::impl::combineBitSlicedPairs<N>(c);
    DeviceGradientCovariance out;
    out.sumXX = static_cast<long long>(g.sumXX);
    out.sumYY = static_cast<long long>(g.sumYY);
    out.sumXY = static_cast<long long>(g.sumXY);
    return out;
}

/// One block's share of an already-clipped window.
/// @tparam AlignedRun The funnel-shift arm. **Its own gate is arithmetic, not a
/// threshold:** one funnel shift reaches 32 bits, so a clipped run wider
/// than 32 pixels takes the general per-word loop below whatever this
/// parameter says. That is the case the benchmark uses as its
/// gate-excluded control -- there the two instantiations run the same loop
/// and must time ~1.00x.
/// @note The two arms distribute DIFFERENT units across the block's threads,
/// which is the whole saving: aligned, a row is exactly one masked word, so
/// a thread takes a ROW; unaligned, it takes a (row, word) pair. At the
/// tracker's 31x31 that is 31 units instead of 62.
template <size_t N, bool AlignedRun>
__device__ __forceinline__ void walkShare(const DevicePlaneBlockConstView& dx,
                                          const DevicePlaneBlockConstView& dy,
                                          const bincv::impl::RegionWords<uint32_t>& r,
                                          PairAcc<N>& acc) {
    uint32_t ax[N];
    uint32_t ay[N];
    uint32_t sel;
    const size_t span = r.x1 - r.x0;
    if (AlignedRun && span <= 32) {
        const unsigned shift = static_cast<unsigned>(r.x0 & size_t{31});
        const uint32_t mask = bincv::impl::lowBitsMask<uint32_t>(span);
        for (size_t t = threadIdx.x; t < r.y1 - r.y0; t += blockDim.x) {
            loadRun<N>(dx, dy, r.y0 + t, r, shift, mask, ax, ay, sel);
            accumulateWord<N>(acc, ax, ay, sel);
        }
        return;
    }
    const size_t rowSpan = r.lastWord - r.firstWord + 1;
    const size_t total = (r.y1 - r.y0) * rowSpan;
    for (size_t t = threadIdx.x; t < total; t += blockDim.x) {
        const size_t ry = t / rowSpan;
        const size_t w = r.firstWord + (t - ry * rowSpan);
        loadWord<N>(dx, dy, r.y0 + ry, w, wordMask(r, w), ax, ay, sel);
        accumulateWord<N>(acc, ax, ay, sel);
    }
}

constexpr unsigned kBatchBlock = 64;  // one block per window; ~62 word visits
// 64, not the 256 this shape started at, and the reason is `gridFor`'s 4096-block
// cap rather than the block size itself. The epilogue is a warp reduce plus
// three 64-bit atomics on ONE address, paid once per warp however many words
// the warp visited; capping the grid at 4096 blocks means a narrower block is
// FEWER warps each visiting more words. Measured on a whole 7680x4320 region,
// where the cap binds: 1.82x faster in 105 of 105 paired rounds, DRAM 41.3% ->
// 84.9% of peak on identical load traffic (518,400 sectors either way), while
// achieved occupancy FALLS 81.9% -> 61.9%. Below the cap -- 3840x2160 and the
// reference frame -- both block sizes give one word per thread and the two are
// a null, so this costs nothing there.
constexpr unsigned kRegionBlock = 64;

// ---------------------------------------------------------------------------
// THE BATCH: one BLOCK per window. reduce.cu's batchKernel shape, so the two
// batched reductions in this backend traverse alike.
// ---------------------------------------------------------------------------
template <size_t N, bool AlignedRun>
__global__ void covBatchKernel(DevicePlaneBlockConstView dx, DevicePlaneBlockConstView dy,
                               const Rect* windows, size_t count,
                               DeviceGradientCovariance* out) {
    const size_t idx = blockIdx.x;
    if (idx >= count) return;  // uniform over the block: grid.x IS count
    // THE HOST'S OWN CLIP, CALLED. Half-open, clamped, negative origins legal,
    // empty yields zeros -- decided in ops/reduce.hpp for both backends at once.
    const bincv::impl::RegionWords<uint32_t> r =
        bincv::impl::clipRegion<uint32_t>(dx.width, dx.height, windows[idx]);

    PairAcc<N> acc;
    clearAcc<N>(acc);
    if (!r.isEmpty) walkShare<N, AlignedRun>(dx, dy, r, acc);

    warpReduceAcc<N>(acc);
    __shared__ PairAcc<N> warpSums[kBatchBlock / 32];
    const unsigned warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31u) == 0u) warpSums[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {
        PairAcc<N> t = warpSums[0];
        for (unsigned k = 1; k < (blockDim.x + 31u) / 32u; ++k) {
#pragma unroll
            for (size_t i = 0; i < N; ++i) {
#pragma unroll
                for (size_t j = i; j < N; ++j) {
                    t.xx[i][j] += warpSums[k].xx[i][j];
                    t.yy[i][j] += warpSums[k].yy[i][j];
                }
#pragma unroll
                for (size_t j = 0; j < N; ++j) {
                    t.xyT[i][j] += warpSums[k].xyT[i][j];
                    t.xyS[i][j] += warpSums[k].xyS[i][j];
                }
            }
        }
        out[idx] = combine<N>(t);
    }
}

// ---------------------------------------------------------------------------
// ONE REGION, which may be a whole frame: grid-stride over (row, word), one
// warp-level reduction, then three atomics.
//
// EACH WARP WEIGHTS ITS OWN PARTIAL COUNTS. That is exact and not an
// approximation: the weighting is a LINEAR map over exact integer counts, so
// weighting per warp and adding is the same integer as adding and weighting
// once. It also means no per-pair count is ever stored or transferred.
// ---------------------------------------------------------------------------
template <size_t N>
__global__ void covRegionKernel(DevicePlaneBlockConstView dx, DevicePlaneBlockConstView dy,
                                bincv::impl::RegionWords<uint32_t> r,
                                DeviceGradientCovariance* out) {
    const size_t rowSpan = r.lastWord - r.firstWord + 1;
    const size_t total = (r.y1 - r.y0) * rowSpan;
    PairAcc<N> acc;
    clearAcc<N>(acc);
    uint32_t ax[N];
    uint32_t ay[N];
    uint32_t sel;
    for (size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += gridDim.x * blockDim.x) {
        const size_t ry = t / rowSpan;
        const size_t w = r.firstWord + (t - ry * rowSpan);
        loadWord<N>(dx, dy, r.y0 + ry, w, wordMask(r, w), ax, ay, sel);
        accumulateWord<N>(acc, ax, ay, sel);
    }
    warpReduceAcc<N>(acc);
    if ((threadIdx.x & 31u) != 0u) return;

    const DeviceGradientCovariance g = combine<N>(acc);
    // atomicAdd takes unsigned long long; two's-complement wraparound addition
    // is what makes that the right instruction for a signed accumulator.
    if (g.sumXX != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(&out->sumXX),
                  static_cast<unsigned long long>(g.sumXX));
    if (g.sumYY != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(&out->sumYY),
                  static_cast<unsigned long long>(g.sumYY));
    if (g.sumXY != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(&out->sumXY),
                  static_cast<unsigned long long>(g.sumXY));
}

unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

/// The two blocks must be N+1 planes of one extent, with strides that cover a
/// row. Read-only, so aliasing between them is unrestricted -- the host's
/// promise 4: `gradientCovariance(dx, dx, w)` is the SumIx^2 case with a cross
/// term equal to it, and that is well defined here too.
bool shapeIsValid(const DevicePlaneBlockConstView& dx, const DevicePlaneBlockConstView& dy) {
    return dx.planes >= 2 && dx.planes <= covarianceMaxPlanes() + 1 &&
           dy.planes == dx.planes && dx.width == dy.width && dx.height == dy.height &&
           dx.stride >= rowWords(dx.width) && dy.stride >= rowWords(dy.width);
}

bool preflight(const DevicePlaneBlockConstView& dx, const DevicePlaneBlockConstView& dy) {
    BINCV_ASSERT(shapeIsValid(dx, dy),
                 "cuda gradientCovariance: both derivatives must be N+1 planes of one "
                 "extent with N in [1, covarianceMaxPlanes()], and every stride must "
                 "cover a row");
    return shapeIsValid(dx, dy);
}

} // namespace

namespace impl {

/// DEFAULTS TO FALSE. The arm is a measured REGRESSION -- see the block
/// comment below -- so the shipped path is the per-word one. It stays
/// reachable because the numbers that rejected it were taken on a contended
/// host and are labelled indicative; a serial pass must be able to re-take
/// them, and a deleted arm cannot be re-measured.
bool& covarianceAlignedRunEnabled() {
    static bool enabled = false;
    return enabled;
}

} // namespace impl

// ---------------------------------------------------------------------------
// TWO OPTIMIZED ARMS WERE WRITTEN FOR THIS KERNEL. NEITHER EARNED ITS PLACE,
// AND BOTH FAILED FOR THE SAME UNDERLYING REASON.
//
// 1. ONE THREAD PER WINDOW -- removes the cross-lane reduction entirely, which
//    at 31x31 is a fold of N(N+1) + 2N^2 counters behind ~62 word visits of
//    real work. LOST 14 readings out of 14: 1.23x slower at N = 1, 1.76x at
//    N = 2. The reason is occupancy and it is arithmetic: 200 windows is 200
//    THREADS, two blocks on a 48-SM part, leaving the machine idle, where one
//    block per window is 12,800 threads and fills it even after paying the
//    reduction. DELETED.
//
// 2. THE FUNNEL-SHIFT RUN EXTRACTION -- halves the word VISITS for any run at
//    most 32 pixels wide. Priced at 200 windows of 31x31 it read 0.94x-0.97x,
//    which is neither a win nor a loss, because there BOTH arms fit one pass of
//    a 64-thread block (31 units and 62, both under 64) and the whole batch
//    sits on the launch floor anyway. Re-priced at 4000 windows of 31x240 --
//    ~30 MB, where the comparison is finally decidable and the formula's
//    ceiling is 2.00x (8 passes against 4) -- it read **1.09x-1.15x SLOWER**,
//    7 processes out of 7, ranges disjoint in 2 of them. KEPT BEHIND A SWITCH
//    THAT DEFAULTS OFF, so a serial pass can re-take an indicative number.
//
// WHY IT LOSES, AS INSTRUCTIONS RATHER THAN AS A SHRUG: the funnel saves
// POPCOUNTS, not LOADS. A run straddling two words must still read both, so the
// per-row cost stays 2N + 2 loads and gains 2N + 2 funnel shifts to halve a
// popcount count that was never the bottleneck -- at N = 1 the kernel is
// traffic-shaped, 4 popcounts on 4 loads. The visit count fell and the byte
// count did not.
//
// THE ARM IS NOT GATED ON N, and that is a reading rather than a choice:
// `nvcc -Xptxas -v` at sm_86 reports 40/56/91/96 registers at N = 1..4 with
// ZERO bytes spilled, so no cutoff is derivable from register pressure and none
// is invented.
// ---------------------------------------------------------------------------

cudaError_t gradientCovarianceBatchAsync(DevicePlaneBlockConstView dxSigned,
                                         DevicePlaneBlockConstView dySigned,
                                         const Rect* dWindows, size_t count,
                                         DeviceGradientCovariance* dResults,
                                         cudaStream_t stream) {
    if (!preflight(dxSigned, dySigned)) return cudaErrorInvalidValue;
    if (count == 0) return cudaSuccess;
    BINCV_ASSERT(dWindows != nullptr && dResults != nullptr,
                 "cuda gradientCovarianceBatch: null window or result array");
    if (dWindows == nullptr || dResults == nullptr) return cudaErrorInvalidValue;
    if (dxSigned.width == 0 || dxSigned.height == 0) {
        // Every window clips to nothing: {0, 0, 0} is a value, not an error.
        return cudaMemsetAsync(dResults, 0, count * sizeof(DeviceGradientCovariance),
                               stream);
    }
    if (dxSigned.ptr == nullptr || dySigned.ptr == nullptr) return cudaErrorInvalidValue;

    const size_t n = dxSigned.planes - 1;
    const bool aligned = impl::covarianceAlignedRunEnabled();
    const unsigned blocks = static_cast<unsigned>(count);

#define BINCV_CUDA_COV_BATCH(NN)                                                       \
    if (aligned)                                                                       \
        covBatchKernel<NN, true><<<blocks, kBatchBlock, 0, stream>>>(                   \
            dxSigned, dySigned, dWindows, count, dResults);                            \
    else                                                                               \
        covBatchKernel<NN, false><<<blocks, kBatchBlock, 0, stream>>>(                  \
            dxSigned, dySigned, dWindows, count, dResults);

    switch (n) {
        case 1: BINCV_CUDA_COV_BATCH(1) break;
        case 2: BINCV_CUDA_COV_BATCH(2) break;
        case 3: BINCV_CUDA_COV_BATCH(3) break;
        case 4: BINCV_CUDA_COV_BATCH(4) break;
        default: return cudaErrorInvalidValue;
    }
#undef BINCV_CUDA_COV_BATCH
    return cudaGetLastError();
}

cudaError_t gradientCovarianceAsync(DevicePlaneBlockConstView dxSigned,
                                    DevicePlaneBlockConstView dySigned, Rect window,
                                    DeviceGradientCovariance* dResult,
                                    cudaStream_t stream) {
    if (!preflight(dxSigned, dySigned)) return cudaErrorInvalidValue;
    BINCV_ASSERT(dResult != nullptr, "cuda gradientCovariance: null result pointer");
    if (dResult == nullptr) return cudaErrorInvalidValue;

    cudaError_t err = cudaMemsetAsync(dResult, 0, sizeof(DeviceGradientCovariance), stream);
    if (err != cudaSuccess) return err;

    const bincv::impl::RegionWords<uint32_t> r =
        bincv::impl::clipRegion<uint32_t>(dxSigned.width, dxSigned.height, window);
    if (r.isEmpty) return cudaSuccess;
    if (dxSigned.ptr == nullptr || dySigned.ptr == nullptr) return cudaErrorInvalidValue;

    const size_t total = (r.y1 - r.y0) * (r.lastWord - r.firstWord + 1);
    const unsigned grid = gridFor(total, kRegionBlock);
    switch (dxSigned.planes - 1) {
        case 1:
            covRegionKernel<1><<<grid, kRegionBlock, 0, stream>>>(dxSigned, dySigned, r,
                                                                  dResult);
            break;
        case 2:
            covRegionKernel<2><<<grid, kRegionBlock, 0, stream>>>(dxSigned, dySigned, r,
                                                                  dResult);
            break;
        case 3:
            covRegionKernel<3><<<grid, kRegionBlock, 0, stream>>>(dxSigned, dySigned, r,
                                                                  dResult);
            break;
        case 4:
            covRegionKernel<4><<<grid, kRegionBlock, 0, stream>>>(dxSigned, dySigned, r,
                                                                  dResult);
            break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

GradientCovariance gradientCovariance(DevicePlaneBlockConstView dxSigned,
                                      DevicePlaneBlockConstView dySigned, Rect window) {
    DeviceGradientCovariance* dOut = nullptr;
    BINCV_CUDA_CHECK(cudaMalloc(&dOut, sizeof(DeviceGradientCovariance)));
    cudaError_t err = gradientCovarianceAsync(dxSigned, dySigned, window, dOut, nullptr);
    DeviceGradientCovariance host{0, 0, 0};
    if (err == cudaSuccess)
        err = cudaMemcpy(&host, dOut, sizeof host, cudaMemcpyDeviceToHost);
    cudaFree(dOut);
    BINCV_CUDA_CHECK(err);
    return toHost(host);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
