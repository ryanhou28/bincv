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

// ---------------------------------------------------------------------------
// The TILED arm. The reference kernel reads the wide image once PER PLANE; a
// 5x5 pattern therefore moves every byte 24 times. Here a block stages a
// 32x8 pixel tile plus the pattern's apron into shared memory once and emits
// all K ballots from it, so each byte crosses the bus once. Patterns whose
// largest offset exceeds the apron cap fall back to the reference arm, which
// stays reachable through censusTiledEnabled() for the benchmark and tests.
// ---------------------------------------------------------------------------

constexpr int kCensusApronCap = 8;

template <typename SrcT>
__global__ void censusKernelTiled(DeviceImageConstView<SrcT> img,
                                  CensusOffsetsPod pattern,
                                  DeviceBinMatView planeBlock, int apron) {
    extern __shared__ unsigned char smemRaw[];
    SrcT* tile = reinterpret_cast<SrcT*>(smemRaw);
    const int tw = 32 + 2 * apron;
    const int th = 8 + 2 * apron;
    const size_t X0 = static_cast<size_t>(blockIdx.x) * 32;
    const size_t Y0 = static_cast<size_t>(blockIdx.y) * 8;

    const unsigned tid = threadIdx.y * 32 + threadIdx.x;
    for (unsigned idx = tid; idx < static_cast<unsigned>(tw * th); idx += 256) {
        const int lx = static_cast<int>(idx) % tw;
        const int ly = static_cast<int>(idx) / tw;
        const long long gx = static_cast<long long>(X0) - apron + lx;
        const long long gy = static_cast<long long>(Y0) - apron + ly;
        tile[idx] = (gx >= 0 && gx < static_cast<long long>(img.width) && gy >= 0 &&
                     gy < static_cast<long long>(img.height))
                        ? img.row(static_cast<size_t>(gy))[gx]
                        : SrcT{0};
    }
    __syncthreads();

    const size_t x = X0 + threadIdx.x;
    const size_t y = Y0 + threadIdx.y;
    if (y >= img.height) return;  // whole warp shares y: uniform exit
    const bool inImg = x < img.width;
    const SrcT c =
        tile[(threadIdx.y + static_cast<unsigned>(apron)) * static_cast<unsigned>(tw) +
             threadIdx.x + static_cast<unsigned>(apron)];
    for (int k = 0; k < pattern.planes; ++k) {
        const int dx = pattern.dx[k];
        const int dy = pattern.dy[k];
        bool pred = false;
        if (inImg) {
            const long long gx = static_cast<long long>(x) + dx;
            const long long gy = static_cast<long long>(y) + dy;
            if (gx >= 0 && gx < static_cast<long long>(img.width) && gy >= 0 &&
                gy < static_cast<long long>(img.height)) {
                const SrcT nb =
                    tile[(threadIdx.y + static_cast<unsigned>(apron + dy)) *
                             static_cast<unsigned>(tw) +
                         threadIdx.x + static_cast<unsigned>(apron + dx)];
                pred = nb > c;
            }
        }
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (threadIdx.x == 0)
            planeBlock.row(static_cast<size_t>(k) * img.height + y)[blockIdx.x] = word;
    }
}

template <typename SrcT>
cudaError_t launchCensus(DeviceImageConstView<SrcT> img, const CensusOffsetsPod& pattern,
                         DeviceBinMatView planeBlock, cudaStream_t stream) {
    if (img.width == 0 || img.height == 0) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr && planeBlock.ptr != nullptr,
                 "cuda censusTransform: a non-empty image needs non-null pointers");
    const size_t words = rowWords(img.width);
    int apron = 0;
    for (int k = 0; k < pattern.planes; ++k) {
        const int ax = pattern.dx[k] < 0 ? -pattern.dx[k] : pattern.dx[k];
        const int ay = pattern.dy[k] < 0 ? -pattern.dy[k] : pattern.dy[k];
        if (ax > apron) apron = ax;
        if (ay > apron) apron = ay;
    }
    if (impl::censusTiledEnabled() && apron <= kCensusApronCap) {
        const dim3 block(32, 8);
        const dim3 grid(static_cast<unsigned>(words),
                        static_cast<unsigned>((img.height + 7) / 8));
        const size_t sharedBytes = static_cast<size_t>(32 + 2 * apron) *
                                   static_cast<size_t>(8 + 2 * apron) * sizeof(SrcT);
        censusKernelTiled<SrcT>
            <<<grid, block, sharedBytes, stream>>>(img, pattern, planeBlock, apron);
        return cudaGetLastError();
    }
    const dim3 block(32, 8);
    const size_t warps = (words * img.height + block.y - 1) / block.y;
    const dim3 grid(static_cast<unsigned>(warps < 2048 ? (warps ? warps : 1) : 2048),
                    1, static_cast<unsigned>(pattern.planes));
    censusKernel<SrcT><<<grid, block, 0, stream>>>(img, pattern, planeBlock, words);
    return cudaGetLastError();
}

/// @brief The packed layout's kernel: one thread, one pixel, one word out.
/// The same tile-and-apron staging as the plane arm, so each source byte is
/// still read once no matter how many comparisons the pattern has.
template <typename SrcT>
__global__ void censusPackedKernel(DeviceImageConstView<SrcT> img,
                                   CensusOffsetsPod pattern,
                                   DeviceImageView<uint32_t> dst, int apron) {
    extern __shared__ unsigned char smemRaw[];
    SrcT* tile = reinterpret_cast<SrcT*>(smemRaw);
    const int tw = 32 + 2 * apron;
    const int th = 8 + 2 * apron;
    const size_t X0 = static_cast<size_t>(blockIdx.x) * 32;
    const size_t Y0 = static_cast<size_t>(blockIdx.y) * 8;

    const unsigned tid = threadIdx.y * 32 + threadIdx.x;
    for (unsigned idx = tid; idx < static_cast<unsigned>(tw * th); idx += 256) {
        const int lx = static_cast<int>(idx) % tw;
        const int ly = static_cast<int>(idx) / tw;
        const long long gx = static_cast<long long>(X0) - apron + lx;
        const long long gy = static_cast<long long>(Y0) - apron + ly;
        tile[idx] = (gx >= 0 && gx < static_cast<long long>(img.width) && gy >= 0 &&
                     gy < static_cast<long long>(img.height))
                        ? img.row(static_cast<size_t>(gy))[gx]
                        : SrcT{0};
    }
    __syncthreads();

    const size_t x = X0 + threadIdx.x;
    const size_t y = Y0 + threadIdx.y;
    if (x >= img.width || y >= img.height) return;
    const SrcT c =
        tile[(threadIdx.y + static_cast<unsigned>(apron)) * static_cast<unsigned>(tw) +
             threadIdx.x + static_cast<unsigned>(apron)];
    uint32_t desc = 0;
    for (int k = 0; k < pattern.planes; ++k) {
        const int dx = pattern.dx[k];
        const int dy = pattern.dy[k];
        const long long gx = static_cast<long long>(x) + dx;
        const long long gy = static_cast<long long>(y) + dy;
        // A neighbour outside the frame contributes 0, exactly as it does in
        // the plane layout.
        if (gx >= 0 && gx < static_cast<long long>(img.width) && gy >= 0 &&
            gy < static_cast<long long>(img.height)) {
            const SrcT nb = tile[(threadIdx.y + static_cast<unsigned>(apron + dy)) *
                                     static_cast<unsigned>(tw) +
                                 threadIdx.x + static_cast<unsigned>(apron + dx)];
            if (nb > c) desc |= (1u << k);
        }
    }
    dst.row(y)[x] = desc;
}

template <typename SrcT>
cudaError_t launchCensusPacked(DeviceImageConstView<SrcT> img,
                               const CensusOffsetsPod& pattern,
                               DeviceImageView<uint32_t> dst, cudaStream_t stream) {
    if (img.width == 0 || img.height == 0) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr && dst.ptr != nullptr,
                 "cuda censusTransformPacked: a non-empty image needs non-null pointers");
    int apron = 0;
    for (int k = 0; k < pattern.planes; ++k) {
        const int ax = pattern.dx[k] < 0 ? -pattern.dx[k] : pattern.dx[k];
        const int ay = pattern.dy[k] < 0 ? -pattern.dy[k] : pattern.dy[k];
        if (ax > apron) apron = ax;
        if (ay > apron) apron = ay;
    }
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((img.width + 31) / 32),
                    static_cast<unsigned>((img.height + 7) / 8));
    const size_t sharedBytes = static_cast<size_t>(32 + 2 * apron) *
                               static_cast<size_t>(8 + 2 * apron) * sizeof(SrcT);
    censusPackedKernel<SrcT><<<grid, block, sharedBytes, stream>>>(img, pattern, dst,
                                                                   apron);
    return cudaGetLastError();
}

} // namespace

cudaError_t censusTransformPackedImpl(DeviceImageConstView<uint8_t> img,
                                      const CensusOffsetsPod& pattern,
                                      DeviceImageView<uint32_t> dst,
                                      cudaStream_t stream) {
    return launchCensusPacked<uint8_t>(img, pattern, dst, stream);
}

cudaError_t censusTransformPackedImpl(DeviceImageConstView<uint16_t> img,
                                      const CensusOffsetsPod& pattern,
                                      DeviceImageView<uint32_t> dst,
                                      cudaStream_t stream) {
    return launchCensusPacked<uint16_t>(img, pattern, dst, stream);
}

bool& censusTiledEnabled() {
    static bool on = true;
    return on;
}

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
