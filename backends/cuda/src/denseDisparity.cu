// The device dense-disparity kernels.
//
// SHAPE, and why it is not the host's: the host must never hold a cost volume,
// so it streams rows and keeps per-disparity bit-sliced accumulators. A GPU
// thread can hold one pixel's running best in registers for the whole
// disparity sweep, which satisfies the same rule with no scratch at all. The
// cost of one candidate is the windowed Hamming distance read directly from
// the packed rows: per window row, extract winWidth bits of each image
// (right image shifted by d) and popcount the XOR. All reads are in-row for
// every anchor a candidate may claim (a >= d and a + winWidth <= width), which
// the host establishes the same way.
//
// Binary and census are ONE kernel: the binary path is the census path at
// planes = 1. Plane k of a block sits at rows [k * H, (k + 1) * H).

#include "bincv/cuda/denseDisparity.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief Bits [bitPos, bitPos + nbits) of a packed row, nbits in 1..32.
/// Reads word i+1 only when the run crosses into it, which the caller's
/// support bounds keep inside the row.
__device__ inline uint32_t extractBits(const uint32_t* row, size_t bitPos,
                                       unsigned nbits) {
    const size_t i = bitPos >> 5;
    const unsigned r = static_cast<unsigned>(bitPos & 31u);
    uint32_t lo = __ldg(row + i) >> r;
    if (r != 0 && r + nbits > 32u) lo |= __ldg(row + i + 1) << (32u - r);
    return nbits == 32u ? lo : lo & ((1u << nbits) - 1u);
}

/// @brief Hamming distance between winW-bit runs at (rowL, aL) and (rowR, aR).
__device__ inline unsigned windowRowHamming(const uint32_t* rowL, size_t aL,
                                            const uint32_t* rowR, size_t aR,
                                            unsigned winW) {
    unsigned cost = 0;
    unsigned off = 0;
    while (off < winW) {
        const unsigned chunk = (winW - off < 32u) ? (winW - off) : 32u;
        const uint32_t l = extractBits(rowL, aL + off, chunk);
        const uint32_t r = extractBits(rowR, aR + off, chunk);
        cost += static_cast<unsigned>(__popc(l ^ r));
        off += chunk;
    }
    return cost;
}

__global__ void denseKernel(DeviceBinMatConstView left, DeviceBinMatConstView right,
                            int minD, int dEnd, int winW, int winH, int planes,
                            size_t imgHeight, DeviceImageView<uint8_t> disparity) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= disparity.width || y >= imgHeight) return;

    const int hw = winW / 2;
    const int hh = winH / 2;
    uint8_t out = kDenseDisparityInvalid;
    if (y >= static_cast<size_t>(hh) && y + static_cast<size_t>(hh) < imgHeight &&
        x >= static_cast<size_t>(hw) && x + static_cast<size_t>(hw) < disparity.width) {
        const size_t a = x - static_cast<size_t>(hw);  // window anchor
        unsigned bestC = 0xFFFFFFFFu;
        unsigned bestD = 0xFFFFu;
        for (int d = minD; d <= dEnd; ++d) {
            // Anchors this disparity may claim: a >= d, full right support.
            if (static_cast<size_t>(d) > a) continue;
            const size_t aR = a - static_cast<size_t>(d);
            unsigned cost = 0;
            for (int k = 0; k < planes; ++k) {
                const size_t planeRow0 = static_cast<size_t>(k) * imgHeight;
                for (int r = -hh; r <= hh; ++r) {
                    const size_t row = planeRow0 + y + static_cast<size_t>(r + hh) -
                                       static_cast<size_t>(hh);
                    cost += windowRowHamming(left.row(row), a, right.row(row), aR,
                                             static_cast<unsigned>(winW));
                }
            }
            // Strictly less, disparities ascending: ties keep the smallest,
            // the host's rule.
            if (cost < bestC) {
                bestC = cost;
                bestD = static_cast<unsigned>(d);
            }
        }
        if (bestD <= 254u) out = static_cast<uint8_t>(bestD);
    }
    disparity.row(y)[x] = out;
}

// ---------------------------------------------------------------------------
// The TILED arm. One output row per block, kTileX pixels per block; the L and
// R word bands every candidate in the block reads -- all planes, all window
// rows, the disparity range's extra R words -- are staged into shared memory
// once, cooperatively, and the sweep runs out of shared. The disparity loop is
// tiled by eight so one L-window extraction serves eight candidates' XORs, and
// the eight running costs live in registers (compile-time indexed, so they
// stay there). Same arithmetic as the reference kernel word for word -- the
// tests hold both arms to identical maps, the reference arm stays reachable
// through denseTiledEnabled(), and the benchmark times both.
// ---------------------------------------------------------------------------

constexpr unsigned kTileX = 128;
constexpr int kDTile = 8;

__device__ inline uint32_t extractShared(const uint32_t* row, size_t relBit,
                                         unsigned nbits) {
    const size_t i = relBit >> 5;
    const unsigned r = static_cast<unsigned>(relBit & 31u);
    uint32_t lo = row[i] >> r;
    if (r != 0 && r + nbits > 32u) lo |= row[i + 1] << (32u - r);
    return nbits == 32u ? lo : lo & ((1u << nbits) - 1u);
}

__global__ void denseKernelTiled(DeviceBinMatConstView left,
                                 DeviceBinMatConstView right, int minD, int dEnd,
                                 int winW, int winH, int planes, size_t imgHeight,
                                 DeviceImageView<uint8_t> disparity, size_t words,
                                 unsigned nwL, unsigned nwR) {
    extern __shared__ uint32_t smem[];
    const size_t y = blockIdx.y;
    const int hw = winW / 2;
    const int hh = winH / 2;
    const size_t X0 = static_cast<size_t>(blockIdx.x) * kTileX;
    const size_t x = X0 + threadIdx.x;

    if (y < static_cast<size_t>(hh) || y + static_cast<size_t>(hh) >= imgHeight) {
        if (x < disparity.width) disparity.row(y)[x] = kDenseDisparityInvalid;
        return;
    }

    // Word bases of the staged bands. Negative bit origins clamp to word 0;
    // every valid candidate's bits still sit at or above the base.
    const long long bL0 = static_cast<long long>(X0) - hw;
    const long long bR0 = bL0 - dEnd;
    const size_t wL0 = bL0 > 0 ? static_cast<size_t>(bL0) >> 5 : 0;
    const size_t wR0 = bR0 > 0 ? static_cast<size_t>(bR0) >> 5 : 0;

    uint32_t* sL = smem;
    uint32_t* sR = smem + static_cast<size_t>(planes) * winH * nwL;
    const unsigned bandRows = static_cast<unsigned>(planes) * winH;
    for (unsigned idx = threadIdx.x; idx < bandRows * nwL; idx += blockDim.x) {
        const unsigned band = idx / nwL;
        const unsigned w = idx - band * nwL;
        const size_t k = band / winH;
        const size_t r = band - k * winH;
        const size_t gw = wL0 + w;
        const size_t row = k * imgHeight + (y - hh) + r;
        sL[idx] = gw < words ? left.row(row)[gw] : 0u;
    }
    for (unsigned idx = threadIdx.x; idx < bandRows * nwR; idx += blockDim.x) {
        const unsigned band = idx / nwR;
        const unsigned w = idx - band * nwR;
        const size_t k = band / winH;
        const size_t r = band - k * winH;
        const size_t gw = wR0 + w;
        const size_t row = k * imgHeight + (y - hh) + r;
        sR[idx] = gw < words ? right.row(row)[gw] : 0u;
    }
    __syncthreads();

    if (x >= disparity.width) return;
    uint8_t out = kDenseDisparityInvalid;
    if (x >= static_cast<size_t>(hw) && x + static_cast<size_t>(hw) < disparity.width) {
        const size_t a = x - static_cast<size_t>(hw);
        const size_t relA = a - (wL0 << 5);
        unsigned bestC = 0xFFFFFFFFu;
        unsigned bestD = 0xFFFFu;
        for (int d0 = minD; d0 <= dEnd; d0 += kDTile) {
            unsigned cost[kDTile];
#pragma unroll
            for (int j = 0; j < kDTile; ++j) cost[j] = 0;
            for (int k = 0; k < planes; ++k) {
                for (int r = 0; r < winH; ++r) {
                    const uint32_t* rowL = sL + (static_cast<size_t>(k) * winH + r) * nwL;
                    const uint32_t* rowR = sR + (static_cast<size_t>(k) * winH + r) * nwR;
                    const uint32_t lw =
                        extractShared(rowL, relA, static_cast<unsigned>(winW));
#pragma unroll
                    for (int j = 0; j < kDTile; ++j) {
                        const int d = d0 + j;
                        if (d <= dEnd && static_cast<size_t>(d) <= a) {
                            const size_t relR = (a - static_cast<size_t>(d)) - (wR0 << 5);
                            cost[j] += static_cast<unsigned>(
                                __popc(lw ^ extractShared(rowR, relR,
                                                          static_cast<unsigned>(winW))));
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < kDTile; ++j) {
                const int d = d0 + j;
                if (d <= dEnd && static_cast<size_t>(d) <= a && cost[j] < bestC) {
                    bestC = cost[j];
                    bestD = static_cast<unsigned>(d);
                }
            }
        }
        if (bestD <= 254u) out = static_cast<uint8_t>(bestD);
    }
    disparity.row(y)[x] = out;
}

cudaError_t launchDense(DeviceBinMatConstView left, DeviceBinMatConstView right,
                        size_t planes, size_t imgHeight,
                        const DenseDisparityParams& params,
                        DeviceImageView<uint8_t> disparity, cudaStream_t stream) {
    const size_t width = left.width;
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "cuda denseDisparity: the pair must share its extent");
    BINCV_ASSERT(left.height == planes * imgHeight,
                 "cuda denseDisparity: plane block height must be planes * imageHeight");
    BINCV_ASSERT(disparity.width == width && disparity.height == imgHeight,
                 "cuda denseDisparity: disparity extent must match the image");
    BINCV_ASSERT(params.winWidth >= 3 && params.winHeight >= 3 &&
                     (params.winWidth & 1) == 1 && (params.winHeight & 1) == 1,
                 "cuda denseDisparity: the window must be odd and at least 3 on a side");
    BINCV_ASSERT(params.minDisparity >= 0 &&
                     params.maxDisparity >= params.minDisparity &&
                     params.maxDisparity <= 254,
                 "cuda denseDisparity: need 0 <= min <= max <= 254 (255 marks invalid)");
    if (width == 0 || imgHeight == 0) return cudaSuccess;
    BINCV_ASSERT(left.ptr != nullptr && right.ptr != nullptr && disparity.ptr != nullptr,
                 "cuda denseDisparity: non-empty views need non-null pointers");

    const long long dMaxSupported = static_cast<long long>(width) -
                                    static_cast<long long>(params.winWidth);
    const int dEnd = params.maxDisparity <= dMaxSupported
                         ? params.maxDisparity
                         : static_cast<int>(dMaxSupported);
    if (imgHeight < static_cast<size_t>(params.winHeight) ||
        static_cast<size_t>(params.winWidth) > width || dEnd < params.minDisparity) {
        // No candidate anywhere: the whole map is the invalid marker, the same
        // early-out the host takes.
        return cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                                 disparity.width, disparity.height, stream);
    }

    // The tiled arm carries two static bounds: a window no wider than one
    // extraction (winW <= 32) and a band that fits default shared memory.
    // Everything else -- and the reference arm the switch forces -- goes
    // through the straightforward kernel.
    const size_t words = rowWords(width);
    const unsigned nwL =
        static_cast<unsigned>((kTileX + static_cast<unsigned>(params.winWidth) + 61) /
                              32) + 1;
    const unsigned nwR = nwL + static_cast<unsigned>((dEnd + 31) / 32) + 1;
    const size_t sharedBytes =
        planes * static_cast<size_t>(params.winHeight) * (nwL + nwR) * sizeof(uint32_t);
    if (impl::denseTiledEnabled() && params.winWidth <= 32 && sharedBytes <= 48 * 1024) {
        const dim3 grid(static_cast<unsigned>((width + kTileX - 1) / kTileX),
                        static_cast<unsigned>(imgHeight));
        denseKernelTiled<<<grid, kTileX, sharedBytes, stream>>>(
            left, right, params.minDisparity, dEnd, params.winWidth, params.winHeight,
            static_cast<int>(planes), imgHeight, disparity, words, nwL, nwR);
        return cudaGetLastError();
    }
    const dim3 block(128, 1);
    const dim3 grid(static_cast<unsigned>((width + block.x - 1) / block.x),
                    static_cast<unsigned>(imgHeight));
    denseKernel<<<grid, block, 0, stream>>>(left, right, params.minDisparity, dEnd,
                                            params.winWidth, params.winHeight,
                                            static_cast<int>(planes), imgHeight,
                                            disparity);
    return cudaGetLastError();
}

} // namespace

namespace impl {
bool& denseTiledEnabled() {
    static bool on = true;
    return on;
}
} // namespace impl

cudaError_t denseDisparityBinary(DeviceBinMatConstView left,
                                 DeviceBinMatConstView right,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream) {
    // The host binary path keeps its extraction in bytes and asserts this; the
    // cost fits wider registers here, but the accepted domain stays the
    // host's so "equal by test" is a statement about the same inputs.
    BINCV_ASSERT(params.winWidth * params.winHeight <= 255,
                 "cuda denseDisparityBinary: winWidth * winHeight must fit a byte");
    return launchDense(left, right, 1, left.height, params, disparity, stream);
}

cudaError_t denseDisparityCensus(DeviceBinMatConstView leftPlanes,
                                 DeviceBinMatConstView rightPlanes, size_t planes,
                                 size_t imageHeight,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream) {
    BINCV_ASSERT(planes >= 1 && planes <= 32,
                 "cuda denseDisparityCensus: 1 to 32 planes");
    return launchDense(leftPlanes, rightPlanes, planes, imageHeight, params, disparity,
                       stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
