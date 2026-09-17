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

constexpr int kDTile = 8;

// ---------------------------------------------------------------------------
// The SLIDING arm, and why it exists.
//
// Both arms above re-evaluate the WHOLE window for every output row: 24 census
// planes x 9 window rows x 64 disparities is ~13,800 popcounts per pixel, and
// the window's height multiplies the cost. It does not have to. The window sum
// for row y+1 is the sum for row y, minus the row that left, plus the row that
// entered -- so a row's cost is paid twice instead of winHeight times.
//
// This is not a new idea here: binCV's own HOST census path already does it,
// and its optimization curve records the sliding vertical accumulator at
// ~4.5x (docs/reports/stereo.md). The device kernel was ported without that
// refinement. The binary entry got away with it -- one plane, ~576 popcounts
// per pixel -- and the census entry, at 24x that, did not.
//
// Each thread owns one output column and a STRIP of output rows, sliding down
// it: winHeight row costs for the first row, two for each row after. The
// leaving row is recomputed rather than cached, which is the host's trade too
// -- a cached ring would need a dynamically indexed register array, and that
// spills to local memory, which costs more than the recomputation saves.
//
// The disparity tile is kept: one L extraction still serves kDTile candidates'
// XORs, so the two savings compose rather than replacing one another.
// ---------------------------------------------------------------------------

constexpr int kStrip = 16;

/// @brief Add (or subtract) one image row's windowed cost to each candidate's
/// running vertical sum. `winW <= 32`, which the launcher's gate enforces.
template <int DT>
__device__ __forceinline__ void accumulateRow(unsigned* sum, DeviceBinMatConstView left,
                                              DeviceBinMatConstView right, int planes,
                                              size_t imgHeight, size_t yy, size_t a,
                                              int d0, int dEnd, unsigned winW,
                                              bool add) {
    for (int k = 0; k < planes; ++k) {
        const size_t row = static_cast<size_t>(k) * imgHeight + yy;
        const uint32_t* rowL = left.row(row);
        const uint32_t* rowR = right.row(row);
        const uint32_t lw = extractBits(rowL, a, winW);
#pragma unroll
        for (int j = 0; j < DT; ++j) {
            const int d = d0 + j;
            // A candidate with no right support at this column is never read,
            // so it is never accumulated -- and the guard also keeps the
            // shifted read inside the row.
            if (d <= dEnd && static_cast<size_t>(d) <= a) {
                const uint32_t rw =
                    extractBits(rowR, a - static_cast<size_t>(d), winW);
                const unsigned c = static_cast<unsigned>(__popc(lw ^ rw));
                sum[j] = add ? sum[j] + c : sum[j] - c;
            }
        }
    }
}

__global__ void denseKernelSliding(DeviceBinMatConstView left,
                                   DeviceBinMatConstView right, int minD, int dEnd,
                                   int winW, int winH, int planes, size_t imgHeight,
                                   DeviceImageView<uint8_t> disparity, size_t outRows) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw = winW / 2;
    const int hh = winH / 2;
    // Border rows and columns are pre-filled with the invalid marker by the
    // launcher, so this kernel writes only pixels a candidate can serve.
    if (x < static_cast<size_t>(hw) || x + static_cast<size_t>(hw) >= disparity.width)
        return;
    const size_t a = x - static_cast<size_t>(hw);
    const size_t sFirst = static_cast<size_t>(blockIdx.y) * kStrip;
    if (sFirst >= outRows) return;
    const size_t yFirst = static_cast<size_t>(hh) + sFirst;
    const size_t rowsHere =
        (outRows - sFirst < static_cast<size_t>(kStrip)) ? (outRows - sFirst)
                                                         : static_cast<size_t>(kStrip);

    unsigned bestCost[kStrip];
    unsigned bestDisp[kStrip];
#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        bestCost[s] = 0xFFFFFFFFu;
        bestDisp[s] = 0xFFFFu;
    }

    const unsigned W = static_cast<unsigned>(winW);
    for (int d0 = minD; d0 <= dEnd; d0 += kDTile) {
        unsigned sum[kDTile];
#pragma unroll
        for (int j = 0; j < kDTile; ++j) sum[j] = 0;

        // The first output row's window, in full: rows [yFirst - hh, yFirst + hh].
        // Counted up from the top row so the index never goes negative --
        // yFirst is at least hh, so the subtraction is safe before the loop.
        const size_t yTop = yFirst - static_cast<size_t>(hh);
        for (int r = 0; r < winH; ++r) {
            accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                  yTop + static_cast<size_t>(r), a, d0, dEnd, W, true);
        }

#pragma unroll
        for (int s = 0; s < kStrip; ++s) {
            if (static_cast<size_t>(s) >= rowsHere) break;
#pragma unroll
            for (int j = 0; j < kDTile; ++j) {
                const int d = d0 + j;
                // Strictly less, disparities ascending: ties keep the
                // smallest, the host's rule.
                if (d <= dEnd && static_cast<size_t>(d) <= a && sum[j] < bestCost[s]) {
                    bestCost[s] = sum[j];
                    bestDisp[s] = static_cast<unsigned>(d);
                }
            }
            if (static_cast<size_t>(s) + 1 < rowsHere) {
                const size_t y = yFirst + static_cast<size_t>(s);
                accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                      y - static_cast<size_t>(hh), a, d0, dEnd, W,
                                      false);
                accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                      y + static_cast<size_t>(hh) + 1, a, d0, dEnd, W,
                                      true);
            }
        }
    }

#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        if (static_cast<size_t>(s) >= rowsHere) break;
        if (bestDisp[s] <= 254u)
            disparity.row(yFirst + static_cast<size_t>(s))[x] =
                static_cast<uint8_t>(bestDisp[s]);
    }
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

    // The sliding arm carries one static bound: a window no wider than a single
    // extraction (winW <= 32). A wider window -- and the reference arm the
    // switch forces -- goes through the straightforward kernel.
    if (impl::denseFastArmEnabled() && params.winWidth <= 32) {
        // Borders first: the sliding kernel writes only pixels a candidate can
        // serve, so the rim's invalid marker comes from one cheap fill rather
        // than from per-thread branches in the hot kernel.
        cudaError_t err =
            cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                              disparity.width, disparity.height, stream);
        if (err != cudaSuccess) return err;
        const size_t outRows = imgHeight - 2 * static_cast<size_t>(params.winHeight / 2);
        constexpr unsigned kColBlock = 128;
        const dim3 grid(static_cast<unsigned>((width + kColBlock - 1) / kColBlock),
                        static_cast<unsigned>((outRows + kStrip - 1) / kStrip));
        denseKernelSliding<<<grid, kColBlock, 0, stream>>>(
            left, right, params.minDisparity, dEnd, params.winWidth, params.winHeight,
            static_cast<int>(planes), imgHeight, disparity, outRows);
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
bool& denseFastArmEnabled() {
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
