// The device sensor stage. A warp produces packed words: lane x of the warp
// reads pixel 32*i + x, evaluates the rule, and __ballot_sync IS the packed
// word -- the format's 32-bit granule and the warp width coinciding. The N-bit
// packer is the same shape, N ballots deep.
//
// ---------------------------------------------------------------------------
// THE ARMS, THEIR SWITCHES, AND WHY THE FIRST ONE ALONE WAS NOT ENOUGH
//
// The grid-stride arm below is the shape this file shipped with, and it is
// still the oracle every other arm is held to. It is not the shape that gets
// launched, because it lost its role comparison: `cuda::threshold` -- which is
// this kernel with a cutoff in front of it -- measured 2.22x SLOWER than
// cv::cuda::threshold at 3840x2160 while moving 1.78x LESS traffic, running at
// 6.8x its bandwidth floor where OpenCV ran at 1.9x. The kernel was not
// memory-bound; it was bound on its own index arithmetic.
//
// cuobjdump located that without a profiler. `packKernel<uint8_t,
// GreaterEqual>` assembles to 184 SASS instructions around ONE LDG and ONE
// STG: 62 IMAD, 26 IADD3, 18 ISETP, and TWO SOFTWARE DIVIDES -- the
// I2F.U32.RP -> MUFU.RCP -> F2I sequence, once at 32 bits and once at 64 --
// from the `wordIdx / words` and `wordIdx - y * words` a flat grid-stride loop
// needs to recover (row, word) from one index. A 32-bit machine has no integer
// divide instruction, so a variable divisor is a function call's worth of
// arithmetic per pixel.
//
//   ROW-GRID ARM (`packRowGridEnabled`, default on). The launch shape carries
//     the row and the word, so both divides are deleted outright rather than
//     strength-reduced. One warp produces EIGHT consecutive words: it issues
//     its eight loads before its first ballot, and lanes 0..7 store the eight
//     words together, one whole 32-byte sector. One word per warp, stored by
//     lane 0, wrote 4 bytes per sector -- 8.0x the output in store sectors --
//     and kept one load per lane in flight; at 7680x4320 the profiler reads
//     DRAM 22% -> 66% for a uint8 source and 41% -> 94% for uint16, store
//     sectors 8.0x -> 1.0x. Costs a gate -- a grid dimension is capped at
//     65535 -- and above that the grid-stride arm runs.
//
// SHARED OWNERSHIP IS THE HAZARD HERE. This kernel is `packBits`',
// `packRows`' and `cuda::threshold`'s, and `packQuantKernel` next to it is the
// N-deep twin with the same two divides. Both got the row grid, and packQuant
// has a wide lane of its own (`packQuantWideLaneEnabled`). The one-word-per-
// warp quantized ballot arm spent ~90 warp instructions per word at N = 2,
// most of them per plane -- a ballot, a branch and a lane-0 store with its own
// 64-bit address -- and read 15% of peak DRAM with SM at up to 66%. The wide
// lane evaluates the scale for four bytes at once in 16-bit fields, extracts
// each plane with one `__dp4a` per four pixels and stores two planes per store
// instruction: ~10x fewer instructions, 92% of peak DRAM at 7680x4320.

#include "bincv/cuda/pack.hpp"

#include <cstdint>

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief Words one warp of a ballot row-grid arm produces: eight, so the eight
/// lanes that store them write one whole 32-byte sector.
constexpr unsigned kBallotWords = 8;

/// @brief Pixels one lane of the wide-lane arm covers: one 16-byte load.
constexpr unsigned kWideLanePixels = 16;

/// @brief Pixels one warp of the wide-lane arm covers: 32 lanes x 16 = 16 words.
constexpr unsigned kWideWarpPixels = 512;

/// @brief The largest `gridDim.y` a launch may ask for. The row-grid family puts
/// a band of rows in that dimension; see `packRowGridApplies` for the gate.
constexpr size_t kMaxGridY = 65535;

// ---------------------------------------------------------------------------
// The grid-stride arm -- the oracle
// ---------------------------------------------------------------------------

template <typename SrcT, PackRule R>
__global__ void packKernel(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                           size_t dstRow, SrcT t, size_t words) {
    const unsigned lane = threadIdx.x;  // 0..31
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        bool pred = false;
        if (x < src.width) {
            const SrcT v = src.row(y)[x];
            if (R == PackRule::NonZero) pred = v != SrcT{0};
            if (R == PackRule::GreaterThan) pred = v > t;
            if (R == PackRule::GreaterEqual) pred = v >= t;
        }
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == 0) dst.row(dstRow + y)[i] = word;
    }
}

// ---------------------------------------------------------------------------
// The row-grid arm -- no divides, eight words per warp
// ---------------------------------------------------------------------------

/// @note The grid-stride arm's rule and ballot, eight times over. What changed
/// is the traversal: a block is eight rows of one eight-word column, so the
/// row and the first word come from the launch shape and neither has to be
/// recovered from a flat index.
/// @note ALL EIGHT LOADS ARE ISSUED BEFORE THE FIRST BALLOT, and lanes 0..7
/// store the eight words in one instruction. One word per warp, stored by
/// lane 0, wrote 4 bytes into each 32-byte sector and kept one load per lane
/// in flight; this writes whole sectors wherever a row starts on one.
/// @note The early return is taken by a WHOLE WARP or none of it -- `threadIdx.y`
/// is uniform across a warp at `blockDim.x == 32` -- so the `__ballot_sync`
/// below still sees all 32 lanes, which is what makes its mask legal. The
/// words past the row's last are ballots of nothing and are not stored.
template <typename SrcT, PackRule R>
__global__ void packKernelRowGrid(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                                  size_t dstRow, SrcT t, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (y >= src.height) return;
    const size_t i0 = static_cast<size_t>(blockIdx.x) * kBallotWords;
    const SrcT* row = src.row(y);
    SrcT v[kBallotWords];
#pragma unroll
    for (unsigned k = 0; k < kBallotWords; ++k) {
        const size_t x = (i0 + k) * 32 + lane;
        v[k] = (x < src.width) ? row[x] : SrcT{0};
    }
    uint32_t mine = 0;
#pragma unroll
    for (unsigned k = 0; k < kBallotWords; ++k) {
        const size_t x = (i0 + k) * 32 + lane;
        bool pred = false;
        if (x < src.width) {
            if (R == PackRule::NonZero) pred = v[k] != SrcT{0};
            if (R == PackRule::GreaterThan) pred = v[k] > t;
            if (R == PackRule::GreaterEqual) pred = v[k] >= t;
        }
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == k) mine = word;
    }
    const size_t i = i0 + lane;
    if (lane < kBallotWords && i < words) dst.row(dstRow + y)[i] = mine;
}

// ---------------------------------------------------------------------------
// The N-bit packer
// ---------------------------------------------------------------------------

template <typename SrcT>
__global__ void packQuantKernel(DeviceImageConstView<SrcT> src,
                                DeviceBinMatView planeBlock, unsigned n,
                                unsigned maxValue, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = words * src.height;
    for (size_t wordIdx = blockIdx.x * warpsPerBlock + threadIdx.y; wordIdx < total;
         wordIdx += gridDim.x * warpsPerBlock) {
        const size_t y = wordIdx / words;
        const size_t i = wordIdx - y * words;
        const size_t x = i * 32 + lane;
        // A lane past the row contributes 0 to every plane, which is the
        // padding invariant holding by construction rather than by masking.
        //
        // THE HOST'S OWN SCALE, CALLED. This was a `quantScaleDevice` that
        // restated the same integer expression here; impl::quantScale is
        // BINCV_HOST_DEVICE, so there is one definition of what level a pixel
        // maps to. It matters more than the usual amount: the `+ srcMax/2`
        // rounding diverges from OpenCV at bytes 1..127 ON PURPOSE, so a device
        // copy that drifted toward OpenCV would read as the divergence finally
        // being fixed rather than as a backend disagreeing with its host.
        const unsigned value = (x < src.width)
                                   ? bincv::impl::quantScale<SrcT>(src.row(y)[x], maxValue)
                                   : 0u;
        for (unsigned p = 0; p < n; ++p) {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((value >> p) & 1u) != 0u);
            if (lane == 0)
                planeBlock.row(static_cast<size_t>(p) * src.height + y)[i] = word;
        }
    }
}

/// @note The N-deep twin of `packKernelRowGrid`, with the same traversal and the
/// grid-stride arm's scale and ballots. Eight words of N planes are 8N
/// ballots; lane `8 * (p % 4) + k` keeps plane p's word k, so planes 0..3
/// leave in one store instruction and planes 4..7 in a second, each plane's
/// eight words one 32-byte sector.
/// @note The plane loop is unrolled to QuantMat's cap of 8 and stops at `n`,
/// which is uniform across the warp, so every ballot is reached by all 32
/// lanes and no register array is indexed at run time.
template <typename SrcT>
__global__ void packQuantKernelRowGrid(DeviceImageConstView<SrcT> src,
                                       DeviceBinMatView planeBlock, unsigned n,
                                       unsigned maxValue, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (y >= src.height) return;
    const size_t i0 = static_cast<size_t>(blockIdx.x) * kBallotWords;
    const SrcT* row = src.row(y);
    unsigned value[kBallotWords];
#pragma unroll
    for (unsigned k = 0; k < kBallotWords; ++k) {
        const size_t x = (i0 + k) * 32 + lane;
        value[k] = (x < src.width) ? bincv::impl::quantScale<SrcT>(row[x], maxValue) : 0u;
    }
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (unsigned p = 0; p < 8; ++p) {
        if (p >= n) break;
#pragma unroll
        for (unsigned k = 0; k < kBallotWords; ++k) {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((value[k] >> p) & 1u) != 0u);
            if (lane == 8u * (p & 3u) + k) {
                if (p < 4) lo = word;
                else hi = word;
            }
        }
    }
    const unsigned plane = lane >> 3;
    const size_t i = i0 + (lane & 7u);
    if (i < words) {
        if (plane < n) planeBlock.row(static_cast<size_t>(plane) * src.height + y)[i] = lo;
        if (plane + 4 < n)
            planeBlock.row(static_cast<size_t>(plane + 4) * src.height + y)[i] = hi;
    }
}

// ---------------------------------------------------------------------------
// The N-bit wide-lane arm -- sixteen pixels per lane, every plane
// ---------------------------------------------------------------------------

/// @brief `impl::quantScale<uint8_t>` for the four pixels of `quad` at once:
/// `(v * maxValue + 127) / 255` per byte, in the same byte order.
/// @note TWO PIXELS PER 32-BIT REGISTER, in 16-bit fields -- bytes 0 and 2 in
/// one, 1 and 3 in the other -- because `v * maxValue + 127` reaches 65152 at
/// v = maxValue = 255, which fits a field and does not carry out of it.
/// @note The divide is `floor(x / 255) == (x + 1 + (x >> 8)) >> 8`, true for
/// every x below 65535 (checked exhaustively; it fails only AT 65535), so the
/// quotient is byte 1 of each field. Every sum stays below 65536.
/// @note A SECOND SPELLING of the host's one definition, which the scalar arms
/// call directly. The suite holds it to the host's quantizer at every byte
/// value, in every byte position, at every depth 1..8 -- the whole domain.
__device__ __forceinline__ uint32_t quantScaleQuad(uint32_t quad, unsigned maxValue) {
    const uint32_t lo = __byte_perm(quad, 0u, 0x4240u) * maxValue + 0x007F007Fu;
    const uint32_t hi = __byte_perm(quad, 0u, 0x4341u) * maxValue + 0x007F007Fu;
    const uint32_t sl = lo + 0x00010001u + __byte_perm(lo, 0u, 0x4341u);
    const uint32_t sh = hi + 0x00010001u + __byte_perm(hi, 0u, 0x4341u);
    return __byte_perm(sl, sh, 0x7351u);
}

/// @brief Bit `p` of sixteen quantized pixels, LSB = the lowest x: one `__dp4a`
/// per quad, the second of each pair weighing its bits 16..128.
__device__ __forceinline__ unsigned planeHalf(const uint32_t (&q)[4], unsigned p) {
    const unsigned lo = __dp4a((q[1] >> p) & 0x01010101u, 0x80402010u,
                               __dp4a((q[0] >> p) & 0x01010101u, 0x08040201u, 0u));
    const unsigned hi = __dp4a((q[3] >> p) & 0x01010101u, 0x80402010u,
                               __dp4a((q[2] >> p) & 0x01010101u, 0x08040201u, 0u));
    return lo | (hi << 8);
}

/// @note One lane, sixteen pixels: one aligned 16-byte load -- 512 B per warp
/// load instruction, sixteen bytes in flight per thread -- scaled four at a
/// time, and every plane extracted from the same four registers. Two lanes are
/// one word, the even lane holding its low half. A lane whose pixels straddle
/// `width` assembles its quads from byte loads with zeros past `width`, and
/// `quantScale(0)` is 0 in every plane -- so the padding invariant holds by
/// arithmetic, and the bytes between `width` and the stride are never read.
/// @note TWO PLANES PER STORE INSTRUCTION. After a pair's shuffle both lanes
/// hold both words; the even lane stores plane p and the odd lane plane p + 1,
/// each set of sixteen lanes writing 64 contiguous bytes of its own plane. An
/// odd N ends on a store with only the even lanes live.
__global__ void packQuantKernelWideLane(DeviceImageConstView<uint8_t> src,
                                        DeviceBinMatView planeBlock, unsigned n,
                                        unsigned maxValue, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (y >= src.height) return;
    const size_t x0 = static_cast<size_t>(blockIdx.x) * kWideWarpPixels +
                      static_cast<size_t>(lane) * kWideLanePixels;
    const uint8_t* row = src.row(y);

    uint32_t q[4];
    if (x0 + kWideLanePixels <= src.width) {
        const uint4 v = *reinterpret_cast<const uint4*>(row + x0);
        q[0] = quantScaleQuad(v.x, maxValue);
        q[1] = quantScaleQuad(v.y, maxValue);
        q[2] = quantScaleQuad(v.z, maxValue);
        q[3] = quantScaleQuad(v.w, maxValue);
    } else {
        uint32_t b[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (unsigned k = 0; k < kWideLanePixels; ++k) {
            const size_t x = x0 + k;
            if (x < src.width)
                b[k >> 2] |= static_cast<uint32_t>(row[x]) << (8u * (k & 3u));
        }
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) q[j] = quantScaleQuad(b[j], maxValue);
    }

    const size_t wordIdx =
        static_cast<size_t>(blockIdx.x) * (kWideWarpPixels / 32) + (lane >> 1);
    const unsigned odd = lane & 1u;
    for (unsigned p = 0; p < n; p += 2) {
        uint32_t a = planeHalf(q, p) << (16u * odd);
        a |= __shfl_xor_sync(0xFFFFFFFFu, a, 1);
        uint32_t b = 0;
        if (p + 1 < n) {
            b = planeHalf(q, p + 1) << (16u * odd);
            b |= __shfl_xor_sync(0xFFFFFFFFu, b, 1);
        }
        const unsigned plane = p + odd;
        if (plane < n && wordIdx < words) {
            planeBlock.row(static_cast<size_t>(plane) * src.height + y)[wordIdx] =
                odd ? b : a;
        }
    }
}

__global__ void unpackKernel(DeviceBinMatConstView src, DeviceImageView<uint8_t> dst,
                             uint8_t onValue, uint8_t zeroValue) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y;
    if (x >= src.width || y >= src.height) return;
    const uint32_t word = src.row(y)[x >> 5];
    dst.row(y)[x] = ((word >> (x & 31u)) & 1u) ? onValue : zeroValue;
}

/// @brief Warp-per-word launch geometry, shared by the grid-stride packers.
dim3 warpGrid(size_t words, size_t height, const dim3& block) {
    const size_t warps = (words * height + block.y - 1) / block.y;
    return dim3(static_cast<unsigned>(warps < 4096 ? (warps ? warps : 1) : 4096));
}

/// @brief One warp per (unit, row), a block being `block.y` consecutive rows of
/// one unit column -- the row-grid arms' and packQuant's wide lane's shape.
/// @param units Work units along a row: eight words for a ballot arm, 512
/// pixels for the wide lane.
/// @note Rows in the block, not units, because a unit is wide: a 752-pixel row
/// is three ballot units and two wide-lane ones, and a block of eight units
/// along one row would leave most of its warps with nothing to do.
dim3 rowBandGrid(size_t units, size_t height, const dim3& block) {
    return dim3(static_cast<unsigned>(units ? units : 1),
                static_cast<unsigned>((height + block.y - 1) / block.y));
}

/// @brief One rule, either ballot arm. The rule stays a TEMPLATE parameter and
/// the arm is the runtime choice, so the switch over `PackRule` is written
/// once rather than once per arm -- two copies of it could select different
/// kernels for the same rule and nothing would say so.
template <typename SrcT, PackRule R>
void launchBallotArm(bool rowGridArm, DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                     size_t dstRow, SrcT t, size_t words, const dim3& block,
                     cudaStream_t stream) {
    if (rowGridArm) {
        const size_t units = (words + kBallotWords - 1) / kBallotWords;
        packKernelRowGrid<SrcT, R><<<rowBandGrid(units, src.height, block), block, 0,
                                     stream>>>(src, dst, dstRow, t, words);
    } else {
        packKernel<SrcT, R><<<warpGrid(words, src.height, block), block, 0, stream>>>(
            src, dst, dstRow, t, words);
    }
}

template <typename SrcT>
cudaError_t launchPack(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                       size_t dstRow, PackRule rule, SrcT t, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width, "cuda packRows: src and dst must share a width");
    BINCV_ASSERT(dstRow + src.height <= dst.height, "cuda packRows: chunk runs past dst");
    if (dst.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda packRows: a non-empty image needs non-null pointers");
    BINCV_ASSERT(src.stride >= src.width,
                 "cuda packRows: src's stride must cover a whole row");
    BINCV_ASSERT(dst.stride >= rowWords(dst.width),
                 "cuda packRows: dst's stride must cover a whole row");
    const size_t words = rowWords(dst.width);
    const dim3 block(32, 8);  // eight warps
    const bool rowGridArm =
        impl::packRowGridEnabled() && impl::packRowGridApplies(src.height);

    switch (rule) {
        case PackRule::NonZero:
            launchBallotArm<SrcT, PackRule::NonZero>(rowGridArm, src, dst, dstRow, t, words,
                                                     block, stream);
            break;
        case PackRule::GreaterThan:
            launchBallotArm<SrcT, PackRule::GreaterThan>(rowGridArm, src, dst, dstRow, t,
                                                         words, block, stream);
            break;
        case PackRule::GreaterEqual:
            launchBallotArm<SrcT, PackRule::GreaterEqual>(rowGridArm, src, dst, dstRow, t,
                                                          words, block, stream);
            break;
    }
    return cudaGetLastError();
}

template <typename SrcT>
cudaError_t launchPackQuant(DeviceImageConstView<SrcT> src, DeviceBinMatView planeBlock,
                            size_t n, cudaStream_t stream) {
    BINCV_ASSERT(n >= 1 && n <= 8, "cuda packQuant: N outside QuantMat's supported range");
    BINCV_ASSERT(src.width == planeBlock.width &&
                     planeBlock.height == n * src.height,
                 "cuda packQuant: plane block must be width x (N * height)");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && planeBlock.ptr != nullptr,
                 "cuda packQuant: a non-empty image needs non-null pointers");
    const size_t words = rowWords(planeBlock.width);
    const dim3 block(32, 8);
    const unsigned maxValue = (1u << n) - 1u;
    // The wide lane is one level below the row grid; with the row grid off it
    // selects nothing.
    if (impl::packRowGridEnabled() && impl::packRowGridApplies(src.height)) {
        if constexpr (sizeof(SrcT) == 1) {
            if (impl::packQuantWideLaneEnabled() &&
                impl::packWideLaneApplies(src.stride, src.ptr, sizeof(SrcT))) {
                const DeviceImageConstView<uint8_t> src8{
                    reinterpret_cast<const uint8_t*>(src.ptr), src.width, src.height,
                    src.stride};
                const size_t units = (src.width + kWideWarpPixels - 1) / kWideWarpPixels;
                packQuantKernelWideLane<<<rowBandGrid(units, src.height, block), block, 0,
                                          stream>>>(src8, planeBlock,
                                                    static_cast<unsigned>(n), maxValue,
                                                    words);
                return cudaGetLastError();
            }
        }
        const size_t units = (words + kBallotWords - 1) / kBallotWords;
        packQuantKernelRowGrid<SrcT><<<rowBandGrid(units, src.height, block), block, 0,
                                       stream>>>(src, planeBlock, static_cast<unsigned>(n),
                                                 maxValue, words);
        return cudaGetLastError();
    }
    const dim3 grid = warpGrid(words, src.height, block);
    packQuantKernel<SrcT><<<grid, block, 0, stream>>>(src, planeBlock,
                                                      static_cast<unsigned>(n), maxValue,
                                                      words);
    return cudaGetLastError();
}

} // namespace

namespace impl {

bool& packRowGridEnabled() {
    static bool on = true;
    return on;
}

bool& packQuantWideLaneEnabled() {
    static bool on = true;
    return on;
}

bool packWideLaneApplies(size_t stride, const void* base, size_t srcElemSize) {
    // uint8 only: the scale treats each 32-bit quarter of the load as four byte
    // pixels.
    if (srcElemSize != 1) return false;
    // The load is an aligned 16-byte access, so the base AND every row it
    // strides to must be 16-byte aligned -- and a tight stride is 16-byte
    // aligned only when the width is.
    if (stride % 16u != 0u) return false;
    return (reinterpret_cast<std::uintptr_t>(base) % 16u) == 0u;
}

bool packRowGridApplies(size_t height) {
    // The row-grid family puts a band of eight rows in gridDim.y, which the
    // hardware caps, and shares this gate so one test selects the whole family.
    // The gate admits the row count against the cap -- stricter than the band
    // launch needs -- and above it the grid-stride arm, whose whole reason to
    // exist is that one flat dimension has no such bound, runs instead.
    return height <= kMaxGridY;
}

} // namespace impl

cudaError_t packBits(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     PackRule rule, uint8_t threshold, cudaStream_t stream) {
    BINCV_ASSERT(src.height == dst.height,
                 "cuda packBits: src and dst must have the same dimensions");
    return launchPack<uint8_t>(src, dst, 0, rule, threshold, stream);
}

cudaError_t packBits(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     PackRule rule, uint16_t threshold, cudaStream_t stream) {
    BINCV_ASSERT(src.height == dst.height,
                 "cuda packBits: src and dst must have the same dimensions");
    return launchPack<uint16_t>(src, dst, 0, rule, threshold, stream);
}

cudaError_t packRows(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint8_t threshold,
                     cudaStream_t stream) {
    return launchPack<uint8_t>(src, dst, dstRow, rule, threshold, stream);
}

cudaError_t packRows(DeviceImageConstView<uint16_t> src, DeviceBinMatView dst,
                     size_t dstRow, PackRule rule, uint16_t threshold,
                     cudaStream_t stream) {
    return launchPack<uint16_t>(src, dst, dstRow, rule, threshold, stream);
}

cudaError_t packQuant(DeviceImageConstView<uint8_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream) {
    return launchPackQuant<uint8_t>(src, planeBlock, n, stream);
}

cudaError_t packQuant(DeviceImageConstView<uint16_t> src, DeviceBinMatView planeBlock,
                      size_t n, cudaStream_t stream) {
    return launchPackQuant<uint16_t>(src, planeBlock, n, stream);
}

cudaError_t unpackTo8Bit(DeviceBinMatConstView src, DeviceImageView<uint8_t> dst,
                         uint8_t onValue, uint8_t zeroValue, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda unpackTo8Bit: src and dst must have the same dimensions");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda unpackTo8Bit: a non-empty image needs non-null pointers");
    const dim3 block(256, 1);
    const dim3 grid(static_cast<unsigned>((src.width + block.x - 1) / block.x),
                    static_cast<unsigned>(src.height));
    unpackKernel<<<grid, block, 0, stream>>>(src, dst, onValue, zeroValue);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
