// The device sensor stage. One warp produces one packed word: lane x of the
// warp reads pixel 32*i + x, evaluates the rule, and __ballot_sync IS the
// packed word -- the format's 32-bit granule and the warp width coinciding.
// The N-bit packer is the same shape, N ballots deep.
//
// ---------------------------------------------------------------------------
// THREE ARMS, TWO SWITCHES, AND WHY THE FIRST ONE ALONE WAS NOT ENOUGH
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
//   ROW-GRID ARM (`packRowGridEnabled`, default on). blockIdx.y IS the row.
//     Both divides are deleted outright rather than strength-reduced: the two
//     quantities the loop was computing are now the two components of the
//     launch shape. Identical body, identical output, one instantiation per
//     (SrcT, rule) exactly as before. Costs a gate -- gridDim.y is capped at
//     65535 -- and above that the grid-stride arm runs.
//
//   BYTE-LANE ARM (`packByteLaneEnabled`, default on, one level down). One
//     lane, FOUR pixels: a 32-bit load where the warp was issuing 32 one-byte
//     loads, `__vsetgeu4` for the four comparisons and `__dp4a` to fold four
//     byte flags into a nibble -- the shape edge.cu's byte-lane arm already
//     runs, on the same part, for the same reason. Eight lanes' nibbles are one
//     output word, gathered by a three-step shuffle butterfly.
//
// The byte-lane arm folds the rule into ONE unsigned cutoff (`v >= cutoff`)
// because `__vsetgeu4` has one spelling: NonZero is cutoff 1, GreaterThan is
// t + 1, GreaterEqual is t. That is the fold edge.cu's launcher already does
// with EdgeRelation, and it is why this arm is one kernel rather than three.
// A cutoff of 256 -- GreaterThan at t == 255, "nothing passes" -- cannot be
// expressed in a byte lane, so it is one of this arm's gates.
//
// SHARED OWNERSHIP IS THE HAZARD HERE. This kernel is `packBits`',
// `packRows`' and `cuda::threshold`'s, and `packQuantKernel` next to it is the
// N-deep twin with the same two divides. Both got the row grid; the byte-lane
// arm is `packKernel`'s alone, because a quantized lane resolves N planes from
// one value and there is no four-at-a-time spelling of that.

#include "bincv/cuda/pack.hpp"

#include <cstdint>

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief Pixels one warp of the byte-lane arm covers: 32 lanes x 4 = 4 words.
constexpr unsigned kPackLanePixelsPerWarp = 128;

/// @brief The largest `gridDim.y` a launch may ask for. The row-grid arms put
/// the image row in that dimension, so this is their domain bound.
constexpr size_t kMaxGridY = 65535;

/// @brief The three rules as ONE unsigned cutoff: `v >= cutoff` is each of them.
/// @note `NonZero` is `v >= 1`; `GreaterThan` is `v >= t + 1`; `GreaterEqual` is
/// `v >= t`. Computed in `unsigned`, so `t + 1` at the type's maximum is 256
/// (or 65536) -- "nothing passes" by arithmetic, with no special case.
template <typename SrcT>
unsigned packCutoff(PackRule rule, SrcT t) {
    switch (rule) {
        case PackRule::NonZero:
            return 1u;
        case PackRule::GreaterThan:
            return static_cast<unsigned>(t) + 1u;
        case PackRule::GreaterEqual:
            return static_cast<unsigned>(t);
    }
    return 1u;
}

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
// The row-grid arm -- the same body, with the two divides deleted
// ---------------------------------------------------------------------------

/// @note The body is the grid-stride arm's, character for character, from `x`
/// onward. What changed is above it: `y` is `blockIdx.y` and `i` is the x
/// dimension, so neither has to be recovered from a flat index.
/// @note The early return is taken by a WHOLE WARP or none of it -- `threadIdx.y`
/// is uniform across a warp at `blockDim.x == 32` -- so the `__ballot_sync`
/// below still sees all 32 lanes, which is what makes its mask legal.
template <typename SrcT, PackRule R>
__global__ void packKernelRowGrid(DeviceImageConstView<SrcT> src, DeviceBinMatView dst,
                                  size_t dstRow, SrcT t, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= words) return;
    const size_t y = blockIdx.y;
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

// ---------------------------------------------------------------------------
// The byte-lane arm -- four pixels per lane
// ---------------------------------------------------------------------------

/// @param cutoff The folded rule, `v >= cutoff`, 0..255 by this arm's gate.
/// @param cutoffV The same cutoff in all four byte lanes.
/// @note A lane whose quad straddles `width` -- at most one per row -- falls to
/// the scalar comparison for its four pixels, which is the SAME expression
/// the fast path folds, not a second spelling of it. Pixels past `width`
/// contribute 0, so the padding invariant holds by construction.
__global__ void packKernelByteLane(DeviceImageConstView<uint8_t> src, DeviceBinMatView dst,
                                   size_t dstRow, unsigned cutoff, uint32_t cutoffV,
                                   size_t words, size_t groups) {
    const unsigned lane = threadIdx.x;
    const size_t g = blockIdx.x * blockDim.y + threadIdx.y;
    // Uniform across the warp, so the shuffles below are still reached by all
    // 32 lanes -- the property the early return in the row-grid arm relies on.
    if (g >= groups) return;
    const size_t y = blockIdx.y;
    const size_t x0 = g * kPackLanePixelsPerWarp + static_cast<size_t>(lane) * 4;

    unsigned nib = 0;
    if (x0 + 4 <= src.width) {
        // The gate guarantees `src.ptr` and `src.stride` are 4-byte aligned, so
        // this quad is one aligned 32-bit load: 128 B per warp load instruction
        // against the 32 B a per-lane byte load moves.
        const uint32_t quad = reinterpret_cast<const uint32_t*>(src.row(y))[x0 >> 2];
        // Bytes of 0 or 1, then 1*b0 + 2*b1 + 4*b2 + 8*b3 in one IDP.4A --
        // LSB = the lowest x, which is the format's own bit order.
        nib = static_cast<unsigned>(__dp4a(__vsetgeu4(quad, cutoffV), 0x08040201u, 0u));
    } else {
        for (unsigned k = 0; k < 4; ++k) {
            const size_t x = x0 + k;
            if (x < src.width && static_cast<unsigned>(src.row(y)[x]) >= cutoff)
                nib |= (1u << k);
        }
    }

    // Eight lanes' nibbles are one output word. A three-step butterfly OR
    // leaves it in every lane of the group; lane 0 of the group stores it, and
    // the four stores of a warp are 16 contiguous bytes.
    uint32_t val = nib << (4u * (lane & 7u));
    val |= __shfl_xor_sync(0xFFFFFFFFu, val, 1);
    val |= __shfl_xor_sync(0xFFFFFFFFu, val, 2);
    val |= __shfl_xor_sync(0xFFFFFFFFu, val, 4);
    const size_t wordIdx = g * 4 + (lane >> 3);
    if ((lane & 7u) == 0 && wordIdx < words) dst.row(dstRow + y)[wordIdx] = val;
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

/// @note The N-deep twin of `packKernelRowGrid`, and the same one change: the
/// row is the grid's y dimension, so the two divides go. The plane loop,
/// the scale and the store are the grid-stride arm's, unchanged.
template <typename SrcT>
__global__ void packQuantKernelRowGrid(DeviceImageConstView<SrcT> src,
                                       DeviceBinMatView planeBlock, unsigned n,
                                       unsigned maxValue, size_t words) {
    const unsigned lane = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= words) return;
    const size_t y = blockIdx.y;
    const size_t x = i * 32 + lane;
    const unsigned value = (x < src.width)
                               ? bincv::impl::quantScale<SrcT>(src.row(y)[x], maxValue)
                               : 0u;
    for (unsigned p = 0; p < n; ++p) {
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((value >> p) & 1u) != 0u);
        if (lane == 0) planeBlock.row(static_cast<size_t>(p) * src.height + y)[i] = word;
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

/// @brief One block per (word group, ROW): the shape that deletes the divides.
/// @param units Work units along a row -- words for the row-grid arm, groups of
/// four words for the byte-lane one.
dim3 rowGrid(size_t units, size_t height, const dim3& block) {
    const size_t xBlocks = (units + block.y - 1) / block.y;
    return dim3(static_cast<unsigned>(xBlocks ? xBlocks : 1),
                static_cast<unsigned>(height));
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
        packKernelRowGrid<SrcT, R><<<rowGrid(words, src.height, block), block, 0, stream>>>(
            src, dst, dstRow, t, words);
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
    const dim3 block(32, 8);  // eight warps, eight words per block iteration
    const bool rowGridArm =
        impl::packRowGridEnabled() && impl::packRowGridApplies(src.height);

    // The byte-lane arm is one level below the row grid, as denseBitSlicedEnabled
    // sits below denseFastArmEnabled: with the row grid off it selects nothing,
    // because its own geometry IS the row grid.
    if constexpr (sizeof(SrcT) == 1) {
        const unsigned cutoff = packCutoff<SrcT>(rule, t);
        if (rowGridArm && impl::packByteLaneEnabled() &&
            impl::packByteLaneApplies(src.stride, src.ptr, sizeof(SrcT), cutoff)) {
            const size_t groups = (words + 3) / 4;
            const DeviceImageConstView<uint8_t> src8{
                reinterpret_cast<const uint8_t*>(src.ptr), src.width, src.height,
                src.stride};
            packKernelByteLane<<<rowGrid(groups, src.height, block), block, 0, stream>>>(
                src8, dst, dstRow, cutoff, cutoff * 0x01010101u, words, groups);
            return cudaGetLastError();
        }
    }

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
    if (impl::packRowGridEnabled() && impl::packRowGridApplies(src.height)) {
        packQuantKernelRowGrid<SrcT><<<rowGrid(words, src.height, block), block, 0,
                                       stream>>>(src, planeBlock,
                                                 static_cast<unsigned>(n), maxValue,
                                                 words);
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

bool& packByteLaneEnabled() {
    static bool on = true;
    return on;
}

bool packRowGridApplies(size_t height) {
    // The row lives in gridDim.y, which the hardware caps. Above the cap there
    // is no launch to make, so the grid-stride arm -- whose whole reason to
    // exist is that one flat dimension has no such bound -- runs instead.
    return height <= kMaxGridY;
}

bool packByteLaneApplies(size_t stride, const void* base, size_t srcElemSize,
                         unsigned cutoff) {
    // uint8 only: `__vsetgeu4` compares four BYTES. A 16-bit source has
    // `__vsetgeu2`, which is half the win for a second kernel to keep
    // bit-exact forever, and no caller has priced that trade.
    if (srcElemSize != 1) return false;
    // A byte lane cannot express "nothing passes" -- GreaterThan at t == 255.
    if (cutoff > 255u) return false;
    // The quad load is an aligned 32-bit access, so the base AND every row it
    // strides to must be 4-byte aligned.
    if (stride % 4u != 0u) return false;
    return (reinterpret_cast<std::uintptr_t>(base) % 4u) == 0u;
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
