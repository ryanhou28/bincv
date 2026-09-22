// The device arm of ops/pyramid.hpp's BOX route.
//
// TWO ARMS, held to the same output in one binary.
//
//   BALLOT (reference)   one WARP per destination word, lane j owning
//                        destination pixel 32i + j. Value arithmetic per lane,
//                        __ballot_sync as the repack -- one instruction that
//                        packs 32 lanes' bits in the format's own bit order.
//                        Generic: NIn and NOut are runtime values, so every one
//                        of the 64 (NIn, NOut) pairs is reachable.
//
//   BIT-SLICED (fast)    ONE THREAD per destination word, running the host
//                        kernel's own word-parallel arithmetic with NIn and
//                        NOut as TEMPLATE parameters. One thread does 32 pixels
//                        of work per instruction, which is binCV's premise
//                        expressed on a GPU lane. Instantiated for a bounded
//                        set of pairs; anything else falls back to the
//                        reference arm, and pyrFastArmCovers() reports which.
//
// THE SOURCE-WORD GUARD IS NOT OPTIONAL AND IT IS THE HOST'S. Destination word
// i reads source words 2i and 2i+1, and 2i+1 does not exist when the source row
// holds an odd number of words -- srcWidth == 32 is the smallest case: one
// source word, one destination word, and sixteen lanes (or the single
// bit-sliced thread) reaching for word 1. Under this backend's tight stride
// there is no slack behind the last plane's last row, so that read is past the
// allocation. Both arms carry the guard and the suite names both widths.

#include "bincv/cuda/pyramid.hpp"

#include "bincv/ops/bitslice.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// Everything uniform over the launch, resolved once on the host and passed as
/// a POD -- the constant bank, at zero register cost.
struct PyrPod {
    size_t srcWords;
    size_t srcHeight;
    size_t dstWords;
    size_t dstWidth;
    size_t dstHeight;
    size_t lastWord;      // (dstWidth - 1) / 32
    unsigned lastBitPos;  // (dstWidth - 1) % 32
    unsigned nIn;
    unsigned nOut;
    unsigned maxOut;   // 2^NOut - 1
    unsigned rounding; // 2 * (2^NIn - 1)
    unsigned divisor;  // 4 * (2^NIn - 1)
    uint32_t tailMask;
    bool oddWidth;
};

// ---------------------------------------------------------------------------
// ARM A -- warp per destination word, per-lane value arithmetic, ballot repack
// ---------------------------------------------------------------------------

__global__ void pyrDownBoxBallotKernel(DevicePlaneBlockConstView src,
                                       DevicePlaneBlockView dst, PyrPod p) {
    const unsigned lane = threadIdx.x;  // 0..31, and x is the warp's axis
    const size_t warpsPerBlock = blockDim.y;
    const size_t total = p.dstWords * p.dstHeight;
    const size_t step = static_cast<size_t>(gridDim.x) * warpsPerBlock;

    for (size_t idx = blockIdx.x * warpsPerBlock + threadIdx.y; idx < total; idx += step) {
        const size_t y = idx / p.dstWords;
        const size_t i = idx - y * p.dstWords;

        // The vertical half of the subsample is a row index. 2y+1 past the last
        // source row replicates 2y -- the same edge rule the odd-width branch
        // applies horizontally.
        const size_t row0 = 2 * y;
        const size_t row1 = (2 * y + 1 < p.srcHeight) ? (2 * y + 1) : row0;

        // Destination word i covers source columns [64i, 64i + 64), i.e. source
        // words 2i and 2i+1 exactly. Lane j's two source pixels are the 2-bit
        // field at bit 2*(j & 15) of word 2i + (j >> 4) -- both of them, in one
        // shift and one mask.
        const size_t x = i * 32u + lane;
        const size_t sw = 2 * i + (lane >> 4);
        const unsigned sh = 2u * (lane & 15u);
        const bool haveWord = sw < p.srcWords;

        unsigned tl = 0u, tr = 0u, bl = 0u, br = 0u;
        for (unsigned q = 0; q < p.nIn; ++q) {
            const uint32_t w0 = haveWord ? src.row(q, row0)[sw] : 0u;
            const uint32_t w1 = haveWord ? src.row(q, row1)[sw] : 0u;
            const unsigned pairTop = (w0 >> sh) & 3u;
            const unsigned pairBottom = (w1 >> sh) & 3u;
            tl |= (pairTop & 1u) << q;
            tr |= ((pairTop >> 1) & 1u) << q;
            bl |= (pairBottom & 1u) << q;
            br |= ((pairBottom >> 1) & 1u) << q;
        }

        // The one destination column whose right partner is source column
        // src.width: replicate the left one, BEFORE the arithmetic, so a dirty
        // source padding bit never reaches a live destination pixel.
        if (p.oddWidth && x + 1 == p.dstWidth) {
            tr = tl;
            br = bl;
        }

        const unsigned s = tl + tr + bl + br;
        // floor((S * (2^NOut - 1) + 2 * (2^NIn - 1)) / (4 * (2^NIn - 1))) --
        // the host expression. S <= 1020 and maxOut <= 255, so the numerator is
        // under 2^18 and a 32-bit divide is exact.
        unsigned v = (s * p.maxOut + p.rounding) / p.divisor;
        // Lanes past the width contribute 0 to every ballot, so the padding bits
        // are zero BY CONSTRUCTION rather than by masking -- the packQuant
        // invariant, reached the same way.
        if (x >= p.dstWidth) v = 0u;

        for (unsigned q = 0; q < p.nOut; ++q) {
            const uint32_t word = __ballot_sync(0xFFFFFFFFu, ((v >> q) & 1u) != 0u);
            if (lane == 0u) dst.row(q, y)[i] = word;
        }
    }
}

// ---------------------------------------------------------------------------
// ARM B -- one thread per destination word, the host's bit-sliced arithmetic
//
// Every function below is ops/pyramid.hpp's, at uint32_t with NIn and NOut
// fixed at compile time so the plane arrays unroll into registers. maj3 and
// thresholdGE are NOT restated: they carry BINCV_HOST_DEVICE, so this kernel
// calls the host library's own definitions. maj3 is one LOP3.LUT and the
// three-term XOR beside it is another, so a full adder is two instructions.
// ---------------------------------------------------------------------------

/// ops/resample.hpp's word-local unshuffle, at 32 bits: even bits of `x`
/// compacted into its low 16.
__device__ inline uint32_t gatherEvenBits32(uint32_t x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    x = (x | (x >> 8)) & 0x0000ffffu;
    return x;
}

/// The four 2x2 phases of one destination word, for one source plane.
__device__ inline void gatherPhases32(uint32_t w0, uint32_t w1, uint32_t& evenPhase,
                                      uint32_t& oddPhase) {
    evenPhase = gatherEvenBits32(w0) | (gatherEvenBits32(w1) << 16);
    oddPhase = gatherEvenBits32(w0 >> 1) | (gatherEvenBits32(w1 >> 1) << 16);
}

template <size_t NA, size_t NB>
__device__ inline void addPlanesD(const uint32_t* a, const uint32_t* b, uint32_t* out) {
    constexpr size_t N = NA > NB ? NA : NB;
    uint32_t carry = 0u;
#pragma unroll
    for (size_t p = 0; p < N; ++p) {
        const uint32_t x = (p < NA) ? a[p] : 0u;
        const uint32_t y = (p < NB) ? b[p] : 0u;
        out[p] = x ^ y ^ carry;
        carry = bincv::maj3<uint32_t>(x, y, carry);
    }
    out[N] = carry;
}

/// `sum = a + b + c + d`, four NIn-bit operands, NIn+2 planes: the tree of
/// three ripple-carry additions, 3*NIn + 1 full-adder stages, linear in NIn.
template <size_t NIn>
__device__ inline void boxSum4D(const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                const uint32_t* d, uint32_t* sum) {
    uint32_t left[NIn + 1];
    uint32_t right[NIn + 1];
    addPlanesD<NIn, NIn>(a, b, left);
    addPlanesD<NIn, NIn>(c, d, right);
    addPlanesD<NIn + 1, NIn + 1>(left, right, sum);
}

/// `out = (v << Shift) - v`, i.e. `v * (2^Shift - 1)`: one borrow chain,
/// because an all-ones constant is one less than a power of two.
template <size_t N, size_t Shift>
__device__ inline void multiplyByAllOnesD(const uint32_t* v, uint32_t* out) {
    constexpr size_t Total = N + Shift;
    uint32_t borrow = 0u;
#pragma unroll
    for (size_t p = 0; p < Total; ++p) {
        const uint32_t hi = (p >= Shift && (p - Shift) < N) ? v[p - Shift] : 0u;
        const uint32_t lo = (p < N) ? v[p] : 0u;
        out[p] = hi ^ lo ^ borrow;
        // The full-subtractor borrow is maj3 with the minuend inverted.
        borrow = bincv::maj3<uint32_t>(~hi, lo, borrow);
    }
}

template <size_t N, unsigned C>
__device__ inline void addConstantD(uint32_t* v) {
    uint32_t carry = 0u;
#pragma unroll
    for (size_t p = 0; p < N; ++p) {
        const uint32_t b = (p < 32u && ((C >> p) & 1u) != 0u) ? 0xFFFFFFFFu : 0u;
        const uint32_t x = v[p];
        v[p] = x ^ b ^ carry;
        carry = bincv::maj3<uint32_t>(x, b, carry);
    }
}

template <size_t N, unsigned C>
__device__ inline void subtractConstantWhereD(uint32_t* v, uint32_t mask) {
    uint32_t borrow = 0u;
#pragma unroll
    for (size_t p = 0; p < N; ++p) {
        const uint32_t b = (p < 32u && ((C >> p) & 1u) != 0u) ? mask : 0u;
        const uint32_t x = v[p];
        v[p] = x ^ b ^ borrow;
        borrow = bincv::maj3<uint32_t>(~x, b, borrow);
    }
}

/// One quotient bit of the restoring division, most significant first.
/// @note NO MULTIPLY ANYWHERE. Arm A spends one 32-bit hardware divide per LANE
/// to produce at most eight meaningful bits; this spends NOut compare-and-
/// subtract steps on whole WORDS, each covering 32 pixels at once. That is
/// where the two arms' work actually differs, and it grows with NOut.
template <unsigned Divisor, size_t N, size_t Q>
__device__ inline void divideStageD(uint32_t* value, uint32_t* quotient) {
    if constexpr (Q > 0) {
        constexpr unsigned kScaled = Divisor << (Q - 1);
        const uint32_t fits = bincv::thresholdGE<uint32_t>(value, N, kScaled);
        quotient[Q - 1] = fits;
        subtractConstantWhereD<N, kScaled>(value, fits);
        divideStageD<Divisor, N, Q - 1>(value, quotient);
    }
}

template <size_t NIn, size_t NOut>
__device__ inline void requantizeD(const uint32_t* sum, uint32_t* out) {
    constexpr unsigned kMaxIn = (1u << NIn) - 1u;
    constexpr size_t kWidth = NIn + NOut + 2;
    uint32_t scaled[kWidth];
    multiplyByAllOnesD<NIn + 2, NOut>(sum, scaled);
    addConstantD<kWidth, 2u * kMaxIn>(scaled);
    divideStageD<4u * kMaxIn, kWidth, NOut>(scaled, out);
}

template <size_t NIn, size_t NOut>
__global__ void pyrDownBoxSlicedKernel(DevicePlaneBlockConstView src,
                                       DevicePlaneBlockView dst, PyrPod p) {
    const size_t total = p.dstWords * p.dstHeight;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t y = idx / p.dstWords;
        const size_t i = idx - y * p.dstWords;

        const size_t row0 = 2 * y;
        const size_t row1 = (2 * y + 1 < p.srcHeight) ? (2 * y + 1) : row0;
        const size_t lo = 2 * i;

        uint32_t topLeft[NIn];
        uint32_t topRight[NIn];
        uint32_t bottomLeft[NIn];
        uint32_t bottomRight[NIn];
#pragma unroll
        for (size_t q = 0; q < NIn; ++q) {
            const uint32_t* r0 = src.row(q, row0);
            const uint32_t* r1 = src.row(q, row1);
            const uint32_t a0 = (lo < p.srcWords) ? r0[lo] : 0u;
            const uint32_t a1 = (lo + 1 < p.srcWords) ? r0[lo + 1] : 0u;
            const uint32_t b0 = (lo < p.srcWords) ? r1[lo] : 0u;
            const uint32_t b1 = (lo + 1 < p.srcWords) ? r1[lo + 1] : 0u;
            gatherPhases32(a0, a1, topLeft[q], topRight[q]);
            gatherPhases32(b0, b1, bottomLeft[q], bottomRight[q]);
        }

        if (p.oddWidth && i == p.lastWord) {
            const uint32_t lastBit = uint32_t{1} << p.lastBitPos;
#pragma unroll
            for (size_t q = 0; q < NIn; ++q) {
                topRight[q] = (topRight[q] & ~lastBit) | (topLeft[q] & lastBit);
                bottomRight[q] = (bottomRight[q] & ~lastBit) | (bottomLeft[q] & lastBit);
            }
        }

        uint32_t sum[NIn + 2];
        boxSum4D<NIn>(topLeft, topRight, bottomLeft, bottomRight, sum);
        uint32_t value[NOut];
        requantizeD<NIn, NOut>(sum, value);

#pragma unroll
        for (size_t q = 0; q < NOut; ++q) {
            dst.row(q, y)[i] =
                (i + 1 == p.dstWords) ? (value[q] & p.tailMask) : value[q];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

inline unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

template <size_t NIn, size_t NOut>
cudaError_t launchSliced(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                         const PyrPod& p, cudaStream_t stream) {
    constexpr unsigned kBlock = 256;
    pyrDownBoxSlicedKernel<NIn, NOut>
        <<<gridFor(p.dstWords * p.dstHeight, kBlock), kBlock, 0, stream>>>(src, dst, p);
    return cudaGetLastError();
}

/// The bounded instantiation set: the pipeline ladder's own steps, the
/// identities, and the 8->8 case. Sixty-four kernels for the other pairs is
/// code size for nothing, and a pair outside the set is not silently slower --
/// pyrFastArmCovers() reports it and the benchmark's gate-excluded row reads
/// ~1.00x because BOTH switch positions run the same reference arm.
bool dispatchSliced(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                    const PyrPod& p, cudaStream_t stream, cudaError_t& err) {
    const size_t nIn = src.planes;
    const size_t nOut = dst.planes;
#define BINCV_PYR_CASE(a, b)                               \
    if (nIn == (a) && nOut == (b)) {                       \
        err = launchSliced<(a), (b)>(src, dst, p, stream); \
        return true;                                       \
    }
    BINCV_PYR_CASE(1, 1)
    BINCV_PYR_CASE(1, 3)
    BINCV_PYR_CASE(3, 3)
    BINCV_PYR_CASE(3, 4)
    BINCV_PYR_CASE(4, 4)
    BINCV_PYR_CASE(4, 5)
    BINCV_PYR_CASE(5, 5)
    BINCV_PYR_CASE(8, 8)
#undef BINCV_PYR_CASE
    return false;
}

} // namespace

namespace impl {

bool& pyrBitSlicedEnabled() {
    static bool enabled = true;
    return enabled;
}

} // namespace impl

bool pyrFastArmCovers(size_t nIn, size_t nOut) {
    return (nIn == 1 && nOut == 1) || (nIn == 1 && nOut == 3) || (nIn == 3 && nOut == 3) ||
           (nIn == 3 && nOut == 4) || (nIn == 4 && nOut == 4) || (nIn == 4 && nOut == 5) ||
           (nIn == 5 && nOut == 5) || (nIn == 8 && nOut == 8);
}

cudaError_t pyrDownBox(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                       cudaStream_t stream) {
    const bool domainOk = src.planes >= 1 && src.planes <= 8 && dst.planes >= 1 &&
                          dst.planes <= 8 && dst.width == pyrDownWidth(src.width) &&
                          dst.height == pyrDownHeight(src.height) &&
                          src.stride >= rowWords(src.width) &&
                          dst.stride >= rowWords(dst.width);
    BINCV_ASSERT(domainOk,
                 "cuda pyrDownBox: planes must be 1..8 and dst must be "
                 "pyrDownWidth(src.width) x pyrDownHeight(src.height) with a stride "
                 "covering a whole row");
    if (!domainOk) return cudaErrorInvalidValue;

    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    if (src.ptr == nullptr || dst.ptr == nullptr) return cudaErrorInvalidValue;

    const unsigned maxIn = (1u << src.planes) - 1u;

    PyrPod p{};
    p.srcWords = rowWords(src.width);
    p.srcHeight = src.height;
    p.dstWords = rowWords(dst.width);
    p.dstWidth = dst.width;
    p.dstHeight = dst.height;
    p.lastWord = (dst.width - 1) / 32u;
    p.lastBitPos = static_cast<unsigned>((dst.width - 1) % 32u);
    p.nIn = static_cast<unsigned>(src.planes);
    p.nOut = static_cast<unsigned>(dst.planes);
    p.maxOut = (1u << dst.planes) - 1u;
    p.rounding = 2u * maxIn;
    p.divisor = 4u * maxIn;
    p.tailMask = rowTailMask(dst.width);
    p.oddWidth = (src.width % 2) != 0;

    if (impl::pyrBitSlicedEnabled()) {
        cudaError_t err = cudaSuccess;
        if (dispatchSliced(src, dst, p, stream, err)) return err;
    }

    // The reference arm: one warp per destination word, eight warps per block.
    constexpr unsigned kWarpsPerBlock = 8;
    const dim3 block(32, kWarpsPerBlock);
    const unsigned grid = gridFor(p.dstWords * p.dstHeight, kWarpsPerBlock);
    pyrDownBoxBallotKernel<<<grid, block, 0, stream>>>(src, dst, p);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
