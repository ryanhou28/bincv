// The device arm of ops/derivative.hpp.
//
// WHAT IS SHARED AND WHAT IS FORKED, at this file's granularity:
//
//   SHARED, by call and not by copy --
//     bincv::impl::borderIndex   the Tier 1 border rule (cv::borderInterpolate),
//                                evaluated on the HOST for the at most four
//                                out-of-image coordinates a 3-tap kernel has,
//                                and passed in as ptrdiff_t. There is no device
//                                border rule here at all.
//     bincv::impl::signedDifference  the sign-magnitude arithmetic: the ternary
//                                three-op spelling at N == 1, the ripple-borrow
//                                subtract plus conditional two's-complement
//                                negate above it. The canonical-zero rule is a
//                                property of those expressions, so it is
//                                inherited rather than re-established.
//     bincv::impl::rowBit        the format's own bit addressing, for the one
//                                border pixel per row per plane that the word
//                                recurrence cannot supply.
//
//   FORKED, because a parallel traversal has nothing in common with a row loop --
//     the host carries the left neighbour in a register across the row; a thread
//     that owns one word cannot, so it LOADS word i-1 instead. Adjacent lanes
//     read adjacent words, so the three loads per plane coalesce. Shared-memory
//     staging of the row is NOT attempted: the dense matcher measured that trade
//     on this device at 1.28x slower, because the redundant loads were already
//     L1 hits.
//
// One thread owns one destination word index (y, i) -- 32 pixels of every
// destination plane -- flattened as y * words + i, grid-strided, which is
// logic.cu's shape for an op that is pointwise in the word index up to the
// +/-1 neighbours.

#include "bincv/cuda/derivative.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// Everything uniform over the launch, resolved once on the host. A POD kernel
/// argument lives in the constant bank at zero register cost -- census.cu's
/// reason for passing its offsets the same way.
struct DerivPod {
    size_t width;
    size_t height;  ///< ONE plane's height
    size_t words;   ///< words per row
    uint32_t tailMask;
    uint32_t lastLiveBit;  ///< the bit holding column width - 1
    ptrdiff_t leftSrc;     ///< source column for the out-of-image column -1
    ptrdiff_t rightSrc;    ///< source column for the out-of-image column width
    ptrdiff_t topSrc;      ///< source row for the out-of-image row -1
    ptrdiff_t bottomSrc;   ///< source row for the out-of-image row height
    uint32_t fill;         ///< a whole out-of-image ROW, under BORDER_CONSTANT
    bool borderValue;
};

/// The two horizontal taps for one plane of one destination word.
/// @note The source's trailing word is deliberately NOT masked, and the device
/// inherits the host's measured coupling verbatim: the dirty padding bit
/// that `cur >> 1` moves lands on exactly the bit the right-border fixup
/// below overwrites. With `width = k*32 + r` and `r != 0` the leak arrives
/// at bit r - 1, and `lastLiveBit` IS bit (width - 1) % 32 == r - 1; at
/// r == 0 a mask would be a no-op anyway. The coupling is stated because
/// anything that narrows the fixup must re-establish both the last column's
/// right tap and the padding invariant explicitly rather than assume they
/// survived.
__device__ __forceinline__ void horizontalTaps(const uint32_t* srcRow, size_t i, bool last,
                                               const DerivPod& p, uint32_t& a, uint32_t& b) {
    const uint32_t cur = srcRow[i];
    // Bit 0 of word i+1 is pixel (i+1)*32, inside `width` for every i that has
    // a successor, so it needs no mask.
    const uint32_t nxt = last ? 0u : srcRow[i + 1];
    uint32_t prev;
    if (i > 0) {
        prev = srcRow[i - 1];
    } else {
        // The synthetic "word before word 0": only its top bit is read, and it
        // holds whatever column 0's left neighbour resolves to. That keeps the
        // shift recurrence identical at i == 0 and everywhere else.
        const bool bit = (p.leftSrc < 0)
                             ? p.borderValue
                             : bincv::impl::rowBit<uint32_t>(
                                   srcRow, static_cast<size_t>(p.leftSrc));
        prev = bit ? 0x80000000u : 0u;
    }
    a = (cur >> 1) | (nxt << 31);
    b = (cur << 1) | (prev >> 31);
    if (last) {
        // Column width - 1's right tap: the one pixel of this row the word
        // recurrence cannot supply.
        const bool bit = (p.rightSrc < 0)
                             ? p.borderValue
                             : bincv::impl::rowBit<uint32_t>(
                                   srcRow, static_cast<size_t>(p.rightSrc));
        a = (a & ~p.lastLiveBit) | (bit ? p.lastLiveBit : 0u);
    }
}

/// Stores N magnitude words and the sign word, trailing word masked in ALL of
/// them. The sign plane needs the mask as much as the magnitude planes do: a
/// set padding bit there is a "negative zero" and so a canonical-zero
/// violation.
template <size_t N>
__device__ __forceinline__ void storeSigned(DevicePlaneBlockView dst, size_t y, size_t i,
                                            bool last, const DerivPod& p,
                                            const uint32_t (&m)[N], uint32_t s) {
#pragma unroll
    for (size_t q = 0; q < N; ++q) dst.row(q, y)[i] = last ? (m[q] & p.tailMask) : m[q];
    dst.row(N, y)[i] = last ? (s & p.tailMask) : s;
}

// ---------------------------------------------------------------------------
// d/dx -- the horizontal taps are bits within a row, so this kernel shifts
// ---------------------------------------------------------------------------
template <size_t N>
__global__ void derivativeXKernel(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                                  DerivPod p) {
    const size_t total = p.words * p.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / p.words;
        const size_t i = idx - y * p.words;
        const bool last = (i + 1 == p.words);

        uint32_t a[N];
        uint32_t b[N];
#pragma unroll
        for (size_t q = 0; q < N; ++q)
            horizontalTaps(src.row(q, y), i, last, p, a[q], b[q]);

        uint32_t m[N];
        uint32_t s = 0u;
        bincv::impl::signedDifference<N, uint32_t, false>(a, b, m, s);
        storeSigned<N>(dst, y, i, last, p, m, s);
    }
}

// ---------------------------------------------------------------------------
// d/dy -- NO BIT MANIPULATION AT ALL. A vertical tap is a row index, so this
// kernel reads two source rows word for word and never shifts; its per-word
// cost is the sign-magnitude subtraction alone. It is also why this axis needs
// no per-word border fixup: an out-of-image ROW is out of image for every
// column at once, so it is a whole-row fill.
//
// The source's padding bits are not masked on the way in here either, and for a
// different reason from the horizontal kernel's: nothing moves between lanes on
// this axis, so a dirty padding bit can only produce a wrong PADDING bit, which
// the masked store clears.
// ---------------------------------------------------------------------------
template <size_t N>
__global__ void derivativeYKernel(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                                  DerivPod p) {
    const size_t total = p.words * p.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / p.words;
        const size_t i = idx - y * p.words;
        const bool last = (i + 1 == p.words);

        const ptrdiff_t ya =
            (y + 1 < p.height) ? static_cast<ptrdiff_t>(y + 1) : p.bottomSrc;
        const ptrdiff_t yb = (y > 0) ? static_cast<ptrdiff_t>(y - 1) : p.topSrc;

        uint32_t a[N];
        uint32_t b[N];
#pragma unroll
        for (size_t q = 0; q < N; ++q) {
            a[q] = (ya >= 0) ? src.row(q, static_cast<size_t>(ya))[i] : p.fill;
            b[q] = (yb >= 0) ? src.row(q, static_cast<size_t>(yb))[i] : p.fill;
        }

        uint32_t m[N];
        uint32_t s = 0u;
        bincv::impl::signedDifference<N, uint32_t, false>(a, b, m, s);
        storeSigned<N>(dst, y, i, last, p, m, s);
    }
}

// ---------------------------------------------------------------------------
// BOTH axes, one thread, one traversal -- the fused arm
//
// 5N loads feeding 2(N+1) destination words: the same load COUNT as the two
// kernels above combined, and ONE launch instead of two. The saving is the
// launch, not the loads -- there is no intra-thread reuse between the axes,
// because d/dy at row y reads rows y-1 and y+1 and never row y.
// ---------------------------------------------------------------------------
template <size_t N>
__global__ void derivativeXYKernel(DevicePlaneBlockConstView src, DevicePlaneBlockView dxDst,
                                   DevicePlaneBlockView dyDst, DerivPod p) {
    const size_t total = p.words * p.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / p.words;
        const size_t i = idx - y * p.words;
        const bool last = (i + 1 == p.words);

        const ptrdiff_t ya =
            (y + 1 < p.height) ? static_cast<ptrdiff_t>(y + 1) : p.bottomSrc;
        const ptrdiff_t yb = (y > 0) ? static_cast<ptrdiff_t>(y - 1) : p.topSrc;

        uint32_t ax[N];
        uint32_t bx[N];
        uint32_t ay[N];
        uint32_t by[N];
#pragma unroll
        for (size_t q = 0; q < N; ++q) {
            horizontalTaps(src.row(q, y), i, last, p, ax[q], bx[q]);
            ay[q] = (ya >= 0) ? src.row(q, static_cast<size_t>(ya))[i] : p.fill;
            by[q] = (yb >= 0) ? src.row(q, static_cast<size_t>(yb))[i] : p.fill;
        }

        uint32_t m[N];
        uint32_t s = 0u;
        bincv::impl::signedDifference<N, uint32_t, false>(ax, bx, m, s);
        storeSigned<N>(dxDst, y, i, last, p, m, s);
        s = 0u;
        bincv::impl::signedDifference<N, uint32_t, false>(ay, by, m, s);
        storeSigned<N>(dyDst, y, i, last, p, m, s);
    }
}

constexpr unsigned kBlock = 256;

unsigned gridFor(size_t total) {
    const size_t blocks = (total + kBlock - 1) / kBlock;
    // Grid-stride kernels cap the grid; 4096 blocks saturate this class of
    // device and the loop covers the rest (logic.cu's shape and cap).
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

/// Words a plane block spans, for the aliasing check. The whole allocation
/// extent rather than the last row's own words: conservative in the only
/// direction that matters, and it cannot reject two blocks laid end to end.
size_t blockSpanWords(const DevicePlaneBlockConstView& v) {
    return v.planes * v.height * v.stride;
}
size_t blockSpanWords(const DevicePlaneBlockView& v) {
    return v.planes * v.height * v.stride;
}

/// True when the two blocks share a word. NO IN-PLACE FORM EXISTS: neither axis
/// is pointwise in the word index, and on the device an aliased block is a
/// silent cross-block race rather than the host's merely wrong answer.
bool blocksOverlap(const uint32_t* a, size_t aWords, const uint32_t* b, size_t bWords) {
    if (a == nullptr || b == nullptr) return false;
    return a < b + bWords && b < a + aWords;
}

bool shapeIsValid(const DevicePlaneBlockConstView& src, const DevicePlaneBlockView& dst) {
    return src.planes >= 1 && src.planes <= derivativeMaxPlanes() &&
           dst.planes == src.planes + 1 && dst.width == src.width &&
           dst.height == src.height && src.stride >= rowWords(src.width) &&
           dst.stride >= rowWords(dst.width) &&
           !blocksOverlap(src.ptr, blockSpanWords(src), dst.ptr, blockSpanWords(dst));
}

DerivPod makePod(const DevicePlaneBlockConstView& src, BorderType borderType,
                 bool borderValue) {
    DerivPod p{};
    p.width = src.width;
    p.height = src.height;
    p.words = rowWords(src.width);
    p.tailMask = rowTailMask(src.width);
    p.lastLiveBit = uint32_t{1} << static_cast<unsigned>((src.width - 1) % 32u);
    // The Tier 1 border rule, evaluated ONCE per image on the host. A 3-tap
    // kernel has exactly these four out-of-image coordinates, and they depend on
    // the extent and the type, not on the row or the column.
    p.leftSrc = bincv::impl::borderIndex(ptrdiff_t{-1}, src.width, borderType);
    p.rightSrc =
        bincv::impl::borderIndex(static_cast<ptrdiff_t>(src.width), src.width, borderType);
    p.topSrc = bincv::impl::borderIndex(ptrdiff_t{-1}, src.height, borderType);
    p.bottomSrc =
        bincv::impl::borderIndex(static_cast<ptrdiff_t>(src.height), src.height, borderType);
    // A `true` constant border means every magnitude plane reads all-ones, i.e.
    // the maximum representable value -- the host's N-bit reading of the bool.
    p.fill = borderValue ? 0xFFFFFFFFu : 0u;
    p.borderValue = borderValue;
    return p;
}

enum class Axis { X, Y };

template <size_t N>
void launchSingle(Axis axis, DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                  const DerivPod& p, cudaStream_t stream) {
    const unsigned grid = gridFor(p.words * p.height);
    if (axis == Axis::X) {
        derivativeXKernel<N><<<grid, kBlock, 0, stream>>>(src, dst, p);
    } else {
        derivativeYKernel<N><<<grid, kBlock, 0, stream>>>(src, dst, p);
    }
}

/// The compile-time N dispatch. `switch` rather than a table: the instantiation
/// set is the named domain and a reader should be able to count it.
cudaError_t dispatchSingle(Axis axis, DevicePlaneBlockConstView src,
                           DevicePlaneBlockView dst, BorderType borderType,
                           bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(bincv::impl::isKnownBorderType(borderType),
                 "cuda derivative: unknown BorderType");
    BINCV_ASSERT(src.planes >= 1 && src.planes <= derivativeMaxPlanes(),
                 "cuda derivative: N outside the DEVICE domain [1, derivativeMaxPlanes()]");
    BINCV_ASSERT(shapeIsValid(src, dst),
                 "cuda derivative: destination must be N+1 planes of the source's extent, "
                 "strides must cover a row, and no destination plane may share a word with "
                 "a source plane -- there is no in-place form");
    if (!bincv::impl::isKnownBorderType(borderType)) return cudaErrorInvalidValue;
    if (!shapeIsValid(src, dst)) return cudaErrorInvalidValue;
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    if (src.ptr == nullptr || dst.ptr == nullptr) return cudaErrorInvalidValue;

    const DerivPod p = makePod(src, borderType, borderValue);
    switch (src.planes) {
        case 1: launchSingle<1>(axis, src, dst, p, stream); break;
        case 2: launchSingle<2>(axis, src, dst, p, stream); break;
        case 3: launchSingle<3>(axis, src, dst, p, stream); break;
        case 4: launchSingle<4>(axis, src, dst, p, stream); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

template <size_t N>
void launchFused(DevicePlaneBlockConstView src, DevicePlaneBlockView dxDst,
                 DevicePlaneBlockView dyDst, const DerivPod& p, cudaStream_t stream) {
    derivativeXYKernel<N><<<gridFor(p.words * p.height), kBlock, 0, stream>>>(src, dxDst,
                                                                              dyDst, p);
}

} // namespace

namespace impl {

bool& derivativeFusedArmEnabled() {
    static bool enabled = true;
    return enabled;
}

} // namespace impl

cudaError_t derivativeX(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                        BorderType borderType, bool borderValue, cudaStream_t stream) {
    return dispatchSingle(Axis::X, src, dst, borderType, borderValue, stream);
}

cudaError_t derivativeY(DevicePlaneBlockConstView src, DevicePlaneBlockView dst,
                        BorderType borderType, bool borderValue, cudaStream_t stream) {
    return dispatchSingle(Axis::Y, src, dst, borderType, borderValue, stream);
}

cudaError_t derivativeXY(DevicePlaneBlockConstView src, DevicePlaneBlockView dxDst,
                         DevicePlaneBlockView dyDst, BorderType borderType,
                         bool borderValue, cudaStream_t stream) {
    // The fused arm's extra precondition: the two destinations must be distinct
    // from each other as well as from the source.
    BINCV_ASSERT(!blocksOverlap(dxDst.ptr, blockSpanWords(dxDst), dyDst.ptr,
                                blockSpanWords(dyDst)),
                 "cuda derivativeXY: the two destination blocks must not share a word");
    if (blocksOverlap(dxDst.ptr, blockSpanWords(dxDst), dyDst.ptr, blockSpanWords(dyDst)))
        return cudaErrorInvalidValue;

    if (!impl::derivativeFusedArmEnabled()) {
        // The off-switch arm: two launches, byte-identical output. Kept
        // reachable so the suite can hold both arms to one answer in ONE
        // binary and the benchmark can time the arm it claims to.
        const cudaError_t ex = derivativeX(src, dxDst, borderType, borderValue, stream);
        if (ex != cudaSuccess) return ex;
        return derivativeY(src, dyDst, borderType, borderValue, stream);
    }

    BINCV_ASSERT(bincv::impl::isKnownBorderType(borderType),
                 "cuda derivativeXY: unknown BorderType");
    BINCV_ASSERT(src.planes >= 1 && src.planes <= derivativeMaxPlanes(),
                 "cuda derivativeXY: N outside the DEVICE domain [1, derivativeMaxPlanes()]");
    BINCV_ASSERT(shapeIsValid(src, dxDst) && shapeIsValid(src, dyDst),
                 "cuda derivativeXY: each destination must be N+1 planes of the source's "
                 "extent and share no word with the source -- there is no in-place form");
    if (!bincv::impl::isKnownBorderType(borderType)) return cudaErrorInvalidValue;
    if (!shapeIsValid(src, dxDst) || !shapeIsValid(src, dyDst)) return cudaErrorInvalidValue;
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    if (src.ptr == nullptr || dxDst.ptr == nullptr || dyDst.ptr == nullptr)
        return cudaErrorInvalidValue;

    const DerivPod p = makePod(src, borderType, borderValue);
    switch (src.planes) {
        case 1: launchFused<1>(src, dxDst, dyDst, p, stream); break;
        case 2: launchFused<2>(src, dxDst, dyDst, p, stream); break;
        case 3: launchFused<3>(src, dxDst, dyDst, p, stream); break;
        case 4: launchFused<4>(src, dxDst, dyDst, p, stream); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
