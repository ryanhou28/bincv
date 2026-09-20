// The device arm of ops/shift.hpp. One thread per destination word, grid-stride
// over (row, word) -- the same geometry the host row loop walks.
//
// TWO ARMS, held to the same output in one binary:
//
//   FUNNEL (default)    one __funnelshift per destination word. It is defined
//                       at a shift count of zero, so the host's `bitShift == 0`
//                       branch -- which exists purely because `x << 32` is
//                       undefined behaviour in C++ -- has no counterpart.
//   TWO-SHIFT-OR (ref)  the host's expression verbatim, branch included, so a
//                       reader can see what the funnel shift replaced.
//
// The border rule is NOT restated here: bincv::impl::borderIndex carries
// BINCV_HOST_DEVICE and this kernel calls it. That function IS
// cv::borderInterpolate, and it is the Tier 1 promise every morphology
// operation built on this file inherits -- a twin of it would be the one copy
// this project's shared-format rule most exists to prevent.

#include "bincv/cuda/shift.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// Everything the kernel needs that is uniform over the launch, resolved once
/// on the host. A POD kernel argument lives in the constant bank at zero
/// register cost, which is the same reason census.cu passes its offsets that
/// way.
struct ShiftPod {
    ptrdiff_t dx;
    ptrdiff_t dy;
    size_t width;         // pixels per row, src and dst alike
    size_t srcHeight;
    size_t words;         // words per destination row
    size_t srcWords;      // words per source row
    size_t k;             // |dx|
    size_t wordShift;     // k / 32
    unsigned bitShift;    // k % 32
    size_t rimFirst;      // non-constant border: the affected column run...
    size_t rimLast;       // ...as [rimFirst, rimLast)
    uint32_t tailMask;
    uint32_t constantFill;  // the border value as a word
    uint32_t wordFill;      // 0 under a non-constant border; see below
    uint32_t srcTailMask;
    BorderType borderType;
    bool borderValue;
    bool nonConstant;
};

__device__ inline uint32_t horizontalWord(const uint32_t* srcRow, size_t i,
                                          const ShiftPod& p, bool funnel) {
    if (p.k >= p.width) return p.wordFill;

    if (p.dx >= 0) {
        // dst[i] gathers from src[i + wordShift] and the word above it.
        const ptrdiff_t base = static_cast<ptrdiff_t>(i + p.wordShift);
        const uint32_t lo = bincv::impl::extendedRowWord<uint32_t>(srcRow, base, p.srcWords,
                                                        p.srcTailMask, p.wordFill);
        const uint32_t hi = bincv::impl::extendedRowWord<uint32_t>(srcRow, base + 1, p.srcWords,
                                                        p.srcTailMask, p.wordFill);
        if (funnel) return __funnelshift_r(lo, hi, p.bitShift);
        if (p.bitShift == 0u) return lo;
        return (lo >> p.bitShift) | (hi << (32u - p.bitShift));
    }

    // The mirror image: dst[i] gathers from src[i - wordShift] and the one below.
    const ptrdiff_t base = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(p.wordShift);
    const uint32_t hi =
        bincv::impl::extendedRowWord<uint32_t>(srcRow, base, p.srcWords, p.srcTailMask, p.wordFill);
    const uint32_t lo =
        bincv::impl::extendedRowWord<uint32_t>(srcRow, base - 1, p.srcWords, p.srcTailMask, p.wordFill);
    if (funnel) return __funnelshift_l(lo, hi, p.bitShift);
    if (p.bitShift == 0u) return hi;
    return (hi << p.bitShift) | (lo >> (32u - p.bitShift));
}

/// The columns whose source column lies outside the row, for the four
/// NON-CONSTANT border types. Per pixel, and unapologetically so: each type
/// maps a different out-of-range column to a different source column, so there
/// is no word-wide answer. Only the words overlapping the run take this path.
__device__ inline uint32_t fixupRim(uint32_t v, const uint32_t* srcRow, size_t i,
                                    const ShiftPod& p) {
    const size_t c0 = i * 32u;
    const size_t c1 = (c0 + 32u < p.width) ? (c0 + 32u) : p.width;
    const size_t a = (c0 > p.rimFirst) ? c0 : p.rimFirst;
    const size_t b = (c1 < p.rimLast) ? c1 : p.rimLast;
    for (size_t c = a; c < b; ++c) {
        const ptrdiff_t sx =
            bincv::impl::borderIndex(static_cast<ptrdiff_t>(c) + p.dx, p.width, p.borderType);
        // sx < 0 is unreachable here: the constant border never reaches this
        // function. Answered anyway rather than indexed with a negative.
        const bool value =
            (sx < 0) ? p.borderValue
                     : ((srcRow[static_cast<size_t>(sx) / 32u] >>
                         (static_cast<unsigned>(static_cast<size_t>(sx) % 32u))) &
                        1u) != 0u;
        const uint32_t mask = uint32_t{1} << static_cast<unsigned>(c - c0);
        v = value ? (v | mask) : (v & ~mask);
    }
    return v;
}

template <bool Funnel>
__global__ void shiftKernel(DeviceBinMatConstView src, DeviceBinMatView dst, ShiftPod p) {
    const size_t total = p.words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / p.words;
        const size_t i = idx - y * p.words;

        // The vertical half of the shift is a row index and nothing else.
        const ptrdiff_t sy = bincv::impl::borderIndex(static_cast<ptrdiff_t>(y) + p.dy,
                                                      p.srcHeight, p.borderType);

        uint32_t v;
        const uint32_t* srcRow = nullptr;
        if (sy < 0) {
            // Outside the image vertically, which only BORDER_CONSTANT reaches:
            // the whole row is the border value, corners included. That is what
            // cv::copyMakeBorder does too.
            v = p.constantFill;
        } else {
            srcRow = src.row(static_cast<size_t>(sy));
            v = horizontalWord(srcRow, i, p, Funnel);
        }

        // Promise 2: the destination's padding bits are zero however the row was
        // built and whatever the fill was.
        if (i + 1 == p.words) v &= p.tailMask;

        if (p.nonConstant && srcRow != nullptr) v = fixupRim(v, srcRow, i, p);

        dst.row(y)[i] = v;
    }
}

inline unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

bool shapeIsValid(DeviceBinMatConstView src, DeviceBinMatView dst, ptrdiff_t dx,
                  ptrdiff_t dy, BorderType borderType) {
    const size_t adx = (dx < 0) ? (size_t{0} - static_cast<size_t>(dx))
                                : static_cast<size_t>(dx);
    const size_t ady = (dy < 0) ? (size_t{0} - static_cast<size_t>(dy))
                                : static_cast<size_t>(dy);
    return src.width == dst.width && src.height == dst.height &&
           bincv::impl::isKnownBorderType(borderType) &&
           adx <= bincv::impl::maxShiftOffset() && ady <= bincv::impl::maxShiftOffset() &&
           src.stride >= rowWords(src.width) && dst.stride >= rowWords(dst.width);
}

} // namespace

namespace impl {

bool& shiftFunnelEnabled() {
    static bool enabled = true;
    return enabled;
}

} // namespace impl

cudaError_t shift(DeviceBinMatConstView src, DeviceBinMatView dst, ptrdiff_t dx,
                  ptrdiff_t dy, BorderType borderType, bool borderValue,
                  cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda shift: src and dst must have the same dimensions");
    BINCV_ASSERT(bincv::impl::isKnownBorderType(borderType), "cuda shift: unknown BorderType");
    BINCV_ASSERT(shapeIsValid(src, dst, dx, dy, borderType),
                 "cuda shift: the shape, stride and offset domain is violated");
    if (!shapeIsValid(src, dst, dx, dy, borderType)) return cudaErrorInvalidValue;

    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    if (src.ptr == nullptr || dst.ptr == nullptr) return cudaErrorInvalidValue;

    ShiftPod p{};
    p.dx = dx;
    p.dy = dy;
    p.width = dst.width;
    p.srcHeight = src.height;
    p.words = rowWords(dst.width);
    p.srcWords = rowWords(src.width);
    p.k = (dx < 0) ? (size_t{0} - static_cast<size_t>(dx)) : static_cast<size_t>(dx);
    p.wordShift = p.k / 32u;
    p.bitShift = static_cast<unsigned>(p.k % 32u);
    p.tailMask = rowTailMask(dst.width);
    p.srcTailMask = rowTailMask(src.width);
    p.constantFill = borderValue ? 0xFFFFFFFFu : 0u;
    p.borderType = borderType;
    p.borderValue = borderValue;
    p.nonConstant = (borderType != BORDER_CONSTANT);
    // Under a non-constant BorderType the word path's fill never survives: the
    // columns it reaches are exactly the ones the rim fixup rewrites. Zero is
    // used there so that a missing fixup is a visibly wrong image rather than a
    // plausible one -- the host's reasoning, kept.
    p.wordFill = p.nonConstant ? 0u : p.constantFill;

    if (dx > 0) {
        const size_t k = static_cast<size_t>(dx);
        p.rimFirst = (k >= p.width) ? 0u : (p.width - k);
        p.rimLast = p.width;
    } else if (dx < 0) {
        const size_t k = size_t{0} - static_cast<size_t>(dx);
        p.rimFirst = 0u;
        p.rimLast = (k >= p.width) ? p.width : k;
    } else {
        p.rimFirst = 0u;
        p.rimLast = 0u;  // nothing leaves the row
    }

    constexpr unsigned kBlock = 256;
    const unsigned grid = gridFor(p.words * dst.height, kBlock);
    if (impl::shiftFunnelEnabled()) {
        shiftKernel<true><<<grid, kBlock, 0, stream>>>(src, dst, p);
    } else {
        shiftKernel<false><<<grid, kBlock, 0, stream>>>(src, dst, p);
    }
    return cudaGetLastError();
}

cudaError_t shiftLeft(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                      BorderType borderType, bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(k <= bincv::impl::maxShiftOffset(),
                 "cuda shiftLeft: the shift distance must not exceed PTRDIFF_MAX / 2");
    if (k > bincv::impl::maxShiftOffset()) return cudaErrorInvalidValue;
    return shift(src, dst, static_cast<ptrdiff_t>(k), ptrdiff_t{0}, borderType, borderValue,
                 stream);
}

cudaError_t shiftRight(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                       BorderType borderType, bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(k <= bincv::impl::maxShiftOffset(),
                 "cuda shiftRight: the shift distance must not exceed PTRDIFF_MAX / 2");
    if (k > bincv::impl::maxShiftOffset()) return cudaErrorInvalidValue;
    return shift(src, dst, ptrdiff_t{0} - static_cast<ptrdiff_t>(k), ptrdiff_t{0}, borderType,
                 borderValue, stream);
}

cudaError_t shiftUp(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                    BorderType borderType, bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(k <= bincv::impl::maxShiftOffset(),
                 "cuda shiftUp: the shift distance must not exceed PTRDIFF_MAX / 2");
    if (k > bincv::impl::maxShiftOffset()) return cudaErrorInvalidValue;
    return shift(src, dst, ptrdiff_t{0}, static_cast<ptrdiff_t>(k), borderType, borderValue,
                 stream);
}

cudaError_t shiftDown(DeviceBinMatConstView src, DeviceBinMatView dst, size_t k,
                      BorderType borderType, bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(k <= bincv::impl::maxShiftOffset(),
                 "cuda shiftDown: the shift distance must not exceed PTRDIFF_MAX / 2");
    if (k > bincv::impl::maxShiftOffset()) return cudaErrorInvalidValue;
    return shift(src, dst, ptrdiff_t{0}, ptrdiff_t{0} - static_cast<ptrdiff_t>(k), borderType,
                 borderValue, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
