// The median family's device kernels: the binary three-pixel L over packed
// words, and the wide median over a caller-chosen neighbourhood.
//
// ---------------------------------------------------------------------------
// THE ARITHMETIC CASE, RE-DERIVED AGAINST WHAT sm_86 ACTUALLY HAS
//
// The obvious spelling of a byte-lane median is __vminu4 / __vmaxu4. On this
// architecture those are EMULATED: six SASS instructions each (three LOP3, a
// SHF, an IADD3 and a PRMT), and a min/max pair over the same two operands is
// twelve -- nothing is shared between them. A design that budgets them at one
// instruction is wrong by a factor of six.
//
// What is cheap is the COMPARE: __vcmpgtu4 is four instructions, and it yields
// a 0xFF-per-lane mask that drives BOTH halves of the pair. Each half is then
// a three-input function of (a, b, mask), which is one LOP3.LUT:
//
//     m  = __vcmpgtu4(a, b)          // 4
//     d  = (a ^ b) & m               // 1
//     lo = a ^ d,  hi = b ^ d        // 1 + 1
//
// Seven instructions for a sorted pair of four lanes against twelve. Measured
// on this machine with cuobjdump -sass, the whole four-pixel L body -- three
// word loads, one PRMT to shift a pixel, three sorted pairs, one word store --
// is 33 instructions for four pixels. The same four pixels cost 37 through the
// intrinsics, 47 a byte at a time, and 21 per pixel one-thread-per-pixel.
// So the packed arm survives the __vminu4 finding, but NOT for the reason the
// intrinsics suggested: what it buys is amortising the loads, the addressing
// and the border test over four pixels, and the selection network is merely
// not allowed to give that back.
//
// At 16 bits the same test comes out the other way. __vcmpgtu2 is also four
// instructions, but it covers two lanes rather than four, and the packed
// two-pixel body measures 33 instructions against 27 for the same two pixels
// done with scalar IMNMX out of one 32-bit load. So uint16_t takes the scalar
// lanes, and the gain there is entirely the wider load.
//
// The binary kernel needs none of this. maj3 is ONE LOP3.LUT and
// __funnelshift_r is ONE SHF, so a destination word -- 32 pixels -- is two
// instructions.

#include "bincv/cuda/median.hpp"

#include "bincv/ops/bitslice.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {
namespace {

inline unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

// ---------------------------------------------------------------------------
// denoiseMedian3 -- the reference arm: one thread, one word, 32 pixels
// ---------------------------------------------------------------------------

__global__ void denoiseMedian3Kernel(DeviceBinMatConstView src, DeviceBinMatView dst,
                                     size_t words, uint32_t tailMask) {
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;
        const uint32_t* cur = src.row(y);
        uint32_t c, r;
        if (i + 1 < words) {
            c = cur[i];
            // One SHF: the right-neighbour word is this word shifted one pixel
            // down with the next word's first pixel arriving in the top bit.
            // cur[i + 1]'s bit 0 is pixel (i + 1) * 32, inside `width` for
            // every i here, so it needs no mask.
            r = __funnelshift_r(c, cur[i + 1], 1);
        } else {
            // THE TRAILING WORD, AND THE MASK IS THE BORDER. `cur[words - 1]`
            // may carry the caller's padding bits, and the right neighbour of
            // pixel `width - 1` is the bit the shift would pull in. Zero fill
            // says that neighbour reads 0, so the mask goes on `c` BEFORE the
            // shift -- which also leaves the destination's padding zero for
            // free, because maj3(anything, 0, 0) == 0 and c & r == 0. Masking
            // the store instead passes every interior test and fails on widths
            // that are not multiples of 32.
            c = cur[i] & tailMask;
            r = c >> 1;
        }
        // Row 0's above-neighbour is the zero border for every column, and
        // maj3(0, b, c) is b & c -- the host's HasAbove = false specialisation.
        dst.row(y)[i] = (y == 0) ? (c & r) : ::bincv::maj3<uint32_t>(src.row(y - 1)[i], c, r);
    }
}

// ---------------------------------------------------------------------------
// THERE IS NO SECOND ARM HERE, AND THAT IS A MEASURED RESULT
//
// A uint4 arm -- one thread, four words, 128 pixels, one 128-bit load -- was
// written, proven bit-exact and timed against the single-word arm on a ladder
// from 752x480 to 4096x2160. Its static case was real: 23 SASS instructions per
// 128 pixels against 14 per 32, and four memory instructions per 128 pixels
// against sixteen. It never separated. At every rung the two arms read the same
// median to three decimals, with per-round ratios between 0.97x and 1.04x and
// overlapping sample ranges.
//
// The reason is arithmetic rather than accidental, so no idle-GPU re-run will
// change it. At 4096x2160 this operation's whole working set is
// 2 * 2160 * 128 * 4 = 2.21 MB, which is about 3.6 us of traffic at the
// reference GPU's 608 GB/s -- against a measured empty-kernel launch of
// 7-10 us on this host. THE KERNEL IS CHEAPER THAN THE LAUNCH THAT CARRIES IT
// AT EVERY FRAME SIZE, and it would take roughly an eightfold increase over 4K
// to change that. No kernel shape can move a number that is not the kernel's.
//
// So one implementation ships, with no off-switch -- the censusTransformPacked
// precedent. A second hand-written traversal that must stay bit-exact forever,
// with no measured operating point where it wins, is a maintenance cost bought
// with nothing. The benchmark still supports the claim: it prints the launch
// floor beside this op at every ladder size, and they are the same number.
// ---------------------------------------------------------------------------

/// @brief Do two device views share any word? A bounding-box test, which is
/// stricter than the host's per-row predicate -- see the header's note.
bool boxesOverlap(const DeviceBinMatConstView& a, const DeviceBinMatView& b) {
    if (a.ptr == nullptr || b.ptr == nullptr) return false;
    const uint32_t* aLo = a.ptr;
    const uint32_t* aHi = a.ptr + (a.height == 0 ? 0 : (a.height - 1) * a.stride +
                                                          rowWords(a.width));
    const uint32_t* bLo = b.ptr;
    const uint32_t* bHi = b.ptr + (b.height == 0 ? 0 : (b.height - 1) * b.stride +
                                                          rowWords(b.width));
    return aLo < bHi && bLo < aHi;
}

// ---------------------------------------------------------------------------
// medianWide -- the lane packs
//
// A Pack is the P pixels one thread owns, P = 4 / sizeof(T). Both arms and
// both types share ONE selection network written over it, so the ordering
// logic has a single definition however the lanes are held.
// ---------------------------------------------------------------------------

struct Pack4 {
    uint32_t w;  ///< four uint8 pixels, pixel l in byte l
};

struct Pack2 {
    unsigned v[2];  ///< two uint16 pixels, one per 32-bit register
};

/// @brief Sort one pair of packs in place: `a` becomes the lane-wise minimum,
/// `b` the lane-wise maximum.
/// @note ONE compare mask drives both halves, and each half is a three-input
/// function of (a, b, mask) -- one LOP3 each. Seven instructions for four
/// lanes, against twelve for __vminu4 plus __vmaxu4.
__device__ __forceinline__ void sortPair(Pack4& a, Pack4& b) {
    const uint32_t m = __vcmpgtu4(a.w, b.w);
    const uint32_t d = (a.w ^ b.w) & m;
    const uint32_t lo = a.w ^ d;
    const uint32_t hi = b.w ^ d;
    a.w = lo;
    b.w = hi;
}

/// @brief The 16-bit spelling. Scalar IMNMX per lane, because the packed
/// halfword compare measured worse here: two lanes do not amortise
/// __vcmpgtu2's four instructions the way four lanes amortise __vcmpgtu4's.
__device__ __forceinline__ void sortPair(Pack2& a, Pack2& b) {
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const unsigned x = a.v[l], y = b.v[l];
        a.v[l] = x < y ? x : y;
        b.v[l] = x < y ? y : x;
    }
}

/// @brief The median sample of `K`, lane by lane. **Compile-time indices only.**
/// @note A PARTIAL SELECTION SORT, not the host's insertion sort. The host body
/// is `while (j > 0 && v[j - 1] > key)`, whose index is data-dependent;
/// ported verbatim that becomes a dynamically indexed local array and
/// spills to local memory. Here every index is a constant after unrolling,
/// so `v` lives in registers. After outer pass `i`, `v[i]` holds the
/// (i+1)-th smallest, so `v[K/2]` is the median -- 3 comparators at K = 3,
/// 9 at K = 5, 18 at K = 7, 30 at K = 9.
/// @note The median of a multiset is a unique VALUE, so any correct selection
/// agrees with the host's sort element for element; ties between equal
/// samples are unobservable.
template <size_t K, typename Pack>
__device__ __forceinline__ Pack medianOfPack(Pack (&v)[K]) {
#pragma unroll
    for (size_t i = 0; i <= K / 2; ++i) {
#pragma unroll
        for (size_t j = i + 1; j < K; ++j) sortPair(v[i], v[j]);
    }
    return v[K / 2];
}

template <typename T>
struct PackTraits;

template <>
struct PackTraits<uint8_t> {
    using Pack = Pack4;
    static constexpr int kLanes = 4;

    /// @brief The P consecutive pixels starting at element `sx` of `row`.
    /// @note `row` is 4-byte aligned and `sx >= 0` by the launcher's gate, so
    /// the word at `sx & ~3` is an aligned 32-bit load.
    /// @note THE SECOND WORD IS LOADED ONLY WHEN `sx` IS NOT WORD-ALIGNED, and
    /// that is what keeps it inside the row. The caller only reaches here
    /// for a thread all of whose samples are interior, so byte `sx + 3` is
    /// inside the image; with `r != 0` that forces `sx & ~3 <= width - 5`,
    /// hence `(sx & ~3) + 4 <= width - 1 < stride`, and since both are
    /// multiples of four, `(sx & ~3) + 7 <= stride - 1`. Move the
    /// interiority test off "every sample of every one of the four pixels"
    /// and that chain breaks silently.
    __device__ static Pack load(const uint8_t* row, long long sx) {
        const unsigned r = static_cast<unsigned>(sx) & 3u;
        const uint32_t* w =
            reinterpret_cast<const uint32_t*>(row + (sx - static_cast<long long>(r)));
        const uint32_t a = w[0];
        if (r == 0u) return Pack4{a};
        // One PRMT: byte j of the result is byte j + r of the (b:a) pair.
        return Pack4{__byte_perm(a, w[1], 0x3210u + r * 0x1111u)};
    }

    __device__ static void store(uint8_t* row, size_t x0, Pack p) {
        *reinterpret_cast<uint32_t*>(row + x0) = p.w;
    }
};

template <>
struct PackTraits<uint16_t> {
    using Pack = Pack2;
    static constexpr int kLanes = 2;

    /// @brief Two consecutive uint16 pixels out of one aligned 32-bit load.
    /// The safety chain is the 8-bit one with 2 in place of 4.
    __device__ static Pack load(const uint16_t* row, long long sx) {
        const unsigned r = static_cast<unsigned>(sx) & 1u;
        const uint32_t* w =
            reinterpret_cast<const uint32_t*>(row + (sx - static_cast<long long>(r)));
        const uint32_t a = w[0];
        if (r == 0u) return Pack2{{a & 0xFFFFu, a >> 16}};
        return Pack2{{a >> 16, w[1] & 0xFFFFu}};
    }

    __device__ static void store(uint16_t* row, size_t x0, Pack p) {
        *reinterpret_cast<uint32_t*>(row + x0) = p.v[0] | (p.v[1] << 16);
    }
};

/// @brief One pixel by the scalar rule, border included. The device twin of
/// `impl::med3Scalar`, and the ONE place "out of range reads as zero" is
/// written on this side of the bus -- both arms call it.
template <size_t K, typename T>
__device__ __forceinline__ void medianWidePixel(DeviceImageConstView<T> src,
                                                DeviceImageView<T> dst,
                                                const MedianOffsetsPod& pat, size_t y,
                                                size_t x) {
    unsigned v[K];
#pragma unroll
    for (size_t k = 0; k < K; ++k) {
        const long long sy = static_cast<long long>(y) + pat.dy[k];
        const long long sx = static_cast<long long>(x) + pat.dx[k];
        const bool inside = sy >= 0 && sx >= 0 &&
                            sy < static_cast<long long>(src.height) &&
                            sx < static_cast<long long>(src.width);
        v[k] = inside ? static_cast<unsigned>(src.row(static_cast<size_t>(sy))[sx]) : 0u;
    }
#pragma unroll
    for (size_t i = 0; i <= K / 2; ++i) {
#pragma unroll
        for (size_t j = i + 1; j < K; ++j) {
            const unsigned a = v[i], b = v[j];
            v[i] = a < b ? a : b;
            v[j] = a < b ? b : a;
        }
    }
    dst.row(y)[x] = static_cast<T>(v[K / 2]);
}

/// @brief The reference arm: one thread, one pixel, K bounds-checked gathers.
template <size_t K, typename T>
__global__ void medianWideRefKernel(DeviceImageConstView<T> src, DeviceImageView<T> dst,
                                    MedianOffsetsPod pat) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= src.width || y >= src.height) return;
    medianWidePixel<K, T>(src, dst, pat, y, x);
}

/// @brief The fast arm: one thread, `4 / sizeof(T)` consecutive pixels, one
/// aligned word load per sample offset, and ONE interiority test for the
/// whole run.
/// @param dyLo,dyHi,dxLo,dxHi The pattern's reach, each clamped to include 0,
/// exactly as the host computes its interior.
template <size_t K, typename T>
__global__ void medianWideFastKernel(DeviceImageConstView<T> src,
                                     DeviceImageView<T> dst, MedianOffsetsPod pat,
                                     int dyLo, int dyHi, int dxLo, int dxHi) {
    using Traits = PackTraits<T>;
    using Pack = typename Traits::Pack;
    constexpr int kLanes = Traits::kLanes;

    const size_t x0 = (blockIdx.x * blockDim.x + threadIdx.x) * static_cast<size_t>(kLanes);
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x0 >= src.width || y >= src.height) return;

    const long long yl = static_cast<long long>(y);
    const long long xl = static_cast<long long>(x0);
    const bool interior = yl + dyLo >= 0 && yl + dyHi < static_cast<long long>(src.height) &&
                          xl + dxLo >= 0 &&
                          xl + (kLanes - 1) + dxHi < static_cast<long long>(src.width);
    if (!interior) {
        // The run straddles an edge. Fall back IN THREAD to the per-pixel
        // helper -- the same one the reference arm calls, so the zero-fill rule
        // has one definition. A second launch for the border would cost more
        // than the border it fixes.
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
            const size_t x = x0 + static_cast<size_t>(l);
            if (x < src.width) medianWidePixel<K, T>(src, dst, pat, y, x);
        }
        return;
    }

    Pack v[K];
#pragma unroll
    for (size_t k = 0; k < K; ++k) {
        const size_t sy = static_cast<size_t>(yl + pat.dy[k]);
        v[k] = Traits::load(src.row(sy), xl + pat.dx[k]);
    }
    // Interiority guarantees x0 + kLanes - 1 < width, so the whole store word
    // holds live pixels: no partial write and no padding to clear.
    Traits::store(dst.row(y), x0, medianOfPack<K, Pack>(v));
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

template <typename T>
bool fastArmAdmits(const DeviceImageConstView<T>& src, const DeviceImageView<T>& dst) {
    // THE GATE READS THE VIEW, not a container's constructor argument: a legal
    // sub-window view carries no alignment promise, and a misaligned 32-bit
    // device load is illegal rather than merely slow.
    const uintptr_t s = reinterpret_cast<uintptr_t>(src.ptr);
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst.ptr);
    return (s % 4u) == 0u && (d % 4u) == 0u &&
           ((src.stride * sizeof(T)) % 4u) == 0u &&
           ((dst.stride * sizeof(T)) % 4u) == 0u;
}

template <size_t K, typename T>
cudaError_t launchMedianWide(DeviceImageConstView<T> src, DeviceImageView<T> dst,
                             const MedianOffsetsPod& pat, cudaStream_t stream) {
    int dyLo = 0, dyHi = 0, dxLo = 0, dxHi = 0;
    for (size_t k = 0; k < K; ++k) {
        dyLo = pat.dy[k] < dyLo ? pat.dy[k] : dyLo;
        dyHi = pat.dy[k] > dyHi ? pat.dy[k] : dyHi;
        dxLo = pat.dx[k] < dxLo ? pat.dx[k] : dxLo;
        dxHi = pat.dx[k] > dxHi ? pat.dx[k] : dxHi;
    }
    const dim3 block(32, 8);
    if (medianWideFastArmEnabled() && fastArmAdmits<T>(src, dst)) {
        constexpr int kLanes = PackTraits<T>::kLanes;
        const size_t perBlock = static_cast<size_t>(block.x) * static_cast<size_t>(kLanes);
        const dim3 grid(static_cast<unsigned>((src.width + perBlock - 1) / perBlock),
                        static_cast<unsigned>((src.height + block.y - 1) / block.y));
        medianWideFastKernel<K, T>
            <<<grid, block, 0, stream>>>(src, dst, pat, dyLo, dyHi, dxLo, dxHi);
        return cudaGetLastError();
    }
    const dim3 grid(static_cast<unsigned>((src.width + block.x - 1) / block.x),
                    static_cast<unsigned>((src.height + block.y - 1) / block.y));
    medianWideRefKernel<K, T><<<grid, block, 0, stream>>>(src, dst, pat);
    return cudaGetLastError();
}

template <typename T>
cudaError_t medianWideDispatch(DeviceImageConstView<T> src, DeviceImageView<T> dst,
                               const MedianOffsetsPod& pat, cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda medianWide: src and dst must have the same dimensions");
    if (src.width == 0 || src.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda medianWide: a non-empty view needs a non-null pointer");
    if (src.width != dst.width || src.height != dst.height || src.ptr == nullptr ||
        dst.ptr == nullptr)
        return cudaErrorInvalidValue;

    // THE DOMAIN, ASSERTED AND THEN REFUSED. K is settled at compile time by
    // the header's static_assert; the offsets are runtime data and this is
    // where a pattern outside the documented reach stops.
    for (int k = 0; k < pat.samples; ++k) {
        BINCV_ASSERT(pat.dx[k] >= -127 && pat.dx[k] <= 127 && pat.dy[k] >= -127 &&
                         pat.dy[k] <= 127,
                     "cuda medianWide: sample offsets must lie within +/-127");
        if (pat.dx[k] < -127 || pat.dx[k] > 127 || pat.dy[k] < -127 || pat.dy[k] > 127)
            return cudaErrorInvalidValue;
    }
    // src and dst must not alias -- every output reads neighbours a partial
    // in-place write would already have changed.
    const T* sLo = src.ptr;
    const T* sHi = src.ptr + (src.height - 1) * src.stride + src.width;
    const T* dLo = dst.ptr;
    const T* dHi = dst.ptr + (dst.height - 1) * dst.stride + dst.width;
    BINCV_ASSERT(!(sLo < dHi && dLo < sHi), "cuda medianWide: src and dst must not alias");
    if (sLo < dHi && dLo < sHi) return cudaErrorInvalidValue;

    switch (pat.samples) {
        case 1: return launchMedianWide<1, T>(src, dst, pat, stream);
        case 3: return launchMedianWide<3, T>(src, dst, pat, stream);
        case 5: return launchMedianWide<5, T>(src, dst, pat, stream);
        case 7: return launchMedianWide<7, T>(src, dst, pat, stream);
        case 9: return launchMedianWide<9, T>(src, dst, pat, stream);
        default: break;
    }
    // Unreachable: the header's static_assert settles K before a pod exists.
    return cudaErrorInvalidValue;
}

} // namespace

bool& medianWideFastArmEnabled() {
    static bool on = true;
    return on;
}

cudaError_t medianWideImpl(DeviceImageConstView<uint8_t> src,
                           DeviceImageView<uint8_t> dst, const MedianOffsetsPod& pattern,
                           cudaStream_t stream) {
    return medianWideDispatch<uint8_t>(src, dst, pattern, stream);
}

cudaError_t medianWideImpl(DeviceImageConstView<uint16_t> src,
                           DeviceImageView<uint16_t> dst,
                           const MedianOffsetsPod& pattern, cudaStream_t stream) {
    return medianWideDispatch<uint16_t>(src, dst, pattern, stream);
}

} // namespace impl

cudaError_t denoiseMedian3(DeviceBinMatConstView src, DeviceBinMatView dst,
                           cudaStream_t stream) {
    BINCV_ASSERT(src.width == dst.width && src.height == dst.height,
                 "cuda denoiseMedian3: src and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda denoiseMedian3: a non-empty view needs a non-null pointer");
    if (src.width != dst.width || src.height != dst.height || src.ptr == nullptr ||
        dst.ptr == nullptr)
        return cudaErrorInvalidValue;

    const size_t words = rowWords(dst.width);
    BINCV_ASSERT(src.stride >= words && dst.stride >= words,
                 "cuda denoiseMedian3: every view's stride must cover a whole row");
    if (src.stride < words || dst.stride < words) return cudaErrorInvalidValue;

    BINCV_ASSERT(!impl::boxesOverlap(src, dst),
                 "cuda denoiseMedian3: dst must share no word with src "
                 "(in place is not supported)");
    if (impl::boxesOverlap(src, dst)) return cudaErrorInvalidValue;

    const uint32_t tail = rowTailMask(dst.width);
    constexpr unsigned kBlock = 256;

    impl::denoiseMedian3Kernel<<<impl::gridFor(words * dst.height, kBlock), kBlock, 0,
                                 stream>>>(src, dst, words, tail);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
