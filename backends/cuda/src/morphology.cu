// The device arm of ops/morphology.hpp. One thread owns one destination WORD
// -- 32 pixels -- and accumulates the whole element in a register before a
// single store, which is what keeps the non-constant border fixup free of the
// lost-update race a per-pixel thread mapping would have (two threads
// read-modify-writing different bits of one word).
//
// Three device-only things the host kernel cannot do, all measured from SASS on
// this machine:
//
//   * __funnelshift_r/l is ONE SHF. The host's (cur >> d) | (next << (32 - d))
//     is FOUR instructions here and nvcc does not recognise the idiom, so
//     every shifted read in this file goes through the intrinsic.
//   * __brev is ONE instruction, and a reflection of 32 consecutive columns IS
//     a bit reversal -- which turns the four non-constant borders from the
//     host's per-pixel band into two instructions per virtual word.
//   * A carry-free blend `b ^ ((a ^ b) & m)` is ONE LOP3 when the mask is a
//     kernel parameter. The obvious (a & m) | (b & ~m) costs two, because nvcc
//     folds ~m into a second literal and LOP3 takes one immediate.

#include "bincv/cuda/morphology.hpp"

#include "bincv/cuda/logic.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

constexpr unsigned kBlock = 256;
constexpr int kWordBits = 32;

inline unsigned gridFor(size_t total, unsigned block) {
    const size_t blocks = (total + block - 1) / block;
    return static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096);
}

/// The fold and its identity from one place, exactly as the host's MorphFold:
/// an identity that does not match its fold gives an image right in the
/// interior and wrong at the edge.
template <bool IsErode>
struct Fold {
    __device__ static uint32_t identity() { return IsErode ? 0xFFFFFFFFu : 0u; }
    __device__ static uint32_t apply(uint32_t a, uint32_t b) {
        return IsErode ? (a & b) : (a | b);
    }
};

// ---------------------------------------------------------------------------
// Reading a source row, with everything outside the row's pixels resolved
// ---------------------------------------------------------------------------

/// @brief Word `j` of a row with everything outside the row's PIXELS reading
/// `fill`. This is `bincv::impl::extendedRowWord` ITSELF, not a twin of it:
/// the host original carries BINCV_HOST_DEVICE, so there is one definition of
/// the row-edge blend for both targets.
/// @note THE SECOND HALF IS THE ONE THAT IS EASY TO OMIT: the LAST word's bits
/// past `width` are replaced by the fill as well, not just the whole words
/// past the row. Omitting it looks correct on every frame binCV produced
/// and is wrong the moment the source wraps a buffer whose padding is
/// dirty. ops/shift.hpp records the concrete failure at 5 pixels wide, and
/// test_cuda_morphology uploads a deliberately dirty source for it.
/// @note This was briefly a local copy spelling the blend
/// `fill ^ ((w ^ fill) & tailMask)`, on the belief that it saved a LOP3 over
/// the host's `(w & tailMask) | (fill & ~tailMask)`. In isolation the belief
/// is backwards: probed on sm_86 with a runtime fill, the host spelling is a
/// pure function of three registers and fuses to ONE LOP3.LUT (0xb8) where
/// the XOR form costs THREE (0x3c, 0xc0, 0x3c) -- the literal-mask penalty
/// that motivated the re-spelling applies only when the mask is a
/// compile-time constant, and `tailMask` arrives in the constant bank.
/// @note AND IT DOES NOT REACH THIS KERNEL, which is the point worth keeping.
/// Compiled both ways, this translation unit is instruction-identical --
/// 17,152 instructions, 1,134 LOP3, 859 SHF, 80 BREV either way -- because
/// the two hot call sites (`bitsAt`) pass a literal zero fill, where both
/// spellings collapse to a single AND. The probe measured a real difference
/// at a call shape this kernel does not have. Switched to the host's
/// function for the ONE-DEFINITION reason, not for instructions.
__device__ inline uint32_t extendedWord(const uint32_t* row, long long j, size_t words,
                                        uint32_t tailMask, uint32_t fill) {
    return bincv::impl::extendedRowWord<uint32_t>(row, static_cast<ptrdiff_t>(j), words,
                                                  tailMask, fill);
}

/// @brief The 32 source bits starting at column `s`, ascending, with anything
/// outside the row reading zero. `s` may be negative.
/// @note One funnel shift over two extended words. Callers below use it only
/// for runs that lie wholly inside the row, so the fill never survives --
/// it is there so the WORD reads cannot leave the allocation.
__device__ inline uint32_t bitsAt(const uint32_t* row, long long s, size_t words,
                                  uint32_t tailMask) {
    long long w = s >> 5;             // floor division, correct for negative s
    const unsigned b = static_cast<unsigned>(s - (w << 5));
    const uint32_t lo = extendedWord(row, w, words, tailMask, 0u);
    if (b == 0) return lo;
    const uint32_t hi = extendedWord(row, w + 1, words, tailMask, 0u);
    return __funnelshift_r(lo, hi, b);
}

/// @brief The 32 columns at word index `j` with EVERY column mapped through
/// `impl::borderIndex` -- the word-parallel form of the four non-constant
/// border types.
///
/// @note THIS IS THE ARM THE HOST DOES NOT HAVE. On the host each out-of-range
/// column maps to a different source column, so there is no word-wide
/// answer and it recomputes a band of `reachX` columns per edge one pixel
/// at a time -- the reason its 1.04x collapses to 0.64x under these four
/// types. On the device a reflection of 32 consecutive columns is exactly
/// `__brev` of the ascending run, so the virtual word is two instructions.
/// @note VALID ONLY FOR j in [-1, words]. That is what the caller's gate
/// (`reachX < 32`) guarantees, and it is why the gate exists: a wider
/// reach would need words further out, where a reflection can fold twice.
/// @note `width >= 64` is the other half of the gate, and it is arithmetic
/// rather than caution. The REFLECT_101 virtual word past the right edge
/// reads source columns down to `2*width - 2 - 32*words - 31`, which is
/// at least `width - 64`; below that the run leaves the row and the
/// closed form stops holding.
__device__ inline uint32_t borderRowWord(const uint32_t* row, long long j, size_t words,
                                         size_t width, uint32_t tailMask, int borderType) {
    const long long W = static_cast<long long>(width);

    if (j >= 0 && static_cast<size_t>(j) + 1 < words) return row[static_cast<size_t>(j)];

    // The run of source columns this word's bits map to, and whether it
    // ascends (replicate is neither -- it is one column repeated).
    if (j < 0) {
        switch (borderType) {
            case BORDER_REPLICATE:
                return 0u - (row[0] & 1u);
            case BORDER_REFLECT:
                // column -32 + b  ->  31 - b : the reversal of columns 0..31.
                return __brev(bitsAt(row, 0, words, tailMask));
            case BORDER_REFLECT_101:
                // column -32 + b  ->  32 - b : the reversal of columns 1..32.
                return __brev(bitsAt(row, 1, words, tailMask));
            case BORDER_WRAP:
            default:
                // column -32 + b  ->  W - 32 + b : the row's own last 32.
                return bitsAt(row, W - 32, words, tailMask);
        }
    }

    // j == words - 1 (bits past `width` need mapping) or j == words (all of it).
    const long long c0 = j * 32;  // the column this word's bit 0 names
    uint32_t outside;
    switch (borderType) {
        case BORDER_REPLICATE: {
            const uint32_t lastWord = row[words - 1];
            outside = 0u - ((lastWord >> static_cast<unsigned>((W - 1) & 31)) & 1u);
            break;
        }
        case BORDER_REFLECT:
            // column c0 + b -> 2W - 1 - c0 - b : descending from 2W-1-c0.
            outside = __brev(bitsAt(row, 2 * W - 1 - c0 - 31, words, tailMask));
            break;
        case BORDER_REFLECT_101:
            outside = __brev(bitsAt(row, 2 * W - 2 - c0 - 31, words, tailMask));
            break;
        case BORDER_WRAP:
        default:
            // column c0 + b -> c0 - W + b : ascending from c0 - W.
            outside = bitsAt(row, c0 - W, words, tailMask);
            break;
    }
    if (static_cast<size_t>(j) >= words) return outside;
    // The partial last word: real pixels below `tailMask`, mapped columns above.
    return outside ^ ((row[words - 1] ^ outside) & tailMask);
}

// ---------------------------------------------------------------------------
// The fold over one element row
// ---------------------------------------------------------------------------

/// @brief The general shifted word: the device twin of impl::morphShiftedWord,
/// with the wordShift / bitShift SPLIT kept.
/// @note The split is not ceremony. The three-word spelling `(cur >> d) |
/// (next << (32 - d))` shifts a uint32 by 32 when |d| == 32, which is
/// undefined behaviour and not merely wrong -- and |d| == 32 is reachable:
/// a 65-wide element centred has reachX 32. __funnelshift wraps its count
/// at 31, so it would silently return the unshifted word there.
template <typename Reader>
__device__ inline uint32_t shiftedWordGeneral(const Reader& read, size_t i, size_t width,
                                              long long dx, uint32_t fill) {
    const long long k = dx < 0 ? -dx : dx;
    if (static_cast<size_t>(k) >= width) return fill;
    const long long wordShift = k >> 5;
    const unsigned bitShift = static_cast<unsigned>(k & 31);
    const long long li = static_cast<long long>(i);

    if (dx >= 0) {
        const uint32_t lo = read(li + wordShift);
        if (bitShift == 0) return lo;
        return __funnelshift_r(lo, read(li + wordShift + 1), bitShift);
    }
    const long long base = li - wordShift;
    const uint32_t hi = read(base);
    if (bitShift == 0) return hi;
    return __funnelshift_l(read(base - 1), hi, bitShift);
}

/// The element as the kernel sees it, with the two fills resolved.
struct MorphArgs {
    DeviceStructuringElement se;
    int borderType;
    uint32_t constantFill;
    uint32_t horizontalFill;
    size_t words;
    uint32_t tailMask;
    size_t bandLeft;        ///< columns [0, bandLeft) can reach past the left edge
    size_t bandRightStart;  ///< columns [bandRightStart, width) past the right
};

/// @brief One destination pixel recomputed from the whole element with every
/// source coordinate mapped through borderIndex -- the device twin of
/// impl::morphFixupPixel, returning the bit instead of writing it.
/// @note Recomputes rather than repairs: more than one element cell can reach
/// past the same edge and their contributions cannot be unpicked.
template <bool IsErode>
__device__ inline bool fixupPixel(DeviceBinMatConstView src, size_t y, size_t c,
                                  const MorphArgs& a) {
    uint32_t acc = Fold<IsErode>::identity();
    for (int ey = 0; ey < a.se.rows; ++ey) {
        const int first = a.se.first[ey];
        const int last = a.se.last[ey];
        if (first >= last) continue;
        const ptrdiff_t sy = bincv::impl::borderIndex(
            static_cast<ptrdiff_t>(y) + (ey - a.se.anchorY), src.height,
            static_cast<BorderType>(a.borderType));
        if (sy < 0) continue;  // unreachable: a constant border never gets here
        const uint32_t* srcRow = src.row(static_cast<size_t>(sy));
        for (int ex = first; ex < last; ++ex) {
            if (a.se.masked && ((a.se.cellBits[ey] >> ex) & 1u) == 0u) continue;
            const ptrdiff_t sx =
                bincv::impl::borderIndex(static_cast<ptrdiff_t>(c) + (ex - a.se.anchorX), src.width,
                                  static_cast<BorderType>(a.borderType));
            if (sx < 0) continue;
            const size_t sxu = static_cast<size_t>(sx);
            const uint32_t bit = (srcRow[sxu >> 5] >> static_cast<unsigned>(sxu & 31)) & 1u;
            acc = Fold<IsErode>::apply(acc, 0u - bit);
        }
    }
    return acc != 0u;
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// @tparam Use3x3 The centre-anchored 3x3 specialization: the cells become
/// three predictable branches on values hoisted to the launch and the
/// shift counts become the constant 1. The host measured exactly this
/// substitution at 2.1x-3.7x against its general path, and what it
/// removes -- a runtime trip count and a data-dependent shift count per
/// cell -- costs a warp more than it costs a core.
/// @tparam WordBorder The word-parallel virtual border words instead of the
/// per-pixel banded fixup.
/// @note Both are TEMPLATE parameters rather than arguments, for the reason the
/// host header records about MorphPath: as a runtime argument the switch
/// became a live branch in the SHIPPED path the moment a second call site
/// passed the other value, and cost 13%. A benchmark that changes the code
/// it measures is not measuring it.
template <bool IsErode, bool Use3x3, bool WordBorder>
__global__ void morphKernel(DeviceBinMatConstView src, DeviceBinMatView dst, MorphArgs a) {
    const size_t words = a.words;
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;

        uint32_t acc = Fold<IsErode>::identity();
        const int rows = Use3x3 ? 3 : a.se.rows;
        for (int ey = 0; ey < rows; ++ey) {
            const int first = a.se.first[ey];
            const int last = a.se.last[ey];
            if (first >= last) continue;

            const ptrdiff_t sy = bincv::impl::borderIndex(
                static_cast<ptrdiff_t>(y) + (ey - a.se.anchorY), src.height,
                static_cast<BorderType>(a.borderType));
            if (sy < 0) {
                // Wholly outside the image under BORDER_CONSTANT: every set
                // cell of this element row contributes the constant, and the
                // spans are TIGHT, so `first < last` already means one is set.
                if (a.constantFill != Fold<IsErode>::identity())
                    acc = Fold<IsErode>::apply(acc, a.constantFill);
                continue;
            }
            const uint32_t* srcRow = src.row(static_cast<size_t>(sy));

            // One reader for the whole element row. Under a constant border it
            // is the extended-word rule; under the word-parallel border arm it
            // is the virtual-word rule, and neither ever leaves the row.
            const auto read = [&](long long j) -> uint32_t {
                uint32_t w;
                if constexpr (WordBorder)
                    w = borderRowWord(srcRow, j, words, src.width, a.tailMask, a.borderType);
                else
                    w = extendedWord(srcRow, j, words, a.tailMask, a.horizontalFill);
                return w;
            };

            if (Use3x3) {
                const uint32_t cur = read(static_cast<long long>(i));
                const int cellRow = (a.se.cells3x3 >> (ey * 3)) & 7;
                // dx = -1: destination column c reads source column c - 1.
                if (cellRow & 1) {
                    const uint32_t prev = read(static_cast<long long>(i) - 1);
                    acc = Fold<IsErode>::apply(acc, __funnelshift_l(prev, cur, 1));
                }
                if (cellRow & 2) acc = Fold<IsErode>::apply(acc, cur);
                // dx = +1: destination column c reads source column c + 1.
                if (cellRow & 4) {
                    const uint32_t next = read(static_cast<long long>(i) + 1);
                    acc = Fold<IsErode>::apply(acc, __funnelshift_r(cur, next, 1));
                }
                continue;
            }

            // THE WINDOW, the host's own: every cell of this element row reads
            // the same source row at a different horizontal offset, and when
            // the row's widest offset is under a word all of them are shifts of
            // the same three words. Three reads per element row instead of two
            // per CELL -- for a 5x5 ellipse row, 3 instead of 10.
            const int reachLeft = a.se.anchorX - first;
            const int reachRight = (last - 1) - a.se.anchorX;
            const int rowReach = max(max(reachLeft, reachRight), 0);

            if (rowReach < kWordBits) {
                const uint32_t prev = read(static_cast<long long>(i) - 1);
                const uint32_t cur = read(static_cast<long long>(i));
                const uint32_t next = read(static_cast<long long>(i) + 1);
                for (int ex = first; ex < last; ++ex) {
                    if (a.se.masked && ((a.se.cellBits[ey] >> ex) & 1u) == 0u) continue;
                    const int d = ex - a.se.anchorX;
                    uint32_t w;
                    if (d == 0) {
                        w = cur;
                    } else if (d > 0) {
                        w = __funnelshift_r(cur, next, static_cast<unsigned>(d));
                    } else {
                        w = __funnelshift_l(prev, cur, static_cast<unsigned>(-d));
                    }
                    acc = Fold<IsErode>::apply(acc, w);
                }
                continue;
            }

            for (int ex = first; ex < last; ++ex) {
                if (a.se.masked && ((a.se.cellBits[ey] >> ex) & 1u) == 0u) continue;
                acc = Fold<IsErode>::apply(
                    acc, shiftedWordGeneral(read, i, src.width, ex - a.se.anchorX,
                                            a.horizontalFill));
            }
        }

        // CLAUDE.md's hard rule: whatever the fold's identity was, the bits past
        // `width` are zero on return.
        if (i == words - 1) acc &= a.tailMask;

        // THE BAND FIXUP, WORD-OWNED AND IN-REGISTER. Only the thread that owns
        // the word touches it, so there is no read-modify-write race between
        // threads -- which a per-pixel thread mapping would have, on different
        // bits of the same word. Taken by the 1-2 edge threads of each row.
        if (!WordBorder && a.borderType != BORDER_CONSTANT &&
            (i * 32 < a.bandLeft || (i + 1) * 32 > a.bandRightStart)) {
            const size_t c0 = i * 32;
            for (unsigned b = 0; b < 32u; ++b) {
                const size_t c = c0 + b;
                if (c >= src.width) break;
                if (c >= a.bandLeft && c < a.bandRightStart) continue;
                const bool bit = fixupPixel<IsErode>(src, y, c, a);
                acc = (acc & ~(1u << b)) | (static_cast<uint32_t>(bit) << b);
            }
        }

        dst.row(y)[i] = acc;
    }
}

__global__ void andNotKernel(DeviceBinMatConstView a, DeviceBinMatConstView b,
                             DeviceBinMatView dst, size_t words, uint32_t tailMask) {
    const size_t total = words * dst.height;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        const size_t y = idx / words;
        const size_t i = idx - y * words;
        uint32_t v = a.row(y)[i] & ~b.row(y)[i];
        if (i == words - 1) v &= tailMask;
        dst.row(y)[i] = v;
    }
}

// ---------------------------------------------------------------------------
// The launcher
// ---------------------------------------------------------------------------

bool argumentsAreSane(DeviceBinMatConstView src, DeviceBinMatView dst) {
    return src.width == dst.width && src.height == dst.height &&
           src.stride >= rowWords(src.width) && dst.stride >= rowWords(dst.width);
}

template <bool IsErode>
cudaError_t launchMorph(DeviceBinMatConstView src, DeviceBinMatView dst,
                        const DeviceStructuringElement& se, BorderType borderType,
                        bool borderValue, cudaStream_t stream) {
    BINCV_ASSERT(argumentsAreSane(src, dst),
                 "cuda morphology: src and dst must have the same dimensions and a "
                 "stride that covers a row");
    BINCV_ASSERT(se.valid != 0,
                 "cuda morphology: the structuring element is outside the device arm's "
                 "accepted domain (see morphology.hpp)");
    BINCV_ASSERT(bincv::impl::isKnownBorderType(borderType),
                 "cuda morphology: unknown BorderType");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    if (se.valid == 0 || !argumentsAreSane(src, dst) ||
        !bincv::impl::isKnownBorderType(borderType))
        return cudaErrorInvalidValue;
    BINCV_ASSERT(src.ptr != nullptr && dst.ptr != nullptr,
                 "cuda morphology: a non-empty view needs a non-null pointer");

    MorphArgs a{};
    a.se = se;
    a.borderType = static_cast<int>(borderType);
    const uint32_t identity = IsErode ? 0xFFFFFFFFu : 0u;
    a.constantFill = borderValue ? 0xFFFFFFFFu : 0u;
    // Under a non-constant BorderType the word path's horizontal fill never
    // survives: the columns it reaches are exactly the ones the other arm
    // rewrites. The fold's identity is used there so a missing fixup shows up
    // as a plainly wrong edge rather than a plausible one.
    a.horizontalFill = (borderType == BORDER_CONSTANT) ? a.constantFill : identity;
    a.words = rowWords(dst.width);
    a.tailMask = rowTailMask(dst.width);
    const size_t reach = static_cast<size_t>(se.reachX);
    a.bandLeft = (reach < dst.width) ? reach : dst.width;
    a.bandRightStart = (dst.width > reach) ? (dst.width - reach) : 0;

    const bool use3x3 = impl::morphFastArmEnabled() && se.rows == 3 && se.cols == 3 &&
                        se.anchorX == 1 && se.anchorY == 1;
    const bool wordBorder = impl::morphWordBorderEnabled() && borderType != BORDER_CONSTANT &&
                            se.reachX < kWordBits && dst.width >= 64;

    const unsigned grid = gridFor(a.words * dst.height, kBlock);
    if (use3x3) {
        if (wordBorder)
            morphKernel<IsErode, true, true><<<grid, kBlock, 0, stream>>>(src, dst, a);
        else
            morphKernel<IsErode, true, false><<<grid, kBlock, 0, stream>>>(src, dst, a);
    } else {
        if (wordBorder)
            morphKernel<IsErode, false, true><<<grid, kBlock, 0, stream>>>(src, dst, a);
        else
            morphKernel<IsErode, false, false><<<grid, kBlock, 0, stream>>>(src, dst, a);
    }
    return cudaGetLastError();
}

} // namespace

// ---------------------------------------------------------------------------
// The host-side element conversion
// ---------------------------------------------------------------------------

bool deviceElementDomainOk(const StructuringElement& se) {
    if (!se.valid()) return false;
    if (se.rows < 1 || se.rows > kMaxDeviceElementRows) return false;
    if (se.cols < 1) return false;
    if (se.mask != nullptr) return se.cols <= kMaxDeviceMaskCols;
    return se.cols <= kMaxDeviceElementCols;
}

DeviceStructuringElement toDeviceElement(const StructuringElement& se) {
    DeviceStructuringElement d{};
    if (!deviceElementDomainOk(se)) return d;  // valid stays 0; launchers refuse it

    d.rows = se.rows;
    d.cols = se.cols;
    d.anchorX = se.anchorCol();
    d.anchorY = se.anchorRow();
    // reachX from the HOST's own helper, not recomputed here: the band
    // arithmetic on the two sides must not be able to drift.
    d.reachX = static_cast<int>(bincv::impl::morphMaxOffsetX(se));
    d.masked = (se.mask != nullptr) ? 1 : 0;

    for (int ey = 0; ey < se.rows; ++ey) {
        int first = 0;
        int last = 0;
        se.spanOfRow(ey, first, last);
        if (d.masked) {
            // TIGHT spans for a mask. spanOfRow hands back [0, cols) there, and
            // the difference is observable: under BORDER_CONSTANT with a
            // non-default borderValue an element row with NO set cell must
            // contribute nothing, and a loose span would fold the constant in
            // at every edge. The host decides that by scanning activeAt; this
            // resolves it once, on the host, into the span itself.
            uint32_t bits = 0;
            int lo = se.cols;
            int hi = -1;
            for (int ex = 0; ex < se.cols; ++ex) {
                if (!se.activeAt(ex, ey)) continue;
                bits |= (uint32_t{1} << ex);
                if (ex < lo) lo = ex;
                hi = ex;
            }
            d.cellBits[ey] = bits;
            d.first[ey] = static_cast<int16_t>(hi >= 0 ? lo : 0);
            d.last[ey] = static_cast<int16_t>(hi >= 0 ? hi + 1 : 0);
        } else {
            d.first[ey] = static_cast<int16_t>(first);
            d.last[ey] = static_cast<int16_t>(last);
        }
    }

    // The nine cells, once per call rather than per row: for MORPH_ELLIPSE
    // activeAt evaluates a square root.
    if (se.rows == 3 && se.cols == 3) {
        for (int ey = 0; ey < 3; ++ey)
            for (int ex = 0; ex < 3; ++ex)
                if (se.activeAt(ex, ey)) d.cells3x3 |= (1 << (ey * 3 + ex));
    }

    d.valid = 1;
    return d;
}

cudaError_t erode(DeviceBinMatConstView src, DeviceBinMatView dst,
                  const DeviceStructuringElement& element, BorderType borderType,
                  bool borderValue, cudaStream_t stream) {
    return launchMorph<true>(src, dst, element, borderType, borderValue, stream);
}

cudaError_t dilate(DeviceBinMatConstView src, DeviceBinMatView dst,
                   const DeviceStructuringElement& element, BorderType borderType,
                   bool borderValue, cudaStream_t stream) {
    return launchMorph<false>(src, dst, element, borderType, borderValue, stream);
}

namespace impl {

bool& morphFastArmEnabled() {
    static bool on = true;
    return on;
}

bool& morphWordBorderEnabled() {
    static bool on = true;
    return on;
}

bool& morphAndNotFusedEnabled() {
    static bool on = true;
    return on;
}

cudaError_t andNot(DeviceBinMatConstView a, DeviceBinMatConstView b, DeviceBinMatView dst,
                   cudaStream_t stream) {
    BINCV_ASSERT(a.width == dst.width && a.height == dst.height && b.width == dst.width &&
                     b.height == dst.height,
                 "cuda andNot: a, b and dst must have the same dimensions");
    if (dst.width == 0 || dst.height == 0) return cudaSuccess;
    BINCV_ASSERT(a.ptr != nullptr && b.ptr != nullptr && dst.ptr != nullptr,
                 "cuda andNot: a non-empty view needs a non-null pointer");
    const size_t words = rowWords(dst.width);
    andNotKernel<<<gridFor(words * dst.height, kBlock), kBlock, 0, stream>>>(
        a, b, dst, words, rowTailMask(dst.width));
    return cudaGetLastError();
}

} // namespace impl

namespace {

/// `dst = a & ~b`, either fused into one launch or spelled the way the host
/// composes it.
/// @param notTarget Where the two-launch arm puts `~b`. It must be a view the
/// caller is free to clobber -- either `b` itself (the host's in-place
/// case) or the dead scratch frame. It exists because `b` can be `src`,
/// which is read-only, and writing `~src` through `dst` would destroy the
/// operand the AND is about to read. The fused arm ignores it, which is
/// exactly why the arms differ in what they leave behind and not in what
/// they produce.
cudaError_t subtractInto(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatView dst, DeviceBinMatView notTarget,
                         cudaStream_t stream) {
    if (impl::morphAndNotFusedEnabled()) return impl::andNot(a, b, dst, stream);
    cudaError_t err = bitwiseNot(b, notTarget, stream);
    if (err != cudaSuccess) return err;
    return bitwiseAnd(a, DeviceBinMatConstView(notTarget), dst, stream);
}

} // namespace

cudaError_t morphologyEx(DeviceBinMatConstView src, DeviceBinMatView dst, MorphOp op,
                         const DeviceStructuringElement& element, DeviceBinMatView scratch,
                         BorderType borderType, cudaStream_t stream) {
    BINCV_ASSERT(op == MORPH_ERODE || op == MORPH_DILATE || op == MORPH_OPEN ||
                     op == MORPH_CLOSE || op == MORPH_GRADIENT || op == MORPH_TOPHAT ||
                     op == MORPH_BLACKHAT,
                 "cuda morphologyEx: unknown MorphOp");
    BINCV_ASSERT(!bincv::morphologyExNeedsScratch(op) ||
                     (scratch.width == src.width && scratch.height == src.height),
                 "cuda morphologyEx: this op needs a caller-provided scratch view of "
                 "src's size");

    const DeviceBinMatConstView cScratch(scratch);
    const DeviceBinMatConstView cDst(dst);
    cudaError_t err = cudaSuccess;

    switch (op) {
        case MORPH_ERODE:
            return erode(src, dst, element, borderType, true, stream);

        case MORPH_DILATE:
            return dilate(src, dst, element, borderType, false, stream);

        case MORPH_OPEN:
            err = erode(src, scratch, element, borderType, true, stream);
            if (err != cudaSuccess) return err;
            return dilate(cScratch, dst, element, borderType, false, stream);

        case MORPH_CLOSE:
            err = dilate(src, scratch, element, borderType, false, stream);
            if (err != cudaSuccess) return err;
            return erode(cScratch, dst, element, borderType, true, stream);

        case MORPH_GRADIENT:
            // dilate - erode. On content in {0,255} a saturating cv::subtract
            // is exactly `a & ~b`, whatever the element -- it does not rest on
            // opening being anti-extensive, which is false for an element whose
            // anchor cell is not set.
            err = dilate(src, dst, element, borderType, false, stream);
            if (err != cudaSuccess) return err;
            err = erode(src, scratch, element, borderType, true, stream);
            if (err != cudaSuccess) return err;
            return subtractInto(cDst, cScratch, dst, scratch, stream);

        case MORPH_TOPHAT:
            // src - open(src). The opening lands in dst, then dst = src & ~dst.
            err = erode(src, scratch, element, borderType, true, stream);
            if (err != cudaSuccess) return err;
            err = dilate(cScratch, dst, element, borderType, false, stream);
            if (err != cudaSuccess) return err;
            return subtractInto(src, cDst, dst, dst, stream);

        case MORPH_BLACKHAT:
        default:
            // close(src) - src. The closing lands in dst; scratch is dead by
            // then, so the subtraction needs no second frame.
            err = dilate(src, scratch, element, borderType, false, stream);
            if (err != cudaSuccess) return err;
            err = erode(cScratch, dst, element, borderType, true, stream);
            if (err != cudaSuccess) return err;
            return subtractInto(cDst, src, dst, scratch, stream);
    }
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
