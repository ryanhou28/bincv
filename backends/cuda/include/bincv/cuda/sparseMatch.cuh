#pragma once

/// @file sparseMatch.cuh
/// @brief The window traversal the sparse-stereo and block-matching kernels
/// share, and the packed argmin every search in this family reduces with.
///
/// ---------------------------------------------------------------------------
/// A .cuh, AND WHY
///
/// Everything here is `__device__`, so only a CUDA translation unit can name
/// it. It is a header rather than a static block inside `sparseMatch.cu`
/// because the two riskiest pieces of arithmetic in this family -- the
/// displaced-row reader and the span extraction built on it -- are worth
/// testing AS UNITS against the host, before any kernel calls them, and a test
/// cannot call what lives in one `.cu`'s anonymous namespace. `packCustom.cuh`
/// is the precedent.
///
/// ---------------------------------------------------------------------------
/// THE BIT ARITHMETIC IS NOT RESTATED. ONLY THE TRAVERSAL IS FORKED.
///
/// `impl::ReplicatedShiftedRow<uint32_t>` -- its seven fields, its
/// `sourceWord(k)` masking of the trailing partial word, its `word(i)`
/// assembly and its `lowOutside`/`highOutside` clamps -- is the HOST's, called
/// here rather than copied. `impl::clipRegion`, `impl::edgeFill`,
/// `impl::lowBitsMask` and `impl::floorDiv` likewise. What this file adds is a
/// BUILDER over a device view (the host's `displacedRow` takes a host view, so
/// the two cannot be one function) and a traversal, which is the half the
/// architecture says a backend forks.
///
/// That split is not stylistic. The shift/mask/clamp logic has four edge cases
/// -- window left of the plane, right of it, straddling, and wholly off -- and
/// a silent divergence in any of them looks like a plausible disparity rather
/// than a crash. There is no second copy to diverge.
///
/// ---------------------------------------------------------------------------
/// WHAT `RowSpan` IS FOR: REUSE ACROSS CANDIDATES
///
/// Every search in this family evaluates one window at many shifts of the SAME
/// rows. Rebuilding a `ReplicatedShiftedRow` per candidate re-derives identical
/// bits: two loads, two shifts, an or and two mask-selects, per row, per
/// candidate. A `RowSpan` pays that ONCE for a row and hands every candidate a
/// single `__funnelshift_r`.
///
/// The correctness argument is that the host's `word(i)` is a PURE FUNCTION OF
/// THE SOURCE COLUMN: bit `j` of `word(i)` is the plane's bit at column
/// `i*32 + off + j` when that column is inside `[0, width)`, `leftFill` when it
/// is negative and `rightFill` when it is at or past `width` -- read straight
/// off the `lowOutside`/`highOutside` arithmetic. Three consecutive `word()`
/// calls therefore hold 96 consecutive source columns under that same rule, and
/// any 32-bit run cut out of them by a funnel shift is bit-for-bit what a
/// `ReplicatedShiftedRow` built at that shift would have produced. The suite
/// pins this against the host's own `impl::hammingAt` rather than leaving it as
/// an argument.
///
/// `__funnelshift_r` is also what removes the `s == 0` hazard: it is defined at
/// a shift of zero, where the hand-written `raw >> s | hi << (32 - s)` is not,
/// and a word-aligned window is the COMMON case, not an exotic one.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/blockMatch.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/reduce.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {

/// @brief The host's displaced-row reader, at device word width.
using DeviceShiftedRow = bincv::impl::ReplicatedShiftedRow<uint32_t>;

/// @brief The host's region geometry, at device word width.
using DeviceRegion = bincv::impl::RegionWords<uint32_t>;

/// @brief `impl::displacedRow` over a DEVICE view: the same seven fields, the
/// same vertical clamp, filled from a `DeviceBinMatConstView`.
/// @note NOT a restatement of the bit arithmetic -- the struct's `word()` and
/// `sourceWord()` are the host's own and carry `BINCV_HOST_DEVICE`. Only the
/// FIELD FILLING is here, because the host builder takes a host view and a
/// device view is a distinct type by design (no call can hide where its
/// memory lives).
/// @note `words` and `tailMask` come from this backend's `rowWords` /
/// `rowTailMask`, which the CUDA suite already holds equal to
/// `impl::minRowWords` / `impl::rowTailMask` across widths.
__device__ inline DeviceShiftedRow deviceDisplacedRow(const DeviceBinMatConstView& plane,
                                                      long long y, long long off) {
    DeviceShiftedRow r;
    if (plane.height == 0 || plane.width == 0) return r;
    long long clamped = y;
    if (clamped < 0) clamped = 0;
    const long long lastRow = static_cast<long long>(plane.height) - 1;
    if (clamped > lastRow) clamped = lastRow;
    r.row = plane.row(static_cast<size_t>(clamped));
    r.words = rowWords(plane.width);
    r.width = plane.width;
    r.tailMask = rowTailMask(plane.width);
    r.off = off;
    r.leftFill = bincv::impl::edgeFill<uint32_t>(r.row, 0);
    r.rightFill = bincv::impl::edgeFill<uint32_t>(r.row, plane.width - 1);
    return r;
}

/// @brief 96 consecutive source columns of one row, replicate border applied,
/// starting at column `base` -- so every candidate shift is one instruction.
/// @note Three words and not two: a run of up to 32 bits starting anywhere in
/// `[0, 64)` of the span must be extractable, which needs a 96-bit window.
/// The kernels' own gate keeps `runStart + runWidth <= 64`.
struct RowSpan {
    uint32_t w0 = 0;
    uint32_t w1 = 0;
    uint32_t w2 = 0;

    /// @brief Bits `[p, p + 32)` of the span, aligned to bit 0. `p < 64`.
    /// @note `__funnelshift_r(lo, hi, p)` is bits `[p, p+32)` of `(hi:lo)` and
    /// is DEFINED at `p == 0`, which the host's own `word()` needs an
    /// `if (s != 0)` guard to be.
    __device__ __forceinline__ uint32_t run(unsigned p) const {
        return p < 32u ? __funnelshift_r(w0, w1, p) : __funnelshift_r(w1, w2, p - 32u);
    }
};

/// @brief A row's span, built through the host's own `word()`.
__device__ inline RowSpan loadRowSpan(const DeviceBinMatConstView& plane, long long y,
                                      long long base) {
    const DeviceShiftedRow r = deviceDisplacedRow(plane, y, base);
    RowSpan s;
    s.w0 = r.word(0);
    s.w1 = r.word(1);
    s.w2 = r.word(2);
    return s;
}

/// @brief One window row of a plane, aligned to bit 0 and masked to `width`
/// pixels -- the form a span's `run()` output is XORed against.
/// @note Through `deviceDisplacedRow` rather than a direct word read, so a
/// plane whose padding bits past `width` are DIRTY gives the clean plane's
/// answer: `sourceWord` masks the trailing partial word. The suite writes
/// garbage past `width` and asserts no result moves.
__device__ __forceinline__ uint32_t windowRowWord(const DeviceBinMatConstView& plane,
                                                  long long y, long long x0) {
    return deviceDisplacedRow(plane, y, x0).word(0);
}

/// @brief The sentinel a search that found nothing reduces to.
/// @note All ones in both halves: the cost half is `0xFFFFFFFF`, which no
/// Hamming distance can reach (a descriptor is at most 2048 bits and a
/// window at most 32x32), so "no candidate" is distinguishable from
/// "distance 0xFFFFFFFF" by construction rather than by a flag.
inline constexpr uint64_t kNoCandidate = 0xFFFFFFFFFFFFFFFFull;

/// @brief `(cost << 32) | index`, the packed form every argmin here reduces.
/// @note THE TIE RULE, FOR FREE. Every host loop in this family keeps the
/// FIRST candidate on a tie ("strictly less, scanning ascending"). A plain
/// 64-bit minimum over `(cost << 32) | index` orders by cost first and by
/// smallest index second, which IS that rule -- so an order-dependent
/// serial loop becomes an order-independent parallel reduction with no
/// correctness argument left to make.
/// @note `index` MUST be non-negative and fit 32 bits. Where a candidate is
/// identified by a signed quantity (a disparity, a displacement) the
/// caller packs `index - lowest` and adds `lowest` back after the
/// reduction: a negative value in the low half inverts the ordering, and
/// `stereoRefineDisparity` accepts a negative `minDisparity` (only
/// `stereoDescriptorMatch` asserts it non-negative), so that case is
/// reachable rather than hypothetical.
__device__ __forceinline__ uint64_t packCost(uint32_t cost, uint32_t index) {
    return (static_cast<uint64_t>(cost) << 32) | static_cast<uint64_t>(index);
}

__device__ __forceinline__ uint32_t packedCost(uint64_t p) {
    return static_cast<uint32_t>(p >> 32);
}

__device__ __forceinline__ uint32_t packedIndex(uint64_t p) {
    return static_cast<uint32_t>(p & 0xFFFFFFFFull);
}

/// @brief One butterfly step of a 64-bit shuffle.
/// @note Through `unsigned long long` explicitly. `uint64_t` is `unsigned long`
/// on LP64, and which shuffle overload that picks is not a question worth
/// leaving to overload resolution in a reduction whose failure mode is a
/// plausible wrong answer.
__device__ __forceinline__ uint64_t shflXor64(uint64_t v, int laneMask) {
    return static_cast<uint64_t>(
        __shfl_xor_sync(0xFFFFFFFFu, static_cast<unsigned long long>(v), laneMask, 32));
}

/// @brief Warp-wide minimum of a packed cost. Every lane returns it.
__device__ __forceinline__ uint64_t warpMinPacked(uint64_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const uint64_t other = shflXor64(v, off);
        if (other < v) v = other;
    }
    return v;
}

/// @brief Warp-wide sum of a 32-bit cost. Every lane returns it.
/// @note A BUTTERFLY, on every architecture, and that is a measured choice
/// rather than the obvious one. sm_80 and later have `__reduce_add_sync`,
/// which is ONE instruction where this is five dependent shuffles, and this
/// reduction runs once per search candidate -- at 25 candidates a level it
/// looked like the inner loop. Priced on the reference GPU against the
/// refinement at 40,000 keypoints, where the kernel is well clear of the
/// launch floor: 0.191-0.208 ms for the intrinsic against 0.194-0.210 ms
/// for the butterfly, ranges overlapping in every round. The reduction is
/// not the limiter -- the profile's dominant stall is `wait` on the whole
/// per-candidate chain (row assembly, popcount, reduce, compare), not on
/// the reduce -- so the intrinsic bought nothing and cost an architecture
/// branch. One spelling ships.
__device__ __forceinline__ uint32_t warpSum(uint32_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off, 32);
    return v;
}

// ---------------------------------------------------------------------------
// The two smallest of a multiset, merged pairwise
// ---------------------------------------------------------------------------

/// @brief One lane's running (best, second) for a ratio test.
/// @note THE HOST'S `second` IS THE SECOND ORDER STATISTIC of the distance
/// multiset, not "the best distance that is not the best match". Trace the
/// host loop on {5,5,7}: 5 sets best=5; the next 5 is not `< best` so it
/// lands in `second`; 7 changes nothing. On {7,5,5}: 7 sets best=7,
/// second=inf; 5 sets best=5, second=7; the next 5 is not `< 5`, is `< 7`,
/// so second=5. Both end at (5, 5). A duplicate of the minimum IS the
/// second order statistic, which is order-independent -- and that is what
/// makes a parallel merge exact rather than approximately right.
struct BestTwo {
    uint64_t best = kNoCandidate;   ///< packed (distance, index)
    uint32_t second = 0xFFFFFFFFu;  ///< the second smallest DISTANCE

    /// @brief Folds one more candidate in, in the host's own order.
    __device__ __forceinline__ void offer(uint32_t distance, uint32_t index) {
        const uint64_t p = packCost(distance, index);
        if (p < best) {
            const uint32_t displaced = packedCost(best);
            best = p;
            if (displaced < second) second = displaced;
        } else if (distance < second) {
            second = distance;
        }
    }

    /// @brief Merges another lane's pair: the two smallest of the four.
    __device__ __forceinline__ void merge(const BestTwo& o) {
        const uint64_t lo = best < o.best ? best : o.best;
        const uint64_t hi = best < o.best ? o.best : best;
        uint32_t s = second < o.second ? second : o.second;
        const uint32_t hiCost = packedCost(hi);
        if (hiCost < s) s = hiCost;
        best = lo;
        second = s;
    }
};

/// @brief Warp-wide merge. Every lane returns the warp's pair.
__device__ __forceinline__ BestTwo warpMergeBestTwo(BestTwo v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        BestTwo o;
        o.best = shflXor64(v.best, off);
        o.second = __shfl_xor_sync(0xFFFFFFFFu, v.second, off, 32);
        v.merge(o);
    }
    return v;
}

/// @brief The `DescriptorMatch` the host would have written for this pair.
/// @note THE NO-CANDIDATE WRITE-BACK IS EXPLICIT. With nothing admitted the
/// host leaves the default-constructed record -- `trainIndex == 0` -- while
/// a packed sentinel minimum carries index `0xFFFFFFFF`. Returning the
/// sentinel's low half would put a fabricated index in a record whose
/// `valid` is 0, which a caller who ignores `valid` would read as a match
/// against train descriptor 4,294,967,295.
/// @note `valid` reproduces the host's `count >= 2 && second != sentinel &&
/// best * 100 <= second * maxRatio`. The first two conditions are
/// EQUIVALENT: `second` leaves the sentinel exactly when a second candidate
/// was scored, and no distance can equal `0xFFFFFFFF`. The suite asserts
/// the equivalence rather than the kernel carrying a redundant counter.
__device__ __forceinline__ DeviceDescriptorMatch finishMatch(const BestTwo& v,
                                                             unsigned maxRatio) {
    DeviceDescriptorMatch m;
    const uint32_t bestDist = packedCost(v.best);
    m.trainIndex = (v.best == kNoCandidate) ? 0u : packedIndex(v.best);
    m.distance = bestDist;
    m.secondDistance = v.second;
    const bool ok = v.second != 0xFFFFFFFFu &&
                    static_cast<uint64_t>(bestDist) * 100ull <=
                        static_cast<uint64_t>(v.second) * static_cast<uint64_t>(maxRatio);
    m.valid = ok ? 1u : 0u;
    return m;
}

/// @brief `popcount(a ^ b)` over `words` device words of two descriptors.
/// @note EIGHT `__popc`s for a 256-bit descriptor, not thirty-two. That is the
/// whole of the matching-side advantage and it comes from the descriptor
/// being held as `uint32_t` words: `cv::cuda`'s BFMatcher dispatches
/// `matchHamming_gpu<uchar>` for a `CV_8U` descriptor -- the type
/// `cv::cuda::ORB` emits -- and its `HammingDist::reduceIter` is
/// `__popc(a ^ b)` per BYTE. (OpenCV also instantiates the `int` caller, so
/// a caller who knows to hand it a `CV_32S` header over the same bytes gets
/// eight too; the benchmark runs both and says which is which.)
__device__ __forceinline__ uint32_t hammingDistanceWords(const uint32_t* a, const uint32_t* b,
                                                         uint32_t words) {
    uint32_t d = 0;
    for (uint32_t i = 0; i < words; ++i)
        d += static_cast<uint32_t>(__popc(a[i] ^ b[i]));
    return d;
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
