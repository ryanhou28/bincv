#pragma once

/// @file reduce.hpp
/// @brief Bulk population counts over device bit matrices: plain counts, masked
/// counts, and the gradient covariance. Reductions stay BULK-ONLY on this
/// backend too -- `__popc` is one instruction here, but a per-word entry point
/// invites per-word round trips over the bus, which is this backend's version
/// of the register-domain crossing the host rule exists to prevent.
///
/// ---------------------------------------------------------------------------
/// THE COVARIANCE IS THIS BACKEND'S BEST CASE
///
/// ARCHITECTURE §1's identity turns the 2x2 gradient covariance into population
/// counts over masks, with no multiplies. On aarch64 that costs two
/// register-domain crossings per word, which is why host reductions are bulk
/// only; on a GPU `__popc` is a single instruction on a value already in a
/// register. The decision that costs binCV on Cortex-M pays here.
///
/// ---------------------------------------------------------------------------
/// TWO SHAPES, AND THE BATCHED ONE IS THE POINT
///
/// A kernel launch is ~5-10 us. A 31x31 window covariance is ~60 word triples
/// -- nanoseconds of work. **One launch per window is latency, not compute**,
/// so the entry point a tracker uses takes the WHOLE keypoint set and issues
/// one launch: `countCovarianceBatchAsync`, one block per window. The
/// single-region forms remain for a caller with one region, and use a
/// grid-stride traversal instead because their region may be a whole frame.
///
/// Both shapes produce identical counts -- integer addition, so combination
/// order cannot change the answer -- and the tests hold the batch to the
/// single-region form as well as to the host.
///
/// ---------------------------------------------------------------------------
/// WHAT IS DELIBERATELY NOT PORTED
///
/// The host's `SlidingWindowCount` exists because consecutive windows in a
/// column re-read almost the same words SERIALLY, and amortizing that is worth
/// 7.3x-20x there. On the device every window is its own block: there is no
/// serial re-read to amortize, the sharing the sliding form hand-builds is what
/// the cache does anyway, and a sliding traversal would serialize what is
/// currently parallel. The batched form is the device answer to the same need,
/// which is the "forked kernels" half of ARCHITECTURE §8.5 doing its job.
///
/// The async forms write into caller-provided DEVICE memory and return without
/// synchronizing, so a resident pipeline consumes the counts on-device. The
/// sync forms are conveniences for tests and callers off the hot path; each
/// costs a stream synchronize and says so.

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/reduce.hpp"
#include "core.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief `SplitCount`'s device-side counterpart: the two halves as the 64-bit
/// unsigned type `atomicAdd` takes.
/// @note A distinct POD rather than the host `SplitCount` because this one is
/// written by device code and must be trivially copyable with no member
/// functions; `toHost` below converts at the boundary, and the signed
/// `crossTerm` subtraction keeps its single implementation there.
struct DeviceSplitCount {
    unsigned long long whenClear;
    unsigned long long whenSet;
};

/// @brief `CovarianceCount`'s device-side counterpart.
struct DeviceCovarianceCount {
    unsigned long long xx;
    unsigned long long yy;
    unsigned long long whenClear;
    unsigned long long whenSet;
};

/// @brief The host spelling of a device count, so `crossTerm()` is reached
/// through `SplitCount`'s one implementation of the signed subtraction.
inline SplitCount toHost(const DeviceSplitCount& d) {
    SplitCount out;
    out.whenClear = static_cast<size_t>(d.whenClear);
    out.whenSet = static_cast<size_t>(d.whenSet);
    return out;
}

inline CovarianceCount toHost(const DeviceCovarianceCount& d) {
    CovarianceCount out;
    out.xx = static_cast<size_t>(d.xx);
    out.yy = static_cast<size_t>(d.yy);
    out.xy.whenClear = static_cast<size_t>(d.whenClear);
    out.xy.whenSet = static_cast<size_t>(d.whenSet);
    return out;
}

// ---------------------------------------------------------------------------
// countNonZero -- plain population count
// ---------------------------------------------------------------------------

/// @brief Sets *dResult (device memory) to the number of set pixels in `src`.
/// Device twin of the host Tier 1 countNonZero, equal to it by test.
cudaError_t countNonZeroAsync(DeviceBinMatConstView src, unsigned long long* dResult,
                              cudaStream_t stream = nullptr);

/// @brief Sets *dResult to the set pixels of `src` inside `region`, clipped to
/// the view exactly as the host clips it (negative origins legal).
cudaError_t countNonZeroAsync(DeviceBinMatConstView src, Rect region,
                              unsigned long long* dResult,
                              cudaStream_t stream = nullptr);

/// @brief Synchronous conveniences: allocate an 8-byte device scalar, count,
/// download. One stream synchronize each -- benchmark the async forms.
size_t countNonZero(DeviceBinMatConstView src);
size_t countNonZero(DeviceBinMatConstView src, Rect region);

// ---------------------------------------------------------------------------
// countAnd -- the masked count, no intermediate image
// ---------------------------------------------------------------------------

/// @brief Sets *dResult to `popcount(a & b)` over `region`. Device twin of the
/// host countAnd; the AND happens a word at a time inside the reduction, so
/// no scratch image exists here either.
cudaError_t countAndAsync(DeviceBinMatConstView a, DeviceBinMatConstView b, Rect region,
                          unsigned long long* dResult, cudaStream_t stream = nullptr);

/// @brief Synchronous convenience; see countNonZero's note.
size_t countAnd(DeviceBinMatConstView a, DeviceBinMatConstView b, Rect region);

// ---------------------------------------------------------------------------
// countAndSplit -- `a & b` split by a selector, one pass
// ---------------------------------------------------------------------------

/// @brief Sets *dResult to `{whenClear, whenSet}` -- the pixels of `a & b`
/// split by selector plane `c` -- over `region`, in one pass.
/// @note Two popcounts per word, never forming `~c`: `whenClear` is
/// `popcount(a&b&mask) - popcount(a&b&mask&c)`, the host's arithmetic, which
/// is also what keeps a trailing word's padding bits from counting.
cudaError_t countAndSplitAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                               DeviceBinMatConstView c, Rect region,
                               DeviceSplitCount* dResult, cudaStream_t stream = nullptr);

/// @brief The no-selector-plane form: the selector is `c0 ^ c1`, XORed a word
/// at a time inside the loop. **This is the form the covariance calls** --
/// with `c0 = sign_x`, `c1 = sign_y` -- and it needs no fifth frame-sized
/// plane, which is the trade CLAUDE.md's memory tiebreak already settled on
/// the host.
cudaError_t countAndSplitAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                               DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                               Rect region, DeviceSplitCount* dResult,
                               cudaStream_t stream = nullptr);

/// @brief Synchronous conveniences; see countNonZero's note.
SplitCount countAndSplit(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatConstView c, Rect region);
SplitCount countAndSplit(DeviceBinMatConstView a, DeviceBinMatConstView b,
                         DeviceBinMatConstView c0, DeviceBinMatConstView c1, Rect region);

// ---------------------------------------------------------------------------
// countCovariance -- all four numbers from one traversal
// ---------------------------------------------------------------------------

/// @brief Sets *dResult to `{xx, yy, xy.whenClear, xy.whenSet}` over `region`,
/// from ONE traversal. Device twin of the host countCovariance.
cudaError_t countCovarianceAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                 DeviceBinMatConstView c, Rect region,
                                 DeviceCovarianceCount* dResult,
                                 cudaStream_t stream = nullptr);

/// @brief The no-selector-plane form; see countAndSplit's four-argument note.
cudaError_t countCovarianceAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                 DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                 Rect region, DeviceCovarianceCount* dResult,
                                 cudaStream_t stream = nullptr);

/// @brief Synchronous conveniences; see countNonZero's note.
CovarianceCount countCovariance(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                DeviceBinMatConstView c, Rect region);
CovarianceCount countCovariance(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                Rect region);

// ---------------------------------------------------------------------------
// The batched form -- N windows, ONE launch
// ---------------------------------------------------------------------------

/// @brief One covariance per region, all in a single launch: one block per
/// window. **THE ENTRY POINT A TRACKER USES.**
/// @param dRegions `count` Rects in DEVICE memory, each clipped exactly as the
/// single-region form clips (negative origins legal, empty ones yield zeros).
/// @param dResults `count` results in DEVICE memory, written in region order.
/// @note Per-window launches are latency-bound -- a launch is ~5-10 us against
/// a 31x31 window's nanoseconds of work -- which is why this exists and why
/// a keypoint-shaped caller should never loop over the single-region form.
/// @note Identical counts to the single-region form: integer addition, so the
/// combination order cannot change the answer. The tests hold it to both
/// that form and the host.
/// @note Allocates nothing. Regions and results are caller-owned device memory,
/// the same contract the host's caller-provided scratch carries.
cudaError_t countCovarianceBatchAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                      DeviceBinMatConstView c0, DeviceBinMatConstView c1,
                                      const Rect* dRegions, size_t count,
                                      DeviceCovarianceCount* dResults,
                                      cudaStream_t stream = nullptr);

/// @brief The selector-plane form of the batch.
cudaError_t countCovarianceBatchAsync(DeviceBinMatConstView a, DeviceBinMatConstView b,
                                      DeviceBinMatConstView c, const Rect* dRegions,
                                      size_t count, DeviceCovarianceCount* dResults,
                                      cudaStream_t stream = nullptr);

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
