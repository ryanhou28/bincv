#pragma once

/// @file fast.hpp
/// @brief The device arm of ops/fast.hpp's BIT-PLANE detector. **API TIER 2.**
///
/// ---------------------------------------------------------------------------
/// WHAT IS PORTED AND WHAT IS NOT
///
/// The **bit-plane** overload only -- `detectFast(BinMatConstView, ...)`. The
/// wide-image overload consumes exactly the bytes `cv::cuda::FastFeatureDetector`
/// consumes and has no representational advantage of any kind, which CLAUDE.md's
/// scope rule puts out on its second prong: binCV would add nothing but a second
/// implementation to keep correct. A caller holding a wide frame reaches this
/// detector through `cuda::packBits` or `cuda::edgeThreshold`, and binarizing
/// first CHANGES THE ANSWER -- said plainly here, exactly as the census entry's
/// on-ramp is labelled in docs/reports/cuda.md.
///
/// ---------------------------------------------------------------------------
/// TIER 2, AND THE SCORE IS WHY -- the host header's note, unchanged
///
/// The DETECTION rule is `cv::FAST`'s: on binary content held as `CV_8U {0,255}`
/// this accepts precisely the corners `cv::FAST` accepts at any threshold in
/// [1,254]. The SCORE is the longest qualifying arc, 9 to 16, not OpenCV's
/// largest-surviving-threshold. Same role, same call shape, different numerics.
///
/// ---------------------------------------------------------------------------
/// WHY A BIT-PLANE FAST IS A SIMT ADVANTAGE AND NOT ONLY A BANDWIDTH ONE
///
/// On a bit-plane the detector is boolean algebra -- `corner = arc9(ring ^ center)`
/// over whole words. **A CUDA lane is a 32-bit machine**, so one `xor` in one lane
/// decides 32 pixels where `cv::cuda::FastFeatureDetector`'s lane decides one and
/// carries 8 useful bits in a 32-bit register.
///
/// **THE MAGNITUDE OF THAT RATIO IS NOT 32x AND THIS HEADER WILL NOT SAY IT IS.**
/// Two measured facts cut against the arithmetic, and both belong next to it:
///
/// * OpenCV's kernel does the four-compass-point early rejection before the
/// sixteen-point test, so the overwhelming majority of pixels cost about four
/// byte loads rather than seventeen. A per-pixel load ratio computed from
/// seventeen is an upper bound on a path almost no pixel takes.
/// * binCV's own host measurement of this identical algebra, at **256-bit**
/// width -- eight times a CUDA lane -- collected 1.50x on x86 and 2.37x on
/// aarch64 against `cv::FAST`. The project has therefore already measured the
/// instruction ratio being collected at a few percent.
///
/// What survives, and it is real: binCV's word form is **branchless** where
/// OpenCV's early exit diverges, and inside a warp a divergent early exit pays
/// both paths. The ratio that ships is the measured one, in
/// benchmark/cuda_fast_benchmark.cpp, not the derived one.
///
/// ---------------------------------------------------------------------------
/// RASTER ORDER, AND WHY THERE IS A SORT
///
/// The host emits in raster order and truncates by keeping the first `capacity`
/// of it. An atomic append cannot produce that order -- compaction.hpp says so
/// and refuses to promise it. So a COMPLETE run is sorted, in place, on the
/// unique key `(y, x)`, which restores exactly one sequence: the host's. That is
/// what lets the suite `memcmp` the two arrays rather than compare them as sets.
///
/// **A TRUNCATED run is still not the host's prefix**, and the sort does not make
/// it one: the atomic decided which corners were stored before the sort saw them.
/// `DeviceAppendResult::truncated()` is how a caller finds out, and
/// `found()` is exactly the capacity a complete re-run needs.
///
/// ---------------------------------------------------------------------------
/// THE DEVICE DOMAIN, NAMED because it is narrower than the host's
///
/// * Word type `uint32_t`, the backend's only device word. The host compiles at
/// 8, 16, 32 and 64.
/// * `arcLength` in [1, 16], as on the host. The TILED arm covers 9 and 12; every
/// other length runs the reference arm, which is the gate-excluded case the
/// benchmark reports at ~1.00x.
/// * Corner counts and capacities are `uint32_t` (compaction.hpp's domain).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/ops/fast.hpp"
#include "compaction.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief Scratch bytes `detectFastAsync` needs for a given output capacity.
///
/// @note **TOTAL, over every arm and every internal path.** It does not depend on
/// which switch is set, on `arcLength`, or on the shared-memory budget: a
/// sizing function whose answer changes when an implementation detail moves
/// turns a correct caller into a device-side out-of-bounds write with no
/// signal. The answer is the sort's padded working area and nothing else --
/// `nextPow2(capacity)` corner records -- and it is the same number whichever
/// detection arm ran.
/// @note Returns 0 for a capacity of 0 or 1, where there is nothing to order.
size_t fastScratchBytes(size_t capacity);

/// @brief Detects FAST corners on a device bit-plane. **API TIER 2.**
/// Bit-exact against `bincv::detectFast(BinMatConstView<uint32_t>, ...)` --
/// positions, order AND the `long long` score -- for every COMPLETE run.
///
/// @param img The frame, one bit per pixel, in device memory.
/// @param out Append target and its counter (compaction.hpp). The counter must be
/// zeroed before every launch; `DeviceAppendCounter::reset` is that call.
/// @param scratch Device scratch of at least `fastScratchBytes(out.capacity)`
/// bytes, 8-byte aligned. May be null when that is 0. **Caller-provided:
/// no kernel here allocates.**
/// @param scratchBytes What `scratch` actually holds, so a short buffer is a
/// refusal rather than a corruption.
/// @param arcLength Contiguous ring pixels required; 9 is `cv::FAST`'s default.
/// @return `cudaErrorInvalidValue` when a documented domain is violated, else the
/// launches' error code. Asynchronous.
///
/// @note **Pixels within 3 of a border are never candidates**, as on the host and
/// in `cv::FAST`: the ring would fall outside and there is no sensible border
/// rule for "is this a corner".
/// @note **Padding bits past `width` never become corners.** They are excluded by
/// the same mask that excludes the border columns, computed arithmetically
/// rather than by a per-bit loop, and the 752-wide case (16 padding bits in
/// the trailing word) is the shape the suite exercises it with.
/// @note Reads `img` only; never allocates; the only device memory written is
/// `out`, its counter, and `scratch`.
cudaError_t detectFastAsync(DeviceBinMatConstView img, DeviceFastCornerBuffer out,
                            void* scratch, size_t scratchBytes, int arcLength = 9,
                            cudaStream_t stream = nullptr);

namespace impl {

/// @brief Forces the reference arm, for the benchmark and the tests. **INTERNAL.**
/// @note A vector arm must be switchable off and the benchmark must show it is
/// on. Both arms are held to the same corner array in one binary by the
/// suite, and the benchmark prints the on/off ratio beside a case the fast
/// arm's own gate excludes -- which must read ~1.00x, or the switch is not
/// switching.
bool& fastTiledEnabled();

/// @brief Selects the scoring arm. **INTERNAL.** `true` (default) reads the score
/// off the eight nested arc-length masks; `false` peels each corner's ring and
/// calls the host's own `impl::fastLongestRun`.
/// @note **The host's crossover is deliberately NOT ported.** The host chooses per
/// chunk because its alternative is a per-corner bit TRANSPOSE at ~78 scalar
/// operations per corner; on the device the alternative is a 16-iteration
/// per-lane peel, which is divergent rather than merely scalar, so the
/// arithmetic the host's threshold was derived from does not hold here.
/// The switch stays because the two arms must be held to one output.
bool& fastMaskScoreEnabled();

/// @brief Whether the tiled arm would run for this `arcLength`. **INTERNAL** --
/// the benchmark and the suite need to name a gate-excluded case without
/// restating the gate, and a restated gate is a gate that can drift.
/// @note The gate is `arcLength == 9 || arcLength == 12`: those are the two
/// lengths whose doubling schedule is a compile-time constant. With a runtime
/// step every index into the sixteen-word ring array is variable, the array
/// leaves registers for local memory, and the dense matcher's own recorded
/// lesson applies verbatim. `arcLength = 10` is therefore the control case
/// that must read ~1.00x between switch positions.
bool fastTiledApplies(int arcLength);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
