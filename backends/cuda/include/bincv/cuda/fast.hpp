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
/// RASTER ORDER, AND WHY THERE IS NO LONGER A SORT
///
/// The host emits in raster order and truncates by keeping the first `capacity`
/// of it. An atomic append cannot produce that order -- compaction.hpp says so
/// and refuses to promise it, and points at the alternative: "a family that
/// genuinely needs the host's truncation ORDER cannot get it from an atomic and
/// must compact by prefix sum instead."
///
/// That is what the shipped arm does, and it is where this operation's cost
/// used to be. A single-block bitonic network over the stored corners was
/// **98.5% of a 752x480 detection at the reference corner density** -- 2.236 ms
/// against 0.0328 ms of detection -- and its cost tracked `nextPow2(found)`
/// rather than the frame. There is nothing to sort: a word's corners are the set
/// bits of one mask and peeling them low bit first is already ascending `x`,
/// while the `(row, word)` UNITS are already in raster order. So the arm counts
/// each unit's corners, prefix-sums the counts over words (11,376 numbers for a
/// 752x480 frame, not 360,960), and writes every corner at the index the host
/// would have put it at. No two corners are ever compared.
///
/// **On that arm a TRUNCATED run IS the host's prefix**, because an index below
/// `capacity` is exactly a raster rank below `capacity`. That is a property of
/// the arm and not a promise of the operation: `impl::fastOrderedEnabled(false)`
/// selects the append-and-sort arm, whose atomic decided which corners were
/// stored before the sort saw them. The contract is unchanged --
/// `DeviceAppendResult::truncated()` is how a caller finds out, and `found()` is
/// exactly the capacity a complete re-run needs -- and the suite compares
/// truncated runs by `found()` for that reason.
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
/// * The SCORE is held in an `int32_t` on the device where the host type is a
/// `long long`, so a corner record is 12 bytes rather than 16. The bit-plane
/// score is an arc length around a 16-pixel ring and cannot leave [1, 16]; the
/// suite sweeps every value it can take rather than sampling, and
/// `DeviceFastCorner::toHost` widens it, so what a caller reads back is
/// unchanged. features.hpp carries the full note.

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

/// @brief Which detection arm a caller is sizing scratch for.
///
/// @note **THE TWO ARMS WANT DIFFERENT AMOUNTS OF DIFFERENT THINGS, so one number
/// could only ever be the larger one.** The ordered arm's scratch is one
/// `uint32_t` per block of the FRAME -- 380 bytes at 752x480, whatever the
/// capacity. The reference arm's is the bitonic network's padded working area,
/// `nextPow2(capacity)` corner records -- 384 KB at a capacity of 32,768,
/// whatever the frame. Naming the arm is what lets a caller of the shipped one
/// stop paying for the other.
enum class FastArm {
    Ordered,   ///< the shipped arm: prefix sum over words, raster order by construction
    Reference  ///< the append-and-sort arm, `impl::fastOrderedEnabled(false)`
};

/// @brief Scratch bytes `detectFastAsync` needs for this frame, this capacity and
/// this arm.
///
/// @note **THE ANSWER IS A PURE FUNCTION OF THE ARGUMENTS AT THE CALL SITE.** It
/// does not depend on a global switch, on `arcLength`, or on the shared-memory
/// budget: a sizing function whose answer changes when an implementation detail
/// moves turns a correct caller into a device-side out-of-bounds write with no
/// signal. What used to keep that property was handing every caller the larger
/// arm's number. What keeps it now is stronger: the arm is an ARGUMENT, and
/// `detectFastAsync` REFUSES rather than writes when the arm it is about to run
/// needs more than the caller passed. Size for `Ordered`, flip
/// `impl::fastOrderedEnabled()` to `false`, and the next call returns
/// `cudaErrorInvalidValue` -- there is no buffer-shaped hole to fall through.
/// @note `Ordered` answers for the arm that switch position actually SELECTS.
/// Where the ordered arm does not apply to this frame and capacity the reference
/// arm runs, and this returns the reference arm's number -- so a caller who names
/// the default always gets a buffer the call can use.
/// @note A caller that intends to run BOTH arms against one buffer -- the suite and
/// the benchmark do, because a switchable arm has to be switchable -- sizes for
/// `Reference`, which covers either.
/// @note Returns 0 where the arm that will run has nothing to keep: a capacity of
/// 0 or 1 on the reference arm, a frame with no detectable row on the ordered one.
size_t fastScratchBytes(size_t width, size_t height, size_t capacity,
                        FastArm arm = FastArm::Ordered);

/// @brief Detects FAST corners on a device bit-plane. **API TIER 2.**
/// Bit-exact against `bincv::detectFast(BinMatConstView<uint32_t>, ...)` --
/// positions, order AND the `long long` score, as `DeviceFastCorner::toHost`
/// hands them back -- for every COMPLETE run.
///
/// @param img The frame, one bit per pixel, in device memory.
/// @param out Append target and its counter (compaction.hpp). The counter must be
/// zeroed before every launch; `DeviceAppendCounter::reset` is that call.
/// @param scratch Device scratch of at least
/// `fastScratchBytes(img.width, img.height, out.capacity, arm)` bytes for the arm
/// that will run, 4-byte aligned. May be null when that is 0. **Caller-provided:
/// no kernel here allocates.**
/// @param scratchBytes What `scratch` actually holds, so a short buffer is a
/// refusal rather than a corruption. The arm is not an argument here -- it is
/// whichever one `impl::fastOrderedEnabled()` and the frame select -- so this is
/// where a buffer sized for the OTHER arm is caught, before a kernel reads it.
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

/// @brief Forces the append-and-sort arm, for the benchmark and the tests.
/// **INTERNAL.** `true` (default) selects the prefix-sum arm: raster order by
/// construction, no comparison between corners at all.
/// @note Both arms are held to the same corner array in one binary for every
/// COMPLETE run, which is the only run the operation's contract orders.
bool& fastOrderedEnabled();

/// @brief Whether the ordered arm would run for this frame and capacity.
/// **INTERNAL** -- the benchmark and the suite need to name the gate-excluded
/// case without restating the gate.
/// @note The gate compares the two arms' working areas: the network this arm
/// replaces runs over `nextPow2(capacity)` corner records, the prefix sum over one
/// `uint32_t` per block of the frame. Below the crossing -- a capacity under 17 on
/// a 752x480 frame -- the network is the smaller job and it runs. That is the case
/// the benchmark must report at ~1.00x between switch positions.
/// @note **THIS IS A HEURISTIC, NOT A DERIVED CROSSOVER, and the honest evidence
/// for that is that it MOVES.** Its left side is `nextPow2(capacity)` times the
/// size of a corner record, so narrowing the record from 16 bytes to 12 changed
/// it by a quarter. At 752x480 nothing moved -- the crossing is capacity 17 at
/// either record size -- but over a sweep of widths 32..2016 and heights 7..1199
/// about 4.5% of (frame, capacity) points now select the reference arm where they
/// used to select the ordered one, always in that direction and never above a
/// capacity of 128. Neither arm is wrong there, because both are bit-exact and
/// both are cheap at those counts; what is wrong is reading a byte count as a
/// work unit. A crossover measured rather than inferred is filed work.
bool fastOrderedApplies(size_t width, size_t height, size_t capacity);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
