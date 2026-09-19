#pragma once

/// @file opticalFlow.hpp
/// @brief The device arm of ops/opticalFlow.hpp: pyramidal Lucas-Kanade tracking
/// of sparse keypoints over bit-packed frames, **one launch per frame**.
/// **API TIER 2** -- `cv::calcOpticalFlowPyrLK`'s role and call shape,
/// deliberately different numerics, NOT bit-exact against OpenCV.
/// **Bit-exact against the HOST binCV tracker, which is this backend's
/// contract.**
///
/// ---------------------------------------------------------------------------
/// WHERE THE FORMAT'S ADVANTAGE IS, AND WHERE IT IS NOT
///
/// The residual identity is the whole of it, and it is not "popcount is fast".
/// Bilinear interpolation is LINEAR in its four taps, and the taps of a binary
/// next frame are BITS, so
///
///     b1 = w00*S(T00) + w01*S(T01) + w10*S(T10) + w11*S(T11) - S(I)
///
/// with `S(M) = sum over the window of M * Ix` a signed masked popcount. The
/// four float weights are applied ONCE PER WINDOW PER ITERATION, not once per
/// pixel. Any byte-per-pixel LK -- CPU or GPU, `cv::cuda`'s included -- must
/// interpolate J per pixel per iteration; this does not.
///
/// **The arithmetic ratio that follows is about 3.5x, not 11x**, and the
/// difference matters because a ratio that dissolves on unit-checking is a
/// claimed advantage the kernel does not have. Counted in LANE-OPS at a 31x31
/// window: binCV pays 31 lanes x ~60 warp-instructions plus 8 warp reductions
/// per iteration; a byte-per-pixel LK pays 961 pixels x ~7 float ops. ~1,860
/// against ~6,727. Counted in WARP-INSTRUCTIONS against the role bar's actual
/// mapping (`cv::cuda` dispatches 31x31 to `block(16,16)`, `patch(2,2)` -- 256
/// threads, 8 warps, ~30 ops per thread per iteration): ~70 against ~240, about
/// 3.4x. Two independent counts, one number.
///
/// **WHAT IS NOT AN ADVANTAGE, STATED BECAUSE AN EARLIER DRAFT CLAIMED IT WAS.**
/// `cv::cuda`'s sparse PyrLK keeps its patch in PER-THREAD REGISTERS
/// (`work_type I_patch[PATCH_Y][PATCH_X]`, 2x2 at a 31x31 window) and uses
/// shared memory only for three 256-float reduction buffers. There is no
/// kilobytes-of-shared-staging to beat, binCV is not on a different side of any
/// shared-memory/register line, and no occupancy claim rests on one. Read out of
/// `cudaoptflow/src/cuda/pyrlk.cu` in the very build the role benchmark links.
///
/// **AND THE PARALLELISM RUNS THE OTHER WAY. IT WAS STATED BEFORE MEASURING AND
/// IT MATERIALISED.** binCV brings one WARP per keypoint where the role bar
/// brings a 256-thread BLOCK -- eight. So binCV does ~3.5x less work per
/// keypoint with about 8x less parallelism to hide latency. Measured on the
/// reference GPU at 61 keypoints over the shipped 1/2/2/2 ladder: achieved
/// occupancy **7.7% against cv::cuda's 21.3%**, and pure kernel time **65.4 us
/// in one launch against 41.9 us across four** -- a 1.56x loss in kernel WORK,
/// which the benchmark's wall-clock win (one launch against six) conceals and
/// says so at the number.
///
/// The stall histogram names the limiter: `math_pipe_throttle` is **0** of 2,753
/// stall samples, `short_scoreboard` -- the warp-shuffle and MIO queue, i.e. the
/// CROSS-LANE REDUCTION -- is **50.9%**, and DRAM throughput is 0.93%. This
/// kernel is reduction-latency-bound at low occupancy: not FP64-bound, which an
/// earlier design predicted at ~85%, and not bandwidth-bound. That is why the
/// self terms are hoisted out of the iteration and why the reduction is an arm
/// with a runtime switch -- swapping it alone moves the kernel 1.16x.
/// The remedy it points at is more warps per keypoint, which is a traversal
/// redesign rather than a tuning knob.
///
/// ---------------------------------------------------------------------------
/// THE LAUNCH SHAPE: ONE LAUNCH PER FRAME, LEVEL LOOP INSIDE THE KERNEL
///
/// Keypoints are independent at every level -- the host's pass 1 writes only
/// `nextPts[p]`, its pass 2 reads and writes only `nextPts[p]`, `status[p]` and
/// `err[p]`, and `entryLevelFor` reads only `prevPts[p]` and the level extents.
/// So fusing the host's two passes and its four levels into ONE kernel is an
/// identity transform on the output, not an approximation, and it is what this
/// file does. Four launches at this host's ~9 us floor would have been 36 us
/// against a few microseconds of kernel work: the same "the cap was the
/// signature's, not the operation's" result the batched covariance recorded at
/// 467x, applied one level up.
///
/// One WARP per keypoint; lane `l` owns window row `l`, and row `l + 32` when the
/// clipped window is taller than 32. That is the mapping the FORMAT hands you: a
/// window at most 32 pixels wide is at most ONE uint32 per row per plane, so the
/// host's 5,766-byte patch copy and its 2 KB of stack staging become **five
/// words per lane and ZERO bytes of shared memory**.
///
/// The level descriptors travel in the kernel's own parameter space, so the
/// operation still allocates nothing: 16 x `DeviceLKLevel` plus the parameters is
/// about 2.3 KB of the 4 KB limit.
///
/// ---------------------------------------------------------------------------
/// THE DOMAIN IS NARROWER THAN THE HOST'S, AND EVERY EDGE OF IT RETURNS A VALUE
///
/// Ruling R4: a device op may accept a narrower domain than its host twin when
/// the docstring NAMES the domain, the op asserts it, and it returns an error
/// outside it. This one does all three. Outside the domain it returns
/// `cudaErrorInvalidValue` and writes nothing -- **a refusal, never a quietly
/// different answer.**
///
///   * `winWidth  <= 32` -- one lane holds one word per row. The host reaches
///     wider windows with multi-word `ReplicatedShiftedRow` spans, which is a
///     different traversal shape and can be added later without changing any
///     answer inside this domain.
///   * `winHeight <= 64` -- two lane slots per warp. This is the host's own
///     `kStagedMaxRows`, not a smaller number invented here.
///   * bits per level `<= 2` -- the shipped 1/2/2/2 ladder. The per-word cost is
///     `20 N^2` popcounts and the register staging is `(3N + 2)` words per lane
///     per slot; N >= 3 is scalar by nature even on the host and the shipped
///     ladder does not reach it.
///   * `levelCount <= 16` -- the host's own `impl::kMaxLevels`, and what fits the
///     parameter space. (The host CLAMPS a longer ladder to 16; this refuses it.
///     A clamp and a refusal are both values, and the refusal is the one a
///     caller can see.)
///   * `err` / `maxResidual` need a ONE-BIT level 0. The `|Jinterp - I|`
///     collapse is exact only because `I` is a bit; the host's N > 1 form is
///     per-pixel and does not port. The shipped ladder's level 0 IS one bit, so
///     this leaves no hole in it.
///
/// ---------------------------------------------------------------------------
/// BIT-EXACTNESS IS CONDITIONAL ON A BUILD FLAG, AND THE PROMISE BELONGS HERE
///
/// The tracker's float layer is O(iterations), never O(pixels), so exactness is
/// cheap to buy and worth a great deal: the epsilon test and the oscillation
/// test are FLOAT BRANCHES, so a one-ulp disagreement changes a point's
/// iteration count and can move its endpoint by pixels. A tolerance would have
/// no stable meaning.
///
/// Equality is achieved by identical IEEE-754 double operations in identical
/// order. `+ - * /` and `sqrt` are correctly rounded; `floor` and `fabs` are
/// exact; integer sums are order-independent. The one hazard is FUSED
/// MULTIPLY-ADD CONTRACTION, which silently changes `a12*b2 - a22*b1`,
/// `w00*t00 + w01*t01 + ...` and `deltaX*deltaX + deltaY*deltaY`. So:
///
///   **`src/opticalFlow.cu` is compiled `-fmad=false`, and the promise is:
///   device output equals host output when the HOST header is compiled without
///   FMA contraction (`-ffp-contract=off`).** GCC's own default is
///   `-ffp-contract=fast`, so a consumer who compiles `ops/opticalFlow.hpp` on
///   an FMA-capable host with default flags may see a point's iteration count
///   differ. That is a property of the build, it is stated here because the
///   header is where a caller reads the promise, and
///   `impl::lkFmaGuardProbeAsync` below exists so the suite fails when the
///   pinning is lost rather than silently changing tracks.
///
/// ---------------------------------------------------------------------------
/// THE COVARIANCE THIS TRACKER DOES NOT CALL, REPORTED RATHER THAN WORKED AROUND
///
/// `cuda/reduce.hpp` calls `countCovarianceBatchAsync` "THE ENTRY POINT A
/// TRACKER USES". This tracker does not call it, and neither does it call
/// `gradientCovarianceBatchAsync`: the warp already holds the staged window in
/// lane registers, so the 2x2 costs four popcounts and four warp reductions in
/// place, where a call into the batch kernel would mean a SECOND full traversal
/// of the window planes plus a device `Rect` array -- exactly the traversal the
/// host removed after measuring the covariance at 27.5% of `track`. CLAUDE.md
/// says a measurement contradicting a documented claim is reported, not worked
/// around, so: one of those two documents is wrong, `impl::lkCovarianceProbeAsync`
/// is held equal to `gradientCovarianceBatchAsync` by test so the two spellings
/// cannot drift, and the wording is the owner's to settle.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "core.hpp"
#include "covariance.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief The most pyramid levels this backend consumes. The host's
/// `impl::kMaxLevels`, restated as a device constant rather than narrowed.
constexpr size_t lkMaxLevels() { return 16; }
/// @brief The widest window a lane's single word can hold.
constexpr int lkMaxWindowWidth() { return 32; }
/// @brief The tallest window two lane slots can hold. The host's
/// `impl::kStagedMaxRows`.
constexpr int lkMaxWindowHeight() { return 64; }
/// @brief The deepest level this backend instantiates. The shipped ladder's
/// deepest rung.
constexpr unsigned lkMaxLevelBits() { return 2; }

/// @brief One pyramid level's `4N + 2` planes, as DEVICE pointers plus the
/// geometry a lane needs. **The kernel takes an array of these BY VALUE in
/// its parameter space, so the operation allocates nothing.**
///
/// @note FOUR STRIDES, NOT ONE. A single stride for all ten planes looks
/// justified by the host's `checkLevelPlanes` and is not: that function
/// asserts equal width and height and never compares stride. Two planes of
/// equal width may legally carry different strides (`DeviceBinMat`'s
/// `rowAlignmentBytes` is a per-object opt-in), and one stride would then
/// read the wrong rows and return a plausible wrong answer. Four strides is
/// not a compromise either: the planes arrive as FOUR device allocations --
/// the previous
/// frame's pyramid level, the next frame's, and the two derivative blocks --
/// and every plane inside one of those shares its stride by construction,
/// because a `DevicePlaneBlockConstView` has exactly one.
/// @note The derivative belongs to the PREVIOUS frame only. LK linearises about
/// the previous frame, which halves the derivative footprint against a naive
/// reading of the algorithm -- the host's note, and the same bytes here.
struct DeviceLKLevel {
    const uint32_t* prev[lkMaxLevelBits()] = {nullptr, nullptr};   ///< LSB-first planes
    const uint32_t* next[lkMaxLevelBits()] = {nullptr, nullptr};
    const uint32_t* dxMag[lkMaxLevelBits()] = {nullptr, nullptr};
    const uint32_t* dyMag[lkMaxLevelBits()] = {nullptr, nullptr};
    const uint32_t* dxSign = nullptr;  ///< a set bit is NEGATIVE
    const uint32_t* dySign = nullptr;

    size_t width = 0;   ///< PIXELS
    size_t height = 0;  ///< rows
    size_t prevStride = 0;  ///< WORDS between rows, previous frame's planes
    size_t nextStride = 0;
    size_t dxStride = 0;
    size_t dyStride = 0;

    unsigned bits = 0;  ///< bits per pixel at this level, 1 or 2
};

/// @brief Names an N-bit level's plane blocks into a `DeviceLKLevel`.
/// **API TIER 2 convenience; no kernel, no allocation.**
/// @param prev,next The two frames at this level, `planes == N`.
/// @param dxSigned,dySigned The PREVIOUS frame's derivative blocks,
/// `planes == N + 1`: magnitude 0..N-1 then the SIGN plane at index N.
/// Exactly what this backend's `derivativeXY` writes, and byte-identical
/// to the host `SignedQuantMat<N>`.
/// @note The one container-shaped entry point here, in the shape the host's
/// `lkLevel` uses: the block knows which plane is magnitude and which is
/// sign; the kernel does not.
/// @note Asserts the shared geometry the host's `checkLevelPlanes` asserts --
/// equal width and per-plane height across all four blocks -- and nothing
/// about stride, because this type carries each block's own.
DeviceLKLevel deviceLkLevel(DevicePlaneBlockConstView prev, DevicePlaneBlockConstView next,
                            DevicePlaneBlockConstView dxSigned,
                            DevicePlaneBlockConstView dySigned);

/// @brief The 1-bit spelling: a pyramid level that is a bit matrix reaches this
/// with no adapter, exactly as the derivative family's binary overloads do.
inline DeviceLKLevel deviceLkLevel(DeviceBinMatConstView prev, DeviceBinMatConstView next,
                                   DevicePlaneBlockConstView dxSigned,
                                   DevicePlaneBlockConstView dySigned) {
    return deviceLkLevel(
        DevicePlaneBlockConstView{prev.ptr, prev.width, prev.height, prev.stride, 1},
        DevicePlaneBlockConstView{next.ptr, next.width, next.height, next.stride, 1},
        dxSigned, dySigned);
}

/// @brief The four keypoint arrays, in DEVICE memory, named once.
///
/// @note **THESE ARE RAW `d`-PREFIXED POINTERS AND THAT IS A DECLENSION, NOT AN
/// OVERSIGHT.** A device-typed span would make "passed a host array to the
/// async form" a compile error, which raw pointers cannot. It is declined
/// here because the whole backend speaks this convention already --
/// `gradientCovarianceBatchAsync(..., const Rect* dWindows, ...)`,
/// `keypointsFromCorners(const DeviceCorner* dCorners, ...)`,
/// `reduce.hpp`'s `dResult` -- and inventing a span for ONE family would
/// leave the other nine unprotected while making this one the odd shape a
/// reader has to learn. Introducing that span is a decision about the
/// backend's vocabulary, and it belongs in `core.hpp` alongside the view
/// types, applied everywhere at once. Grouping the four arrays into one
/// named aggregate is what this file can do about it: the hazard now has
/// one place to be read about rather than four parameters to be missed in.
/// @note `xy` is the host family's own contract -- `count` interleaved `(x, y)`
/// float pairs, which is `bincv::Point2f[count]` byte for byte -- so an
/// upload is a raw copy of the array the host tracker already works on.
/// @note `dNextXY` must not overlap `dPrevXY`. The kernel reads `dPrevXY[p]` and
/// `dNextXY[p]` into registers before writing anything, so an in-place call
/// would not corrupt its own input the way the host's would -- but it would
/// still track from the wrong anchor, so the contract is the host's and the
/// launcher asserts it.
/// @note `dErr` is OPTIONAL. Null means the residual is not computed at all,
/// which is the same 2x-class saving the host records, and the same
/// `maxResidual` caveat: setting `maxResidual` turns the cost on whether or
/// not `dErr` was asked for.
struct DeviceLKTracks {
    const float* dPrevXY = nullptr;  ///< 2 * count floats, DEVICE. Read only.
    float* dNextXY = nullptr;        ///< 2 * count floats, DEVICE. Out; also IN under useInitialFlow.
    uint8_t* dStatus = nullptr;      ///< count bytes, DEVICE. Every entry written.
    float* dErr = nullptr;           ///< count floats, DEVICE, or null.
    uint32_t count = 0;              ///< keypoints, NOT floats
};

/// @brief Pyramidal Lucas-Kanade tracking of a whole keypoint set, in ONE
/// launch. **THE ENTRY POINT A RESIDENT TRACKER USES. API TIER 2.**
///
/// @param levels `levelCount` HOST-side level bundles, **level 0 (the finest)
/// first**, copied into the kernel's parameter space by value. The PLANES
/// they name are device memory; the array itself is not.
/// @param levelCount 1..16. Levels at or below the window size are ignored as a
/// prefix (the host's deviation (vi)); 0 is legal and loses every point.
/// @param tracks The four device arrays and the point count.
/// @param params The host's `LKParams`, unchanged and with the same defaults.
/// @param stream The stream the launch is enqueued on. **No synchronize, no
/// allocation, no download** -- this is the per-frame device work of a
/// resident tracker, and a synchronize in it would be the whole cost.
/// @return `cudaSuccess`, the launch's error, or `cudaErrorInvalidValue` for a
/// shape, depth or level count outside the domain named in the file header.
///
/// @note **Bit-exact against `bincv::calcOpticalFlowPyrLK` on the same planes
/// and keypoints**, `nextPts` compared as raw 32-bit words, `status` as
/// bytes, `err` as bits -- subject to the FMA note in the file header. It
/// reproduces all seven of the host's documented deviations, both entry-level
/// policies, `useInitialFlow`, all three loss rules, the oscillation rule with
/// its half-step back-off, the unconditional final range test, and the
/// iteration and epsilon clamps. Those ARE the algorithm; a device tracker
/// that dropped one of them would be a different operation wearing the name.
/// @note **Zero scratch, and structurally rather than by budget.** The window
/// staging lives in lane registers, the running estimate lives in registers,
/// the level descriptors live in the kernel's parameter space. Shared memory:
/// 0 bytes. There is no caller-provided scratch buffer and therefore no
/// sizing formula -- the host's "not one byte" property, reached by a
/// different mechanism.
cudaError_t calcOpticalFlowPyrLKAsync(const DeviceLKLevel* levels, size_t levelCount,
                                      const DeviceLKTracks& tracks,
                                      const LKParams& params = LKParams(),
                                      cudaStream_t stream = nullptr);

/// @brief Synchronous convenience taking HOST keypoint arrays: allocates the four
/// device arrays, uploads, launches, downloads, synchronizes. **API TIER 2.**
///
/// @note FOUR ALLOCATIONS AND ONE STREAM SYNCHRONIZE, exactly the contract
/// `reduce.hpp`'s `countNonZero(view)` carries. For tests and callers off the
/// hot path. **A resident tracker MUST use the async form**, where the
/// keypoint arrays never leave the device -- that is the entire argument for
/// a device tracker and this overload gives it away.
/// @note Takes `bincv::Point2f*` where the async form takes `float*`, which is
/// the one thing separating the two signatures at a glance. See
/// `DeviceLKTracks` for why that is thinner than it should be.
void calcOpticalFlowPyrLK(const DeviceLKLevel* levels, size_t levelCount,
                          const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                          float* err, size_t pointCount,
                          const LKParams& params = LKParams());

/// @brief Which arms a launch from THIS binary would actually take, as a string.
/// **API TIER 3, DIAGNOSTIC -- print it.**
///
/// @note **IT IS MEASURED, NOT ASSERTED, AND IT IS NOT CACHED.** This launches a
/// one-thread probe that takes the same branches the tracker takes and
/// reports the arm it landed on. That is the difference between "the arm we
/// believe is compiled in" and "the arm that ran", and it is the whole reason
/// the function exists: two arms of this operation produce byte-identical
/// output BY CONSTRUCTION, so equality between them cannot distinguish an arm
/// that ran from an arm that was compiled out. That is precisely the
/// mis-attached-`#define` failure CLAUDE.md records, and it is why the
/// `__reduce_add_sync` arm is a RUNTIME switch below rather than a
/// compile-time one.
/// @note The benchmark prints this and the test suite asserts it, so an arm that
/// silently stopped running fails the gate instead of quietly costing time.
/// @note One one-thread launch and one SYNCHRONIZE per call, re-probed every
/// time -- a cached first reading would tell a caller who flipped a switch
/// that the old arm is still running, which is the misreport this function
/// exists to prevent. Call it outside a timed region.
/// @note Not thread-safe against a concurrent flip of the switches below; it is
/// a diagnostic, and so are they.
const char* lkPathName();

namespace impl {

/// @brief Runtime switch for the `__reduce_add_sync` warp reduction (sm_80+),
/// against the portable five-step `__shfl_down_sync` tree. **INTERNAL,
/// defaults ON where the intrinsic is compiled in.**
///
/// @note **A RUNTIME switch, deliberately, and not a `#if` on the
/// architecture.** A compile-time gate does not satisfy this project's
/// arm rule at all: the suite has to hold BOTH arms to one output in ONE
/// binary, and a `#if` cannot be flipped between two calls. The shuffle tree
/// is also the arm a Jetson (sm_72) build ships, and an untested fallback is
/// not a fallback -- so it is compiled and RUN on the reference GPU even
/// though the intrinsic is available there.
/// @note The two arms cannot disagree: both are integer addition over the same
/// 32 lane values, and every reduced quantity is bounded well inside 32 bits
/// (a window is at most 32 x 64 pixels, so the largest weighted tap sum at
/// N = 2 is 9 x 2048 = 18,432). That is a property, not a hope, and it is
/// why this pair has no gate that excludes it -- which the benchmark says at
/// the number rather than substituting a control that cannot fail.
bool& lkWarpReduceIntrinsicEnabled();

/// @brief Runtime switch for the iteration-to-iteration TAP CACHE. **INTERNAL,
/// defaults ON.**
///
/// @note The four tap words move, but they move as `floor(offX)`, and the
/// iteration is SHRINKING `off` -- once the estimate settles inside a pixel
/// the integer part stops changing and the same words are re-extracted every
/// remaining iteration. The host's `impl::TapCache`, with lane registers
/// where the host has a 2 KB stack buffer. Sound by construction: the tap
/// words are a pure function of `next`, the clipped region and
/// `(tapX, tapY)`; the first two are fixed for the point and the third is
/// the key.
/// @note **THIS IS THE ARM WITH A GATE THAT EXCLUDES IT**, and it is this
/// family's answer to "is the fast path actually running": at
/// `maxIterations == 1` the cache can never hit, so on-vs-off must read
/// ~1.00x. If it does not, the switch is not selecting what it claims to.
bool& lkTapCacheEnabled();

/// @brief The five exact integers one gradient component's residual needs, as
/// the device writes them. **INTERNAL test hook.**
/// @note The device spelling of the host's `impl::TapSums`. Kept a separate POD
/// because it is written by device code and must be trivially copyable; the
/// host type's meaning is not restated here.
struct DeviceTapSums {
    long long t00;
    long long t01;
    long long t10;
    long long t11;
    long long self;
};

/// @brief `b = sum(diff * grad)` over ONE window at ONE displacement, as ten
/// exact integers. **API TIER 3, INTERNAL TEST HOOK.**
/// @param dOut TWO entries in device memory: the X component then the Y.
/// @note It exists so the EXACT-INTEGER layer is compared against the host
/// DIRECTLY rather than only through the tracker's floating-point tail.
/// These ten numbers are integers, so equality is a real check and a
/// one-ulp float difference cannot hide inside it.
/// @note One warp, one window, one displacement -- a launch per window, which is
/// why this is a test hook and not an entry point. The tracker computes the
/// same sums from the same staged registers without a second launch.
cudaError_t lkResidualSumsProbeAsync(const DeviceLKLevel& lv, Rect window, long long tapX,
                                     long long tapY, DeviceTapSums* dOut,
                                     cudaStream_t stream = nullptr);

/// @brief The 2x2 gradient covariance from the SAME staged lane registers the
/// tracker uses. **API TIER 3, INTERNAL TEST HOOK.**
/// @note Not a call into `covariance.cu`, and see the file header for why. The
/// suite holds this equal to BOTH the host's `gradientCovariance` AND this
/// backend's `gradientCovarianceBatchAsync` on the same windows, which is
/// what keeps the tracker's staged spelling and the batch kernel's traversal
/// spelling from drifting apart while each stays correct on its own.
cudaError_t lkCovarianceProbeAsync(const DeviceLKLevel& lv, Rect window,
                                   DeviceGradientCovariance* dOut,
                                   cudaStream_t stream = nullptr);

/// @brief `a*b - c*d` on the device, in one double. **API TIER 3, INTERNAL TEST
/// HOOK, AND A GATE NOBODY HAS WATCHED FAIL IS NOT KNOWN TO WORK.**
/// @note The bit-exactness claim in the file header rests on `src/opticalFlow.cu`
/// being compiled `-fmad=false` and on the host oracle being compiled
/// `-ffp-contract=off`. Both are BUILD FLAGS, so both can be lost without a
/// line of code changing. The suite evaluates this expression on operands
/// chosen so that the contracted and uncontracted results DIFFER, on both
/// targets, and requires them equal -- so a future flag change breaks the
/// suite instead of silently changing which points converge.
cudaError_t lkFmaGuardProbeAsync(double a, double b, double c, double d, double* dOut,
                                 cudaStream_t stream = nullptr);

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
