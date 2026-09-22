#pragma once

/// @file sparseMatch.hpp
/// @brief Hamming descriptor matching, sparse rectified stereo and Hamming
/// block matching on the device: the twins of ops/descriptor.hpp's
/// `matchDescriptors` / `matchDescriptorsGated`, ops/stereo.hpp's whole
/// family, and ops/blockMatch.hpp's `calcOpticalFlowBlockMatch`.
/// **API TIER 3** throughout -- none of these has an OpenCV CPU equivalent, so
/// none of them borrows an OpenCV name.
///
/// ---------------------------------------------------------------------------
/// WHAT THE REPRESENTATION PAYS FOR HERE, PER OP, AND WHERE IT PAYS NOTHING
///
/// This family does NOT have one answer, and the two halves are different
/// enough that a single sentence about it would be wrong about one of them.
///
/// **THE DESCRIPTOR HALF: the instruction ratio is real and it is not where the
/// time goes. Both halves of that sentence are measured.**
///
/// A BRIEF descriptor is already a bit string on both sides of the comparison,
/// so "the caller already holds bits" buys nothing by itself -- OpenCV's caller
/// holds bits too. What is not equal is the WORD the popcount runs on.
/// `cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING)` dispatches on
/// the descriptor Mat's depth (`brute_force_matcher.cpp`'s `callersHamming[]`),
/// and for the `CV_8U` descriptor `cv::cuda::ORB` emits it selects
/// `matchHamming_gpu<uchar>`, whose `HammingDist::reduceIter` is
/// `__popc(a ^ b)` **per byte** -- 32 popcounts per 256-bit descriptor, eight
/// useful bits in each, against eight for a matcher reading `uint32` words.
///
/// **That 4x instruction ratio buys nothing measurable, and the measurement is
/// the point.** OpenCV also instantiates `matchHamming_gpu<int>`, so a `CV_32S`
/// header over the IDENTICAL bytes gets eight popcounts too -- and on the
/// reference GPU the two OpenCV arms are indistinguishable: 5.198 ms against
/// 5.186 ms at 5000 x 5000, 1.00x, with the ranges overlapping. The profiler
/// says why in one reading: that kernel's dominant stall is `barrier` (35,540
/// samples of ~108,000), then long-scoreboard latency; `math_pipe_throttle` --
/// the stall that would rise if popcount ISSUE were the limit -- is 871, under
/// 1% either way. A kernel that is `__syncthreads()`-bound does not care how
/// many popcounts it issues.
///
/// So the descriptor half's advantage is **kernel shape, not representation**,
/// and the same profile says which shape: binCV's tiled matcher runs at 75.9%
/// of peak SM throughput with 0.25% DRAM and ~8,000 stall samples, where
/// `matchUnrolledCached` runs at 62.9% with 12.5% DRAM and ~108,000. Two
/// `__syncthreads()` per launch against one per descriptor chunk per train
/// block. Measured end to end: 0.471 ms against 5.19 ms at 5000 x 5000, ranges
/// disjoint.
///
/// Memory is NOT a wash, and that was a surprise rather than a claim. By
/// arithmetic the two sides hold the same arrays -- two descriptor blocks and a
/// 16-byte record per query, 36.7 KB at 470 x 256-bit. Measured on one meter
/// (`cudaMemGetInfo` over 256 replicas, both sides), binCV reads 0.03 MB per
/// working set against cv::cuda's 0.50 MB. The cause is not the algorithm: a
/// 256-bit descriptor is a 32-BYTE ROW, and `GpuMat` pitches a row to a
/// 512-byte multiple, so OpenCV's descriptor matrices carry 16x their own data.
/// A packed device array does not. The OpenCV reading is an upper bound (it may
/// pool, and the metered scope runs one `knnMatchAsync` so its match buffer is
/// counted, as binCV's record array is) -- but the pitch is arithmetic, not
/// allocator behaviour, and it is the bulk of the gap.
///
/// **THE WINDOW HALF: the format's own advantage, and it is partial.** Stereo
/// refinement and block matching both score a window by
/// `popcount((L ^ R) & mask)` over packed rows, where a byte-per-pixel matcher
/// moves one byte per pixel. Quantified for the traversal THIS family actually
/// has -- one warp per keypoint, lanes over window rows -- and not for the
/// dense matcher's (whose lanes own consecutive anchors and therefore share
/// words, which nothing here does):
///
/// | op | window row | packed touches | bytes touched | vs one byte per pixel |
/// |---|---|---|---|---|
/// | block match | 31 wide | 2 words | 8 B | 31 B -> ~3.9x |
/// | stereo refine | 11 wide | 2 words | 8 B | 11 B -> ~1.4x |
///
/// And the `__popc` utilisation that goes with it, stated rather than left for
/// a reader to derive: an 11-wide window masked out of a 32-bit word puts 11
/// useful bits into a popcount over two loaded words -- **17%** -- and a
/// 31-wide window **48%**. This family is not the case where the
/// representation pays in full, and saying so here is cheaper than having it
/// found.
///
/// What the family DOES exploit, and what the design it came from missed, is
/// that every candidate of a search reads the SAME rows at consecutive shifts.
/// Three `word()` calls per row hand every candidate a single
/// `__funnelshift_r` (see sparseMatch.cuh), so `2R + 3` candidates cost one
/// row assembly instead of `2R + 3`. That is the same lesson the dense
/// matcher's sliding arm records -- pay a row twice, not `winHeight` times --
/// applied along the candidate axis instead of the row axis.
///
/// `__ballot_sync` appears nowhere in this file. Its primitive is `__popc`.
///
/// ---------------------------------------------------------------------------
/// BATCHED, TWO ARMS, ONE BINARY
///
/// Every entry point takes a WHOLE SET and issues ONE launch (block matching:
/// one launch per pyramid level, because coarse-to-fine is a serial dependency
/// between levels and nothing else). Each carries a REFERENCE arm -- a thread
/// per output element, transcribing the host loop -- reachable at runtime
/// through the switches in `impl::`, so both arms are held to the same output
/// in one binary and the benchmark can time either.
///
/// Each fast arm also has an input its OWN gate rejects, named here so the
/// ~1.00x control is runnable rather than asserted:
///   * the matchers and the stereo coarse stage: a descriptor wider than
///     `impl::kMatchTileMaxWords` words, which the shared-memory staging
///     refuses;
///   * stereo refinement and block matching: a window the span bound rejects
///     (`impl::windowSpanFits`).
///
/// ---------------------------------------------------------------------------
/// NO HEAP, AND ONE SCRATCH BUFFER
///
/// Nothing here allocates. Four of the five entry points need no scratch at
/// all: their reductions live in registers and warp shuffles, which is this
/// backend's spelling of the host's running-minimum rule. Block matching is the
/// exception and for a structural reason -- its per-keypoint estimate must
/// survive BETWEEN level launches -- so it takes caller-provided device scratch
/// sized by `blockMatchScratchBytes`.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "bincv/core/types.hpp"
#include "bincv/ops/blockMatch.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/stereo.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

// ---------------------------------------------------------------------------
// Block matching: the level type and the scratch
// ---------------------------------------------------------------------------

/// @brief One pyramid level for device block matching: both frames, and **no
/// derivative** -- the host `BlockMatchLevel`'s two fields, device-typed.
/// @note Deliberately not the tracker's level type, for the host's reason:
/// route (a) never forms a gradient, and a level that carried four unused
/// planes would misstate the footprint this operation is partly measured on.
struct DeviceBlockMatchLevel {
    DeviceBinMatConstView prev;
    DeviceBinMatConstView next;

    BINCV_CUDA_HD size_t width() const { return prev.width; }
    BINCV_CUDA_HD size_t height() const { return prev.height; }
};

/// @brief One keypoint's state between level launches.
/// @note THIS IS WHY THE OP TAKES SCRATCH AT ALL. Coarse-to-fine is a serial
/// dependency, so the level loop is a loop over LAUNCHES, and a register
/// does not survive one. Sixteen bytes, one aligned store per keypoint per
/// level.
struct alignas(16) DeviceBlockMatchState {
    int32_t estX;   ///< displacement estimate in the CURRENT level's pixels
    int32_t estY;
    uint32_t lost;  ///< 1 once a level clipped this keypoint's window to nothing
    uint32_t pad;
};

/// @brief Scratch bytes `calcOpticalFlowBlockMatch` needs for `pointCount`
/// keypoints.
/// @note `sizeof(DeviceBlockMatchState)` and not the 16 it happens to be: a
/// hardcoded width silently under-allocates the day a field widens.
inline constexpr size_t blockMatchScratchBytes(size_t pointCount) {
    return pointCount * sizeof(DeviceBlockMatchState);
}

/// @brief Pyramid levels a device call accepts.
/// @note A DOCUMENTED, ASSERTED DOMAIN NARROWING, not an oversight. The
/// reference arm carries the whole ladder as a by-value kernel argument so
/// that its coarse-to-fine lives in registers exactly as the host's does,
/// and a kernel's parameter space is 4 KB. Twelve levels of a 752x480 frame
/// bottom out at 1x1; the host's own cap rule stops at the window size long
/// before that, so the bound is not reachable by a real ladder.
inline constexpr size_t kMaxBlockMatchLevels = 12;

/// @brief Search radius a device call accepts.
/// @note A DOCUMENTED, ASSERTED DOMAIN NARROWING. The device estimate is
/// `int32_t` where the host's is `long long`, and a coarse-to-fine estimate
/// is bounded by `searchRadius * (2^levels - 1)`. With the level cap above,
/// this bound keeps that product under 2^31 for every input the op accepts,
/// so the narrowing is exact over its whole domain rather than usually.
inline constexpr int kMaxBlockMatchRadius = 1024;

namespace impl {

// -------------------------------------------------------------------------
// The runtime switches. Same contract as `censusTiledEnabled` and
// `briefBallotArmEnabled`: flip to false and the reference arm runs, in the
// same binary, on the same inputs.
// -------------------------------------------------------------------------

/// @brief Force the reference matcher (one thread per query, the host's serial
/// scan). Applies to both the ungated and the gated form. **INTERNAL.**
bool& matchTiledArmEnabled();

/// @brief Force the reference sparse-stereo arms (one thread per keypoint).
/// Applies to the coarse descriptor stage and to the refinement.
/// **INTERNAL.**
bool& sparseStereoFastArmEnabled();

/// @brief Force the reference block matcher (one thread per keypoint, one
/// launch, coarse-to-fine in registers). **INTERNAL.**
bool& blockMatchFastArmEnabled();

/// @brief Queries one matcher block owns, streaming the train set once for all
/// of them. **INTERNAL.**
/// @note A RUNTIME knob rather than a constant, because a tile width is
/// MEASURED here and not inherited: the benchmark sweeps it at BOTH scales
/// and prints the lines that chose the default. 1, 4 and 8 are
/// instantiated; anything else takes the default.
unsigned& matchQueryTile();

/// @brief The shipped tile width.
/// @note Not a guess and not inherited from another kernel. Measured on the
/// reference GPU at 5000 x 5000, 256-bit: tile 1 at 1.744 ms, tile 4 at
/// 0.658 ms, tile 8 at 0.441 ms. A wider tile trades blocks for reuse --
/// one train word load feeds `Tile` XOR-popcount pairs -- and at eight the
/// grid is still wide enough to fill the device at both the pipeline and
/// the map-scale point. The benchmark reprints the sweep next to this
/// value, so the number that chose it travels with it.
inline constexpr unsigned kMatchDefaultTile = 8;

/// @brief Descriptor words above which the tiled arms' own gate excludes them.
/// @note The tiled matcher stages its query tile in shared memory, so the
/// pitch it accepts is bounded; the coarse stereo stage stages one left
/// descriptor per block and shares the bound. A wider descriptor falls to
/// the reference arm, which is what gives the benchmark a case the fast
/// path's gate rejects -- it must read ~1.00x, or the switch is not wired.
inline constexpr uint32_t kMatchTileMaxWords = 16;

/// @brief Whether the funnel-shift arms accept this window and search span.
/// @param windowWidth The window's width in pixels (its CLIPPED width is never
/// larger, so testing the unclipped one is the conservative side).
/// @param candidateSpan The number of distinct shifts minus one -- `2R` for
/// block matching, `2 * refineRadius + 2` for stereo refinement.
/// @note The fast arms cache 96 source columns of a row and cut every
/// candidate out of them with one funnel shift, so they need the window to
/// fit a word and the run start to stay inside the first 64 bits. Both
/// bounds are structural, they are the arms' own gate, and a caller outside
/// them gets the reference arm rather than a wrong answer.
BINCV_CUDA_HD inline bool windowSpanFits(int windowWidth, int windowHeight,
                                         int candidateSpan) {
    return windowWidth > 0 && windowWidth <= 32 && windowHeight > 0 &&
           windowHeight <= 32 && candidateSpan >= 0 &&
           windowWidth + candidateSpan <= 64;
}

cudaError_t matchDescriptorsImpl(DeviceDescriptorSetConstView query,
                                 DeviceKeypointSetConstView queryPts,
                                 DeviceDescriptorSetConstView train,
                                 DeviceKeypointSetConstView trainPts, bool gated,
                                 float maxDx, float maxDy, int maxOctaveDelta,
                                 unsigned maxRatio, DeviceDescriptorMatch* dOut,
                                 cudaStream_t stream);

cudaError_t stereoDescriptorMatchImpl(DeviceKeypointSetConstView leftPts,
                                      DeviceDescriptorSetConstView leftDesc,
                                      DeviceKeypointSetConstView rightPts,
                                      DeviceDescriptorSetConstView rightDesc,
                                      const StereoMatchParams& params,
                                      DeviceStereoMatch* dOut, cudaStream_t stream);

cudaError_t stereoRefineDisparityImpl(DeviceBinMatConstView left,
                                      DeviceBinMatConstView right,
                                      DeviceKeypointSetConstView leftPts,
                                      const StereoMatchParams& params,
                                      DeviceStereoMatch* dInOut, cudaStream_t stream);

cudaError_t blockMatchImpl(const DeviceBlockMatchLevel* levels, size_t levelCount,
                           const Point2f* dPrevPts, Point2f* dNextPts, uint8_t* dStatus,
                           size_t pointCount, void* dScratch, size_t scratchBytes,
                           const BlockMatchParams& params, cudaStream_t stream);

} // namespace impl

// ---------------------------------------------------------------------------
// Descriptor matching
// ---------------------------------------------------------------------------

/// @brief Brute-force nearest neighbour with Lowe's ratio test, for a whole
/// query set in one launch. **API TIER 3.**
/// @param query,train Descriptor sets of the SAME word pitch. `train` may be
/// empty; `query` empty is a no-op.
/// @param dOut `query.count` records in device memory, every entry written.
/// @param maxRatio Reject unless `best * 100 <= secondBest * maxRatio`. Lowe's
/// 0.8 is `maxRatio == 80` -- an integer percentage, exactly as the host's,
/// so the rule is exact and needs no float compare.
/// @note Bit-exact against `bincv::matchDescriptors<uint32_t>` on the same
/// bytes, ties included: the host keeps the FIRST train index on a tie and
/// the device reproduces that with a plain 64-bit minimum over
/// `(distance << 32) | index` rather than with an ordering argument.
/// @note **The device record narrows `trainIndex` to `uint32_t`** where the
/// host's is `size_t`; see features.hpp for the domain and why it is not a
/// restriction on this backend. `train.count` is checked against it.
/// @note Never allocates and takes no scratch.
inline cudaError_t matchDescriptors(DeviceDescriptorSetConstView query,
                                    DeviceDescriptorSetConstView train,
                                    DeviceDescriptorMatch* dOut, unsigned maxRatio = 80,
                                    cudaStream_t stream = nullptr) {
    return impl::matchDescriptorsImpl(query, DeviceKeypointSetConstView{}, train,
                                      DeviceKeypointSetConstView{}, false, 0.0f, 0.0f, 0,
                                      maxRatio, dOut, stream);
}

/// @brief `matchDescriptors` restricted to candidates a pipeline's priors
/// admit: a position window, and optionally an octave band. **API TIER 3.**
/// @param queryPts,trainPts The positions the gate tests, and -- when
/// `octave` is non-null on BOTH -- the pyramid levels the band tests. Null
/// on one and not the other is refused, exactly as the host asserts.
/// @param maxDx,maxDy Half-extents of the admission window, in the positions'
/// own units. Must not be negative.
/// @note THE RATIO TEST RUNS INSIDE THE GATE, as the host's does: best and
/// second-best are the best ADMITTED candidates. An unbounded window
/// therefore reproduces `matchDescriptors` exactly, which the suite pins --
/// and which doubles as a structural check that the gate is the ONLY
/// difference between the two kernels.
/// @note **ON DEVICE THE GATE IS NOT A SPEED FEATURE, and this header does not
/// claim it is.** It rejects a candidate on two float compares before any
/// descriptor word is read, which on a CPU saves the `words` XORs and
/// popcounts behind it. On the device at pipeline scale the whole
/// brute-force match is a launch's worth of work, so there is nothing for
/// the gate to save; and it COSTS memory, since it carries two position
/// arrays the ungated form does not. What it still does is change the
/// ADMITTED SET, which is an accuracy property and is computed here bit for
/// bit as the host computes it. The benchmark measures both arms so the
/// claim is a number rather than a sentence.
inline cudaError_t matchDescriptorsGated(DeviceDescriptorSetConstView query,
                                         DeviceKeypointSetConstView queryPts,
                                         DeviceDescriptorSetConstView train,
                                         DeviceKeypointSetConstView trainPts, float maxDx,
                                         float maxDy, DeviceDescriptorMatch* dOut,
                                         unsigned maxRatio = 80, int maxOctaveDelta = 1,
                                         cudaStream_t stream = nullptr) {
    return impl::matchDescriptorsImpl(query, queryPts, train, trainPts, true, maxDx, maxDy,
                                      maxOctaveDelta, maxRatio, dOut, stream);
}

// ---------------------------------------------------------------------------
// Sparse rectified stereo
// ---------------------------------------------------------------------------

/// @brief COARSE stage: each left descriptor against the right keypoints in its
/// row band and disparity range, in one launch. **API TIER 3.**
/// @param dOut One `DeviceStereoMatch` per LEFT keypoint, every entry written.
/// @note **The no-candidate sentinel is excluded EXPLICITLY**, not by the
/// Hamming gate -- the host header is emphatic about this and so is the
/// kernel. A caller who raises `maxHamming` to "accept everything" must get
/// "no candidate", never a match fabricated from right keypoint 0.
/// @note No ratio test, for the host's stated reason: along an epipolar row the
/// second-best candidate is often the true match's neighbour.
/// @note Never allocates and takes no scratch.
inline cudaError_t stereoDescriptorMatch(
    DeviceKeypointSetConstView leftPts, DeviceDescriptorSetConstView leftDesc,
    DeviceKeypointSetConstView rightPts, DeviceDescriptorSetConstView rightDesc,
    DeviceStereoMatch* dOut, const StereoMatchParams& params = StereoMatchParams(),
    cudaStream_t stream = nullptr) {
    return impl::stereoDescriptorMatchImpl(leftPts, leftDesc, rightPts, rightDesc, params,
                                           dOut, stream);
}

/// @brief FINE stage: slide a window along the epipolar row around each valid
/// match's disparity and refine to sub-pixel, in one launch. **API TIER 3.**
/// @param left,right The RECTIFIED pair, one bit per pixel, same extent.
/// @param dInOut One entry per left keypoint. `valid` and `disparity` are read
/// on the way in -- from `stereoDescriptorMatch`, or filled by a caller
/// refining an initial disparity of their own -- and `disparity` and `valid`
/// are written on the way out.
/// @note The scan is HORIZONTAL only and the result is clamped to the params'
/// disparity range, exactly as the host's is.
/// @note `minDisparity` MAY be negative here: this is a separate entry point
/// and the host asserts non-negativity only in the coarse stage. The
/// device's packed argmin therefore packs `d - lo` and adds `lo` back after
/// the reduction, because a negative value in the low half would invert the
/// ordering.
/// @note **This op and block matching are the only floating point in the
/// family**, and both are compiled without fused multiply-add. See the
/// comment in backends/cuda/CMakeLists.txt: contraction would change the
/// window ANCHOR by a whole pixel, not the parabola by a last bit.
/// @note Never allocates and takes no scratch.
inline cudaError_t stereoRefineDisparity(
    DeviceBinMatConstView left, DeviceBinMatConstView right,
    DeviceKeypointSetConstView leftPts, DeviceStereoMatch* dInOut,
    const StereoMatchParams& params = StereoMatchParams(),
    cudaStream_t stream = nullptr) {
    return impl::stereoRefineDisparityImpl(left, right, leftPts, params, dInOut, stream);
}

/// @brief Both stages, two enqueues on one stream. **API TIER 3.**
/// @note Exactly `stereoDescriptorMatch` into `stereoRefineDisparity`; it
/// exists so the common caller cannot get the hand-off wrong, not because
/// the composition adds anything. It makes no independent performance
/// claim and has no benchmark arm of its own.
inline cudaError_t stereoMatchRectified(
    DeviceBinMatConstView left, DeviceBinMatConstView right,
    DeviceKeypointSetConstView leftPts, DeviceDescriptorSetConstView leftDesc,
    DeviceKeypointSetConstView rightPts, DeviceDescriptorSetConstView rightDesc,
    DeviceStereoMatch* dOut, const StereoMatchParams& params = StereoMatchParams(),
    cudaStream_t stream = nullptr) {
    const cudaError_t e = impl::stereoDescriptorMatchImpl(leftPts, leftDesc, rightPts,
                                                          rightDesc, params, dOut, stream);
    if (e != cudaSuccess) return e;
    return impl::stereoRefineDisparityImpl(left, right, leftPts, params, dOut, stream);
}

// ---------------------------------------------------------------------------
// Block matching -- route (a)
// ---------------------------------------------------------------------------

/// @brief Pyramidal keypoint tracking by integer Hamming block matching, for a
/// whole keypoint set. **API TIER 3.**
/// @param levels `levelCount` device-typed levels, **LEVEL 0 FIRST**, in a HOST
/// array -- exactly as the host entry point takes a host array of host
/// views. At most `kMaxBlockMatchLevels`.
/// @param dPrevPts,dNextPts,dStatus Device arrays of `pointCount` entries.
/// `dNextPts` and `dStatus` are written for every point whether or not it
/// was tracked, as the host writes them.
/// @param dScratch,scratchBytes Caller-provided device scratch of at least
/// `blockMatchScratchBytes(pointCount)`. **Nothing here allocates.**
/// @note ONE LAUNCH PER LEVEL, not one per keypoint and not one for the whole
/// job: coarse-to-fine is a serial dependency between levels and nothing
/// else, so the level loop is the only thing that has to be a loop over
/// launches. The reference arm does it in one launch with the ladder in
/// registers, and the suite holds the two to the same output.
/// @note Bit-exact against `bincv::calcOpticalFlowBlockMatch<uint32_t>`,
/// including the tie rule the host header calls out: ties keep the first
/// candidate in scan order (dy outer, dx inner, both from -radius), which
/// resolves a flat plateau to its top-left. The device reproduces it with a
/// packed minimum over `(cost << 32) | candidateIndex` in that same order.
/// @note `searchRadius <= kMaxBlockMatchRadius` and
/// `levelCount <= kMaxBlockMatchLevels` are ASSERTED and, outside Debug,
/// REFUSED with `cudaErrorInvalidValue` rather than truncated.
inline cudaError_t calcOpticalFlowBlockMatch(
    const DeviceBlockMatchLevel* levels, size_t levelCount, const Point2f* dPrevPts,
    Point2f* dNextPts, uint8_t* dStatus, size_t pointCount, void* dScratch,
    size_t scratchBytes, const BlockMatchParams& params = BlockMatchParams(),
    cudaStream_t stream = nullptr) {
    return impl::blockMatchImpl(levels, levelCount, dPrevPts, dNextPts, dStatus, pointCount,
                                dScratch, scratchBytes, params, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
