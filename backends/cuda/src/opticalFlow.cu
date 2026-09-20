// Pyramidal Lucas-Kanade on the device: ONE LAUNCH PER FRAME, one warp per
// keypoint, the level loop inside the kernel.
//
// WHAT PORTED AND WHAT DID NOT
//
// include/bincv/ops/opticalFlow.hpp is 3,267 lines and most of it is CPU SHAPE
// that has no device meaning: the AVX2 eight-keypoint batch and
// impl/lkBatch_impl.hpp, the four NEON residual kernels, StagedWindow/TapCache
// as STACK buffers with BINCV_STAGING_BUDGET_BYTES (a Cortex-M stack budget),
// RowReader's Staged specialization, narrowLevel and the 64-bit-word trap (the
// device word type is uint32_t and nothing else, so there is nothing to
// reproduce), and the parallelFor backend. What ported is the ALGORITHM -- the
// propagation, the entry level, the clip, the staging, the covariance, the three
// loss rules, the iteration with its epsilon and oscillation rules, the final
// range test and the error term -- and the bit-exactness target. Roughly 400
// lines of algorithm out of 3,267 lines of file.
//
// WHAT IS SHARED RATHER THAN FORKED, WHICH IS THE POINT OF BINCV_HOST_DEVICE
//
// The bit arithmetic a device tap extraction is most able to get subtly wrong is
// the HOST's, called here rather than restated: impl::ReplicatedShiftedRow::word
// (the replicate fill and its lowOutside/highOutside clamps), impl::floorDiv,
// impl::edgeFill, impl::alignedWord, impl::clipRegion, impl::lowBitsMask,
// impl::TapSums::combine (the residual identity's four multiplies in a FIXED
// order, which is what the exactness claim rests on), impl::referenceMinEigScale,
// impl::minEigenValue and impl::combineBitSlicedPairs. Every one of them already
// carried the annotation or gained it with this work, and none of them is a
// traversal.
//
// THE THREE STRUCTURAL WINS THIS FILE TAKES OVER A NAIVE PORT, each an identity
// transform on integer sums and therefore unable to break bit-exactness:
//
//  1. THE SELF TERMS ARE HOISTED OUT OF THE ITERATION. sum(I * Ix) reads only the
//     PREVIOUS frame and has no dependence on (tapX, tapY), so two of the ten
//     warp reductions and two of the twenty popcount terms per lane move to the
//     per-level staging step, next to the covariance that already reads the same
//     registers. A 20% cut in the quantity that is the inner loop's cost.
//  2. THE SIGN PLANE IS PRE-SPLIT AT STAGING. The host spells a signed masked sum
//     as popc(mag & m) - 2*popc(mag & m & sign): three ANDs and two popcounts.
//     Staging pos = mag & ~sign and neg = mag & sign instead -- computed ONCE per
//     level, in the same registers -- makes every term popc(v & pos) - popc(v &
//     neg): two ANDs, two popcounts, identical integers. At 20 terms per lane per
//     iteration that is ~20 AND instructions removed from a ~60-instruction
//     integer budget, for two extra registers at N = 2 and none at N = 1.
//  3. THE `+1` TAP IS A SHIFT WHENEVER THE **CLIPPED WINDOW SPAN** IS UNDER 32,
//     which is the host's own rule (RowReader's tapIsShift_) and not "whenever
//     the plane is narrower than 32". No level of the reference ladder
//     (752/376/188/94) is narrower than 32, so the second reading would have
//     issued FOUR displaced extractions per lane per slot instead of two at every
//     level of the shipped operating point.
//
// ZERO SHARED MEMORY, ZERO SCRATCH, ZERO ALLOCATION. The window staging is five
// words per lane per slot in registers; the level descriptors ride in the
// kernel's parameter space.
//
// -fmad=false ON THIS TRANSLATION UNIT, and it is a correctness flag. See the
// header's bit-exactness section and impl::lkFmaGuardProbeAsync.

#include <cstdio>

#include "bincv/cuda/opticalFlow.hpp"

#include "bincv/core/types.hpp"
#include "bincv/cuda/core.hpp"
#include "bincv/ops/corner.hpp"
#include "bincv/ops/covariance.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/reduce.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

namespace {

using bincv::impl::RegionWords;

constexpr unsigned kFullMask = 0xFFFFFFFFu;
/// Warps per block. Four keypoints per block at 128 threads -- small enough that
/// a 200-keypoint frame still spreads over 50 blocks and 48 SMs.
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kBlockThreads = kWarpsPerBlock * 32u;

/// Bit 0: the `__reduce_add_sync` arm ran. Bit 1: the tap cache arm ran.
/// Bit 2: the warp-per-keypoint traversal ran (there is one traversal today, and
/// naming it is what makes a second one visible when it arrives).
constexpr unsigned kArmReduceIntrinsic = 1u;
constexpr unsigned kArmTapCache = 2u;
constexpr unsigned kArmWarpPerPoint = 4u;

/// @brief Is the `__reduce_add_sync` arm COMPILED IN for the architecture this
/// translation unit is being built for?
/// @note ONE definition of that question, called by both the tracker's reduction
/// and the arm probe -- so `lkPathName()` reports the arm that ran rather
/// than the arm this file believes it compiled.
__device__ __forceinline__ bool reduceIntrinsicAvailable() {
#if __CUDA_ARCH__ >= 800
    return true;
#else
    return false;
#endif
}

/// @brief Sum one 32-bit integer across the warp, result in EVERY lane.
/// @note Both arms are integer addition over the same 32 values, so they cannot
/// disagree. Every quantity reduced here is bounded well inside 32 bits: a
/// window is at most 32 x 64 = 2048 pixels, so a tap sum is in [-2048, 2048]
/// at N = 1 and, with the 2^(i+j) weights, in [-18432, 18432] at N = 2.
/// @note The shuffle tree leaves its answer in lane 0 only, so it broadcasts;
/// `__reduce_add_sync` already gives every lane the sum. Both spellings end
/// with the same value in all 32 lanes, which is what lets every lane
/// compute the scalar tail redundantly and never need a second broadcast.
__device__ __forceinline__ int warpSum(int v, bool intrinsic) {
#if __CUDA_ARCH__ >= 800
    if (reduceIntrinsicAvailable() && intrinsic) {
        return static_cast<int>(__reduce_add_sync(kFullMask, static_cast<unsigned>(v)));
    }
#else
    (void)intrinsic;
#endif
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(kFullMask, v, off);
    }
    return __shfl_sync(kFullMask, v, 0);
}

/// @brief One word of a NEXT-frame row, displaced by `off` pixels, with columns
/// outside the plane reading as the nearest edge pixel and rows outside it
/// clamped.
/// @note The bit arithmetic is the HOST's `impl::ReplicatedShiftedRow::word`,
/// called rather than restated. Only the two things that cannot be shared
/// are written here: the vertical clamp (the host's `displacedRow` does it
/// over a host view) and filling the aggregate from a DEVICE row pointer.
/// @note `word(0)` rather than `word(i)`: the window is at most one word wide in
/// this backend's domain, and the offset carries the window's own left edge,
/// so bit `b` of the result is plane column `off + b`.
__device__ __forceinline__ uint32_t replicatedShiftedWord(const uint32_t* base, size_t stride,
                                                          size_t width, size_t height,
                                                          long long y, long long off) {
    if (width == 0 || height == 0) return 0u;
    long long cy = y;
    if (cy < 0) cy = 0;
    const long long lastRow = static_cast<long long>(height) - 1;
    if (cy > lastRow) cy = lastRow;
    const uint32_t* row = base + static_cast<size_t>(cy) * stride;

    bincv::impl::ReplicatedShiftedRow<uint32_t> r;
    r.row = row;
    r.words = rowWords(width);
    r.width = width;
    r.tailMask = rowTailMask(width);
    r.off = off;
    r.leftFill = bincv::impl::edgeFill<uint32_t>(row, 0);
    r.rightFill = bincv::impl::edgeFill<uint32_t>(row, width - 1);
    return r.word(0);
}

// ---------------------------------------------------------------------------
// The staged window: five words per lane per slot, in REGISTERS
// ---------------------------------------------------------------------------

/// @brief One keypoint's window, staged into the warp's lane registers.
/// @note `live[s]` is per lane: lane `l` owns window rows `l` and `l + 32`, and a
/// clipped window shorter than 64 rows leaves some of them unowned. Nothing
/// outside the accumulation depends on it, so the warp never diverges around
/// a reduction.
/// @note `posX`/`negX` rather than `magX`/`signX` -- the pre-split described at
/// the top of this file. The magnitudes are already masked to the clipped
/// span, so both derived words are, and a sign plane's padding bits can
/// never reach a count.
template <unsigned MaxBits>
struct StagedWindow {
    uint32_t self[2][MaxBits];
    uint32_t posX[2][MaxBits];
    uint32_t negX[2][MaxBits];
    uint32_t posY[2][MaxBits];
    uint32_t negY[2][MaxBits];
    uint32_t mask;
    bool live[2];
};

/// @brief Stage the window, and take the two things that read ONLY the staged
/// registers with it: the 2x2 covariance and the loop-invariant self terms.
/// @param covOut The 2x2, identical in every lane.
/// @param selfXOut,selfYOut `sum(I * Ix)` and `sum(I * Iy)` over the window,
/// identical in every lane. Hoisted out of the iteration because they have
/// no dependence on the displacement -- see structural win 1.
/// @note The covariance is computed from the SAME registers the residual will
/// use, not from a second traversal of the level's planes. That is the whole
/// reason this tracker does not call `gradientCovarianceBatchAsync`; the
/// header says so and a probe holds the two equal.
template <unsigned MaxBits>
__device__ void stageWindow(const DeviceLKLevel& lv, const RegionWords<uint32_t>& r,
                            unsigned bits, unsigned lane, bool intrinsic,
                            StagedWindow<MaxBits>& s, GradientCovariance& covOut,
                            long long& selfXOut, long long& selfYOut) {
    const size_t rows = r.y1 - r.y0;
    const size_t span = r.x1 - r.x0;
    const size_t words = rowWords(lv.width);
    s.mask = bincv::impl::lowBitsMask<uint32_t>(span);

    // The covariance needs the magnitudes and the sign SELECTOR; the residual
    // needs the pos/neg split. Both come off the same five loads, and the
    // magnitude and sign words die at the end of this function.
    uint32_t magX[2][MaxBits];
    uint32_t magY[2][MaxBits];
    uint32_t sel[2];

#pragma unroll
    for (unsigned sl = 0; sl < 2; ++sl) {
        const size_t i = static_cast<size_t>(sl) * 32u + lane;
        s.live[sl] = (i < rows);
        const size_t y = r.y0 + (s.live[sl] ? i : 0);
        uint32_t sgnX = 0;
        uint32_t sgnY = 0;
        if (s.live[sl]) {
            sgnX = bincv::impl::alignedWord<uint32_t>(lv.dxSign + y * lv.dxStride, words,
                                                      r.x0);
            sgnY = bincv::impl::alignedWord<uint32_t>(lv.dySign + y * lv.dyStride, words,
                                                      r.x0);
        }
        sel[sl] = sgnX ^ sgnY;
#pragma unroll
        for (unsigned k = 0; k < MaxBits; ++k) {
            uint32_t sf = 0, mx = 0, my = 0;
            if (s.live[sl] && k < bits) {
                sf = bincv::impl::alignedWord<uint32_t>(lv.prev[k] + y * lv.prevStride,
                                                        words, r.x0);
                mx = bincv::impl::alignedWord<uint32_t>(lv.dxMag[k] + y * lv.dxStride,
                                                        words, r.x0) &
                     s.mask;
                my = bincv::impl::alignedWord<uint32_t>(lv.dyMag[k] + y * lv.dyStride,
                                                        words, r.x0) &
                     s.mask;
            }
            s.self[sl][k] = sf;
            magX[sl][k] = mx;
            magY[sl][k] = my;
            s.posX[sl][k] = mx & ~sgnX;
            s.negX[sl][k] = mx & sgnX;
            s.posY[sl][k] = my & ~sgnY;
            s.negY[sl][k] = my & sgnY;
        }
    }

    // THE 2x2, from the staged registers. `3N^2 + N` popcounts per lane per slot
    // -- 4 at N = 1, 14 at N = 2 -- then one warp reduction per counter, then the
    // HOST's own weighting. combineBitSlicedPairs carries BINCV_HOST_DEVICE and is
    // CALLED, not forked: the doubled off-diagonal and SplitCount::crossTerm have
    // one definition across both backends.
    //
    // At bits == 1 with MaxBits == 2 every counter of plane index 1 is zero, and
    // combineBitSlicedPairs<2> over those zeros is combineBitSlicedPairs<1> of the
    // rest -- the weights multiply zero and crossTerm(0, 0) is 0 -- so the deeper
    // instantiation gives the shallower level's exact integers.
    bincv::impl::BitSlicedPairCounts<MaxBits> counts;
#pragma unroll
    for (unsigned a = 0; a < MaxBits; ++a) {
#pragma unroll
        for (unsigned b = a; b < MaxBits; ++b) {
            int xx = 0, yy = 0;
#pragma unroll
            for (unsigned sl = 0; sl < 2; ++sl) {
                if (!s.live[sl]) continue;
                xx += __popc(magX[sl][a] & magX[sl][b]);
                yy += __popc(magY[sl][a] & magY[sl][b]);
            }
            counts.xx[a][b] = static_cast<size_t>(warpSum(xx, intrinsic));
            counts.yy[a][b] = static_cast<size_t>(warpSum(yy, intrinsic));
        }
#pragma unroll
        for (unsigned b = 0; b < MaxBits; ++b) {
            int tot = 0, set = 0;
#pragma unroll
            for (unsigned sl = 0; sl < 2; ++sl) {
                if (!s.live[sl]) continue;
                const uint32_t both = magX[sl][a] & magY[sl][b];
                tot += __popc(both);
                set += __popc(both & sel[sl]);
            }
            counts.xyTotal[a][b] = static_cast<size_t>(warpSum(tot, intrinsic));
            counts.xySet[a][b] = static_cast<size_t>(warpSum(set, intrinsic));
        }
    }
    covOut = bincv::impl::combineBitSlicedPairs<MaxBits>(counts);

    // THE SELF TERMS, hoisted. Same shape as one tap term, run once per level.
    int sx = 0, sy = 0;
#pragma unroll
    for (unsigned sl = 0; sl < 2; ++sl) {
        if (!s.live[sl]) continue;
#pragma unroll
        for (unsigned i = 0; i < MaxBits; ++i) {
            if (i >= bits) continue;
#pragma unroll
            for (unsigned j = 0; j < MaxBits; ++j) {
                if (j >= bits) continue;
                const int w = 1 << (i + j);
                const uint32_t v = s.self[sl][i];
                sx += w * (__popc(v & s.posX[sl][j]) - __popc(v & s.negX[sl][j]));
                sy += w * (__popc(v & s.posY[sl][j]) - __popc(v & s.negY[sl][j]));
            }
        }
    }
    selfXOut = warpSum(sx, intrinsic);
    selfYOut = warpSum(sy, intrinsic);
}

/// @brief The four displaced tap words per lane per slot per plane.
/// @note `t01 = t00 >> 1` whenever the CLIPPED SPAN is under 32 -- the host's
/// own rule, exact because bit `b` of the `+1` extraction is bit `b + 1` of
/// the same 32-bit extraction and the replicate fill agrees on both. See
/// structural win 3.
template <unsigned MaxBits>
__device__ __forceinline__ void extractTaps(const DeviceLKLevel& lv,
                                            const RegionWords<uint32_t>& r, unsigned bits,
                                            unsigned lane, const StagedWindow<MaxBits>& s,
                                            long long tapX, long long tapY, bool tapIsShift,
                                            uint32_t (&tap)[2][MaxBits][4]) {
    const long long srcX = static_cast<long long>(r.x0) + tapX;
#pragma unroll
    for (unsigned sl = 0; sl < 2; ++sl) {
        const long long srcY =
            static_cast<long long>(r.y0 + static_cast<size_t>(sl) * 32u + lane) + tapY;
#pragma unroll
        for (unsigned k = 0; k < MaxBits; ++k) {
            uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
            if (s.live[sl] && k < bits) {
                t0 = replicatedShiftedWord(lv.next[k], lv.nextStride, lv.width, lv.height,
                                           srcY, srcX);
                t2 = replicatedShiftedWord(lv.next[k], lv.nextStride, lv.width, lv.height,
                                           srcY + 1, srcX);
                if (tapIsShift) {
                    t1 = t0 >> 1;
                    t3 = t2 >> 1;
                } else {
                    t1 = replicatedShiftedWord(lv.next[k], lv.nextStride, lv.width,
                                               lv.height, srcY, srcX + 1);
                    t3 = replicatedShiftedWord(lv.next[k], lv.nextStride, lv.width,
                                               lv.height, srcY + 1, srcX + 1);
                }
            }
            tap[sl][k][0] = t0;
            tap[sl][k][1] = t1;
            tap[sl][k][2] = t2;
            tap[sl][k][3] = t3;
        }
    }
}

/// @brief The eight per-tap residual sums, warp-reduced. The self terms are NOT
/// here -- they are loop-invariant and were taken at staging.
template <unsigned MaxBits>
__device__ __forceinline__ void residualSums(const StagedWindow<MaxBits>& s,
                                             const uint32_t (&tap)[2][MaxBits][4],
                                             unsigned bits, bool intrinsic, int (&rx)[4],
                                             int (&ry)[4]) {
    int lx[4] = {0, 0, 0, 0};
    int ly[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned sl = 0; sl < 2; ++sl) {
        if (!s.live[sl]) continue;
#pragma unroll
        for (unsigned i = 0; i < MaxBits; ++i) {
            if (i >= bits) continue;
#pragma unroll
            for (unsigned j = 0; j < MaxBits; ++j) {
                if (j >= bits) continue;
                const int w = 1 << (i + j);
                const uint32_t px = s.posX[sl][j];
                const uint32_t nx = s.negX[sl][j];
                const uint32_t py = s.posY[sl][j];
                const uint32_t ny = s.negY[sl][j];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t v = tap[sl][i][t];
                    lx[t] += w * (__popc(v & px) - __popc(v & nx));
                    ly[t] += w * (__popc(v & py) - __popc(v & ny));
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        rx[t] = warpSum(lx[t], intrinsic);
        ry[t] = warpSum(ly[t], intrinsic);
    }
}

/// @brief `w00*t00 + w01*t01 + w10*t10 + w11*t11 - self`, through the HOST's own
/// `impl::TapSums::combine` so the four multiplies keep their order.
__device__ __forceinline__ double combineTaps(const int (&r)[4], long long self, double w00,
                                              double w01, double w10, double w11) {
    bincv::impl::TapSums t;
    t.t00 = r[0];
    t.t01 = r[1];
    t.t10 = r[2];
    t.t11 = r[3];
    t.self = self;
    return t.combine(w00, w01, w10, w11);
}

// ---------------------------------------------------------------------------
// The tracker
// ---------------------------------------------------------------------------

/// Everything one launch needs, in the kernel's 4 KB parameter space.
/// 16 x sizeof(DeviceLKLevel) is about 2.2 KB of it, which is why the level
/// descriptors need no device buffer and this operation allocates nothing.
template <unsigned MaxBits>
struct TrackArgs {
    DeviceLKLevel levels[lkMaxLevels()];
    const float* prevXY;
    float* nextXY;
    uint8_t* status;
    float* err;
    uint32_t count;
    unsigned usableLevels;
    int winW;
    int winH;
    float halfWinX;
    float halfWinY;
    int maxIterations;
    double eps2;
    double minEigThreshold;
    double maxResidual;
    bool useInitialFlow;
    bool deepestFitting;
    bool tapCache;
    bool intrinsic;
    bool wantErr;
};

/// @brief The host's `impl::windowFitsAtLevel`, in the host's own doubles.
__device__ __forceinline__ bool windowFitsAtLevel(const DeviceLKLevel& lv, float px, float py,
                                                  unsigned li, float halfWinX,
                                                  float halfWinY) {
    const double scale = 1.0 / static_cast<double>(size_t{1} << li);
    const double x = static_cast<double>(px) * scale;
    const double y = static_cast<double>(py) * scale;
    return x - static_cast<double>(halfWinX) >= 0.0 &&
           y - static_cast<double>(halfWinY) >= 0.0 &&
           x + static_cast<double>(halfWinX) < static_cast<double>(lv.width) &&
           y + static_cast<double>(halfWinY) < static_cast<double>(lv.height);
}

template <unsigned MaxBits>
__global__ __launch_bounds__(kBlockThreads) void trackKernel(TrackArgs<MaxBits> a) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warpInBlock = threadIdx.x >> 5;
    const uint32_t p = blockIdx.x * kWarpsPerBlock + warpInBlock;
    if (p >= a.count) return;

    // THE FOUR OUTPUT VALUES LIVE IN REGISTERS FOR THE WHOLE KERNEL and are
    // written once at the end. The host writes `nextPts[p]` at every level and
    // inside every iteration; nothing ever reads it back except this same point,
    // so keeping it in a register and storing the last value is an identity
    // transform -- and it is what lets the four levels fuse into one launch.
    const float prevPx = a.prevXY[2u * p];
    const float prevPy = a.prevXY[2u * p + 1u];
    // Read BEFORE anything is written, which is what makes `useInitialFlow`'s
    // read of the caller's guess safe in a fused kernel.
    const float initX = a.nextXY[2u * p];
    const float initY = a.nextXY[2u * p + 1u];
    float curX = initX;
    float curY = initY;
    uint8_t status = 1u;
    float errVal = 0.0f;

    if (a.usableLevels == 0u) {
        // Degenerate but legal, and a VALUE rather than an error: with no levels
        // there is nothing to track on. Under `useInitialFlow` the caller's guess
        // IS the last estimate and overwriting it would throw away the better of
        // the two.
        if (!a.useInitialFlow) {
            curX = prevPx;
            curY = prevPy;
        }
        status = 0u;
    } else {
        unsigned entry = a.usableLevels - 1u;
        if (a.deepestFitting) {
            entry = 0u;
            for (unsigned li = a.usableLevels; li-- > 0;) {
                if (windowFitsAtLevel(a.levels[li], prevPx, prevPy, li, a.halfWinX,
                                      a.halfWinY)) {
                    entry = li;
                    break;
                }
            }
        }

        for (unsigned lidx = a.usableLevels; lidx-- > 0;) {
            const unsigned li = lidx;
            if (li > entry) continue;  // pass 1 AND pass 2 both skip

            // PASS 1 -- propagate into this level's coordinates.
            const float scale = 1.0f / static_cast<float>(1u << li);
            if (li == entry) {
                const float seedX = a.useInitialFlow ? initX : prevPx;
                const float seedY = a.useInitialFlow ? initY : prevPy;
                curX = seedX * scale;
                curY = seedY * scale;
            } else {
                curX *= 2.0f;
                curY *= 2.0f;
            }

            // PASS 2 -- track.
            const DeviceLKLevel& lv = a.levels[li];
            const bool finest = (li == 0u);
            const long long levelWidth = static_cast<long long>(lv.width);
            const long long levelHeight = static_cast<long long>(lv.height);
            const double levelMinEigScale = bincv::impl::referenceMinEigScale(lv.bits);

            const float prevX = prevPx * scale - a.halfWinX;
            const float prevY = prevPy * scale - a.halfWinY;
            const long long anchorX = static_cast<long long>(floorf(prevX));
            const long long anchorY = static_cast<long long>(floorf(prevY));

            // LOSS RULE 1 -- the window's origin is out of range.
            if (anchorX < -static_cast<long long>(a.winW) || anchorX >= levelWidth ||
                anchorY < -static_cast<long long>(a.winH) || anchorY >= levelHeight) {
                if (finest) status = 0u;
                continue;
            }

            const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), a.winW,
                              a.winH);
            const RegionWords<uint32_t> r =
                bincv::impl::clipRegion<uint32_t>(lv.width, lv.height, window);
            if (r.isEmpty) {
                if (finest) status = 0u;
                continue;
            }

            StagedWindow<MaxBits> s;
            GradientCovariance cov;
            long long selfX = 0, selfY = 0;
            stageWindow<MaxBits>(lv, r, lv.bits, lane, a.intrinsic, s, cov, selfX, selfY);

            const double a11 = static_cast<double>(cov.sumXX);
            const double a22 = static_cast<double>(cov.sumYY);
            const double a12 = static_cast<double>(cov.sumXY);
            const double det = a11 * a22 - a12 * a12;

            // LOSS RULE 2 -- a degenerate window. `det` is a difference of
            // products of exact popcounts, so it is 0 or at least 1 and the test
            // needs no epsilon.
            const double minEig = static_cast<double>(
                bincv::impl::minEigenValue(cov.sumXX, cov.sumYY, cov.sumXY));
            const double referenceMinEig =
                levelMinEigScale * minEig / static_cast<double>(a.winW * a.winH);
            if (det <= 0.0 || referenceMinEig < a.minEigThreshold) {
                if (finest) status = 0u;
                continue;
            }

            const bool tapIsShift = (r.x1 - r.x0) < 32u;
            uint32_t tap[2][MaxBits][4];
            long long cachedTapX = 0, cachedTapY = 0;
            bool cacheValid = false;

            float nextX = curX - a.halfWinX;
            float nextY = curY - a.halfWinY;
            double prevDeltaX = 0.0;
            double prevDeltaY = 0.0;
            bool inRange = true;

            for (int it = 0; it < a.maxIterations; ++it) {
                const long long originX = static_cast<long long>(floorf(nextX));
                const long long originY = static_cast<long long>(floorf(nextY));

                // LOSS RULE 3 -- the estimate walked out of range mid-iteration.
                if (originX < -static_cast<long long>(a.winW) || originX >= levelWidth ||
                    originY < -static_cast<long long>(a.winH) || originY >= levelHeight) {
                    if (finest) status = 0u;
                    inRange = false;
                    break;
                }

                // THE DISPLACEMENT IS MEASURED FROM `prevX`, NOT FROM THE INTEGER
                // ANCHOR. They differ by frac(prevX), which is zero at level 0 for
                // integer keypoints and is NOT zero at any coarser level; the host
                // records the wrong spelling costing up to 1.4 px through four
                // levels.
                const double offX = static_cast<double>(nextX) - static_cast<double>(prevX);
                const double offY = static_cast<double>(nextY) - static_cast<double>(prevY);
                const long long tapX = static_cast<long long>(floor(offX));
                const long long tapY = static_cast<long long>(floor(offY));
                const double fx = offX - static_cast<double>(tapX);
                const double fy = offY - static_cast<double>(tapY);
                const double w00 = (1.0 - fx) * (1.0 - fy);
                const double w01 = fx * (1.0 - fy);
                const double w10 = (1.0 - fx) * fy;
                const double w11 = fx * fy;

                const bool fresh =
                    a.tapCache && cacheValid && cachedTapX == tapX && cachedTapY == tapY;
                if (!fresh) {
                    extractTaps<MaxBits>(lv, r, lv.bits, lane, s, tapX, tapY, tapIsShift,
                                         tap);
                    cachedTapX = tapX;
                    cachedTapY = tapY;
                    cacheValid = a.tapCache;
                }

                int rx[4], ry[4];
                residualSums<MaxBits>(s, tap, lv.bits, a.intrinsic, rx, ry);
                const double b1 = combineTaps(rx, selfX, w00, w01, w10, w11);
                const double b2 = combineTaps(ry, selfY, w00, w01, w10, w11);

                // The factor of 2 turns the raw [-1, 0, 1] tap into a central
                // difference -- the host's UNITS section, and a derivation rather
                // than a tuning knob.
                const double deltaX =
                    bincv::impl::kCentralDifferenceScale * (a12 * b2 - a22 * b1) / det;
                const double deltaY =
                    bincv::impl::kCentralDifferenceScale * (a12 * b1 - a11 * b2) / det;

                nextX += static_cast<float>(deltaX);
                nextY += static_cast<float>(deltaY);
                curX = nextX + a.halfWinX;
                curY = nextY + a.halfWinY;

                // TERMINATION 1 -- converged.
                if (deltaX * deltaX + deltaY * deltaY <= a.eps2) break;

                // TERMINATION 2 -- oscillation: this step almost exactly undoes
                // the last one. Back off by half a step and stop.
                if (it > 0 && fabs(deltaX + prevDeltaX) < 0.01 &&
                    fabs(deltaY + prevDeltaY) < 0.01) {
                    curX -= static_cast<float>(deltaX * 0.5);
                    curY -= static_cast<float>(deltaY * 0.5);
                    nextX = curX - a.halfWinX;
                    nextY = curY - a.halfWinY;
                    break;
                }
                prevDeltaX = deltaX;
                prevDeltaY = deltaY;
            }

            // THE FINAL PASS, AND IT IS ABOUT THE POSITION THAT IS RETURNED, not
            // about the last iterate: the range test is re-applied to it, and --
            // only if it survives -- the error term is measured there, from taps
            // and weights recomputed from the returned position.
            if (finest && status != 0u && inRange) {
                const float finalX = curX - a.halfWinX;
                const float finalY = curY - a.halfWinY;
                const long long fx0 = static_cast<long long>(floorf(finalX));
                const long long fy0 = static_cast<long long>(floorf(finalY));
                if (fx0 < -static_cast<long long>(a.winW) || fx0 >= levelWidth ||
                    fy0 < -static_cast<long long>(a.winH) || fy0 >= levelHeight) {
                    // The last iteration's step can carry the point out of range
                    // after the in-loop test has already passed, and `status`
                    // describes the position the caller gets. Unconditional here,
                    // where OpenCV makes it depend on whether `err` was asked for.
                    status = 0u;
                } else if (a.wantErr) {
                    const double offX =
                        static_cast<double>(finalX) - static_cast<double>(prevX);
                    const double offY =
                        static_cast<double>(finalY) - static_cast<double>(prevY);
                    const long long tapX = static_cast<long long>(floor(offX));
                    const long long tapY = static_cast<long long>(floor(offY));
                    const double ffx = offX - static_cast<double>(tapX);
                    const double ffy = offY - static_cast<double>(tapY);
                    const double w[4] = {(1.0 - ffx) * (1.0 - ffy), ffx * (1.0 - ffy),
                                         (1.0 - ffx) * ffy, ffx * ffy};
                    extractTaps<MaxBits>(lv, r, lv.bits, lane, s, tapX, tapY, tapIsShift,
                                         tap);

                    // THE ERROR TERM IS THE POPCOUNT IDENTITY, and it is exact
                    // only because I is a BIT: |Jinterp - I| = I + (1 - 2I)*Jinterp
                    // collapses the window sum to ten counts. The N > 1 form is
                    // per-pixel on the host and is refused here (see the header).
                    int pixels = 0, countI = 0, allT[4] = {0, 0, 0, 0},
                        selfT[4] = {0, 0, 0, 0};
#pragma unroll
                    for (unsigned sl = 0; sl < 2; ++sl) {
                        if (!s.live[sl]) continue;
                        pixels += __popc(s.mask);
                        const uint32_t iw = s.self[sl][0] & s.mask;
                        countI += __popc(iw);
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const uint32_t tw = tap[sl][0][t] & s.mask;
                            allT[t] += __popc(tw);
                            selfT[t] += __popc(tw & iw);
                        }
                    }
                    pixels = warpSum(pixels, a.intrinsic);
                    countI = warpSum(countI, a.intrinsic);
                    double sumJ = 0.0;
                    double sumJoverI = 0.0;
#pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int at = warpSum(allT[t], a.intrinsic);
                        const int st = warpSum(selfT[t], a.intrinsic);
                        sumJ += w[t] * static_cast<double>(at);
                        sumJoverI += w[t] * static_cast<double>(st);
                    }
                    if (pixels != 0) {
                        const double total =
                            static_cast<double>(countI) + sumJ - 2.0 * sumJoverI;
                        // THE ROUNDING TO FLOAT HAPPENS BEFORE THE REJECT, not after.
                        // The host's `windowMeanAbsDiff` RETURNS float and the
                        // caller widens it back, so the value `maxResidual` is
                        // compared against is the float-rounded one. Comparing the
                        // full double here would keep a point the host drops in
                        // the narrow band between the two.
                        const float residual =
                            static_cast<float>(total / static_cast<double>(pixels));
                        errVal = residual;
                        if (a.maxResidual > 0.0 &&
                            static_cast<double>(residual) > a.maxResidual) {
                            status = 0u;
                        }
                    }
                }
            }
        }
    }

    if (lane == 0u) {
        a.nextXY[2u * p] = curX;
        a.nextXY[2u * p + 1u] = curY;
        a.status[p] = status;
        if (a.err != nullptr) a.err[p] = errVal;
    }
}

// ---------------------------------------------------------------------------
// The probes
// ---------------------------------------------------------------------------

template <unsigned MaxBits>
__global__ void residualProbeKernel(DeviceLKLevel lv, Rect window, long long tapX,
                                    long long tapY, bool intrinsic,
                                    impl::DeviceTapSums* out) {
    const unsigned lane = threadIdx.x & 31u;
    const RegionWords<uint32_t> r =
        bincv::impl::clipRegion<uint32_t>(lv.width, lv.height, window);
    impl::DeviceTapSums zero{0, 0, 0, 0, 0};
    if (r.isEmpty) {
        if (lane == 0u) {
            out[0] = zero;
            out[1] = zero;
        }
        return;
    }
    StagedWindow<MaxBits> s;
    GradientCovariance cov;
    long long selfX = 0, selfY = 0;
    stageWindow<MaxBits>(lv, r, lv.bits, lane, intrinsic, s, cov, selfX, selfY);

    uint32_t tap[2][MaxBits][4];
    extractTaps<MaxBits>(lv, r, lv.bits, lane, s, tapX, tapY, (r.x1 - r.x0) < 32u, tap);
    int rx[4], ry[4];
    residualSums<MaxBits>(s, tap, lv.bits, intrinsic, rx, ry);
    if (lane == 0u) {
        impl::DeviceTapSums sx{rx[0], rx[1], rx[2], rx[3], selfX};
        impl::DeviceTapSums sy{ry[0], ry[1], ry[2], ry[3], selfY};
        out[0] = sx;
        out[1] = sy;
    }
}

template <unsigned MaxBits>
__global__ void covarianceProbeKernel(DeviceLKLevel lv, Rect window, bool intrinsic,
                                      DeviceGradientCovariance* out) {
    const unsigned lane = threadIdx.x & 31u;
    const RegionWords<uint32_t> r =
        bincv::impl::clipRegion<uint32_t>(lv.width, lv.height, window);
    if (r.isEmpty) {
        if (lane == 0u) *out = DeviceGradientCovariance{0, 0, 0};
        return;
    }
    StagedWindow<MaxBits> s;
    GradientCovariance cov;
    long long selfX = 0, selfY = 0;
    stageWindow<MaxBits>(lv, r, lv.bits, lane, intrinsic, s, cov, selfX, selfY);
    if (lane == 0u) {
        *out = DeviceGradientCovariance{static_cast<long long>(cov.sumXX),
                                        static_cast<long long>(cov.sumYY),
                                        static_cast<long long>(cov.sumXY)};
    }
}

__global__ void fmaGuardKernel(double a, double b, double c, double d, double* out) {
    *out = a * b - c * d;
}

/// Reports the arms a launch from THIS binary takes, by TAKING THEM.
__global__ void armProbeKernel(bool intrinsic, bool tapCache, unsigned* out) {
    unsigned mask = kArmWarpPerPoint;
    if (reduceIntrinsicAvailable() && intrinsic) mask |= kArmReduceIntrinsic;
    if (tapCache) mask |= kArmTapCache;
    *out = mask;
}

// ---------------------------------------------------------------------------
// Launch-side validation
// ---------------------------------------------------------------------------

bool levelInDomain(const DeviceLKLevel& lv) {
    if (lv.bits < 1u || lv.bits > lkMaxLevelBits()) return false;
    if (lv.width == 0 || lv.height == 0) return false;
    if (lv.next[0] == nullptr || lv.dxSign == nullptr || lv.dySign == nullptr) return false;
    for (unsigned k = 0; k < lv.bits; ++k) {
        if (lv.prev[k] == nullptr || lv.next[k] == nullptr || lv.dxMag[k] == nullptr ||
            lv.dyMag[k] == nullptr) {
            return false;
        }
    }
    return true;
}

template <unsigned MaxBits>
cudaError_t launchTrack(const DeviceLKLevel* levels, size_t usableLevels,
                        const DeviceLKTracks& tracks, const LKParams& params,
                        int maxIterations, double eps2, cudaStream_t stream) {
    TrackArgs<MaxBits> a{};
    for (size_t i = 0; i < usableLevels; ++i) a.levels[i] = levels[i];
    a.prevXY = tracks.dPrevXY;
    a.nextXY = tracks.dNextXY;
    a.status = tracks.dStatus;
    a.err = tracks.dErr;
    a.count = tracks.count;
    a.usableLevels = static_cast<unsigned>(usableLevels);
    a.winW = params.winWidth;
    a.winH = params.winHeight;
    a.halfWinX = static_cast<float>(params.winWidth - 1) * 0.5f;
    a.halfWinY = static_cast<float>(params.winHeight - 1) * 0.5f;
    a.maxIterations = maxIterations;
    a.eps2 = eps2;
    a.minEigThreshold = static_cast<double>(params.minEigThreshold);
    a.maxResidual = static_cast<double>(params.maxResidual);
    a.useInitialFlow = params.useInitialFlow;
    a.deepestFitting = (params.entryLevel == LKEntryLevel::DeepestFitting);
    a.tapCache = impl::lkTapCacheEnabled();
    a.intrinsic = impl::lkWarpReduceIntrinsicEnabled();
    a.wantErr = (tracks.dErr != nullptr) || (params.maxResidual > 0.0f);

    const unsigned blocks =
        static_cast<unsigned>((tracks.count + kWarpsPerBlock - 1u) / kWarpsPerBlock);
    trackKernel<MaxBits><<<blocks, kBlockThreads, 0, stream>>>(a);
    return cudaGetLastError();
}

} // namespace

// ---------------------------------------------------------------------------

DeviceLKLevel deviceLkLevel(DevicePlaneBlockConstView prev, DevicePlaneBlockConstView next,
                            DevicePlaneBlockConstView dxSigned,
                            DevicePlaneBlockConstView dySigned) {
    const size_t n = prev.planes;
    BINCV_ASSERT(n >= 1 && n <= lkMaxLevelBits(),
                 "cuda::deviceLkLevel: this backend tracks 1- or 2-bit levels");
    BINCV_ASSERT(next.planes == n && dxSigned.planes == n + 1 && dySigned.planes == n + 1,
                 "cuda::deviceLkLevel: next must have N planes and each derivative N + 1");
    // The host's checkLevelPlanes rule -- equal width and per-plane height across
    // every plane of the level. NOT equal stride: this type carries each block's
    // own, because two blocks of equal width may legally differ there.
    BINCV_ASSERT(prev.width == next.width && prev.height == next.height &&
                     prev.width == dxSigned.width && prev.height == dxSigned.height &&
                     prev.width == dySigned.width && prev.height == dySigned.height,
                 "cuda::deviceLkLevel: a level's planes must share its dimensions");

    DeviceLKLevel lv;
    for (size_t k = 0; k < n; ++k) {
        lv.prev[k] = prev.planeData(k);
        lv.next[k] = next.planeData(k);
        lv.dxMag[k] = dxSigned.planeData(k);
        lv.dyMag[k] = dySigned.planeData(k);
    }
    lv.dxSign = dxSigned.planeData(n);
    lv.dySign = dySigned.planeData(n);
    lv.width = prev.width;
    lv.height = prev.height;
    lv.prevStride = prev.stride;
    lv.nextStride = next.stride;
    lv.dxStride = dxSigned.stride;
    lv.dyStride = dySigned.stride;
    lv.bits = static_cast<unsigned>(n);
    return lv;
}

cudaError_t calcOpticalFlowPyrLKAsync(const DeviceLKLevel* levels, size_t levelCount,
                                      const DeviceLKTracks& tracks, const LKParams& params,
                                      cudaStream_t stream) {
    if (tracks.count == 0u) return cudaSuccess;
    if (tracks.dPrevXY == nullptr || tracks.dNextXY == nullptr ||
        tracks.dStatus == nullptr) {
        return cudaErrorInvalidValue;
    }
    // The host's own precondition: pass 1 writes nextPts before pass 2 reads
    // prevPts, so an in-place call tracks from the wrong anchor.
    BINCV_ASSERT(tracks.dPrevXY != tracks.dNextXY,
                 "cuda::calcOpticalFlowPyrLK: dNextXY must not alias dPrevXY");
    if (params.winWidth <= 2 || params.winHeight <= 2) return cudaErrorInvalidValue;
    if (params.winWidth > lkMaxWindowWidth() || params.winHeight > lkMaxWindowHeight()) {
        return cudaErrorInvalidValue;
    }
    if (levelCount > lkMaxLevels()) return cudaErrorInvalidValue;
    if (levelCount > 0 && levels == nullptr) return cudaErrorInvalidValue;

    unsigned maxBits = 1u;
    for (size_t i = 0; i < levelCount; ++i) {
        if (!levelInDomain(levels[i])) return cudaErrorInvalidValue;
        if (levels[i].bits > maxBits) maxBits = levels[i].bits;
    }
    // The error term's popcount identity is exact only for a ONE-BIT level 0.
    const bool wantErr = (tracks.dErr != nullptr) || (params.maxResidual > 0.0f);
    if (levelCount > 0 && wantErr && levels[0].bits != 1u) return cudaErrorInvalidValue;

    // The host's own prefix rule (deviation (vi)): levels at or below the window
    // size are ignored. Called rather than restated, so the two backends cannot
    // consume different ladders.
    size_t usable = 0;
    if (levelCount > 0) {
        usable = bincv::impl::usableLevelCount(
            levelCount, params.winWidth, params.winHeight, [&](size_t i) {
                return bincv::impl::LevelDims{levels[i].width, levels[i].height};
            });
        if (usable > lkMaxLevels()) usable = lkMaxLevels();
    }

    // The host clamps both criteria before use, because they arrive from a config
    // file. Same clamps, same order.
    int maxIterations = params.maxIterations;
    if (maxIterations < 0) maxIterations = 0;
    if (maxIterations > 100) maxIterations = 100;
    float eps = params.epsilon;
    if (eps < 0.0f) eps = 0.0f;
    if (eps > 10.0f) eps = 10.0f;
    const double eps2 = static_cast<double>(eps) * static_cast<double>(eps);

    if (maxBits == 1u) {
        return launchTrack<1>(levels, usable, tracks, params, maxIterations, eps2, stream);
    }
    return launchTrack<2>(levels, usable, tracks, params, maxIterations, eps2, stream);
}

void calcOpticalFlowPyrLK(const DeviceLKLevel* levels, size_t levelCount,
                          const Point2f* prevPts, Point2f* nextPts, uint8_t* status,
                          float* err, size_t pointCount, const LKParams& params) {
    if (pointCount == 0) return;
    BINCV_ASSERT(prevPts != nullptr && nextPts != nullptr && status != nullptr,
                 "cuda::calcOpticalFlowPyrLK: prevPts, nextPts and status must be non-null");

    const size_t xyBytes = pointCount * 2 * sizeof(float);
    float* dPrev = nullptr;
    float* dNext = nullptr;
    uint8_t* dStatus = nullptr;
    float* dErr = nullptr;
    BINCV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPrev), xyBytes));
    BINCV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dNext), xyBytes));
    BINCV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dStatus), pointCount));
    if (err != nullptr) {
        BINCV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dErr),
                                    pointCount * sizeof(float)));
    }

    BINCV_CUDA_CHECK(cudaMemcpy(dPrev, prevPts, xyBytes, cudaMemcpyHostToDevice));
    // nextPts is an INPUT under useInitialFlow, and uploading it unconditionally
    // costs one copy of an array that is about to be overwritten -- which is
    // cheaper than a branch that would make the two modes take different code.
    BINCV_CUDA_CHECK(cudaMemcpy(dNext, nextPts, xyBytes, cudaMemcpyHostToDevice));

    DeviceLKTracks tracks;
    tracks.dPrevXY = dPrev;
    tracks.dNextXY = dNext;
    tracks.dStatus = dStatus;
    tracks.dErr = dErr;
    tracks.count = static_cast<uint32_t>(pointCount);

    const cudaError_t rc = calcOpticalFlowPyrLKAsync(levels, levelCount, tracks, params);
    if (rc == cudaSuccess) {
        BINCV_CUDA_CHECK(cudaDeviceSynchronize());
        BINCV_CUDA_CHECK(cudaMemcpy(nextPts, dNext, xyBytes, cudaMemcpyDeviceToHost));
        BINCV_CUDA_CHECK(cudaMemcpy(status, dStatus, pointCount, cudaMemcpyDeviceToHost));
        if (err != nullptr) {
            BINCV_CUDA_CHECK(
                cudaMemcpy(err, dErr, pointCount * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    cudaFree(dPrev);
    cudaFree(dNext);
    cudaFree(dStatus);
    if (dErr != nullptr) cudaFree(dErr);
    if (rc != cudaSuccess) {
        BINCV_THROW(std::runtime_error,
                    std::string("cuda::calcOpticalFlowPyrLK: ") + cudaGetErrorString(rc));
    }
}

const char* lkPathName() {
    // NO CACHE. The answer depends on the switches BELOW, and a cached first
    // reading would tell a caller who flipped one that the old arm is still
    // running -- which is precisely the misreport this function exists to
    // prevent. It costs one one-thread launch and one synchronize per call, so
    // it is called outside a timed region, and both callers in this repository
    // do.
    static char buf[128];
    unsigned* dMask = nullptr;
    unsigned mask = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&dMask), sizeof(unsigned)) != cudaSuccess) {
        return "lk: arm probe unavailable (no device allocation)";
    }
    armProbeKernel<<<1, 1>>>(impl::lkWarpReduceIntrinsicEnabled(), impl::lkTapCacheEnabled(),
                             dMask);
    if (cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(&mask, dMask, sizeof(unsigned), cudaMemcpyDeviceToHost);
    }
    cudaFree(dMask);
    std::snprintf(buf, sizeof(buf), "lk: %s + %s + %s",
                  (mask & kArmWarpPerPoint) ? "warp-per-keypoint" : "NO TRAVERSAL",
                  (mask & kArmReduceIntrinsic) ? "reduce_add_sync" : "shuffle-tree",
                  (mask & kArmTapCache) ? "tap-cache" : "no-tap-cache");
    return buf;
}

namespace impl {

bool& lkWarpReduceIntrinsicEnabled() {
    static bool on = true;
    return on;
}

bool& lkTapCacheEnabled() {
    static bool on = true;
    return on;
}

cudaError_t lkResidualSumsProbeAsync(const DeviceLKLevel& lv, Rect window, long long tapX,
                                     long long tapY, DeviceTapSums* dOut,
                                     cudaStream_t stream) {
    if (dOut == nullptr || !levelInDomain(lv)) return cudaErrorInvalidValue;
    const bool intrinsic = lkWarpReduceIntrinsicEnabled();
    if (lv.bits == 1u) {
        residualProbeKernel<1><<<1, 32, 0, stream>>>(lv, window, tapX, tapY, intrinsic, dOut);
    } else {
        residualProbeKernel<2><<<1, 32, 0, stream>>>(lv, window, tapX, tapY, intrinsic, dOut);
    }
    return cudaGetLastError();
}

cudaError_t lkCovarianceProbeAsync(const DeviceLKLevel& lv, Rect window,
                                   DeviceGradientCovariance* dOut, cudaStream_t stream) {
    if (dOut == nullptr || !levelInDomain(lv)) return cudaErrorInvalidValue;
    const bool intrinsic = lkWarpReduceIntrinsicEnabled();
    if (lv.bits == 1u) {
        covarianceProbeKernel<1><<<1, 32, 0, stream>>>(lv, window, intrinsic, dOut);
    } else {
        covarianceProbeKernel<2><<<1, 32, 0, stream>>>(lv, window, intrinsic, dOut);
    }
    return cudaGetLastError();
}

cudaError_t lkFmaGuardProbeAsync(double a, double b, double c, double d, double* dOut,
                                 cudaStream_t stream) {
    if (dOut == nullptr) return cudaErrorInvalidValue;
    fmaGuardKernel<<<1, 1, 0, stream>>>(a, b, c, d, dOut);
    return cudaGetLastError();
}

} // namespace impl

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
