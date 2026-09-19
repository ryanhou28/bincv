// Descriptor matching, sparse rectified stereo and Hamming block matching on
// the device. Five ops, two arms each, one binary.
//
// ===========================================================================
// WHERE THE WORK ACTUALLY IS, AND WHAT THAT DICTATED
//
// Two different shapes, because the two halves of this family have different
// bottlenecks and pretending otherwise would give one of them the other's
// kernel.
//
//   * THE MATCHER is Q x T independent Hamming distances, and the only traffic
//     that scales with the product is the TRAIN descriptors. So a block owns a
//     TILE of queries, stages them in shared memory, and streams the train set
//     once for the whole tile: one train word load feeds `tile` XOR-popcount
//     pairs instead of one. The tile width is a runtime switch here rather
//     than a constant inherited from somebody else's kernel, because the
//     backend's own record says a tile width is measured, not assumed.
//
//   * THE WINDOW OPS evaluate one window at many SHIFTS of the same rows.
//     Rebuilding the displaced row per candidate re-derives identical bits --
//     two loads, two shifts, an or and two mask-selects, per row, per
//     candidate -- so each row's 96 covering columns are assembled ONCE into a
//     RowSpan and every candidate is one `__funnelshift_r` out of it. That is
//     the dense matcher's recorded lesson (pay a row twice, not winHeight
//     times) turned along the candidate axis.
//
// ===========================================================================
// THE TIE RULES ARE REPRODUCED, NOT APPROXIMATED
//
// Every search here is "strictly less, scanning ascending", so every argmin is
// a plain 64-bit minimum over `(cost << 32) | candidateIndex` and the host's
// first-wins rule falls out of the ordering instead of out of an argument
// about traversal order. Two places need care and get it:
//
//   * `stereoRefineDisparity` accepts a NEGATIVE minDisparity (only the coarse
//     stage asserts non-negativity), so the low half packs `d - lo`, not `d`.
//   * block matching's candidate index is the host's own scan order --
//     `(dy + R) * (2R + 1) + (dx + R)`, dy outer, dx inner -- which is what
//     makes a flat plateau resolve to its top-left the way the host header
//     says it does.
//
// ===========================================================================
// WHAT IS SHARED WITH THE HOST AND WHAT IS FORKED
//
// Shared, by call and not by copy: `impl::clipRegion` and its head/tail masks,
// `impl::ReplicatedShiftedRow`'s `word()` / `sourceWord()`, `impl::edgeFill`,
// `impl::floorToLL`, `impl::parabolicOffset`. Forked: the traversals, and the
// builder that fills a shifted row from a DEVICE view (the host builder takes
// a host view, and the two view types are distinct on purpose).

#include "bincv/cuda/sparseMatch.cuh"
#include "bincv/cuda/sparseMatch.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {
namespace {

using bincv::impl::clipRegion;
using bincv::impl::floorToLL;
using bincv::impl::lowBitsMask;
using bincv::impl::parabolicOffset;

// ---------------------------------------------------------------------------
// Shapes. Each is recorded with the number that chose it in the benchmark; a
// constant inherited from another kernel is the thing this backend's report
// explicitly says not to do.
// ---------------------------------------------------------------------------

constexpr unsigned kMatchBlock = 128;      ///< threads per matcher block (4 warps)
constexpr unsigned kMatchWarps = kMatchBlock / 32;
constexpr unsigned kRefBlock = 128;        ///< threads per reference-arm block
constexpr unsigned kStereoBlock = 128;     ///< threads per coarse-stereo block
constexpr unsigned kStereoWarps = kStereoBlock / 32;
constexpr unsigned kWarpsPerWindowBlock = 4;  ///< keypoints per window-op block

/// @brief The largest tile width instantiated. Shared memory is sized from it,
/// so it bounds the staging rather than the tile width bounding it.
constexpr unsigned kMaxQueryTile = 8;

// ---------------------------------------------------------------------------
// The admission gate, spelled once
// ---------------------------------------------------------------------------

/// @brief `matchDescriptorsGated`'s predicate, in the host's own order.
/// @note The float comparisons are the host's four, not `fabsf(dx) <= maxDx`:
/// a NaN position must fall through both of them and be ADMITTED, which is
/// what the host does, and an absolute value would reject it instead.
__device__ __forceinline__ bool gateAdmits(float qx, float qy, int qo, float tx, float ty,
                                           int to, float maxDx, float maxDy, bool hasOctave,
                                           int maxOctaveDelta) {
    const float dx = tx - qx;
    const float dy = ty - qy;
    if (dx > maxDx || dx < -maxDx || dy > maxDy || dy < -maxDy) return false;
    if (hasOctave) {
        const int od = qo - to;
        if (od > maxOctaveDelta || od < -maxOctaveDelta) return false;
    }
    return true;
}

/// @brief Everything the gate needs, so the two kernels take one argument
/// instead of six and the ungated instantiation drops them all.
struct GateArgs {
    const float* queryXY = nullptr;
    const float* trainXY = nullptr;
    const int32_t* queryOctave = nullptr;
    const int32_t* trainOctave = nullptr;
    float maxDx = 0.0f;
    float maxDy = 0.0f;
    int maxOctaveDelta = 0;
};

// ---------------------------------------------------------------------------
// OP 1 / 2: the matcher, reference arm
//
// One thread per query, a serial scan over the whole train set: the host loop
// transcribed, including `best` / `second` / `bestIdx` and the `valid`
// predicate. This is what the runtime switch forces and what the suite
// compares against.
// ---------------------------------------------------------------------------

template <bool Gated>
__global__ void matchRefKernel(DeviceDescriptorSetConstView query,
                               DeviceDescriptorSetConstView train, GateArgs gate,
                               unsigned maxRatio, DeviceDescriptorMatch* out) {
    const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= query.count) return;

    const uint32_t words = query.wordsPerDescriptor;
    const uint32_t* qd = query.descriptor(q);
    float qx = 0.0f, qy = 0.0f;
    int qo = 0;
    if (Gated) {
        qx = gate.queryXY[2 * static_cast<size_t>(q)];
        qy = gate.queryXY[2 * static_cast<size_t>(q) + 1];
        if (gate.queryOctave != nullptr) qo = gate.queryOctave[q];
    }

    BestTwo v;
    for (uint32_t t = 0; t < train.count; ++t) {
        if (Gated) {
            const float tx = gate.trainXY[2 * static_cast<size_t>(t)];
            const float ty = gate.trainXY[2 * static_cast<size_t>(t) + 1];
            const int to = gate.trainOctave != nullptr ? gate.trainOctave[t] : 0;
            if (!gateAdmits(qx, qy, qo, tx, ty, to, gate.maxDx, gate.maxDy,
                            gate.queryOctave != nullptr, gate.maxOctaveDelta))
                continue;
        }
        v.offer(hammingDistanceWords(qd, train.descriptor(t), words), t);
    }
    out[q] = finishMatch(v, maxRatio);
}

// ---------------------------------------------------------------------------
// OP 1 / 2: the matcher, tiled arm
//
// A block owns `Tile` consecutive queries, staged in shared memory, and its
// 128 threads stride the train set. ONE train word load feeds `Tile` XOR +
// popcount pairs, which is the only traffic in the kernel that scales as
// Q x T.
//
// The per-thread state is a (best, second) PAIR per staged query, merged
// pairwise. That is exact rather than nearly right because the host's `second`
// is the second ORDER STATISTIC of the distance multiset -- see BestTwo in
// sparseMatch.cuh, which traces the host loop for it.
//
// NO EARLY RETURN inside the block. Every thread reaches the warp shuffles,
// including threads whose strided loop ran zero times, because a lane missing
// from `__shfl_xor_sync(0xFFFFFFFF, ...)` makes the reduction undefined --
// which produces a plausible match rather than a crash.
// ---------------------------------------------------------------------------

template <bool Gated, unsigned Tile>
__global__ void matchTiledKernel(DeviceDescriptorSetConstView query,
                                 DeviceDescriptorSetConstView train, GateArgs gate,
                                 unsigned maxRatio, DeviceDescriptorMatch* out) {
    __shared__ uint32_t sQuery[kMaxQueryTile * kMatchTileMaxWords];
    __shared__ float sQx[kMaxQueryTile];
    __shared__ float sQy[kMaxQueryTile];
    __shared__ int sQo[kMaxQueryTile];
    __shared__ uint64_t sBest[kMaxQueryTile][kMatchWarps];
    __shared__ uint32_t sSecond[kMaxQueryTile][kMatchWarps];

    const uint32_t q0 = blockIdx.x * Tile;
    if (q0 >= query.count) return;   // block-uniform: a whole block leaves

    const uint32_t words = query.wordsPerDescriptor;
    const uint32_t staged = Tile * words;
    for (uint32_t i = threadIdx.x; i < staged; i += kMatchBlock) {
        const uint32_t j = i / words;
        const uint32_t w = i - j * words;
        const uint32_t qi = q0 + j;
        sQuery[j * kMatchTileMaxWords + w] =
            qi < query.count ? query.descriptor(qi)[w] : 0u;
    }
    if (threadIdx.x < Tile) {
        const uint32_t qi = q0 + threadIdx.x;
        const bool live = qi < query.count;
        // A slot past the end of the query set is staged with a position that
        // admits nothing rather than left unread: its record is never written,
        // but a wild position would make the gate load descriptors for it.
        sQx[threadIdx.x] = (Gated && live) ? gate.queryXY[2 * static_cast<size_t>(qi)] : 0.0f;
        sQy[threadIdx.x] =
            (Gated && live) ? gate.queryXY[2 * static_cast<size_t>(qi) + 1] : 0.0f;
        sQo[threadIdx.x] =
            (Gated && live && gate.queryOctave != nullptr) ? gate.queryOctave[qi] : 0;
    }
    __syncthreads();

    BestTwo v[Tile];
    const bool hasOctave = Gated && gate.queryOctave != nullptr;

    for (uint32_t t = threadIdx.x; t < train.count; t += kMatchBlock) {
        bool admit[Tile];
        if (Gated) {
            const float tx = gate.trainXY[2 * static_cast<size_t>(t)];
            const float ty = gate.trainXY[2 * static_cast<size_t>(t) + 1];
            const int to = gate.trainOctave != nullptr ? gate.trainOctave[t] : 0;
            bool any = false;
#pragma unroll
            for (unsigned j = 0; j < Tile; ++j) {
                admit[j] = gateAdmits(sQx[j], sQy[j], sQo[j], tx, ty, to, gate.maxDx,
                                      gate.maxDy, hasOctave, gate.maxOctaveDelta);
                any = any || admit[j];
            }
            // The gate's whole traffic claim, in one branch: a candidate no
            // query in the tile admits costs no descriptor word.
            if (!any) continue;
        } else {
#pragma unroll
            for (unsigned j = 0; j < Tile; ++j) admit[j] = true;
        }

        uint32_t d[Tile];
#pragma unroll
        for (unsigned j = 0; j < Tile; ++j) d[j] = 0;

        const uint32_t* td = train.descriptor(t);
        for (uint32_t w = 0; w < words; ++w) {
            const uint32_t tw = __ldg(td + w);
#pragma unroll
            for (unsigned j = 0; j < Tile; ++j)
                if (admit[j]) d[j] += static_cast<uint32_t>(
                                   __popc(tw ^ sQuery[j * kMatchTileMaxWords + w]));
        }
#pragma unroll
        for (unsigned j = 0; j < Tile; ++j)
            if (admit[j]) v[j].offer(d[j], t);
    }

    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
#pragma unroll
    for (unsigned j = 0; j < Tile; ++j) {
        const BestTwo m = warpMergeBestTwo(v[j]);
        if (lane == 0) {
            sBest[j][warp] = m.best;
            sSecond[j][warp] = m.second;
        }
    }
    __syncthreads();

    if (threadIdx.x < Tile) {
        const uint32_t qi = q0 + threadIdx.x;
        if (qi < query.count) {
            BestTwo m;
            m.best = sBest[threadIdx.x][0];
            m.second = sSecond[threadIdx.x][0];
            for (unsigned wp = 1; wp < kMatchWarps; ++wp) {
                BestTwo o;
                o.best = sBest[threadIdx.x][wp];
                o.second = sSecond[threadIdx.x][wp];
                m.merge(o);
            }
            out[qi] = finishMatch(m, maxRatio);
        }
    }
}

// ---------------------------------------------------------------------------
// OP 3: the coarse sparse-stereo stage
//
// One block per LEFT keypoint; its threads stride the right set. The row-band
// and disparity-range gate is evaluated BEFORE any descriptor word is read,
// which is the only lever this stage has -- and the left descriptor is staged
// once in shared memory so the Q-side read is not repeated per candidate.
//
// No ratio test, and the no-candidate sentinel is excluded EXPLICITLY rather
// than by the Hamming threshold: a caller who raises maxHamming to "accept
// everything" must still get "no candidate" and not a match fabricated from
// right keypoint 0.
// ---------------------------------------------------------------------------

struct StereoCoarseArgs {
    float rowTol;
    float minD;
    float maxD;
    unsigned maxHamming;
};

__global__ void stereoCoarseRefKernel(DeviceKeypointSetConstView leftPts,
                                      DeviceDescriptorSetConstView leftDesc,
                                      DeviceKeypointSetConstView rightPts,
                                      DeviceDescriptorSetConstView rightDesc,
                                      StereoCoarseArgs a, DeviceStereoMatch* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftPts.count) return;

    const float xL = leftPts.x(i);
    const float yL = leftPts.y(i);
    const uint32_t words = leftDesc.wordsPerDescriptor;
    const uint32_t* ld = leftDesc.descriptor(i);

    uint64_t best = kNoCandidate;
    for (uint32_t j = 0; j < rightPts.count; ++j) {
        const float dy = rightPts.y(j) - yL;
        if (dy > a.rowTol || dy < -a.rowTol) continue;
        const float d = xL - rightPts.x(j);
        if (d < a.minD || d > a.maxD) continue;
        const uint64_t p = packCost(hammingDistanceWords(ld, rightDesc.descriptor(j), words), j);
        if (p < best) best = p;
    }

    DeviceStereoMatch m;
    m.disparity = 0.0f;
    m.distance = 0;
    m.rightIndex = 0;
    m.valid = 0;
    if (best != kNoCandidate && packedCost(best) <= a.maxHamming) {
        const uint32_t bestIdx = packedIndex(best);
        m.disparity = xL - rightPts.x(bestIdx);
        m.distance = packedCost(best);
        m.rightIndex = bestIdx;
        m.valid = 1;
    }
    out[i] = m;
}

__global__ void stereoCoarseFastKernel(DeviceKeypointSetConstView leftPts,
                                       DeviceDescriptorSetConstView leftDesc,
                                       DeviceKeypointSetConstView rightPts,
                                       DeviceDescriptorSetConstView rightDesc,
                                       StereoCoarseArgs a, DeviceStereoMatch* out) {
    __shared__ uint32_t sLeft[kMatchTileMaxWords];
    __shared__ uint64_t sBest[kStereoWarps];

    const uint32_t i = blockIdx.x;
    if (i >= leftPts.count) return;

    const uint32_t words = leftDesc.wordsPerDescriptor;
    for (uint32_t w = threadIdx.x; w < words; w += kStereoBlock)
        sLeft[w] = leftDesc.descriptor(i)[w];
    __syncthreads();

    const float xL = leftPts.x(i);
    const float yL = leftPts.y(i);

    uint64_t best = kNoCandidate;
    for (uint32_t j = threadIdx.x; j < rightPts.count; j += kStereoBlock) {
        const float dy = rightPts.y(j) - yL;
        if (dy > a.rowTol || dy < -a.rowTol) continue;
        const float d = xL - rightPts.x(j);
        if (d < a.minD || d > a.maxD) continue;
        const uint32_t* rd = rightDesc.descriptor(j);
        uint32_t dist = 0;
        for (uint32_t w = 0; w < words; ++w)
            dist += static_cast<uint32_t>(__popc(__ldg(rd + w) ^ sLeft[w]));
        const uint64_t p = packCost(dist, j);
        if (p < best) best = p;
    }

    const uint64_t warpBest = warpMinPacked(best);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) sBest[warp] = warpBest;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t b = sBest[0];
        for (unsigned wp = 1; wp < kStereoWarps; ++wp)
            if (sBest[wp] < b) b = sBest[wp];
        DeviceStereoMatch m;
        m.disparity = 0.0f;
        m.distance = 0;
        m.rightIndex = 0;
        m.valid = 0;
        if (b != kNoCandidate && packedCost(b) <= a.maxHamming) {
            const uint32_t bestIdx = packedIndex(b);
            m.disparity = xL - rightPts.x(bestIdx);
            m.distance = packedCost(b);
            m.rightIndex = bestIdx;
            m.valid = 1;
        }
        out[i] = m;
    }
}

// ---------------------------------------------------------------------------
// The window traversal, both shapes
// ---------------------------------------------------------------------------

/// @brief One window's Hamming cost, the general way: every region word of
/// every region row, head and tail masked. The REFERENCE arms' body, and the
/// transcription of `impl::hammingAt`'s traversal.
__device__ uint32_t windowCostGeneral(DeviceBinMatConstView prev, DeviceBinMatConstView next,
                                      const DeviceRegion& r, long long tapX, long long tapY) {
    uint32_t cost = 0;
    for (size_t y = r.y0; y < r.y1; ++y) {
        const uint32_t* ip = prev.row(y);
        const DeviceShiftedRow row =
            deviceDisplacedRow(next, static_cast<long long>(y) + tapY, tapX);
        if (r.firstWord == r.lastWord) {
            const uint32_t m = r.headMask & r.tailMask;
            cost += static_cast<uint32_t>(__popc((ip[r.firstWord] ^ row.word(r.firstWord)) & m));
            continue;
        }
        cost += static_cast<uint32_t>(
            __popc((ip[r.firstWord] ^ row.word(r.firstWord)) & r.headMask));
        for (size_t i = r.firstWord + 1; i < r.lastWord; ++i)
            cost += static_cast<uint32_t>(__popc(ip[i] ^ row.word(i)));
        cost += static_cast<uint32_t>(
            __popc((ip[r.lastWord] ^ row.word(r.lastWord)) & r.tailMask));
    }
    return cost;
}

/// @brief One window's Hamming cost with the window in ONE word per row and
/// one lane per row -- the fast arms' body for a single tap.
/// @note `leftWord` is the lane's own row of the previous frame, already
/// aligned to bit 0; `active` is false for the lanes past the region's row
/// count, and they contribute zero rather than leaving, because every lane
/// must reach the warp reduction.
__device__ __forceinline__ uint32_t windowCostLane(DeviceBinMatConstView next, long long y,
                                                   long long x0, long long tapX,
                                                   long long tapY, uint32_t leftWord,
                                                   uint32_t mask, bool active) {
    uint32_t c = 0;
    if (active) {
        const uint32_t rw = windowRowWord(next, y + tapY, x0 + tapX);
        c = static_cast<uint32_t>(__popc((leftWord ^ rw) & mask));
    }
    return warpSum(c);
}

// ---------------------------------------------------------------------------
// OP 4: stereo refinement
//
// One WARP per keypoint, one LANE per window row, and the candidate loop
// inside the lane's registers. Every lane assembles its row's 96 covering
// columns ONCE (a RowSpan) and cuts each of the 2R+1 candidates out of it with
// a single funnel shift, so the 2R+1 disparities cost one row assembly rather
// than 2R+1.
//
// The parabola's two extra scores sit one column outside that span at each
// end, which is why the span is built over [lo - 1, hi + 1] when subPixel is
// on: the neighbours are part of the plan, not an afterthought that walks off
// the cached columns.
// ---------------------------------------------------------------------------

struct StereoRefineArgs {
    int winW;
    int winH;
    int refineRadius;
    int minDisparity;
    int maxDisparity;
    bool subPixel;
};

/// @brief The scalars both refinement arms derive per keypoint, so the two
/// cannot disagree about the window, the range or the empty cases.
struct RefineSetup {
    DeviceRegion region;
    long long lo;
    long long hi;
    bool ok;
};

__device__ __forceinline__ RefineSetup refineSetup(DeviceBinMatConstView left,
                                                   DeviceKeypointSetConstView leftPts,
                                                   uint32_t i, float disparity,
                                                   const StereoRefineArgs& a) {
    RefineSetup s;
    s.lo = 0;
    s.hi = 0;
    s.ok = false;
    const float halfWinX = static_cast<float>(a.winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(a.winH - 1) * 0.5f;
    const long long anchorX = floorToLL(leftPts.x(i) - halfWinX);
    const long long anchorY = floorToLL(leftPts.y(i) - halfWinY);
    const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), a.winW, a.winH);
    s.region = clipRegion<uint32_t>(left.width, left.height, window);
    if (s.region.isEmpty) return s;

    const long long d0 = floorToLL(disparity + 0.5f);
    long long lo = d0 - a.refineRadius;
    long long hi = d0 + a.refineRadius;
    const long long minD = static_cast<long long>(a.minDisparity);
    const long long maxD = static_cast<long long>(a.maxDisparity);
    if (lo < minD) lo = minD;
    if (hi > maxD) hi = maxD;
    if (lo > hi) return s;
    s.lo = lo;
    s.hi = hi;
    s.ok = true;
    return s;
}

/// @brief The host's final clamp and store, spelled once for both arms.
__device__ __forceinline__ float refineFinish(long long bestD, double sub, int minDisparity,
                                              int maxDisparity) {
    double refined = static_cast<double>(bestD) + sub;
    const double minD = static_cast<double>(minDisparity);
    const double maxD = static_cast<double>(maxDisparity);
    if (refined < minD) refined = minD;
    if (refined > maxD) refined = maxD;
    return static_cast<float>(refined);
}

__global__ void stereoRefineRefKernel(DeviceBinMatConstView left, DeviceBinMatConstView right,
                                      DeviceKeypointSetConstView leftPts, StereoRefineArgs a,
                                      DeviceStereoMatch* inout) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leftPts.count) return;
    if (inout[i].valid == 0u) return;

    const RefineSetup s = refineSetup(left, leftPts, i, inout[i].disparity, a);
    if (!s.ok) {
        inout[i].valid = 0;
        return;
    }

    uint32_t bestCost = 0xFFFFFFFFu;
    long long bestD = s.lo;
    for (long long d = s.lo; d <= s.hi; ++d) {
        const uint32_t cost = windowCostGeneral(left, right, s.region, -d, 0);
        if (cost < bestCost) {
            bestCost = cost;
            bestD = d;
        }
    }

    double sub = 0.0;
    if (a.subPixel) {
        const uint32_t cm = windowCostGeneral(left, right, s.region, -(bestD - 1), 0);
        const uint32_t cp = windowCostGeneral(left, right, s.region, -(bestD + 1), 0);
        sub = parabolicOffset(static_cast<long long>(cm), static_cast<long long>(bestCost),
                              static_cast<long long>(cp));
    }
    inout[i].disparity = refineFinish(bestD, sub, a.minDisparity, a.maxDisparity);
}

__global__ void stereoRefineFastKernel(DeviceBinMatConstView left, DeviceBinMatConstView right,
                                       DeviceKeypointSetConstView leftPts, StereoRefineArgs a,
                                       DeviceStereoMatch* inout) {
    const unsigned lane = threadIdx.x;
    const uint32_t i = blockIdx.x * blockDim.y + threadIdx.y;
    // WARP-UNIFORM. Every predicate that can make a lane leave depends on
    // blockIdx and threadIdx.y alone, so a whole warp leaves or a whole warp
    // stays -- a lane missing from the reductions below would make them
    // undefined and produce a plausible disparity.
    if (i >= leftPts.count) return;
    if (inout[i].valid == 0u) return;

    const RefineSetup s = refineSetup(left, leftPts, i, inout[i].disparity, a);
    if (!s.ok) {
        if (lane == 0) inout[i].valid = 0;
        return;
    }

    const long long x0 = static_cast<long long>(s.region.x0);
    const uint32_t w = static_cast<uint32_t>(s.region.x1 - s.region.x0);
    const uint32_t mask = lowBitsMask<uint32_t>(w);
    const uint32_t nRows = static_cast<uint32_t>(s.region.y1 - s.region.y0);
    const bool active = lane < nRows;
    const long long y = static_cast<long long>(s.region.y0) + lane;

    // The candidate span runs from `lo` to `hi`, widened by one at each end
    // when the parabola's two neighbours will be asked for -- so every tap this
    // keypoint can need is inside the cached 96 columns rather than walking off
    // them. The span's own width is what the launcher's gate tested.
    const long long dHigh = a.subPixel ? s.hi + 1 : s.hi;
    const long long base = x0 - dHigh;

    const uint32_t leftWord = active ? windowRowWord(left, y, x0) : 0u;
    RowSpan span;
    if (active) span = loadRowSpan(right, y, base);

    uint64_t best = kNoCandidate;
    for (long long d = s.lo; d <= s.hi; ++d) {
        const unsigned p = static_cast<unsigned>(dHigh - d);
        uint32_t c = 0;
        if (active) c = static_cast<uint32_t>(__popc((leftWord ^ span.run(p)) & mask));
        const uint32_t cost = warpSum(c);
        // `d - lo`, not `d`: minDisparity may be negative here, and a negative
        // value in the packed low half inverts the ordering.
        const uint64_t q = packCost(cost, static_cast<uint32_t>(d - s.lo));
        if (q < best) best = q;
    }

    const long long bestD = s.lo + static_cast<long long>(packedIndex(best));
    const uint32_t bestCost = packedCost(best);

    double sub = 0.0;
    if (a.subPixel) {
        uint32_t cmLane = 0, cpLane = 0;
        if (active) {
            cmLane = static_cast<uint32_t>(
                __popc((leftWord ^ span.run(static_cast<unsigned>(dHigh - (bestD - 1)))) & mask));
            cpLane = static_cast<uint32_t>(
                __popc((leftWord ^ span.run(static_cast<unsigned>(dHigh - (bestD + 1)))) & mask));
        }
        const uint32_t cm = warpSum(cmLane);
        const uint32_t cp = warpSum(cpLane);
        sub = parabolicOffset(static_cast<long long>(cm), static_cast<long long>(bestCost),
                              static_cast<long long>(cp));
    }
    if (lane == 0)
        inout[i].disparity = refineFinish(bestD, sub, a.minDisparity, a.maxDisparity);
}

// ---------------------------------------------------------------------------
// OP 5: block matching, route (a)
//
// REFERENCE ARM: one thread per keypoint, ONE launch, the whole ladder carried
// by value and the coarse-to-fine estimate in registers -- the host function
// transcribed. It is the reason the fast arm's per-level launches can be
// checked at all: the two differ in WHERE the estimate lives and nowhere else.
//
// FAST ARM: one launch per pyramid level, one warp per keypoint, one lane per
// window row. Within a level the 2R+1 horizontal candidates at a fixed dy come
// out of ONE RowSpan per row.
// ---------------------------------------------------------------------------

struct BlockMatchArgs {
    int winW;
    int winH;
    int radius;
    bool subPixel;
};

struct BlockMatchLadder {
    DeviceBlockMatchLevel lv[kMaxBlockMatchLevels];
};

__global__ void blockMatchRefKernel(BlockMatchLadder ladder, unsigned usableLevels,
                                    const Point2f* prevPts, Point2f* nextPts,
                                    uint8_t* status, uint32_t pointCount,
                                    BlockMatchArgs a) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pointCount) return;

    nextPts[p] = prevPts[p];
    status[p] = 1;
    if (usableLevels == 0) {
        status[p] = 0;
        return;
    }

    const float halfWinX = static_cast<float>(a.winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(a.winH - 1) * 0.5f;

    long long estX = 0, estY = 0;
    double subX = 0.0, subY = 0.0;
    bool lost = false;

    for (unsigned li = usableLevels; li-- > 0;) {
        const DeviceBlockMatchLevel& lv = ladder.lv[li];
        const float scale = 1.0f / static_cast<float>(1u << li);
        if (li + 1 != usableLevels) {
            estX *= 2;
            estY *= 2;
        }
        const float prevX = prevPts[p].x * scale - halfWinX;
        const float prevY = prevPts[p].y * scale - halfWinY;
        const long long anchorX = floorToLL(prevX);
        const long long anchorY = floorToLL(prevY);
        const Rect window(static_cast<int>(anchorX), static_cast<int>(anchorY), a.winW,
                          a.winH);
        const DeviceRegion region =
            clipRegion<uint32_t>(lv.width(), lv.height(), window);
        if (region.isEmpty) {
            lost = true;
            break;
        }

        uint32_t bestCost = 0xFFFFFFFFu;
        long long bestDx = 0, bestDy = 0;
        for (long long dy = -a.radius; dy <= a.radius; ++dy) {
            for (long long dx = -a.radius; dx <= a.radius; ++dx) {
                const uint32_t cost =
                    windowCostGeneral(lv.prev, lv.next, region, estX + dx, estY + dy);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestDx = dx;
                    bestDy = dy;
                }
            }
        }
        estX += bestDx;
        estY += bestDy;

        if (li == 0 && a.subPixel) {
            const long long c0 = windowCostGeneral(lv.prev, lv.next, region, estX, estY);
            const long long cxm = windowCostGeneral(lv.prev, lv.next, region, estX - 1, estY);
            const long long cxp = windowCostGeneral(lv.prev, lv.next, region, estX + 1, estY);
            const long long cym = windowCostGeneral(lv.prev, lv.next, region, estX, estY - 1);
            const long long cyp = windowCostGeneral(lv.prev, lv.next, region, estX, estY + 1);
            subX = parabolicOffset(cxm, c0, cxp);
            subY = parabolicOffset(cym, c0, cyp);
        }
    }

    if (lost) {
        status[p] = 0;
        return;
    }
    nextPts[p].x = prevPts[p].x + static_cast<float>(static_cast<double>(estX) + subX);
    nextPts[p].y = prevPts[p].y + static_cast<float>(static_cast<double>(estY) + subY);
}

__global__ void blockMatchLevelKernel(DeviceBlockMatchLevel lv, unsigned li,
                                      unsigned usableLevels, const Point2f* prevPts,
                                      Point2f* nextPts, uint8_t* status, uint32_t pointCount,
                                      DeviceBlockMatchState* state, BlockMatchArgs a) {
    const unsigned lane = threadIdx.x;
    const uint32_t p = blockIdx.x * blockDim.y + threadIdx.y;
    if (p >= pointCount) return;   // warp-uniform

    // The coarsest level owns the initialization, so the caller needs no
    // separate memset and the scratch has no state before the first launch.
    DeviceBlockMatchState st;
    if (li + 1 == usableLevels) {
        st.estX = 0;
        st.estY = 0;
        st.lost = 0;
        st.pad = 0;
    } else {
        st = state[p];
        st.estX *= 2;
        st.estY *= 2;
    }

    if (st.lost != 0u) {
        if (lane == 0) {
            state[p] = st;
            if (li == 0) {
                nextPts[p] = prevPts[p];
                status[p] = 0;
            }
        }
        return;
    }

    const float halfWinX = static_cast<float>(a.winW - 1) * 0.5f;
    const float halfWinY = static_cast<float>(a.winH - 1) * 0.5f;
    const float scale = 1.0f / static_cast<float>(1u << li);
    const float prevX = prevPts[p].x * scale - halfWinX;
    const float prevY = prevPts[p].y * scale - halfWinY;
    const Rect window(static_cast<int>(floorToLL(prevX)), static_cast<int>(floorToLL(prevY)),
                      a.winW, a.winH);
    const DeviceRegion region = clipRegion<uint32_t>(lv.width(), lv.height(), window);
    if (region.isEmpty) {
        st.lost = 1;
        if (lane == 0) {
            state[p] = st;
            if (li == 0) {
                nextPts[p] = prevPts[p];
                status[p] = 0;
            }
        }
        return;
    }

    const long long x0 = static_cast<long long>(region.x0);
    const uint32_t w = static_cast<uint32_t>(region.x1 - region.x0);
    const uint32_t mask = lowBitsMask<uint32_t>(w);
    const uint32_t nRows = static_cast<uint32_t>(region.y1 - region.y0);
    const bool active = lane < nRows;
    const long long y = static_cast<long long>(region.y0) + lane;
    const uint32_t leftWord = active ? windowRowWord(lv.prev, y, x0) : 0u;

    const long long estX = st.estX;
    const long long estY = st.estY;
    const long long R = a.radius;
    const uint32_t stride = static_cast<uint32_t>(2 * R + 1);

    uint64_t best = kNoCandidate;
    for (long long dy = -R; dy <= R; ++dy) {
        // ONE row assembly for the whole row of candidates: the 2R+1 shifts at
        // this dy are funnel shifts out of these 96 columns.
        RowSpan span;
        if (active) span = loadRowSpan(lv.next, y + estY + dy, x0 + estX - R);
        for (long long dx = -R; dx <= R; ++dx) {
            uint32_t c = 0;
            if (active)
                c = static_cast<uint32_t>(
                    __popc((leftWord ^ span.run(static_cast<unsigned>(dx + R))) & mask));
            const uint32_t cost = warpSum(c);
            // The host's own scan order -- dy outer, dx inner, both from
            // -radius -- so the packed minimum keeps the first candidate in
            // that order and a flat plateau resolves to its top-left.
            const uint32_t cand =
                static_cast<uint32_t>(dy + R) * stride + static_cast<uint32_t>(dx + R);
            const uint64_t q = packCost(cost, cand);
            if (q < best) best = q;
        }
    }

    const uint32_t cand = packedIndex(best);
    const long long bestDy = static_cast<long long>(cand / stride) - R;
    const long long bestDx = static_cast<long long>(cand - (cand / stride) * stride) - R;
    const long long newX = estX + bestDx;
    const long long newY = estY + bestDy;

    double subX = 0.0, subY = 0.0;
    if (li == 0 && a.subPixel) {
        // The host recomputes c0 rather than reusing the search minimum, and
        // so does this: at a clipped border the two are the same number, but
        // reusing it would tie the parabola to the search's packing.
        const uint32_t c0 = windowCostLane(lv.next, y, x0, newX, newY, leftWord, mask, active);
        const uint32_t cxm =
            windowCostLane(lv.next, y, x0, newX - 1, newY, leftWord, mask, active);
        const uint32_t cxp =
            windowCostLane(lv.next, y, x0, newX + 1, newY, leftWord, mask, active);
        const uint32_t cym =
            windowCostLane(lv.next, y, x0, newX, newY - 1, leftWord, mask, active);
        const uint32_t cyp =
            windowCostLane(lv.next, y, x0, newX, newY + 1, leftWord, mask, active);
        subX = parabolicOffset(static_cast<long long>(cxm), static_cast<long long>(c0),
                               static_cast<long long>(cxp));
        subY = parabolicOffset(static_cast<long long>(cym), static_cast<long long>(c0),
                               static_cast<long long>(cyp));
    }

    if (lane == 0) {
        st.estX = static_cast<int32_t>(newX);
        st.estY = static_cast<int32_t>(newY);
        state[p] = st;
        if (li == 0) {
            Point2f out;
            out.x = prevPts[p].x + static_cast<float>(static_cast<double>(newX) + subX);
            out.y = prevPts[p].y + static_cast<float>(static_cast<double>(newY) + subY);
            nextPts[p] = out;
            status[p] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

template <bool Gated>
cudaError_t launchMatch(DeviceDescriptorSetConstView query,
                        DeviceDescriptorSetConstView train, const GateArgs& gate,
                        unsigned maxRatio, DeviceDescriptorMatch* out, cudaStream_t stream) {
    const uint32_t words = query.wordsPerDescriptor;
    if (matchTiledArmEnabled() && words <= kMatchTileMaxWords) {
        const unsigned tile = matchQueryTile();
        const uint32_t blocks = (query.count + tile - 1u) / tile;
        switch (tile) {
            case 1:
                matchTiledKernel<Gated, 1><<<blocks, kMatchBlock, 0, stream>>>(
                    query, train, gate, maxRatio, out);
                break;
            case 4:
                matchTiledKernel<Gated, 4><<<blocks, kMatchBlock, 0, stream>>>(
                    query, train, gate, maxRatio, out);
                break;
            case 8:
            default:
                matchTiledKernel<Gated, kMatchDefaultTile><<<blocks, kMatchBlock, 0, stream>>>(
                    query, train, gate, maxRatio, out);
                break;
        }
        return cudaGetLastError();
    }
    const uint32_t blocks = (query.count + kRefBlock - 1u) / kRefBlock;
    matchRefKernel<Gated><<<blocks, kRefBlock, 0, stream>>>(query, train, gate, maxRatio, out);
    return cudaGetLastError();
}

} // namespace

// ---------------------------------------------------------------------------
// The runtime switches
// ---------------------------------------------------------------------------

bool& matchTiledArmEnabled() {
    static bool on = true;
    return on;
}

bool& sparseStereoFastArmEnabled() {
    static bool on = true;
    return on;
}

bool& blockMatchFastArmEnabled() {
    static bool on = true;
    return on;
}

unsigned& matchQueryTile() {
    static unsigned tile = kMatchDefaultTile;
    return tile;
}

// ---------------------------------------------------------------------------
// The entry points' implementations
// ---------------------------------------------------------------------------

cudaError_t matchDescriptorsImpl(DeviceDescriptorSetConstView query,
                                 DeviceKeypointSetConstView queryPts,
                                 DeviceDescriptorSetConstView train,
                                 DeviceKeypointSetConstView trainPts, bool gated,
                                 float maxDx, float maxDy, int maxOctaveDelta,
                                 unsigned maxRatio, DeviceDescriptorMatch* dOut,
                                 cudaStream_t stream) {
    if (query.count == 0) return cudaSuccess;
    BINCV_ASSERT(query.words != nullptr && dOut != nullptr,
                 "cuda matchDescriptors: null query argument");
    BINCV_ASSERT(train.count == 0 || train.words != nullptr,
                 "cuda matchDescriptors: null train argument");
    BINCV_ASSERT(query.wordsPerDescriptor == train.wordsPerDescriptor || train.count == 0,
                 "cuda matchDescriptors: the two sets must share their word pitch");
    if (query.words == nullptr || dOut == nullptr) return cudaErrorInvalidValue;
    if (train.count != 0 &&
        (train.words == nullptr || train.wordsPerDescriptor != query.wordsPerDescriptor))
        return cudaErrorInvalidValue;

    GateArgs g;
    if (gated) {
        BINCV_ASSERT(queryPts.xy != nullptr, "cuda matchDescriptorsGated: null query positions");
        BINCV_ASSERT(train.count == 0 || trainPts.xy != nullptr,
                     "cuda matchDescriptorsGated: null train positions");
        BINCV_ASSERT(queryPts.hasOctave() == trainPts.hasOctave(),
                     "cuda matchDescriptorsGated: octave arrays come as a pair or not at all");
        BINCV_ASSERT(maxDx >= 0.0f && maxDy >= 0.0f,
                     "cuda matchDescriptorsGated: the window must not be negative");
        BINCV_ASSERT(queryPts.count == query.count && trainPts.count == train.count,
                     "cuda matchDescriptorsGated: one position per descriptor");
        if (queryPts.xy == nullptr || (train.count != 0 && trainPts.xy == nullptr) ||
            queryPts.hasOctave() != trainPts.hasOctave() || maxDx < 0.0f || maxDy < 0.0f)
            return cudaErrorInvalidValue;
        g.queryXY = queryPts.xy;
        g.trainXY = trainPts.xy;
        g.queryOctave = queryPts.octave;
        g.trainOctave = trainPts.octave;
        g.maxDx = maxDx;
        g.maxDy = maxDy;
        g.maxOctaveDelta = maxOctaveDelta;
    }
    return gated ? launchMatch<true>(query, train, g, maxRatio, dOut, stream)
                 : launchMatch<false>(query, train, g, maxRatio, dOut, stream);
}

cudaError_t stereoDescriptorMatchImpl(DeviceKeypointSetConstView leftPts,
                                      DeviceDescriptorSetConstView leftDesc,
                                      DeviceKeypointSetConstView rightPts,
                                      DeviceDescriptorSetConstView rightDesc,
                                      const StereoMatchParams& params,
                                      DeviceStereoMatch* dOut, cudaStream_t stream) {
    if (leftPts.count == 0) return cudaSuccess;
    BINCV_ASSERT(leftPts.xy != nullptr && leftDesc.words != nullptr && dOut != nullptr,
                 "cuda stereoDescriptorMatch: null left argument");
    BINCV_ASSERT(rightPts.count == 0 ||
                     (rightPts.xy != nullptr && rightDesc.words != nullptr),
                 "cuda stereoDescriptorMatch: null right argument");
    BINCV_ASSERT(params.minDisparity >= 0 && params.maxDisparity > params.minDisparity,
                 "cuda stereoDescriptorMatch: need 0 <= minDisparity < maxDisparity");
    BINCV_ASSERT(params.rowTolerance >= 0, "cuda stereoDescriptorMatch: rowTolerance < 0");
    BINCV_ASSERT(leftDesc.count == leftPts.count && rightDesc.count == rightPts.count,
                 "cuda stereoDescriptorMatch: one descriptor per keypoint");
    if (leftPts.xy == nullptr || leftDesc.words == nullptr || dOut == nullptr)
        return cudaErrorInvalidValue;
    if (params.minDisparity < 0 || params.maxDisparity <= params.minDisparity ||
        params.rowTolerance < 0)
        return cudaErrorInvalidValue;
    if (rightPts.count != 0 &&
        (rightPts.xy == nullptr || rightDesc.words == nullptr ||
         rightDesc.wordsPerDescriptor != leftDesc.wordsPerDescriptor))
        return cudaErrorInvalidValue;

    StereoCoarseArgs a;
    a.rowTol = static_cast<float>(params.rowTolerance);
    a.minD = static_cast<float>(params.minDisparity);
    a.maxD = static_cast<float>(params.maxDisparity);
    a.maxHamming = params.maxHamming;

    if (sparseStereoFastArmEnabled() &&
        leftDesc.wordsPerDescriptor <= kMatchTileMaxWords) {
        stereoCoarseFastKernel<<<leftPts.count, kStereoBlock, 0, stream>>>(
            leftPts, leftDesc, rightPts, rightDesc, a, dOut);
        return cudaGetLastError();
    }
    const uint32_t blocks = (leftPts.count + kRefBlock - 1u) / kRefBlock;
    stereoCoarseRefKernel<<<blocks, kRefBlock, 0, stream>>>(leftPts, leftDesc, rightPts,
                                                            rightDesc, a, dOut);
    return cudaGetLastError();
}

cudaError_t stereoRefineDisparityImpl(DeviceBinMatConstView left, DeviceBinMatConstView right,
                                      DeviceKeypointSetConstView leftPts,
                                      const StereoMatchParams& params,
                                      DeviceStereoMatch* dInOut, cudaStream_t stream) {
    if (leftPts.count == 0) return cudaSuccess;
    BINCV_ASSERT(leftPts.xy != nullptr && dInOut != nullptr,
                 "cuda stereoRefineDisparity: null argument");
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "cuda stereoRefineDisparity: the pair must share its extent");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "cuda stereoRefineDisparity: the window must be more than 2 pixels a side");
    BINCV_ASSERT(params.refineRadius >= 1, "cuda stereoRefineDisparity: refineRadius < 1");
    if (leftPts.xy == nullptr || dInOut == nullptr) return cudaErrorInvalidValue;
    if (left.width != right.width || left.height != right.height) return cudaErrorInvalidValue;
    if (params.winWidth <= 2 || params.winHeight <= 2 || params.refineRadius < 1)
        return cudaErrorInvalidValue;

    StereoRefineArgs a;
    a.winW = params.winWidth;
    a.winH = params.winHeight;
    a.refineRadius = params.refineRadius;
    a.minDisparity = params.minDisparity;
    a.maxDisparity = params.maxDisparity;
    a.subPixel = params.subPixel;

    // The parabola's neighbours widen the span by one at each end, so the gate
    // asks about the span the kernel will actually build.
    const int span = 2 * params.refineRadius + (params.subPixel ? 2 : 0);
    if (sparseStereoFastArmEnabled() &&
        windowSpanFits(params.winWidth, params.winHeight, span)) {
        const dim3 block(32, kWarpsPerWindowBlock);
        const dim3 grid((leftPts.count + kWarpsPerWindowBlock - 1u) / kWarpsPerWindowBlock);
        stereoRefineFastKernel<<<grid, block, 0, stream>>>(left, right, leftPts, a, dInOut);
        return cudaGetLastError();
    }
    const uint32_t blocks = (leftPts.count + kRefBlock - 1u) / kRefBlock;
    stereoRefineRefKernel<<<blocks, kRefBlock, 0, stream>>>(left, right, leftPts, a, dInOut);
    return cudaGetLastError();
}

cudaError_t blockMatchImpl(const DeviceBlockMatchLevel* levels, size_t levelCount,
                           const Point2f* dPrevPts, Point2f* dNextPts, uint8_t* dStatus,
                           size_t pointCount, void* dScratch, size_t scratchBytes,
                           const BlockMatchParams& params, cudaStream_t stream) {
    if (pointCount == 0) return cudaSuccess;
    BINCV_ASSERT(dPrevPts != nullptr && dNextPts != nullptr && dStatus != nullptr,
                 "cuda blockMatch: prevPts, nextPts and status must be non-null");
    BINCV_ASSERT(params.winWidth > 2 && params.winHeight > 2,
                 "cuda blockMatch: the window must be more than 2 pixels on a side");
    BINCV_ASSERT(params.searchRadius >= 1, "cuda blockMatch: searchRadius must be at least 1");
    BINCV_ASSERT(params.searchRadius <= kMaxBlockMatchRadius,
                 "cuda blockMatch: searchRadius outside the device's documented domain");
    BINCV_ASSERT(levelCount <= kMaxBlockMatchLevels,
                 "cuda blockMatch: levelCount outside the device's documented domain");
    BINCV_ASSERT(pointCount <= kDeviceMaxKeypoints,
                 "cuda blockMatch: pointCount outside the device's uint32 keypoint domain");
    BINCV_ASSERT(levelCount == 0 || levels != nullptr, "cuda blockMatch: levels must be non-null");
    if (dPrevPts == nullptr || dNextPts == nullptr || dStatus == nullptr)
        return cudaErrorInvalidValue;
    if (params.winWidth <= 2 || params.winHeight <= 2 || params.searchRadius < 1 ||
        params.searchRadius > kMaxBlockMatchRadius || levelCount > kMaxBlockMatchLevels ||
        pointCount > kDeviceMaxKeypoints)
        return cudaErrorInvalidValue;
    if (levelCount != 0 && levels == nullptr) return cudaErrorInvalidValue;

    const uint32_t points = static_cast<uint32_t>(pointCount);
    BlockMatchArgs a;
    a.winW = params.winWidth;
    a.winH = params.winHeight;
    a.radius = params.searchRadius;
    a.subPixel = params.subPixel;

    // The host's own pyramid cap, applied where the host applies it: a level at
    // or below the window size gives every point nearly the same window.
    unsigned usableLevels = levelCount == 0 ? 0u : 1u;
    while (usableLevels < levelCount &&
           levels[usableLevels].width() > static_cast<size_t>(params.winWidth) &&
           levels[usableLevels].height() > static_cast<size_t>(params.winHeight)) {
        ++usableLevels;
    }

    const bool fast = blockMatchFastArmEnabled() &&
                      windowSpanFits(params.winWidth, params.winHeight,
                                     2 * params.searchRadius) &&
                      levelCount != 0;
    if (!fast) {
        BlockMatchLadder ladder;
        for (size_t i = 0; i < levelCount; ++i) ladder.lv[i] = levels[i];
        const uint32_t blocks = (points + kRefBlock - 1u) / kRefBlock;
        blockMatchRefKernel<<<blocks, kRefBlock, 0, stream>>>(
            ladder, usableLevels, dPrevPts, dNextPts, dStatus, points, a);
        return cudaGetLastError();
    }

    BINCV_ASSERT(dScratch != nullptr && scratchBytes >= blockMatchScratchBytes(pointCount),
                 "cuda blockMatch: the fast arm needs blockMatchScratchBytes of scratch");
    if (dScratch == nullptr || scratchBytes < blockMatchScratchBytes(pointCount))
        return cudaErrorInvalidValue;

    DeviceBlockMatchState* state = static_cast<DeviceBlockMatchState*>(dScratch);
    const dim3 block(32, kWarpsPerWindowBlock);
    const dim3 grid((points + kWarpsPerWindowBlock - 1u) / kWarpsPerWindowBlock);
    for (unsigned li = usableLevels; li-- > 0;) {
        blockMatchLevelKernel<<<grid, block, 0, stream>>>(levels[li], li, usableLevels,
                                                          dPrevPts, dNextPts, dStatus, points,
                                                          state, a);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
