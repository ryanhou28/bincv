// The packed census matcher's warp-cooperative separable-box arm.
//
// WHAT THE PROFILE SAID, AND WHAT THIS SHAPE IS ANSWERING. Nsight Compute on
// the shipped per-pixel kernel at 752x480 / D=0..64 / 9x9 / K=24: 255 registers
// per thread, two blocks resident per SM, warps_active 15% of peak. The stall
// histogram is 45% `wait` -- fixed-latency dependency with too few warps to
// hide it -- and 12% `mio_throttle`, the load/store instruction queue backing
// up. DRAM throughput is 0.7% of peak and `long_scoreboard` is 2%, so nothing
// here is waiting on memory; what is saturated is the ISSUE of load
// instructions and the register file. `math_pipe_throttle` is 3.5%, so the
// kernel is not popcount-bound either.
//
// Both leading terms point one way: issue FEWER instructions per output, and
// hold FEWER registers. The window sum is separable, so the horizontal half
// need not be recomputed per column at all -- and the vertical half is already
// slid. Per output pixel-disparity that takes 2*winWidth popcounts and about
// as many loads down to two popcounts, two loads, and a fixed handful of warp
// shuffles, on a kernel small enough to keep its accumulators in registers.
// Measured on the reference frame: 77.4 M warp-instructions against the shipped
// kernel's 158.5 M, and 3.77 M global-load instructions against its 20.2 M.
//
// WHY LANE-TO-LANE AND NOT A THREAD MARCHING ALONG X. Columns that share a
// horizontal window are adjacent, and on a GPU adjacent columns must be
// adjacent LANES or the loads stop coalescing: a thread owning eight adjacent
// columns makes its warp straddle eight 128-byte lines per load instruction,
// an eightfold tax on the exact pipe `mio_throttle` says is already the
// throttle. So the sharing runs across lanes. Shared memory is one way to do
// that and it is a recorded LOSS here (1.16 ms against 0.91: the redundant
// loads were already L1 hits, so staging bought nothing and cost 624 barriers
// per block plus byte-wide bank conflicts). `__shfl_down_sync` has no barrier,
// no shared memory and no bank conflicts.
//
// THE COST OF THE SHAPE IS THE HALO. Lane `l` aggregates lanes `l` through
// `l + winWidth - 1`, so only `33 - winWidth` of a warp's 32 lanes produce
// output -- 24 at winWidth 9, a 1.33x tax on the vertical-slide work. That is
// what buys the barrier-free horizontal sum, and it is why the arm's gate stops
// at winWidth 17, where the tax reaches 2x.
//
// AND THE FIRST VERSION OF IT WAS WORTH ONLY 1.10x, which is the part worth
// recording. It removed exactly the throttles it was aimed at --
// `mio_throttle` and `math_pipe_throttle` fell to under 1% of samples each --
// and was barely faster for it, because the profile had simply moved:
// `short_scoreboard` (the shuffle chain's own result dependency) and
// `long_scoreboard` rose to 22% and 19%, `wait` stayed at 37%, and the
// register allocation was still 254, so still two blocks per SM. Every leading
// term was exposed LATENCY at an occupancy too low to hide it. What closed the
// gap was not another arithmetic trick but the register budget: a SHORTER strip
// and a deeper disparity tile, which together do MORE work per output row --
// the window seed amortizes over four rows instead of sixteen -- and are 2.2x
// faster for it, 1.10x to 2.4x on the same interleaved comparison. 80
// registers, no spill, six blocks per SM by the profiler's own occupancy limit,
// and 40% of peak warps active against the shipped kernel's 15%.
//
// MEASURED NEGATIVE, do not retry: forcing the budget with `__launch_bounds__`
// instead of shrinking the state. At four blocks per SM the register cap is
// 128 and the kernel spilled 396 bytes, measuring 0.98x -- slower than the
// kernel it replaces. At six and eight blocks it spilled 148 and 244 bytes for
// 1.70x and 1.59x, against 2.53x for the same shape with no bound at all.
// Telling ptxas the answer is not the same as giving it less to hold.

#include "bincv/cuda/denseCensusBox.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief Disparities one pass of the vertical slide carries, so that one left
/// descriptor load serves this many candidates.
/// @note Swept WITH the strip below, because the two trade against the same
/// register budget and neither is readable alone. At kBoxStrip = 4 on the
/// reference frame: 4 gave 1.83x, 8 gave 2.37x, 16 gave 2.53x and 32 gave
/// 2.15x against the shipped kernel. Nothing here is inherited -- the shipped
/// kernel's own tile of 8 was swept against a completely different register
/// profile.
constexpr int kBoxDTile = 16;

/// @brief Output rows one warp walks, sliding its vertical window down them.
/// @note THE COUNTER-INTUITIVE ONE, and the profile is what explains it. A
/// longer strip does strictly LESS work -- the winHeight-row window seed is
/// amortized over more output rows -- and it is slower anyway. At the tile
/// above: 32 gave 0.72x and spilled, 16 gave 1.32x, 8 gave 1.77x, 4 gives
/// 2.53x, and 2 falls back to 2.34x because the seed stops amortizing at all.
/// The reason is that best[] costs one register per strip row and the whole
/// body is unrolled over it: four keeps the kernel at 80 registers with no
/// spill, which is six resident blocks per SM against the shipped kernel's two.
constexpr int kBoxStrip = 4;

/// @brief Warps per block. 32 x 4 = 128 threads, the shipped kernel's block.
constexpr unsigned kBoxWarps = 4;

/// @brief Lanes of a warp that produce OUTPUT at this window width. The other
/// `winWidth - 1` are the halo the horizontal aggregation reaches into.
/// @note ONE definition, used by the kernel AND by the launch geometry. They
/// were briefly two, and two copies that disagree by one lane still produced a
/// byte-identical map on a shifted test pair while silently aggregating a
/// column twice -- a constant that says how far a lane reaches into its
/// neighbours cannot have a second spelling.
BINCV_CUDA_HD constexpr int boxOutLanes(int winW) { return 33 - winW; }

/// @brief The widest window this arm accepts: the last one that still leaves
/// half a warp producing output. Above it the halo tax exceeds 2x and the
/// shipped per-pixel kernel is the better arm, so the launcher falls back
/// rather than instantiating a shape nobody measured.
constexpr int kBoxMaxWinWidth = 17;
static_assert(boxOutLanes(kBoxMaxWinWidth) >= 16 &&
                  boxOutLanes(kBoxMaxWinWidth + 2) < 16,
              "kBoxMaxWinWidth is DERIVED: the widest window at which at least "
              "half a warp's lanes still produce output");

/// @brief The largest window cost the packed `(cost << 8) | disparity` fold can
/// hold exactly. A descriptor carries at most 32 comparisons, so the cost is at
/// most `winWidth * winHeight * 32`; anything that fits 24 bits packs exactly,
/// and 254 is the largest disparity the params allow, so the packed value stays
/// strictly below the 0xFFFFFFFF sentinel.
constexpr long long kBoxMaxCost = 0xFFFFFE;

/// @brief One step of the binary-decomposition window sum across warp lanes.
///
/// Invariant entering the step: `blk` at lane `l` holds
/// `v(l) + ... + v(l + 2^B - 1)`, and lanes `[0, OFF)` of the window have
/// already been folded into the caller's running sum. If bit `B` of `WINW` is
/// set, the run of `2^B` lanes starting at `l + OFF` is exactly what is wanted
/// next, and one `__shfl_down_sync` by `OFF` fetches it.
///
/// EVERY OPERAND IS WARP-UNIFORM -- `WINW`, `B` and `OFF` are template
/// parameters, so all 32 lanes reach every shuffle with the same delta. That is
/// the property `__shfl_down_sync(0xFFFFFFFF, ...)` requires and the one a map
/// comparison cannot check.
///
/// @note Lanes near the top of the warp receive their OWN value where the
/// source lane is out of range. Those results are garbage and are DISCARDED:
/// lane `l` is only ever read by lanes `l - winWidth + 1 .. l`, and a lane that
/// writes output satisfies `lane < 33 - winWidth`, so its highest contributor
/// is lane 31. Widening the output-lane count past that bound would make this
/// silently wrong, which is why the bound is a named constant checked at the
/// launch site.
template <int WINW, int B, int OFF>
__device__ __forceinline__ unsigned warpWindowStep(unsigned blk) {
    constexpr unsigned kAll = 0xFFFFFFFFu;
    static_assert(B <= 4, "WINW <= 31, so the top set bit is at most bit 4");
    // Bit B of the width says whether the run of 2^B lanes at offset OFF is
    // part of the window; a higher set bit says the run must be doubled and the
    // walk continued. The recursion ends at the top set bit, so it cannot pass
    // bit 4 for any accepted width.
    constexpr bool kTake = ((WINW >> B) & 1) != 0;
    constexpr bool kMore = (WINW >> (B + 1)) != 0;
    unsigned s = 0u;
    if constexpr (kTake)
        s = (OFF == 0) ? blk : __shfl_down_sync(kAll, blk, static_cast<unsigned>(OFF));
    if constexpr (kMore)
        s += warpWindowStep<WINW, B + 1, kTake ? OFF + (1 << B) : OFF>(
            blk + __shfl_down_sync(kAll, blk, static_cast<unsigned>(1 << B)));
    return s;
}

/// @brief `S(lane) = v(lane) + v(lane + 1) + ... + v(lane + WINW - 1)`, with
/// the operands living in WINW different lanes' registers.
/// @note Costs one shuffle per set bit of `WINW` plus one per doubling:
/// 4 at WINW = 9 (1001b), 2 at 3, 3 at 5, 4 at 7, 4 at 15, 5 at 17. The
/// decomposition is the host census path's own horizontal doubling, spelled
/// across warp lanes instead of across a register's bit-planes.
template <int WINW>
__device__ __forceinline__ unsigned warpWindowSum(unsigned v) {
    static_assert(WINW >= 3 && WINW <= 31, "the decomposition covers 3..31");
    return warpWindowStep<WINW, 0, 0>(v);
}

/// @brief Add (or subtract) one image row's contribution to this lane's COLUMN
/// sum, for each disparity of the tile.
///
/// One left load serves the whole tile. The loads are warp-coalesced by
/// construction: lane `l` reads word `c0 + l`, so a warp's 32 lanes cover 128
/// consecutive bytes.
///
/// @param colOk False for a halo lane whose column has run off the row. Its
/// contribution is zero and its loads are predicated away -- the column is
/// never a contributor to a lane that writes, because a writing lane's
/// contributors all satisfy `column <= anchor + winWidth - 1 < width`.
template <int DT>
__device__ __forceinline__ void accumulateColumn(unsigned* v,
                                                 DeviceImageConstView<uint32_t> left,
                                                 DeviceImageConstView<uint32_t> right,
                                                 size_t yy, size_t c, bool colOk, int d0,
                                                 int dEnd, bool add) {
    const uint32_t lv = colOk ? __ldg(left.row(yy) + c) : 0u;
    const uint32_t* rowR = right.row(yy);
#pragma unroll
    for (int j = 0; j < DT; ++j) {
        const int d = d0 + j;
        if (d > dEnd) break;  // warp-uniform: d0 and dEnd are kernel arguments
        // `c < d` would read BEFORE the row -- before the allocation on row 0 --
        // so the load is predicated, not clamped. A candidate with no right
        // support at this column is never accumulated, which is also how the
        // shipped kernel and the host keep the shifted read inside the row.
        const bool ok = colOk && c >= static_cast<size_t>(d);
        const uint32_t rv = ok ? __ldg(rowR + c - static_cast<size_t>(d)) : 0u;
        const unsigned k = ok ? static_cast<unsigned>(__popc(lv ^ rv)) : 0u;
        v[j] = add ? v[j] + k : v[j] - k;
    }
}

/// @brief One warp owns `33 - WINW` anchor columns and a strip of output rows.
///
/// THE INVARIANT THAT MAKES THE REORDERING LEGAL, stated because it is the
/// single fact the whole shape rests on: a lane's ANCHOR IS ITS COLUMN. Lane
/// `l` writes only when its anchor `a` satisfies `a >= d` and
/// `a + WINW <= width`; its contributors are columns `a .. a + WINW - 1`, every
/// one of which then satisfies `column >= a >= d` and `column < width`. So no
/// contributor of a writing lane is ever the zeroed halo case, and the sum this
/// kernel forms over 81 (row, column) pairs is the same multiset the shipped
/// kernel sums pixel by pixel. Unsigned addition is associative and
/// commutative and the total cannot overflow (see kBoxMaxCost), so the two maps
/// are equal by arithmetic rather than by hope.
///
/// The running best is folded into ONE register per strip row as
/// `(cost << 8) | disparity`. THE PACKING IS WHAT ENFORCES THE TIE RULE, not
/// the comparison: an equal cost at a larger disparity gives a strictly larger
/// packed value, so it loses under `<` and under `<=` alike, and the host's
/// "ties keep the smallest disparity" survives whatever a later reader does to
/// the operator. Inverting the packed order is the change that breaks it, and
/// that is what the suite's flat-frame and periodic-frame cases catch. The
/// sentinel 0xFFFFFFFF decodes to byte 255 -- `kDenseDisparityInvalid` -- so a
/// pixel no candidate can serve needs no branch at write time.
template <int WINW, int DT, int STRIP>
__global__ void denseKernelPackedWarpBox(DeviceImageConstView<uint32_t> left,
                                         DeviceImageConstView<uint32_t> right, int minD,
                                         int dEnd, int winH,
                                         DeviceImageView<uint8_t> disparity,
                                         size_t outRows, size_t anchors) {
    constexpr int kOutLanes = boxOutLanes(WINW);
    const unsigned lane = threadIdx.x;
    // Warp-uniform early exits ONLY. A lane that leaves early would break every
    // shuffle below, and a kernel with that bug still produces correct maps on
    // most frames -- so every per-lane condition lives on a load or on the
    // output write, never on control flow around a shuffle.
    const size_t warpBase =
        (static_cast<size_t>(blockIdx.x) * kBoxWarps + threadIdx.y) *
        static_cast<size_t>(kOutLanes);
    if (warpBase >= anchors) return;
    const size_t sFirst = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(STRIP);
    if (sFirst >= outRows) return;

    const int hh = winH / 2;
    constexpr size_t kHalfW = static_cast<size_t>(WINW / 2);
    const size_t yFirst = static_cast<size_t>(hh) + sFirst;
    const int rowsHere = (outRows - sFirst < static_cast<size_t>(STRIP))
                             ? static_cast<int>(outRows - sFirst)
                             : STRIP;

    const size_t c = warpBase + lane;  // this lane's column, and its anchor
    const bool colOk = c < left.width;
    const bool writes = lane < static_cast<unsigned>(kOutLanes) && c < anchors;

    unsigned best[STRIP];
#pragma unroll
    for (int s = 0; s < STRIP; ++s) best[s] = 0xFFFFFFFFu;

    for (int d0 = minD; d0 <= dEnd; d0 += DT) {
        unsigned v[DT];
#pragma unroll
        for (int j = 0; j < DT; ++j) v[j] = 0u;

        // The first output row's window in full, counted up from the top row so
        // the index never goes negative: yFirst is at least hh.
        const size_t yTop = yFirst - static_cast<size_t>(hh);
        for (int r = 0; r < winH; ++r)
            accumulateColumn<DT>(v, left, right, yTop + static_cast<size_t>(r), c, colOk,
                                 d0, dEnd, true);

        // STRIP is the loop bound and `rowsHere` only breaks out of it: a
        // runtime bound would index best[] dynamically, which puts it in local
        // memory and gives away the occupancy this shape exists for.
#pragma unroll
        for (int s = 0; s < STRIP; ++s) {
            if (s >= rowsHere) break;  // warp-uniform
#pragma unroll
            for (int j = 0; j < DT; ++j) {
                const int d = d0 + j;
                if (d > dEnd) break;  // warp-uniform
                const unsigned sum = warpWindowSum<WINW>(v[j]);
                if (writes && static_cast<size_t>(d) <= c) {
                    const unsigned cand = (sum << 8) | static_cast<unsigned>(d);
                    if (cand < best[s]) best[s] = cand;
                }
            }
            if (s + 1 < rowsHere) {
                const size_t y = yFirst + static_cast<size_t>(s);
                accumulateColumn<DT>(v, left, right, y - static_cast<size_t>(hh), c,
                                     colOk, d0, dEnd, false);
                accumulateColumn<DT>(v, left, right, y + static_cast<size_t>(hh) + 1, c,
                                     colOk, d0, dEnd, true);
            }
        }
    }

    if (!writes) return;  // every shuffle is behind us
#pragma unroll
    for (int s = 0; s < STRIP; ++s) {
        if (s >= rowsHere) break;
        disparity.row(yFirst + static_cast<size_t>(s))[c + kHalfW] =
            static_cast<uint8_t>(best[s] & 0xFFu);
    }
}

template <int WINW>
cudaError_t launchBoxArm(DeviceImageConstView<uint32_t> left,
                         DeviceImageConstView<uint32_t> right, int minD, int dEnd,
                         int winH, DeviceImageView<uint8_t> disparity, size_t outRows,
                         cudaStream_t stream) {
    constexpr unsigned kOutLanes = static_cast<unsigned>(boxOutLanes(WINW));
    const size_t anchors = left.width - static_cast<size_t>(WINW) + 1;
    const size_t perBlock = static_cast<size_t>(kOutLanes) * kBoxWarps;
    const dim3 block(32, kBoxWarps);
    const dim3 grid(static_cast<unsigned>((anchors + perBlock - 1) / perBlock),
                    static_cast<unsigned>((outRows + kBoxStrip - 1) / kBoxStrip));
    denseKernelPackedWarpBox<WINW, kBoxDTile, kBoxStrip><<<grid, block, 0, stream>>>(
        left, right, minD, dEnd, winH, disparity, outRows, anchors);
    return cudaGetLastError();
}

} // namespace

namespace impl {

bool& densePackedBoxEnabled() {
    static bool on = true;
    return on;
}

bool densePackedBoxAccepts(const DenseDisparityParams& params) {
    if (params.winWidth < 3 || params.winWidth > kBoxMaxWinWidth) return false;
    if ((params.winWidth & 1) == 0 || params.winHeight < 3) return false;
    const long long maxCost = static_cast<long long>(params.winWidth) *
                              static_cast<long long>(params.winHeight) * 32;
    return maxCost <= kBoxMaxCost;
}

cudaError_t launchDensePackedWarpBox(DeviceImageConstView<uint32_t> left,
                                     DeviceImageConstView<uint32_t> right, int minD,
                                     int dEnd, const DenseDisparityParams& params,
                                     DeviceImageView<uint8_t> disparity, size_t outRows,
                                     cudaStream_t stream) {
    if (!densePackedBoxAccepts(params) ||
        static_cast<size_t>(params.winWidth) > left.width)
        return cudaErrorInvalidValue;
    switch (params.winWidth) {
        case 3:
            return launchBoxArm<3>(left, right, minD, dEnd, params.winHeight, disparity,
                                   outRows, stream);
        case 5:
            return launchBoxArm<5>(left, right, minD, dEnd, params.winHeight, disparity,
                                   outRows, stream);
        case 7:
            return launchBoxArm<7>(left, right, minD, dEnd, params.winHeight, disparity,
                                   outRows, stream);
        case 9:
            return launchBoxArm<9>(left, right, minD, dEnd, params.winHeight, disparity,
                                   outRows, stream);
        case 11:
            return launchBoxArm<11>(left, right, minD, dEnd, params.winHeight, disparity,
                                    outRows, stream);
        case 13:
            return launchBoxArm<13>(left, right, minD, dEnd, params.winHeight, disparity,
                                    outRows, stream);
        case 15:
            return launchBoxArm<15>(left, right, minD, dEnd, params.winHeight, disparity,
                                    outRows, stream);
        case 17:
            return launchBoxArm<17>(left, right, minD, dEnd, params.winHeight, disparity,
                                    outRows, stream);
        default:
            break;
    }
    // Unreachable through densePackedBoxAccepts, which admits only the odd
    // widths above. An error rather than a ninth instantiation: a kernel that
    // ran here would aggregate the wrong number of columns and return a
    // fully-formed, wrong map.
    return cudaErrorInvalidValue;
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
