// The device dense-disparity kernels.
//
// SHAPE, and why it is not the host's: the host must never hold a cost volume,
// so it streams rows and keeps per-disparity bit-sliced accumulators. A GPU
// thread can hold one pixel's running best in registers for the whole
// disparity sweep, which satisfies the same rule with no scratch at all. The
// cost of one candidate is the windowed Hamming distance read directly from
// the packed rows: per window row, extract winWidth bits of each image
// (right image shifted by d) and popcount the XOR. All reads are in-row for
// every anchor a candidate may claim (a >= d and a + winWidth <= width), which
// the host establishes the same way.
//
// Binary and census are ONE kernel: the binary path is the census path at
// planes = 1. Plane k of a block sits at rows [k * H, (k + 1) * H).

#include "bincv/cuda/denseDisparity.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief Bits [bitPos, bitPos + nbits) of a packed row, nbits in 1..32.
/// Reads word i+1 only when the run crosses into it, which the caller's
/// support bounds keep inside the row.
__device__ inline uint32_t extractBits(const uint32_t* row, size_t bitPos,
                                       unsigned nbits) {
    const size_t i = bitPos >> 5;
    const unsigned r = static_cast<unsigned>(bitPos & 31u);
    uint32_t lo = __ldg(row + i) >> r;
    if (r != 0 && r + nbits > 32u) lo |= __ldg(row + i + 1) << (32u - r);
    return nbits == 32u ? lo : lo & ((1u << nbits) - 1u);
}

/// @brief Hamming distance between winW-bit runs at (rowL, aL) and (rowR, aR).
__device__ inline unsigned windowRowHamming(const uint32_t* rowL, size_t aL,
                                            const uint32_t* rowR, size_t aR,
                                            unsigned winW) {
    unsigned cost = 0;
    unsigned off = 0;
    while (off < winW) {
        const unsigned chunk = (winW - off < 32u) ? (winW - off) : 32u;
        const uint32_t l = extractBits(rowL, aL + off, chunk);
        const uint32_t r = extractBits(rowR, aR + off, chunk);
        cost += static_cast<unsigned>(__popc(l ^ r));
        off += chunk;
    }
    return cost;
}

__global__ void denseKernel(DeviceBinMatConstView left, DeviceBinMatConstView right,
                            int minD, int dEnd, int winW, int winH, int planes,
                            size_t imgHeight, DeviceImageView<uint8_t> disparity) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= disparity.width || y >= imgHeight) return;

    const int hw = winW / 2;
    const int hh = winH / 2;
    uint8_t out = kDenseDisparityInvalid;
    if (y >= static_cast<size_t>(hh) && y + static_cast<size_t>(hh) < imgHeight &&
        x >= static_cast<size_t>(hw) && x + static_cast<size_t>(hw) < disparity.width) {
        const size_t a = x - static_cast<size_t>(hw);  // window anchor
        unsigned bestC = 0xFFFFFFFFu;
        unsigned bestD = 0xFFFFu;
        for (int d = minD; d <= dEnd; ++d) {
            // Anchors this disparity may claim: a >= d, full right support.
            if (static_cast<size_t>(d) > a) continue;
            const size_t aR = a - static_cast<size_t>(d);
            unsigned cost = 0;
            for (int k = 0; k < planes; ++k) {
                const size_t planeRow0 = static_cast<size_t>(k) * imgHeight;
                for (int r = -hh; r <= hh; ++r) {
                    const size_t row = planeRow0 + y + static_cast<size_t>(r + hh) -
                                       static_cast<size_t>(hh);
                    cost += windowRowHamming(left.row(row), a, right.row(row), aR,
                                             static_cast<unsigned>(winW));
                }
            }
            // Strictly less, disparities ascending: ties keep the smallest,
            // the host's rule.
            if (cost < bestC) {
                bestC = cost;
                bestD = static_cast<unsigned>(d);
            }
        }
        if (bestD <= 254u) out = static_cast<uint8_t>(bestD);
    }
    disparity.row(y)[x] = out;
}

constexpr int kDTile = 8;

// ---------------------------------------------------------------------------
// The SLIDING arm, and why it exists.
//
// Both arms above re-evaluate the WHOLE window for every output row: 24 census
// planes x 9 window rows x 64 disparities is ~13,800 popcounts per pixel, and
// the window's height multiplies the cost. It does not have to. The window sum
// for row y+1 is the sum for row y, minus the row that left, plus the row that
// entered -- so a row's cost is paid twice instead of winHeight times.
//
// This is not a new idea here: binCV's own HOST census path already does it,
// and its optimization curve records the sliding vertical accumulator at
// ~4.5x (docs/reports/stereo.md). The device kernel was ported without that
// refinement. The binary entry got away with it -- one plane, ~576 popcounts
// per pixel -- and the census entry, at 24x that, did not.
//
// Each thread owns one output column and a STRIP of output rows, sliding down
// it: winHeight row costs for the first row, two for each row after. The
// leaving row is recomputed rather than cached, which is the host's trade too
// -- a cached ring would need a dynamically indexed register array, and that
// spills to local memory, which costs more than the recomputation saves.
//
// The disparity tile is kept: one L extraction still serves kDTile candidates'
// XORs, so the two savings compose rather than replacing one another.
// ---------------------------------------------------------------------------

constexpr int kStrip = 16;

/// @brief Add (or subtract) one image row's windowed cost to each candidate's
/// running vertical sum. `winW <= 32`, which the launcher's gate enforces.
template <int DT>
__device__ __forceinline__ void accumulateRow(unsigned* sum, DeviceBinMatConstView left,
                                              DeviceBinMatConstView right, int planes,
                                              size_t imgHeight, size_t yy, size_t a,
                                              int d0, int dEnd, unsigned winW,
                                              bool add) {
    for (int k = 0; k < planes; ++k) {
        const size_t row = static_cast<size_t>(k) * imgHeight + yy;
        const uint32_t* rowL = left.row(row);
        const uint32_t* rowR = right.row(row);
        const uint32_t lw = extractBits(rowL, a, winW);
#pragma unroll
        for (int j = 0; j < DT; ++j) {
            const int d = d0 + j;
            // A candidate with no right support at this column is never read,
            // so it is never accumulated -- and the guard also keeps the
            // shifted read inside the row.
            if (d <= dEnd && static_cast<size_t>(d) <= a) {
                const uint32_t rw =
                    extractBits(rowR, a - static_cast<size_t>(d), winW);
                const unsigned c = static_cast<unsigned>(__popc(lw ^ rw));
                sum[j] = add ? sum[j] + c : sum[j] - c;
            }
        }
    }
}

__global__ void denseKernelSliding(DeviceBinMatConstView left,
                                   DeviceBinMatConstView right, int minD, int dEnd,
                                   int winW, int winH, int planes, size_t imgHeight,
                                   DeviceImageView<uint8_t> disparity, size_t outRows) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw = winW / 2;
    const int hh = winH / 2;
    // Border rows and columns are pre-filled with the invalid marker by the
    // launcher, so this kernel writes only pixels a candidate can serve.
    if (x < static_cast<size_t>(hw) || x + static_cast<size_t>(hw) >= disparity.width)
        return;
    const size_t a = x - static_cast<size_t>(hw);
    const size_t sFirst = static_cast<size_t>(blockIdx.y) * kStrip;
    if (sFirst >= outRows) return;
    const size_t yFirst = static_cast<size_t>(hh) + sFirst;
    const size_t rowsHere =
        (outRows - sFirst < static_cast<size_t>(kStrip)) ? (outRows - sFirst)
                                                         : static_cast<size_t>(kStrip);

    unsigned bestCost[kStrip];
    unsigned bestDisp[kStrip];
#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        bestCost[s] = 0xFFFFFFFFu;
        bestDisp[s] = 0xFFFFu;
    }

    const unsigned W = static_cast<unsigned>(winW);
    for (int d0 = minD; d0 <= dEnd; d0 += kDTile) {
        unsigned sum[kDTile];
#pragma unroll
        for (int j = 0; j < kDTile; ++j) sum[j] = 0;

        // The first output row's window, in full: rows [yFirst - hh, yFirst + hh].
        // Counted up from the top row so the index never goes negative --
        // yFirst is at least hh, so the subtraction is safe before the loop.
        const size_t yTop = yFirst - static_cast<size_t>(hh);
        for (int r = 0; r < winH; ++r) {
            accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                  yTop + static_cast<size_t>(r), a, d0, dEnd, W, true);
        }

#pragma unroll
        for (int s = 0; s < kStrip; ++s) {
            if (static_cast<size_t>(s) >= rowsHere) break;
#pragma unroll
            for (int j = 0; j < kDTile; ++j) {
                const int d = d0 + j;
                // Strictly less, disparities ascending: ties keep the
                // smallest, the host's rule.
                if (d <= dEnd && static_cast<size_t>(d) <= a && sum[j] < bestCost[s]) {
                    bestCost[s] = sum[j];
                    bestDisp[s] = static_cast<unsigned>(d);
                }
            }
            if (static_cast<size_t>(s) + 1 < rowsHere) {
                const size_t y = yFirst + static_cast<size_t>(s);
                accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                      y - static_cast<size_t>(hh), a, d0, dEnd, W,
                                      false);
                accumulateRow<kDTile>(sum, left, right, planes, imgHeight,
                                      y + static_cast<size_t>(hh) + 1, a, d0, dEnd, W,
                                      true);
            }
        }
    }

#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        if (static_cast<size_t>(s) >= rowsHere) break;
        if (bestDisp[s] <= 254u)
            disparity.row(yFirst + static_cast<size_t>(s))[x] =
                static_cast<uint8_t>(bestDisp[s]);
    }
}

// ---------------------------------------------------------------------------
// The WORD-PARALLEL BIT-SLICED arm, and why it exists.
//
// Every arm above maps one thread to one output PIXEL: per candidate and
// window row it extracts a winWidth-bit run from each image and popcounts the
// XOR -- a 32-bit instruction doing nine bits of useful work, making no use at
// all of the fact that 32 pixels already live in one word. That is where the
// binary path's advantage was going: against the same frame's census matcher
// the device gained 2.3x where the host's own bit-sliced kernel gains 7.6x on
// aarch64 and ~17x on x86-64.
//
// Here one thread owns one WORD of output anchors -- 32 pixels -- and the
// arithmetic is the host kernel's, ported rather than reinvented:
//
//   * The raw cost of 32 pixels at a candidate is ONE xor of the left word
//     against the right word shifted by d.
//   * The horizontal winWidth-window sum is a bit-sliced count of winWidth
//     shifted copies of that word into bitSlicedSumPlanes(winWidth) planes.
//   * The vertical accumulation slides down a strip of output rows, bit-sliced.
//   * The winner-take-all is a bit-sliced compare -- a borrow chain -- and a
//     masked select: the host's planesLess and planesSelect.
//
// HORIZONTAL BEFORE VERTICAL, which reverses the host's order and is the one
// decision the port does not inherit. The host sums vertically first and then
// runs a doubling tree over the lane-shifted ACCUMULATOR planes; on the device
// a lane shift crosses the word boundary into the neighbouring THREAD's
// registers, so that tree would cost a warp shuffle per plane per stage.
// Summing horizontally first shifts only the raw cost word, whose neighbour a
// thread rebuilds from two more loads -- no cross-thread traffic at all.
// Integer addition commutes, so the map is identical either way.
// ---------------------------------------------------------------------------

/// @brief Output rows one thread walks, sliding its vertical accumulator.
/// @note Swept on the reference frame: 4 gave 0.078 ms and 16 gave 0.148 --
/// sixteen strip rows hold 256 words of running best, which spills (1,768 bytes
/// of spill stores at the 255-register ceiling) -- against this value's 0.067.
constexpr int kBsStrip = 8;

/// @brief Threads that split one output word's disparity range between them.
/// Word-parallel work is 32x denser per thread, and the reference frame's rows
/// are only 24 words wide: without this split the whole frame is 1,416 threads
/// -- under one warp per SM -- and the kernel starves whatever its arithmetic
/// costs.
/// @note Swept with the strip: 4 gave 0.083 ms and 16 gave 0.080 against this
/// value's 0.067. Four leaves too few threads; sixteen pays a fourth fold step
/// over chunks only four disparities long.
constexpr int kBsChunks = 8;

/// @brief Output words per block. `blockDim` is (kBsChunks, kBsWords) and the
/// product is exactly one warp, so a word's chunk threads always share a warp
/// and the fold between them is a shuffle rather than shared memory.
constexpr int kBsWords = 32 / kBsChunks;

/// @brief Planes the vertical accumulator and the disparity carry. The binary
/// path asserts winWidth * winHeight <= 255, so eight always suffice -- and
/// fixing the count at compile time is what keeps every plane array in
/// registers: a runtime bound makes the indexing dynamic and puts the arrays in
/// local memory, which costs far more than the one plane it would save.
constexpr int kBsAccPlanes = 8;
constexpr int kBsDispPlanes = 8;

/// @brief Word `j` of a packed row, or zero outside it -- the device spelling
/// of the host's `extendedRowWord` with a zero fill.
__device__ __forceinline__ uint32_t bsRowWord(const uint32_t* row, long long j,
                                              long long words) {
    return (j < 0 || j >= words) ? 0u : __ldg(row + j);
}

/// @brief `planes += v << W0`, bit-sliced ripple. The plane budget bounds the
/// value, so the carry out of the last plane is zero by construction.
template <int HP, int W0>
__device__ __forceinline__ void bsAddPlane(uint32_t* planes, uint32_t v) {
#pragma unroll
    for (int p = W0; p < HP; ++p) {
        const uint32_t t = planes[p] ^ v;
        v &= planes[p];
        planes[p] = t;
    }
}

/// @brief The horizontal window sum of one output word at one image row and
/// disparity: lane b holds the sum over j in [0, winW) of the raw cost at lane
/// b + j, as `HP` bit-planes. `winW <= 32`, which the launcher's gate enforces.
template <int HP>
__device__ __forceinline__ void bsWindowRow(const uint32_t* rowL, const uint32_t* rowR,
                                            long long words, long long i, int d,
                                            int winW, uint32_t* h) {
    const long long s = d >> 5;
    const unsigned r = static_cast<unsigned>(d) & 31u;
    const long long j = i - s;
    const uint32_t r0 = bsRowWord(rowR, j - 1, words);
    const uint32_t r1 = bsRowWord(rowR, j, words);
    const uint32_t r2 = bsRowWord(rowR, j + 1, words);
    // cost lane x = L[x] ^ R[x - d]: the right row shifted right by d pixels is
    // one funnel shift of the two words it spans.
    const uint32_t c0 = __ldg(rowL + i) ^ __funnelshift_l(r0, r1, r);
    // The neighbouring word, which the lane shifts below read into. Past the
    // row it reads zero, exactly as the host's laneShiftDown does.
    const uint32_t c1 = (i + 1 < words)
                            ? (__ldg(rowL + i + 1) ^ __funnelshift_l(r1, r2, r))
                            : 0u;

    // Three shifted copies at a time: one full adder collapses them into a
    // weight-1 and a weight-2 word, so the plane ripple runs twice per three
    // inputs instead of three times. Measured as instructions rather than
    // argued: nine inputs cost 28 logic ops this way against 42 one at a time.
    uint32_t a = c0;
    uint32_t b = __funnelshift_r(c0, c1, 1u);
    uint32_t c = __funnelshift_r(c0, c1, 2u);
    h[0] = a ^ b ^ c;
    h[1] = (a & b) | (a & c) | (b & c);
#pragma unroll
    for (int p = 2; p < HP; ++p) h[p] = 0u;
    int k = 3;
    for (; k + 3 <= winW; k += 3) {
        a = __funnelshift_r(c0, c1, static_cast<unsigned>(k));
        b = __funnelshift_r(c0, c1, static_cast<unsigned>(k + 1));
        c = __funnelshift_r(c0, c1, static_cast<unsigned>(k + 2));
        bsAddPlane<HP, 0>(h, a ^ b ^ c);
        bsAddPlane<HP, 1>(h, (a & b) | (a & c) | (b & c));
    }
    for (; k < winW; ++k)
        bsAddPlane<HP, 0>(h, __funnelshift_r(c0, c1, static_cast<unsigned>(k)));
}

/// @brief `acc += h`, bit-sliced carry ripple; the host's accAddWord.
template <int HP>
__device__ __forceinline__ void bsAccAdd(uint32_t* acc, const uint32_t* h) {
    uint32_t carry = 0u;
#pragma unroll
    for (int p = 0; p < HP; ++p) {
        const uint32_t v = h[p];
        const uint32_t sum = acc[p] ^ v ^ carry;
        carry = (acc[p] & v) | (acc[p] & carry) | (v & carry);
        acc[p] = sum;
    }
    // Planes above the addend's are known zero there, so the ripple continues
    // in the two-op carry-only form.
#pragma unroll
    for (int p = HP; p < kBsAccPlanes; ++p) {
        const uint32_t sum = acc[p] ^ carry;
        carry &= acc[p];
        acc[p] = sum;
    }
}

/// @brief `acc -= h`, bit-sliced borrow ripple; the host's accSubWord. The
/// accumulator always holds at least the row being removed -- the same
/// arithmetic added it -- so the final borrow is zero by construction.
template <int HP>
__device__ __forceinline__ void bsAccSub(uint32_t* acc, const uint32_t* h) {
    uint32_t borrow = 0u;
#pragma unroll
    for (int p = 0; p < HP; ++p) {
        const uint32_t v = h[p];
        const uint32_t diff = acc[p] ^ v ^ borrow;
        borrow = (~acc[p] & (v | borrow)) | (v & borrow);
        acc[p] = diff;
    }
#pragma unroll
    for (int p = HP; p < kBsAccPlanes; ++p) {
        const uint32_t diff = acc[p] ^ borrow;
        borrow = ~acc[p] & borrow;
        acc[p] = diff;
    }
}

/// @brief The lanes where `v < best`, as a mask word: the borrow out of the
/// lane-wise subtraction. The host's planesLess.
__device__ __forceinline__ uint32_t bsLess(const uint32_t* v, const uint32_t* best) {
    uint32_t borrow = 0u;
#pragma unroll
    for (int p = 0; p < kBsAccPlanes; ++p)
        borrow = (~v[p] & (best[p] | borrow)) | (best[p] & borrow);
    return borrow;
}

/// @brief The lanes where `(costA, dispA)` beats `(costB, dispB)`: lower cost,
/// or the same cost at a lower disparity. One borrow chain over the disparity
/// planes and the cost planes in that order, which IS the comparison of the
/// concatenated value.
/// @note Lexicographic rather than cost-only because the chunk fold below is a
/// TREE: its second step already holds partial winners from non-adjacent
/// chunks, so "keep it only if strictly less" no longer means "keep the
/// smallest disparity on a tie". Comparing the disparity too makes the fold
/// order irrelevant, which is what the host's sequential sweep gets for free.
__device__ __forceinline__ uint32_t bsBeats(const uint32_t* costA, const uint32_t* dispA,
                                            const uint32_t* costB,
                                            const uint32_t* dispB) {
    uint32_t borrow = 0u;
#pragma unroll
    for (int p = 0; p < kBsDispPlanes; ++p)
        borrow = (~dispA[p] & (dispB[p] | borrow)) | (dispB[p] & borrow);
#pragma unroll
    for (int p = 0; p < kBsAccPlanes; ++p)
        borrow = (~costA[p] & (costB[p] | borrow)) | (costB[p] & borrow);
    return borrow;
}

/// @brief `best = mask ? v : best`, per plane. The host's planesSelect.
template <int N>
__device__ __forceinline__ void bsSelect(uint32_t* best, const uint32_t* v, uint32_t m) {
#pragma unroll
    for (int p = 0; p < N; ++p) best[p] = (best[p] & ~m) | (v[p] & m);
}

/// @brief `bestD = mask ? d : bestD`, with `d` broadcast into bit-planes.
__device__ __forceinline__ void bsSelectDisp(uint32_t* bestD, int d, uint32_t m) {
#pragma unroll
    for (int p = 0; p < kBsDispPlanes; ++p) {
        const uint32_t bit =
            ((static_cast<unsigned>(d) >> p) & 1u) != 0u ? 0xFFFFFFFFu : 0u;
        bestD[p] = (bestD[p] & ~m) | (bit & m);
    }
}

/// @brief The lanes of output word `i` whose anchor lies in [lo, hi] -- the
/// anchors a disparity may claim. The host's laneRangeMask.
__device__ __forceinline__ uint32_t bsLaneRange(long long i, long long lo, long long hi) {
    const long long base = i * 32;
    long long a = lo - base;
    long long b = hi - base;
    if (b < 0 || a >= 32) return 0u;
    if (a < 0) a = 0;
    if (b >= 32) b = 31;
    const uint32_t upTo =
        (b == 31) ? 0xFFFFFFFFu : ((1u << static_cast<unsigned>(b + 1)) - 1u);
    return upTo & (0xFFFFFFFFu << static_cast<unsigned>(a));
}

template <int HP>
__global__ void denseKernelBitSliced(DeviceBinMatConstView left,
                                     DeviceBinMatConstView right, int minD, int dEnd,
                                     int winW, int winH,
                                     DeviceImageView<uint8_t> disparity, size_t outRows,
                                     size_t rowWordCount) {
    const long long words = static_cast<long long>(rowWordCount);
    const long long iRaw = static_cast<long long>(blockIdx.x) * kBsWords +
                           static_cast<long long>(threadIdx.y);
    // A thread past the row's last word still runs the fold below -- that is a
    // warp shuffle, so every lane has to reach it. It just never writes.
    const bool active = iRaw < words;
    const long long i = active ? iRaw : 0;

    const size_t sFirst = static_cast<size_t>(blockIdx.y) * kBsStrip;
    if (sFirst >= outRows) return;   // block-uniform: the whole warp leaves
    const int hh = winH / 2;
    const int hw = winW / 2;
    const size_t yFirst = static_cast<size_t>(hh) + sFirst;
    const int rowsHere = (outRows - sFirst < static_cast<size_t>(kBsStrip))
                             ? static_cast<int>(outRows - sFirst)
                             : kBsStrip;

    // The running best per output row of the strip, bit-sliced. All-ones is the
    // host's initial cost (nothing can be strictly less than a saturated
    // accumulator) and all-ones in the disparity planes is 255, the invalid
    // marker, so a lane no candidate claims reads out correct with no branch.
    uint32_t bestC[kBsStrip][kBsAccPlanes];
    uint32_t bestD[kBsStrip][kBsDispPlanes];
#pragma unroll
    for (int s = 0; s < kBsStrip; ++s) {
#pragma unroll
        for (int p = 0; p < kBsAccPlanes; ++p) bestC[s][p] = 0xFFFFFFFFu;
#pragma unroll
        for (int p = 0; p < kBsDispPlanes; ++p) bestD[s][p] = 0xFFFFFFFFu;
    }

    // BLOCKED, never strided: the fold below keeps a candidate only when it is
    // strictly less, which reproduces the host's tie rule -- smallest disparity
    // wins -- only if a lower chunk really does hold the lower disparities.
    const int chunk = static_cast<int>(threadIdx.x);
    const int span = (dEnd - minD + kBsChunks) / kBsChunks;
    const int dLo = minD + chunk * span;
    const int dHi = (dLo + span - 1 < dEnd) ? (dLo + span - 1) : dEnd;

    const long long anchorHi = static_cast<long long>(disparity.width) - winW;
    const size_t stL = left.stride;
    const size_t stR = right.stride;

    for (int d = dLo; d <= dHi; ++d) {
        uint32_t acc[kBsAccPlanes];
#pragma unroll
        for (int p = 0; p < kBsAccPlanes; ++p) acc[p] = 0u;
        uint32_t h[HP];
        // The first output row's window in full: rows [yFirst - hh, yFirst + hh],
        // counted up from the top so the index never goes negative.
        const uint32_t* lRow = left.ptr + sFirst * stL;
        const uint32_t* rRow = right.ptr + sFirst * stR;
        for (int r = 0; r < winH; ++r) {
            bsWindowRow<HP>(lRow, rRow, words, i, d, winW, h);
            bsAccAdd<HP>(acc, h);
            lRow += stL;
            rRow += stR;
        }
        const uint32_t rng = bsLaneRange(i, d, anchorHi);

#pragma unroll
        for (int s = 0; s < kBsStrip; ++s) {
            if (s >= rowsHere) break;
            const uint32_t m = bsLess(acc, bestC[s]) & rng;
            bsSelect<kBsAccPlanes>(bestC[s], acc, m);
            bsSelectDisp(bestD[s], d, m);
            if (s + 1 < rowsHere) {
                const size_t y = yFirst + static_cast<size_t>(s);
                const size_t yLeave = y - static_cast<size_t>(hh);
                const size_t yEnter = y + static_cast<size_t>(hh) + 1;
                bsWindowRow<HP>(left.ptr + yLeave * stL, right.ptr + yLeave * stR, words,
                                i, d, winW, h);
                bsAccSub<HP>(acc, h);
                bsWindowRow<HP>(left.ptr + yEnter * stL, right.ptr + yEnter * stR, words,
                                i, d, winW, h);
                bsAccAdd<HP>(acc, h);
            }
        }
    }

    if (kBsChunks > 1) {
        // Fold the chunks pairwise.
#pragma unroll
        for (int off = kBsChunks / 2; off > 0; off >>= 1) {
#pragma unroll
            for (int s = 0; s < kBsStrip; ++s) {
                if (s >= rowsHere) break;
                uint32_t oC[kBsAccPlanes];
                uint32_t oD[kBsDispPlanes];
#pragma unroll
                for (int p = 0; p < kBsAccPlanes; ++p)
                    oC[p] = __shfl_down_sync(0xFFFFFFFFu, bestC[s][p],
                                             static_cast<unsigned>(off));
#pragma unroll
                for (int p = 0; p < kBsDispPlanes; ++p)
                    oD[p] = __shfl_down_sync(0xFFFFFFFFu, bestD[s][p],
                                             static_cast<unsigned>(off));
                const uint32_t m = bsBeats(oC, oD, bestC[s], bestD[s]);
                bsSelect<kBsAccPlanes>(bestC[s], oC, m);
                bsSelect<kBsDispPlanes>(bestD[s], oD, m);
            }
        }
        // Back to every chunk thread, so the byte extraction below is split
        // kBsChunks ways and neighbouring threads write neighbouring bytes.
        const unsigned base = static_cast<unsigned>(kBsChunks) * threadIdx.y;
#pragma unroll
        for (int s = 0; s < kBsStrip; ++s) {
            if (s >= rowsHere) break;
#pragma unroll
            for (int p = 0; p < kBsDispPlanes; ++p)
                bestD[s][p] = __shfl_sync(0xFFFFFFFFu, bestD[s][p],
                                          static_cast<int>(base));
        }
    }

    if (!active) return;
    constexpr int kLanesPerThread = 32 / kBsChunks;
    const int laneLo = chunk * kLanesPerThread;
#pragma unroll
    for (int s = 0; s < kBsStrip; ++s) {
        if (s >= rowsHere) break;
        uint8_t* out = disparity.row(yFirst + static_cast<size_t>(s));
#pragma unroll
        for (int b = 0; b < kLanesPerThread; ++b) {
            const long long a = i * 32 + laneLo + b;
            // Anchors ascend, so past the last one the rest of the word is rim:
            // the launcher's fill already wrote the invalid marker there.
            if (a > anchorHi) break;
            unsigned v = 0;
#pragma unroll
            for (int p = 0; p < kBsDispPlanes; ++p)
                v |= ((bestD[s][p] >> (laneLo + b)) & 1u) << p;
            out[a + hw] = static_cast<uint8_t>(v);
        }
    }
}

template <int HP>
cudaError_t launchBitSlicedArm(DeviceBinMatConstView left, DeviceBinMatConstView right,
                               int minD, int dEnd, const DenseDisparityParams& params,
                               DeviceImageView<uint8_t> disparity, size_t outRows,
                               size_t words, cudaStream_t stream) {
    const dim3 block(kBsChunks, kBsWords);
    const dim3 grid(static_cast<unsigned>((words + kBsWords - 1) / kBsWords),
                    static_cast<unsigned>((outRows + kBsStrip - 1) / kBsStrip));
    denseKernelBitSliced<HP><<<grid, block, 0, stream>>>(
        left, right, minD, dEnd, params.winWidth, params.winHeight, disparity, outRows,
        words);
    return cudaGetLastError();
}

/// @brief Pick the instantiation whose plane count matches this window width.
/// The horizontal sum's plane count is what the kernel must know statically --
/// see kBsAccPlanes -- and it takes five values over the accepted widths, so
/// five instantiations cover them exactly.
cudaError_t launchBitSliced(DeviceBinMatConstView left, DeviceBinMatConstView right,
                            int minD, int dEnd, const DenseDisparityParams& params,
                            DeviceImageView<uint8_t> disparity, size_t outRows,
                            size_t words, cudaStream_t stream) {
    switch (bitSlicedSumPlanes(static_cast<size_t>(params.winWidth))) {
        case 2:
            return launchBitSlicedArm<2>(left, right, minD, dEnd, params, disparity,
                                         outRows, words, stream);
        case 3:
            return launchBitSlicedArm<3>(left, right, minD, dEnd, params, disparity,
                                         outRows, words, stream);
        case 4:
            return launchBitSlicedArm<4>(left, right, minD, dEnd, params, disparity,
                                         outRows, words, stream);
        case 5:
            return launchBitSlicedArm<5>(left, right, minD, dEnd, params, disparity,
                                         outRows, words, stream);
        default:
            return launchBitSlicedArm<6>(left, right, minD, dEnd, params, disparity,
                                         outRows, words, stream);
    }
}

/// @brief The packed matcher's disparity tile, swept independently of the
/// plane kernel's because its register pressure differs (no plane loop).
/// Measured on the reference frame: 4 gave 1.97 ms and 16 gave 1.11 ms
/// against this value's 0.91 -- eight is the optimum, not an inherited default.
constexpr int kPackDTile = 8;

// ---------------------------------------------------------------------------
// The PACKED-DESCRIPTOR matcher. Same sliding strip, same disparity tile; the
// only change is the input layout, and it is the change that matters. Reading
// a pixel's K census comparisons from K plane arrays costs K loads and K
// popcounts, each counting ONE useful bit out of 32. Reading them from one
// packed word costs one load and one popcount, with K bits doing useful work.
// ---------------------------------------------------------------------------

/// @brief Add (or subtract) one image row's windowed cost, packed layout.
template <int DT>
__device__ __forceinline__ void accumulateRowPacked(
    unsigned* sum, DeviceImageConstView<uint32_t> left,
    DeviceImageConstView<uint32_t> right, size_t yy, size_t a, int d0, int dEnd,
    int winW, bool add) {
    const uint32_t* rowL = left.row(yy);
    const uint32_t* rowR = right.row(yy);
    for (int i = 0; i < winW; ++i) {
        const uint32_t lv = __ldg(rowL + a + static_cast<size_t>(i));
#pragma unroll
        for (int j = 0; j < DT; ++j) {
            const int d = d0 + j;
            if (d <= dEnd && static_cast<size_t>(d) <= a) {
                const uint32_t rv =
                    __ldg(rowR + a + static_cast<size_t>(i) - static_cast<size_t>(d));
                const unsigned c = static_cast<unsigned>(__popc(lv ^ rv));
                sum[j] = add ? sum[j] + c : sum[j] - c;
            }
        }
    }
}

__global__ void denseKernelPacked(DeviceImageConstView<uint32_t> left,
                                  DeviceImageConstView<uint32_t> right, int minD,
                                  int dEnd, int winW, int winH,
                                  DeviceImageView<uint8_t> disparity, size_t outRows) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw = winW / 2;
    const int hh = winH / 2;
    if (x < static_cast<size_t>(hw) || x + static_cast<size_t>(hw) >= disparity.width)
        return;
    const size_t a = x - static_cast<size_t>(hw);
    const size_t sFirst = static_cast<size_t>(blockIdx.y) * kStrip;
    if (sFirst >= outRows) return;
    const size_t yFirst = static_cast<size_t>(hh) + sFirst;
    const size_t rowsHere =
        (outRows - sFirst < static_cast<size_t>(kStrip)) ? (outRows - sFirst)
                                                         : static_cast<size_t>(kStrip);

    unsigned bestCost[kStrip];
    unsigned bestDisp[kStrip];
#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        bestCost[s] = 0xFFFFFFFFu;
        bestDisp[s] = 0xFFFFu;
    }

    for (int d0 = minD; d0 <= dEnd; d0 += kPackDTile) {
        unsigned sum[kPackDTile];
#pragma unroll
        for (int j = 0; j < kPackDTile; ++j) sum[j] = 0;

        const size_t yTop = yFirst - static_cast<size_t>(hh);
        for (int r = 0; r < winH; ++r) {
            accumulateRowPacked<kPackDTile>(sum, left, right, yTop + static_cast<size_t>(r),
                                        a, d0, dEnd, winW, true);
        }

#pragma unroll
        for (int s = 0; s < kStrip; ++s) {
            if (static_cast<size_t>(s) >= rowsHere) break;
#pragma unroll
            for (int j = 0; j < kPackDTile; ++j) {
                const int d = d0 + j;
                if (d <= dEnd && static_cast<size_t>(d) <= a && sum[j] < bestCost[s]) {
                    bestCost[s] = sum[j];
                    bestDisp[s] = static_cast<unsigned>(d);
                }
            }
            if (static_cast<size_t>(s) + 1 < rowsHere) {
                const size_t y = yFirst + static_cast<size_t>(s);
                accumulateRowPacked<kPackDTile>(sum, left, right,
                                            y - static_cast<size_t>(hh), a, d0, dEnd,
                                            winW, false);
                accumulateRowPacked<kPackDTile>(sum, left, right,
                                            y + static_cast<size_t>(hh) + 1, a, d0,
                                            dEnd, winW, true);
            }
        }
    }

#pragma unroll
    for (int s = 0; s < kStrip; ++s) {
        if (static_cast<size_t>(s) >= rowsHere) break;
        if (bestDisp[s] <= 254u)
            disparity.row(yFirst + static_cast<size_t>(s))[x] =
                static_cast<uint8_t>(bestDisp[s]);
    }
}

constexpr unsigned kPackBlock = 128;

cudaError_t launchDense(DeviceBinMatConstView left, DeviceBinMatConstView right,
                        size_t planes, size_t imgHeight,
                        const DenseDisparityParams& params,
                        DeviceImageView<uint8_t> disparity, cudaStream_t stream) {
    const size_t width = left.width;
    BINCV_ASSERT(left.width == right.width && left.height == right.height,
                 "cuda denseDisparity: the pair must share its extent");
    BINCV_ASSERT(left.height == planes * imgHeight,
                 "cuda denseDisparity: plane block height must be planes * imageHeight");
    BINCV_ASSERT(disparity.width == width && disparity.height == imgHeight,
                 "cuda denseDisparity: disparity extent must match the image");
    BINCV_ASSERT(params.winWidth >= 3 && params.winHeight >= 3 &&
                     (params.winWidth & 1) == 1 && (params.winHeight & 1) == 1,
                 "cuda denseDisparity: the window must be odd and at least 3 on a side");
    BINCV_ASSERT(params.minDisparity >= 0 &&
                     params.maxDisparity >= params.minDisparity &&
                     params.maxDisparity <= 254,
                 "cuda denseDisparity: need 0 <= min <= max <= 254 (255 marks invalid)");
    if (width == 0 || imgHeight == 0) return cudaSuccess;
    BINCV_ASSERT(left.ptr != nullptr && right.ptr != nullptr && disparity.ptr != nullptr,
                 "cuda denseDisparity: non-empty views need non-null pointers");

    const long long dMaxSupported = static_cast<long long>(width) -
                                    static_cast<long long>(params.winWidth);
    const int dEnd = params.maxDisparity <= dMaxSupported
                         ? params.maxDisparity
                         : static_cast<int>(dMaxSupported);
    if (imgHeight < static_cast<size_t>(params.winHeight) ||
        static_cast<size_t>(params.winWidth) > width || dEnd < params.minDisparity) {
        // No candidate anywhere: the whole map is the invalid marker, the same
        // early-out the host takes.
        return cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                                 disparity.width, disparity.height, stream);
    }

    // Both fast arms carry one static bound: a window no wider than a single
    // extraction (winW <= 32). A wider window -- and the reference arm the
    // switch forces -- goes through the straightforward kernel.
    if (impl::denseFastArmEnabled() && params.winWidth <= 32) {
        // Borders first: neither fast kernel writes a pixel no candidate can
        // serve, so the rim's invalid marker comes from one cheap fill rather
        // than from per-thread branches in the hot kernel.
        cudaError_t err =
            cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                              disparity.width, disparity.height, stream);
        if (err != cudaSuccess) return err;
        const size_t outRows = imgHeight - 2 * static_cast<size_t>(params.winHeight / 2);
        // The word-parallel arm is the BINARY path's: at one plane a candidate's
        // raw cost is one bit per pixel, which is what makes 32 pixels fit one
        // xor. A census plane block carries K bits per pixel and has no such
        // form, so it stays on the per-pixel sliding kernel.
        if (impl::denseBitSlicedEnabled() && planes == 1) {
            return launchBitSliced(left, right, params.minDisparity, dEnd, params,
                                   disparity, outRows, rowWords(width), stream);
        }
        constexpr unsigned kColBlock = 128;
        const dim3 grid(static_cast<unsigned>((width + kColBlock - 1) / kColBlock),
                        static_cast<unsigned>((outRows + kStrip - 1) / kStrip));
        denseKernelSliding<<<grid, kColBlock, 0, stream>>>(
            left, right, params.minDisparity, dEnd, params.winWidth, params.winHeight,
            static_cast<int>(planes), imgHeight, disparity, outRows);
        return cudaGetLastError();
    }
    const dim3 block(128, 1);
    const dim3 grid(static_cast<unsigned>((width + block.x - 1) / block.x),
                    static_cast<unsigned>(imgHeight));
    denseKernel<<<grid, block, 0, stream>>>(left, right, params.minDisparity, dEnd,
                                            params.winWidth, params.winHeight,
                                            static_cast<int>(planes), imgHeight,
                                            disparity);
    return cudaGetLastError();
}

} // namespace

namespace impl {
bool& denseFastArmEnabled() {
    static bool on = true;
    return on;
}
bool& denseBitSlicedEnabled() {
    static bool on = true;
    return on;
}
} // namespace impl

cudaError_t denseDisparityBinary(DeviceBinMatConstView left,
                                 DeviceBinMatConstView right,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream) {
    // The host binary path keeps its extraction in bytes and asserts this; the
    // cost fits wider registers here, but the accepted domain stays the
    // host's so "equal by test" is a statement about the same inputs.
    BINCV_ASSERT(params.winWidth * params.winHeight <= 255,
                 "cuda denseDisparityBinary: winWidth * winHeight must fit a byte");
    return launchDense(left, right, 1, left.height, params, disparity, stream);
}

cudaError_t denseDisparityCensusPacked(DeviceImageConstView<uint32_t> leftDesc,
                                       DeviceImageConstView<uint32_t> rightDesc,
                                       const DenseDisparityParams& params,
                                       DeviceImageView<uint8_t> disparity,
                                       cudaStream_t stream) {
    const size_t width = leftDesc.width, height = leftDesc.height;
    BINCV_ASSERT(leftDesc.width == rightDesc.width && leftDesc.height == rightDesc.height,
                 "cuda denseDisparityCensusPacked: the pair must share its extent");
    BINCV_ASSERT(disparity.width == width && disparity.height == height,
                 "cuda denseDisparityCensusPacked: disparity extent must match");
    BINCV_ASSERT(params.winWidth >= 3 && params.winHeight >= 3 &&
                     (params.winWidth & 1) == 1 && (params.winHeight & 1) == 1,
                 "cuda denseDisparityCensusPacked: window odd and >= 3 on a side");
    BINCV_ASSERT(params.minDisparity >= 0 &&
                     params.maxDisparity >= params.minDisparity &&
                     params.maxDisparity <= 254,
                 "cuda denseDisparityCensusPacked: need 0 <= min <= max <= 254");
    if (width == 0 || height == 0) return cudaSuccess;
    BINCV_ASSERT(leftDesc.ptr != nullptr && rightDesc.ptr != nullptr &&
                     disparity.ptr != nullptr,
                 "cuda denseDisparityCensusPacked: non-empty views need pointers");

    const long long dMaxSupported =
        static_cast<long long>(width) - static_cast<long long>(params.winWidth);
    const int dEnd = params.maxDisparity <= dMaxSupported
                         ? params.maxDisparity
                         : static_cast<int>(dMaxSupported);
    if (height < static_cast<size_t>(params.winHeight) ||
        static_cast<size_t>(params.winWidth) > width || dEnd < params.minDisparity) {
        return cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                                 disparity.width, disparity.height, stream);
    }
    cudaError_t err =
        cudaMemset2DAsync(disparity.ptr, disparity.stride, kDenseDisparityInvalid,
                          disparity.width, disparity.height, stream);
    if (err != cudaSuccess) return err;
    const size_t outRows = height - 2 * static_cast<size_t>(params.winHeight / 2);
    const dim3 grid(static_cast<unsigned>((width + kPackBlock - 1) / kPackBlock),
                    static_cast<unsigned>((outRows + kStrip - 1) / kStrip));
    // MEASURED NEGATIVE, do not retry on this shape: staging each pixel pair's
    // raw cost in shared memory so the winWidth overlapping windows share one
    // load -- the obvious next step, since neighbouring threads' windows differ by
    // one pixel -- measured 1.16 ms against this kernel's 0.91 ms, 1.28x
    // SLOWER. The redundant loads were already L1 hits, so what the staging
    // bought was nothing and what it cost was 624 __syncthreads() per block
    // plus byte-wide shared-memory bank conflicts.
    denseKernelPacked<<<grid, kPackBlock, 0, stream>>>(leftDesc, rightDesc,
                                                       params.minDisparity, dEnd,
                                                       params.winWidth, params.winHeight,
                                                       disparity, outRows);
    return cudaGetLastError();
}

cudaError_t denseDisparityCensus(DeviceBinMatConstView leftPlanes,
                                 DeviceBinMatConstView rightPlanes, size_t planes,
                                 size_t imageHeight,
                                 const DenseDisparityParams& params,
                                 DeviceImageView<uint8_t> disparity,
                                 cudaStream_t stream) {
    BINCV_ASSERT(planes >= 1 && planes <= 32,
                 "cuda denseDisparityCensus: 1 to 32 planes");
    return launchDense(leftPlanes, rightPlanes, planes, imageHeight, params, disparity,
                       stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
