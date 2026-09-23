// The device arm of ops/corner.hpp. Two response arms, two whole-operation
// arms, one selection kernel.
//
// THE RESPONSE. `bincv::impl::minEigenValue` carries BINCV_HOST_DEVICE and the
// CUDA suite already sweeps nvcc's device compilation of it against the host's,
// so the kernels CALL it -- there is no transcription and no uploaded table to
// stand in for one. The same is true of the 3x3 box sums: `boxHorizontal3`,
// `boxVertical3`, `boxValueAt` and `boxWordAt` are the host's own functions,
// and the full adder that makes the box sum exact has one definition.
//
// THE ONE PIECE OF IT THAT IS MEMOIZED is the square root, and only where a 3x3
// window bounds its argument to an integer: `cornerSqrtMemo.cuh` holds
// `sqrt(d^2 + 4c^2)` for `|d|, |c| <= 9` and the file header there states why a
// memo of `sqrt` is not a second definition of the response. The formula, the
// order of its operations and every other caller are unchanged.
//
// SLICED ARM (blockSize 3). One thread per output WORD. A box sum of bits is
// word-parallel -- horizontally `a(x-1) + a(x) + a(x+1)` is ONE full adder into
// two planes, vertically three of those sum into four planes (0..9 fits
// exactly) -- so ~80 integer operations advance 32 pixels of all four sums. The
// host's sparsity skip ports unchanged. What is NOT ported is the host's 8x8 bit
// transpose in the extraction: that is a CPU device for turning per-bit gathers
// into byte moves, and on a machine whose variable shift is one instruction
// there is nothing to transpose.
//
// WINDOW ARM (any blockSize). One thread per PIXEL, counting the clipped window
// directly. It is the oracle for the sliced arm at blockSize 3 and the only arm
// above it, which is also why `blockSize = 7` is the control case that must read
// ~1.00x between switch positions.
//
// FUSED ARM. A block computes a (kFuseWords + 2) x (kFuseRows + 2) word tile of
// responses into shared memory, reduces the frame maximum over the pixels it
// OWNS, runs the 3x3 suppression inside the tile and warp-aggregates the
// survivors into the candidate buffer. No frame-sized float map exists at any
// point. The reference arm writes one and scans it, and the two are held to the
// same corner array in one binary.
//
// TILE GEOMETRY IS NOT SWEPT, and that is stated rather than implied: 8 words by
// 8 rows with a one-pixel apron is 12,800 B of shared memory and was chosen to
// fit three blocks per SM, not measured against 16x8 or 8x16. The dense
// matcher's recorded negatives say a tile sweep must be measured and not
// reasoned, so this one is an open item and not a result.
//
// THE SELECTION. One block, because the greedy spacing filter is sequential in
// its acceptances and nothing else in the tail is large. The shape is NOT the
// obvious serial-outer/parallel-inner loop, which pays a block barrier per
// RANKED candidate: accepted points only ever accumulate, so a candidate killed
// by an accepted point stays killed, and the first surviving candidate is always
// the next one the host's loop would accept. That turns the barrier count from
// `ranked` (thousands) into `kept` (at most `maxCorners`) for the same total
// distance work.

#include "bincv/cuda/corner.hpp"

#include <cmath>
#include <cstdint>

#include "bincv/cuda/cornerSqrtMemo.cuh"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

/// @brief The four derivative planes, travelling together.
struct CovViews {
    DeviceBinMatConstView magX;
    DeviceBinMatConstView magY;
    DeviceBinMatConstView signX;
    DeviceBinMatConstView signY;
};

// ---------------------------------------------------------------------------
// The response, two ways
// ---------------------------------------------------------------------------

/// @brief 32 responses of row `y`, word `w`, at blockSize 3. Bit-sliced.
/// @param out 32 floats. Lanes past `width` are written 0 -- they are not frame
/// pixels, and their box sums are NOT zero (the window of `x == width` reaches
/// `x - 1`, which is a real pixel), so leaving them would put a response where
/// the host writes nothing.
/// @tparam Memo true to look the square root up (cornerSqrtMemo.cuh), false to
/// evaluate `impl::minEigenValue` in full. A TEMPLATE PARAMETER and not a
/// runtime flag: threading a runtime flag into a device helper of this family
/// took `fusedCandidateKernel` from 64 to 89 registers and +9.5% instructions
/// in the OFF position, which moves the baseline the ON position is measured
/// against.
template <bool Memo>
__device__ void slicedResponseWord(const CovViews& v, size_t words, size_t y, size_t w,
                                   float* out) {
    const uint32_t* mxr[3];
    const uint32_t* myr[3];
    const uint32_t* sxr[3];
    const uint32_t* syr[3];
    for (int k = -1; k <= 1; ++k) {
        const long long r = static_cast<long long>(y) + k;
        const bool ok = r >= 0 && r < static_cast<long long>(v.magX.height);
        const int i = k + 1;
        const size_t ur = static_cast<size_t>(r < 0 ? 0 : r);
        mxr[i] = ok ? v.magX.row(ur) : nullptr;
        myr[i] = ok ? v.magY.row(ur) : nullptr;
        sxr[i] = ok ? v.signX.row(ur) : nullptr;
        syr[i] = ok ? v.signY.row(ur) : nullptr;
    }

    // THE SPARSITY SKIP, the host's. A word's outputs depend on source columns
    // in words w-1, w and w+1 of three rows; if magX | magY is zero across all
    // nine, every box sum is zero and the response is exactly 0 for 32 pixels.
    uint32_t any = 0u;
    for (int i = 0; i < 3; ++i) {
        for (long long d = -1; d <= 1; ++d) {
            const long long ww = static_cast<long long>(w) + d;
            any |= bincv::impl::boxWordAt<uint32_t>(mxr[i], words, ww) |
                   bincv::impl::boxWordAt<uint32_t>(myr[i], words, ww);
        }
    }
    const long long x0 = static_cast<long long>(w) * 32;
    if (any == 0u) {
        for (int b = 0; b < 32; ++b) out[b] = 0.0f;
        return;
    }

    uint32_t hA[3][2], hB[3][2], hP[3][2], hN[3][2];
    for (int i = 0; i < 3; ++i) {
        uint32_t ca, cb, cp, cn, pa, pb, pp, pn, na, nb, np, nn;
        const long long wc = static_cast<long long>(w);
        // magnitude, magnitude, and the cross term split by signX ^ signY
        const auto load = [&](long long ww, uint32_t& oa, uint32_t& ob, uint32_t& op,
                              uint32_t& on) {
            const uint32_t a = bincv::impl::boxWordAt<uint32_t>(mxr[i], words, ww);
            const uint32_t b = bincv::impl::boxWordAt<uint32_t>(myr[i], words, ww);
            const uint32_t sel = bincv::impl::boxWordAt<uint32_t>(sxr[i], words, ww) ^
                                 bincv::impl::boxWordAt<uint32_t>(syr[i], words, ww);
            const uint32_t both = a & b;
            oa = a;
            ob = b;
            on = both & sel;
            op = both ^ on;
        };
        load(wc, ca, cb, cp, cn);
        load(wc - 1, pa, pb, pp, pn);
        load(wc + 1, na, nb, np, nn);
        // Pixel x lives at bit x % 32, so column x-1 is one bit lower: a LEFT
        // shift brings it to x, pulling in the previous word's top bit.
        const auto sl = [](uint32_t cur, uint32_t prev) { return (cur << 1) | (prev >> 31); };
        const auto sr = [](uint32_t cur, uint32_t next) { return (cur >> 1) | (next << 31); };
        bincv::impl::boxHorizontal3<uint32_t>(sl(ca, pa), ca, sr(ca, na), hA[i][0], hA[i][1]);
        bincv::impl::boxHorizontal3<uint32_t>(sl(cb, pb), cb, sr(cb, nb), hB[i][0], hB[i][1]);
        bincv::impl::boxHorizontal3<uint32_t>(sl(cp, pp), cp, sr(cp, np), hP[i][0], hP[i][1]);
        bincv::impl::boxHorizontal3<uint32_t>(sl(cn, pn), cn, sr(cn, nn), hN[i][0], hN[i][1]);
    }

    uint32_t vA[4], vB[4], vP[4], vN[4];
    bincv::impl::boxVertical3<uint32_t>(hA[0], hA[1], hA[2], vA);
    bincv::impl::boxVertical3<uint32_t>(hB[0], hB[1], hB[2], vB);
    bincv::impl::boxVertical3<uint32_t>(hP[0], hP[1], hP[2], vP);
    bincv::impl::boxVertical3<uint32_t>(hN[0], hN[1], hN[2], vN);

    for (int b = 0; b < 32; ++b) {
        if (x0 + b >= static_cast<long long>(v.magX.width)) {
            out[b] = 0.0f;
            continue;
        }
        const size_t bit = static_cast<size_t>(b);
        const long long xx = bincv::impl::boxValueAt<uint32_t>(vA, bit);
        const long long yy = bincv::impl::boxValueAt<uint32_t>(vB, bit);
        const long long pos = bincv::impl::boxValueAt<uint32_t>(vP, bit);
        const long long neg = bincv::impl::boxValueAt<uint32_t>(vN, bit);
        // The two arms differ in NOTHING but the root -- same operands, same
        // expression, same order -- so they agree bit for bit rather than
        // closely, and the suite pins that over the whole reachable domain. The
        // branch is `if constexpr` and the expression is written out twice
        // rather than factored behind a helper: an extra inlining layer moved
        // this kernel's OFF position by eight SASS instructions, and an OFF
        // position that is not the committed kernel is not a baseline.
        if constexpr (Memo) {
            out[b] = ((xx | yy | pos | neg) == 0)
                         ? 0.0f
                         : impl::minEigenValueMemo3(xx, yy, pos - neg);
        } else {
            out[b] = ((xx | yy | pos | neg) == 0)
                         ? 0.0f
                         : bincv::impl::minEigenValue(xx, yy, pos - neg);
        }
    }
}

/// @brief One pixel's response at any blockSize, from the clipped window.
/// @note The window is `impl::blockWindow`'s -- anchored where `cv::boxFilter`
/// anchors it -- and CLIPS at the frame edge, which is the host's promise 2.
/// The three sums are popcounts of exact integers, so they cannot differ from
/// the host's by a traversal order.
/// @note NO SQUARE-ROOT MEMO HERE, and that is the gate rather than an
/// oversight: `blockSize` arrives at runtime, so the discriminant is bounded by
/// `5*blockSize^4` and not by a table. This arm evaluates the root, which is
/// also what makes a blockSize other than 3 the control that must read ~1.00x
/// between the memo switch's positions.
__device__ float windowResponse(const CovViews& v, int blockSize, long long x, long long y) {
    const long long off = blockSize / 2;
    long long xs = x - off, ys = y - off;
    long long xe = xs + blockSize, ye = ys + blockSize;
    if (xs < 0) xs = 0;
    if (ys < 0) ys = 0;
    if (xe > static_cast<long long>(v.magX.width)) xe = static_cast<long long>(v.magX.width);
    if (ye > static_cast<long long>(v.magX.height)) ye = static_cast<long long>(v.magX.height);
    if (xs >= xe || ys >= ye) return bincv::impl::minEigenValue(0, 0, 0);

    long long xx = 0, yy = 0, xy = 0;
    for (long long r = ys; r < ye; ++r) {
        const size_t ur = static_cast<size_t>(r);
        const uint32_t* mx = v.magX.row(ur);
        const uint32_t* my = v.magY.row(ur);
        const uint32_t* sx = v.signX.row(ur);
        const uint32_t* sy = v.signY.row(ur);
        const long long wLo = xs / 32, wHi = (xe - 1) / 32;
        for (long long wi = wLo; wi <= wHi; ++wi) {
            const long long lo = wi * 32 > xs ? 0 : xs - wi * 32;
            const long long hiExcl = xe >= (wi + 1) * 32 ? 32 : xe - wi * 32;
            uint32_t m = 0xFFFFFFFFu;
            if (lo > 0) m &= ~static_cast<uint32_t>((1ull << lo) - 1ull);
            if (hiExcl < 32) m &= static_cast<uint32_t>((1ull << hiExcl) - 1ull);
            const size_t uw = static_cast<size_t>(wi);
            const uint32_t a = mx[uw] & m;
            const uint32_t b = my[uw] & m;
            const uint32_t both = a & b;
            const uint32_t neg = both & (sx[uw] ^ sy[uw]);
            const uint32_t pos = both ^ neg;
            xx += __popc(static_cast<int>(a));
            yy += __popc(static_cast<int>(b));
            xy += __popc(static_cast<int>(pos)) - __popc(static_cast<int>(neg));
        }
    }
    return bincv::impl::minEigenValue(xx, yy, xy);
}

// ---------------------------------------------------------------------------
// cornerMinEigenValAsync
// ---------------------------------------------------------------------------

// The map is written THROUGH SHARED MEMORY, and the reason is measured rather
// than stylistic. One thread owns one word, so a thread's 32 responses are 128
// consecutive bytes and consecutive threads are 128 bytes apart: every store
// instruction of the direct spelling touched 32 different 32-byte sectors and
// filled one float of each. The profiler read 360,960 global store sectors for a
// 1.44 MB map -- 11.5 MB of traffic for 1.44 MB of answer -- with
// `short_scoreboard` at 78% of the stall samples and 46,080 local-load sectors
// beside it, because the 32-float staging array was passed by pointer and so
// lived in local memory. Staging the block's 128 words in shared instead and
// re-reading them one pixel per thread makes the store a plain coalesced row
// write and removes the local array entirely. The 33-float pitch is the
// bank-conflict pad: at 32 the whole warp would hit one bank on every write.
constexpr int kRespThreads = 128;
constexpr int kRespPitch = 33;

template <bool Memo>
__global__ void responseKernelSliced(CovViews v, DeviceImageView<float> dst, size_t words) {
    __shared__ float stage[kRespThreads][kRespPitch];
    const size_t total = words * v.magX.height;
    const size_t stride = static_cast<size_t>(gridDim.x) * kRespThreads;
    for (size_t base = static_cast<size_t>(blockIdx.x) * kRespThreads; base < total;
         base += stride) {
        const size_t idx = base + threadIdx.x;
        if (idx < total) {
            const size_t y = idx / words;
            const size_t w = idx - y * words;
            slicedResponseWord<Memo>(v, words, y, w, &stage[threadIdx.x][0]);
        }
        __syncthreads();
        const size_t units = (total - base) < static_cast<size_t>(kRespThreads)
                                 ? (total - base)
                                 : static_cast<size_t>(kRespThreads);
        for (size_t i = threadIdx.x; i < units * 32; i += kRespThreads) {
            const size_t u = i >> 5;
            const size_t b = i & 31u;
            const size_t idx2 = base + u;
            const size_t y = idx2 / words;
            const size_t w = idx2 - y * words;
            const size_t x = w * 32 + b;
            if (x < dst.width) dst.row(y)[x] = stage[u][b];
        }
        __syncthreads();
    }
}

__global__ void responseKernelWindow(CovViews v, DeviceImageView<float> dst, int blockSize) {
    const size_t total = dst.width * dst.height;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t y = idx / dst.width;
        const size_t x = idx - y * dst.width;
        dst.row(y)[x] = windowResponse(v, blockSize, static_cast<long long>(x),
                                       static_cast<long long>(y));
    }
}

// ---------------------------------------------------------------------------
// Candidates: the frame maximum, the 3x3 suppression, the append
// ---------------------------------------------------------------------------

constexpr int kFuseWords = 8;                          // 256 owned pixels per tile row
constexpr int kFuseRows = 8;                           // owned rows per tile
constexpr int kFuseTileWords = kFuseWords + 2;         // one apron word each side
constexpr int kFuseTileRows = kFuseRows + 2;           // one apron row each side
constexpr int kFuseCols = kFuseTileWords * 32;
constexpr int kFuseThreads = 128;
// ONE BLOCK does the whole selection -- a bitonic network needs a barrier
// between every stage, and __syncthreads() is the only barrier a kernel has. So
// the block size divides the entire tail, and 1024 is the largest this device
// takes. Measured: at 256 threads the selection was 24.4 ms of a 24.5 ms
// operation on a frame with 42,153 raw maxima. It is still the operation's cost
// centre and that is recorded rather than hidden.
constexpr int kSelThreads = 1024;

/// @brief Warp-aggregated append of at most one candidate per lane.
/// @note Every lane of the warp must reach this; the loops that call it have a
/// compile-time-constant trip count, so they do.
__device__ inline void appendCandidateWarp(const DeviceCornerBuffer& buf, bool hit,
                                           const DeviceCorner& c) {
    const unsigned ballot = __ballot_sync(0xFFFFFFFFu, hit);
    const unsigned lane = threadIdx.x & 31u;
    uint32_t base = 0u;
    if (lane == 0u && ballot != 0u) {
        base = buf.reserve(static_cast<uint32_t>(__popc(static_cast<int>(ballot))));
    }
    base = __shfl_sync(0xFFFFFFFFu, base, 0);
    if (hit) {
        const unsigned below = ballot & ((1u << lane) - 1u);
        buf.store(base + static_cast<uint32_t>(__popc(static_cast<int>(below))), c);
    }
}

/// @brief Block maximum of `myMax`, then ONE atomicMax into the frame maximum.
/// @note The bit pattern comparison IS the float comparison here: corner.hpp's
/// PRECISION section proves a response is either exactly 0.0f or at least
/// `1/(2*blockSize^2)`, so it is never negative and never NaN.
template <int Threads>
__device__ inline void reduceFrameMax(float myMax, float* shared, uint32_t* maxBits) {
    shared[threadIdx.x] = myMax;
    __syncthreads();
    for (int s = Threads / 2; s > 0; s >>= 1) {
        if (static_cast<int>(threadIdx.x) < s) {
            const float o = shared[threadIdx.x + s];
            if (o > shared[threadIdx.x]) shared[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(maxBits, __float_as_uint(shared[0]));
}

template <bool Sliced, bool Memo>
__global__ void fusedCandidateKernel(CovViews v, int blockSize, DeviceCornerBuffer cand,
                                     uint32_t* maxBits, size_t words) {
    __shared__ float resp[kFuseTileRows][kFuseCols];
    __shared__ float redux[kFuseThreads];

    const long long w0 = static_cast<long long>(blockIdx.x) * kFuseWords;
    const long long y0 = static_cast<long long>(blockIdx.y) * kFuseRows;
    const long long width = static_cast<long long>(v.magX.width);
    const long long height = static_cast<long long>(v.magX.height);

    // Phase A: the tile's responses, one (row, word) unit per thread.
    for (int u = static_cast<int>(threadIdx.x); u < kFuseTileRows * kFuseTileWords;
         u += kFuseThreads) {
        const int r = u / kFuseTileWords;
        const int cw = u - r * kFuseTileWords;
        const long long gy = y0 + r - 1;
        const long long gw = w0 + cw - 1;
        float* out = &resp[r][cw * 32];
        if (gy < 0 || gy >= height || gw < 0 || gw >= static_cast<long long>(words)) {
            for (int b = 0; b < 32; ++b) out[b] = 0.0f;
        } else if (Sliced) {
            slicedResponseWord<Memo>(v, words, static_cast<size_t>(gy), static_cast<size_t>(gw),
                                     out);
        } else {
            for (int b = 0; b < 32; ++b) {
                const long long x = gw * 32 + b;
                out[b] = x < width ? windowResponse(v, blockSize, x, gy) : 0.0f;
            }
        }
    }
    __syncthreads();

    // Phase B: the frame maximum over the pixels this tile OWNS -- border row
    // and column included, because that is the region cv::minMaxLoc covers and
    // the host's maxVal with it -- and the 3x3 suppression over the interior.
    float myMax = 0.0f;
    for (int p = static_cast<int>(threadIdx.x); p < kFuseWords * 32 * kFuseRows;
         p += kFuseThreads) {
        const int py = p / (kFuseWords * 32);
        const int px = p - py * (kFuseWords * 32);
        const long long x = w0 * 32 + px;
        const long long y = y0 + py;
        bool hit = false;
        DeviceCorner c;
        c.x = 0;
        c.y = 0;
        c.response = 0.0f;
        if (x < width && y < height) {
            const int r = py + 1;
            const int cc = px + 32;
            const float val = resp[r][cc];
            if (val > myMax) myMax = val;
            if (val > 0.0f && x >= 1 && x + 1 < width && y >= 1 && y + 1 < height) {
                bool isMax = true;
                for (int dx = -1; dx <= 1 && isMax; ++dx) {
                    if (resp[r - 1][cc + dx] > val || resp[r][cc + dx] > val ||
                        resp[r + 1][cc + dx] > val) {
                        isMax = false;
                    }
                }
                if (isMax) {
                    hit = true;
                    c.x = static_cast<int>(x);
                    c.y = static_cast<int>(y);
                    c.response = val;
                }
            }
        }
        appendCandidateWarp(cand, hit, c);
    }
    reduceFrameMax<kFuseThreads>(myMax, redux, maxBits);
}

/// @brief The reference arm's candidate pass, over a materialised frame map.
__global__ void mapCandidateKernel(DeviceImageConstView<float> map, DeviceCornerBuffer cand,
                                   uint32_t* maxBits) {
    __shared__ float redux[kSelThreads];
    const size_t total = map.width * map.height;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const long long width = static_cast<long long>(map.width);
    const long long height = static_cast<long long>(map.height);

    float myMax = 0.0f;
    for (size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x; base < total;
         base += stride) {
        const size_t idx = base + threadIdx.x;
        bool hit = false;
        DeviceCorner c;
        c.x = 0;
        c.y = 0;
        c.response = 0.0f;
        if (idx < total) {
            const size_t y = idx / map.width;
            const size_t x = idx - y * map.width;
            const float val = map.row(y)[x];
            if (val > myMax) myMax = val;
            const long long lx = static_cast<long long>(x), ly = static_cast<long long>(y);
            if (val > 0.0f && lx >= 1 && lx + 1 < width && ly >= 1 && ly + 1 < height) {
                const float* prev = map.row(y - 1);
                const float* cur = map.row(y);
                const float* next = map.row(y + 1);
                bool isMax = true;
                for (int dx = -1; dx <= 1 && isMax; ++dx) {
                    const size_t q = static_cast<size_t>(lx + dx);
                    if (prev[q] > val || cur[q] > val || next[q] > val) isMax = false;
                }
                if (isMax) {
                    hit = true;
                    c.x = static_cast<int>(x);
                    c.y = static_cast<int>(y);
                    c.response = val;
                }
            }
        }
        appendCandidateWarp(cand, hit, c);
    }
    reduceFrameMax<kSelThreads>(myMax, redux, maxBits);
}

// ---------------------------------------------------------------------------
// The selection
// ---------------------------------------------------------------------------
//
// WHAT USED TO BE HERE, AND WHY IT IS NOT. One block did the whole tail: a
// bitonic network over `nextPow2(S)` corner records in global memory, then a
// greedy spacing filter that rescanned every surviving candidate once per
// acceptance. On the reference frame -- 25,115 candidates, 200 corners kept --
// that was 1.29% of one SM's throughput on a 48-SM part and it was 98% of the
// operation. Measured split, same frame, same binary: candidates 0.13 ms, sort
// 5.6 ms, spacing 23.1 ms.
//
// Two changes, and both come out of the same observation -- a corner's ORDER is
// smaller than the corner.
//
// 1. **THE ORDERING KEY IS THE CORNER.** `impl::CornerStronger` is response
//    descending, then y descending, then x descending. A response is either
//    exactly 0.0f or at least 1/(2*blockSize^2) (corner.hpp's PRECISION
//    section), so it is never negative and never NaN and its IEEE bit pattern
//    orders exactly as the float does. Packing
//
//        key = (~responseBits) << 32 | (0xFFFF - y) << 16 | (0xFFFF - x)
//
//    makes ASCENDING `uint64_t` order identical to CornerStronger, and the pack
//    is LOSSLESS -- the corner is read back out of the key with three shifts.
//    So the sort moves 8 bytes per element instead of 16, the greedy reads a
//    position out of 4 bytes instead of a 16-byte record, and no separate
//    payload array exists. The keys are unique (positions are), so the sorted
//    sequence is unique and the sort's own order-dependence cannot reach the
//    answer -- the same argument the append order already rests on.
//
// 2. **THE SPACING FILTER TESTS, IT DOES NOT KILL.** The two formulations are
//    the same answer: a candidate is alive when it is reached exactly when no
//    ACCEPTED point is within `minDistance` of it, and accepted points only
//    accumulate. Killing costs one pass over every surviving candidate per
//    acceptance -- 200 passes over 25,115 records through one SM. Testing costs
//    one pass over the candidates, each tested against the accepted set, which
//    is at most `maxCorners` points and lives in shared memory. The chunk is a
//    block's width: 1024 candidates are tested in parallel against the accepted
//    set, then the acceptances INSIDE that chunk are resolved serially, which
//    is at most one round per corner kept.
//
// The sort is a standard bitonic network, and it is device-wide: each block
// sorts 2048 keys in shared memory, and the stages wider than a chunk are their
// own launches because a block barrier is not a device barrier. The ladder is
// sized from `candidateCapacity`, which the host knows; each kernel reads the
// actual candidate count and returns immediately when its stage is past it, so
// a frame with few candidates pays launches rather than work.

/// @brief `impl::CornerStronger` as an ascending `uint64_t`. LOSSLESS.
/// @note The response is never negative and never NaN, so the bit pattern is
/// order-isomorphic to the float, and complementing it turns "stronger first"
/// into "smaller first". `x` and `y` are below 65536 -- the device domain
/// corner.hpp names -- so the tie rule is the low 32 bits, complemented for the
/// same reason.
__device__ inline uint64_t cornerKey(const DeviceCorner& c) {
    const uint32_t r = __float_as_uint(c.response);
    const uint32_t lo = ((0xFFFFu - static_cast<uint32_t>(c.y)) << 16) |
                        (0xFFFFu - static_cast<uint32_t>(c.x));
    return (static_cast<uint64_t>(~r) << 32) | static_cast<uint64_t>(lo);
}

__device__ inline int keyX(uint64_t k) {
    return static_cast<int>(0xFFFFu - (static_cast<uint32_t>(k) & 0xFFFFu));
}
__device__ inline int keyY(uint64_t k) {
    return static_cast<int>(0xFFFFu - ((static_cast<uint32_t>(k) >> 16) & 0xFFFFu));
}
__device__ inline DeviceCorner keyCorner(uint64_t k) {
    DeviceCorner c;
    c.x = keyX(k);
    c.y = keyY(k);
    c.response = __uint_as_float(~static_cast<uint32_t>(k >> 32));
    return c;
}

/// @brief The sentinel that pads a run up to a power of two. Sorts LAST, and a
/// real key can never equal it: it would need response -NaN at (-1, -1).
__device__ inline uint64_t keySentinel() { return 0xFFFFFFFFFFFFFFFFull; }

constexpr int kSortThreads = 512;
// 4096 keys = 32 KB of shared memory, which is what a block may hold
// statically. It is chosen for the LAUNCH COUNT rather than the occupancy:
// the ladder is one launch per doubling plus one per stage wider than a
// chunk, so doubling the chunk removes five launches from a 32,768-key
// sort -- 15 down to 10 -- and a launch is ~5 us of a sort whose work is
// ~20 us.
constexpr uint32_t kSortChunk = 4096;  // keys a block sorts in shared memory

__device__ inline uint32_t deviceNextPow2(uint32_t v) {
    uint32_t p = 1u;
    while (p < v) p <<= 1;
    return p;
}

/// @brief The quality cut, and the keys it survives as. **Multi-block.**
/// @note The append order does not matter: the keys are unique and everything
/// after this sorts them.
__global__ void keyKernel(const DeviceCorner* cand, const uint32_t* candCount,
                          uint32_t candCapacity, const uint32_t* maxBits, double qualityLevel,
                          uint64_t* keys, uint32_t* state) {
    const uint32_t found = *candCount;
    if (found > candCapacity) {
        if (blockIdx.x == 0u && threadIdx.x == 0u) state[1] = 1u;
        return;
    }
    // The reference's arithmetic exactly: the product is formed in double and
    // narrowed to float, and the comparison below is STRICTLY greater.
    const float thr =
        static_cast<float>(static_cast<double>(__uint_as_float(*maxBits)) * qualityLevel);
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const unsigned lane = threadIdx.x & 31u;
    for (size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x; base < found;
         base += stride) {
        const size_t i = base + threadIdx.x;
        bool hit = false;
        uint64_t k = 0ull;
        if (i < found) {
            const DeviceCorner c = cand[i];
            if (c.response > thr) {
                hit = true;
                k = cornerKey(c);
            }
        }
        const unsigned ballot = __ballot_sync(0xFFFFFFFFu, hit);
        uint32_t slot = 0u;
        if (lane == 0u && ballot != 0u) {
            slot = atomicAdd(&state[0], static_cast<uint32_t>(__popc(static_cast<int>(ballot))));
        }
        slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
        if (hit) {
            const unsigned below = ballot & ((1u << lane) - 1u);
            keys[slot + static_cast<uint32_t>(__popc(static_cast<int>(below)))] = k;
        }
    }
}

__device__ inline void keyExchange(uint64_t& a, uint64_t& b, bool ascending) {
    if ((a > b) == ascending) {
        const uint64_t t = a;
        a = b;
        b = t;
    }
}

/// @brief Sorts one `kSortChunk` block of keys in shared memory, padding the run
/// up to the power of two the network needs.
__global__ void bitonicChunkKernel(uint64_t* keys, const uint32_t* state, uint32_t capacity) {
    __shared__ uint64_t buf[kSortChunk];
    const uint32_t n = state[0] < capacity ? state[0] : capacity;
    const uint32_t n2 = deviceNextPow2(n < 1u ? 1u : n);
    const uint32_t start = blockIdx.x * kSortChunk;
    if (start >= n2) return;
    const uint32_t span = (n2 - start) < kSortChunk ? (n2 - start) : kSortChunk;

    for (uint32_t i = threadIdx.x; i < span; i += kSortThreads) {
        const uint32_t g = start + i;
        buf[i] = g < n ? keys[g] : keySentinel();
    }
    __syncthreads();
    for (uint32_t k = 2u; k <= span; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < span; i += kSortThreads) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    // The direction is the GLOBAL index's k bit, not the local
                    // one's: at k == kSortChunk the two differ, and every other
                    // chunk would then be sorted the wrong way round.
                    keyExchange(buf[i], buf[l], ((start + i) & k) == 0u);
                }
            }
            __syncthreads();
        }
    }
    for (uint32_t i = threadIdx.x; i < span; i += kSortThreads) keys[start + i] = buf[i];
}

/// @brief One bitonic stage wider than a chunk. **One launch, because a block
/// barrier is not a device barrier.**
__global__ void bitonicStageKernel(uint64_t* keys, const uint32_t* state, uint32_t capacity,
                                   uint32_t k, uint32_t j) {
    const uint32_t n = state[0] < capacity ? state[0] : capacity;
    const uint32_t n2 = deviceNextPow2(n < 1u ? 1u : n);
    if (k > n2) return;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += stride) {
        const uint32_t l = i ^ j;
        if (l > i) keyExchange(keys[i], keys[l], (i & k) == 0u);
    }
}

/// @brief The tail of one bitonic merge -- every stage with `j < kSortChunk` --
/// done inside shared memory, which is what keeps the launch count at one per
/// doubling rather than one per stage.
__global__ void bitonicMergeKernel(uint64_t* keys, const uint32_t* state, uint32_t capacity,
                                   uint32_t k) {
    __shared__ uint64_t buf[kSortChunk];
    const uint32_t n = state[0] < capacity ? state[0] : capacity;
    const uint32_t n2 = deviceNextPow2(n < 1u ? 1u : n);
    if (k > n2) return;
    const uint32_t start = blockIdx.x * kSortChunk;
    if (start >= n2) return;
    for (uint32_t i = threadIdx.x; i < kSortChunk; i += kSortThreads) buf[i] = keys[start + i];
    __syncthreads();
    for (uint32_t j = kSortChunk >> 1; j > 0u; j >>= 1) {
        for (uint32_t i = threadIdx.x; i < kSortChunk; i += kSortThreads) {
            const uint32_t l = i ^ j;
            if (l > i) keyExchange(buf[i], buf[l], ((start + i) & k) == 0u);
        }
        __syncthreads();
    }
    for (uint32_t i = threadIdx.x; i < kSortChunk; i += kSortThreads) keys[start + i] = buf[i];
}

/// @brief The whole network in ONE block -- the reference arm the ladder above
/// is held to. Same keys, same network, one barrier domain.
__global__ void bitonicOneBlockKernel(uint64_t* keys, const uint32_t* state, uint32_t capacity) {
    const uint32_t n = state[0] < capacity ? state[0] : capacity;
    if (n < 2u) return;
    const uint32_t n2 = deviceNextPow2(n);
    for (uint32_t i = n + threadIdx.x; i < n2; i += kSelThreads) keys[i] = keySentinel();
    __syncthreads();
    for (uint32_t k = 2u; k <= n2; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < n2; i += kSelThreads) {
                const uint32_t l = i ^ j;
                if (l > i) keyExchange(keys[i], keys[l], (i & k) == 0u);
            }
            __syncthreads();
        }
    }
}

/// @brief Accepted points cached in shared memory; beyond the cache the output
/// array itself is the store, and every thread reads the same element, so the
/// read is a broadcast rather than a gather.
constexpr uint32_t kAccShared = 1024;

__device__ inline void acceptedAt(uint32_t i, const uint32_t* sAcc, const DeviceCorner* out,
                                  int& ax, int& ay) {
    if (i < kAccShared) {
        const uint32_t p = sAcc[i];
        ax = static_cast<int>(p & 0xFFFFu);
        ay = static_cast<int>(p >> 16);
    } else {
        ax = out[i].x;
        ay = out[i].y;
    }
}

/// @brief `dx*dx + dy*dy < minDistanceSq`, the host's comparison exactly.
/// @note dx and dy are integers well under 2^26, so both products and their sum
/// are exact in double and no contraction can change the comparison.
__device__ inline bool tooCloseDouble(int cx, int cy, int ax, int ay, double minDistanceSq) {
    const double dx = static_cast<double>(cx) - static_cast<double>(ax);
    const double dy = static_cast<double>(cy) - static_cast<double>(ay);
    return dx * dx + dy * dy < minDistanceSq;
}

/// @brief The same comparison, in INTEGERS, and it is the same answer.
/// @note This is the spacing filter's whole inner loop and the arithmetic in it
/// decides the operation's cost: `sm_86` runs FP64 at 1/64 of FP32, so the two
/// multiplies, the add and the two int-to-double conversions above are about
/// eighty cycles of one SM's two FP64 pipes per test, and the filter performs
/// `ranked x kept` of them. Measured on the reference frame, that was 6.8 ms of
/// a 7.9 ms detection.
/// @note **The equivalence is exact, not close.** `|dx|` and `|dy|` are below
/// 65536 (the device domain corner.hpp names), so `s = dx*dx + dy*dy` is a
/// non-negative integer below 2^33 -- exactly representable as a double, which
/// is why the host's own double form computes the same integer. With
/// `T = ceil(minDistanceSq)` taken once on the host, `(double)s < minDistanceSq`
/// and `s < T` are the same predicate for every non-negative integer `s`: if
/// `minDistanceSq` is an integer the two bounds coincide, and if it is not then
/// `s < T` means `s <= floor(minDistanceSq) < minDistanceSq`. The reference
/// spacing arm keeps the double form, so the suite holding the two arms to one
/// corner array in one binary is the proof rather than this paragraph.
__device__ inline bool tooCloseInt(int cx, int cy, int ax, int ay,
                                   unsigned long long minDistanceSqCeil) {
    const int dx = cx - ax;
    const int dy = cy - ay;
    const unsigned ux = static_cast<unsigned>(dx < 0 ? -dx : dx);
    const unsigned uy = static_cast<unsigned>(dy < 0 ? -dy : dy);
    const unsigned long long s = static_cast<unsigned long long>(ux) * ux +
                                 static_cast<unsigned long long>(uy) * uy;
    return s < minDistanceSqCeil;
}

/// @brief Threshold count, rank truncation and the greedy spacing filter.
/// @tparam Chunked true for the tested-in-parallel arm, false for the reference
/// arm that kills forward once per acceptance.
template <bool Chunked>
__global__ void spacingKernel(const uint64_t* keys, const uint32_t* state, uint32_t candCapacity,
                              double minDistanceSq, unsigned long long minDistanceSqCeil,
                              int spacing, uint32_t limit, DeviceCorner* out, uint32_t capacity,
                              uint8_t* alive, DeviceCornerResult* result) {
    __shared__ uint32_t sAcc[kAccShared];
    __shared__ uint32_t sFree[kSelThreads / 32];
    __shared__ uint32_t sKept;
    __shared__ int sPick;
    __shared__ uint32_t sCursor;
    __shared__ uint32_t sMin[kSelThreads];

    if (state[1] != 0u) {
        if (threadIdx.x == 0u) {
            result->count = 0u;
            result->candidatesRanked = 0u;
            result->candidatesTruncated = 0u;
            result->candidateOverflow = 1u;
        }
        return;
    }

    const uint32_t S = state[0] < candCapacity ? state[0] : candCapacity;
    const uint32_t ranked = S < capacity ? S : capacity;
    const uint32_t truncated = S > capacity ? 1u : 0u;

    if (spacing == 0) {
        // gftt.cpp's `else` branch: no spacing at all, just the strongest.
        const uint32_t kept = ranked < limit ? ranked : limit;
        for (uint32_t i = threadIdx.x; i < kept; i += kSelThreads) out[i] = keyCorner(keys[i]);
        __syncthreads();
        if (threadIdx.x == 0u) {
            result->count = kept;
            result->candidatesRanked = ranked;
            result->candidatesTruncated = truncated;
            result->candidateOverflow = 0u;
        }
        return;
    }

    if (threadIdx.x == 0u) {
        sKept = 0u;
        sCursor = 0u;
    }
    __syncthreads();

    if (Chunked) {
        const unsigned lane = threadIdx.x & 31u;
        const unsigned warp = threadIdx.x >> 5;
        for (uint32_t base = 0u; base < ranked; base += kSelThreads) {
            if (sKept >= limit) break;
            const uint32_t j = base + threadIdx.x;
            const bool valid = j < ranked;
            const uint64_t key = valid ? keys[j] : 0ull;
            const int cx = valid ? keyX(key) : 0;
            const int cy = valid ? keyY(key) : 0;

            // Every candidate of the chunk, against the accepted set as it stands.
            bool freeFlag = valid;
            const uint32_t keptNow = sKept;
            for (uint32_t a = 0u; a < keptNow && freeFlag; ++a) {
                int ax, ay;
                acceptedAt(a, sAcc, out, ax, ay);
                if (tooCloseInt(cx, cy, ax, ay, minDistanceSqCeil)) freeFlag = false;
            }

            // The acceptances INSIDE the chunk, resolved in order. At most one
            // round per corner kept over the whole call.
            for (;;) {
                const unsigned ballot = __ballot_sync(0xFFFFFFFFu, freeFlag);
                if (lane == 0u) sFree[warp] = ballot;
                __syncthreads();
                // The lowest free lane of the chunk, found by ONE warp with a
                // ballot rather than by thread 0 with a 32-word scan: every other
                // warp is sitting at the barrier below while this runs, so the
                // serial step is paid 32 times over.
                if (threadIdx.x < 32u) {
                    const uint32_t m = sFree[threadIdx.x];
                    const unsigned any = __ballot_sync(0xFFFFFFFFu, m != 0u);
                    if (threadIdx.x == 0u) {
                        int p = -1;
                        if (any != 0u) {
                            const int wi = __ffs(static_cast<int>(any)) - 1;
                            p = wi * 32 +
                                __ffs(static_cast<int>(sFree[static_cast<size_t>(wi)])) - 1;
                        }
                        if (p >= 0 && sKept < limit) {
                            const uint64_t k = keys[base + static_cast<uint32_t>(p)];
                            out[sKept] = keyCorner(k);
                            if (sKept < kAccShared) {
                                sAcc[sKept] = (static_cast<uint32_t>(keyY(k)) << 16) |
                                              static_cast<uint32_t>(keyX(k));
                            }
                            sKept += 1u;
                            sPick = p;
                        } else {
                            sPick = -1;
                        }
                    }
                }
                __syncthreads();
                if (sPick < 0) break;
                const uint32_t pick = static_cast<uint32_t>(sPick);
                int ax, ay;
                acceptedAt(sKept - 1u, sAcc, out, ax, ay);
                if (threadIdx.x <= pick) freeFlag = false;
                if (freeFlag && tooCloseInt(cx, cy, ax, ay, minDistanceSqCeil)) freeFlag = false;
                __syncthreads();
                if (sKept >= limit) break;
            }
            __syncthreads();
        }
    } else {
        // THE REFERENCE ARM: kill forward once per acceptance. One barrier per
        // ACCEPTANCE rather than per candidate, but one pass over every
        // surviving candidate per acceptance -- which is the cost the arm above
        // exists to not pay.
        for (uint32_t i = threadIdx.x; i < ranked; i += kSelThreads) alive[i] = 1u;
        __syncthreads();
        for (;;) {
            if (sKept >= limit) break;
            uint32_t best = 0xFFFFFFFFu;
            for (uint32_t i = sCursor + threadIdx.x; i < ranked; i += kSelThreads) {
                if (alive[i] != 0u) {
                    best = i;
                    break;
                }
            }
            sMin[threadIdx.x] = best;
            __syncthreads();
            for (int s = kSelThreads / 2; s > 0; s >>= 1) {
                if (static_cast<int>(threadIdx.x) < s) {
                    const uint32_t o = sMin[threadIdx.x + s];
                    if (o < sMin[threadIdx.x]) sMin[threadIdx.x] = o;
                }
                __syncthreads();
            }
            const uint32_t pick = sMin[0];
            if (pick == 0xFFFFFFFFu) break;
            const uint64_t k = keys[pick];
            const int ax = keyX(k), ay = keyY(k);
            if (threadIdx.x == 0u) {
                out[sKept] = keyCorner(k);
                alive[pick] = 0u;
                sKept += 1u;
                sCursor = pick + 1u;
            }
            __syncthreads();
            for (uint32_t j = pick + 1u + threadIdx.x; j < ranked; j += kSelThreads) {
                if (alive[j] == 0u) continue;
                const uint64_t kj = keys[j];
                if (tooCloseDouble(keyX(kj), keyY(kj), ax, ay, minDistanceSq)) alive[j] = 0u;
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0u) {
        result->count = sKept;
        result->candidatesRanked = ranked;
        result->candidatesTruncated = truncated;
        result->candidateOverflow = 0u;
    }
}

uint32_t nextPow2(uint32_t v) {
    uint32_t p = 1u;
    while (p < v) p <<= 1;
    return p;
}

size_t alignUp16(size_t n) { return (n + 15u) & ~static_cast<size_t>(15u); }

dim3 linearGrid(size_t units, unsigned threads) {
    const size_t blocks = (units + threads - 1) / threads;
    return dim3(static_cast<unsigned>(blocks < 8192 ? (blocks ? blocks : 1) : 8192));
}

bool dimensionsAgree(const CovViews& v) {
    return v.magX.width == v.magY.width && v.magX.height == v.magY.height &&
           v.magX.width == v.signX.width && v.magX.height == v.signX.height &&
           v.magX.width == v.signY.width && v.magX.height == v.signY.height;
}

} // namespace

namespace impl {

bool& cornerSlicedEnabled() {
    static bool on = true;
    return on;
}

bool cornerSlicedApplies(int blockSize) { return blockSize == 3; }

bool& cornerSqrtMemoEnabled() {
    static bool on = true;
    return on;
}

bool cornerSqrtMemoApplies(int blockSize) { return cornerSlicedApplies(blockSize); }

bool& cornerFusedEnabled() {
    static bool on = true;
    return on;
}

bool& cornerSortParallelEnabled() {
    static bool on = true;
    return on;
}

bool cornerSortParallelApplies(size_t candidateCapacity) {
    return nextPow2(static_cast<uint32_t>(candidateCapacity > 0xFFFFFFFFu
                                              ? 0xFFFFFFFFu
                                              : candidateCapacity)) > kSortChunk;
}

bool& cornerSpacingChunkedEnabled() {
    static bool on = true;
    return on;
}

bool cornerSpacingChunkedApplies(const GoodFeaturesParams& params) {
    return params.minDistance >= 1.0;
}

} // namespace impl

size_t goodFeaturesScratchBytes(size_t candidateCapacity) {
    if (candidateCapacity == 0) return 0;
    BINCV_ASSERT(candidateCapacity <= 0xFFFFFFFFu,
                 "goodFeaturesScratchBytes: capacity outside the device's uint32 domain");
    const size_t padded = nextPow2(static_cast<uint32_t>(candidateCapacity));
    // The ordering key (8 B, and it IS the corner), the reference spacing arm's
    // alive byte, and the two state words the candidate count and the overflow
    // flag live in. The key replaced a 16-byte record and a separate payload, so
    // this number went DOWN when the selection was parallelised.
    return alignUp16(padded * sizeof(uint64_t)) + alignUp16(padded) + 16;
}

cudaError_t cornerMinEigenValAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                                   DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                                   int blockSize, DeviceImageView<float> dst,
                                   cudaStream_t stream) {
    const CovViews v{magX, magY, signX, signY};
    BINCV_ASSERT(dimensionsAgree(v),
                 "cuda cornerMinEigenVal: the four derivative planes must have the same "
                 "dimensions");
    BINCV_ASSERT(dst.width == magX.width && dst.height == magX.height,
                 "cuda cornerMinEigenVal: the response map must have the planes' dimensions");
    BINCV_ASSERT(blockSize > 0, "cuda cornerMinEigenVal: blockSize must be positive");
    if (!dimensionsAgree(v) || blockSize <= 0) return cudaErrorInvalidValue;
    if (dst.width != magX.width || dst.height != magX.height) return cudaErrorInvalidValue;
    if (magX.width == 0 || magX.height == 0) return cudaSuccess;
    BINCV_ASSERT(dst.ptr != nullptr && magX.ptr != nullptr,
                 "cuda cornerMinEigenVal: a non-empty call needs non-null pointers");
    if (dst.ptr == nullptr || magX.ptr == nullptr) return cudaErrorInvalidValue;

    const size_t words = rowWords(magX.width);
    if (impl::cornerSlicedEnabled() && impl::cornerSlicedApplies(blockSize)) {
        const dim3 grid = linearGrid(words * magX.height, kRespThreads);
        if (impl::cornerSqrtMemoEnabled()) {
            responseKernelSliced<true><<<grid, kRespThreads, 0, stream>>>(v, dst, words);
        } else {
            responseKernelSliced<false><<<grid, kRespThreads, 0, stream>>>(v, dst, words);
        }
    } else {
        responseKernelWindow<<<linearGrid(dst.width * dst.height, 128), 128, 0, stream>>>(
            v, dst, blockSize);
    }
    return cudaGetLastError();
}

cudaError_t goodFeaturesToTrackAsync(DeviceBinMatConstView magX, DeviceBinMatConstView magY,
                                     DeviceBinMatConstView signX, DeviceBinMatConstView signY,
                                     const GoodFeaturesParams& params,
                                     const DeviceGoodFeaturesWorkspace& work,
                                     DeviceCorner* corners, uint32_t capacity,
                                     DeviceCornerResult* result, cudaStream_t stream) {
    const CovViews v{magX, magY, signX, signY};
    BINCV_ASSERT(dimensionsAgree(v),
                 "cuda goodFeaturesToTrack: the four derivative planes must have the same "
                 "dimensions");
    BINCV_ASSERT(params.blockSize > 0, "cuda goodFeaturesToTrack: blockSize must be positive");
    BINCV_ASSERT(params.qualityLevel > 0.0,
                 "cuda goodFeaturesToTrack: qualityLevel must be positive");
    BINCV_ASSERT(params.minDistance >= 0.0,
                 "cuda goodFeaturesToTrack: minDistance must not be negative");
    BINCV_ASSERT(result != nullptr, "cuda goodFeaturesToTrack: a result pointer is required");
    BINCV_ASSERT(corners != nullptr || capacity == 0,
                 "cuda goodFeaturesToTrack: a non-zero capacity needs a corner array");
    if (!dimensionsAgree(v) || params.blockSize <= 0 || params.qualityLevel <= 0.0 ||
        params.minDistance < 0.0 || result == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (corners == nullptr && capacity != 0u) return cudaErrorInvalidValue;
    // R4: the selection's ordering key packs the tie rule into 16 bits per axis,
    // so a frame wider or taller than 65536 is REFUSED rather than wrapped.
    BINCV_ASSERT(magX.width <= 65536 && magX.height <= 65536,
                 "cuda goodFeaturesToTrack: the frame must be at most 65536 pixels on a side");
    if (magX.width > 65536 || magX.height > 65536) return cudaErrorInvalidValue;
    if (magX.width == 0 || magX.height == 0) {
        return cudaMemsetAsync(result, 0, sizeof(DeviceCornerResult), stream);
    }

    const bool fused = impl::cornerFusedEnabled();
    BINCV_ASSERT(work.candidates.out != nullptr && work.candidates.counter != nullptr,
                 "cuda goodFeaturesToTrack: the candidate buffer needs a pointer and a counter");
    BINCV_ASSERT(work.maxBits != nullptr,
                 "cuda goodFeaturesToTrack: the frame-maximum word is required");
    if (work.candidates.out == nullptr || work.candidates.counter == nullptr ||
        work.maxBits == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (work.scratchBytes < goodFeaturesScratchBytes(work.candidates.capacity)) {
        return cudaErrorInvalidValue;
    }
    if (!fused && (work.frameMap.ptr == nullptr || work.frameMap.width != magX.width ||
                   work.frameMap.height != magX.height)) {
        // The reference arm needs the map the fused arm exists not to have.
        return cudaErrorInvalidValue;
    }

    const size_t words = rowWords(magX.width);
    const bool sliced = impl::cornerSlicedEnabled() && impl::cornerSlicedApplies(params.blockSize);

    if (fused) {
        const dim3 grid(static_cast<unsigned>((words + kFuseWords - 1) / kFuseWords),
                        static_cast<unsigned>((magX.height + kFuseRows - 1) / kFuseRows));
        if (sliced && impl::cornerSqrtMemoEnabled()) {
            fusedCandidateKernel<true, true><<<grid, kFuseThreads, 0, stream>>>(
                v, params.blockSize, work.candidates, work.maxBits, words);
        } else if (sliced) {
            fusedCandidateKernel<true, false><<<grid, kFuseThreads, 0, stream>>>(
                v, params.blockSize, work.candidates, work.maxBits, words);
        } else {
            // The window arm takes its blockSize at runtime, so its discriminant
            // has no bounded integer domain and there is no memo instantiation
            // of it to choose between.
            fusedCandidateKernel<false, false><<<grid, kFuseThreads, 0, stream>>>(
                v, params.blockSize, work.candidates, work.maxBits, words);
        }
    } else {
        const cudaError_t mapErr = cornerMinEigenValAsync(magX, magY, signX, signY,
                                                          params.blockSize, work.frameMap, stream);
        if (mapErr != cudaSuccess) return mapErr;
        mapCandidateKernel<<<linearGrid(magX.width * magX.height, kSelThreads), kSelThreads, 0,
                             stream>>>(work.frameMap, work.candidates, work.maxBits);
    }
    const cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) return launchErr;

    const uint32_t padded = nextPow2(work.candidates.capacity);
    uint64_t* keys = static_cast<uint64_t*>(work.scratch);
    uint8_t* alive = reinterpret_cast<uint8_t*>(work.scratch) +
                     alignUp16(static_cast<size_t>(padded) * sizeof(uint64_t));
    uint32_t* state = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(work.scratch) +
                                                  alignUp16(static_cast<size_t>(padded) *
                                                            sizeof(uint64_t)) +
                                                  alignUp16(padded));
    const uint32_t limit =
        params.maxCorners > 0
            ? (capacity < static_cast<uint32_t>(params.maxCorners)
                   ? capacity
                   : static_cast<uint32_t>(params.maxCorners))
            : capacity;

    // The candidate count and the overflow flag. Zeroed here rather than by the
    // caller: they are this call's own bookkeeping, not part of the workspace a
    // caller reasons about.
    const cudaError_t stateErr = cudaMemsetAsync(state, 0, 2 * sizeof(uint32_t), stream);
    if (stateErr != cudaSuccess) return stateErr;

    keyKernel<<<linearGrid(work.candidates.capacity, kSelThreads), kSelThreads, 0, stream>>>(
        work.candidates.out, work.candidates.counter, work.candidates.capacity, work.maxBits,
        params.qualityLevel, keys, state);

    const bool parallelSort = impl::cornerSortParallelEnabled();
    if (parallelSort) {
        const unsigned chunks = static_cast<unsigned>((padded + kSortChunk - 1u) / kSortChunk);
        bitonicChunkKernel<<<chunks, kSortThreads, 0, stream>>>(keys, state,
                                                                work.candidates.capacity);
        // The ladder is sized from the CAPACITY, which the host knows; every
        // kernel reads the actual count and returns at once when its stage is
        // past it, so a sparse frame pays launches rather than work.
        for (uint32_t k = 2u * kSortChunk; k <= padded && k >= 2u * kSortChunk; k <<= 1) {
            for (uint32_t j = k >> 1; j >= kSortChunk; j >>= 1) {
                bitonicStageKernel<<<linearGrid(padded, kSortThreads), kSortThreads, 0, stream>>>(
                    keys, state, work.candidates.capacity, k, j);
            }
            bitonicMergeKernel<<<chunks, kSortThreads, 0, stream>>>(keys, state,
                                                                    work.candidates.capacity, k);
        }
    } else {
        bitonicOneBlockKernel<<<1, kSelThreads, 0, stream>>>(keys, state,
                                                             work.candidates.capacity);
    }
    const cudaError_t sortErr = cudaGetLastError();
    if (sortErr != cudaSuccess) return sortErr;

    const int spacing = params.minDistance >= 1.0 ? 1 : 0;
    const double minDistanceSq = params.minDistance * params.minDistance;
    // ceil, once, on the host: see tooCloseInt for why this is the same
    // predicate as the double comparison and not an approximation of it.
    const double ceilSq = std::ceil(minDistanceSq);
    const unsigned long long minDistanceSqCeil =
        ceilSq >= 18446744073709549568.0 ? 0xFFFFFFFFFFFFFFFFull
                                          : static_cast<unsigned long long>(ceilSq);
    if (impl::cornerSpacingChunkedEnabled()) {
        spacingKernel<true><<<1, kSelThreads, 0, stream>>>(
            keys, state, work.candidates.capacity, minDistanceSq, minDistanceSqCeil, spacing,
            limit, corners, capacity, alive, result);
    } else {
        spacingKernel<false><<<1, kSelThreads, 0, stream>>>(
            keys, state, work.candidates.capacity, minDistanceSq, minDistanceSqCeil, spacing,
            limit, corners, capacity, alive, result);
    }
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
