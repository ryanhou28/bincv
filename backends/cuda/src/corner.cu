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

#include <cstdint>

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
        out[b] = ((xx | yy | pos | neg) == 0) ? 0.0f
                                              : bincv::impl::minEigenValue(xx, yy, pos - neg);
    }
}

/// @brief One pixel's response at any blockSize, from the clipped window.
/// @note The window is `impl::blockWindow`'s -- anchored where `cv::boxFilter`
/// anchors it -- and CLIPS at the frame edge, which is the host's promise 2.
/// The three sums are popcounts of exact integers, so they cannot differ from
/// the host's by a traversal order.
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

__global__ void responseKernelSliced(CovViews v, DeviceImageView<float> dst, size_t words) {
    const size_t total = words * v.magX.height;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t y = idx / words;
        const size_t w = idx - y * words;
        float tmp[32];
        slicedResponseWord(v, words, y, w, tmp);
        float* row = dst.row(y);
        const size_t x0 = w * 32;
        for (int b = 0; b < 32; ++b) {
            if (x0 + static_cast<size_t>(b) < dst.width) row[x0 + static_cast<size_t>(b)] = tmp[b];
        }
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

template <bool Sliced>
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
            slicedResponseWord(v, words, static_cast<size_t>(gy), static_cast<size_t>(gw), out);
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

/// @brief `impl::CornerStronger`, on the device: response DESCENDING, ties broken
/// by DESCENDING raster position. The reference's `greaterThanPtr` rule spelled
/// on coordinates, and a TOTAL order over distinct positions -- which is what
/// makes the sorted sequence unique and the append order unobservable.
__device__ inline bool cornerStronger(const DeviceCorner& a, const DeviceCorner& b) {
    if (a.response != b.response) return a.response > b.response;
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

__global__ void selectKernel(const DeviceCorner* cand, const uint32_t* candCount,
                             uint32_t candCapacity, const uint32_t* maxBits,
                             double qualityLevel, double minDistanceSq, int spacing,
                             uint32_t limit, DeviceCorner* out, uint32_t capacity,
                             DeviceCorner* sortBuf, uint8_t* alive,
                             DeviceCornerResult* result) {
    __shared__ uint32_t sKept;
    __shared__ uint32_t sCursor;
    __shared__ uint32_t sCompact;
    __shared__ uint32_t sN;
    __shared__ uint32_t sOverflow;
    __shared__ float sThr;
    __shared__ uint32_t sMin[kSelThreads];

    if (threadIdx.x == 0) {
        const uint32_t found = *candCount;
        sOverflow = found > candCapacity ? 1u : 0u;
        sN = found < candCapacity ? found : candCapacity;
        sCompact = 0u;
        // The reference's arithmetic exactly: the product is formed in double
        // (cv::threshold takes a double) and narrowed to float, and the
        // comparison below is STRICTLY greater.
        sThr = static_cast<float>(static_cast<double>(__uint_as_float(*maxBits)) * qualityLevel);
    }
    __syncthreads();

    if (sOverflow != 0u) {
        if (threadIdx.x == 0) {
            result->count = 0u;
            result->candidatesRanked = 0u;
            result->candidatesTruncated = 0u;
            result->candidateOverflow = 1u;
        }
        return;
    }

    // The quality cut. What survives is `S`, and CornerStronger orders on
    // response first, so S is UPWARD CLOSED in the raw maxima -- which is what
    // makes "the strongest `capacity` of S" the host's ranked set exactly.
    for (uint32_t i = threadIdx.x; i < sN; i += kSelThreads) {
        const DeviceCorner c = cand[i];
        if (c.response > sThr) sortBuf[atomicAdd(&sCompact, 1u)] = c;
    }
    __syncthreads();
    const uint32_t S = sCompact;

    uint32_t p2 = 1u;
    while (p2 < S) p2 <<= 1;
    for (uint32_t i = S + threadIdx.x; i < p2; i += kSelThreads) {
        DeviceCorner s;
        s.x = -2147483647 - 1;
        s.y = -2147483647 - 1;
        s.response = -1.0f;  // below every response, which is never negative
        sortBuf[i] = s;
    }
    __syncthreads();

    // A bitonic network, ascending under CornerStronger, so index 0 is the
    // strongest. One block, so __syncthreads() is a full barrier over it.
    for (uint32_t k = 2u; k <= p2; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < p2; i += kSelThreads) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    const bool ascending = (i & k) == 0u;
                    const bool swapNeeded = ascending ? cornerStronger(sortBuf[l], sortBuf[i])
                                                      : cornerStronger(sortBuf[i], sortBuf[l]);
                    if (swapNeeded) {
                        const DeviceCorner t = sortBuf[i];
                        sortBuf[i] = sortBuf[l];
                        sortBuf[l] = t;
                    }
                }
            }
            __syncthreads();
        }
    }

    const uint32_t ranked = S < capacity ? S : capacity;
    const uint32_t truncated = S > capacity ? 1u : 0u;

    if (spacing == 0) {
        // gftt.cpp's `else` branch: no spacing at all, just the strongest.
        const uint32_t kept = ranked < limit ? ranked : limit;
        for (uint32_t i = threadIdx.x; i < kept; i += kSelThreads) out[i] = sortBuf[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            result->count = kept;
            result->candidatesRanked = ranked;
            result->candidatesTruncated = truncated;
            result->candidateOverflow = 0u;
        }
        return;
    }

    for (uint32_t i = threadIdx.x; i < ranked; i += kSelThreads) alive[i] = 1u;
    if (threadIdx.x == 0) {
        sKept = 0u;
        sCursor = 0u;
    }
    __syncthreads();

    // THE GREEDY FILTER. Accepted points only ever accumulate, so a candidate
    // killed by one stays killed and the FIRST surviving candidate is always the
    // next one the host's sequential loop accepts. That is the whole answer with
    // one barrier per ACCEPTANCE instead of one per candidate.
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

        const DeviceCorner accepted = sortBuf[pick];
        if (threadIdx.x == 0) {
            out[sKept] = accepted;
            alive[pick] = 0u;
            sKept += 1u;
            sCursor = pick + 1u;
        }
        __syncthreads();
        // dx and dy are integers well under 2^26, so both products and their sum
        // are exact in double and no contraction can change the comparison.
        for (uint32_t j = pick + 1u + threadIdx.x; j < ranked; j += kSelThreads) {
            if (alive[j] == 0u) continue;
            const double dx = static_cast<double>(sortBuf[j].x) - static_cast<double>(accepted.x);
            const double dy = static_cast<double>(sortBuf[j].y) - static_cast<double>(accepted.y);
            if (dx * dx + dy * dy < minDistanceSq) alive[j] = 0u;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
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

bool& cornerFusedEnabled() {
    static bool on = true;
    return on;
}

} // namespace impl

size_t goodFeaturesScratchBytes(size_t candidateCapacity) {
    if (candidateCapacity == 0) return 0;
    BINCV_ASSERT(candidateCapacity <= 0xFFFFFFFFu,
                 "goodFeaturesScratchBytes: capacity outside the device's uint32 domain");
    const size_t padded = nextPow2(static_cast<uint32_t>(candidateCapacity));
    return alignUp16(padded * sizeof(DeviceCorner)) + alignUp16(padded);
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
        responseKernelSliced<<<linearGrid(words * magX.height, 128), 128, 0, stream>>>(v, dst,
                                                                                       words);
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
        if (sliced) {
            fusedCandidateKernel<true><<<grid, kFuseThreads, 0, stream>>>(
                v, params.blockSize, work.candidates, work.maxBits, words);
        } else {
            fusedCandidateKernel<false><<<grid, kFuseThreads, 0, stream>>>(
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
    DeviceCorner* sortBuf = static_cast<DeviceCorner*>(work.scratch);
    uint8_t* alive = reinterpret_cast<uint8_t*>(work.scratch) +
                     alignUp16(static_cast<size_t>(padded) * sizeof(DeviceCorner));
    const uint32_t limit =
        params.maxCorners > 0
            ? (capacity < static_cast<uint32_t>(params.maxCorners)
                   ? capacity
                   : static_cast<uint32_t>(params.maxCorners))
            : capacity;

    selectKernel<<<1, kSelThreads, 0, stream>>>(
        work.candidates.out, work.candidates.counter, work.candidates.capacity, work.maxBits,
        params.qualityLevel, params.minDistance * params.minDistance,
        params.minDistance >= 1.0 ? 1 : 0, limit, corners, capacity, sortBuf, alive, result);
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
