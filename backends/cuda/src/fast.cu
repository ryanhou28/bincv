// The device arm of ops/fast.hpp's bit-plane detector. Two detection arms, two
// scoring arms, one sort, one switch each.
//
// REFERENCE ARM: one thread per output WORD, reading the seven ring rows
// straight from global memory through the HOST's own `impl::fastShiftedWord` and
// `impl::fastArcAny`. Not a transcription of them -- the functions themselves,
// which is what ruling R3 buys: the `w + 1 < words` guard that decides the last
// word of every row has one definition, and the arc schedule that decides which
// pixels are corners has one definition. This arm is the oracle the tiled arm is
// held to.
//
// TILED ARM: block (32, 4), a 34 x 10 word tile in shared memory, and the
// doubling schedule unrolled at compile time for arcLength 9 and 12. Two things
// it buys over the reference arm, and the second is the larger:
//
//   * Each source word is read from global ONCE per tile instead of up to seven
//     times (once per ring row that reaches it).
//   * `arcLength` becomes a template parameter. With a runtime step every index
//     into the sixteen-word ring array is variable, so the array cannot live in
//     registers and each `and` becomes load-and-store. The host header records
//     the same effect at 1.6x on its own AVX2 arm, and the dense matcher
//     recorded it again here.
//
// SCORING. `fastMaskScoreEnabled()` chooses between reading the score off the
// eight nested arc-length masks -- which the three doublings have already half
// computed -- and peeling each corner's ring and calling the host's
// `impl::fastLongestRun`. The host chooses between the masks and a per-corner
// bit TRANSPOSE, at a measured crossover of three corners per chunk; that
// arithmetic does not port, because the device's alternative is a divergent
// per-lane loop and not a scalar one. Both arms exist so they can be held to one
// output, which is also how the mask identity gets proven.
//
// THE SORT. An atomic append produces no particular order and compaction.hpp
// refuses to promise one. The host emits in raster order, so a complete run is
// sorted on the unique key (y, x) by a single-block bitonic network before it is
// returned -- after which `memcmp` against the host array is the test, not a set
// comparison. A truncated run is still not the host's prefix and the header says
// so.

#include "bincv/cuda/fast.hpp"

#include <cstdint>

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace {

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/// @brief The 16-pixel Bresenham ring, in a form a kernel can read.
/// @note **A RESTATEMENT, AND IT IS HELD TO THE HOST'S RATHER THAN TRUSTED.**
/// nvcc cannot odr-use a host namespace-scope array from device code, so the
/// alternative to restating the ring is not "no copy" -- it is "no kernel".
/// The WINDING is the detector: contiguity is defined along this order, and a
/// different one accepts different corners. `kRingMatchesHost` below is a
/// compile-time comparison against `bincv::impl::kFastRingX/Y`, so a host-side
/// change breaks this file rather than quietly detecting something else.
struct Ring16 {
    int x[16];
    int y[16];
};

BINCV_CUDA_HD constexpr Ring16 fastRing() {
    return Ring16{{0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1},
                  {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3}};
}

constexpr bool kRingMatchesHost() {
    const Ring16 r = fastRing();
    for (int k = 0; k < 16; ++k) {
        if (r.x[k] != bincv::impl::kFastRingX[k]) return false;
        if (r.y[k] != bincv::impl::kFastRingY[k]) return false;
    }
    return true;
}
static_assert(kRingMatchesHost(),
              "the device ring must be the host ring: a different winding is a "
              "different detector");

/// @brief The bits of output word `w` that may be corners: `3 <= x <= width - 4`.
/// @note Two shifts and two masks, not the host's 32-iteration per-bit loop. The
/// host writes the loop because it compiles at 8, 16, 32 and 64-bit words; the
/// device has one word width, and exactly two words per row are affected.
/// @note Padding bits past `width` fall out for free: `width - 4 < width`, so a
/// bit at or past the frame edge is above the high cut.
__device__ inline uint32_t fastBorderMask(size_t w, size_t width) {
    if (width < 7) return 0u;
    const long long base = static_cast<long long>(w) * 32;
    const long long bLo = 3 - base;                                 // need bit >= bLo
    const long long bHi = static_cast<long long>(width) - 4 - base; // need bit <= bHi
    if (bHi < 0 || bLo > 31) return 0u;
    uint32_t m = 0xFFFFFFFFu;
    if (bLo > 0) m &= ~static_cast<uint32_t>((1ull << bLo) - 1ull);
    if (bHi < 31) m &= static_cast<uint32_t>((1ull << (bHi + 1)) - 1ull);
    return m;
}

/// @brief Reserves `n` consecutive slots for this lane with ONE atomic per warp.
/// @return This lane's first slot, which may be at or past capacity -- `store`
/// decides per element, and the counter has already counted the overflow so
/// `found()` stays the true total.
/// @note Every lane of the warp must reach this. The kernels below iterate on a
/// block-uniform bound and pass `n == 0` for a lane with nothing to append,
/// which is what makes the full-mask shuffles legal.
__device__ inline uint32_t warpReserveSlots(const DeviceFastCornerBuffer& buf, uint32_t n,
                                            unsigned lane) {
    uint32_t scan = n;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, scan, off);
        if (lane >= static_cast<unsigned>(off)) scan += v;
    }
    const uint32_t total = __shfl_sync(0xFFFFFFFFu, scan, 31);
    uint32_t base = 0u;
    if (lane == 31u && total != 0u) base = buf.reserve(total);
    base = __shfl_sync(0xFFFFFFFFu, base, 31);
    return base + (scan - n);
}

/// @brief Writes one word's corners, scored by peeling each ring from `diff`.
/// @note The score comes from the HOST's `impl::fastLongestRun`, so it is the
/// host's number by construction rather than by agreement.
__device__ inline void emitPeeled(const DeviceFastCornerBuffer& buf, const uint32_t (&diff)[16],
                                  uint32_t mask, size_t x0, size_t y, int arcLength,
                                  uint32_t slot) {
    while (mask != 0u) {
        const unsigned b = static_cast<unsigned>(__ffs(static_cast<int>(mask)) - 1);
        mask &= mask - 1u;
        unsigned ring = 0u;
#pragma unroll
        for (int k = 0; k < 16; ++k) ring |= ((diff[k] >> b) & 1u) << k;
        DeviceFastCorner c;
        c.x = static_cast<int>(x0 + b);
        c.y = static_cast<int>(y);
        c.score = bincv::impl::fastLongestRun(ring, arcLength);
        buf.store(slot++, c);
    }
}

/// @brief Writes one word's corners, scored off the eight nested arc masks.
/// @note `masks[0]` is the corner mask itself (a run of nine), so a pixel's score
/// is `8 + the number of masks holding its bit` -- a run of L implies a run of
/// every shorter length, so the count IS the maximum. The host's `emitScored`
/// says the same thing on the host side, and the suite holds the two spellings
/// to one output.
__device__ inline void emitScored(const DeviceFastCornerBuffer& buf, const uint32_t (&masks)[8],
                                  uint32_t mask, size_t x0, size_t y, uint32_t slot) {
    while (mask != 0u) {
        const unsigned b = static_cast<unsigned>(__ffs(static_cast<int>(mask)) - 1);
        mask &= mask - 1u;
        int score = 8;
#pragma unroll
        for (int L = 0; L < 8; ++L) score += static_cast<int>((masks[L] >> b) & 1u);
        DeviceFastCorner c;
        c.x = static_cast<int>(x0 + b);
        c.y = static_cast<int>(y);
        c.score = score;
        buf.store(slot++, c);
    }
}

// ---------------------------------------------------------------------------
// The reference arm
// ---------------------------------------------------------------------------

__global__ void fastKernelRef(DeviceBinMatConstView img, DeviceFastCornerBuffer buf,
                              int arcLength, size_t words) {
    const size_t rows = img.height - 6;  // y in [3, height - 3); the caller checked height >= 7
    const size_t total = words * rows;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const unsigned lane = threadIdx.x & 31u;

    // The loop bound is BLOCK-UNIFORM on purpose: every lane must reach the warp
    // scan below, so a lane with no work carries n == 0 rather than exiting.
    for (size_t base = static_cast<size_t>(blockIdx.x) * blockDim.x; base < total;
         base += stride) {
        const size_t idx = base + threadIdx.x;
        uint32_t diff[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) diff[k] = 0u;
        uint32_t mask = 0u;
        size_t y = 0, w = 0;
        if (idx < total) {
            const Ring16 ring = fastRing();
            y = 3 + idx / words;
            w = idx - (y - 3) * words;
            const uint32_t center = img.row(y)[w];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint32_t* ringRow = img.row(
                    static_cast<size_t>(static_cast<long long>(y) + ring.y[k]));
                const uint32_t r =
                    bincv::impl::fastShiftedWord<uint32_t>(ringRow, words, w, ring.x[k]);
                diff[k] = r ^ center;
            }
            mask = bincv::impl::fastArcAny<uint32_t>(diff, arcLength) &
                   fastBorderMask(w, img.width);
        }
        const uint32_t n = static_cast<uint32_t>(__popc(static_cast<int>(mask)));
        const uint32_t slot = warpReserveSlots(buf, n, lane);
        if (mask != 0u) emitPeeled(buf, diff, mask, w * 32, y, arcLength, slot);
    }
}

// ---------------------------------------------------------------------------
// The tiled arm
// ---------------------------------------------------------------------------

constexpr int kTileWords = 32;   // threadIdx.x
constexpr int kTileRows = 4;     // threadIdx.y
constexpr int kTileCols = kTileWords + 2;
constexpr int kTileHeight = kTileRows + 6;

/// @brief `fastShiftedWord` over the staged tile. The neighbours are in the tile
/// by construction (col 1..32 of 0..33), and words outside the row were staged
/// as ZERO -- which is the host's own rule for them, not an approximation of it.
__device__ inline uint32_t shiftedTileWord(const uint32_t* row, int c, int dx) {
    if (dx == 0) return row[c];
    if (dx > 0) {
        const unsigned s = static_cast<unsigned>(dx);
        return (row[c] >> s) | (row[c + 1] << (32u - s));
    }
    const unsigned s = static_cast<unsigned>(-dx);
    return (row[c] << s) | (row[c - 1] >> (32u - s));
}

/// @brief One doubling step of the arc test, the step a COMPILE-TIME constant.
/// @note A template and not a parameter for the reason the host header records
/// for its own vector arm: with a runtime step every index into `v` is
/// variable, so `v` leaves the register file.
template <int Step>
__device__ inline void arcStep(uint32_t (&v)[16]) {
    uint32_t save[Step];
#pragma unroll
    for (int i = 0; i < Step; ++i) save[i] = v[i];
#pragma unroll
    for (int k = 0; k + Step < 16; ++k) v[k] &= v[k + Step];
#pragma unroll
    for (int k = 16 - Step; k < 16; ++k) v[k] &= save[k + Step - 16];
}

/// @brief `OR over all 16 starts of (AND of L consecutive)`, given `v` holding
/// runs of eight. Two overlapping runs of eight cover any L in 9..16.
template <int L>
__device__ inline uint32_t arcMask(const uint32_t (&v)[16]) {
    constexpr int kOff = L - 8;
    uint32_t a = v[0] & v[kOff & 15];
#pragma unroll
    for (int k = 1; k < 16; ++k) a |= v[k] & v[(k + kOff) & 15];
    return a;
}

template <int ArcLength, bool MaskScore>
__global__ void fastKernelTiled(DeviceBinMatConstView img, DeviceFastCornerBuffer buf,
                                size_t words) {
    __shared__ uint32_t tile[kTileHeight][kTileCols];

    const size_t w0 = static_cast<size_t>(blockIdx.x) * kTileWords;
    const size_t y0 = 3 + static_cast<size_t>(blockIdx.y) * kTileRows;
    const unsigned t = threadIdx.y * kTileWords + threadIdx.x;

    for (unsigned i = t; i < kTileHeight * kTileCols; i += kTileRows * kTileWords) {
        const unsigned r = i / kTileCols;
        const unsigned c = i - r * kTileCols;
        const long long gw = static_cast<long long>(w0) + c - 1;
        const long long gy = static_cast<long long>(y0) + r - 3;
        const bool ok = gw >= 0 && gw < static_cast<long long>(words) && gy >= 0 &&
                        gy < static_cast<long long>(img.height);
        tile[r][c] = ok ? img.row(static_cast<size_t>(gy))[static_cast<size_t>(gw)] : 0u;
    }
    __syncthreads();

    const size_t w = w0 + threadIdx.x;
    const size_t y = y0 + threadIdx.y;
    const int c0 = static_cast<int>(threadIdx.x) + 1;
    const int r0 = static_cast<int>(threadIdx.y) + 3;

    // A lane outside the frame carries all-zero ring differences rather than
    // exiting: the warp-aggregated append below is a full-mask shuffle and every
    // lane has to reach it.
    const bool active = w < words && y + 3 < img.height;
    const Ring16 ring = fastRing();

    uint32_t v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = 0u;
    if (active) {
        const uint32_t center = tile[r0][c0];
#pragma unroll
        for (int k = 0; k < 16; ++k)
            v[k] = center ^ shiftedTileWord(tile[r0 + ring.y[k]], c0, ring.x[k]);
    }

    if constexpr (MaskScore) {
        // MaskScore is instantiated at ArcLength 9 only: after the three
        // doublings `v[k]` is a run of EIGHT, and two overlapping runs of eight
        // cover every L in 9..16 -- which is what makes the score a count of
        // masks rather than a second pass over the ring.
        arcStep<1>(v);
        arcStep<2>(v);
        arcStep<4>(v);
        uint32_t masks[8];
        masks[0] = arcMask<9>(v);
        masks[1] = arcMask<10>(v);
        masks[2] = arcMask<11>(v);
        masks[3] = arcMask<12>(v);
        masks[4] = arcMask<13>(v);
        masks[5] = arcMask<14>(v);
        masks[6] = arcMask<15>(v);
        masks[7] = arcMask<16>(v);
        // The border cut lands on the CORNER mask only. `masks[1..7]` are read
        // solely at bits this mask still holds, so a scored pixel's count is
        // never taken from a bit the border removed.
        const uint32_t mask = active ? (masks[0] & fastBorderMask(w, img.width)) : 0u;
        const uint32_t n = static_cast<uint32_t>(__popc(static_cast<int>(mask)));
        const uint32_t slot = warpReserveSlots(buf, n, threadIdx.x & 31u);
        if (mask != 0u) emitScored(buf, masks, mask, w * 32, y, slot);
    } else {
        uint32_t d[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) d[k] = v[k];
        uint32_t any;
        if constexpr (ArcLength == 9) {
            arcStep<1>(v);
            arcStep<2>(v);
            arcStep<4>(v);
            any = arcMask<9>(v);
        } else {  // ArcLength == 12: the host's schedule 1, 2, 4, 4
            arcStep<1>(v);
            arcStep<2>(v);
            arcStep<4>(v);
            arcStep<4>(v);
            any = v[0];
#pragma unroll
            for (int k = 1; k < 16; ++k) any |= v[k];
        }
        const uint32_t mask = active ? (any & fastBorderMask(w, img.width)) : 0u;
        const uint32_t n = static_cast<uint32_t>(__popc(static_cast<int>(mask)));
        const uint32_t slot = warpReserveSlots(buf, n, threadIdx.x & 31u);
        if (mask != 0u) emitPeeled(buf, d, mask, w * 32, y, ArcLength, slot);
    }
}

// ---------------------------------------------------------------------------
// Raster order
// ---------------------------------------------------------------------------

/// @brief Is `a` later in raster order than `b`? The sort key is `(y, x)`, which
/// is UNIQUE per corner -- so any correct sort produces exactly one sequence,
/// and the sort's own order-dependence cannot reach the answer.
__device__ inline bool rasterGreater(const DeviceFastCorner& a, const DeviceFastCorner& b) {
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

/// @brief Sorts the stored corners into the host's raster order. ONE BLOCK, so
/// `__syncthreads()` is a full barrier over the whole network.
/// @note The padded tail is filled with a sentinel that sorts LAST, which is what
/// lets an arbitrary count run through a power-of-two network without the
/// network ever moving a real element into a slot that does not exist.
__global__ void fastSortKernel(DeviceFastCorner* out, const uint32_t* counter,
                               uint32_t capacity, DeviceFastCorner* scratch) {
    const uint32_t found = *counter;
    const uint32_t n = found < capacity ? found : capacity;
    if (n < 2u) return;
    // Padded to the next power of two above what was actually STORED, not above
    // the capacity the caller sized for. A frontend that sizes generously and
    // detects two hundred corners sorts 256 slots, not its capacity's worth --
    // the network is O(n log^2 n) and that difference is most of this kernel.
    uint32_t padded = 1u;
    while (padded < n) padded <<= 1;

    for (uint32_t i = threadIdx.x; i < padded; i += blockDim.x) {
        if (i < n) {
            scratch[i] = out[i];
        } else {
            DeviceFastCorner sentinel;
            sentinel.x = 0x7FFFFFFF;
            sentinel.y = 0x7FFFFFFF;
            sentinel.score = 0;
            scratch[i] = sentinel;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= padded; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
            for (uint32_t i = threadIdx.x; i < padded; i += blockDim.x) {
                const uint32_t l = i ^ j;
                if (l > i) {
                    const bool ascending = (i & k) == 0u;
                    if (ascending == rasterGreater(scratch[i], scratch[l])) {
                        const DeviceFastCorner tmp = scratch[i];
                        scratch[i] = scratch[l];
                        scratch[l] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) out[i] = scratch[i];
}

uint32_t nextPow2(uint32_t v) {
    uint32_t p = 1u;
    while (p < v) p <<= 1;
    return p;
}

} // namespace

namespace impl {

bool& fastTiledEnabled() {
    static bool on = true;
    return on;
}

bool& fastMaskScoreEnabled() {
    static bool on = true;
    return on;
}

bool fastTiledApplies(int arcLength) { return arcLength == 9 || arcLength == 12; }

} // namespace impl

size_t fastScratchBytes(size_t capacity) {
    if (capacity < 2) return 0;
    BINCV_ASSERT(capacity <= 0xFFFFFFFFu,
                 "fastScratchBytes: capacity outside the device's uint32 count domain");
    return static_cast<size_t>(nextPow2(static_cast<uint32_t>(capacity))) *
           sizeof(DeviceFastCorner);
}

cudaError_t detectFastAsync(DeviceBinMatConstView img, DeviceFastCornerBuffer out,
                            void* scratch, size_t scratchBytes, int arcLength,
                            cudaStream_t stream) {
    BINCV_ASSERT(arcLength >= 1 && arcLength <= 16,
                 "cuda detectFast: arcLength must be within the ring");
    BINCV_ASSERT(out.counter != nullptr, "cuda detectFast: the append buffer needs a counter");
    // R4: the domain is narrower than the host's and is refused rather than
    // truncated. arcLength is the only one a release build can still get wrong.
    if (arcLength < 1 || arcLength > 16) return cudaErrorInvalidValue;
    if (out.counter == nullptr) return cudaErrorInvalidValue;
    if (img.width < 7 || img.height < 7 || out.capacity == 0u) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr, "cuda detectFast: a non-empty frame needs a non-null pointer");
    BINCV_ASSERT(img.stride >= rowWords(img.width),
                 "cuda detectFast: the frame's stride must cover a whole row");
    if (scratchBytes < fastScratchBytes(out.capacity)) return cudaErrorInvalidValue;

    const size_t words = rowWords(img.width);
    const bool tiled = impl::fastTiledEnabled() && impl::fastTiledApplies(arcLength);

    if (tiled) {
        const dim3 block(kTileWords, kTileRows);
        const dim3 grid(static_cast<unsigned>((words + kTileWords - 1) / kTileWords),
                        static_cast<unsigned>((img.height - 6 + kTileRows - 1) / kTileRows));
        if (arcLength == 9 && impl::fastMaskScoreEnabled()) {
            fastKernelTiled<9, true><<<grid, block, 0, stream>>>(img, out, words);
        } else if (arcLength == 9) {
            fastKernelTiled<9, false><<<grid, block, 0, stream>>>(img, out, words);
        } else {
            fastKernelTiled<12, false><<<grid, block, 0, stream>>>(img, out, words);
        }
    } else {
        const size_t units = words * (img.height - 6);
        const unsigned threads = 256u;
        const size_t blocks = (units + threads - 1) / threads;
        const dim3 grid(static_cast<unsigned>(blocks < 4096 ? (blocks ? blocks : 1) : 4096));
        fastKernelRef<<<grid, dim3(threads), 0, stream>>>(img, out, arcLength, words);
    }
    const cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) return launchErr;

    if (out.capacity >= 2u && out.out != nullptr) {
        // ONE BLOCK, because a bitonic network needs a barrier between every
        // stage and __syncthreads() is the only one a kernel has. 1024 threads is
        // the largest block this device takes, and the network's inner loop is
        // exactly `padded / threads` iterations wide, so the block size divides
        // the whole kernel.
        fastSortKernel<<<1, 1024, 0, stream>>>(out.out, out.counter, out.capacity,
                                               static_cast<DeviceFastCorner*>(scratch));
    }
    return cudaGetLastError();
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
