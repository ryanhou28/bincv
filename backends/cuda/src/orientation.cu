// Keypoint orientation on the device, in both of ops/orientation.hpp's
// spellings. Five kernels: three arms over a wide image and two over a
// bit-plane block, each reachable in one binary through the switches in
// cuda/orientation.hpp.
//
// The wide side has three arms rather than two because the denominator of this
// family's only format claim is the wide arm, and a scalar denominator would
// have decided that claim by omission rather than by measurement. W1 isolates
// warp parallelism over W0; W2 isolates vectorization over W1.

#include <type_traits>

#include "bincv/cuda/orientation.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {
namespace {

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/// @brief The host's truncating float-to-integer keypoint cast, reproduced.
__device__ inline void keypointCenter(DeviceKeypointSetConstView kp, uint32_t k,
                                      long long& cx, long long& cy) {
    cx = static_cast<long long>(kp.x(k));
    cy = static_cast<long long>(kp.y(k));
}

/// @brief The host's bounding-SQUARE test: the descriptor that consumes this
/// angle samples the square, so a keypoint this rejects is one the descriptor
/// was going to reject anyway.
__device__ inline bool insideSquare(long long cx, long long cy, int r, size_t width,
                                    size_t height) {
    return bincv::impl::squareInsideImage(cx, cy, r, width, height);
}

/// @brief The host's `angle[k] = atan2(m01, m10)`, with its flat-patch rule.
__device__ inline float momentAngle(long long m10, long long m01) {
    return (m10 == 0 && m01 == 0)
               ? 0.0f
               : atan2f(static_cast<float>(m01), static_cast<float>(m10));
}

__device__ inline void writeResult(uint32_t k, long long m10, long long m01, bool inside,
                                   float* angles, uint8_t* keep, long long* moments) {
    angles[k] = momentAngle(m10, m01);
    if (keep != nullptr) keep[k] = inside ? uint8_t{1} : uint8_t{0};
    if (moments != nullptr) {
        moments[2 * static_cast<size_t>(k)] = m10;
        moments[2 * static_cast<size_t>(k) + 1] = m01;
    }
}

/// @brief Sums one value across the whole warp, result valid in lane 0.
/// @note 32-bit ONLY, and every warp arm here is written to stay inside it.
/// A 64-bit reduction is ten `__shfl_down_sync` and ten 64-bit adds rather
/// than five and five, which on kernels whose whole disc is a few hundred
/// instructions is not a detail: it is measured below at 1.76x.
__device__ inline int warpSumInt(int v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}


/// @brief The disc's half-width at |dy|, from the two packed registers.
/// @note Warp arms only, and they gate on radius <= 15 -- which is exactly the
/// range in which every half-width fits four bits. No memory: nvcc answers a
/// runtime index into a by-value array parameter with a LOCAL-memory copy, and
/// a stack round trip inside the disc loop is the cost the quad arm exists to
/// avoid.
__device__ inline int discHalf(const DiscPod& disc, int dy) {
    const unsigned ady = static_cast<unsigned>(dy < 0 ? -dy : dy);
    const uint32_t pack = ady < 8u ? disc.packLo : disc.packHi;
    return static_cast<int>((pack >> ((ady & 7u) * 4u)) & 15u);
}

// ---------------------------------------------------------------------------
// W0 -- the reference arm: one thread per keypoint, the host row loop
// ---------------------------------------------------------------------------

template <typename SrcT>
__global__ void orientWideRefKernel(DeviceImageConstView<SrcT> img,
                                    DeviceKeypointSetConstView kp, float* angles,
                                    uint8_t* keep, long long* moments, DiscPod disc) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kp.count) return;
    const int r = disc.radius;
    long long cx = 0, cy = 0;
    keypointCenter(kp, k, cx, cy);
    const bool inside = insideSquare(cx, cy, r, img.width, img.height);
    long long m10 = 0, m01 = 0;
    if (inside) {
        const SrcT* center =
            img.ptr + static_cast<size_t>(cy) * img.stride + static_cast<size_t>(cx);
        for (int dy = -r; dy <= r; ++dy) {
            const int h2 = disc.halfWidth[dy < 0 ? -dy : dy];
            const SrcT* row = center + static_cast<long long>(dy) *
                                           static_cast<long long>(img.stride);
            long long rowSum = 0, rowXSum = 0;
            for (int dx = -h2; dx <= h2; ++dx) {
                const long long v = static_cast<long long>(row[dx]);
                rowSum += v;
                rowXSum += dx * v;
            }
            m10 += rowXSum;
            m01 += dy * rowSum;
        }
    }
    writeResult(k, m10, m01, inside, angles, keep, moments);
}

// ---------------------------------------------------------------------------
// W1 -- a warp per keypoint, a lane per disc COLUMN
//
// 2*radius+1 <= 31 columns at radius 15, so one lane per column with lane 31
// idle. Each row is ONE coalesced read of up to 31 consecutive bytes. No
// per-row reduction: both weights are known per lane per row, so the lane
// accumulates directly and the warp reduces ONCE at the end.
// ---------------------------------------------------------------------------

template <typename SrcT>
__global__ void orientWideWarpKernel(DeviceImageConstView<SrcT> img,
                                     DeviceKeypointSetConstView kp, float* angles,
                                     uint8_t* keep, long long* moments, DiscPod disc) {
    const unsigned lane = threadIdx.x;
    const uint32_t k = blockIdx.x * blockDim.y + threadIdx.y;
    if (k >= kp.count) return;  // warp-uniform: threadIdx.y is constant across a warp
    const int r = disc.radius;
    long long cx = 0, cy = 0;
    keypointCenter(kp, k, cx, cy);
    const bool inside = insideSquare(cx, cy, r, img.width, img.height);

    // int accumulators: at radius 15 a lane's |sum dx*v| is at most
    // 15 * 65535 * 31 = 3.0e7, and the warp total at most 2.6e8. Both inside
    // int32 at either pixel width, which is why this arm does not pay for 64-bit
    // adds it cannot need.
    int m10 = 0, m01 = 0;
    if (inside) {
        const SrcT* center =
            img.ptr + static_cast<size_t>(cy) * img.stride + static_cast<size_t>(cx);
        for (int dy = -r; dy <= r; ++dy) {
            const int h2 = discHalf(disc, dy);
            const int dx = -h2 + static_cast<int>(lane);
            if (dx <= h2) {
                const SrcT* row = center + static_cast<long long>(dy) *
                                               static_cast<long long>(img.stride);
                const int v = static_cast<int>(row[dx]);
                m10 += dx * v;
                m01 += dy * v;
            }
        }
    }
    m10 = warpSumInt(m10);
    m01 = warpSumInt(m01);
    if (lane == 0)
        writeResult(k, static_cast<long long>(m10), static_cast<long long>(m01), inside,
                    angles, keep, moments);
}

// ---------------------------------------------------------------------------
// W2 -- a warp per keypoint, FOUR PIXELS PER LANE: one aligned 4-byte load and
// two `__dp4a` in place of four byte loads and four multiply-adds.
//
// THE WEIGHTS ARE BIASED SO BOTH DOT PRODUCTS ARE UNSIGNED. `__dp4a`'s
// same-signedness overloads are the ones with an intrinsic, and a pixel read as
// a signed byte is wrong above 127. So the x-weight carried in the vector is
// the POSITION p = dx + radius, which is non-negative and at most 30, and the
// re-centering `m10 = sum(p*v) - radius * sum(v)` happens once at the end --
// the same move the bit-plane arm's `- h2 * cnt` makes, for the same reason.
//
// OUT-OF-DISC BYTES ARE MASKED IN THE PIXEL WORD, not in the weights: a pixel
// of zero contributes nothing to either dot product, so one AND covers both.
//
// THE ADDRESSING DOMAIN is checked by the launcher (quadArmApplies): a 4-byte
// aligned base and a stride that is a multiple of four. Together they make
// every row start 4-aligned, so the aligned word holding a row's LAST disc
// pixel begins at or before `stride - 4` and the read cannot leave the row.
// ---------------------------------------------------------------------------

/// @brief The low `n` bytes of a 32-bit word, as a mask, for `n` in [0, 4]. The
/// byte range `[a, b)` is then `byteRun(b) & ~byteRun(a)`.
/// @note Two selects and a shift rather than a five-entry lookup: a local array
/// indexed by a runtime value lands in LOCAL MEMORY, which would put a stack
/// round trip inside the innermost loop of the arm that exists to remove
/// instructions. Neither end can shift by 32.
__device__ inline uint32_t byteRun(int n) {
    if (n <= 0) return 0u;
    if (n >= 4) return 0xFFFFFFFFu;
    return 0xFFFFFFFFu >> ((4 - n) * 8);
}

/// @brief One quad's contribution: 4 pixels, 2 dot products, 1 row weight.
__device__ inline void quadAccumulate(const uint8_t* alignedWindow, int q, int s, int r,
                                      int h2, int dy, unsigned& posAcc, int& cntAcc,
                                      int& m01Acc) {
    const uint32_t px =
        __ldg(reinterpret_cast<const uint32_t*>(alignedWindow + 4 * q));

    // Byte j of this quad holds position p = s + j. The disc row covers
    // positions [r - h2, r + h2].
    const int lo = (r - h2) - s;
    const int hi = (r + h2) - s;
    int a = lo < 0 ? 0 : (lo > 4 ? 4 : lo);
    int b = hi + 1;
    b = b < 0 ? 0 : (b > 4 ? 4 : b);
    const uint32_t mask = byteRun(b) & ~byteRun(a);
    const uint32_t pxm = px & mask;

    // Weight bytes are p = s + j. Only q == 0 can have s < 0, and there the
    // bytes below -s are masked away, so shifting the 0,1,2,3 ramp left by -s
    // bytes puts the right value in every byte that survives.
    const uint32_t weights =
        s >= 0 ? (static_cast<uint32_t>(s) * 0x01010101u + 0x03020100u)
               : (0x03020100u << (static_cast<unsigned>(-s) * 8u));

    posAcc = __dp4a(pxm, weights, posAcc);
    const int c = static_cast<int>(__dp4a(pxm, 0x01010101u, 0u));
    cntAcc += c;
    m01Acc += dy * c;
}

__global__ void orientWideQuadKernel(DeviceImageConstView<uint8_t> img,
                                     DeviceKeypointSetConstView kp, float* angles,
                                     uint8_t* keep, long long* moments, DiscPod disc) {
    const int lane = static_cast<int>(threadIdx.x);
    const uint32_t k = blockIdx.x * blockDim.y + threadIdx.y;
    if (k >= kp.count) return;  // warp-uniform
    const int r = disc.radius;
    long long cx = 0, cy = 0;
    keypointCenter(kp, k, cx, cy);
    const bool inside = insideSquare(cx, cy, r, img.width, img.height);

    unsigned posAcc = 0;
    int cntAcc = 0, m01Acc = 0;
    if (inside) {
        const int rows = 2 * r + 1;
        const long long x0 = cx - r;            // window's first column
        const int m = static_cast<int>(x0 & 3);  // warp-uniform: every row is 4-aligned
        const int qLast = (m + 2 * r) >> 2;      // last quad any row needs, at most 8
        const uint8_t* windowTop = img.ptr +
                                   static_cast<size_t>(cy - r) * img.stride +
                                   static_cast<size_t>(x0 - m);

        // Quads 0..7 of four rows at a time: `lane >> 3` and `lane & 7` rather
        // than a division by a runtime quad count, which would cost more than
        // the quads save.
        const int sub = lane >> 3;
        const int q = lane & 7;
        for (int base = 0; base < rows; base += 4) {
            const int rowIdx = base + sub;
            if (rowIdx < rows && q <= qLast) {
                const int dy = rowIdx - r;
                const int h2 = discHalf(disc, dy);
                quadAccumulate(windowTop + static_cast<size_t>(rowIdx) * img.stride, q,
                               4 * q - m, r, h2, dy, posAcc, cntAcc, m01Acc);
            }
        }
        // The ninth quad, which only a window straddling nine words needs.
        if (qLast >= 8) {
            for (int rowIdx = lane; rowIdx < rows; rowIdx += 32) {
                const int dy = rowIdx - r;
                const int h2 = discHalf(disc, dy);
                quadAccumulate(windowTop + static_cast<size_t>(rowIdx) * img.stride, 8,
                               32 - m, r, h2, dy, posAcc, cntAcc, m01Acc);
            }
        }
    }

    // The re-centering happens PER LANE, before the reduction, because
    // `sum(p*v) - r*sum(v)` is `sum(p*v - r*v)`: two warp reductions instead of
    // three, and a warp reduction is five `__shfl_down_sync` plus five adds --
    // real money on a kernel whose entire disc is ~500 instructions.
    const int m10 = warpSumInt(static_cast<int>(posAcc) - r * cntAcc);
    const int m01 = warpSumInt(m01Acc);
    if (lane == 0)
        writeResult(k, static_cast<long long>(m10), static_cast<long long>(m01), inside,
                    angles, keep, moments);
}

// ---------------------------------------------------------------------------
// The BIT-PLANE spelling
// ---------------------------------------------------------------------------

/// @brief Up to 32 consecutive bits of a bit-plane row.
/// @note The same shape as the dense matcher's own extractor, down to reading
/// word i+1 ONLY when the run crosses -- which is what keeps a disc ending
/// on the row's last word from reading past it. It is spelled again here
/// rather than shared because a `__device__` helper cannot cross a .cu
/// boundary without relocatable device code, which this backend does not
/// build with; the shape is short enough that a test comparing this arm
/// against the host's own segment extraction pins it.
__device__ inline uint32_t extractBits32(const uint32_t* row, size_t bitPos,
                                         unsigned nbits) {
    const size_t i = bitPos >> 5;
    const unsigned off = static_cast<unsigned>(bitPos & 31u);
    uint32_t lo = __ldg(row + i) >> off;
    if (off != 0 && off + nbits > 32u) lo |= __ldg(row + i + 1) << (32u - off);
    return nbits == 32u ? lo : lo & ((1u << nbits) - 1u);
}

/// @brief The same, up to 63 bits -- the reference arm's domain, radius 16..31.
__device__ inline uint64_t extractBits64(const uint32_t* row, size_t bitPos,
                                         unsigned nbits) {
    size_t word = bitPos >> 5;
    const unsigned off = static_cast<unsigned>(bitPos & 31u);
    uint64_t seg = static_cast<uint64_t>(row[word]) >> off;
    unsigned got = 32u - off;
    while (got < nbits) {
        ++word;
        seg |= static_cast<uint64_t>(row[word]) << got;
        got += 32u;
    }
    return seg & ((uint64_t{1} << nbits) - 1ull);
}

/// @brief `sum over set bits of their position`, positions 0..63 -- the host's
/// six-mask decomposition of the position index.
__device__ inline long long bitPositionSum64(uint64_t seg) {
    const uint64_t kMask[6] = {0xAAAAAAAAAAAAAAAAull, 0xCCCCCCCCCCCCCCCCull,
                               0xF0F0F0F0F0F0F0F0ull, 0xFF00FF00FF00FF00ull,
                               0xFFFF0000FFFF0000ull, 0xFFFFFFFF00000000ull};
    long long sum = 0;
#pragma unroll
    for (int b = 0; b < 6; ++b)
        sum += static_cast<long long>(__popcll(seg & kMask[b])) << b;
    return sum;
}

/// @brief The five-mask form: positions 0..31 only.
/// @note The host's sixth mask is exactly the "+32 per high-word bit" term,
/// which exists only once a segment exceeds 32 bits -- that is, only at
/// radius > 15, the reference arm's own domain. Dropping it here is EXACT,
/// not approximate.
__device__ inline int bitPositionSum32(uint32_t seg) {
    return __popc(seg & 0xAAAAAAAAu) + (__popc(seg & 0xCCCCCCCCu) << 1) +
           (__popc(seg & 0xF0F0F0F0u) << 2) + (__popc(seg & 0xFF00FF00u) << 3) +
           (__popc(seg & 0xFFFF0000u) << 4);
}

__global__ void orientPlaneRefKernel(DevicePlaneBlockConstView planes,
                                     DeviceKeypointSetConstView kp, float* angles,
                                     uint8_t* keep, long long* moments, DiscPod disc) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kp.count) return;
    const int r = disc.radius;
    long long cx = 0, cy = 0;
    keypointCenter(kp, k, cx, cy);
    const bool inside = insideSquare(cx, cy, r, planes.width, planes.height);
    long long m10 = 0, m01 = 0;
    if (inside) {
        for (size_t p = 0; p < planes.planes; ++p) {
            const long long wp = 1LL << p;
            long long pm10 = 0, pm01 = 0;
            for (int dy = -r; dy <= r; ++dy) {
                const int h2 = disc.halfWidth[dy < 0 ? -dy : dy];
                const unsigned len = static_cast<unsigned>(2 * h2 + 1);
                const uint32_t* row = planes.row(p, static_cast<size_t>(cy + dy));
                const uint64_t seg =
                    extractBits64(row, static_cast<size_t>(cx - h2), len);
                const long long cnt = static_cast<long long>(__popcll(seg));
                // Positions run 0..2*h2 with position 0 at dx == -h2, so the
                // weighted x-sum re-centers by subtracting h2 per set bit.
                pm10 += bitPositionSum64(seg) - h2 * cnt;
                pm01 += dy * cnt;
            }
            m10 += wp * pm10;
            m01 += wp * pm01;
        }
    }
    writeResult(k, m10, m01, inside, angles, keep, moments);
}

__global__ void orientPlaneWarpKernel(DevicePlaneBlockConstView planes,
                                      DeviceKeypointSetConstView kp, float* angles,
                                      uint8_t* keep, long long* moments, DiscPod disc) {
    const int lane = static_cast<int>(threadIdx.x);
    const uint32_t k = blockIdx.x * blockDim.y + threadIdx.y;
    if (k >= kp.count) return;  // warp-uniform
    const int r = disc.radius;
    long long cx = 0, cy = 0;
    keypointCenter(kp, k, cx, cy);
    const bool inside = insideSquare(cx, cy, r, planes.width, planes.height);

    // 32-BIT ACCUMULATORS, and the bound is why the launcher gates this arm at
    // 16 planes. A lane holds one disc ROW: |pm10| and |pm01| are at most
    // h2 * len <= 15 * 31 = 465, the plane weights sum to 2^P - 1, and the warp
    // reduces 31 lanes -- so at P = 16 the total is 65535 * 465 * 31 = 9.4e8,
    // inside int32. Measured cost of getting this wrong: the 64-bit reduction
    // this replaced was TEN `__shfl_down_sync` plus ten 64-bit adds on a kernel
    // whose whole disc is ~41 instructions, and it made this arm 1.76x SLOWER
    // than its own reference at 100,000 keypoints.
    int m10 = 0, m01 = 0;
    if (inside) {
        const int rows = 2 * r + 1;  // at most 31: one disc ROW per lane
        const bool active = lane < rows;
        const int dy = lane - r;
        const int h2 = active ? discHalf(disc, dy) : 0;
        const unsigned len = static_cast<unsigned>(2 * h2 + 1);
        for (size_t p = 0; p < planes.planes; ++p) {
            int pm10 = 0, pm01 = 0;
            if (active) {
                const uint32_t* row = planes.row(p, static_cast<size_t>(cy + dy));
                const uint32_t seg =
                    extractBits32(row, static_cast<size_t>(cx - h2), len);
                const int cnt = __popc(seg);
                pm10 = bitPositionSum32(seg) - h2 * cnt;
                pm01 = dy * cnt;
            }
            const int wp = 1 << p;
            m10 += wp * pm10;
            m01 += wp * pm01;
        }
    }
    m10 = warpSumInt(m10);
    m01 = warpSumInt(m01);
    if (lane == 0)
        writeResult(k, static_cast<long long>(m10), static_cast<long long>(m01), inside,
                    angles, keep, moments);
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

constexpr unsigned kWarpsPerBlock = 8;
constexpr unsigned kRefBlock = 128;

template <typename SrcT>
cudaError_t launchWide(DeviceImageConstView<SrcT> img, DeviceKeypointSetConstView kp,
                       float* angles, uint8_t* keep, long long* moments,
                       const DiscPod& disc, cudaStream_t stream) {
    if (kp.count == 0) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr && kp.xy != nullptr && angles != nullptr,
                 "cuda keypointOrientation: a non-empty call needs non-null pointers");
    if (orientationWideWarpEnabled() && disc.radius <= 15) {
        const dim3 block(32, kWarpsPerBlock);
        const dim3 grid((kp.count + kWarpsPerBlock - 1) / kWarpsPerBlock);
        if (orientationWideQuadEnabled() && quadArmApplies<SrcT>(img, disc.radius)) {
            if constexpr (std::is_same<SrcT, uint8_t>::value) {
                orientWideQuadKernel<<<grid, block, 0, stream>>>(img, kp, angles, keep,
                                                                 moments, disc);
                return cudaGetLastError();
            }
        }
        orientWideWarpKernel<SrcT>
            <<<grid, block, 0, stream>>>(img, kp, angles, keep, moments, disc);
        return cudaGetLastError();
    }
    const dim3 grid((kp.count + kRefBlock - 1) / kRefBlock);
    orientWideRefKernel<SrcT>
        <<<grid, kRefBlock, 0, stream>>>(img, kp, angles, keep, moments, disc);
    return cudaGetLastError();
}

} // namespace

bool& orientationWideWarpEnabled() {
    static bool on = true;
    return on;
}
bool& orientationWideQuadEnabled() {
    // DEFAULT OFF, and it is a measured decision rather than a doubt about the
    // code. The rule written before the measurement said the quad arm becomes
    // the default only if it beats the lane-per-column arm by more than both
    // printed spreads at 470 and 1000 keypoints -- the rate the named pipeline
    // runs at. Median of nine independent process runs: 1.05x at 470 (slower)
    // and 0.89x at 1000, with the two arms' sample ranges overlapping in every
    // run. At those counts the whole kernel sits on a ~0.009 ms launch floor,
    // so the 3.4x fewer loads and 1.55x fewer instructions the SASS shows have
    // nothing to buy. At 100,000 keypoints, where the kernel is no longer
    // launch, the same arm is 1.46x FASTER with disjoint ranges in six runs of
    // nine. So it stays -- reachable, correct, held to the same output -- and
    // it is not what a caller gets by default at the size a caller uses.
    static bool on = false;
    return on;
}
bool& orientationBitPlaneWarpEnabled() {
    static bool on = true;
    return on;
}

cudaError_t keypointOrientationImpl(DeviceImageConstView<uint8_t> img,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream) {
    return launchWide<uint8_t>(img, keypoints, dAngles, dKeep, dMoments, disc, stream);
}

cudaError_t keypointOrientationImpl(DeviceImageConstView<uint16_t> img,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream) {
    return launchWide<uint16_t>(img, keypoints, dAngles, dKeep, dMoments, disc, stream);
}

cudaError_t keypointOrientationImpl(DevicePlaneBlockConstView planes,
                                    DeviceKeypointSetConstView keypoints, float* dAngles,
                                    uint8_t* dKeep, long long* dMoments, const DiscPod& disc,
                                    cudaStream_t stream) {
    if (keypoints.count == 0) return cudaSuccess;
    BINCV_ASSERT(planes.ptr != nullptr && keypoints.xy != nullptr && dAngles != nullptr,
                 "cuda keypointOrientation: a non-empty call needs non-null pointers");
    // 16 planes is the warp arm's 32-bit accumulator bound, derived at the
    // kernel. QuantMat tops out at 8, so the gate is not binding in practice --
    // it is there because the bound is real and the reference arm, which
    // accumulates in 64 bits like the host, is the answer above it.
    if (orientationBitPlaneWarpEnabled() && disc.radius <= 15 && planes.planes <= 16) {
        const dim3 block(32, kWarpsPerBlock);
        const dim3 grid((keypoints.count + kWarpsPerBlock - 1) / kWarpsPerBlock);
        orientPlaneWarpKernel<<<grid, block, 0, stream>>>(planes, keypoints, dAngles,
                                                          dKeep, dMoments, disc);
        return cudaGetLastError();
    }
    const dim3 grid((keypoints.count + kRefBlock - 1) / kRefBlock);
    orientPlaneRefKernel<<<grid, kRefBlock, 0, stream>>>(planes, keypoints, dAngles, dKeep,
                                                         dMoments, disc);
    return cudaGetLastError();
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
