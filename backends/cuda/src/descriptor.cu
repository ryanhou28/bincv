// BRIEF and steered BRIEF on the device. Two arms, one binary.
//
// THE STRUCTURAL POINT OF THIS FILE: a BRIEF descriptor bit is a comparison,
// 32 of them pack into a word, and `__ballot_sync` builds that word in ONE
// instruction when the pattern is laid out PER LANE. The bit order needs no
// fixing afterwards: the host sets bit b of word w from pair w*32+b,
// `__ballot_sync` sets bit `lane`, and lane l evaluates pair w*32+l, so
// lane == b. Nothing to permute, nothing to check at runtime.
//
// WHERE THE MEASURED WIN ACTUALLY COMES FROM, which is NOT where it was
// expected to. The ballot arm runs 2.6x faster than the reference arm, and the
// obvious explanation -- one VOTE in place of ~32 dependent shift-ors -- is not
// it. Counted out of the SASS, per keypoint and per descriptor word, the
// reference issues ~33 warp instructions and the ballot arm ~36: the ballot arm
// issues slightly MORE work per unit of output, because each of its lanes
// unpacks its own pattern entry where the reference's thread reuses one.
//
// What separates them is the GATHER. A warp of 32 keypoint-threads reads from
// 32 DIFFERENT 39x39 patches, so every pixel load is 32 scattered sectors, and
// with steering the 32 lanes sit in different rotation bins so even the pattern
// read diverges. A warp on ONE keypoint reads inside a single patch -- L1
// resident after the first word -- and its pattern read is 128 contiguous
// bytes. The win is locality, and the ballot is what makes the one-warp-one-
// keypoint shape cost nothing to collect.

#include "bincv/cuda/descriptor.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {
namespace impl {
namespace {

/// @brief One pattern entry as a single 4-byte load.
/// @note `BriefPair` is four `int8_t`, so `alignof` is 1 and a struct read can
/// compile to four byte loads. The array is device memory holding 4-byte
/// elements, so every entry IS 4-aligned; the launcher checks the base
/// pointer and this reads the entry as the word it is. Field order is the
/// struct's, which is the same little-endian assumption `core/view.hpp`
/// makes about planes and which the suite pins by comparing whole
/// descriptors against the host's.
struct PairWord {
    int ax, ay, bx, by;
};

__device__ inline PairWord loadPair(const BriefPair* pairs, size_t i) {
    const uint32_t w = __ldg(reinterpret_cast<const uint32_t*>(pairs) + i);
    PairWord p;
    p.ax = static_cast<int>(static_cast<int8_t>(w & 0xFFu));
    p.ay = static_cast<int>(static_cast<int8_t>((w >> 8) & 0xFFu));
    p.bx = static_cast<int>(static_cast<int8_t>((w >> 16) & 0xFFu));
    p.by = static_cast<int>(static_cast<int8_t>((w >> 24) & 0xFFu));
    return p;
}

/// @brief The rotation bin this keypoint's pattern comes from.
/// @note Calls the HOST's `bincv::briefAngleBin`, which carries
/// BINCV_HOST_DEVICE for exactly this: one ULP of disagreement at a bin
/// boundary does not shift an angle slightly, it replaces a whole 256-bit
/// descriptor, and a transcription of that expression is the last thing
/// that should be allowed to drift. Its domain assertion comes along, which
/// is what the gate's Debug configuration exists to compile.
__device__ inline unsigned selectBin(const float* angles, uint32_t k) {
    return angles == nullptr ? 0u : bincv::briefAngleBin(angles[k]);
}

/// @brief The host's bounding-square test against the SELECTED bin's reach.
__device__ inline bool insidePatch(long long cx, long long cy, int reach, size_t width,
                                   size_t height) {
    return bincv::impl::squareInsideImage(cx, cy, reach, width, height);
}

// ---------------------------------------------------------------------------
// The reference arm: one thread per keypoint, the host's serial shift-or
// ---------------------------------------------------------------------------

template <typename SrcT>
__global__ void briefRefKernel(DeviceImageConstView<SrcT> img,
                               DeviceKeypointSetConstView kp, const float* angles,
                               DeviceBriefPattern pat, DeviceDescriptorSetView out) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= kp.count) return;
    const uint32_t kWords = pat.bits >> 5;
    uint32_t* d = out.descriptor(k);
    // The host zeroes before the inside test, so a rejected keypoint's
    // descriptor is all zero rather than stale.
    for (uint32_t w = 0; w < kWords; ++w) d[w] = 0;

    const unsigned bin = selectBin(angles, k);
    const BriefPair* pairs = pat.bin(bin);
    const int reach = pat.reach[bin];
    const long long cx = static_cast<long long>(kp.x(k));
    const long long cy = static_cast<long long>(kp.y(k));
    const bool inside = insidePatch(cx, cy, reach, img.width, img.height);
    if (inside) {
        const SrcT* center =
            img.ptr + static_cast<size_t>(cy) * img.stride + static_cast<size_t>(cx);
        const long long stride = static_cast<long long>(img.stride);
        for (uint32_t w = 0; w < kWords; ++w) {
            uint32_t acc = 0;
            for (uint32_t b = 0; b < 32u; ++b) {
                const PairWord q = loadPair(pairs, static_cast<size_t>(w) * 32u + b);
                const bool bit = center[q.ay * stride + q.ax] < center[q.by * stride + q.bx];
                acc |= static_cast<uint32_t>(bit) << b;
            }
            d[w] = acc;
        }
    }
    if (out.keep != nullptr) out.keep[k] = inside ? uint8_t{1} : uint8_t{0};
}

// ---------------------------------------------------------------------------
// The ballot arm: a warp per keypoint, a LANE PER DESCRIPTOR BIT
//
// A warp of 32 keypoint-threads would gather from 32 DIFFERENT patches -- 32
// scattered sectors per load, and with steering the 32 lanes sit in different
// bins so even the pattern read diverges. A warp on ONE keypoint gathers inside
// a single patch and reads 128 CONTIGUOUS bytes of pattern per word.
// ---------------------------------------------------------------------------

template <typename SrcT>
__global__ void briefBallotKernel(DeviceImageConstView<SrcT> img,
                                  DeviceKeypointSetConstView kp, const float* angles,
                                  DeviceBriefPattern pat, DeviceDescriptorSetView out) {
    const unsigned lane = threadIdx.x;
    const uint32_t k = blockIdx.x * blockDim.y + threadIdx.y;
    // WARP-UNIFORM, and that is load-bearing: a lane that exits while its
    // neighbours reach `__ballot_sync(0xFFFFFFFF, ...)` makes the ballot
    // undefined, which produces a silently wrong descriptor with no crash and
    // no assert. `k` depends on threadIdx.y alone, so a whole warp leaves or a
    // whole warp stays.
    if (k >= kp.count) return;

    const uint32_t kWords = pat.bits >> 5;
    const unsigned bin = selectBin(angles, k);
    const BriefPair* pairs = pat.bin(bin);
    const int reach = pat.reach[bin];
    const long long cx = static_cast<long long>(kp.x(k));
    const long long cy = static_cast<long long>(kp.y(k));
    const bool inside = insidePatch(cx, cy, reach, img.width, img.height);

    // Formed only when the patch is in the image: a rejected keypoint can carry
    // a negative or enormous coordinate, and forming that address -- even
    // without reading it -- is not something to leave to luck.
    const SrcT* center =
        inside ? img.ptr + static_cast<size_t>(cy) * img.stride + static_cast<size_t>(cx)
               : img.ptr;
    const long long stride = static_cast<long long>(img.stride);

    uint32_t mine = 0;
    for (uint32_t w = 0; w < kWords; ++w) {
        const PairWord q = loadPair(pairs, static_cast<size_t>(w) * 32u + lane);
        bool pred = false;
        if (inside)
            pred = center[q.ay * stride + q.ax] < center[q.by * stride + q.bx];
        // bit `lane` of `word` IS host bit `lane` of word `w`. Nothing to
        // permute, nothing to check at runtime.
        const uint32_t word = __ballot_sync(0xFFFFFFFFu, pred);
        if (lane == w) mine = word;
    }
    // Lanes 0..kWords-1 hold the whole descriptor: one contiguous store, and
    // an out-of-patch keypoint stores the zeros its all-false ballots produced.
    if (lane < kWords) out.descriptor(k)[lane] = mine;
    if (out.keep != nullptr && lane == 0) out.keep[k] = inside ? uint8_t{1} : uint8_t{0};
}

constexpr unsigned kWarpsPerBlock = 8;
// 64: this arm is one thread per keypoint, so the block size is how finely the
// work spreads over SMs, and 2,000 keypoints is 16 blocks at 128 threads on a
// 48-SM part. At 64 it is 32 blocks and 1.41x faster (101 of 105 rounds); at
// 20,000 keypoints, where the grid fills, it is still 0.943x, 105 rounds to 0.
// Widening loses: 256 costs 1.09-1.48x and 512 up to 3.00x.
constexpr unsigned kRefBlock = 64;

template <typename SrcT>
cudaError_t launchBrief(DeviceImageConstView<SrcT> img, DeviceKeypointSetConstView kp,
                        const float* angles, const DeviceBriefPattern& pat,
                        DeviceDescriptorSetView out, cudaStream_t stream) {
    if (kp.count == 0) return cudaSuccess;
    BINCV_ASSERT(img.ptr != nullptr && kp.xy != nullptr && out.words != nullptr &&
                     pat.pairs != nullptr,
                 "cuda computeBrief: a non-empty call needs non-null pointers");
    BINCV_ASSERT((reinterpret_cast<uintptr_t>(pat.pairs) & 3u) == 0u,
                 "cuda computeBrief: the device pattern must be 4-byte aligned");
    const uint32_t kWords = pat.bits >> 5;
    if (briefBallotArmEnabled() && kWords <= kBriefBallotMaxWords) {
        const dim3 block(32, kWarpsPerBlock);
        const dim3 grid((kp.count + kWarpsPerBlock - 1) / kWarpsPerBlock);
        briefBallotKernel<SrcT><<<grid, block, 0, stream>>>(img, kp, angles, pat, out);
        return cudaGetLastError();
    }
    const dim3 grid((kp.count + kRefBlock - 1) / kRefBlock);
    briefRefKernel<SrcT><<<grid, kRefBlock, 0, stream>>>(img, kp, angles, pat, out);
    return cudaGetLastError();
}

} // namespace

bool& briefBallotArmEnabled() {
    static bool on = true;
    return on;
}

cudaError_t computeBriefImpl(DeviceImageConstView<uint8_t> img,
                             DeviceKeypointSetConstView keypoints, const float* dAngles,
                             const DeviceBriefPattern& pattern, DeviceDescriptorSetView out,
                             cudaStream_t stream) {
    return launchBrief<uint8_t>(img, keypoints, dAngles, pattern, out, stream);
}

cudaError_t computeBriefImpl(DeviceImageConstView<uint16_t> img,
                             DeviceKeypointSetConstView keypoints, const float* dAngles,
                             const DeviceBriefPattern& pattern, DeviceDescriptorSetView out,
                             cudaStream_t stream) {
    return launchBrief<uint16_t>(img, keypoints, dAngles, pattern, out, stream);
}

} // namespace impl
} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
