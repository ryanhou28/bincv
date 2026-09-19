#pragma once

/// @file descriptor.hpp
/// @brief BRIEF and steered BRIEF on the device: the twin of
/// ops/descriptor.hpp's `computeBrief` / `computeBriefSteered`, for a whole
/// keypoint set in one launch. **API TIER 3.**
///
/// ---------------------------------------------------------------------------
/// WHAT IS AND IS NOT A STRUCTURAL ADVANTAGE HERE, STATED AT THE TOP
///
/// **There is no format advantage in this op, and claiming one would be
/// false.** ops/descriptor.hpp is explicit that the test is `img[a] < img[b]`
/// on the GRAYSCALE image, because a comparison between two one-bit pixels
/// carries almost nothing. So a 256-bit descriptor reads 512 bytes of the same
/// wide frame `cv::cuda::ORB` reads, and writes the same 32 bytes: the
/// memory-traffic ratio against ORB's describe is **1.0x**.
///
/// What this op does bring is three things, none of them the representation:
///
/// 1. A warp-per-keypoint gather collected by `__ballot_sync`: 2.6x over the
///    thread-per-keypoint arm, measured. Not for the reason it looks like --
///    the SASS says the two arms issue about the same number of instructions
///    per descriptor word (~33 against ~36), so the one VOTE in place of ~32
///    dependent shift-ors is not what pays. What pays is that a warp on ONE
///    keypoint gathers inside a single 39x39 patch instead of 32 scattered
///    ones. A device-kernel win, and one anybody could take.
/// 2. No wide pyramid and no blurred copy in the working set.
/// 3. **The descriptor comes out as `uint32_t` WORDS, and that is load-bearing
///    downstream.** `cv::cuda`'s BFMatcher popcounts descriptors as `uchar`
///    elements -- `HammingDist::reduceIter` is `__popc(a ^ b)` instantiated at
///    `uchar` -- so it issues 32 popcounts per 256-bit descriptor where a
///    matcher reading these words issues 8. That 4x is real, it is on the
///    MATCHING side rather than here, and the only way to give it away is to
///    emit bytes. This header does not.
///
/// The case for a device arm rests on residency and on memory: once keypoints
/// are on the device, the alternative is downloading a 361 KB frame to describe
/// a few hundred keypoints on the CPU. That is the honest sentence, and it is
/// the one the report carries.
///
/// ---------------------------------------------------------------------------
/// THE BIT ORDER IS EXACT BY CONSTRUCTION, NOT BY LUCK
///
/// The host sets bit `b` of word `w` from pair index `w * 32 + b`.
/// `__ballot_sync` sets bit `lane`. Lane `l` evaluates pair `w * 32 + l`.
/// Therefore `lane == b`: there is nothing to permute and nothing to check at
/// runtime, and a device descriptor matches against a host or `cv::ORB` map
/// directly.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "bincv/ops/descriptor.hpp"
#include "core.hpp"
#include "features.hpp"

namespace bincv {
inline namespace BINCV_ABI_NAMESPACE {
namespace cuda {

/// @brief A BRIEF pattern as the device reads it: the HOST'S OWN BYTES at a
/// device address, plus the per-bin reach the bounds test needs.
/// @note NOTHING IS RE-TABULATED. `BriefPattern<Bits>` and
/// `SteeredBriefPattern<Bits>` are plain aggregates of `BriefPair`, so the
/// upload is one raw `cudaMemcpy` of the host object and the kernel reads
/// the host's bytes -- the shared-format rule applied to a table. In
/// particular `ops/orbPattern.hpp`'s vendored `bit_pattern_31_` is reached
/// by pointer and copied nowhere new.
/// @note `reach` is carried BY VALUE (30 bytes) so the kernel does not rescan
/// `Bits` pairs per keypoint to find the bounds test's radius. It is the
/// value `bincv::impl::briefPatternReach` returns -- the host's own
/// function, called once per bin, so the two sides cannot disagree about
/// which keypoints are describable.
struct DeviceBriefPattern {
    const BriefPair* pairs = nullptr;  ///< device memory, `bins * bits` pairs
    uint32_t bits = 0;                 ///< descriptor length
    uint32_t bins = 0;                 ///< 1 unsteered, `kBriefAngleBins` steered
    int8_t reach[kBriefAngleBins] = {};

    BINCV_CUDA_HD bool empty() const { return pairs == nullptr || bits == 0 || bins == 0; }
    /// @brief First pair of rotation bin `b`.
    BINCV_CUDA_HD const BriefPair* bin(uint32_t b) const {
        return pairs + static_cast<size_t>(b) * bits;
    }
};

/// @brief Pairs a caller must allocate on the device for an unsteered pattern.
template <size_t Bits>
constexpr size_t briefPatternPairs() {
    return Bits;
}

/// @brief Pairs a caller must allocate on the device for a steered pattern:
/// 30 bins x `Bits`, 30,720 pairs at 256 bits.
template <size_t Bits>
constexpr size_t steeredBriefPatternPairs() {
    return kBriefAngleBins * Bits;
}

/// @brief Copies a host pattern to caller-owned device memory and names it.
/// **API TIER 3** (setup, no kernel).
/// @param dPairs Device memory for `briefPatternPairs<Bits>()` pairs. CALLER
/// OWNED, like every other array this backend's feature families take --
/// nothing here allocates.
/// @note Asynchronous with respect to `stream` only from pinned host memory;
/// a pattern normally lives in a static or on the stack, so the copy is
/// effectively synchronous and happens once per process, not per frame.
template <size_t Bits>
inline cudaError_t uploadBriefPattern(const BriefPattern<Bits>& host, BriefPair* dPairs,
                                      DeviceBriefPattern& out,
                                      cudaStream_t stream = nullptr) {
    BINCV_ASSERT(dPairs != nullptr, "uploadBriefPattern: null device pattern buffer");
    out = DeviceBriefPattern{};
    out.pairs = dPairs;
    out.bits = static_cast<uint32_t>(Bits);
    out.bins = 1;
    out.reach[0] = static_cast<int8_t>(bincv::impl::briefPatternReach<Bits>(host.pair));
    return cudaMemcpyAsync(dPairs, host.pair, sizeof(BriefPair) * Bits,
                           cudaMemcpyHostToDevice, stream);
}

/// @brief The steered spelling: all 30 rotated copies, one copy.
/// **API TIER 3** (setup, no kernel).
/// @note `SteeredBriefPattern<Bits>` is 30 contiguous `BriefPattern<Bits>`, so
/// bin `b` lands at `dPairs + b * Bits` and the kernel's `bin()` is the
/// host's `pattern.bin[b].pair`. That is one `static_assert` away from being
/// checked rather than asserted in prose, and it is checked below.
template <size_t Bits>
inline cudaError_t uploadBriefPattern(const SteeredBriefPattern<Bits>& host,
                                      BriefPair* dPairs, DeviceBriefPattern& out,
                                      cudaStream_t stream = nullptr) {
    static_assert(sizeof(SteeredBriefPattern<Bits>) ==
                      kBriefAngleBins * sizeof(BriefPair) * Bits,
                  "a steered pattern must be 30 contiguous bins with no padding");
    BINCV_ASSERT(dPairs != nullptr, "uploadBriefPattern: null device pattern buffer");
    out = DeviceBriefPattern{};
    out.pairs = dPairs;
    out.bits = static_cast<uint32_t>(Bits);
    out.bins = static_cast<uint32_t>(kBriefAngleBins);
    for (size_t b = 0; b < kBriefAngleBins; ++b)
        out.reach[b] =
            static_cast<int8_t>(bincv::impl::briefPatternReach<Bits>(host.bin[b].pair));
    return cudaMemcpyAsync(dPairs, host.bin[0].pair,
                           sizeof(BriefPair) * kBriefAngleBins * Bits,
                           cudaMemcpyHostToDevice, stream);
}

namespace impl {

/// @brief Force the reference arm (one thread per keypoint, the host's serial
/// shift-or). **INTERNAL.** Same contract as `censusTiledEnabled`.
bool& briefBallotArmEnabled();

/// @brief Descriptor words above which the ballot arm's own gate excludes it.
/// @note The ballot arm keeps word `w` in lane `w` and stores `kWords`
/// contiguous words from lanes `0..kWords-1`, so it needs `kWords <= 32`.
/// A longer descriptor falls to the reference arm, which is what gives the
/// benchmark a case the fast path's gate rejects -- it must read ~1.00x, or
/// the switch is not wired.
inline constexpr uint32_t kBriefBallotMaxWords = 32;

cudaError_t computeBriefImpl(DeviceImageConstView<uint8_t> img,
                             DeviceKeypointSetConstView keypoints, const float* dAngles,
                             const DeviceBriefPattern& pattern,
                             DeviceDescriptorSetView out, cudaStream_t stream);
cudaError_t computeBriefImpl(DeviceImageConstView<uint16_t> img,
                             DeviceKeypointSetConstView keypoints, const float* dAngles,
                             const DeviceBriefPattern& pattern,
                             DeviceDescriptorSetView out, cudaStream_t stream);

/// @brief The shape checks both entry points share.
/// @note The three `(void)` casts are the project's idiom for a parameter that
/// exists only for a `BINCV_ASSERT`: the release build compiles the checks
/// away, and `-Wunused-parameter` is fatal under the gate.
inline void checkBriefArgs(const DeviceBriefPattern& pattern,
                           DeviceKeypointSetConstView keypoints,
                           DeviceDescriptorSetView out) {
    (void)pattern;
    (void)keypoints;
    (void)out;
    BINCV_ASSERT(pattern.bits % 32 == 0,
                 "cuda computeBrief: descriptor length must be a multiple of 32 bits");
    BINCV_ASSERT(out.wordsPerDescriptor == pattern.bits / 32,
                 "cuda computeBrief: the output's word pitch must match the pattern");
    BINCV_ASSERT(out.count == keypoints.count,
                 "cuda computeBrief: one descriptor per keypoint");
}

} // namespace impl

/// @brief Descriptors for a whole keypoint set, unsteered. **API TIER 3.**
/// @param pattern An UNSTEERED pattern (`bins == 1`), uploaded by
/// `uploadBriefPattern`.
/// @param out `keypoints.count` descriptors of `pattern.bits / 32` words, and
/// optionally the per-keypoint `keep` byte. **A keypoint whose patch falls
/// outside the image has no descriptor**: its words are written ZERO and its
/// `keep` byte 0, exactly as the host does it, because inventing one by
/// clamping would produce a confident match against nothing.
/// @note Bit `i` is `img[a_i] < img[b_i]`, the reference test. Never allocates,
/// and takes no scratch.
inline cudaError_t computeBrief(DeviceImageConstView<uint8_t> img,
                                DeviceKeypointSetConstView keypoints,
                                const DeviceBriefPattern& pattern,
                                DeviceDescriptorSetView out,
                                cudaStream_t stream = nullptr) {
    BINCV_ASSERT(pattern.bins == 1, "cuda computeBrief: an unsteered pattern has one bin");
    impl::checkBriefArgs(pattern, keypoints, out);
    return impl::computeBriefImpl(img, keypoints, nullptr, pattern, out, stream);
}
inline cudaError_t computeBrief(DeviceImageConstView<uint16_t> img,
                                DeviceKeypointSetConstView keypoints,
                                const DeviceBriefPattern& pattern,
                                DeviceDescriptorSetView out,
                                cudaStream_t stream = nullptr) {
    BINCV_ASSERT(pattern.bins == 1, "cuda computeBrief: an unsteered pattern has one bin");
    impl::checkBriefArgs(pattern, keypoints, out);
    return impl::computeBriefImpl(img, keypoints, nullptr, pattern, out, stream);
}

/// @brief `computeBrief` steered by per-keypoint angles. **API TIER 3.**
/// @param dAngles One angle per keypoint in device memory, radians --
/// `keypointOrientation`'s output, byte for byte.
/// @param pattern A STEERED pattern (`bins == kBriefAngleBins`).
/// @note **THE ANGLE DOMAIN IS [-2*pi, 2*pi] AND THE KERNEL ASSERTS IT.** Bin
/// selection ends in a float-to-unsigned cast, which is undefined behaviour
/// outside that range and which the two host ISAs already resolve
/// differently; the device is a third resolution. `keypointOrientation`'s
/// (-pi, pi] needs no pre-conditioning. An accumulated or otherwise
/// unwrapped angle is the CALLER's to wrap first, and a Debug build says so
/// by trapping in the kernel rather than returning a plausible descriptor.
/// @note NOT the host's bin-by-bin traversal. The host runs bin by bin because
/// it flattens one ~4 KB offset table per bin on the stack; the device
/// computes each sample's address inline -- one integer multiply-add per
/// sample, free here -- so keypoints stay in input order and nothing is
/// sorted. The answer is identical; the traversal is not, which is the
/// backend's whole shape.
/// @note `keep` is the SELECTED BIN's reach, as the host's is: a border
/// keypoint can have a descriptor at one angle and not another.
inline cudaError_t computeBriefSteered(DeviceImageConstView<uint8_t> img,
                                       DeviceKeypointSetConstView keypoints,
                                       const float* dAngles,
                                       const DeviceBriefPattern& pattern,
                                       DeviceDescriptorSetView out,
                                       cudaStream_t stream = nullptr) {
    BINCV_ASSERT(pattern.bins == kBriefAngleBins,
                 "cuda computeBriefSteered: a steered pattern has kBriefAngleBins bins");
    BINCV_ASSERT(keypoints.count == 0 || dAngles != nullptr,
                 "cuda computeBriefSteered: null angle array");
    impl::checkBriefArgs(pattern, keypoints, out);
    return impl::computeBriefImpl(img, keypoints, dAngles, pattern, out, stream);
}
inline cudaError_t computeBriefSteered(DeviceImageConstView<uint16_t> img,
                                       DeviceKeypointSetConstView keypoints,
                                       const float* dAngles,
                                       const DeviceBriefPattern& pattern,
                                       DeviceDescriptorSetView out,
                                       cudaStream_t stream = nullptr) {
    BINCV_ASSERT(pattern.bins == kBriefAngleBins,
                 "cuda computeBriefSteered: a steered pattern has kBriefAngleBins bins");
    BINCV_ASSERT(keypoints.count == 0 || dAngles != nullptr,
                 "cuda computeBriefSteered: null angle array");
    impl::checkBriefArgs(pattern, keypoints, out);
    return impl::computeBriefImpl(img, keypoints, dAngles, pattern, out, stream);
}

} // namespace cuda
} // inline namespace BINCV_ABI_NAMESPACE
} // namespace bincv
