// Device-versus-host bit-exactness for keypoint orientation and BRIEF.
//
// The host library is the truth and the arrays are shared byte for byte, so
// every case here is a raw comparison: run both, download, compare. A device
// kernel that is faster and different is not an optimization.
//
// A CUDA translation unit, because two of the claims below can only be asked
// from a kernel. `bincv::briefAngleBin` now carries BINCV_HOST_DEVICE and the
// describe kernel CALLS IT rather than restating it, so what is worth pinning
// is that nvcc's device pass selects the host's own bin -- including at the
// 12-degree boundaries, where one ULP replaces a whole 256-bit descriptor. And
// the moment claim is an INTEGER claim underneath a float output: the angle
// goes through `atan2f` on one side and `std::atan2` on the other, so the
// suite pins the integers exactly and MEASURES the float's divergence rather
// than asserting a bit-exactness that no transcendental can promise.
//
// WHAT IS EXACT HERE AND WHAT IS NOT, stated once:
//   exact    descriptor words, keep bytes, integer moments, bin selection
//   measured angle ULP against the host's atan2 -- reported, not asserted at 0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/descriptor.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/orientation.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/orbPattern.hpp"
#include "bincv/ops/orientation.hpp"
#include "test_util.hpp"

namespace bc = bincv::cuda;

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A frame with structure as well as noise: a flat or purely random patch makes
/// a centroid uninformative, and an uninformative centroid hides a bug in the
/// weighting rather than showing it.
template <typename T>
std::vector<T> makeFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            const uint64_t n = splitmix(seed);
            const size_t ramp = (x * 3u + y * 5u) % 251u;
            img[y * w + x] = static_cast<T>((ramp * 7u + (n & 0x3Fu)) *
                                            (sizeof(T) == 1 ? 1u : 61u));
        }
    return img;
}

/// Keypoints spread over the frame, including deliberate border and
/// out-of-frame cases and fractional coordinates.
std::vector<float> makeKeypoints(size_t w, size_t h, size_t count, int radius,
                                 uint64_t seed) {
    std::vector<float> xy;
    xy.reserve(count * 2);
    // Exactly on the border: kept and described.
    if (count > 0) {
        xy.push_back(static_cast<float>(radius));
        xy.push_back(static_cast<float>(radius));
    }
    if (count > 1) {
        xy.push_back(static_cast<float>(w - 1 - static_cast<size_t>(radius)));
        xy.push_back(static_cast<float>(h - 1 - static_cast<size_t>(radius)));
    }
    // One pixel outside: keep == 0, descriptor all zero, angle 0.
    if (count > 2) {
        xy.push_back(static_cast<float>(radius - 1));
        xy.push_back(static_cast<float>(radius));
    }
    // Negative and fractional, to pin the truncating float-to-integer cast.
    if (count > 3) {
        xy.push_back(-3.75f);
        xy.push_back(2.5f);
    }
    if (count > 4) {
        xy.push_back(static_cast<float>(w / 2) + 0.75f);
        xy.push_back(static_cast<float>(h / 2) + 0.25f);
    }
    while (xy.size() < count * 2) {
        const uint64_t a = splitmix(seed), b = splitmix(seed);
        xy.push_back(static_cast<float>(a % w) + 0.5f);
        xy.push_back(static_cast<float>(b % h) + 0.5f);
    }
    return xy;
}

/// @brief The moments computed PER PIXEL, owing nothing to either kernel.
/// @note Written from the definition in ops/orientation.hpp's header comment --
/// `m10 = sum x * I`, `m01 = sum y * I` over the disc -- and not from either
/// implementation's loop, so a shared mistake in the row decomposition has
/// somewhere to show up. The disc membership test is the inequality itself,
/// not `discHalfWidth`.
template <typename T>
void referenceMoments(const std::vector<T>& img, size_t w, size_t h,
                      const std::vector<float>& xy, size_t count, int radius,
                      std::vector<long long>& moments, std::vector<uint8_t>& keep) {
    moments.assign(count * 2, 0);
    keep.assign(count, 0);
    for (size_t k = 0; k < count; ++k) {
        const long long cx = static_cast<long long>(xy[2 * k]);
        const long long cy = static_cast<long long>(xy[2 * k + 1]);
        const bool inside = cx - radius >= 0 && cy - radius >= 0 &&
                            cx + radius < static_cast<long long>(w) &&
                            cy + radius < static_cast<long long>(h);
        keep[k] = inside ? uint8_t{1} : uint8_t{0};
        if (!inside) continue;
        long long m10 = 0, m01 = 0;
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy > radius * radius) continue;
                const long long v = static_cast<long long>(
                    img[static_cast<size_t>(cy + dy) * w + static_cast<size_t>(cx + dx)]);
                m10 += dx * v;
                m01 += dy * v;
            }
        moments[2 * k] = m10;
        moments[2 * k + 1] = m01;
    }
}

/// @brief ULP distance between two floats of the same sign convention.
unsigned ulpDistance(float a, float b) {
    if (a == b) return 0;
    int32_t ia = 0, ib = 0;
    std::memcpy(&ia, &a, 4);
    std::memcpy(&ib, &b, 4);
    if (ia < 0) ia = static_cast<int32_t>(0x80000000u) - ia;
    if (ib < 0) ib = static_cast<int32_t>(0x80000000u) - ib;
    const long long d = static_cast<long long>(ia) - static_cast<long long>(ib);
    const long long ad = d < 0 ? -d : d;
    return ad > 0xFFFFFFFFll ? 0xFFFFFFFFu : static_cast<unsigned>(ad);
}

/// One device run of the wide orientation, with the arm switches set.
template <typename T>
struct WideRun {
    std::vector<float> angle;
    std::vector<uint8_t> keep;
    std::vector<long long> moments;
    cudaError_t err = cudaSuccess;
};

template <typename T>
WideRun<T> runWide(const std::vector<T>& img, size_t w, size_t h,
                   const std::vector<float>& xy, size_t count, int radius, bool warpArm,
                   bool quadArm) {
    bc::DeviceImage<T> dimg(static_cast<int>(w), static_cast<int>(h));
    bc::uploadImage<T>(img.data(), w, h, w, dimg.view());
    bc::DeviceArray<float> dxy(count * 2 > 0 ? count * 2 : 1);
    if (count > 0)
        cudaMemcpy(dxy.data(), xy.data(), count * 2 * sizeof(float), cudaMemcpyHostToDevice);
    bc::DeviceArray<float> dang(count > 0 ? count : 1);
    bc::DeviceArray<uint8_t> dkeep(count > 0 ? count : 1);
    bc::DeviceArray<long long> dmom(count > 0 ? count * 2 : 1);
    cudaMemset(dang.data(), 0xCD, dang.size() * sizeof(float));
    cudaMemset(dkeep.data(), 0xCD, dkeep.size());
    cudaMemset(dmom.data(), 0xCD, dmom.size() * sizeof(long long));

    bc::impl::orientationWideWarpEnabled() = warpArm;
    bc::impl::orientationWideQuadEnabled() = quadArm;
    WideRun<T> out;
    out.err = bc::keypointOrientation(dimg.constView(), bc::keypointSet(dxy.data(), count),
                                      dang.data(), dkeep.data(), radius, dmom.data());
    cudaDeviceSynchronize();
    bc::impl::orientationWideWarpEnabled() = true;
    bc::impl::orientationWideQuadEnabled() = true;

    out.angle.assign(count, 0.0f);
    out.keep.assign(count, 0);
    out.moments.assign(count * 2, 0);
    if (count > 0) {
        cudaMemcpy(out.angle.data(), dang.data(), count * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(out.keep.data(), dkeep.data(), count, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.moments.data(), dmom.data(), count * 2 * sizeof(long long),
                   cudaMemcpyDeviceToHost);
    }
    return out;
}

/// Compares one device run against the host library and the independent
/// per-pixel reference. Returns the largest angle ULP difference seen.
template <typename T>
unsigned checkWideRun(const char* what, const std::vector<T>& img, size_t w, size_t h,
                      const std::vector<float>& xy, size_t count, int radius,
                      const WideRun<T>& run) {
    std::vector<float> hostAngle(count, 0.0f);
    std::vector<uint8_t> hostKeep(count, 0);
    if (count > 0)
        bincv::keypointOrientation<T>(img.data(), w, h, w, xy.data(), count,
                                      hostAngle.data(), hostKeep.data(), radius);
    std::vector<long long> refMom;
    std::vector<uint8_t> refKeep;
    referenceMoments<T>(img, w, h, xy, count, radius, refMom, refKeep);

    size_t badKeep = 0, badMoment = 0;
    unsigned maxUlp = 0;
    size_t exactAngles = 0;
    for (size_t k = 0; k < count; ++k) {
        if (run.keep[k] != hostKeep[k] || run.keep[k] != refKeep[k]) ++badKeep;
        if (run.moments[2 * k] != refMom[2 * k] ||
            run.moments[2 * k + 1] != refMom[2 * k + 1])
            ++badMoment;
        const unsigned u = ulpDistance(run.angle[k], hostAngle[k]);
        if (u == 0) ++exactAngles;
        if (u > maxUlp) maxUlp = u;
    }
    BINCV_CHECK_EQ(run.err, cudaSuccess);
    BINCV_CHECK_EQ(badKeep, size_t{0});
    BINCV_CHECK_EQ(badMoment, size_t{0});
    std::printf("   %-42s n=%4zu r=%2d  angles exact %zu/%zu, max %u ULP\n", what, count,
                radius, exactAngles, count, maxUlp);
    return maxUlp;
}

// ---------------------------------------------------------------------------
// Bit-plane helpers
// ---------------------------------------------------------------------------

/// Host planes for an N-bit image, plane p holding bit p of the pixel value.
std::vector<bincv::BinMat<uint32_t>> makePlanes(const std::vector<uint8_t>& img, size_t w,
                                                size_t h, size_t planeCount) {
    std::vector<bincv::BinMat<uint32_t>> planes;
    for (size_t p = 0; p < planeCount; ++p) {
        bincv::BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
        for (size_t y = 0; y < h; ++y) {
            uint32_t* row = m.view().row(y);
            for (size_t x = 0; x < w; ++x)
                if (((static_cast<unsigned>(img[y * w + x]) >> p) & 1u) != 0u)
                    row[x / 32] |= (uint32_t{1} << (x % 32));
        }
        planes.push_back(std::move(m));
    }
    return planes;
}

/// The pixel values those planes encode -- the wide image the bit-plane arm is
/// claimed to agree with.
std::vector<uint8_t> planeValues(const std::vector<uint8_t>& img, size_t planeCount) {
    std::vector<uint8_t> out(img.size());
    const uint8_t mask = static_cast<uint8_t>((1u << planeCount) - 1u);
    for (size_t i = 0; i < img.size(); ++i) out[i] = static_cast<uint8_t>(img[i] & mask);
    return out;
}

struct PlaneRun {
    std::vector<float> angle;
    std::vector<uint8_t> keep;
    std::vector<long long> moments;
    cudaError_t err = cudaSuccess;
};

PlaneRun runPlane(const std::vector<bincv::BinMat<uint32_t>>& planes, size_t w, size_t h,
                  const std::vector<float>& xy, size_t count, int radius, bool warpArm,
                  bool dirtyPadding) {
    const size_t planeCount = planes.size();
    bc::DeviceBinMat block(static_cast<int>(w),
                           static_cast<int>(planeCount * h));
    bc::DevicePlaneBlockView pb = bc::planeBlock(block.view(), planeCount);
    for (size_t p = 0; p < planeCount; ++p)
        bc::upload<uint32_t>(planes[p].constView(), pb.plane(p));
    if (dirtyPadding && (w % 32) != 0) {
        // Set every bit past `width` in every row. The container guarantees
        // these are zero; this case exists to prove the kernel's extraction
        // masks to the run length regardless, so a dirty padding bit cannot
        // reach a popcount.
        const size_t words = bc::rowWords(w);
        const uint32_t tail = ~bc::rowTailMask(w);
        std::vector<uint32_t> row(words, 0u);
        for (size_t y = 0; y < planeCount * h; ++y) {
            cudaMemcpy(row.data(), block.view().row(y), words * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            row[words - 1] |= tail;
            cudaMemcpy(block.view().row(y), row.data(), words * sizeof(uint32_t),
                       cudaMemcpyHostToDevice);
        }
    }

    bc::DeviceArray<float> dxy(count * 2 > 0 ? count * 2 : 1);
    if (count > 0)
        cudaMemcpy(dxy.data(), xy.data(), count * 2 * sizeof(float), cudaMemcpyHostToDevice);
    bc::DeviceArray<float> dang(count > 0 ? count : 1);
    bc::DeviceArray<uint8_t> dkeep(count > 0 ? count : 1);
    bc::DeviceArray<long long> dmom(count > 0 ? count * 2 : 1);

    bc::impl::orientationBitPlaneWarpEnabled() = warpArm;
    PlaneRun out;
    out.err = bc::keypointOrientation(bc::DevicePlaneBlockConstView(pb),
                                      bc::keypointSet(dxy.data(), count), dang.data(),
                                      dkeep.data(), radius, dmom.data());
    cudaDeviceSynchronize();
    bc::impl::orientationBitPlaneWarpEnabled() = true;

    out.angle.assign(count, 0.0f);
    out.keep.assign(count, 0);
    out.moments.assign(count * 2, 0);
    if (count > 0) {
        cudaMemcpy(out.angle.data(), dang.data(), count * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(out.keep.data(), dkeep.data(), count, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.moments.data(), dmom.data(), count * 2 * sizeof(long long),
                   cudaMemcpyDeviceToHost);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

template <size_t Bits, typename T>
struct BriefRun {
    std::vector<uint32_t> words;
    std::vector<uint8_t> keep;
    cudaError_t err = cudaSuccess;
};

/// Runs the device describe with a chosen arm. `angles == nullptr` selects the
/// unsteered entry point and the unsteered pattern.
template <size_t Bits, typename T>
BriefRun<Bits, T> runBrief(const std::vector<T>& img, size_t w, size_t h,
                           const std::vector<float>& xy, size_t count,
                           const bincv::BriefPattern<Bits>& base,
                           const bincv::SteeredBriefPattern<Bits>* steered,
                           const std::vector<float>* angles, bool ballotArm) {
    constexpr size_t kWords = Bits / 32;
    bc::DeviceImage<T> dimg(static_cast<int>(w), static_cast<int>(h));
    bc::uploadImage<T>(img.data(), w, h, w, dimg.view());
    bc::DeviceArray<float> dxy(count * 2 > 0 ? count * 2 : 1);
    if (count > 0)
        cudaMemcpy(dxy.data(), xy.data(), count * 2 * sizeof(float), cudaMemcpyHostToDevice);
    bc::DeviceArray<uint32_t> dwords(count * kWords > 0 ? count * kWords : 1);
    bc::DeviceArray<uint8_t> dkeep(count > 0 ? count : 1);
    cudaMemset(dwords.data(), 0xCD, dwords.size() * sizeof(uint32_t));
    cudaMemset(dkeep.data(), 0xCD, dkeep.size());

    const size_t pairCount = steered != nullptr ? bc::steeredBriefPatternPairs<Bits>()
                                                : bc::briefPatternPairs<Bits>();
    bc::DeviceArray<bincv::BriefPair> dpairs(pairCount);
    bc::DeviceBriefPattern pat{};
    if (steered != nullptr)
        bc::uploadBriefPattern<Bits>(*steered, dpairs.data(), pat);
    else
        bc::uploadBriefPattern<Bits>(base, dpairs.data(), pat);

    bc::DeviceArray<float> dang(count > 0 ? count : 1);
    if (angles != nullptr && count > 0)
        cudaMemcpy(dang.data(), angles->data(), count * sizeof(float),
                   cudaMemcpyHostToDevice);

    bc::impl::briefBallotArmEnabled() = ballotArm;
    BriefRun<Bits, T> out;
    const bc::DeviceDescriptorSetView set =
        bc::descriptorSet(dwords.data(), count, kWords, dkeep.data());
    if (steered != nullptr)
        out.err = bc::computeBriefSteered(dimg.constView(), bc::keypointSet(dxy.data(), count),
                                          dang.data(), pat, set);
    else
        out.err = bc::computeBrief(dimg.constView(), bc::keypointSet(dxy.data(), count), pat,
                                   set);
    cudaDeviceSynchronize();
    bc::impl::briefBallotArmEnabled() = true;

    out.words.assign(count * kWords, 0u);
    out.keep.assign(count, 0);
    if (count > 0) {
        cudaMemcpy(out.words.data(), dwords.data(), count * kWords * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(out.keep.data(), dkeep.data(), count, cudaMemcpyDeviceToHost);
    }
    return out;
}

bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return false;
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// WIDE ORIENTATION
// ---------------------------------------------------------------------------

BINCV_TEST(CudaOrientation, WideAllThreeArmsAgreeWithHost) {
    struct Shape {
        size_t w, h;
    };
    // 131 is not a multiple of 32 and not a multiple of 4, so it is also the
    // case whose stride puts the __dp4a arm outside its addressing domain --
    // one shape exercising both the odd-width packing and the arm fallback.
    const Shape shapes[] = {{160, 96}, {131, 47}, {752, 480}};
    const int radii[] = {1, 15, 16, 31};
    const size_t counts[] = {0, 1, 7, 470};
    uint64_t seed = 0xB0BACAFEull;
    for (const Shape& s : shapes) {
        const auto img8 = makeFrame<uint8_t>(s.w, s.h, seed);
        const auto img16 = makeFrame<uint16_t>(s.w, s.h, seed + 1);
        for (int r : radii) {
            if (static_cast<size_t>(2 * r + 1) > s.h) continue;
            for (size_t n : counts) {
                const auto xy = makeKeypoints(s.w, s.h, n, r, seed + n);
                // W0 reference, W1 warp-scalar, W2 warp-quad: all three held to
                // the host AND to the independent per-pixel reference.
                checkWideRun<uint8_t>("u8 W0 reference", img8, s.w, s.h, xy, n, r,
                                      runWide<uint8_t>(img8, s.w, s.h, xy, n, r, false,
                                                       false));
                checkWideRun<uint8_t>("u8 W1 warp/column", img8, s.w, s.h, xy, n, r,
                                      runWide<uint8_t>(img8, s.w, s.h, xy, n, r, true,
                                                       false));
                checkWideRun<uint8_t>("u8 W2 warp/quad dp4a", img8, s.w, s.h, xy, n, r,
                                      runWide<uint8_t>(img8, s.w, s.h, xy, n, r, true,
                                                       true));
                checkWideRun<uint16_t>("u16 W0 reference", img16, s.w, s.h, xy, n, r,
                                       runWide<uint16_t>(img16, s.w, s.h, xy, n, r, false,
                                                         false));
                checkWideRun<uint16_t>("u16 W1 warp/column", img16, s.w, s.h, xy, n, r,
                                       runWide<uint16_t>(img16, s.w, s.h, xy, n, r, true,
                                                         true));
            }
        }
    }
}

BINCV_TEST(CudaOrientation, WideArmsAgreeWithEachOtherBitForBit) {
    // The switch test: three arms, one binary, one output. If the switch were
    // not wired, this would pass by comparing an arm against itself -- which is
    // why the arm-selection predicate is checked separately below.
    const size_t w = 752, h = 480;
    uint64_t seed = 0x1234567ull;
    const auto img = makeFrame<uint8_t>(w, h, seed);
    const auto xy = makeKeypoints(w, h, 1000, 15, seed);
    const auto a = runWide<uint8_t>(img, w, h, xy, 1000, 15, false, false);
    const auto b = runWide<uint8_t>(img, w, h, xy, 1000, 15, true, false);
    const auto c = runWide<uint8_t>(img, w, h, xy, 1000, 15, true, true);
    size_t bad = 0;
    for (size_t k = 0; k < 1000; ++k) {
        if (a.moments[2 * k] != b.moments[2 * k] ||
            a.moments[2 * k + 1] != b.moments[2 * k + 1])
            ++bad;
        if (a.moments[2 * k] != c.moments[2 * k] ||
            a.moments[2 * k + 1] != c.moments[2 * k + 1])
            ++bad;
        if (a.keep[k] != b.keep[k] || a.keep[k] != c.keep[k]) ++bad;
    }
    BINCV_CHECK_EQ(bad, size_t{0});
}

BINCV_TEST(CudaOrientation, QuadArmGateIsWhatItClaims) {
    // The mis-attached-#define scar, applied to a device arm: the gate must
    // EXCLUDE what it says it excludes, or the benchmark's ~1.00x control is
    // measuring the fast arm against itself and means nothing.
    bc::DeviceImage<uint8_t> wide(752, 480);
    bc::DeviceImage<uint8_t> odd(131, 47);
    bc::DeviceImage<uint16_t> wide16(752, 480);
    BINCV_CHECK(bc::impl::quadArmApplies<uint8_t>(wide.constView(), 15));
    BINCV_CHECK(!bc::impl::quadArmApplies<uint8_t>(wide.constView(), 16));
    BINCV_CHECK(!bc::impl::quadArmApplies<uint8_t>(odd.constView(), 15));
    BINCV_CHECK(!bc::impl::quadArmApplies<uint16_t>(wide16.constView(), 15));
}

// ---------------------------------------------------------------------------
// BIT-PLANE ORIENTATION
// ---------------------------------------------------------------------------

BINCV_TEST(CudaOrientation, BitPlaneAgreesWithHostAndWithTheWideArm) {
    struct Shape {
        size_t w, h;
    };
    const Shape shapes[] = {{160, 96}, {131, 47}, {752, 480}};
    const size_t planeCounts[] = {1, 2, 4};
    const int radii[] = {15, 16};
    uint64_t seed = 0xFEEDFACEull;
    for (const Shape& s : shapes) {
        const auto img = makeFrame<uint8_t>(s.w, s.h, seed);
        for (size_t pc : planeCounts) {
            const auto planes = makePlanes(img, s.w, s.h, pc);
            const auto values = planeValues(img, pc);
            std::vector<bincv::BinMatConstView<uint32_t>> hostViews;
            for (const auto& p : planes) hostViews.push_back(p.constView());
            for (int r : radii) {
                const size_t n = 470;
                const auto xy = makeKeypoints(s.w, s.h, n, r, seed + pc);
                std::vector<float> hostAngle(n, 0.0f);
                std::vector<uint8_t> hostKeep(n, 0);
                bincv::keypointOrientation<uint32_t>(hostViews.data(), pc, xy.data(), n,
                                                     hostAngle.data(), hostKeep.data(), r);
                // The wide arm on the EQUIVALENT pixel values: the two
                // spellings must produce the same integers, not close ones.
                std::vector<long long> wideMom;
                std::vector<uint8_t> wideKeep;
                referenceMoments<uint8_t>(values, s.w, s.h, xy, n, r, wideMom, wideKeep);

                for (int arm = 0; arm < 2; ++arm) {
                    const PlaneRun run =
                        runPlane(planes, s.w, s.h, xy, n, r, arm != 0, false);
                    size_t bad = 0, badKeep = 0;
                    unsigned maxUlp = 0;
                    for (size_t k = 0; k < n; ++k) {
                        if (run.moments[2 * k] != wideMom[2 * k] ||
                            run.moments[2 * k + 1] != wideMom[2 * k + 1])
                            ++bad;
                        if (run.keep[k] != hostKeep[k]) ++badKeep;
                        const unsigned u = ulpDistance(run.angle[k], hostAngle[k]);
                        if (u > maxUlp) maxUlp = u;
                    }
                    BINCV_CHECK_EQ(run.err, cudaSuccess);
                    BINCV_CHECK_EQ(bad, size_t{0});
                    BINCV_CHECK_EQ(badKeep, size_t{0});
                    std::printf("   bit-plane %zu-plane r=%2d arm=%d  max %u ULP\n", pc, r,
                                arm, maxUlp);
                }
            }
        }
    }
}

BINCV_TEST(CudaOrientation, BitPlanePaddingBitsCannotReachAPopcount) {
    // This family WRITES no padding -- its outputs are floats, flags and
    // descriptor words. It READS bit-planes whose padding the container
    // guarantees zero. What is pinned here is that the guarantee is not what
    // the answer depends on: the extraction masks to the run length, so a row
    // whose bits past `width` are all ones gives the same moments.
    const size_t w = 131, h = 47;
    uint64_t seed = 0xABCDEFull;
    const auto img = makeFrame<uint8_t>(w, h, seed);
    const auto planes = makePlanes(img, w, h, 1);
    const auto xy = makeKeypoints(w, h, 200, 15, seed);
    for (int arm = 0; arm < 2; ++arm) {
        const PlaneRun clean = runPlane(planes, w, h, xy, 200, 15, arm != 0, false);
        const PlaneRun dirty = runPlane(planes, w, h, xy, 200, 15, arm != 0, true);
        size_t bad = 0;
        for (size_t k = 0; k < 400; ++k)
            if (clean.moments[k] != dirty.moments[k]) ++bad;
        BINCV_CHECK_EQ(bad, size_t{0});
    }
}

// ---------------------------------------------------------------------------
// BRIEF
// ---------------------------------------------------------------------------

namespace {

template <size_t Bits, typename T>
void checkBriefAgainstHost(const char* what, const std::vector<T>& img, size_t w, size_t h,
                           const std::vector<float>& xy, size_t count,
                           const bincv::BriefPattern<Bits>& base,
                           const bincv::SteeredBriefPattern<Bits>* steered,
                           const std::vector<float>* angles) {
    constexpr size_t kWords = Bits / 32;
    std::vector<uint32_t> hostWords(count * kWords, 0u);
    std::vector<uint8_t> hostKeep(count, 0);
    if (count > 0) {
        if (steered != nullptr)
            bincv::computeBriefSteered<Bits, T, uint32_t>(img.data(), w, h, w, xy.data(),
                                                          count, angles->data(), *steered,
                                                          hostWords.data(), hostKeep.data());
        else
            bincv::computeBrief<Bits, T, uint32_t>(img.data(), w, h, w, xy.data(), count,
                                                   base, hostWords.data(), hostKeep.data());
    }
    for (int arm = 0; arm < 2; ++arm) {
        const BriefRun<Bits, T> run =
            runBrief<Bits, T>(img, w, h, xy, count, base, steered, angles, arm != 0);
        size_t badWords = 0, badKeep = 0;
        for (size_t i = 0; i < count * kWords; ++i)
            if (run.words[i] != hostWords[i]) ++badWords;
        for (size_t k = 0; k < count; ++k)
            if (run.keep[k] != hostKeep[k]) ++badKeep;
        BINCV_CHECK_EQ(run.err, cudaSuccess);
        BINCV_CHECK_EQ(badWords, size_t{0});
        BINCV_CHECK_EQ(badKeep, size_t{0});
        if (badWords != 0 || badKeep != 0)
            std::printf("   %s arm=%d: %zu/%zu words, %zu keep differ\n", what, arm,
                        badWords, count * kWords, badKeep);
    }
}

} // namespace

BINCV_TEST(CudaDescriptor, UnsteeredMatchesHostBitForBit) {
    const size_t w = 752, h = 480;
    uint64_t seed = 0x5A5A5Aull;
    const auto img8 = makeFrame<uint8_t>(w, h, seed);
    const auto img16 = makeFrame<uint16_t>(w, h, seed + 7);
    const size_t counts[] = {0, 1, 7, 470, 1000};

    bincv::BriefPattern<128> p128{};
    bincv::makeBriefPattern<128>(p128);
    bincv::BriefPattern<256> p256{};
    bincv::makeBriefPattern<256>(p256);
    // 33 words: above the ballot arm's own gate, so the reference arm runs and
    // must still match. This is the configuration the benchmark's ~1.00x
    // control uses.
    bincv::BriefPattern<1056> p1056{};
    bincv::makeBriefPattern<1056>(p1056);

    for (size_t n : counts) {
        const auto xy = makeKeypoints(w, h, n, 21, seed + n);
        checkBriefAgainstHost<128, uint8_t>("u8 128-bit", img8, w, h, xy, n, p128, nullptr,
                                            nullptr);
        checkBriefAgainstHost<256, uint8_t>("u8 256-bit", img8, w, h, xy, n, p256, nullptr,
                                            nullptr);
        checkBriefAgainstHost<1056, uint8_t>("u8 1056-bit (gate-excluded)", img8, w, h, xy,
                                             n, p1056, nullptr, nullptr);
        checkBriefAgainstHost<256, uint16_t>("u16 256-bit", img16, w, h, xy, n, p256,
                                             nullptr, nullptr);
    }
}

BINCV_TEST(CudaDescriptor, OrbTableMatchesHostBitForBit) {
    // ops/orbPattern.hpp's vendored table, reached BY POINTER and copied
    // nowhere new. tests/test_descriptor.cpp already pins the host's answer
    // with this table to cv::ORB::compute byte for byte; device == host here
    // therefore carries that pin across without re-measuring it.
    const size_t w = 640, h = 400;
    uint64_t seed = 0x0B0B0ull;
    const auto img = makeFrame<uint8_t>(w, h, seed);
    const auto xy = makeKeypoints(w, h, 800, 20, seed);
    checkBriefAgainstHost<256, uint8_t>("ORB table, unsteered", img, w, h, xy, 800,
                                        bincv::kOrbBriefPattern, nullptr, nullptr);
}

BINCV_TEST(CudaDescriptor, SteeredMatchesHostAndBinsAgree) {
    const size_t w = 752, h = 480;
    uint64_t seed = 0xC0FFEEull;
    const auto img = makeFrame<uint8_t>(w, h, seed);
    const size_t n = 1000;
    const auto xy = makeKeypoints(w, h, n, 21, seed);

    bincv::BriefPattern<256> base{};
    bincv::makeBriefPattern<256>(base);
    static bincv::SteeredBriefPattern<256> steered{};
    bincv::makeSteeredBriefPattern<256>(steered, base);

    // (a) Every bin exercised, plus angles placed EXACTLY on the 12-degree bin
    // boundaries -- the band where one ULP changes a whole descriptor.
    std::vector<float> angles(n, 0.0f);
    const float kTwoPi = 6.28318530717958647692f;
    const float kBinWidth = kTwoPi / 30.0f;
    for (size_t k = 0; k < n; ++k) {
        if (k % 3 == 0) {
            // exactly a boundary: the midpoint between two bin centres
            const float b = static_cast<float>(k % 30);
            angles[k] = (b + 0.5f) * kBinWidth - kTwoPi * 0.5f;
        } else {
            angles[k] = (static_cast<float>(k % 601) / 600.0f) * kTwoPi - kTwoPi * 0.5f;
        }
    }
    checkBriefAgainstHost<256, uint8_t>("steered, boundary-heavy angles", img, w, h, xy, n,
                                        base, &steered, &angles);

    // (b) All-zero angles must reproduce the UNSTEERED descriptor bit for bit:
    // bin 0 is the identity rotation, which the host's own test pins too.
    const std::vector<float> zeros(n, 0.0f);
    std::vector<uint32_t> unsteeredHost(n * 8, 0u);
    std::vector<uint8_t> unsteeredKeep(n, 0);
    bincv::computeBrief<256, uint8_t, uint32_t>(img.data(), w, h, w, xy.data(), n, base,
                                                unsteeredHost.data(), unsteeredKeep.data());
    for (int arm = 0; arm < 2; ++arm) {
        const auto run = runBrief<256, uint8_t>(img, w, h, xy, n, base, &steered, &zeros,
                                                arm != 0);
        size_t bad = 0;
        for (size_t i = 0; i < n * 8; ++i)
            if (run.words[i] != unsteeredHost[i]) ++bad;
        BINCV_CHECK_EQ(bad, size_t{0});
    }

    // (c) Steered by REAL angles out of the device orientation, which is how a
    // resident pipeline chains them. The bins the device selects must be the
    // bins the host selects from the host's own angles -- a divergence here is
    // the one failure mode that turns a 1-ULP angle into a wrong descriptor.
    const auto oriented = runWide<uint8_t>(img, w, h, xy, n, 15, true, true);
    std::vector<float> hostAngle(n, 0.0f);
    bincv::keypointOrientation<uint8_t>(img.data(), w, h, w, xy.data(), n, hostAngle.data(),
                                        nullptr, 15);
    size_t binDivergence = 0;
    for (size_t k = 0; k < n; ++k)
        if (bincv::briefAngleBin(oriented.angle[k]) != bincv::briefAngleBin(hostAngle[k]))
            ++binDivergence;
    BINCV_CHECK_EQ(binDivergence, size_t{0});
    checkBriefAgainstHost<256, uint8_t>("steered by device angles", img, w, h, xy, n, base,
                                        &steered, &hostAngle);
}

BINCV_TEST(CudaDescriptor, BinBoundariesWalkedInSingleUlpSteps) {
    // THE TEST THAT CATCHES A LOST `-fmad=false`.
    //
    // `briefAngleBin` ends in `angle * kBinsPerRadian + kBriefAngleBins`, a
    // CONTRACTIBLE multiply-add. Left to its defaults nvcc fuses it into one
    // `FFMA` (verified in the SASS); the backend compiles the translation unit
    // that calls it with `-fmad=false`, which emits `FMUL` then `FADD` -- the
    // unfused sequence a baseline x86-64 host produces. The two differ by up to
    // one ULP, and one ULP at a 12-degree boundary does not shift an angle
    // slightly: it selects a different rotation of the pattern and replaces the
    // whole 256-bit descriptor.
    //
    // So this walks each of the 30 boundaries in SINGLE-ULP steps either side
    // and requires the device's descriptor to be the host's, word for word.
    // It exercises the real shipped path -- the bin is never read out, it is
    // inferred from the descriptor it selects -- which is the only way to ask
    // the question that does not need a second copy of the expression.
    const size_t w = 752, h = 480;
    uint64_t seed = 0xB1Bull;
    const auto img = makeFrame<uint8_t>(w, h, seed);

    const float kTwoPi = 6.28318530717958647692f;
    const float kBinWidth = kTwoPi / 30.0f;
    std::vector<float> angles;
    for (int b = 0; b < 30; ++b) {
        const float boundary = (static_cast<float>(b) + 0.5f) * kBinWidth - kTwoPi * 0.5f;
        float up = boundary, down = boundary;
        angles.push_back(boundary);
        for (int u = 0; u < 8; ++u) {
            up = std::nextafterf(up, kTwoPi);
            down = std::nextafterf(down, -kTwoPi);
            angles.push_back(up);
            angles.push_back(down);
        }
    }
    const size_t n = angles.size();
    const auto xy = makeKeypoints(w, h, n, 21, seed);

    bincv::BriefPattern<256> base{};
    bincv::makeBriefPattern<256>(base);
    static bincv::SteeredBriefPattern<256> steered{};
    bincv::makeSteeredBriefPattern<256>(steered, base);
    checkBriefAgainstHost<256, uint8_t>("510 single-ULP boundary angles", img, w, h, xy, n,
                                        base, &steered, &angles);
}

BINCV_TEST(CudaDescriptor, WordWidthIsAByteIdentity) {
    // A descriptor computed at uint64_t on the host is the SAME BYTES as the
    // device's uint32 words on a little-endian host -- the narrowPlane
    // argument core/view.hpp makes about planes, applied to descriptors.
    // Checked rather than assumed.
    const size_t w = 320, h = 240, n = 300;
    uint64_t seed = 0x99ull;
    const auto img = makeFrame<uint8_t>(w, h, seed);
    const auto xy = makeKeypoints(w, h, n, 21, seed);
    bincv::BriefPattern<256> base{};
    bincv::makeBriefPattern<256>(base);

    std::vector<uint64_t> host64(n * 4, 0ull);
    bincv::computeBrief<256, uint8_t, uint64_t>(img.data(), w, h, w, xy.data(), n, base,
                                                host64.data(), nullptr);
    const auto run = runBrief<256, uint8_t>(img, w, h, xy, n, base, nullptr, nullptr, true);
    BINCV_CHECK_EQ(std::memcmp(host64.data(), run.words.data(), n * 32), 0);
}

BINCV_TEST(CudaDescriptor, EmptySetLaunchesNothingAndSucceeds) {
    bc::DeviceImage<uint8_t> img(64, 64);
    bc::DeviceArray<uint32_t> words(8);
    bincv::BriefPattern<256> base{};
    bincv::makeBriefPattern<256>(base);
    bc::DeviceArray<bincv::BriefPair> dpairs(bc::briefPatternPairs<256>());
    bc::DeviceBriefPattern pat{};
    BINCV_CHECK_EQ(bc::uploadBriefPattern<256>(base, dpairs.data(), pat), cudaSuccess);
    BINCV_CHECK_EQ(bc::computeBrief(img.constView(), bc::keypointSet(nullptr, 0), pat,
                                    bc::descriptorSet(words.data(), 0, 8)),
                   cudaSuccess);
    bc::DeviceArray<float> ang(1);
    BINCV_CHECK_EQ(bc::keypointOrientation(img.constView(), bc::keypointSet(nullptr, 0),
                                           ang.data()),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA orientation/descriptor tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA orientation/descriptor tests", argc, argv);
}
#endif
