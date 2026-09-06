// The orientation + steered-descriptor family, priced at birth.
//
// CLAUDE.md's rule, and the scar it came from: an operation with no benchmark arm
// ships correct and untimed, and nothing notices until something calls it -- at
// which point it was 78% of the frontend. These arms exist the day the kernels do.
//
// Core-only on purpose: the family needs no OpenCV, so the reference device's
// default build can produce these numbers. There is no OpenCV denominator here --
// cv::ORB computes orientation and steering fused into its describe, so the
// like-for-like comparison is the FUSED pair below against cv::ORB::compute, and
// that lives in feature_benchmark.cpp where OpenCV is available.
//
// MEMORY, stated rather than discovered: the steered pattern set is 30 rotations
// x Bits pairs x 4 bytes -- 30 720 B at 256 bits -- built once at setup, the
// price of rotation invariance without per-keypoint trigonometry. The kernels
// themselves take no scratch and allocate nothing; descriptors are the caller's
// output buffer (32 B per keypoint at 256 bits).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/orientation.hpp"
#include "bincv/quantMat.hpp"
#include "measure_util.hpp"

namespace {

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;
constexpr int kW = 752, kH = 480;   // the reference frame size
constexpr size_t kKp = 1000;

std::vector<uint8_t> smoothFrame(size_t w, size_t h) {
    // The same LCG-then-blur shape the tests use: sampling-error tolerance is not
    // being measured here, but a frame with realistic spatial correlation keeps the
    // branch behavior honest.
    std::vector<uint8_t> img(w * h);
    uint64_t st = 0xFEEDFACEULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    std::vector<uint8_t> tmp(w * h);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            tmp[y * w + x] = static_cast<uint8_t>(
                (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(
                (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) / 4u);
    return img;
}

} // namespace

int main() {
    const std::vector<uint8_t> img = smoothFrame(kW, kH);

    bincv::BinMat<uint32_t> bin1(kW, kH);
    bincv::QuantMat<2, uint32_t> bin2(kW, kH);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            const uint8_t v = img[static_cast<size_t>(y) * kW + static_cast<size_t>(x)];
            bin1.set(y, x, v >= 128 ? 1u : 0u);
            bin2.set(y, x, v >> 6);
        }

    std::vector<float> kp(kKp * 2);
    uint64_t st = 0x5EA15EEDULL;
    for (size_t i = 0; i < kKp; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        kp[2 * i] = static_cast<float>(20 + static_cast<int>((st >> 33) % (kW - 44)));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        kp[2 * i + 1] = static_cast<float>(20 + static_cast<int>((st >> 33) % (kH - 44)));
    }

    bincv::BriefPattern<kBits> base;
    bincv::makeBriefPattern<kBits>(base);
    bincv::SteeredBriefPattern<kBits> steered;
    bincv::makeSteeredBriefPattern<kBits>(steered, base);

    std::vector<float> angles(kKp);
    std::vector<uint32_t> desc(kKp * kWords);
    const bincv::BinMatConstView<uint32_t> planes2[2] = {bin2.constPlane(0),
                                                         bin2.constPlane(1)};

    std::printf("=== orientation and steered descriptors ===\n");
    std::printf(" %dx%d, %zu keypoints, %zu-bit descriptors, radius 15\n", kW, kH, kKp,
                kBits);
    std::printf(" steered pattern set: %zu B, built once\n\n", sizeof(steered));

    std::vector<measure::Bench> bs = {
        {"orientation, wide (uint8)",
         [&](int) {
             bincv::keypointOrientation<uint8_t>(img.data(), kW, kH, kW, kp.data(), kKp,
                                                 angles.data());
             measure::g_sink += static_cast<size_t>(angles[0] * 0.0f);
         }},
        {"orientation, bit-plane 1-bit (u32)",
         [&](int) {
             bincv::keypointOrientation<uint32_t>(bin1.constView(), kp.data(), kKp,
                                                  angles.data());
             measure::g_sink += static_cast<size_t>(angles[0] * 0.0f);
         }},
        {"orientation, bit-plane 2-bit (u32)",
         [&](int) {
             bincv::keypointOrientation<uint32_t>(planes2, 2, kp.data(), kKp,
                                                  angles.data());
             measure::g_sink += static_cast<size_t>(angles[0] * 0.0f);
         }},
        {"describe, unsteered BRIEF",
         [&](int) {
             bincv::computeBrief<kBits, uint8_t, uint32_t>(
                 img.data(), kW, kH, kW, kp.data(), kKp, base, desc.data());
             measure::g_sink += desc[0];
         }},
        {"describe, steered (angles precomputed)",
         [&](int) {
             bincv::computeBriefSteered<kBits, uint8_t, uint32_t>(
                 img.data(), kW, kH, kW, kp.data(), kKp, angles.data(), steered,
                 desc.data());
             measure::g_sink += desc[0];
         }},
        {"orient + describe steered (the ORB-shaped pair)",
         [&](int) {
             bincv::keypointOrientation<uint8_t>(img.data(), kW, kH, kW, kp.data(), kKp,
                                                 angles.data());
             bincv::computeBriefSteered<kBits, uint8_t, uint32_t>(
                 img.data(), kW, kH, kW, kp.data(), kKp, angles.data(), steered,
                 desc.data());
             measure::g_sink += desc[0];
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 60.0);
    std::printf(" %-46s %12s %14s\n", "arm", "ns/call", "ns/keypoint");
    for (size_t i = 0; i < bs.size(); ++i) {
        std::printf(" %-46s %12.0f %14.1f  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs, t[i].medianNs / static_cast<double>(kKp),
                    t[i].spreadPct());
    }
    std::printf("\n no scratch, no allocation in any arm; output buffers are the\n"
                " caller's %zu B of descriptors and %zu B of angles.\n",
                kKp * kWords * sizeof(uint32_t), kKp * sizeof(float));
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
