// The sparse rectified stereo matcher, priced at birth (CLAUDE.md's rule).
//
// Core-only: no OpenCV denominator exists -- cv::StereoBM is dense and prices a
// different operation entirely -- so these arms compare binCV's stages against
// each other and against the keypoint count, which is what a frontend budget
// needs. The synthetic pair is a shifted frame: every keypoint has a true
// disparity, so the arms are running the code path that matters (candidates
// exist and survive the gates) rather than an early-out storm.
//
// MEMORY: no scratch, no allocation in any stage; the output is the caller's
// 24 B per left keypoint of StereoMatch records.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/shift.hpp"
#include "bincv/ops/stereo.hpp"
#include "bincv/ops/pack.hpp"
#include "measure_util.hpp"

namespace {

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;
constexpr int kW = 752, kH = 480;   // the reference frame size
constexpr int kDisp = 21;
constexpr size_t kKp = 500;

std::vector<uint8_t> smoothFrame(size_t w, size_t h) {
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
    const std::vector<uint8_t> lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(static_cast<size_t>(kW) * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kDisp];

    bincv::BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), size_t{kW}, size_t{kH}, size_t{kW}, lb.view(), uint8_t{127});
    bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), size_t{kW}, size_t{kH}, size_t{kW}, rb.view(), uint8_t{127});

    std::vector<float> kpL(kKp * 2), kpR(kKp * 2);
    uint64_t st = 0x5EA15EEDULL;
    for (size_t i = 0; i < kKp; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        const int x = 80 + static_cast<int>((st >> 33) % (kW - 160));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        const int y = 40 + static_cast<int>((st >> 33) % (kH - 80));
        kpL[2 * i] = static_cast<float>(x);
        kpL[2 * i + 1] = static_cast<float>(y);
        kpR[2 * i] = static_cast<float>(x - kDisp);
        kpR[2 * i + 1] = static_cast<float>(y);
    }

    bincv::BriefPattern<kBits> pat;
    bincv::makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> dl(kKp * kWords), dr(kKp * kWords);
    bincv::computeBrief<kBits, uint8_t, uint32_t>(lw.data(), kW, kH, kW, kpL.data(), kKp,
                                                  pat, dl.data());
    bincv::computeBrief<kBits, uint8_t, uint32_t>(rw.data(), kW, kH, kW, kpR.data(), kKp,
                                                  pat, dr.data());

    bincv::StereoMatchParams params;
    params.maxDisparity = 64;
    std::vector<bincv::StereoMatch> m(kKp);

    std::printf("=== sparse rectified stereo ===\n");
    std::printf(" %dx%d pair, %zu vs %zu keypoints, disparity range [0, %d],"
                " %dx%d refine window\n\n",
                kW, kH, kKp, kKp, params.maxDisparity, params.winWidth, params.winHeight);

    std::vector<measure::Bench> bs = {
        {"descriptor stage (row band + range gate)",
         [&](int) {
             bincv::stereoDescriptorMatch<uint32_t>(kpL.data(), kKp, dl.data(), kpR.data(),
                                                    kKp, dr.data(), kWords, m.data(),
                                                    params);
             measure::g_sink += m[0].rightIndex;
         }},
        {"refinement stage (Hamming windows + parabola)",
         [&](int) {
             for (size_t i = 0; i < kKp; ++i) {
                 m[i].valid = 1;
                 m[i].disparity = static_cast<float>(kDisp);
             }
             bincv::stereoRefineDisparity<uint32_t>(lb.constView(), rb.constView(),
                                                    kpL.data(), kKp, m.data(), params);
             measure::g_sink += static_cast<size_t>(m[0].disparity);
         }},
        {"both (stereoMatchRectified)",
         [&](int) {
             bincv::stereoMatchRectified<uint32_t>(lb.constView(), rb.constView(),
                                                   kpL.data(), kKp, dl.data(), kpR.data(),
                                                   kKp, dr.data(), kWords, m.data(),
                                                   params);
             measure::g_sink += m[0].valid;
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 60.0);
    std::printf(" %-46s %12s %14s\n", "arm", "ns/call", "ns/keypoint");
    for (size_t i = 0; i < bs.size(); ++i) {
        std::printf(" %-46s %12.0f %14.1f  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs, t[i].medianNs / static_cast<double>(kKp),
                    t[i].spreadPct());
    }

    size_t valid = 0;
    for (size_t i = 0; i < kKp; ++i)
        if (m[i].valid) ++valid;
    std::printf("\n %zu of %zu keypoints matched (the arms are exercising the full\n"
                " path, not the reject).\n",
                valid, kKp);
    std::printf(" no scratch; output is %zu B of StereoMatch records.\n",
                kKp * sizeof(bincv::StereoMatch));
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
