// The sparse rectified stereo matcher, priced at birth: an operation gets its
// benchmark arm when it is written.
//
// Core-only: no OpenCV denominator exists -- cv::StereoBM is dense and prices a
// different operation entirely -- so these arms compare binCV's stages against
// each other and against the keypoint count, which is what a pipeline budget
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
#include "stereo_scene.hpp"
#include "bincv/ops/stereo.hpp"
#include "bincv/ops/pack.hpp"
#include "measure_util.hpp"

using namespace stereoscene;  // the pair, built as bincv_m7_stereo builds it

int main() {
    std::vector<uint8_t> lw(kW * kH), rw(kW * kH), ring(3 * kW);
    buildLeft(lw.data(), ring.data());
    rw = lw;
    shiftToRight(rw.data());
    bincv::BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    pack(lw.data(), lb.view());
    pack(rw.data(), rb.view());

    std::vector<float> kpL(kKp * 2), kpR(kKp * 2);
    buildKeypoints(kpL.data(), kpR.data());

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
    std::printf(" %zux%zu pair, %zu vs %zu keypoints, disparity range [0, %d],"
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
    // The digests bincv_m7_stereo prints, for the board's to be held to, after both
    // stages once more: the timed arms ran in an interleaved order, and the board
    // digests what both stages leave.
    bincv::stereoMatchRectified<uint32_t>(lb.constView(), rb.constView(), kpL.data(), kKp,
                                          dl.data(), kpR.data(), kKp, dr.data(), kWords,
                                          m.data(), params);
    std::printf("\n-- digests, for the board's --\n");
    std::printf("  left bits        %08x\n", digestBits(lb.constView()));
    std::printf("  right bits       %08x\n", digestBits(rb.constView()));
    std::printf("  left brief       %08x\n", fnv1a(kFnvSeed, dl.data(), dl.size() * 4));
    std::printf("  right brief      %08x\n", fnv1a(kFnvSeed, dr.data(), dr.size() * 4));
    std::printf("  matches          %08x\n", digestMatches(m.data(), kKp));
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
