// ===========================================================================
// X-30 / E-12 -- WHERE THE FRONTEND'S TIME ACTUALLY GOES, and therefore what
// Phase 5.1 should vectorize.
//
// E-12 asks how much of an `ops/` kernel's PER-ROW cost is genericity that is not
// in N. It was registered against T3.5's derivative -- and against "every ops/
// kernel with a per-row prologue", which is the half that still matters, because
// [X-28] then measured the frontend end to end and the derivative turned out to be
// worth almost nothing:
//
//     corner detection   31.590 ms/frame   68.83%     (X-23, 640x480)
//     LK track           14.020 ms/frame   30.55%     (X-26, 140 keypoints)
//     build               0.285 ms/frame    0.62%     (pyrDown x2 + BOTH
//                                                      derivative ladders)
//
// ELIMINATING THE ENTIRE BUILD STAGE CAPS THE FRONTEND GAIN AT 1.0062x. So
// answering E-12 precisely on the derivative would be optimizing 0.6%, and the
// question worth the measurement is the same question asked of the 99%.
//
// This file splits the two hot stages, on the reference device, at the frontend's
// real operating point. The splits are made by DIFFERENCE rather than by
// instrumentation, so nothing is perturbed by a timer inside a loop:
//
//   LK      maxIterations = 0 runs per-point setup, clipping, the covariance and
//           the minEig test, and NO residual. Subtracting it from the full call
//           separates "the 2x2 matrix" from "the iteration".
//   corner  cornerMinEigenVal is the response sweep alone; goodFeaturesToTrack
//           adds NMS, ranking and the spacing filter. The difference is selection.
// ===========================================================================

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

using W = uint32_t;

int main() {
    const int width = 640, height = 480;
    bincv::Pyramid<W, 1, 2, 2, 2> prev(width, height), next(width, height);
    bincv::SignedQuantMat<1, W> dx0(width, height), dy0(width, height);
    bincv::SignedQuantMat<2, W> dx1(320, 240), dy1(320, 240), dx2(160, 120), dy2(160, 120),
        dx3(80, 60), dy3(80, 60);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            prev.level<0>().set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);
            next.level<0>().set(y, x,
                                (((x - 1) * 7 + y * 13) % 29 == 0 || (x - 1 + y) % 37 == 0) ? 1u
                                                                                           : 0u);
        }
    }
    prev.build();
    next.build();
    bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
    bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
    bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
    bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);

    bincv::LKLevels<W, 1, 2, 2, 2> levels;
    levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
    levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
    levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
    levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);

    std::vector<bincv::Point2f> pts;
    for (int y = 40; y < height - 40; y += 40) {
        for (int x = 40; x < width - 40; x += 40) {
            pts.push_back(bincv::Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    std::vector<bincv::Point2f> out(pts.size());
    std::vector<uint8_t> status(pts.size());

    std::vector<float> response(static_cast<size_t>(width) * static_cast<size_t>(height));
    std::vector<float> ring(bincv::kResponseRingRows * static_cast<size_t>(width));
    std::vector<bincv::Corner> corners(20000);
    bincv::GoodFeaturesParams gftt;

    std::printf("=== X-30 / E-12: where the frontend's time goes ===\n");
    std::printf("640x480, ladder 1/2/2/2, %zu keypoints, 31x31 window, 4 levels\n\n", pts.size());

    auto lkAt = [&](int iters) {
        return [&, iters](int) {
            bincv::LKParams p;
            p.maxIterations = iters;
            bincv::calcOpticalFlowPyrLK(levels, pts.data(), out.data(), status.data(), nullptr,
                                        pts.size(), p);
        };
    };
    std::vector<measure::Bench> benches = {
        {"LK, maxIterations = 20 (full)", lkAt(20)},
        {"LK, maxIterations =  0 (setup + covariance + minEig only)", lkAt(0)},
        {"corner: response sweep only (cornerMinEigenVal, frame map)",
         [&](int) {
             bincv::ResponseMap m{response.data(), static_cast<size_t>(width),
                                  static_cast<size_t>(height), static_cast<size_t>(width)};
             bincv::cornerMinEigenVal<W>(dx0, dy0, gftt.blockSize, m);
         }},
        {"corner: full streaming detect (response + NMS + rank + spacing)",
         [&](int) {
             bincv::ResponseMap r{ring.data(), static_cast<size_t>(width),
                                  bincv::kResponseRingRows, static_cast<size_t>(width)};
             const auto res = bincv::goodFeaturesToTrackStreaming<W>(dx0, dy0, gftt, r,
                                                                     corners.data(),
                                                                     corners.size());
             measure::g_sink += res.count;
         }},
        {"build: pyrDown x2 + both derivative ladders",
         [&](int) {
             prev.build();
             next.build();
             bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
             bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
             bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
             bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);
         }},
    };
    const auto t = measure::measureInterleaved(benches, 7, 60.0);
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf("  %-58s %9.3f ms   spread %.0f%%\n", benches[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    }
    const double lkFull = t[0].medianNs, lkSetup = t[1].medianNs;
    const double respOnly = t[2].medianNs, detFull = t[3].medianNs;
    std::printf("\n  DERIVED SPLITS (by difference, nothing instrumented inside a loop)\n");
    std::printf("    LK covariance + setup   : %9.3f ms  (%.1f%% of LK)\n", lkSetup / 1e6,
                100.0 * lkSetup / lkFull);
    std::printf("    LK residual + solve     : %9.3f ms  (%.1f%% of LK)\n",
                (lkFull - lkSetup) / 1e6, 100.0 * (lkFull - lkSetup) / lkFull);
    std::printf("    corner response sweep   : %9.3f ms  (%.1f%% of detect)\n", respOnly / 1e6,
                100.0 * respOnly / detFull);
    std::printf("    corner selection        : %9.3f ms  (%.1f%% of detect)\n",
                (detFull - respOnly) / 1e6, 100.0 * (detFull - respOnly) / detFull);
    return 0;
}
