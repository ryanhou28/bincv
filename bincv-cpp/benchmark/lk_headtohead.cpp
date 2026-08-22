// WHY IS binCV'S TRACKER SLOWER? LK against LK, same points, same window, same
// levels, same binary content, OpenCV pinned to one thread. Everything that is not
// the tracker is removed.
//
// The op-count model this is testing, written before the numbers:
//
//   binCV, per WORD (32 pixels), at depth N:
//     residualSums issues 5 tap sums x 2 components x N^2 plane pairs x 2
//     popcounts = 20N^2 popcounts. At N = 2 that is 80 popcounts per 32 pixels =
//     2.5 popcount-ops PER PIXEL.
//
//   OpenCV, per PIXEL:
//     one bilinear diff and two multiply-accumulates, ~7 ops, SIMD 8-16 wide
//     on CV_16S => ~0.5-0.9 ops per pixel.
//
// So the 32x packing advantage is spent three times over: N^2 plane pairs (4x at
// N = 2), the five-tap decomposition (binCV keeps four bilinear taps plus self as
// SEPARATE sums so the float weights can leave the pixel loop; OpenCV forms `diff`
// once per pixel), and OpenCV's vectorization. If the model is right, binCV should
// be issuing roughly an order of magnitude more work per pixel.
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"

using W = uint32_t;
using Clock = std::chrono::steady_clock;

int main() {
    cv::setNumThreads(1);
    const int w = 640, h = 480;
    cv::Mat b0(h, w, CV_8U, cv::Scalar(0)), b1(h, w, CV_8U, cv::Scalar(0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            b0.at<uchar>(y, x) = ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 255 : 0;
            b1.at<uchar>(y, x) =
                (((x - 1) * 7 + y * 13) % 29 == 0 || (x - 1 + y) % 37 == 0) ? 255 : 0;
        }
    }
    std::vector<cv::Point2f> cvPts;
    std::vector<bincv::Point2f> bPts;
    for (int y = 40; y < h - 40; y += 40) {
        for (int x = 40; x < w - 40; x += 40) {
            cvPts.emplace_back(static_cast<float>(x), static_cast<float>(y));
            bPts.push_back(bincv::Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    // ITERATION COUNT IS A CONFOUND AND IS CONTROLLED HERE. Both trackers stop
    // early on their own convergence rules, so at maxIterations = 20 they may do
    // DIFFERENT amounts of work and the ratio would not be of the kernels. Sweeping
    // a fixed low count with epsilon 0 forces both to run exactly that many.
    const int levels = 4, win = 31;
    int iters = 20;
    if (const char* e = std::getenv("BINCV_ITERS")) iters = std::atoi(e);
    const bool forceAll = std::getenv("BINCV_FORCE_ITERS") != nullptr;
    const float epsB = forceAll ? 0.0f : 0.03f;
    const double epsC = forceAll ? 0.0 : 0.03;

    // ---- binCV, the shipped 1/2/2/2 ladder ----
    bincv::Pyramid<W, 1, 2, 2, 2> p(w, h), n(w, h);
    p.level<0>().fromCVMat(b0);
    n.level<0>().fromCVMat(b1);
    p.build(); n.build();
    bincv::SignedQuantMat<1, W> dx0(w, h), dy0(w, h);
    bincv::SignedQuantMat<2, W> dx1(320,240), dy1(320,240), dx2(160,120), dy2(160,120),
        dx3(80,60), dy3(80,60);
    bincv::derivativeX(p.level<0>(), dx0); bincv::derivativeY(p.level<0>(), dy0);
    bincv::derivativeX(p.level<1>(), dx1); bincv::derivativeY(p.level<1>(), dy1);
    bincv::derivativeX(p.level<2>(), dx2); bincv::derivativeY(p.level<2>(), dy2);
    bincv::derivativeX(p.level<3>(), dx3); bincv::derivativeY(p.level<3>(), dy3);
    bincv::LKLevels<W, 1, 2, 2, 2> L;
    L.get<0>() = bincv::lkLevel<1>(p.level<0>(), n.level<0>(), dx0, dy0);
    L.get<1>() = bincv::lkLevel<2>(p.level<1>(), n.level<1>(), dx1, dy1);
    L.get<2>() = bincv::lkLevel<2>(p.level<2>(), n.level<2>(), dx2, dy2);
    L.get<3>() = bincv::lkLevel<2>(p.level<3>(), n.level<3>(), dx3, dy3);

    // ---- binCV, an all-1-bit ladder, to isolate the N^2 term ----
    bincv::Pyramid<W, 1, 1, 1, 1> p1(w, h), n1(w, h);
    p1.level<0>().fromCVMat(b0); n1.level<0>().fromCVMat(b1);
    p1.build(); n1.build();
    bincv::SignedQuantMat<1, W> e0(w,h), f0(w,h), e1(320,240), f1(320,240),
        e2(160,120), f2(160,120), e3(80,60), f3(80,60);
    bincv::derivativeX(p1.level<0>(), e0); bincv::derivativeY(p1.level<0>(), f0);
    bincv::derivativeX(p1.level<1>(), e1); bincv::derivativeY(p1.level<1>(), f1);
    bincv::derivativeX(p1.level<2>(), e2); bincv::derivativeY(p1.level<2>(), f2);
    bincv::derivativeX(p1.level<3>(), e3); bincv::derivativeY(p1.level<3>(), f3);
    bincv::LKLevels<W, 1, 1, 1, 1> L1;
    L1.get<0>() = bincv::lkLevel<1>(p1.level<0>(), n1.level<0>(), e0, f0);
    L1.get<1>() = bincv::lkLevel<1>(p1.level<1>(), n1.level<1>(), e1, f1);
    L1.get<2>() = bincv::lkLevel<1>(p1.level<2>(), n1.level<1>(), e2, f2);
    L1.get<3>() = bincv::lkLevel<1>(p1.level<3>(), n1.level<3>(), e3, f3);

    bincv::LKParams lk; lk.winWidth = win; lk.winHeight = win; lk.maxIterations = iters;
    lk.epsilon = epsB;
    std::vector<bincv::Point2f> out(bPts.size());
    std::vector<uint8_t> stt(bPts.size());

    auto timeIt = [](const char* name, auto&& fn, int reps) {
        fn();  // warm
        const auto t0 = Clock::now();
        for (int i = 0; i < reps; ++i) fn();
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;
        std::printf("  %-42s %8.3f ms\n", name, ms);
        return ms;
    };

    std::printf("=== LK vs LK: %zu points, %dx%d window, %d levels, %d iterations%s,"
                " OpenCV 1 thread ===\n\n", bPts.size(), win, win, levels, iters,
                forceAll ? " FORCED (eps=0)" : "");
    const double tB2 = timeIt("binCV LK, ladder 1/2/2/2 (shipped)", [&] {
        bincv::calcOpticalFlowPyrLK(L, bPts.data(), out.data(), stt.data(), nullptr,
                                    bPts.size(), lk); }, 20);
    const double tB1 = timeIt("binCV LK, ladder 1/1/1/1 (N=1 everywhere)", [&] {
        bincv::calcOpticalFlowPyrLK(L1, bPts.data(), out.data(), stt.data(), nullptr,
                                    bPts.size(), lk); }, 20);
    std::vector<cv::Point2f> cvOut; std::vector<uchar> cvSt; std::vector<float> cvErr;
    const double tCV = timeIt("OpenCV LK on the same bits as CV_8U", [&] {
        cv::calcOpticalFlowPyrLK(b0, b1, cvPts, cvOut, cvSt, cvErr, cv::Size(win, win),
                                 levels - 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                                  iters, epsC), 0, 1e-4); }, 20);

    auto ratio = [](double a, double b) {
        // Printing "0.29x slower" is a trap: below 1.0 it means FASTER. Say which.
        return a <= b ? std::string("FASTER") : std::string("slower");
    };
    std::printf("\n  binCV 1/2/2/2 vs OpenCV : %.2fx  (%s)\n", tB2 <= tCV ? tCV / tB2 : tB2 / tCV,
                ratio(tB2, tCV).c_str());
    std::printf("  binCV 1/1/1/1 vs OpenCV : %.2fx  (%s)\n", tB1 <= tCV ? tCV / tB1 : tB1 / tCV,
                ratio(tB1, tCV).c_str());
    std::printf("  cost of the N=2 ladder  : %.2fx\n", tB2 / tB1);

    const double wordsPerWin = 2.0 * 31.0;
    const double pxPerWin = 31.0 * 31.0;
    std::printf("\n  OP-COUNT MODEL, per pixel per iteration per level:\n");
    for (int N : {1, 2}) {
        const double pc = wordsPerWin * 20.0 * N * N;
        std::printf("    binCV N=%d : %6.0f popcounts per window = %.2f per pixel\n", N, pc,
                    pc / pxPerWin);
    }
    std::printf("    OpenCV    : ~7 ops per pixel, SIMD 8-16 wide = ~0.4-0.9 per pixel\n");
    return 0;
}
