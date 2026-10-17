// cornerSubPix, priced against cv::cornerSubPix.
//
// THE RULE, WRITTEN BEFORE THE FIRST RUN: this is a pricing benchmark, not an
// adopt/reject gate -- the op is shipped. Metrics: milliseconds to refine the
// same corner set from the same seeds, binCV on its already-computed ternary
// derivatives against cv::cornerSubPix on the 8-bit image, plus each side's
// working set stated. If binCV loses its role comparison badly, that is a
// finding to report -- an operation ships only when it holds up on speed AND
// memory -- not a number to bury.
//
// THE RULE FOR SAMPLING THE WINDOW AT THE SUB-PIXEL POSITION, WRITTEN BEFORE THAT
// KERNEL WAS TIMED: the window moved off the rounded position because the rounded
// one was a different refinement -- a median of 0.98 px from cv::cornerSubPix on
// this file's frame -- and that is decided on agreement, which the test suite holds,
// not on speed. This file prices the result against cv::cornerSubPix, the role
// comparison; if binCV loses it on the device, the op does not ship as it is but
// goes to review with the gap. Its cost against the rounded-anchor kernel is
// recorded once, beside the figure it changes, because that kernel computed a
// different answer and is not a baseline anyone should choose.
//
// WHAT THE COMPARISON COVERS: refinement only. binCV's input premise is that
// the pipeline already holds ternary derivatives (that is the operation's
// documented shape); OpenCV's premise is the 8-bit image it resamples per
// call. Each side is timed doing its own whole job from its own natural input
// -- neither pays the other's preprocessing.
//
// CONTENT: realframe.bin, corners from cv::goodFeaturesToTrack -- the seeds a
// tracker would actually refine. Both refiners run the identical parameters
// (winHalf 5, no zero zone, 40 iterations, eps 0.001).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "bincv/binMat.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/subpix.hpp"
#include "measure_util.hpp"

using W = uint32_t;

namespace {

std::vector<uint8_t> loadRealFrame(int& w, int& h) {
    std::vector<uint8_t> px;
    FILE* f = std::fopen(BINCV_REALFRAME_PATH, "rb");
    if (!f) return px;
    uint32_t fw = 0, fh = 0;
    if (std::fread(&fw, 4, 1, f) != 1 || std::fread(&fh, 4, 1, f) != 1) {
        std::fclose(f);
        return px;
    }
    px.resize(static_cast<size_t>(fw) * fh);
    if (std::fread(px.data(), 1, px.size(), f) != px.size()) px.clear();
    std::fclose(f);
    w = static_cast<int>(fw);
    h = static_cast<int>(fh);
    return px;
}

} // namespace

int main() {
    int gw = 0, gh = 0;
    const std::vector<uint8_t> frame = loadRealFrame(gw, gh);
    if (frame.empty()) {
        std::fprintf(stderr, "realframe.bin unreadable -- the corner seeds ARE the "
                             "workload; refusing a synthetic fallback\n");
        return 1;
    }
    const size_t w = static_cast<size_t>(gw), h = static_cast<size_t>(gh);

    // The 8-bit image OpenCV refines on. realframe.bin holds {0,1}; the
    // refinement is invariant to a scale on the image, so 0/255 only aids gFTT.
    cv::Mat img(gh, gw, CV_8UC1);
    for (size_t i = 0; i < w * h; ++i)
        img.data[i] = frame[i] ? uint8_t{255} : uint8_t{0};

    std::vector<cv::Point2f> seeds;
    cv::goodFeaturesToTrack(img, seeds, 140, 0.01, 10.0);
    if (seeds.size() < 32) {
        std::fprintf(stderr, "only %zu seeds -- content too sparse to price\n",
                     seeds.size());
        return 1;
    }

    // binCV's natural input: the ternary derivatives the pipeline already holds --
    // with cv::cornerSubPix's replicated border, so the agreement printed below is
    // OpenCV's in the outermost rows and columns too.
    bincv::BinMat<W> bin(gw, gh);
    bincv::packBits<bincv::PackRule::NonZero>(frame.data(), w, h, w, bin.view());
    bincv::SignedQuantMat<1, W> dx(gw, gh), dy(gw, gh);
    bincv::derivativeX(bin, dx, bincv::BORDER_REPLICATE);
    bincv::derivativeY(bin, dy, bincv::BORDER_REPLICATE);

    bincv::SubPixParams p;   // winHalf 5, zeroHalf -1, 40 iterations, eps 0.001
    const cv::Size win(p.winHalf, p.winHalf), zero(-1, -1);
    const cv::TermCriteria crit(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                p.maxIterations, p.epsilon);

    // Agreement first, once, so the reader knows the two timed jobs end at the
    // same answers before caring which is faster.
    std::vector<bincv::Point2f> oursPts(seeds.size());
    std::vector<cv::Point2f> cvPts(seeds.begin(), seeds.end());
    for (size_t i = 0; i < seeds.size(); ++i)
        oursPts[i] = bincv::Point2f{seeds[i].x, seeds[i].y};
    bincv::cornerSubPix<1, W>(dx, dy, oursPts.data(), oursPts.size(), p);
    cv::cornerSubPix(img, cvPts, win, zero, crit);
    double sum = 0.0, worst = 0.0;
    for (size_t i = 0; i < seeds.size(); ++i) {
        const double ex = static_cast<double>(oursPts[i].x) - cvPts[i].x;
        const double ey = static_cast<double>(oursPts[i].y) - cvPts[i].y;
        const double d = std::sqrt(ex * ex + ey * ey);
        sum += d;
        worst = d > worst ? d : worst;
    }
    std::printf("=== cornerSubPix vs cv::cornerSubPix (role), %dx%d, %zu corners ===\n",
                gw, gh, seeds.size());
    std::printf(" |ours - cv| after refining the same seeds: mean %.2e px, worst %.2e px\n"
                " (the same refinement on the same bits; they differ by rounding, and the\n"
                "  test suite holds that to OpenCV's own 0/1-vs-0/255 floor)\n\n",
                sum / static_cast<double>(seeds.size()), worst);

    std::vector<bincv::Point2f> oursWork(seeds.size());
    std::vector<cv::Point2f> cvWork(seeds.size());
    std::vector<measure::Bench> bs = {
        {"bincv::cornerSubPix (ternary derivs)", [&](int) {
             for (size_t i = 0; i < seeds.size(); ++i)
                 oursWork[i] = bincv::Point2f{seeds[i].x, seeds[i].y};
             const auto r =
                 bincv::cornerSubPix<1, W>(dx, dy, oursWork.data(), oursWork.size(), p);
             measure::g_sink += r.refined;
         }},
        {"cv::cornerSubPix (8-bit image)", [&](int) {
             std::copy(seeds.begin(), seeds.end(), cvWork.begin());
             cv::cornerSubPix(img, cvWork, win, zero, crit);
             measure::g_sink += static_cast<size_t>(cvWork[0].x);
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 30.0);
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf(" %-38s %10.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    std::printf("\n ratio %.2fx  (>1 means binCV is faster)\n",
                t[1].medianNs / t[0].medianNs);
    std::printf(" working sets: binCV reads the pipeline's existing ternary planes, no\n"
                " allocation in the kernel; cv resamples the 8-bit frame per call.\n");
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
