// ===========================================================================
// X-83 -- WHERE `track` TIME ACTUALLY GOES ON THE REFERENCE DEVICE, BY STAGE.
//
// An iteration-cap sweep on the device put roughly 45% of `track` OUTSIDE the
// iteration loop:
//
//     cap 1   4.207 ms      cap 4   5.759
//     cap 2   5.522         cap 20  5.766     (mean iterations 1.98)
//
// Nothing in this project has ever measured WHICH of the per-point stages that is --
// staging, the covariance, the clip. Two guesses had already been made and measured at
// 1.9% and 0.0%, which is what this benchmark is for. X-67/D-59 is the same lesson from
// the frontend's side: `build` looked like one thing and decomposed into three, one of
// which was 3.6%.
//
// Same frontend, ladder and parameters as `frontend_sequence`, so the shares are the
// shipped tracker's and not a synthetic frame's.
//
// Usage: lk_stage_profile <frame-dir> [max-frames]
// ===========================================================================

// Before the include: the hook is off by default and this is the only consumer.
#define BINCV_LK_STAGE_TIMING 1

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"

using W = uint32_t;

namespace {

// ---- frontend_sequence's preprocessing, verbatim -------------------------
cv::Mat referenceDenoise(const cv::Mat& img) {
    cv::Mat right = cv::Mat::zeros(img.size(), img.type());
    cv::Mat above = cv::Mat::zeros(img.size(), img.type());
    img.colRange(1, img.cols).copyTo(right.colRange(0, img.cols - 1));
    img.rowRange(0, img.rows - 1).copyTo(above.rowRange(1, img.rows));
    cv::Mat a, b, c, out;
    cv::min(above, img, a);
    cv::max(above, img, b);
    cv::min(b, right, c);
    cv::max(a, c, out);
    return out;
}
cv::Mat preprocess(const cv::Mat& gray, int thr) {
    const cv::Mat den = referenceDenoise(gray);
    const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(den, dx, CV_32F, kx);
    cv::filter2D(den, dy, CV_32F, ky);
    dx = cv::abs(dx);
    dy = cv::abs(dy);
    const cv::Mat mask = (dx >= thr) | (dy >= thr);
    cv::Mat out = cv::Mat::zeros(den.size(), CV_8U);
    out.setTo(255, mask);
    return out;
}

struct Frontend {
    bincv::Pyramid<W, 1, 2, 2, 2> prev, next;
    bincv::SignedQuantMat<1, W> dx0, dy0;
    bincv::SignedQuantMat<2, W> dx1, dy1, dx2, dy2, dx3, dy3;
    bincv::LKLevels<W, 1, 2, 2, 2> levels;
    std::vector<float> ring;

    Frontend(int w, int h)
        : prev(w, h), next(w, h), dx0(w, h), dy0(w, h),
          dx1(w / 2 + (w & 1), h / 2 + (h & 1)), dy1(w / 2 + (w & 1), h / 2 + (h & 1)),
          dx2((w + 3) / 4, (h + 3) / 4), dy2((w + 3) / 4, (h + 3) / 4),
          dx3((w + 7) / 8, (h + 7) / 8), dy3((w + 7) / 8, (h + 7) / 8),
          ring(bincv::kResponseRingRows * static_cast<size_t>(w)) {}

    void build(const cv::Mat& binPrev, const cv::Mat& binNext) {
        prev.level<0>().fromCVMat(binPrev);
        next.level<0>().fromCVMat(binNext);
        prev.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        next.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
        bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
        bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
        bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);
        levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
        levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
        levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
        levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);
    }
};

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc < 2) {
        std::printf("usage: lk_iteration_histogram <frame-dir> [max-frames]\n");
        return 2;
    }
    const size_t maxFrames = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 0;

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (maxFrames && files.size() > maxFrames) files.resize(maxFrames);
    if (files.size() < 2) { std::printf("need at least 2 frames\n"); return 2; }

    const cv::Mat first = cv::imread(files[0].string(), cv::IMREAD_GRAYSCALE);
    const int w = first.cols, h = first.rows;

    bincv::LKParams lk;                     // seal_params.yaml verbatim
    bincv::GoodFeaturesParams gftt;
    const int kMinTracks = 60;
    // (the ladder depth; the stage counters are per point-level, not per level)

    // THE SCALAR PATH ON BOTH ARCHITECTURES, DELIBERATELY. x86 takes D-66's keypoint
    // batch, which does not go through `trackOnePoint` and so records nothing -- and
    // the reference device has no batch at all. Profiling the same code both places is
    // what makes the two columns comparable, and the device column is the one X-83 is
    // about.
#if defined(BINCV_X86_LK_BATCH)
    bincv::impl::lkBatchEnabled() = false;
#endif

    Frontend fe(w, h);
    std::vector<bincv::Corner> corners(20000);
    std::vector<bincv::Point2f> pts;

    cv::Mat binPrev = preprocess(first, 17);
    for (size_t f = 1; f < files.size(); ++f) {
        const cv::Mat gray = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;
        const cv::Mat binNext = preprocess(gray, 17);
        fe.build(binPrev, binNext);

        if (pts.size() < static_cast<size_t>(kMinTracks)) {
            bincv::ResponseMap ringMap{fe.ring.data(), static_cast<size_t>(w),
                                       bincv::kResponseRingRows, static_cast<size_t>(w)};
            const bincv::CornerResult r = bincv::goodFeaturesToTrackStreaming<W>(
                fe.dx0, fe.dy0, gftt, ringMap, corners.data(), corners.size());
            pts.clear();
            for (size_t i = 0; i < r.count; ++i) {
                pts.push_back(bincv::Point2f{static_cast<float>(corners[i].x),
                                             static_cast<float>(corners[i].y)});
            }
        }
        if (!pts.empty()) {
            std::vector<bincv::Point2f> out(pts.size());
            std::vector<uint8_t> status(pts.size());
            bincv::calcOpticalFlowPyrLK(fe.levels, pts.data(), out.data(), status.data(),
                                        nullptr, pts.size(), lk);

            std::vector<bincv::Point2f> kept;
            for (size_t i = 0; i < pts.size(); ++i) {
                if (status[i]) kept.push_back(out[i]);
            }
            pts.swap(kept);
        }
        binPrev = binNext;
    }

    const bincv::impl::StageTiming& s = bincv::impl::stageTiming();
    const double total = static_cast<double>(s.setup + s.staging + s.covariance + s.residual);
    if (total <= 0.0) {
        std::printf("no stages recorded\n");
        return 1;
    }
    std::printf("=== X-83: `track` by stage, %zu frames, %llu point-levels ===\n",
                files.size(), s.points);
    const struct { const char* name; unsigned long long ns; } rows[] = {
        {"setup (bounds, clipRegion)", s.setup},
        {"staging (stageWindow)", s.staging},
        {"covariance + eigen test", s.covariance},
        {"iteration loop (taps + residualSums)", s.residual},
    };
    for (const auto& r : rows) {
        std::printf("  %-38s %10.3f ms   %5.1f%%   %7.1f ns/point-level\n", r.name,
                    static_cast<double>(r.ns) / 1e6,
                    100.0 * static_cast<double>(r.ns) / total,
                    s.points ? static_cast<double>(r.ns) / static_cast<double>(s.points) : 0.0);
    }
    std::printf("  %-38s %10.3f ms\n", "TOTAL (instrumented)", total / 1e6);
    // How much of the iteration loop is TAP EXTRACTION rather than arithmetic. A window
    // is 31 rows, so `tapRows / iterations / 31` is the fraction of iterations that had
    // to refresh their taps -- X-70's cache absorbing the rest.
    std::printf("\n  residualSums calls per point-level  %8.3f\n",
                s.points ? static_cast<double>(s.iterations) / static_cast<double>(s.points) : 0.0);
    std::printf("  tap ROWS extracted per point-level %8.1f\n",
                s.points ? static_cast<double>(s.tapRows) / static_cast<double>(s.points) : 0.0);
    std::printf("  tap refreshes per residualSums     %8.3f   (1.0 = the cache never hits)\n",
                s.iterations ? static_cast<double>(s.tapRows) / static_cast<double>(s.iterations) / 31.0
                             : 0.0);
    return 0;
}
