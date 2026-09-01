// ===========================================================================
// -- HOW MANY ITERATIONS DOES A POINT ACTUALLY RUN, AND WHAT DOES THE
// MAXIMUM OVER EIGHT COST?
//
// that work’s AVX2 keypoint batch puts eight keypoints in lanes and iterates them IN
// LOCKSTEP, so a batch runs until its LAST lane converges. a measurement measured the MEAN
// at 4.29 iterations per point per level; the batch pays the MAXIMUM OVER EIGHT,
// and that number decides whether the batch is worth writing.
//
// naive lockstep = kernel x mean(iters) / mean(batch max over 8)
// with lane refill = kernel x mean(iters) / (mean(iters) + refill)
//
// THE POINT OF MEASURING THIS FIRST is that it is decisive and nearly free.
// exists because a measurement measured 1.75x in a kernel and 0.31x on the frontend, and
// lockstep batching changes exactly the quantity that did that -- how many
// iterations run.
//
// This harness reproduces frontend_sequence's frontend exactly: the same
// preprocessing, the same 1/2/2/2 ladder, the same seal_params.yaml parameters and
// the same re-detection schedule, so the distribution is the one the shipped
// tracker sees and not one a synthetic frame produced.
//
// Usage: lk_iteration_histogram <frame-dir> [max-frames]
// ===========================================================================

// Before the include: the hook is off by default and this is the only consumer.
#define BINCV_LK_ITERATION_HISTOGRAM 1

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

/// EVERY iteration count the run produced, one entry per (frame, level, ACTIVE
/// point) in the order the tracker visited them. Batches are formed from this in
/// scan order, which is what the batched tracker would actually do.
std::vector<unsigned> gAll;
/// Batch maxima, in the same scan order, per (frame, level) group -- a batch never
/// spans two levels, because the tracker does not.
std::vector<unsigned> gBatchMax;
/// Total lane-slots the naive batch would run, including the wasted ones.
double gSlots = 0.0, gUsed = 0.0;

void accumulate(const unsigned* iters, size_t levelCount, size_t n) {
    constexpr size_t kLanes = 8;
    for (size_t li = 0; li < levelCount; ++li) {
        std::vector<unsigned> active;
        for (size_t p = 0; p < n; ++p) {
            const unsigned v = iters[li * n + p];
            if (v != 0) active.push_back(v);
        }
        for (unsigned v : active) gAll.push_back(v);
        for (size_t i = 0; i < active.size(); i += kLanes) {
            unsigned m = 0;
            const size_t end = std::min(i + kLanes, active.size());
            for (size_t j = i; j < end; ++j) m = std::max(m, active[j]);
            gBatchMax.push_back(m);
            // A short final batch still costs a full vector iteration.
            gSlots += static_cast<double>(m) * static_cast<double>(kLanes);
            for (size_t j = i; j < end; ++j) gUsed += static_cast<double>(active[j]);
        }
    }
}

double mean(const std::vector<unsigned>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (unsigned x : v) s += x;
    return s / static_cast<double>(v.size());
}

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
    constexpr size_t kLevels = 4;

    Frontend fe(w, h);
    std::vector<bincv::Corner> corners(20000);
    std::vector<bincv::Point2f> pts;
    std::vector<unsigned> iters;

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
            iters.assign(kLevels * pts.size(), 0u);
            bincv::impl::iterationTrace().counts = iters.data();
            bincv::impl::iterationTrace().pointCount = pts.size();
            bincv::calcOpticalFlowPyrLK(fe.levels, pts.data(), out.data(), status.data(),
                                        nullptr, pts.size(), lk);
            bincv::impl::iterationTrace().counts = nullptr;
            accumulate(iters.data(), kLevels, pts.size());

            std::vector<bincv::Point2f> kept;
            for (size_t i = 0; i < pts.size(); ++i) {
                if (status[i]) kept.push_back(out[i]);
            }
            pts.swap(kept);
        }
        binPrev = binNext;
    }

    if (gAll.empty()) { std::printf("no iterations recorded\n"); return 1; }

    unsigned hi = 0;
    for (unsigned v : gAll) hi = std::max(hi, v);
    std::vector<size_t> hist(hi + 1, 0);
    for (unsigned v : gAll) ++hist[v];

    const double meanIters = mean(gAll);
    const double meanMax = mean(gBatchMax);
    const double ratio = meanMax > 0.0 ? meanIters / meanMax : 0.0;

    std::printf("=== LK iteration distribution, %zu frames, cap %d ===\n",
                files.size(), lk.maxIterations);
    std::printf("point-levels tracked: %zu batches of 8: %zu\n\n",
                gAll.size(), gBatchMax.size());
    std::printf(" iters points share cumulative\n");
    double cum = 0.0;
    for (unsigned k = 1; k <= hi; ++k) {
        if (hist[k] == 0) continue;
        const double share = static_cast<double>(hist[k]) / static_cast<double>(gAll.size());
        cum += share;
        std::printf(" %5u %7zu %6.2f%% %6.2f%%\n", k, hist[k], share * 100.0,
                    cum * 100.0);
    }
    std::printf("\n mean iterations per point-level %8.3f\n", meanIters);
    std::printf(" mean MAXIMUM over a batch of 8 %8.3f\n", meanMax);
    std::printf(" ratio mean / mean-of-max-8 %8.3f\n", ratio);
    std::printf(" lane slots run %.0f, of which useful %.0f -> %.1f%% wasted\n",
                gSlots, gUsed, (1.0 - gUsed / gSlots) * 100.0);

    // The projection the decision rule is written against. The kernel factor is
    // that measurement’s arm D, measured; everything else here is this run's distribution.
    constexpr double kKernel = 2.1;
    std::printf("\n projected `track` speedup, naive lockstep %5.2fx\n", kKernel * ratio);
    std::printf(" projected `track` speedup, with lane refill %5.2fx (refill excluded)\n",
                kKernel);
    const char* band = ratio >= 0.70 ? "A -- naive lockstep is enough"
                     : ratio >= 0.45 ? "B -- lane refill"
                                     : "C -- refill MANDATORY, naive lockstep regresses";
    std::printf(" BAND: %s\n", band);
    return 0;
}
