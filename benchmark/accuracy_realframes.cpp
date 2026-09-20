// ===========================================================================
// THE REAL-FRAME ACCURACY HARNESS -- the candidate resolution to the synthetic
// harness's known failure, priced before anyone adopts it.
//
// The synthetic-warp harness and the pipeline disagree by ~4.2 yield points on
// the same configuration, and the float-cascade hypothesis is dead: correcting
// the cascade moved the number 0.12 where the gap is 4.2. What remains is
// structural -- the warp harness tracks binarizations of ONE image, so `prev`
// and `next` have near-identical edge maps, while the pipeline tracks real
// consecutive frames whose binarizations differ wherever a pixel sits near the
// threshold. The standing rule is therefore that no synthetic-harness accuracy
// conclusion may be promoted to a shipped default.
//
// THIS FILE IS THE OTHER BARGAIN: real consecutive frame pairs, with OpenCV's
// LK on the SAME binary content as the reference instead of a known warp. It
// trades exact ground truth for representativeness. The question that decided
// whether the trade was worth taking: does this harness reproduce the PIPELINE's
// configuration deltas (which the synthetic harness does not), while staying
// cheap enough to sweep with? Measured: it does -- on the axis where the
// synthetic harness said -0.42 and the pipeline said -4.60, this said -7.24 --
// and the owner ADOPTED it (2026-09-11): this harness may guide ladder/filter
// accuracy decisions, with a full pipeline run remaining the final gate before
// any shipped default changes. The synthetic harness stays restricted to
// sensitivity questions.
//
// Yield here is: of the keypoints BOTH trackers report tracked, the fraction
// whose flows agree within 1 px. It is agreement with a reference
// implementation, not truth -- where OpenCV's own flow is wrong, agreement
// rewards being wrong the same way. That is the priced imperfection, and it is
// the same one every recorded pipeline comparison already carries.
//
// Usage: accuracy_realframes <frame-dir> [pair-stride] [max-pairs]
//   pair-stride N takes every Nth consecutive pair (default 5), so the sweep
//   covers the sequence without paying for every frame.
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "reference_sensor.hpp"

#include "bincv/ops/corner.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/medianWide.hpp"
#include "bincv/ops/opticalFlow.hpp"
#include "bincv/ops/pyramid.hpp"

namespace fs = std::filesystem;
using W = uint32_t;
using bincv::Point2f;

namespace {

// The reference sensor stage -- ONE definition shared with feature_tracking_sequence.cpp,
// which checks binCV's own sensor stage bit-exact against it every frame. Both
// trackers here see identical binary content, so the only variable is the
// tracking configuration under test.
using refsensor::preprocess;

struct PairResult {
    size_t both = 0;      ///< tracked by both binCV and the reference
    size_t agree = 0;     ///< ... and within 1 px of each other's flow
};

/// One configuration, over one frame pair. Templated on the ladder and filter,
/// because that is the axis the synthetic harness and the pipeline disagree on.
template <size_t N1, size_t N2, size_t N3, bincv::PyrDownFilter F>
PairResult runPair(const cv::Mat& binPrev, const cv::Mat& binNext,
                   const std::vector<cv::Point2f>& pts) {
    const int w = binPrev.cols, h = binPrev.rows;
    bincv::Pyramid<W, 1, N1, N2, N3> prev(w, h), next(w, h);
    prev.template level<0>().fromCVMat(binPrev);
    next.template level<0>().fromCVMat(binNext);
    prev.template build<F, bincv::PyrDownBorder::Replicate>();
    next.template build<F, bincv::PyrDownBorder::Replicate>();

    bincv::SignedQuantMat<1, W> dx0(w, h), dy0(w, h);
    const int w1 = static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w)));
    const int h1 = static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h)));
    const int w2 = static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w1)));
    const int h2 = static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h1)));
    const int w3 = static_cast<int>(bincv::pyrDownWidth(static_cast<size_t>(w2)));
    const int h3 = static_cast<int>(bincv::pyrDownHeight(static_cast<size_t>(h2)));
    bincv::SignedQuantMat<N1, W> dx1(w1, h1), dy1(w1, h1);
    bincv::SignedQuantMat<N2, W> dx2(w2, h2), dy2(w2, h2);
    bincv::SignedQuantMat<N3, W> dx3(w3, h3), dy3(w3, h3);
    bincv::derivativeX(prev.template level<0>(), dx0);
    bincv::derivativeY(prev.template level<0>(), dy0);
    bincv::derivativeX(prev.template level<1>(), dx1);
    bincv::derivativeY(prev.template level<1>(), dy1);
    bincv::derivativeX(prev.template level<2>(), dx2);
    bincv::derivativeY(prev.template level<2>(), dy2);
    bincv::derivativeX(prev.template level<3>(), dx3);
    bincv::derivativeY(prev.template level<3>(), dy3);

    bincv::LKLevels<W, 1, N1, N2, N3> levels;
    levels.template get<0>() =
        bincv::lkLevel<1>(prev.template level<0>(), next.template level<0>(), dx0, dy0);
    levels.template get<1>() =
        bincv::lkLevel<N1>(prev.template level<1>(), next.template level<1>(), dx1, dy1);
    levels.template get<2>() =
        bincv::lkLevel<N2>(prev.template level<2>(), next.template level<2>(), dx2, dy2);
    levels.template get<3>() =
        bincv::lkLevel<N3>(prev.template level<3>(), next.template level<3>(), dx3, dy3);

    std::vector<Point2f> src, dst;
    std::vector<uint8_t> status;
    for (const auto& p : pts) src.push_back(Point2f{p.x, p.y});
    dst.assign(src.size(), Point2f{});
    status.assign(src.size(), 0);
    bincv::LKParams lk;
    bincv::calcOpticalFlowPyrLK(levels, src.data(), dst.data(), status.data(), nullptr,
                                src.size(), lk);

    // The reference: OpenCV on the same binary content, from the same points.
    std::vector<cv::Point2f> cvOut;
    std::vector<uchar> cvStatus;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(binPrev, binNext, pts, cvOut, cvStatus, err,
                             cv::Size(lk.winWidth, lk.winHeight), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                                              lk.maxIterations, lk.epsilon),
                             0, static_cast<double>(lk.minEigThreshold));

    PairResult r;
    for (size_t i = 0; i < src.size(); ++i) {
        if (!status[i] || i >= cvStatus.size() || !cvStatus[i]) continue;
        ++r.both;
        const double fx = (static_cast<double>(dst[i].x) - src[i].x) -
                          (static_cast<double>(cvOut[i].x) - pts[i].x);
        const double fy = (static_cast<double>(dst[i].y) - src[i].y) -
                          (static_cast<double>(cvOut[i].y) - pts[i].y);
        if (fx * fx + fy * fy <= 1.0) ++r.agree;
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: accuracy_realframes <frame-dir> [pair-stride] [max-pairs]\n");
        return 2;
    }
    const size_t stride = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 5;
    const size_t maxPairs = argc > 3 ? static_cast<size_t>(std::atoi(argv[3])) : 0;
    cv::setNumThreads(1);

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.size() < 2) { std::printf("need at least 2 frames\n"); return 2; }

    struct Config {
        const char* name;
        PairResult (*run)(const cv::Mat&, const cv::Mat&, const std::vector<cv::Point2f>&);
        PairResult total;
    };
    // The axis the disagreement lives on: the synthetic harness said the ladder
    // barely matters (-0.42 for dropping level 3's second bit at BOX_3x3) and the
    // pipeline said it matters a lot (-4.60). These six cells reproduce that
    // recorded sweep on real pairs.
    Config configs[] = {
        {"1/1/1/1 BOX_2x2", &runPair<1, 1, 1, bincv::PyrDownFilter::Box2x2>, {}},
        {"1/1/1/1 BOX_3x3", &runPair<1, 1, 1, bincv::PyrDownFilter::Box3x3>, {}},
        {"1/2/2/1 BOX_2x2", &runPair<2, 2, 1, bincv::PyrDownFilter::Box2x2>, {}},
        {"1/2/2/1 BOX_3x3", &runPair<2, 2, 1, bincv::PyrDownFilter::Box3x3>, {}},
        {"1/2/2/2 BOX_2x2", &runPair<2, 2, 2, bincv::PyrDownFilter::Box2x2>, {}},
        {"1/2/2/2 BOX_3x3", &runPair<2, 2, 2, bincv::PyrDownFilter::Box3x3>, {}},
    };

    const auto t0 = std::chrono::steady_clock::now();
    size_t pairs = 0;
    for (size_t f = 0; f + 1 < files.size(); f += stride) {
        const cv::Mat a = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        const cv::Mat b = cv::imread(files[f + 1].string(), cv::IMREAD_GRAYSCALE);
        if (a.empty() || b.empty()) continue;
        const cv::Mat binA = preprocess(a, 17);
        const cv::Mat binB = preprocess(b, 17);

        // The pipeline's own detector picks the points, so the harness scores the
        // pixels a real caller would actually track, not a synthetic grid.
        std::vector<cv::Point2f> pts;
        cv::goodFeaturesToTrack(binA, pts, 200, 0.01, 33.0, cv::noArray(), 3, false);
        if (pts.size() < 20) continue;

        for (auto& cfg : configs) {
            const PairResult r = cfg.run(binA, binB, pts);
            cfg.total.both += r.both;
            cfg.total.agree += r.agree;
        }
        ++pairs;
        if (maxPairs && pairs >= maxPairs) break;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("=== real-frame accuracy harness: %zu consecutive pairs, stride %zu ===\n",
                pairs, stride);
    std::printf(" yield = %% of points tracked by BOTH whose flows agree within 1 px\n");
    std::printf(" reference = cv::calcOpticalFlowPyrLK on the SAME binary frames\n\n");
    double anchor = -1.0;
    for (const auto& cfg : configs) {
        const double y = cfg.total.both
                             ? 100.0 * static_cast<double>(cfg.total.agree) /
                                   static_cast<double>(cfg.total.both)
                             : 0.0;
        if (anchor < 0.0) anchor = y;
        std::printf(" %-18s yield %6.2f%%  (delta %+5.2f vs first row; %zu compared)\n",
                    cfg.name, y, y - anchor, cfg.total.both);
    }
    std::printf("\n whole sweep: %.1f s for six configurations -- the price of a harness\n"
                " that tracks real pairs. Adopted for ladder/filter DIRECTION decisions\n"
                " (owner, 2026-09-11); a full pipeline run remains the final gate before\n"
                " a shipped default changes, and the synthetic harness answers only\n"
                " sensitivity questions.\n",
                secs);
    return 0;
}
