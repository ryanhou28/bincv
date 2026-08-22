// ===========================================================================
// T4.3a / E-5 -- END-TO-END VALIDATION OVER A REAL SEQUENCE.
//
// Three of the four ROADMAP success criteria, and the fourth (tier-1
// bit-exactness) is already enforced per operation:
//
//   2. tier-2 operations agreeing with the REFERENCE FRONTEND frame by frame --
//      feature positions, flow vectors, track lifetimes
//   3. several-fold smaller PEAK FOOTPRINT over the frontend operation set
//   4. FASTER execution against the byte-per-pixel denominator
//
// THE DENOMINATOR IS CLAUDE.md's, NOT A FLATTERING ONE: OpenCV doing the same
// semantic operation on the SAME BINARY CONTENT stored as CV_8U. Both frontends
// see bit-identical input -- the reference pipeline's two-stage preprocessing,
// median_filter then rl_fast_edge_filter_wide -- so the comparison is of the
// implementations and not of the content.
//
// TWO INDEPENDENT FRONTENDS, NOT ONE DRIVING THE OTHER. Each detects its own
// corners, maintains its own track set and re-detects on its own schedule. That is
// what makes track LIFETIME a comparable quantity: a harness that fed binCV's
// points to OpenCV would measure per-frame flow agreement and nothing else, and
// lifetime is the criterion that most directly reflects whether a frontend is
// usable.
//
// Usage: frontend_sequence <frame-dir> [max-frames]
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"

using W = uint32_t;
using Clock = std::chrono::steady_clock;

namespace {

// ---- the reference pipeline's preprocessing, both stages ------------------
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
cv::Mat referenceEdgeFilter(const cv::Mat& gray, int thr) {
    const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(gray, dx, CV_32F, kx);
    cv::filter2D(gray, dy, CV_32F, ky);
    dx = cv::abs(dx);
    dy = cv::abs(dy);
    const cv::Mat mask = (dx >= thr) | (dy >= thr);
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8U);
    out.setTo(255, mask);
    return out;
}
cv::Mat preprocess(const cv::Mat& g, int thr) {
    return referenceEdgeFilter(referenceDenoise(g), thr);
}

// ---- binCV's frontend state, ladder 1/2/2/2 (D-23) -----------------------
struct BincvFrontend {
    static constexpr size_t kLevels = 4;
    bincv::Pyramid<W, 1, 2, 2, 2> prev, next;
    bincv::SignedQuantMat<1, W> dx0, dy0;
    bincv::SignedQuantMat<2, W> dx1, dy1, dx2, dy2, dx3, dy3;
    bincv::LKLevels<W, 1, 2, 2, 2> levels;
    std::vector<float> ring;  // kResponseRingRows rows, the streaming response
    int w, h;

    BincvFrontend(int width, int height)
        : prev(width, height), next(width, height), dx0(width, height), dy0(width, height),
          dx1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dy1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dx2((width + 3) / 4, (height + 3) / 4), dy2((width + 3) / 4, (height + 3) / 4),
          dx3((width + 7) / 8, (height + 7) / 8), dy3((width + 7) / 8, (height + 7) / 8),
          ring(bincv::kResponseRingRows * static_cast<size_t>(width)), w(width), h(height) {}

    void loadLevel0(const cv::Mat& binPrev, const cv::Mat& binNext) {
        prev.level<0>().fromCVMat(binPrev);
        next.level<0>().fromCVMat(binNext);
    }
    void build() {
        prev.build();
        next.build();
        bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
        bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
        bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
        bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);
        levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
        levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
        levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
        levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);
    }
    /// Peak working set of the whole frontend operation set, by construction.
    size_t bytes() const {
        const size_t pyr = prev.sizeInBytes() + next.sizeInBytes();
        const size_t der = (dx0.sizeInWords() + dy0.sizeInWords() + dx1.sizeInWords() +
                            dy1.sizeInWords() + dx2.sizeInWords() + dy2.sizeInWords() +
                            dx3.sizeInWords() + dy3.sizeInWords()) * sizeof(W);
        return pyr + der + ring.size() * sizeof(float);
    }
};

/// OpenCV's frontend footprint on the same content, computed the way it allocates:
/// a CV_8U pyramid with a winSize border on every level (buildOpticalFlowPyramid),
/// plus goodFeaturesToTrack's CV_32F response map.
size_t opencvBytes(int w, int h, int levels, int win) {
    size_t bytes = 0;
    int cw = w, ch = h;
    for (int l = 0; l < levels; ++l) {
        bytes += static_cast<size_t>(cw + 2 * win) * static_cast<size_t>(ch + 2 * win);
        cw = (cw + 1) / 2;
        ch = (ch + 1) / 2;
    }
    bytes *= 2;                                                    // two frames
    bytes += static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(float);   // eigen map
    return bytes;
}

struct Stats {
    size_t frames = 0, detections = 0;
    size_t bTried = 0, bSurvived = 0, oTried = 0, oSurvived = 0;
    double flowRmsPx = 0.0, flowMaxPx = 0.0;
    size_t compared = 0, agreeWithin1px = 0;
    std::vector<double> flowErrs;   // X-25's lesson: this distribution has a tail
    double bincvMs = 0.0, opencvMs = 0.0;
    // PER-STAGE, INSIDE THE REAL LOOP. X-30 profiled one detection and one track
    // per frame; the real frontend re-detects on a few percent of frames, and that
    // over-weighted detection ~33x and sent D-27's target list to the wrong kernel.
    // These timers are taken at the ACTUAL duty cycle.
    double msBuild = 0.0, msDetect = 0.0, msTrack = 0.0;
    std::vector<int> bincvLifetimes, opencvLifetimes;
};

double median(std::vector<int> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc < 2) { std::printf("usage: frontend_sequence <frame-dir> [max-frames]\n"); return 2; }
    const size_t maxFrames = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 0;

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (maxFrames && files.size() > maxFrames) files.resize(maxFrames);
    if (files.size() < 2) { std::printf("need at least 2 frames\n"); return 2; }

    // THREAD CONTROL. The default comparison lets OpenCV use every core while
    // binCV runs on one, which conflates "OpenCV has more cores" with "OpenCV has
    // better code". Setting BINCV_OPENCV_THREADS=1 separates them. Neither run is
    // the honest one on its own -- both are reported.
    if (const char* t = std::getenv("BINCV_OPENCV_THREADS")) cv::setNumThreads(std::atoi(t));

    const cv::Mat first = cv::imread(files[0].string(), cv::IMREAD_GRAYSCALE);
    const int w = first.cols, h = first.rows;
    std::printf("=== T4.3a / E-5: frontend over %zu frames, %dx%d ===\n", files.size(), w, h);
    std::printf("both frontends see bit-identical input: median_filter then "
                "rl_fast_edge_filter_wide(17)\n\n");

    bincv::LKParams lk;                       // seal_params.yaml verbatim
    // The iteration cap is seal_params.yaml's 20. Nothing in this project has ever
    // measured how many iterations the tracker actually NEEDS, and at 94.7% of
    // frontend time an unnecessary iteration is the most expensive thing there is.
    if (const char* it = std::getenv("BINCV_LK_ITERS")) lk.maxIterations = std::atoi(it);
    bincv::GoodFeaturesParams gftt;           // ditto
    const int kMinTracks = 60;

    BincvFrontend fe(w, h);
    Stats st;
    // CAPACITY MUST EXCEED maxCorners, AND BY A LOT. `goodFeaturesToTrack` ranks
    // NMS survivors into this array and only THEN applies the minDistance spacing
    // filter, so a capacity of maxCorners truncates the candidate pool before
    // spacing thins it -- and spacing at minDistance 33 rejects most candidates.
    // Measured on the first V1_02 frame: capacity 200 yields 61 corners, capacity
    // 20000 yields 193 against OpenCV's 200. The earlier version of this harness
    // passed maxCorners and read the shortfall as a binCV defect; `CornerResult
    // ::candidatesTruncated` was reporting the truncation the whole time, which is
    // exactly what T3.11 added it for.
    std::vector<bincv::Corner> corners(20000);
    size_t truncatedDetections = 0;

    std::vector<bincv::Point2f> bPts;                 // binCV's live tracks
    std::vector<int> bAge;
    std::vector<cv::Point2f> oPts;                    // OpenCV's live tracks
    std::vector<int> oAge;

    cv::Mat binPrev = preprocess(first, 17);
    for (size_t f = 1; f < files.size(); ++f) {
        const cv::Mat gray = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;
        const cv::Mat binNext = preprocess(gray, 17);

        // ---------------- binCV ----------------
        auto t0 = Clock::now();
        auto tStage = t0;
        fe.loadLevel0(binPrev, binNext);
        fe.build();
        st.msBuild += std::chrono::duration<double, std::milli>(Clock::now() - tStage).count();
        tStage = Clock::now();
        if (bPts.size() < static_cast<size_t>(kMinTracks)) {
            bincv::ResponseMap ringMap{fe.ring.data(), static_cast<size_t>(w),
                                       bincv::kResponseRingRows, static_cast<size_t>(w)};
            const bincv::CornerResult r = bincv::goodFeaturesToTrackStreaming<W>(
                fe.dx0, fe.dy0, gftt, ringMap, corners.data(), corners.size());
            for (int a : bAge) st.bincvLifetimes.push_back(a);   // discarded, not immortal
            bPts.clear(); bAge.clear();
            for (size_t i = 0; i < r.count; ++i) {
                bPts.push_back(bincv::Point2f{static_cast<float>(corners[i].x),
                                              static_cast<float>(corners[i].y)});
                bAge.push_back(0);
            }
            if (r.candidatesTruncated) ++truncatedDetections;
            ++st.detections;
        }
        st.msDetect += std::chrono::duration<double, std::milli>(Clock::now() - tStage).count();
        tStage = Clock::now();
        std::vector<bincv::Point2f> bOut(bPts.size());
        std::vector<uint8_t> bStatus(bPts.size());
        if (!bPts.empty()) {
            bincv::calcOpticalFlowPyrLK(fe.levels, bPts.data(), bOut.data(), bStatus.data(),
                                        nullptr, bPts.size(), lk);
        }
        st.msTrack += std::chrono::duration<double, std::milli>(Clock::now() - tStage).count();
        st.bincvMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // ---------------- OpenCV, same binary content as CV_8U ----------------
        t0 = Clock::now();
        if (oPts.size() < static_cast<size_t>(kMinTracks)) {
            std::vector<cv::Point2f> found;
            cv::goodFeaturesToTrack(binPrev, found, gftt.maxCorners, gftt.qualityLevel,
                                    gftt.minDistance, cv::noArray(), gftt.blockSize, false);
            for (int a : oAge) st.opencvLifetimes.push_back(a);
            oPts = found;
            oAge.assign(oPts.size(), 0);
        }
        std::vector<cv::Point2f> oOut;
        std::vector<uchar> oStatus;
        std::vector<float> oErr;
        if (!oPts.empty()) {
            cv::calcOpticalFlowPyrLK(binPrev, binNext, oPts, oOut, oStatus, oErr,
                                     cv::Size(lk.winWidth, lk.winHeight), 3,
                                     cv::TermCriteria(cv::TermCriteria::COUNT +
                                                          cv::TermCriteria::EPS,
                                                      lk.maxIterations, lk.epsilon),
                                     0, static_cast<double>(lk.minEigThreshold));
        }
        st.opencvMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        // ---------------- criterion 2: flow agreement ----------------
        // NEAREST-NEIGHBOUR MATCHING. The two frontends detect independently, so
        // their arrays share no ordering and index-matching compares unrelated
        // points -- which is what an earlier version of this file did, and it
        // reported zero comparisons rather than wrong ones, which is how it was
        // caught. A pair counts only if the two START positions coincide to well
        // under a pixel, so the flow difference is of the same physical feature.
        for (size_t i = 0; i < bPts.size(); ++i) {
            if (i >= bOut.size() || !bStatus[i]) continue;
            size_t best = oPts.size();
            double bestD2 = 0.25 * 0.25;
            for (size_t j2 = 0; j2 < oPts.size(); ++j2) {
                if (j2 >= oOut.size() || !oStatus[j2]) continue;
                const double ddx = static_cast<double>(bPts[i].x) - oPts[j2].x;
                const double ddy = static_cast<double>(bPts[i].y) - oPts[j2].y;
                const double d2 = ddx * ddx + ddy * ddy;
                if (d2 < bestD2) { bestD2 = d2; best = j2; }
            }
            if (best == oPts.size()) continue;
            const size_t j = best;
            const double fx = (static_cast<double>(bOut[i].x) - bPts[i].x) -
                              (oOut[j].x - oPts[j].x);
            const double fy = (static_cast<double>(bOut[i].y) - bPts[i].y) -
                              (oOut[j].y - oPts[j].y);
            const double e = std::sqrt(fx * fx + fy * fy);
            st.flowErrs.push_back(e);
            st.flowRmsPx += e * e;
            st.flowMaxPx = std::max(st.flowMaxPx, e);
            if (e <= 1.0) ++st.agreeWithin1px;
            ++st.compared;
        }

        // ---------------- criterion 2: track lifetimes ----------------
        auto compact = [&](auto& pts, auto& age, const auto& out, const auto& status,
                           std::vector<int>& deaths, int W2, int H2) {
            size_t k = 0;
            for (size_t i = 0; i < pts.size(); ++i) {
                const bool alive = status[i] && out[i].x >= 0 && out[i].y >= 0 &&
                                   out[i].x < static_cast<float>(W2) &&
                                   out[i].y < static_cast<float>(H2);
                if (alive) {
                    pts[k] = out[i];
                    age[k] = age[i] + 1;
                    ++k;
                } else {
                    deaths.push_back(age[i] + 1);
                }
            }
            pts.resize(k);
            age.resize(k);
        };
        st.bTried += bPts.size();
        st.oTried += oPts.size();
        if (!bPts.empty()) compact(bPts, bAge, bOut, bStatus, st.bincvLifetimes, w, h);
        if (!oPts.empty()) compact(oPts, oAge, oOut, oStatus, st.opencvLifetimes, w, h);
        st.bSurvived += bPts.size();
        st.oSurvived += oPts.size();

        binPrev = binNext;
        ++st.frames;
        if (st.frames % 200 == 0) std::printf("  ... %zu frames\n", st.frames);
    }

    // surviving tracks count too, or long-lived tracks are invisible
    for (int a : bAge) st.bincvLifetimes.push_back(a);
    for (int a : oAge) st.opencvLifetimes.push_back(a);

    const size_t bcvBytes = fe.bytes();
    const size_t ocvBytes = opencvBytes(w, h, 4, lk.winWidth);

    std::printf("\n--- CRITERION 2: agreement with the reference frontend ---\n");
    std::printf("  flow vectors compared : %zu over %zu frames (%zu re-detections)\n",
                st.compared, st.frames, st.detections);
    // REPORTED AS PERCENTILES, NOT AS RMS. X-25 established that this project's
    // flow errors are a tight body with a small catastrophic tail, and that an RMS
    // over everything reports the tail as though it were the body -- that finding
    // cost two experiments' worth of misattribution, so it is applied here.
    std::sort(st.flowErrs.begin(), st.flowErrs.end());
    auto pct = [&](double f) {
        return st.flowErrs.empty() ? 0.0
                                   : st.flowErrs[std::min(st.flowErrs.size() - 1,
                                                          static_cast<size_t>(
                                                              f * static_cast<double>(
                                                                      st.flowErrs.size())))];
    };
    std::printf("  flow difference       : median %.4f px   p90 %.4f   p99 %.4f   max %.4f\n",
                pct(0.50), pct(0.90), pct(0.99), st.flowMaxPx);
    std::printf("  (RMS over all         : %.4f px -- reported for completeness; the\n"
                "   comparisons)           percentiles above are the honest summary, see X-25\n",
                st.compared ? std::sqrt(st.flowRmsPx / static_cast<double>(st.compared)) : 0.0);
    std::printf("  agreeing within 1 px  : %.1f%%\n",
                st.compared ? 100.0 * static_cast<double>(st.agreeWithin1px) /
                                  static_cast<double>(st.compared) : 0.0);
    std::printf("  track lifetime median : binCV %.0f frames   OpenCV %.0f frames\n",
                median(st.bincvLifetimes), median(st.opencvLifetimes));
    std::printf("  tracks observed       : binCV %zu   OpenCV %zu\n", st.bincvLifetimes.size(),
                st.opencvLifetimes.size());
    std::printf("  per-frame survival    : binCV %.1f%%   OpenCV %.1f%%\n",
                st.bTried ? 100.0 * static_cast<double>(st.bSurvived) /
                                static_cast<double>(st.bTried) : 0.0,
                st.oTried ? 100.0 * static_cast<double>(st.oSurvived) /
                                static_cast<double>(st.oTried) : 0.0);

    std::printf("  detections truncated  : %zu of %zu (capacity %zu)\n", truncatedDetections,
                st.detections, corners.size());
    std::printf("\n--- CRITERION 3: peak footprint over the frontend operation set ---\n");
    std::printf("  binCV  : %8zu B   (1/2/2/2 pyramid x2, derivative ladders, 3-row response ring)\n",
                bcvBytes);
    std::printf("  OpenCV : %8zu B   (CV_8U pyramid x2 with %d-px border/level, CV_32F eigen map)\n",
                ocvBytes, lk.winWidth);
    std::printf("  RATIO  : %.2fx smaller\n",
                static_cast<double>(ocvBytes) / static_cast<double>(bcvBytes));

    std::printf("\n--- binCV per-stage, AT THE REAL DUTY CYCLE ---\n");
    {
        const double f = static_cast<double>(st.frames);
        const double tot = st.msBuild + st.msDetect + st.msTrack;
        std::printf("  %-28s %8.3f ms/frame  %5.1f%%   (%zu detections in %zu frames = %.1f%%)\n",
                    "detect", st.msDetect / f, 100.0 * st.msDetect / tot, st.detections,
                    st.frames, 100.0 * static_cast<double>(st.detections) / f);
        std::printf("  %-28s %8.3f ms/frame  %5.1f%%\n", "track (LK)", st.msTrack / f,
                    100.0 * st.msTrack / tot);
        std::printf("  %-28s %8.3f ms/frame  %5.1f%%\n", "build (pyrDown + derivatives)",
                    st.msBuild / f, 100.0 * st.msBuild / tot);
    }
    std::printf("\n--- CRITERION 4: speed against the byte-per-pixel denominator ---\n");
    std::printf("  binCV  : %8.3f ms/frame\n", st.bincvMs / static_cast<double>(st.frames));
    std::printf("  OpenCV : %8.3f ms/frame\n", st.opencvMs / static_cast<double>(st.frames));
    std::printf("  RATIO  : %.2fx\n", st.opencvMs / st.bincvMs);
    std::printf("  NOTE: OpenCV threads = %d, and its LK and gftt are SIMD-vectorized.\n",
                cv::getNumThreads());
#if defined(BINCV_HAVE_NEON) && defined(__aarch64__)
    std::printf("        binCV has its NEON path here (D-30, D-33), so this is SIMD against\n"
                "        SIMD on the deployment target -- the comparison criterion 4 is about.\n");
#else
    std::printf("        binCV has NO VECTOR PATH ON x86 (ROADMAP 5.3 is unwritten), so this\n"
                "        is binCV SCALAR against OpenCV SSE. It is a fact about this machine,\n"
                "        not about the product, whose target is Cortex-A. See X-37.\n");
#endif
    return 0;
}
