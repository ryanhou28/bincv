// ===========================================================================
// A BINARY-FRAME VIO VISION FRONTEND, END TO END, ON binCV KERNELS.
//
// T4.3b asked whether binCV's kernel set is SUFFICIENT for a real VIO frontend.
// Every prior end-to-end measurement in this project (X-28, X-38, X-49) runs a
// benchmark loop: detect wholesale every N frames, track, compare. A real
// frontend does something structurally different, and this is that loop --
// modelled on HybVIO's, which is what the SEAL paper's pipeline drives:
//
//   1. TEMPORAL / SENSOR STAGE, in OpenCV.  median_filter then
//      rl_fast_edge_filter_wide (SEAL/src/temporal_processing/) turn an 8-bit
//      camera frame into a binary one. In SEAL this is dedicated hardware, and
//      it is NOT binCV's claim: binCV's domain starts at the binary frame,
//      which ARCHITECTURE 7.2 calls "the input, not a choice".
//
//   2. EVERYTHING AFTER, in binCV.  Pyramid, derivatives, LK tracking,
//      detection, and the track lifecycle.
//
// WHAT THIS EXERCISES THAT A BENCHMARK LOOP DOES NOT:
//   * a PERSISTENT TRACK SET carried across frames, not a per-frame rematch;
//   * CULLING on LK status and on leaving the frame -- HybVIO's
//     FAILED_FLOW / FLOW_OUT_OF_RANGE (src/tracker/optical_flow.cpp);
//   * TOPPING UP by detection only when the count falls below a target, with
//     applyMinDistance against the SURVIVORS so new corners do not land on
//     features already being tracked (src/tracker/feature_detector_legacy.cpp).
//     binCV has no mask parameter by design -- ops/corner.hpp documents the
//     spacing filter as the route -- and this is that route taken.
//
// It reports what a frontend is judged on: how many features it holds, how long
// they live, and why they die.
// ===========================================================================
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/denoise.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/ops/pyramid.hpp"
#include "bincv-cpp/quantMat.hpp"

namespace fs = std::filesystem;
using W = uint32_t;
using bincv::Point2f;

namespace {

// ---- 1. The temporal/sensor stage. OpenCV, and deliberately not binCV. ----
// SEAL/src/temporal_processing/denoise.cpp: a three-pixel median.
// SEAL/src/temporal_processing/edge_filter.cpp: |d/dx| + |d/dy| over [-1,0,1],
// thresholded. Read from the reference rather than inferred (CLAUDE.md).
cv::Mat sensorStage(const cv::Mat& gray, int edgeThreshold) {
    cv::Mat med;
    cv::medianBlur(gray, med, 3);
    cv::Mat kx = (cv::Mat_<float>(1, 3) << -1, 0, 1);
    cv::Mat ky = (cv::Mat_<float>(3, 1) << -1, 0, 1);
    cv::Mat dx, dy;
    cv::filter2D(med, dx, CV_32F, kx);
    cv::filter2D(med, dy, CV_32F, ky);
    cv::Mat mag = cv::abs(dx) + cv::abs(dy);
    cv::Mat bin;
    cv::threshold(mag, bin, static_cast<double>(edgeThreshold), 255.0, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8U);
    return bin;
}

/// One tracked feature. `age` is what a VIO backend cares about: a feature seen
/// in two frames constrains nothing, one seen in ten constrains a lot.
struct Track {
    Point2f p;
    int age = 0;
    int id = 0;
};

/// HybVIO's applyMinDistance, against the SURVIVING tracks rather than against
/// the previous detection -- which is what makes this a top-up and not a
/// re-detect. O(new x live); both are a few hundred at most.
void applyMinDistance(std::vector<Point2f>& fresh, const std::vector<Track>& live, float r) {
    if (r < 1.0f) return;
    const float r2 = r * r;
    size_t out = 0;
    for (size_t i = 0; i < fresh.size(); ++i) {
        bool ok = true;
        for (const Track& t : live) {
            const float ddx = fresh[i].x - t.p.x, ddy = fresh[i].y - t.p.y;
            if (ddx * ddx + ddy * ddy < r2) { ok = false; break; }
        }
        if (!ok) continue;
        for (size_t j = 0; j < out; ++j) {   // and against each other
            const float ddx = fresh[i].x - fresh[j].x, ddy = fresh[i].y - fresh[j].y;
            if (ddx * ddx + ddy * ddy < r2) { ok = false; break; }
        }
        if (ok) fresh[out++] = fresh[i];
    }
    fresh.resize(out);
}

/// binCV's half of the frontend: the shipped 1/2/2/2 ladder (D-23), the box
/// downsample (D-39 -- `build()` would default to cv::pyrDown's Gaussian), the
/// derivative ladder, and the streaming response ring (D-26) so detection needs
/// no frame-sized float map.
struct Frontend {
    bincv::Pyramid<W, 1, 2, 2, 2> prev, next;
    bincv::SignedQuantMat<1, W> dx0, dy0;
    bincv::SignedQuantMat<2, W> dx1, dy1, dx2, dy2, dx3, dy3;
    bincv::LKLevels<W, 1, 2, 2, 2> levels;
    std::vector<float> ring;
    int w, h;

    Frontend(int width, int height)
        : prev(width, height), next(width, height), dx0(width, height), dy0(width, height),
          dx1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dy1(width / 2 + (width & 1), height / 2 + (height & 1)),
          dx2((width + 3) / 4, (height + 3) / 4), dy2((width + 3) / 4, (height + 3) / 4),
          dx3((width + 7) / 8, (height + 7) / 8), dy3((width + 7) / 8, (height + 7) / 8),
          ring(bincv::kResponseRingRows * static_cast<size_t>(width)), w(width), h(height) {}

    void buildFrom(const cv::Mat& binary) {
        next.level<0>().fromCVMat(binary);
        next.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
    }
    /// Derivatives come from `prev`, which is what LK linearises about.
    void bindAll() {
        bincv::derivativeX(prev.level<0>(), dx0); bincv::derivativeY(prev.level<0>(), dy0);
        bincv::derivativeX(prev.level<1>(), dx1); bincv::derivativeY(prev.level<1>(), dy1);
        bincv::derivativeX(prev.level<2>(), dx2); bincv::derivativeY(prev.level<2>(), dy2);
        bincv::derivativeX(prev.level<3>(), dx3); bincv::derivativeY(prev.level<3>(), dy3);
        levels.get<0>() = bincv::lkLevel<1>(prev.level<0>(), next.level<0>(), dx0, dy0);
        levels.get<1>() = bincv::lkLevel<2>(prev.level<1>(), next.level<1>(), dx1, dy1);
        levels.get<2>() = bincv::lkLevel<2>(prev.level<2>(), next.level<2>(), dx2, dy2);
        levels.get<3>() = bincv::lkLevel<2>(prev.level<3>(), next.level<3>(), dx3, dy3);
    }
    size_t bytes() const {
        return prev.sizeInBytes() + next.sizeInBytes() +
               (dx0.sizeInWords() + dy0.sizeInWords() + dx1.sizeInWords() + dy1.sizeInWords() +
                dx2.sizeInWords() + dy2.sizeInWords() + dx3.sizeInWords() + dy3.sizeInWords()) *
                   sizeof(W) +
               ring.size() * sizeof(float);
    }
};

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: vio_frontend <frame-dir> [max-frames]\n");
        return 2;
    }
    const size_t maxFrames = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 0;
    cv::setNumThreads(1);

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (maxFrames && files.size() > maxFrames) files.resize(maxFrames);
    if (files.size() < 2) { std::printf("need at least 2 frames\n"); return 2; }

    const cv::Mat first = cv::imread(files[0].string(), cv::IMREAD_GRAYSCALE);
    if (first.empty()) { std::printf("cannot read %s\n", files[0].string().c_str()); return 2; }
    const int w = first.cols, h = first.rows;

    constexpr int kTarget = 200;        // features a VIO frontend wants live
    constexpr int kEdgeThreshold = 17;  // seal_params.yaml
    bincv::GoodFeaturesParams gf;
    gf.maxCorners = kTarget;
    bincv::LKParams lk;
    double lowFrac = 1.0;
    if (const char* lw = std::getenv("BINCV_VIO_LOW")) {
        const double v = std::atof(lw);
        if (v > 0.0 && v <= 1.0) lowFrac = v;
    }
    const int lowWater = static_cast<int>(lowFrac * kTarget);

    Frontend fe(w, h);
    std::vector<Track> tracks;
    std::vector<Point2f> src, dst;
    std::vector<uint8_t> status;
    // Sized from the NMS pool, NOT from maxCorners. Getting this wrong truncates the
    // pool BEFORE the spacing filter runs: the `capacity` STRONGEST survivors are kept
    // and the rest are dropped, so the returned count is a lower bound on what the
    // reference would have produced -- a dropped survivor might have been accepted by
    // the spacing filter after the ranked ones ran out. `candidatesTruncated` is the
    // only way a caller learns of it (the capacity contract on `selectGoodFeatures`),
    // and this program checks it every frame.
    std::vector<bincv::Corner> cand(static_cast<size_t>(w) * static_cast<size_t>(h) / 2);
    int nextId = 0;

    // Statistics a VIO backend would care about.
    std::vector<int> lifetimes;
    size_t frames = 0, detections = 0, diedFlow = 0, diedRange = 0, spawned = 0, truncated = 0;
    size_t maxRanked = 0;
    double sumLive = 0.0, msSensor = 0, msBuild = 0, msTrack = 0, msDetect = 0;

    std::printf("=== A binary-frame VIO vision frontend on binCV kernels ===\n");
    std::printf("    %zu frames, %dx%d, target %d live features, 1/2/2/2 ladder\n",
                files.size(), w, h, kTarget);
    std::printf("    sensor stage (median + edge filter) in OpenCV; everything after in binCV\n");
    std::printf("    detection policy: top up when live < %d (%.0f%% of target)\n\n",
                lowWater, 100.0 * lowFrac);

    for (size_t f = 0; f < files.size(); ++f) {
        const cv::Mat gray = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty() || gray.cols != w || gray.rows != h) continue;

        auto t0 = std::chrono::steady_clock::now();
        const cv::Mat binary = sensorStage(gray, kEdgeThreshold);
        msSensor += msSince(t0);

        t0 = std::chrono::steady_clock::now();
        fe.buildFrom(binary);
        msBuild += msSince(t0);

        if (f == 0) { std::swap(fe.prev, fe.next); ++frames; continue; }

        t0 = std::chrono::steady_clock::now();
        fe.bindAll();
        msBuild += msSince(t0);

        // ---- track the live set ----
        if (!tracks.empty()) {
            src.clear();
            for (const Track& t : tracks) src.push_back(t.p);
            dst.assign(src.size(), Point2f{});
            status.assign(src.size(), 0);
            t0 = std::chrono::steady_clock::now();
            bincv::calcOpticalFlowPyrLK(fe.levels, src.data(), dst.data(), status.data(), nullptr,
                                        src.size(), lk);
            msTrack += msSince(t0);

            size_t out = 0;
            for (size_t i = 0; i < tracks.size(); ++i) {
                if (!status[i]) { ++diedFlow; lifetimes.push_back(tracks[i].age); continue; }
                if (dst[i].x < 0.0f || dst[i].x >= static_cast<float>(w) || dst[i].y < 0.0f ||
                    dst[i].y >= static_cast<float>(h)) {
                    ++diedRange; lifetimes.push_back(tracks[i].age); continue;
                }
                tracks[out] = tracks[i];
                tracks[out].p = dst[i];
                tracks[out].age += 1;
                ++out;
            }
            tracks.resize(out);
        }

        // ---- top up by detection, only when short ----
        // DETECTION POLICY, and it is a choice rather than a property of binCV.
        // Detecting whenever the count dips below target runs the detector on
        // almost every frame; a hysteresis band runs it far less. BINCV_VIO_LOW
        // sets the low-water mark as a fraction of target so the two can be
        // compared -- see the note under COST PER FRAME.
        if (static_cast<int>(tracks.size()) < lowWater) {
            t0 = std::chrono::steady_clock::now();
            bincv::ResponseMap ringView{fe.ring.data(), static_cast<size_t>(w),
                                        bincv::kResponseRingRows, static_cast<size_t>(w)};
            const bincv::CornerResult r = bincv::goodFeaturesToTrackStreaming<W>(
                fe.dx0, fe.dy0, gf, ringView, cand.data(), cand.size());
            msDetect += msSince(t0);
            ++detections;
                    if (r.candidatesTruncated) ++truncated;
            maxRanked = std::max(maxRanked, r.candidatesRanked);

            std::vector<Point2f> fresh;
            fresh.reserve(r.count);
            for (size_t i = 0; i < r.count; ++i) {
                fresh.push_back(Point2f{static_cast<float>(cand[i].x), static_cast<float>(cand[i].y)});
            }
            applyMinDistance(fresh, tracks, static_cast<float>(gf.minDistance));
            for (const Point2f& p : fresh) {
                if (static_cast<int>(tracks.size()) >= kTarget) break;
                tracks.push_back(Track{p, 0, nextId++});
                ++spawned;
            }
        }

        sumLive += static_cast<double>(tracks.size());
        std::swap(fe.prev, fe.next);
        ++frames;
        if (frames % 200 == 0) {
            std::printf("  ... %zu frames, %zu live\n", frames, tracks.size());
        }
    }
    for (const Track& t : tracks) lifetimes.push_back(t.age);   // survivors count too

    std::sort(lifetimes.begin(), lifetimes.end());
    auto pct = [&](double p) {
        return lifetimes.empty() ? 0
                                 : lifetimes[std::min(lifetimes.size() - 1,
                                                      static_cast<size_t>(
                                                          p * static_cast<double>(lifetimes.size())))];
    };
    const double fd = static_cast<double>(frames ? frames - 1 : 1);
    std::printf("\n--- TRACK LIFECYCLE (what a VIO backend is handed) ---\n");
    std::printf("  frames processed      : %zu\n", frames);
    std::printf("  mean live features    : %.1f  (target %d)\n", sumLive / fd, kTarget);
    std::printf("  features spawned      : %zu over %zu detections (%.1f%% of frames)\n",
                spawned, detections, 100.0 * static_cast<double>(detections) / fd);
    std::printf("  track lifetime        : p50 %d  p90 %d  max %d frames\n",
                pct(0.50), pct(0.90), lifetimes.empty() ? 0 : lifetimes.back());
    std::printf("  died: FAILED_FLOW %zu, FLOW_OUT_OF_RANGE %zu  (%.1f%% / %.1f%% of spawns)\n",
                diedFlow, diedRange, 100.0 * static_cast<double>(diedFlow) /
                    static_cast<double>(std::max<size_t>(spawned, 1)),
                100.0 * static_cast<double>(diedRange) /
                    static_cast<double>(std::max<size_t>(spawned, 1)));
    std::printf("  NMS pool peak         : %zu survivors ranked (capacity %zu)%s\n", maxRanked,
                cand.size(), truncated ? "  <-- TRUNCATED, see below" : "");
    if (truncated) {
        std::printf("  *** %zu detections truncated the NMS pool. The pool keeps the STRONGEST\n"
                    "      survivors it can hold, so the loss is the ones dropped before the\n"
                    "      spacing filter ever saw them: the count returned is a LOWER BOUND on\n"
                    "      the reference\'s. Size the pool from candidatesRanked, not from\n"
                    "      maxCorners. ***\n", truncated);
    }

    std::printf("\n--- COST PER FRAME ---\n");
    std::printf("  sensor stage (OpenCV, NOT binCV) %7.3f ms\n", msSensor / fd);
    std::printf("  build  (pyrDown + derivatives)   %7.3f ms\n", msBuild / fd);
    std::printf("  track  (LK)                      %7.3f ms\n", msTrack / fd);
    std::printf("  detect (streaming gftt)          %7.3f ms\n", msDetect / fd);
    std::printf("  binCV total                      %7.3f ms\n",
                (msBuild + msTrack + msDetect) / fd);
    std::printf("  peak binCV working set           %7zu B\n", fe.bytes());
    std::printf("\n  DETECTION RAN ON %.1f%% OF FRAMES, and that is set by the policy\n"
                "  above rather than by binCV. Re-run with BINCV_VIO_LOW=0.6 to see how far\n"
                "  the profile moves.\n\n"
                "  The sensor stage is listed separately because it is NOT binCV's claim:\n"
                "  in SEAL it is dedicated hardware, and binCV's domain starts at the binary\n"
                "  frame. Judge binCV on the three lines above it.\n",
                100.0 * static_cast<double>(detections) / fd);
    return 0;
}
