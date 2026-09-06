// ===========================================================================
// A BINARY-FRAME SLAM VISION FRONTEND, END TO END, ON binCV KERNELS.
//
// vio_frontend.cpp answers "can binCV's kernels carry a VIO tracking loop"; this
// file asks the same of the SLAM-shaped loop, which is structurally different:
// association is by DESCRIPTOR, not by optical flow, because a SLAM system needs
// to recognise a feature it has not tracked continuously -- relocalisation, loop
// closure and map-point association all run on descriptors. The loop here is the
// ORB-SLAM family's tracking front half, on binCV's representation:
//
//   1. SENSOR STAGE, in OpenCV, exactly as vio_frontend.cpp runs it: the caller's
//      half of the input boundary. binCV's domain starts at the binary frame.
//   2. A BINARY PYRAMID (1 bit at every level -- there is no tracker here wanting
//      deeper levels), built with the Box2x2 kernel.
//   3. FAST per level on the BIT-PLANE frame, budgeted per level and spaced with
//      spaceCandidates -- the mechanism is binCV's, the budget split is a POLICY
//      and marked as such.
//   4. ORIENTATION from the intensity centroid, on the SAME wide frame the
//      descriptor samples, at level-0 coordinates -- angle and patch must
//      measure the same support, or the steering answers a different image.
//   5. STEERED BRIEF descriptors on the incoming wide frame at level-0
//      coordinates, angles from step 4.
//   6. MATCHING against the previous frame's descriptors (ratio test), then the
//      five-point essential matrix under RANSAC over the matches.
//
// The headline is the RANSAC INLIER RATE: it is what a SLAM system does with the
// matches, and it prices the whole chain -- a weak detector, a drifting angle or
// a bad descriptor all land in that one number.
//
// TWO CHOICES A READER SHOULD SEE ARGUED, NOT ASSUMED:
//
// * DESCRIPTORS READ THE WIDE FRAME AT BASE SCALE, whatever level the keypoint
//   came from. Scale-adapted patches need either a wide-image pyramid resident
//   (the memory binCV exists not to spend: an 8-bit ladder of the reference
//   frame is ~480 KB against the binary pyramid's ~60 KB) or pattern offsets
//   wider than int8 once a x8 pattern is rotated. Frame-to-frame matching at
//   video rate sees scale changes far under one pyramid octave, so base-scale
//   description holds up here; matching across LARGE scale gaps (relocalisation
//   against an old map) is where this choice would start to cost, and that is a
//   measurement for the day that caller exists.
// * THE WIDE FRAME ITSELF IS THE INCOMING ONE, held only for the current
//   iteration -- in a deployed system it is the sensor's own transient buffer.
//   Nothing wide persists across frames; what persists is descriptors
//   (32 B/keypoint) and keypoint records.
//
// Usage: slam_frontend <frame-dir> [max-frames]
//   BINCV_SLAM_FX/FY/CX/CY override the intrinsics (defaults: EuRoC cam0).
// ===========================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bincv/core/simd.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/essential.hpp"
#include "bincv/ops/fast.hpp"
#include "bincv/ops/occupancy.hpp"
#include "bincv/ops/orientation.hpp"
#include "bincv/ops/pyramid.hpp"
#include "bincv/quantMat.hpp"

namespace fs = std::filesystem;
using W = uint32_t;
using bincv::Point2f;

namespace {

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;
constexpr size_t kLevels = 4;

// The per-level feature budget. A POLICY, not a property of binCV: this split is
// roughly area-proportional with the tail rounded up, the shape the ORB-SLAM
// family uses so coarse levels contribute structure without flooding the set.
constexpr size_t kBudget[kLevels] = {300, 120, 60, 20};
constexpr float kMinSpacing = 12.0f;   // spacing radius at level 0, halved per level

// ---- 1. The sensor stage. OpenCV, and deliberately not binCV -- the same
// spelling as vio_frontend.cpp, which is the file to read on why.
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

/// The keypoint record the descriptor and geometry stages consume: position at
/// LEVEL 0 (the exact Box2x2 center map, so no per-level bias accumulates), the
/// octave it was detected at, and its orientation. The FAST arc score orders the
/// per-level candidates and is spent there -- nothing downstream reads it, so it
/// is deliberately not carried.
struct SlamKeypoint {
    float x, y;
    int octave;
    float angle;
};

/// One frame's feature set: records plus packed descriptors, the only state that
/// persists across frames.
struct FrameFeatures {
    std::vector<SlamKeypoint> kp;
    std::vector<float> xy;        // interleaved, for the kernels
    std::vector<uint32_t> desc;   // kWords per keypoint
    size_t count = 0;

    size_t bytes() const {
        return kp.capacity() * sizeof(SlamKeypoint) + xy.capacity() * sizeof(float) +
               desc.capacity() * sizeof(uint32_t);
    }
};

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

float envF(const char* name, float dflt) {
    const char* v = std::getenv(name);
    return v ? static_cast<float>(std::atof(v)) : dflt;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: slam_frontend <frame-dir> [max-frames]\n");
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

    // Intrinsics for the essential-matrix stage. EuRoC cam0 by default; the
    // geometry is a CONSUMER of the matches, so a wrong K degrades the inlier
    // headline rather than the frontend itself.
    const float fx = envF("BINCV_SLAM_FX", 458.654f);
    const float fy = envF("BINCV_SLAM_FY", 457.296f);
    const float cx = envF("BINCV_SLAM_CX", 367.215f);
    const float cy = envF("BINCV_SLAM_CY", 248.375f);

    constexpr int kEdgeThreshold = 17;   // the reference frontend's edge_threshold

    std::printf("%s\n", bincv::simdStatusString());

    // The binary pyramid: 1 bit at every level. ~60 KB at 752x480 against the
    // ~480 KB an 8-bit ladder would hold resident.
    bincv::Pyramid<W, 1, 1, 1, 1> pyr(w, h);

    // The steered pattern set: built once, 30 720 B, the price of rotation
    // invariance with no per-keypoint trigonometry.
    bincv::BriefPattern<kBits> base;
    bincv::makeBriefPattern<kBits>(base);
    bincv::SteeredBriefPattern<kBits> steered;
    bincv::makeSteeredBriefPattern<kBits>(steered, base);

    FrameFeatures cur, prev;
    std::vector<bincv::FastCorner> corners(static_cast<size_t>(w) *
                                           static_cast<size_t>(h) / 8);
    std::vector<Point2f> cand;
    std::vector<float> angles;
    std::vector<bincv::DescriptorMatch> matches;
    std::vector<Point2f> fromN, toN;   // matched pairs, normalized coordinates
    std::vector<uint8_t> keep;

    size_t frames = 0, truncated = 0;
    size_t sumKp = 0, sumQueried = 0, sumAccepted = 0, sumInliers = 0, geomFrames = 0;
    size_t perLevelKept[kLevels] = {0, 0, 0, 0};
    double perLevelMs[kLevels] = {0, 0, 0, 0};
    double msSensor = 0, msBuild = 0, msDetect = 0, msOrient = 0, msDescribe = 0,
           msMatch = 0, msGeom = 0;

    std::printf("=== A binary-frame SLAM vision frontend on binCV kernels ===\n");
    std::printf(" %zu frames, %dx%d, budgets/level {%zu, %zu, %zu, %zu}, %zu-bit"
                " steered descriptors\n",
                files.size(), w, h, kBudget[0], kBudget[1], kBudget[2], kBudget[3], kBits);
    std::printf(" sensor stage in OpenCV; detection, orientation, description,\n"
                " matching and the essential matrix in binCV\n\n");

    for (size_t f = 0; f < files.size(); ++f) {
        const cv::Mat gray = cv::imread(files[f].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty() || gray.cols != w || gray.rows != h) continue;

        auto t0 = std::chrono::steady_clock::now();
        const cv::Mat binary = sensorStage(gray, kEdgeThreshold);
        msSensor += msSince(t0);

        t0 = std::chrono::steady_clock::now();
        pyr.level<0>().fromCVMat(binary);
        pyr.build<bincv::PyrDownFilter::Box2x2, bincv::PyrDownBorder::Replicate>();
        msBuild += msSince(t0);

        // ---- detect per level, budget, space, record the octave ----
        cur.kp.clear();
        const auto detectLevel = [&](const auto& level, size_t li) {
            const auto tL = std::chrono::steady_clock::now();
            bool trunc = false;
            const size_t n =
                bincv::detectFast<W>(level.constView(), corners.data(), corners.size(),
                                     &trunc, 9);
            if (trunc) ++truncated;
            // Strongest first, then the greedy spacing filter -- the same order
            // vio_frontend.cpp argues for. Ties break on raster position so the
            // result is deterministic. The arc score is spent here: it decides
            // which candidates the spacing filter sees first, and nothing after
            // this lambda reads it.
            std::sort(corners.begin(), corners.begin() + static_cast<ptrdiff_t>(n),
                      [](const bincv::FastCorner& a, const bincv::FastCorner& b) {
                          if (a.score != b.score) return a.score > b.score;
                          if (a.y != b.y) return a.y < b.y;
                          return a.x < b.x;
                      });
            cand.clear();
            for (size_t i = 0; i < n; ++i) {
                cand.push_back(Point2f{static_cast<float>(corners[i].x),
                                       static_cast<float>(corners[i].y)});
            }
            const float spacing = kMinSpacing / static_cast<float>(1u << li);
            const size_t kept = bincv::spaceCandidates(cand.data(), cand.size(), nullptr,
                                                       0, spacing, kBudget[li]);
            for (size_t i = 0; i < kept; ++i) {
                SlamKeypoint kp;
                kp.x = bincv::pyrLevelToBase(cand[i].x, li);
                kp.y = bincv::pyrLevelToBase(cand[i].y, li);
                kp.octave = static_cast<int>(li);
                kp.angle = 0.0f;
                cur.kp.push_back(kp);
            }
            perLevelKept[li] += kept;
            perLevelMs[li] += msSince(tL);
        };
        t0 = std::chrono::steady_clock::now();
        detectLevel(pyr.level<0>(), 0);
        detectLevel(pyr.level<1>(), 1);
        detectLevel(pyr.level<2>(), 2);
        detectLevel(pyr.level<3>(), 3);
        msDetect += msSince(t0);

        // ---- orient, then describe, both on the incoming wide frame ----
        // The angle and the patch it steers must measure the same support: both
        // read the wide frame at level-0 coordinates, radius 15 pairing with the
        // 31-pixel patch. (The bit-plane orientation spelling serves a pipeline
        // that detects AND describes on bits; feeding a wide-frame descriptor an
        // angle measured over a 2^octave-wider binary disc would steer a patch by
        // a different image's structure.) Coordinates are ROUNDED to the nearest
        // pixel for the sampling kernels -- the exact center map lands coarse
        // keypoints on half-pixels, and truncation would bias every one of their
        // patches half a pixel toward the origin. The record keeps the exact
        // coordinates; the geometry stage wants those.
        cur.count = cur.kp.size();
        cur.xy.clear();
        angles.assign(cur.count, 0.0f);
        for (size_t i = 0; i < cur.count; ++i) {
            cur.xy.push_back(std::floor(cur.kp[i].x + 0.5f));
            cur.xy.push_back(std::floor(cur.kp[i].y + 0.5f));
        }
        t0 = std::chrono::steady_clock::now();
        bincv::keypointOrientation<uint8_t>(gray.ptr<uint8_t>(0), static_cast<size_t>(w),
                                            static_cast<size_t>(h), gray.step,
                                            cur.xy.data(), cur.count, angles.data());
        for (size_t i = 0; i < cur.count; ++i) cur.kp[i].angle = angles[i];
        msOrient += msSince(t0);

        t0 = std::chrono::steady_clock::now();
        cur.desc.resize(cur.count * kWords);   // every keypoint's words are written
        keep.assign(cur.count, 0);
        bincv::computeBriefSteered<kBits, uint8_t, uint32_t>(
            gray.ptr<uint8_t>(0), static_cast<size_t>(w), static_cast<size_t>(h),
            gray.step, cur.xy.data(), cur.count, angles.data(), steered, cur.desc.data(),
            keep.data());
        // Keypoints whose patch left the frame have no descriptor; drop them so the
        // matcher never sees a zeroed 256-bit string pretending to be data. The
        // steered reach (~21 px for a square-sampled base pattern) covers the
        // orientation disc's 15, so no surviving keypoint carries a border-zeroed
        // angle either.
        size_t out = 0;
        for (size_t i = 0; i < cur.count; ++i) {
            if (!keep[i]) continue;
            cur.kp[out] = cur.kp[i];
            cur.xy[2 * out] = cur.xy[2 * i];
            cur.xy[2 * out + 1] = cur.xy[2 * i + 1];
            for (size_t j = 0; j < kWords; ++j)
                cur.desc[out * kWords + j] = cur.desc[i * kWords + j];
            ++out;
        }
        cur.count = out;
        cur.kp.resize(out);
        cur.xy.resize(2 * out);
        msDescribe += msSince(t0);
        sumKp += cur.count;

        // ---- match against the previous frame, then the essential matrix ----
        if (prev.count > 0 && cur.count > 0) {
            t0 = std::chrono::steady_clock::now();
            matches.assign(cur.count, bincv::DescriptorMatch{});
            bincv::matchDescriptors<uint32_t>(cur.desc.data(), cur.count, prev.desc.data(),
                                              prev.count, kWords, matches.data(), 80);
            msMatch += msSince(t0);

            t0 = std::chrono::steady_clock::now();
            fromN.clear();
            toN.clear();
            for (size_t i = 0; i < cur.count; ++i) {
                if (!matches[i].valid) continue;
                const SlamKeypoint& a = prev.kp[matches[i].trainIndex];
                const SlamKeypoint& b = cur.kp[i];
                fromN.push_back(Point2f{(a.x - cx) / fx, (a.y - cy) / fy});
                toN.push_back(Point2f{(b.x - cx) / fx, (b.y - cy) / fy});
            }
            // fromN holds exactly the ratio-test survivors; the denominator the
            // summary prints is the QUERY count, and it is labeled as such.
            sumQueried += cur.count;
            sumAccepted += fromN.size();
            if (fromN.size() >= 8) {
                bincv::RansacParams rp;
                rp.threshold = 1.5 / static_cast<double>(fx);   // ~1.5 px, in normalized units
                bincv::EssentialMatrix E;
                const bincv::RansacResult rr = bincv::findEssentialMat(
                    fromN.data(), toN.data(), fromN.size(), rp, &E);
                if (rr.found) {
                    sumInliers += rr.inliers;
                    ++geomFrames;
                }
            }
            msGeom += msSince(t0);
        }

        std::swap(prev, cur);
        ++frames;
        if (frames % 200 == 0) {
            std::printf(" ... %zu frames, %zu keypoints, %zu matches survived RANSAC\n",
                        frames, prev.count, fromN.size());
        }
    }

    const double fd = static_cast<double>(frames ? frames : 1);
    const double fg = static_cast<double>(geomFrames ? geomFrames : 1);
    std::printf("\n--- WHAT THE BACKEND IS HANDED ---\n");
    std::printf(" frames processed        : %zu\n", frames);
    std::printf(" keypoints/frame         : %.1f (budgeted %zu)\n",
                static_cast<double>(sumKp) / fd,
                kBudget[0] + kBudget[1] + kBudget[2] + kBudget[3]);
    for (size_t li = 0; li < kLevels; ++li) {
        std::printf("   level %zu               : %.1f kept/frame (budget %zu)  %.3f ms\n",
                    li, static_cast<double>(perLevelKept[li]) / fd, kBudget[li],
                    perLevelMs[li] / fd);
    }
    std::printf(" ratio-test accepted     : %.1f of %.1f queried/frame (%.1f%%)\n",
                static_cast<double>(sumAccepted) / fd, static_cast<double>(sumQueried) / fd,
                sumQueried ? 100.0 * static_cast<double>(sumAccepted) /
                                 static_cast<double>(sumQueried)
                           : 0.0);
    std::printf(" RANSAC inliers          : %.1f/frame = %.1f%% of accepted -- THE HEADLINE\n",
                static_cast<double>(sumInliers) / fg,
                sumAccepted ? 100.0 * static_cast<double>(sumInliers) /
                                  static_cast<double>(sumAccepted)
                            : 0.0);
    if (truncated) {
        std::printf(" *** %zu detections truncated the corner buffer -- counts above are\n"
                    " lower bounds; size the buffer up. ***\n", truncated);
    }

    std::printf("\n--- COST PER FRAME ---\n");
    std::printf(" sensor stage (OpenCV, NOT binCV) %7.3f ms\n", msSensor / fd);
    std::printf(" build (pack + binary pyramid)    %7.3f ms\n", msBuild / fd);
    std::printf(" detect (FAST x%zu + space)        %7.3f ms\n", kLevels, msDetect / fd);
    std::printf(" orient (intensity centroid)      %7.3f ms\n", msOrient / fd);
    std::printf(" describe (steered BRIEF)         %7.3f ms\n", msDescribe / fd);
    std::printf(" match (Hamming, ratio test)      %7.3f ms\n", msMatch / fd);
    std::printf(" geometry (5-pt RANSAC)           %7.3f ms\n", msGeom / fd);
    std::printf(" binCV total                      %7.3f ms\n",
                (msBuild + msDetect + msOrient + msDescribe + msMatch + msGeom) / fd);

    std::printf("\n--- MEMORY (persistent across frames) ---\n");
    std::printf(" binary pyramid                   %7zu B\n", pyr.sizeInBytes());
    std::printf(" steered pattern set              %7zu B\n", sizeof(steered));
    std::printf(" feature sets (cur + prev)        %7zu B\n", cur.bytes() + prev.bytes());
    std::printf("\n Unlike the VIO loop, detection runs EVERY frame -- descriptor\n"
                " association has no duty cycle to hide behind, which is exactly why\n"
                " this caller exists: it prices FAST, orientation, description and\n"
                " matching at the rate a SLAM frontend actually pays them.\n");
    return 0;
}
