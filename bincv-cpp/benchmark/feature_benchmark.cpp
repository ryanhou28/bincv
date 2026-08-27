// How fast are the descriptor family and FAST (T5.4 / T5.5)?
//
// THREE COMPARISONS, AND ONLY TWO OF THEM ARE FAIR. Saying which is which is the
// point:
//
//   FAST vs cv::FAST            FAIR. Same algorithm, same ring, same detections --
//                               tests/test_fast.cpp proves the SETS are identical.
//
//   matching vs cv::BFMatcher   FAIR, and the one binCV should win. Both are brute
//                               force over Hamming distance; OpenCV stores these as
//                               bits too, so for once nobody is paying a
//                               representation penalty.
//
//   BRIEF vs cv::ORB::compute   NOT FAIR, and reported anyway with the reason.
//                               cv::ORB additionally computes an intensity-centroid
//                               orientation and rotates its sampling pattern per
//                               keypoint. It is doing strictly more work. The number
//                               is here because "how fast is our descriptor" deserves
//                               SOME answer, not because it settles anything.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/descriptor.hpp"
#include "bincv-cpp/ops/fast.hpp"
#include "measure_util.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;

// A SMOOTHED frame, not white noise. Raw noise at threshold 40 makes 14% of pixels
// FAST corners; a real 752x480 frame makes about 1%. Benchmarking the corner-dense
// case measures the arc scan and almost nothing else, and it flatters whichever
// implementation happens to reject late.
cv::Mat makeFrame(int w, int h, int seedShift = 0) {
    cv::Mat m(h, w, CV_8U);
    uint64_t st = 0xFEEDFACEULL + static_cast<uint64_t>(static_cast<unsigned>(seedShift));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            m.at<uint8_t>(y, x) = static_cast<uint8_t>(st >> 40);
        }
    cv::Mat blurred;
    cv::GaussianBlur(m, blurred, cv::Size(5, 5), 1.1);
    return blurred;
}

} // namespace

int main() {
    const int w = 752, h = 480;   // the reference frame size
    const cv::Mat img = makeFrame(w, h);
    const int fastThreshold = 40;
    cv::setNumThreads(1);   // X-64: binCV is single-threaded here, so OpenCV is too

    std::printf("=== T5.4 / T5.5: descriptors, matching and FAST ===\n");
    std::printf("  %dx%d, FAST threshold %d, %zu-bit descriptors, OpenCV at 1 thread\n\n",
                w, h, fastThreshold, kBits);

    // ---- FAST ----
    std::vector<bincv::FastCorner> corners(200000);
    size_t nCorners = bincv::detectFast<uint8_t>(img.ptr<uint8_t>(0), static_cast<size_t>(w),
                                                 static_cast<size_t>(h), img.step,
                                                 fastThreshold, corners.data(), corners.size());
    std::vector<cv::KeyPoint> cvKp;
    cv::FAST(img, cvKp, fastThreshold, false, cv::FastFeatureDetector::TYPE_9_16);
    std::printf("  detections: binCV %zu, cv::FAST %zu  (%.1f%% of pixels)\n", nCorners,
                cvKp.size(), 100.0 * static_cast<double>(nCorners) /
                                 static_cast<double>(w) / static_cast<double>(h));
    if (nCorners < 32) {
        // A benchmark with nothing to describe or match would index past the end and
        // report a meaningless time. Say so rather than segfaulting.
        std::printf("  too few corners to describe or match -- lower the threshold.\n");
        return 1;
    }

    // ---- descriptors on a bounded keypoint set ----
    const size_t kKp = nCorners < 1000 ? nCorners : size_t{1000};
    std::vector<float> kpXY(kKp * 2);
    std::vector<cv::KeyPoint> cvSub;
    for (size_t i = 0; i < kKp; ++i) {
        kpXY[2 * i] = static_cast<float>(corners[i].x);
        kpXY[2 * i + 1] = static_cast<float>(corners[i].y);
        cvSub.emplace_back(static_cast<float>(corners[i].x), static_cast<float>(corners[i].y),
                           31.0f);
    }
    bincv::BriefPattern<kBits> pat;
    bincv::makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> desc(kKp * kWords), desc2(kKp * kWords);
    std::vector<bincv::DescriptorMatch> matches(kKp);
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Mat cvDesc;

    // A second descriptor set to match against: the same points on a shifted frame.
    cv::Mat img2 = makeFrame(w, h, 991);
    bincv::computeBrief<kBits, uint8_t, uint32_t>(img.ptr<uint8_t>(0), static_cast<size_t>(w),
                                                  static_cast<size_t>(h), img.step,
                                                  kpXY.data(), kKp, pat, desc.data());
    bincv::computeBrief<kBits, uint8_t, uint32_t>(img2.ptr<uint8_t>(0), static_cast<size_t>(w),
                                                  static_cast<size_t>(h), img2.step,
                                                  kpXY.data(), kKp, pat, desc2.data());
    cv::Mat q(static_cast<int>(kKp), static_cast<int>(kBits / 8), CV_8U, desc.data());
    cv::Mat t(static_cast<int>(kKp), static_cast<int>(kBits / 8), CV_8U, desc2.data());
    cv::BFMatcher bf(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;

    std::vector<measure::Bench> bs = {
        {"FAST          binCV", [&](int) {
             measure::g_sink += bincv::detectFast<uint8_t>(
                 img.ptr<uint8_t>(0), static_cast<size_t>(w), static_cast<size_t>(h),
                 img.step, fastThreshold, corners.data(), corners.size());
         }},
        {"FAST          cv::FAST", [&](int) {
             std::vector<cv::KeyPoint> k;
             cv::FAST(img, k, fastThreshold, false, cv::FastFeatureDetector::TYPE_9_16);
             measure::g_sink += k.size();
         }},
        {"describe      binCV BRIEF", [&](int) {
             bincv::computeBrief<kBits, uint8_t, uint32_t>(
                 img.ptr<uint8_t>(0), static_cast<size_t>(w), static_cast<size_t>(h),
                 img.step, kpXY.data(), kKp, pat, desc.data());
             measure::g_sink += desc[0];
         }},
        {"describe      cv::ORB (does more)", [&](int) {
             std::vector<cv::KeyPoint> k = cvSub;
             orb->compute(img, k, cvDesc);
             measure::g_sink += static_cast<size_t>(cvDesc.rows);
         }},
        {"match kNN=2   binCV", [&](int) {
             bincv::matchDescriptors<uint32_t>(desc.data(), kKp, desc2.data(), kKp, kWords,
                                               matches.data(), 80);
             measure::g_sink += matches[0].distance;
         }},
        {"match kNN=2   cv::BFMatcher", [&](int) {
             knn.clear();
             bf.knnMatch(q, t, knn, 2);
             measure::g_sink += knn.size();
         }},
    };
    const auto tt = measure::measureInterleaved(bs, 7, 60.0);
    std::printf("\n  %-36s %14s %12s\n", "arm", "ns/frame", "vs OpenCV");
    for (size_t i = 0; i < bs.size(); ++i) {
        const char* rel = "";
        char buf[32] = {0};
        if (i % 2 == 0 && i + 1 < bs.size()) {
            std::snprintf(buf, sizeof(buf), "%.2fx", tt[i + 1].medianNs / tt[i].medianNs);
            rel = buf;
        }
        std::printf("  %-36s %14.1f %12s\n", bs[i].name.c_str(), tt[i].medianNs, rel);
    }
    std::printf("\n  %zu keypoints described and matched.\n", kKp);
    std::printf("  cv::ORB::compute ALSO computes orientation and rotates its pattern per\n"
                "  keypoint, so that row is not a like-for-like comparison and is not a\n"
                "  claim. FAST and matching are.\n");
    std::printf("\n  sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
