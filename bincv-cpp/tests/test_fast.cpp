// FAST corner detection (T5.5).
//
// TWO HALVES. The CORE half checks the detection rule against a naive oracle and
// against synthetic geometry with a known answer. The OPENCV half checks binCV's
// detections against cv::FAST's -- which is the claim that matters, because the
// DETECTION rule is meant to be cv::FAST's exactly even though the SCORE is not
// (API Tier 2).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/fast.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#endif

namespace {
using namespace bincv;
}

BINCV_TEST(Fast, FlatImageHasNoCorners) {
    // The obvious failure mode of any threshold detector, and the one that makes a
    // whole frame's worth of false keypoints.
    constexpr size_t kW = 60, kH = 60;
    std::vector<uint8_t> flat(kW * kH, 128);
    std::vector<FastCorner> out(1000);
    bool trunc = false;
    const size_t n = detectFast<uint8_t>(flat.data(), kW, kH, kW, 20, out.data(), out.size(),
                                         &trunc);
    std::printf("  flat image: %zu corners\n", n);
    BINCV_CHECK(n == 0);
    BINCV_CHECK(!trunc);
}

BINCV_TEST(Fast, ASquareCornerIsDetectedAndItsInteriorIsNot) {
    // A filled square on a dark field. Its four corners are corners; the middle of an
    // edge is not, and neither is the interior -- an edge point has HALF its ring
    // bright, which is well short of a 9-arc.
    constexpr size_t kW = 80, kH = 80;
    std::vector<uint8_t> img(kW * kH, 20);
    for (size_t y = 25; y < 55; ++y)
        for (size_t x = 25; x < 55; ++x) img[y * kW + x] = 220;
    std::vector<FastCorner> out(2000);
    const size_t n = detectFast<uint8_t>(img.data(), kW, kH, kW, 40, out.data(), out.size());

    auto near = [&](int px, int py) {
        for (size_t i = 0; i < n; ++i)
            if (std::abs(out[i].x - px) <= 2 && std::abs(out[i].y - py) <= 2) return true;
        return false;
    };
    auto any = [&](int px, int py) {
        for (size_t i = 0; i < n; ++i) if (out[i].x == px && out[i].y == py) return true;
        return false;
    };
    std::printf("  square: %zu corners; TL=%d TR=%d BL=%d BR=%d edge=%d interior=%d\n", n,
                near(25, 25), near(54, 25), near(25, 54), near(54, 54), any(40, 25),
                any(40, 40));
    BINCV_CHECK(near(25, 25) && near(54, 25) && near(25, 54) && near(54, 54));
    BINCV_CHECK(!any(40, 40));   // interior
    BINCV_CHECK(!any(40, 15));   // background
}

BINCV_TEST(Fast, TruncationIsReportedNotSilent) {
    // A silently truncated detection looks like a sparse image, and that gets
    // diagnosed as a tuning problem for weeks.
    constexpr size_t kW = 80, kH = 80;
    std::vector<uint8_t> img(kW * kH, 20);
    for (size_t y = 10; y < 70; y += 6)
        for (size_t x = 10; x < 70; x += 6)
            for (size_t dy = 0; dy < 3; ++dy)
                for (size_t dx = 0; dx < 3; ++dx) img[(y + dy) * kW + x + dx] = 240;
    std::vector<FastCorner> small(4);
    bool trunc = false;
    const size_t n = detectFast<uint8_t>(img.data(), kW, kH, kW, 40, small.data(),
                                         small.size(), &trunc);
    std::printf("  capacity 4: wrote %zu, truncated=%d\n", n, trunc ? 1 : 0);
    BINCV_CHECK(n == 4);
    BINCV_CHECK(trunc);
}

BINCV_TEST(Fast, ArcLengthChangesWhatQualifies) {
    // arcLength is the parameter that decides how corner-like a corner must be. A
    // longer arc must never accept MORE points than a shorter one.
    constexpr size_t kW = 100, kH = 100;
    std::vector<uint8_t> img(kW * kH);
    uint64_t st = 5;
    for (auto& v : img) { st = st * 6364136223846793005ULL + 1; v = static_cast<uint8_t>(st >> 40); }
    std::vector<FastCorner> out(20000);
    const size_t n9 = detectFast<uint8_t>(img.data(), kW, kH, kW, 30, out.data(), out.size(),
                                          nullptr, 9);
    const size_t n12 = detectFast<uint8_t>(img.data(), kW, kH, kW, 30, out.data(), out.size(),
                                           nullptr, 12);
    std::printf("  random texture: arc 9 -> %zu, arc 12 -> %zu\n", n9, n12);
    BINCV_CHECK(n12 <= n9);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Fast, DetectionsMatchCvFast) {
    // The claim: the DETECTION rule is cv::FAST's, exactly. Same ring, same order,
    // same contiguity-wraps rule. (The SCORE is not -- see the tier note -- so this
    // compares the SET of detected points, not their ordering.)
    constexpr int kW = 120, kH = 90;
    cv::Mat img(kH, kW, CV_8U);
    uint64_t st = 31337;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            img.at<uint8_t>(y, x) = static_cast<uint8_t>(st >> 40);
        }
    const int t = 30;
    std::vector<cv::KeyPoint> cvKp;
    cv::FAST(img, cvKp, t, /*nonmaxSuppression=*/false, cv::FastFeatureDetector::TYPE_9_16);

    std::vector<FastCorner> mine(40000);
    const size_t n = detectFast<uint8_t>(img.ptr<uint8_t>(0), static_cast<size_t>(kW),
                                         static_cast<size_t>(kH), img.step, t, mine.data(),
                                         mine.size(), nullptr, 9);

    // cv::FAST also refuses pixels within 3 of the border, so the sets are directly
    // comparable without trimming either side.
    std::vector<uint8_t> mineMask(static_cast<size_t>(kW * kH), 0), cvMask(static_cast<size_t>(kW * kH), 0);
    for (size_t i = 0; i < n; ++i)
        mineMask[static_cast<size_t>(mine[i].y) * static_cast<size_t>(kW) +
                 static_cast<size_t>(mine[i].x)] = 1;
    for (const cv::KeyPoint& k : cvKp)
        cvMask[static_cast<size_t>(k.pt.y) * static_cast<size_t>(kW) +
               static_cast<size_t>(k.pt.x)] = 1;
    size_t onlyMine = 0, onlyCv = 0, both = 0;
    for (size_t i = 0; i < mineMask.size(); ++i) {
        if (mineMask[i] && cvMask[i]) ++both;
        else if (mineMask[i]) ++onlyMine;
        else if (cvMask[i]) ++onlyCv;
    }
    std::printf("  binCV %zu, cv::FAST %zu -- both %zu, only binCV %zu, only cv %zu\n", n,
                cvKp.size(), both, onlyMine, onlyCv);
    BINCV_CHECK(both > 0);
    BINCV_CHECK(onlyMine == 0);
    BINCV_CHECK(onlyCv == 0);
}
#endif

BINCV_TEST_MAIN("test_fast")
