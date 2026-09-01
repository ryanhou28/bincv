// FAST corner detection.
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

#include "bincv-cpp/binMat.hpp"
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
    std::printf(" flat image: %zu corners\n", n);
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
    std::printf(" square: %zu corners; TL=%d TR=%d BL=%d BR=%d edge=%d interior=%d\n", n,
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
    std::printf(" capacity 4: wrote %zu, truncated=%d\n", n, trunc ? 1 : 0);
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
    std::printf(" random texture: arc 9 -> %zu, arc 12 -> %zu\n", n9, n12);
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
            st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
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
    std::printf(" binCV %zu, cv::FAST %zu -- both %zu, only binCV %zu, only cv %zu\n", n,
                cvKp.size(), both, onlyMine, onlyCv);
    BINCV_CHECK(both > 0);
    BINCV_CHECK(onlyMine == 0);
    BINCV_CHECK(onlyCv == 0);
}
#endif

// ---------------------------------------------------------------------------
// earlier work: THE BIT-PLANE OVERLOAD, WHICH IS FAST ON binCV'S OWN TYPE.
//
// THE EQUIVALENCE THIS RESTS ON. For binary content stored as CV_8U in {0, 255},
// `cv::FAST` at ANY threshold in [1, 254] accepts exactly the corners the bit-plane
// form accepts: `255 > 0 + t` holds for every such `t`, and `0 < 255 - t` likewise, so
// a brighter arc needs a zero center and a darker arc a set one. That is why the
// bit-plane form takes no threshold at all -- there is only one -- and why the
// comparison below is CORNER FOR CORNER AND IN ORDER rather than set-against-set.
//
// THE SIZES ARE CHOSEN TO EXERCISE BOTH PATHS. The AVX2 kernel handles 256 pixels at a
// time and declines the first and last row of the sweep (a bit-plane row has no stride
// padding, so its one-byte-either-side reads would leave the allocation). A width
// under 256 leaves leftover words for the scalar path; 100 px is ENTIRELY scalar.
// ---------------------------------------------------------------------------
#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Fast, BitPlaneMatchesCvFastExactly) {
    struct Case { int w, h; const char* what; };
    const Case cases[] = {
        {100, 60, "all scalar -- narrower than one vector chunk"},
        {320, 90, "one vector chunk plus scalar leftovers"},
        {752, 120, "the reference frame's width"},
        {256, 40, "exactly one chunk, no leftovers"},
    };
    for (const Case& c : cases) {
        cv::Mat img(c.h, c.w, CV_8U);
        uint64_t st = UINT64_C(0xC0FFEE) + static_cast<uint64_t>(c.w);
        for (int y = 0; y < c.h; ++y) {
            for (int x = 0; x < c.w; ++x) {
                st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
                // Binary content, and DENSE enough to make real arcs: a coin flip
                // gives almost no 9-runs, so this biases toward runs of set pixels.
                img.at<uint8_t>(y, x) = ((st >> 41) % 100u) < 62u ? 255 : 0;
            }
        }
        bincv::BinMat<uint32_t> plane(c.w, c.h);
        plane.fromCVMat(img);

        std::vector<cv::KeyPoint> cvKp;
        cv::FAST(img, cvKp, 100, /*nonmaxSuppression=*/false,
                 cv::FastFeatureDetector::TYPE_9_16);
        std::vector<FastCorner> mine(200000);
        bool truncated = false;
        const size_t n = detectFast(plane.constView(), mine.data(), mine.size(), &truncated);

        size_t mismatched = 0;
        for (size_t i = 0; i < (n > cvKp.size() ? n : cvKp.size()); ++i) {
            if (i >= n || i >= cvKp.size()) { ++mismatched; continue; }
            if (mine[i].x != static_cast<int>(cvKp[i].pt.x) ||
                mine[i].y != static_cast<int>(cvKp[i].pt.y)) {
                ++mismatched;
            }
        }
        // The score is binCV's own -- the longest qualifying arc, 9..16 -- because
        // OpenCV's is the same number for every corner on binary content and orders
        // nothing. It must still be in range and consistent with detection.
        size_t badScore = 0;
        for (size_t i = 0; i < n; ++i) {
            if (mine[i].score < 9 || mine[i].score > 16) ++badScore;
        }
        std::printf(" %4dx%-4d %-46s binCV %6zu cv %6zu mismatched %zu\n", c.w, c.h,
                    c.what, n, cvKp.size(), mismatched);
        BINCV_CHECK(n > 0);
        BINCV_CHECK_EQ(n, cvKp.size());
        BINCV_CHECK_EQ(mismatched, size_t{0});
        BINCV_CHECK_EQ(badScore, size_t{0});
        BINCV_CHECK(!truncated);
    }
}

BINCV_TEST(Fast, BitPlaneScoringArmsAgree) {
    // the score can be computed two ways -- by transposing each corner's ring, or
    // by counting how many of the nested arc-length masks hold that pixel's bit. They
    // are the same number by construction, and the shipped path CHOOSES BETWEEN THEM
    // PER CHUNK on a measured density crossover. That choice must not be observable in
    // the output, which is exactly what this checks.
    constexpr int kW = 400, kH = 120;
    cv::Mat img(kH, kW, CV_8U);
    uint64_t st = UINT64_C(4242);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            img.at<uint8_t>(y, x) = ((st >> 41) % 100u) < 66u ? 255 : 0;
        }
    }
    bincv::BinMat<uint32_t> plane(kW, kH);
    plane.fromCVMat(img);

    std::vector<FastCorner> transposed(200000), masked(200000), adaptive(200000);
    const int saved = bincv::impl::fastScoreMaskThreshold();
    bincv::impl::fastScoreMaskThreshold() = 1 << 30;   // never use the masks
    const size_t nT = detectFast(plane.constView(), transposed.data(), transposed.size());
    bincv::impl::fastScoreMaskThreshold() = 0;         // always use the masks
    const size_t nM = detectFast(plane.constView(), masked.data(), masked.size());
    bincv::impl::fastScoreMaskThreshold() = saved;     // the shipped crossover
    const size_t nA = detectFast(plane.constView(), adaptive.data(), adaptive.size());

    size_t diffM = 0, diffA = 0, scoreRange = 0;
    for (size_t i = 0; i < nT; ++i) {
        if (i < nM && (transposed[i].x != masked[i].x || transposed[i].y != masked[i].y ||
                       transposed[i].score != masked[i].score)) {
            ++diffM;
        }
        if (i < nA && (transposed[i].x != adaptive[i].x || transposed[i].y != adaptive[i].y ||
                       transposed[i].score != adaptive[i].score)) {
            ++diffA;
        }
        if (transposed[i].score < 9 || transposed[i].score > 16) ++scoreRange;
    }
    std::printf(" scoring arms: %zu corners; transpose vs masks %zu differ, "
                "vs adaptive %zu differ\n", nT, diffM, diffA);
    BINCV_CHECK(nT > 500);
    BINCV_CHECK_EQ(nM, nT);
    BINCV_CHECK_EQ(nA, nT);
    BINCV_CHECK_EQ(diffM, size_t{0});
    BINCV_CHECK_EQ(diffA, size_t{0});
    BINCV_CHECK_EQ(scoreRange, size_t{0});
}

BINCV_TEST(Fast, BitPlaneThresholdIsUniqueOnOneBitContent) {
    // The claim the missing threshold parameter rests on, checked rather than argued:
    // on {0, 255} content every legal `cv::FAST` threshold gives the SAME corners, so
    // there is exactly one detector to expose and no knob to offer.
    constexpr int kW = 200, kH = 80;
    cv::Mat img(kH, kW, CV_8U);
    uint64_t st = UINT64_C(991);
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            img.at<uint8_t>(y, x) = ((st >> 41) % 100u) < 62u ? 255 : 0;
        }
    }
    size_t base = 0, differing = 0;
    for (int t : {1, 2, 37, 128, 200, 254}) {
        std::vector<cv::KeyPoint> kp;
        cv::FAST(img, kp, t, false, cv::FastFeatureDetector::TYPE_9_16);
        if (t == 1) base = kp.size();
        else if (kp.size() != base) ++differing;
    }
    std::printf(" cv::FAST on {0,255} content: %zu corners at every threshold tried, "
                "%zu thresholds disagreed\n", base, differing);
    BINCV_CHECK(base > 0);
    BINCV_CHECK_EQ(differing, size_t{0});
}
#endif

BINCV_TEST_MAIN("test_fast")
