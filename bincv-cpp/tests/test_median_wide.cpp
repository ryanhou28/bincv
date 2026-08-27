// The reference pipeline's median filter on a WIDE image (T5.10).
//
// ops/denoise.hpp already implements this neighbourhood for BINARY input, where the
// median collapses to maj3. But the reference filters the GRAYSCALE image before
// binarisation -- SEALProcessor.cpp runs three_pix_median_filter(img) and only then
// rl_fast_edge_filter_wide -- so a binary-only median cannot sit where the reference
// puts it. This is the wide-input form and the caller-chosen neighbourhood.
//
// THE OPENCV HALF IS THE ONE THAT MATTERS: it checks medianWide against the
// reference's OWN spelling, ported call for call --
//     Median = max(min(p1,p2), min(max(p1,p2), p3))
// over zero-filled shifted matrices -- so "the default pattern is the reference's" is
// a checked claim rather than a comment. The zero-fill border is the reference's too,
// and it comes from cv::Mat::zeros: the row and column that fall off the edge KEEP
// the zeros rather than being replicated or reflected.
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/medianWide.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

namespace {

using namespace bincv;

template <size_t K, typename SrcT>
void checkAgainstOracle(const char* label, const MedianPattern<K>& pat) {
    constexpr size_t kW = 37, kH = 19;
    std::vector<SrcT> in(kW * kH), out(kW * kH, SrcT{0});
    uint64_t st = 0xD00DULL;
    for (auto& v : in) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<SrcT>(st >> 40);
    }
    medianWide<K, SrcT>(in.data(), kW, kH, kW, out.data(), kW, pat);

    size_t wrong = 0;
    for (size_t y = 0; y < kH; ++y) {
        for (size_t x = 0; x < kW; ++x) {
            std::vector<SrcT> v;
            for (size_t k = 0; k < K; ++k) {
                const long long sy = static_cast<long long>(y) + pat.offset[k].dy;
                const long long sx = static_cast<long long>(x) + pat.offset[k].dx;
                const bool inside = sy >= 0 && sx >= 0 &&
                                    sy < static_cast<long long>(kH) &&
                                    sx < static_cast<long long>(kW);
                v.push_back(inside ? in[static_cast<size_t>(sy) * kW +
                                        static_cast<size_t>(sx)]
                                   : SrcT{0});
            }
            std::sort(v.begin(), v.end());
            if (out[y * kW + x] != v[K / 2]) ++wrong;
        }
    }
    std::printf("  %-34s %4zu pixels differ from the oracle\n", label, wrong);
    BINCV_CHECK(wrong == 0);
}

} // namespace

BINCV_TEST(MedianWide, PatternsAgainstOracle) {
    checkAgainstOracle<3, uint8_t>("reference L      u8", kMedianReferenceL);
    checkAgainstOracle<5, uint8_t>("reference plus   u8", kMedianReferencePlus);
    checkAgainstOracle<3, uint16_t>("reference L      u16", kMedianReferenceL);
    checkAgainstOracle<5, uint16_t>("reference plus   u16", kMedianReferencePlus);
    // An arbitrary neighbourhood is a template argument, not a fork -- that is the
    // whole point of the pattern parameter, so one that is neither shipped constant
    // is exercised here.
    constexpr MedianPattern<3> diagonal{{{-1, -1}, {0, 0}, {1, 1}}};
    checkAgainstOracle<3, uint8_t>("caller's diagonal u8", diagonal);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(MedianWide, ReferenceLMatchesTheReferenceSpelling) {
    // three_pix_median_filter, ported call for call from
    // SEAL/src/temporal_processing/denoise.cpp.
    constexpr int kW = 53, kH = 23;
    cv::Mat img(kH, kW, CV_8U);
    uint64_t st = 4242;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            img.at<uint8_t>(y, x) = static_cast<uint8_t>(st >> 40);
        }
    cv::Mat right = cv::Mat::zeros(img.size(), img.type());
    cv::Mat above = cv::Mat::zeros(img.size(), img.type());
    img.colRange(1, img.cols).copyTo(right.colRange(0, img.cols - 1));
    img.rowRange(0, img.rows - 1).copyTo(above.rowRange(1, img.rows));
    cv::Mat minP, maxP, minMax, want;
    cv::min(above, img, minP);
    cv::max(above, img, maxP);
    cv::min(maxP, right, minMax);
    cv::max(minP, minMax, want);

    std::vector<uint8_t> got(static_cast<size_t>(kW) * static_cast<size_t>(kH));
    medianWide<3, uint8_t>(img.ptr<uint8_t>(0), static_cast<size_t>(kW),
                           static_cast<size_t>(kH), img.step, got.data(),
                           static_cast<size_t>(kW), kMedianReferenceL);
    size_t diff = 0;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
            if (got[static_cast<size_t>(y) * static_cast<size_t>(kW) +
                    static_cast<size_t>(x)] != want.at<uint8_t>(y, x)) ++diff;
    std::printf("  vs three_pix_median_filter: %zu of %d pixels differ\n", diff, kW * kH);
    BINCV_CHECK(diff == 0);
}
#endif

BINCV_TEST_MAIN("test_median_wide")
