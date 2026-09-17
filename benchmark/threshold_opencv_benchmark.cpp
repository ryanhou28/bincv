// bincv::threshold against cv::threshold -- the Tier 1 claim, priced (issue #55).
//
// threshold.hpp's cv::Mat overload is API TIER 1: bit-exact against
// cv::threshold's THRESH_BINARY by test. It had no benchmark, so the claim
// "same answer" carried no price tag. This states one: milliseconds per frame
// and the output each side hands the next stage -- binCV writes 1 bit per
// pixel, OpenCV 8 bits, and that 8x is the working-set argument the library is
// built on, so it is printed beside the time rather than left implied.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "bincv/binMat.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/threshold.hpp"
#include "measure_util.hpp"

namespace {
constexpr int kW = 752, kH = 480;
}

int main() {
    cv::Mat src(kH, kW, CV_8UC1);
    uint64_t st = 0xB1A5EDULL;
    for (size_t i = 0; i < static_cast<size_t>(kW) * kH; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        src.data[i] = static_cast<uint8_t>(st >> 40);
    }

    bincv::BinMat<uint32_t> bits(kW, kH);
    cv::Mat cvOut;

    std::vector<measure::Bench> bs = {
        {"bincv::threshold -> BinMat (1 bpp)", [&](int) {
             bincv::threshold(src, bits.view(), 127.0);
             measure::g_sink += bits.view().row(kH / 2)[0];
         }},
        {"cv::threshold -> cv::Mat (8 bpp)", [&](int) {
             cv::threshold(src, cvOut, 127.0, 255.0, cv::THRESH_BINARY);
             measure::g_sink += cvOut.data[kW * (kH / 2)];
         }},
        // The route a binCV consumer would actually take through OpenCV: the
        // next stage needs bits, so cv's bytes still have to be packed. This
        // is the right baseline; the raw cv line above is the wrong one for a
        // bit-plane pipeline and is printed to show the difference.
        {"cv::threshold + packBits (to bits)", [&](int) {
             cv::threshold(src, cvOut, 127.0, 255.0, cv::THRESH_BINARY);
             bincv::packBits<bincv::PackRule::NonZero>(cvOut.data, size_t{kW},
                                                       size_t{kH}, size_t{kW},
                                                       bits.view());
             measure::g_sink += bits.view().row(kH / 2)[0];
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 30.0);
    std::printf("=== threshold, %dx%d (Tier 1: bit-exact by test; this is the price) ===\n\n",
                kW, kH);
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf(" %-36s %9.3f ms  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    std::printf("\n to-bits ratio %.2fx (>1 means binCV is faster) -- and the raw cv\n"
                " line's outputs differ 8x: %zu B against %zu B handed on.\n",
                t[2].medianNs / t[0].medianNs,
                static_cast<size_t>(bits.view().stride) * kH * sizeof(uint32_t),
                static_cast<size_t>(kW) * kH);
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
