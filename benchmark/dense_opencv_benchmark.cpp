// Dense disparity against the cv::StereoBM denominator.
//
// THE COMPARISON IS OF ROLE, NOT NUMERICS -- StereoBM runs SAD over prefiltered
// bytes with its own validity rules; binCV runs census Hamming. What a caller
// weighs is the pair (time, working set) each pays for a disparity map of the
// same frame, and that is what this prints. The correctness triangle (both
// against synthetic truth) lives in tests/test_dense_disparity.cpp; this file
// prices it.
//
// StereoBM's working-set figure below is the honest visible part -- its output
// (CV_16S, 2 bytes/px) plus the buffers create() sizes internally are not all
// observable, so the printed figure is a LOWER bound labelled as such, against
// binCV's exactly-computable scratch.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv/ops/denseDisparity.hpp"
#include "measure_util.hpp"

namespace {
constexpr int kW = 752, kH = 480;
constexpr int kDisp = 21;

std::vector<uint8_t> smoothFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = 0xFEEDFACEULL;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    std::vector<uint8_t> tmp(w * h);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            tmp[y * w + x] = static_cast<uint8_t>(
                (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(
                (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) / 4u);
    return img;
}
} // namespace

int main() {
    cv::setNumThreads(1);   // binCV is single-threaded, so OpenCV is too
    const std::vector<uint8_t> lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(static_cast<size_t>(kW) * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kDisp];

    cv::Mat lcv(kH, kW, CV_8U), rcv(kH, kW, CV_8U);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            lcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = lw[y * kW + x];
            rcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = rw[y * kW + x];
        }

    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    std::vector<uint32_t> sw(bincv::denseDisparityScratchWords<24, uint32_t>(kW, p));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(static_cast<size_t>(kW) * kH);
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(64, 21);
    cv::Mat disp16;

    std::printf("=== dense disparity vs cv::StereoBM (role, not numerics) ===\n");
    std::printf(" %dx%d, 64 disparities, one thread each\n\n", kW, kH);

    std::vector<measure::Bench> bs = {
        {"binCV denseDisparity (census 5x5, 9x9)",
         [&](int) {
             bincv::denseDisparity<24, uint8_t, uint32_t>(
                 lw.data(), rw.data(), kW, kH, kW, kW, bincv::kCensus5x5, p, sw.data(),
                 sw.size(), sr.data(), sr.size(), disp.data(), kW);
             measure::g_sink += disp[static_cast<size_t>(kH / 2) * kW + kW / 2];
         }},
        {"cv::StereoBM (SAD, blockSize 21)",
         [&](int) {
             bm->compute(lcv, rcv, disp16);
             measure::g_sink +=
                 static_cast<size_t>(disp16.at<short>(kH / 2, kW / 2));
         }},
    };
    const auto t = measure::measureInterleaved(bs, 5, 60.0);
    for (size_t i = 0; i < bs.size(); ++i)
        std::printf(" %-42s %10.2f ms  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6, t[i].spreadPct());
    std::printf("\n working sets: binCV scratch %zu B + 1 B/px output;"
                " StereoBM >= %zu B output alone (2 B/px, internal buffers"
                " unobserved)\n",
                sw.size() * sizeof(uint32_t) + sr.size() * sizeof(uint16_t),
                static_cast<size_t>(kW) * kH * 2);
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
