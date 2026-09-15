// Dense disparity, priced at birth (CLAUDE.md's rule).
//
// Core-only, binCV against binCV: the roofline questions the dense design left
// open are answered here -- what the disparity range costs (the memory rule made
// peak INDEPENDENT of D; time is linear in it), what the window costs, and what
// the word type buys. The cv::StereoBM denominator lives in
// dense_opencv_benchmark.cpp, so the reference device's core-only default build
// still produces these numbers.
//
// MEMORY, stated: scratch is the two census bands plus the accumulator ladder
// and three uint16 rows -- printed below from the sizing functions themselves,
// so the number in the log is the number the contract computes. The naive cost
// volume this design refuses would be width * height * D bytes.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/ops/denseDisparity.hpp"
#include "measure_util.hpp"

namespace {

constexpr int kW = 752, kH = 480;   // the reference frame size
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

template <typename W>
double runArm(const std::vector<uint8_t>& lw, const std::vector<uint8_t>& rw,
              const bincv::DenseDisparityParams& p, std::vector<uint8_t>& disp,
              const char* name) {
    std::vector<W> sw(bincv::denseDisparityScratchWords<24, W>(kW, p));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(kW));
    std::vector<measure::Bench> bs = {
        {name, [&](int) {
             bincv::denseDisparity<24, uint8_t, W>(lw.data(), rw.data(), kW, kH, kW, kW,
                                                   bincv::kCensus5x5, p, sw.data(),
                                                   sw.size(), sr.data(), sr.size(),
                                                   disp.data(), kW);
             measure::g_sink += disp[static_cast<size_t>(kH / 2) * kW + kW / 2];
         }}};
    const auto t = measure::measureInterleaved(bs, 5, 60.0);
    std::printf(" %-40s %10.2f ms  scratch %zu B  spread %.0f%%\n", name,
                t[0].medianNs / 1e6,
                sw.size() * sizeof(W) + sr.size() * sizeof(uint16_t), t[0].spreadPct());
    return t[0].medianNs;
}

} // namespace

int main() {
    const std::vector<uint8_t> lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(static_cast<size_t>(kW) * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kDisp];
    std::vector<uint8_t> disp(static_cast<size_t>(kW) * kH);

    std::printf("=== dense disparity (census 5x5, streamed) ===\n");
    std::printf(" %dx%d pair; the refused cost volume would be %d B per byte of cell\n\n",
                kW, kH, kW * kH * 64);

    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    const double d64 = runArm<uint32_t>(lw, rw, p, disp, "D=64, 9x9, u32 sliding");
    const double d64w = runArm<uint64_t>(lw, rw, p, disp, "D=64, 9x9, u64 sliding");
    p.recomputeVertical = true;
    const double d64r = runArm<uint64_t>(lw, rw, p, disp, "D=64, 9x9, u64 RECOMPUTE arm");
    p.recomputeVertical = false;
    p.maxDisparity = 32;
    const double d32 = runArm<uint32_t>(lw, rw, p, disp, "D=32, 9x9, u32 sliding");
    p.maxDisparity = 64;
    p.winWidth = 5;
    p.winHeight = 5;
    runArm<uint32_t>(lw, rw, p, disp, "D=64, 5x5, u32 sliding");
    std::printf("\n sliding vs recompute at u64: %.2fx (the arm the default buys)\n",
                d64r / d64w);

    std::printf("\n D=64 vs D=32 time ratio %.2fx (the design trades time linear in D\n"
                " for peak memory independent of it; this is the linearity check).\n",
                d64 / d32);
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
