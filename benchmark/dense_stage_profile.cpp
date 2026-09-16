// Where `denseDisparityBinary`'s time goes on this machine, by stage.
//
// The binary dense path is four distinct loop families -- the vertical
// accumulator slide, the horizontal doubling tree, the compare-and-select, and
// the byte extraction -- and a vector pass should land on the one that owns the
// time. The LK stage profile exists because three guessed optimizations in a
// row measured under 2%; this is the same instrument pointed at the dense path.
//
// Same frame synthesis and parameters as dense_benchmark's BINARY-NATIVE arm,
// so the shares decompose that arm's number and not a synthetic one.

// Before the include: the hook is off by default and this is the only consumer.
#define BINCV_DENSE_STAGE_TIMING 1

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/denseDisparity.hpp"
#include "bincv/ops/pack.hpp"
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
    const std::vector<uint8_t> lw = smoothFrame(kW, kH);
    std::vector<uint8_t> rw(static_cast<size_t>(kW) * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x)
            rw[y * kW + x] = lw[y * kW + x + kDisp];

    bincv::BinMat<uint64_t> lb(kW, kH), rb(kW, kH);
    bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), size_t{kW}, size_t{kH},
                                                  size_t{kW}, lb.view(), uint8_t{127});
    bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), size_t{kW}, size_t{kH},
                                                  size_t{kW}, rb.view(), uint8_t{127});

    bincv::DenseDisparityParams bp;
    bp.maxDisparity = 64;
    std::vector<uint64_t> sw(bincv::denseDisparityBinaryScratchWords<uint64_t>(kW, bp));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(static_cast<size_t>(kW) * kH);

    constexpr int kWarmup = 2, kReps = 10;
    for (int r = 0; r < kWarmup; ++r)
        bincv::denseDisparityBinary<uint64_t>(lb.constView(), rb.constView(), bp,
                                              sw.data(), sw.size(), sr.data(), sr.size(),
                                              disp.data(), kW);
    bincv::impl::denseStageTiming() = {};

    const uint64_t w0 = bincv::impl::denseStageNow();
    for (int r = 0; r < kReps; ++r) {
        bincv::denseDisparityBinary<uint64_t>(lb.constView(), rb.constView(), bp,
                                              sw.data(), sw.size(), sr.data(), sr.size(),
                                              disp.data(), kW);
        measure::g_sink += disp[static_cast<size_t>(kH / 2) * kW + kW / 2];
    }
    const uint64_t wallNs = bincv::impl::denseStageNow() - w0;

    const auto& t = bincv::impl::denseStageTiming();
    const uint64_t sum = t.ringNs + t.treeNs + t.wtaNs + t.extractNs;
    const auto line = [&](const char* name, uint64_t ns) {
        std::printf(" %-28s %8.2f ms/frame  %5.1f%%\n", name,
                    static_cast<double>(ns) / kReps / 1e6,
                    100.0 * static_cast<double>(ns) / static_cast<double>(sum));
    };
    std::printf("=== denseDisparityBinary stages, %dx%d, D=64, 9x9, u64, %d reps ===\n\n",
                kW, kH, kReps);
    line("vertical ring (add/sub)", t.ringNs);
    line("horizontal doubling tree", t.treeNs);
    line("compare + masked selects", t.wtaNs);
    line("byte extraction", t.extractNs);
    std::printf("\n stages sum %.2f ms/frame; wall %.2f ms/frame (the gap is timing\n"
                " overhead plus the row setup the stages do not cover)\n",
                static_cast<double>(sum) / kReps / 1e6,
                static_cast<double>(wallNs) / kReps / 1e6);
    std::printf(" sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
