// The census transform, priced at birth (CLAUDE.md's rule).
//
// Core-only, binCV against binCV: mainline OpenCV has no census to be a
// denominator (one lives in opencv_contrib, which no configuration here builds).
// The arms price the two shipped neighbourhoods per word type, which is what the
// dense-disparity design needs to budget its transform stage -- and what makes
// the shipped per-pixel v1 an honest baseline for the word-parallel
// restructuring to beat when the pipeline's share justifies it.
//
// MEMORY, stated rather than discovered: the FULL-FRAME spelling holds K planes
// resident -- at 752x480/uint32, 8 planes are 361 KB and 24 planes are 1 083 KB.
// The dense pipeline's recorded memory rule streams an ~11-row band instead
// (~50 KB at 24 planes); this benchmark prices the transform's compute, and the
// full-frame numbers below are the CONVENIENCE spelling's footprint, not the
// pipeline's.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/census.hpp"
#include "measure_util.hpp"

namespace {

constexpr int kW = 752, kH = 480;   // the reference frame size

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

template <size_t K, typename W>
struct Planes {
    std::vector<bincv::BinMat<W>> store;
    std::vector<bincv::BinMatView<W>> views;
    Planes() {
        store.reserve(K);
        for (size_t k = 0; k < K; ++k) {
            store.emplace_back(kW, kH);
            views.push_back(store.back().view());
        }
    }
    size_t bytes() const { return K * store[0].sizeInWords() * sizeof(W); }
};

} // namespace

int main() {
    const std::vector<uint8_t> img = smoothFrame(kW, kH);
    Planes<8, uint32_t> p8;
    Planes<24, uint32_t> p24;
    Planes<24, uint64_t> p24w;

    std::printf("=== census transform ===\n");
    std::printf(" %dx%d uint8 frame; full-frame plane sets: 3x3 %zu B, 5x5 %zu B\n\n",
                kW, kH, p8.bytes(), p24.bytes());

    std::vector<measure::Bench> bs = {
        {"census 3x3 (8 planes, u32)",
         [&](int) {
             bincv::censusTransform<8, uint8_t, uint32_t>(
                 img.data(), kW, kH, kW, bincv::kCensus3x3, p8.views.data());
             measure::g_sink += p8.store[0].data()[0];
         }},
        {"census 5x5 (24 planes, u32)",
         [&](int) {
             bincv::censusTransform<24, uint8_t, uint32_t>(
                 img.data(), kW, kH, kW, bincv::kCensus5x5, p24.views.data());
             measure::g_sink += p24.store[0].data()[0];
         }},
        {"census 5x5 (24 planes, u64)",
         [&](int) {
             bincv::censusTransform<24, uint8_t, uint64_t>(
                 img.data(), kW, kH, kW, bincv::kCensus5x5, p24w.views.data());
             measure::g_sink += static_cast<size_t>(p24w.store[0].data()[0]);
         }},
    };
    const auto t = measure::measureInterleaved(bs, 7, 60.0);
    std::printf(" %-34s %12s %14s\n", "arm", "ms/frame", "ns/px/plane");
    for (size_t i = 0; i < bs.size(); ++i) {
        const double planes = i == 0 ? 8.0 : 24.0;
        std::printf(" %-34s %12.3f %14.2f  spread %.0f%%\n", bs[i].name.c_str(),
                    t[i].medianNs / 1e6,
                    t[i].medianNs / (static_cast<double>(kW) * kH * planes),
                    t[i].spreadPct());
    }
    std::printf("\n per-pixel v1: the number the word-parallel restructuring has to"
                " beat.\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
