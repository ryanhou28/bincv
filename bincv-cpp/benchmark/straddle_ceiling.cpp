// X-34's CEILING, and it needs no new kernel.
//
// A 31-pixel window at x0 % 32 == 0 occupies ONE uint32 word; at x0 % 32 == 5 it
// occupies TWO. Timing the shipped residualSums on each bounds what aligning the
// window could buy, with the alignment cost removed entirely. Under 1.3x and
// X-34's arm is not written.
//
// Both arms run the SAME number of windows over the SAME content and differ only
// in the x-offset, so the pixel count is identical and only the word count moves.
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "measure_util.hpp"

using W = uint32_t;

int main() {
    const int w = 640, h = 480;
    bincv::QuantMat<2, W> prev(w, h), next(w, h);
    uint64_t st = 7;
    auto rnd = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                         return static_cast<unsigned>(st >> 33); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) { prev.set(y, x, rnd() % 4u); next.set(y, x, rnd() % 4u); }
    }
    bincv::SignedQuantMat<2, W> dx(w, h), dy(w, h);
    bincv::derivativeX(prev, dx);
    bincv::derivativeY(prev, dy);
    const auto lv = bincv::lkLevel<2>(prev, next, dx, dy);

    auto build = [&](int phase) {
        std::vector<bincv::impl::RegionWords<W>> v;
        for (int y = 40; y + 31 < h - 40; y += 37) {
            for (int x = 64; x + 32 < w - 64; x += 64) {
                v.push_back(bincv::impl::clipRegion<W>(static_cast<size_t>(w),
                                                       static_cast<size_t>(h),
                                                       bincv::Rect(x + phase, y, 31, 31)));
            }
        }
        return v;
    };
    const auto aligned = build(0);    // x0 % 32 == 0 -> one word
    const auto straddling = build(5); // x0 % 32 == 5 -> two words

    auto words = [](const std::vector<bincv::impl::RegionWords<W>>& v) {
        double n = 0;
        for (const auto& r : v) n += static_cast<double>(r.lastWord - r.firstWord + 1) *
                                     static_cast<double>(r.y1 - r.y0);
        return n;
    };
    std::printf("=== X-34 CEILING: aligned vs straddling window ===\n");
    std::printf("  %zu windows of 31x31 each, N=2, uint32_t\n", aligned.size());
    std::printf("  word-visits: aligned %.0f, straddling %.0f (%.2fx)\n\n", words(aligned),
                words(straddling), words(straddling) / words(aligned));

    std::vector<measure::Bench> b = {
        {"straddling (x0 % 32 == 5, two words)", [&](int) {
             bincv::impl::TapSums a, c;
             for (const auto& r : straddling) bincv::impl::residualSums(lv, r, 1, 1, a, c);
             measure::g_sink += static_cast<size_t>(a.t00);
         }},
        {"aligned    (x0 % 32 == 0, one word)", [&](int) {
             bincv::impl::TapSums a, c;
             for (const auto& r : aligned) bincv::impl::residualSums(lv, r, 1, 1, a, c);
             measure::g_sink += static_cast<size_t>(a.t00);
         }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("  %-40s %10s %9s\n", "arm", "us", "vs strad");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf("  %-40s %10.1f %8.3fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[0].medianNs / t[i].medianNs);
    }
    std::printf("\n  This is an UPPER BOUND: the real kernel must also pay to align.\n"
                "  X-34's rule: under 1.3x here and the arm is not written.\n");
    return 0;
}
