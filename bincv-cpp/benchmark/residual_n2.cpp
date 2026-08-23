// X-40 -- residualSums at N = 2, the DELIVERED arm against the shipped one.
//
// X-40's ceiling measured the two SHAPES in isolation at 1.461x. This measures
// the real kernel: the same `impl::residualSums`, with and without the window-
// carried accumulator, over 31x31 windows on an N = 2 level -- the depth three of
// the four levels of the shipped 1/2/2/2 ladder run at (D-23).
//
// Equality is checked before anything is timed. The reshaping is exact -- the
// weight is constant across rows, so folding it per pair and reducing per window
// gives the same integers -- so the ten sums must come out IDENTICAL, and a
// mismatch is a failure rather than a tolerance.
#include <cstdint>
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
    uint64_t st = 11;
    auto rnd = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                         return static_cast<unsigned>(st >> 33); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Sparse, like a real edge map's levels rather than uniform noise.
            prev.set(y, x, (rnd() % 8u < 3u) ? rnd() % 4u : 0u);
            next.set(y, x, (rnd() % 8u < 3u) ? rnd() % 4u : 0u);
        }
    }
    bincv::SignedQuantMat<2, W> dx(w, h), dy(w, h);
    bincv::derivativeX(prev, dx);
    bincv::derivativeY(prev, dy);
    const auto lv = bincv::lkLevel<2>(prev, next, dx, dy);

    std::vector<bincv::impl::RegionWords<W>> regs;
    std::vector<long long> tx, ty;
    for (int y = 40; y + 31 < h - 40; y += 37) {
        for (int x = 40; x + 31 < w - 40; x += 41) {
            regs.push_back(bincv::impl::clipRegion<W>(static_cast<size_t>(w),
                                                      static_cast<size_t>(h),
                                                      bincv::Rect(x, y, 31, 31)));
            tx.push_back(static_cast<long long>(rnd() % 7u) - 3);
            ty.push_back(static_cast<long long>(rnd() % 7u) - 3);
        }
    }
    std::printf("=== X-40: residualSums at N=2, %zu windows of 31x31 ===\n\n", regs.size());

    size_t bad = 0;
    for (size_t k = 0; k < regs.size(); ++k) {
        bincv::impl::TapSums a1, b1, a2, b2;
        bincv::impl::residualSums<2, W, false>(lv, regs[k], tx[k], ty[k], a1, b1);
        bincv::impl::residualSums<2, W, true>(lv, regs[k], tx[k], ty[k], a2, b2);
        if (a1.t00 != a2.t00 || a1.t01 != a2.t01 || a1.t10 != a2.t10 || a1.t11 != a2.t11 ||
            a1.self != a2.self || b1.t00 != b2.t00 || b1.t01 != b2.t01 || b1.t10 != b2.t10 ||
            b1.t11 != b2.t11 || b1.self != b2.self) {
            if (bad < 3)
                std::printf("  MISMATCH window %zu: x.t01 %lld vs %lld\n", k, a1.t01, a2.t01);
            ++bad;
        }
    }
    std::printf("  EQUALITY: %zu of %zu windows differ\n", bad, regs.size());
    if (bad) return 1;

    // EXTRACTION ONLY. The same per-row tap machinery -- alignedWord, the interior
    // test, the t01 = t00 >> 1 identity, the masks -- with the counting REMOVED and
    // the words merely XORed into a sink. Everything the counting arms do to GET
    // the words, and nothing they do WITH them. The gap between this and the arms
    // above is the entire budget any future counting optimisation can address, so
    // it is measured rather than inferred from a ratio.
    auto extractOnly = [&](const bincv::impl::RegionWords<W>& r, long long tapX,
                           long long tapY) {
        const size_t width = r.x1 - r.x0;
        const size_t words = bincv::impl::minRowWords<W>(lv.prev[0].width);
        const W mask = bincv::impl::lowBitsMask<W>(width);
        const long long x0 = static_cast<long long>(r.x0);
        const bool tapIsShift = width < bincv::impl::bitsPerWord<W>();
        const long long srcX = x0 + tapX;
        const long long lastCol = static_cast<long long>(lv.next[0].width) - 1;
        const bool colsInside = srcX >= 0 && srcX + static_cast<long long>(width) <= lastCol;
        W sink = 0;
        for (size_t y = r.y0; y < r.y1; ++y) {
            const long long srcY = static_cast<long long>(y) + tapY;
            const bool rowsInside =
                srcY >= 0 && srcY + 1 < static_cast<long long>(lv.next[0].height);
            const bool interior = colsInside && rowsInside;
            for (size_t k = 0; k < 2; ++k) {
                W t00, t10;
                if (interior) {
                    t00 = bincv::impl::alignedWord<W>(lv.next[k].row(static_cast<size_t>(srcY)),
                                                      words, static_cast<size_t>(srcX));
                    t10 = bincv::impl::alignedWord<W>(lv.next[k].row(static_cast<size_t>(srcY) + 1),
                                                      words, static_cast<size_t>(srcX));
                } else {
                    t00 = bincv::impl::displacedRow<W>(lv.next[k], srcY, srcX).word(0);
                    t10 = bincv::impl::displacedRow<W>(lv.next[k], srcY + 1, srcX).word(0);
                }
                const W t01 = tapIsShift
                    ? static_cast<W>(t00 >> 1)
                    : bincv::impl::displacedRow<W>(lv.next[k], srcY, srcX + 1).word(0);
                const W t11 = tapIsShift
                    ? static_cast<W>(t10 >> 1)
                    : bincv::impl::displacedRow<W>(lv.next[k], srcY + 1, srcX + 1).word(0);
                sink = static_cast<W>(sink ^ t00 ^ t01 ^ t10 ^ t11);
                sink = static_cast<W>(sink ^ bincv::impl::alignedWord<W>(lv.prev[k].row(y), words, r.x0));
                sink = static_cast<W>(sink ^ (bincv::impl::alignedWord<W>(lv.dxMag[k].row(y), words, r.x0) & mask));
                sink = static_cast<W>(sink ^ (bincv::impl::alignedWord<W>(lv.dyMag[k].row(y), words, r.x0) & mask));
            }
            sink = static_cast<W>(sink ^ bincv::impl::alignedWord<W>(lv.dxSign.row(y), words, r.x0));
            sink = static_cast<W>(sink ^ bincv::impl::alignedWord<W>(lv.dySign.row(y), words, r.x0));
        }
        return sink;
    };

    std::vector<measure::Bench> b = {
        {"scalar (UseNeon=false)", [&](int) { bincv::impl::TapSums a, c;
             for (size_t k = 0; k < regs.size(); ++k)
                 bincv::impl::residualSums<2, W, false>(lv, regs[k], tx[k], ty[k], a, c);
             measure::g_sink += static_cast<size_t>(a.t00); }},
        {"shipped NEON (reduce per call)", [&](int) { bincv::impl::TapSums a, c;
             for (size_t k = 0; k < regs.size(); ++k)
                 bincv::impl::alignedResidualSums<2, W, true>(lv, regs[k], tx[k], ty[k], a, c);
             measure::g_sink += static_cast<size_t>(a.t00); }},
        {"X-40 (reduce per window)", [&](int) { bincv::impl::TapSums a, c;
             for (size_t k = 0; k < regs.size(); ++k)
                 bincv::impl::residualSums<2, W, true>(lv, regs[k], tx[k], ty[k], a, c);
             measure::g_sink += static_cast<size_t>(a.t00); }},
        {"extraction only (no counting)", [&](int) { W s2 = 0;
             for (size_t k = 0; k < regs.size(); ++k) s2 ^= extractOnly(regs[k], tx[k], ty[k]);
             measure::g_sink += static_cast<size_t>(s2); }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("\n  %-34s %10s %11s %11s\n", "arm", "us", "vs scalar", "vs shipped");
    for (size_t i = 0; i < b.size(); ++i)
        std::printf("  %-34s %10.1f %10.3fx %10.3fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[0].medianNs / t[i].medianNs, t[1].medianNs / t[i].medianNs);
    std::printf("\n  X-40's ceiling for this reshaping, measured on the shapes alone: 1.461x.\n");
    std::printf("  The last arm is the FLOOR: what the kernel costs with the counting removed\n"
                "  entirely. Everything above it that is not counting cannot be optimised by\n"
                "  any further reshaping of the counts.\n");
    return 0;
}
