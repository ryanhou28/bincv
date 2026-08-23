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
#include <algorithm>
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

    // X-41's ARM. The same words, the same sink, with everything alignedWord
    // decides ONCE PER WINDOW instead of ~372 times:
    //   * the two (w0, s) pairs -- one for r.x0, one for srcX -- and their shifts,
    //     their `s == 0` case and their `w0 + 1 < words` bounds test;
    //   * strided row pointers carried down the window instead of `.row(y)`'s
    //     multiply, which the compiler cannot hoist across five distinct objects
    //     without a no-alias proof the signatures do not offer;
    //   * the y-loop SPLIT at the rows where `rowsInside` changes, so the interior
    //     bulk pays no per-row branch.
    auto extractHoisted = [&](const bincv::impl::RegionWords<W>& r, long long tapX,
                              long long tapY) {
        const size_t width = r.x1 - r.x0;
        const size_t words = bincv::impl::minRowWords<W>(lv.prev[0].width);
        const W mask = bincv::impl::lowBitsMask<W>(width);
        constexpr size_t B = bincv::impl::bitsPerWord<W>();
        const long long x0 = static_cast<long long>(r.x0);
        const bool tapIsShift = width < B;
        const long long srcX = x0 + tapX;
        const long long lastCol = static_cast<long long>(lv.next[0].width) - 1;
        const bool colsInside = srcX >= 0 && srcX + static_cast<long long>(width) <= lastCol;

        // Invariant #1: the two extraction descriptors.
        const size_t wSelf = r.x0 / B, sSelf = r.x0 % B;
        const bool hiSelf = (wSelf + 1) < words;
        const size_t wTap = colsInside ? static_cast<size_t>(srcX) / B : 0;
        const size_t sTap = colsInside ? static_cast<size_t>(srcX) % B : 0;
        const bool hiTap = (wTap + 1) < words;
        auto pick = [](const W* row, size_t w0, size_t sh, bool hasHi) -> W {
            const W lo = row[w0];
            if (sh == 0) return lo;
            const W hi = hasHi ? row[w0 + 1] : static_cast<W>(0);
            return static_cast<W>((lo >> sh) | (hi << (B - sh)));
        };

        // Invariant #2: the row where `rowsInside` starts and stops holding.
        const long long srcH = static_cast<long long>(lv.next[0].height);
        const long long yLo = -tapY;              // first y with srcY >= 0
        const long long yHi = srcH - 1 - tapY;       // first y with srcY + 1 >= h
        const size_t iBeg = static_cast<size_t>(
            std::max<long long>(static_cast<long long>(r.y0), yLo));
        const size_t iEnd = static_cast<size_t>(
            std::max<long long>(static_cast<long long>(iBeg),
                                std::min<long long>(static_cast<long long>(r.y1), yHi)));

        W sink = 0;
        // Invariant #3: strided row pointers, advanced rather than recomputed.
        const size_t rowStride = lv.prev[0].stride;
        auto sweep = [&](size_t yBeg, size_t yEnd, bool interior) {
            if (yBeg >= yEnd) return;
            const W* pPrev[2] = {lv.prev[0].row(yBeg), lv.prev[1].row(yBeg)};
            const W* pMagX[2] = {lv.dxMag[0].row(yBeg), lv.dxMag[1].row(yBeg)};
            const W* pMagY[2] = {lv.dyMag[0].row(yBeg), lv.dyMag[1].row(yBeg)};
            const W* pSgnX = lv.dxSign.row(yBeg);
            const W* pSgnY = lv.dySign.row(yBeg);
            for (size_t y = yBeg; y < yEnd; ++y) {
                const long long srcY = static_cast<long long>(y) + tapY;
                for (size_t k = 0; k < 2; ++k) {
                    W t00, t10;
                    if (interior) {
                        // Both source rows are inside, so they are a stride apart.
                        const W* n0 = lv.next[k].row(static_cast<size_t>(srcY));
                        t00 = pick(n0, wTap, sTap, hiTap);
                        t10 = pick(n0 + rowStride, wTap, sTap, hiTap);
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
                    sink = static_cast<W>(sink ^ pick(pPrev[k], wSelf, sSelf, hiSelf));
                    sink = static_cast<W>(sink ^ (pick(pMagX[k], wSelf, sSelf, hiSelf) & mask));
                    sink = static_cast<W>(sink ^ (pick(pMagY[k], wSelf, sSelf, hiSelf) & mask));
                    pPrev[k] += rowStride; pMagX[k] += rowStride; pMagY[k] += rowStride;
                }
                sink = static_cast<W>(sink ^ pick(pSgnX, wSelf, sSelf, hiSelf));
                sink = static_cast<W>(sink ^ pick(pSgnY, wSelf, sSelf, hiSelf));
                pSgnX += rowStride; pSgnY += rowStride;
            }
        };
        sweep(r.y0, iBeg, false);
        sweep(iBeg, iEnd, colsInside);
        sweep(iEnd, r.y1, false);
        return sink;
    };

    // Equality before timing, as everywhere else in this project.
    size_t badX = 0;
    for (size_t k = 0; k < regs.size(); ++k)
        if (extractOnly(regs[k], tx[k], ty[k]) != extractHoisted(regs[k], tx[k], ty[k])) ++badX;
    std::printf("  EXTRACTION EQUALITY: %zu of %zu windows differ\n", badX, regs.size());
    if (badX) return 1;

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
        {"X-41 extraction, hoisted+strided", [&](int) { W s2 = 0;
             for (size_t k = 0; k < regs.size(); ++k) s2 ^= extractHoisted(regs[k], tx[k], ty[k]);
             measure::g_sink += static_cast<size_t>(s2); }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    // -------------------------------------------------------------------------
    // X-41's SECOND QUESTION, asked because the first was answered NO.
    //
    // If the extraction is not addressing, it is the LOADS. A 31x31 window touches
    // 31 rows of TEN separate planes -- prev x2, next x2, dxMag x2, dyMag x2 and
    // the two sign planes -- and consecutive rows of one plane are a stride apart,
    // so every one of those 310 row-touches is its own cache line. That is ~19.8 KB
    // of lines fetched for ~2.5 KB of useful bits, an 8x overfetch, against a 32 KB
    // L1D.
    //
    // The discriminator: run the SAME extraction over a level small enough that all
    // ten planes fit in L1 together. If the cost per window collapses, the
    // extraction is bound by the layout; if it does not, it is bound by the work.
    // -------------------------------------------------------------------------
    const int sw = 128, sh2 = 96;   // 10 planes x 128x96 bits = ~15 KB, inside 32 KB L1D
    bincv::QuantMat<2, W> sPrev(sw, sh2), sNext(sw, sh2);
    for (int y = 0; y < sh2; ++y)
        for (int x = 0; x < sw; ++x) {
            sPrev.set(y, x, (rnd() % 8u < 3u) ? rnd() % 4u : 0u);
            sNext.set(y, x, (rnd() % 8u < 3u) ? rnd() % 4u : 0u);
        }
    bincv::SignedQuantMat<2, W> sdx(sw, sh2), sdy(sw, sh2);
    bincv::derivativeX(sPrev, sdx);
    bincv::derivativeY(sPrev, sdy);
    const auto slv = bincv::lkLevel<2>(sPrev, sNext, sdx, sdy);
    std::vector<bincv::impl::RegionWords<W>> sregs;
    std::vector<long long> stx, sty;
    for (size_t k = 0; k < regs.size(); ++k) {
        const int yy = 33 + static_cast<int>((k * 7) % 30);
        const int xx = 33 + static_cast<int>((k * 11) % 60);
        sregs.push_back(bincv::impl::clipRegion<W>(static_cast<size_t>(sw),
                                                   static_cast<size_t>(sh2),
                                                   bincv::Rect(xx, yy, 31, 31)));
        stx.push_back(tx[k]);
        sty.push_back(ty[k]);
    }
    // Same code, different level: a copy of extractOnly bound to `slv`.
    auto extractSmall = [&](const bincv::impl::RegionWords<W>& r, long long tapX,
                            long long tapY) {
        const size_t width = r.x1 - r.x0;
        const size_t words = bincv::impl::minRowWords<W>(slv.prev[0].width);
        const W mask = bincv::impl::lowBitsMask<W>(width);
        const long long x0 = static_cast<long long>(r.x0);
        const bool tapIsShift = width < bincv::impl::bitsPerWord<W>();
        const long long srcX = x0 + tapX;
        const long long lastCol = static_cast<long long>(slv.next[0].width) - 1;
        const bool colsInside = srcX >= 0 && srcX + static_cast<long long>(width) <= lastCol;
        W sk = 0;
        for (size_t y = r.y0; y < r.y1; ++y) {
            const long long srcY = static_cast<long long>(y) + tapY;
            const bool rowsInside =
                srcY >= 0 && srcY + 1 < static_cast<long long>(slv.next[0].height);
            const bool interior = colsInside && rowsInside;
            for (size_t k = 0; k < 2; ++k) {
                W t00, t10;
                if (interior) {
                    t00 = bincv::impl::alignedWord<W>(slv.next[k].row(static_cast<size_t>(srcY)),
                                                      words, static_cast<size_t>(srcX));
                    t10 = bincv::impl::alignedWord<W>(slv.next[k].row(static_cast<size_t>(srcY) + 1),
                                                      words, static_cast<size_t>(srcX));
                } else {
                    t00 = bincv::impl::displacedRow<W>(slv.next[k], srcY, srcX).word(0);
                    t10 = bincv::impl::displacedRow<W>(slv.next[k], srcY + 1, srcX).word(0);
                }
                const W t01 = tapIsShift ? static_cast<W>(t00 >> 1)
                    : bincv::impl::displacedRow<W>(slv.next[k], srcY, srcX + 1).word(0);
                const W t11 = tapIsShift ? static_cast<W>(t10 >> 1)
                    : bincv::impl::displacedRow<W>(slv.next[k], srcY + 1, srcX + 1).word(0);
                sk = static_cast<W>(sk ^ t00 ^ t01 ^ t10 ^ t11);
                sk = static_cast<W>(sk ^ bincv::impl::alignedWord<W>(slv.prev[k].row(y), words, r.x0));
                sk = static_cast<W>(sk ^ (bincv::impl::alignedWord<W>(slv.dxMag[k].row(y), words, r.x0) & mask));
                sk = static_cast<W>(sk ^ (bincv::impl::alignedWord<W>(slv.dyMag[k].row(y), words, r.x0) & mask));
            }
            sk = static_cast<W>(sk ^ bincv::impl::alignedWord<W>(slv.dxSign.row(y), words, r.x0));
            sk = static_cast<W>(sk ^ bincv::impl::alignedWord<W>(slv.dySign.row(y), words, r.x0));
        }
        return sk;
    };
    std::vector<measure::Bench> b2 = {
        {"extraction, 640x480 level (384 KB)", [&](int) { W s2 = 0;
             for (size_t k = 0; k < regs.size(); ++k) s2 ^= extractOnly(regs[k], tx[k], ty[k]);
             measure::g_sink += static_cast<size_t>(s2); }},
        {"extraction, 128x96 level (~15 KB)", [&](int) { W s2 = 0;
             for (size_t k = 0; k < sregs.size(); ++k) s2 ^= extractSmall(sregs[k], stx[k], sty[k]);
             measure::g_sink += static_cast<size_t>(s2); }},
    };
    const auto t2 = measure::measureInterleaved(b2, 9, 60.0);
    std::printf("\n  %-34s %10s %11s %11s\n", "arm", "us", "vs scalar", "vs shipped");
    for (size_t i = 0; i < b.size(); ++i)
        std::printf("  %-34s %10.1f %10.3fx %10.3fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[0].medianNs / t[i].medianNs, t[1].medianNs / t[i].medianNs);
    std::printf("\n  X-40's ceiling for this reshaping, measured on the shapes alone: 1.461x.\n");
    std::printf("  The floor arm is what the kernel costs with the counting removed entirely.\n"
                "  Everything above it that is not counting cannot be optimised by any\n"
                "  further reshaping of the counts.\n");
    std::printf("\n  === X-41's second question: is the extraction the LAYOUT? ===\n");
    std::printf("  Same extraction, same window count, working set in and out of L1D:\n\n");
    std::printf("  %-38s %10s %10s\n", "arm", "us", "vs large");
    for (size_t i = 0; i < b2.size(); ++i)
        std::printf("  %-38s %10.1f %9.3fx\n", b2[i].name.c_str(), t2[i].medianNs / 1000.0,
                    t2[0].medianNs / t2[i].medianNs);
    std::printf("\n  A 31x31 window touches 31 rows of TEN separate planes: 310 distinct cache\n"
                "  lines, ~19.8 KB fetched for ~2.5 KB of useful bits. If the small level is\n"
                "  much faster, the extraction is bound by that layout and not by its work.\n");
    return 0;
}
