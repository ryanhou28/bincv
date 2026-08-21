// X-32 -- residualSums' tap extraction. Equality is checked before any timing:
// the derivation is an identity, so the ten sums must be IDENTICAL.
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "measure_util.hpp"
#include "residual_arms.hpp"

using W = uint32_t;

int main() {
    const int w = 640, h = 480;
    bincv::QuantMat<2, W> prev(w, h), next(w, h);
    uint64_t st = 7;
    auto rnd = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                         return static_cast<unsigned>(st >> 33); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            prev.set(y, x, rnd() % 4u);
            next.set(y, x, rnd() % 4u);
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
    std::printf("=== X-32: residualSums tap extraction, N=2, %zu windows of 31x31 ===\n\n",
                regs.size());

    size_t bad = 0;
    for (size_t k = 0; k < regs.size(); ++k) {
        bincv::impl::TapSums a1, a2, b1, b2;
        residual::shipped(lv, regs[k], tx[k], ty[k], a1, b1);
        residual::hoisted(lv, regs[k], tx[k], ty[k], a2, b2);
        if (a1.t00 != a2.t00 || a1.t01 != a2.t01 || a1.t10 != a2.t10 || a1.t11 != a2.t11 ||
            a1.self != a2.self || b1.t00 != b2.t00 || b1.t01 != b2.t01 || b1.t10 != b2.t10 ||
            b1.t11 != b2.t11 || b1.self != b2.self) {
            if (bad < 3) {
                std::printf("  MISMATCH window %zu tap(%lld,%lld): x.t01 %lld vs %lld\n", k, tx[k],
                            ty[k], a1.t01, a2.t01);
            }
            ++bad;
        }
    }
    std::printf("  EQUALITY: %zu of %zu windows differ\n", bad, regs.size());
    if (bad != 0) {
        std::printf("  The derivation is an IDENTITY (X-32). Not timing a wrong kernel.\n");
        return 1;
    }

    std::vector<measure::Bench> b = {
        {"S  shipped (4 word() per word)", [&](int) {
            bincv::impl::TapSums a, c;
            for (size_t k = 0; k < regs.size(); ++k) residual::shipped(lv, regs[k], tx[k], ty[k], a, c);
            measure::g_sink += static_cast<size_t>(a.t00);
        }},
        {"H  hoisted (2 word() + 2 shifts)", [&](int) {
            bincv::impl::TapSums a, c;
            for (size_t k = 0; k < regs.size(); ++k) residual::hoisted(lv, regs[k], tx[k], ty[k], a, c);
            measure::g_sink += static_cast<size_t>(a.t00);
        }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("\n  %-34s %10s %8s\n", "arm", "us", "vs S");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf("  %-34s %10.1f %7.3fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[0].medianNs / t[i].medianNs);
    }
    return 0;
}
