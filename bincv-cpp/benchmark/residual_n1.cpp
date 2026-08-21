// X-36 isolated: residualSums at N = 1, tap-batched NEON vs scalar. The shipped
// 1/2/2/2 ladder has only ONE level at N = 1, so a whole-ladder measurement dilutes
// this by four; the kernel ratio is what the decision rule asks for.
#include <cstdio>
#include <string>
#include <vector>
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "measure_util.hpp"
using W = uint32_t;
int main() {
    const int w = 640, h = 480;
    bincv::QuantMat<1, W> prev(w, h), next(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            prev.set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);
            next.set(y, x, (((x - 1) * 7 + y * 13) % 29 == 0 || (x - 1 + y) % 37 == 0) ? 1u : 0u);
        }
    bincv::SignedQuantMat<1, W> dx(w, h), dy(w, h);
    bincv::derivativeX(prev, dx); bincv::derivativeY(prev, dy);
    const auto lv = bincv::lkLevel<1>(prev, next, dx, dy);

    std::vector<bincv::impl::RegionWords<W>> regs;
    std::vector<long long> tx, ty;
    uint64_t st = 3;
    auto rnd = [&st]{ st = st*6364136223846793005ULL+1442695040888963407ULL; return (unsigned)(st>>33); };
    for (int y = 40; y + 31 < h - 40; y += 37)
        for (int x = 40; x + 31 < w - 40; x += 41) {
            regs.push_back(bincv::impl::clipRegion<W>((size_t)w, (size_t)h, bincv::Rect(x, y, 31, 31)));
            tx.push_back((long long)(rnd() % 7u) - 3);
            ty.push_back((long long)(rnd() % 7u) - 3);
        }
    std::printf("=== X-36: residualSums at N=1, %zu windows of 31x31 ===\n\n", regs.size());

    size_t bad = 0;
    for (size_t k = 0; k < regs.size(); ++k) {
        bincv::impl::TapSums a1, b1, a2, b2;
        bincv::impl::residualSums<1, W, false>(lv, regs[k], tx[k], ty[k], a1, b1);
        bincv::impl::residualSums<1, W, true>(lv, regs[k], tx[k], ty[k], a2, b2);
        if (a1.t00!=a2.t00||a1.t01!=a2.t01||a1.t10!=a2.t10||a1.t11!=a2.t11||a1.self!=a2.self||
            b1.t00!=b2.t00||b1.t01!=b2.t01||b1.t10!=b2.t10||b1.t11!=b2.t11||b1.self!=b2.self) {
            if (bad < 3) std::printf("  MISMATCH window %zu: x.t01 %lld vs %lld\n", k, a1.t01, a2.t01);
            ++bad;
        }
    }
    std::printf("  EQUALITY: %zu of %zu windows differ\n", bad, regs.size());
    if (bad) return 1;

    std::vector<measure::Bench> b = {
        {"scalar (UseNeon=false)", [&](int){ bincv::impl::TapSums a,c;
            for (size_t k=0;k<regs.size();++k)
                bincv::impl::residualSums<1,W,false>(lv,regs[k],tx[k],ty[k],a,c);
            measure::g_sink += (size_t)a.t00; }},
        {"tap-batched NEON", [&](int){ bincv::impl::TapSums a,c;
            for (size_t k=0;k<regs.size();++k)
                bincv::impl::residualSums<1,W,true>(lv,regs[k],tx[k],ty[k],a,c);
            measure::g_sink += (size_t)a.t00; }},
    };
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("\n  %-30s %10s %9s\n", "arm", "us", "vs scalar");
    for (size_t i = 0; i < b.size(); ++i)
        std::printf("  %-30s %10.1f %8.3fx\n", b[i].name.c_str(), t[i].medianNs/1000.0,
                    t[0].medianNs/t[i].medianNs);
    return 0;
}
