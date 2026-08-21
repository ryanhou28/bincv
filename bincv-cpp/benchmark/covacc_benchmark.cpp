// ===========================================================================
// X-29 / E-13 -- does D-15 item 4's per-row partial accumulator still pay
// above N = 1?
//
// THE NOISE FLOOR IS MEASURED, NOT ASSUMED. Arm P' is bit-identical source to
// arm P in a second translation unit, so P vs P' is pure code layout. That
// spread is L, and W vs P is judged against L rather than against zero --
// because X-22 measured this same kernel moving 1.46x between binaries built
// from unchanged source, and declined to close on a 1.14x reading for exactly
// that reason. See covacc_arms.hpp.
// ===========================================================================

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "bincv-cpp/quantMat.hpp"
#include "covacc_arms.hpp"
#include "measure_util.hpp"

using W = uint32_t;

int main() {
    const int width = 640, height = 480;
    constexpr size_t kMaxN = 4;

    // One 4-plane magnitude pair plus two sign planes; a given N uses the first N.
    std::vector<bincv::BinMat<W>> mx, my;
    for (size_t p = 0; p < kMaxN; ++p) { mx.emplace_back(width, height); my.emplace_back(width, height); }
    bincv::BinMat<W> sx(width, height), sy(width, height);
    uint64_t state = 12345;
    auto rnd = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<unsigned>(state >> 33);
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (size_t p = 0; p < kMaxN; ++p) {
                mx[p].set(y, x, (rnd() & 7u) == 0 ? 1u : 0u);
                my[p].set(y, x, (rnd() & 7u) == 0 ? 1u : 0u);
            }
            sx.set(y, x, rnd() & 1u);
            sy.set(y, x, rnd() & 1u);
        }
    }
    std::vector<bincv::BinMatConstView<W>> vx, vy;
    for (size_t p = 0; p < kMaxN; ++p) { vx.push_back(mx[p].constView()); vy.push_back(my[p].constView()); }

    // Window positions swept so the head/tail region masks vary, not one aligned case.
    std::vector<bincv::Rect> windows;
    for (int y = 0; y + 31 < height; y += 29) {
        for (int x = 0; x + 31 < width; x += 23) windows.emplace_back(x, y, 31, 31);
    }
    std::printf("=== X-29 / E-13: per-row vs window-wide accumulator ===\n");
    std::printf("640x480, %zu windows of 31x31, uint32_t\n", windows.size());
    std::printf("P and P' are BIT-IDENTICAL SOURCE in different translation units:\n"
                "their spread is the code-layout NOISE FLOOR L.\n\n");
    std::printf("   N |      P us |     P' us |      W us |   L (P vs P') |  W vs P |  verdict\n");
    std::printf("  ---+-----------+-----------+-----------+---------------+---------+---------\n");

    for (size_t n = 1; n <= kMaxN; ++n) {
        auto mk = [&](covacc::Fn f) {
            return [f, &vx, &vy, &sx, &sy, n, &windows](int) {
                const int64_t r = f(vx.data(), vy.data(), sx.constView(), sy.constView(), n,
                                    windows.data(), windows.size());
                measure::g_sink += static_cast<size_t>(r);
            };
        };
        std::vector<measure::Bench> benches = {
            {"P", mk(&covacc::perRow)},
            {"P'", mk(&covacc::perRowB)},
            {"W", mk(&covacc::windowWide)},
        };
        const auto t = measure::measureInterleaved(benches, 9, 50.0);
        const double p = t[0].medianNs, pb = t[1].medianNs, w = t[2].medianNs;
        const double L = std::fabs(p - pb) / std::min(p, pb);
        const double gain = p / w;                       // >1 means W is faster
        const double effect = std::fabs(gain - 1.0);
        const char* verdict = (effect <= L) ? "IN NOISE" : (gain > 1.0 ? "W wins" : "P wins");
        std::printf("  %2zu | %9.1f | %9.1f | %9.1f | %12.1f%% | %6.3fx | %s\n", n,
                    p / 1000.0, pb / 1000.0, w / 1000.0, 100.0 * L, gain, verdict);
    }
    std::printf("\n  L is |P - P'| / min(P, P') -- the same algorithm, different objects.\n"
                "  A 'W vs P' ratio whose deviation from 1.0 is inside L is not a result.\n");
    return 0;
}
