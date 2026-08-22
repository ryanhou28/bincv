// ===========================================================================
// X-39 / E-21 -- THE SPEED AXIS of the pyramid design space.
//
// X-39 measured the accuracy frontier with reference implementations and found the
// two axes are NOT independent: BOX_2x2 saturates at 3 bits (+0.82 yield points
// from N=2 to N=7) where GAUSSIAN_5x5 keeps paying (+3.93). The points worth
// pricing are GAUSSIAN_5x5 @ N=3 (0.65 below the anchor) and BOX_3x3 @ N=3.
//
// This is the cost side. It matters because X-38 measured pyrDown at 25.8% of the
// frontend -- up from 4.5%, because LK got 3.44x faster and the build did not.
//
// The shipped `pyrDown` (hand-written BOX_2x2 route) is included as a control: the
// generic framework has to be compared against what it would replace, not only
// against its own other settings.
// ===========================================================================
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

using W = uint32_t;
using bincv::PyrDownFilter;

int main() {
    const int w = 640, h = 480;
    bincv::QuantMat<1, W> src(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            src.set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);

    bincv::QuantMat<3, W> d3(320, 240);
    bincv::QuantMat<2, W> d2(320, 240);
    bincv::QuantMat<1, W> d1(320, 240);

    std::printf("=== X-39 speed axis: pyrDown filters, 640x480 -> 320x240, NIn=1 ===\n\n");
    std::vector<measure::Bench> b = {
        {"pyrDown (shipped BOX_2x2 route)  N=2", [&](int) { bincv::pyrDown<2, 1, W>(src, d2); }},
        {"pyrDown (shipped BOX_2x2 route)  N=3", [&](int) { bincv::pyrDown<3, 1, W>(src, d3); }},
        {"filtered DIRECT_SUBSAMPLE        N=1", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::DirectSubsample, 1, 1, W>(src, d1); }},
        {"filtered BOX_2x2                 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box2x2, 3, 1, W>(src, d3); }},
        {"filtered BOX_3x3                 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box3x3, 3, 1, W>(src, d3); }},
        {"filtered GAUSSIAN_3x3            N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian3x3, 3, 1, W>(src, d3); }},
        {"filtered GAUSSIAN_5x5            N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 3, 1, W>(src, d3); }},
    };
    const auto t = measure::measureInterleaved(b, 7, 60.0);
    std::printf("  %-38s %10s %9s\n", "arm", "us", "vs shipped");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf("  %-38s %10.1f %8.2fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[i].medianNs / t[1].medianNs);
    }
    std::printf("\n  'vs shipped' is against the hand-written BOX_2x2 route at N=3.\n"
                "  pyrDown is 25.8%% of the frontend (X-38), and a level-0 pass is the\n"
                "  largest of the three the ladder runs.\n");
    return 0;
}
