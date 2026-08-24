// X-46 -- WHERE DOES BIT-SLICING STOP PAYING?
//
// X-45 measured the two endpoints of pyrDown against cv::pyrDown: 5.52x FASTER at
// 1 -> 3 bits, 13.7x SLOWER at 8 -> 8. The crossover between them has never been
// measured, and it is the number that decides binCV's operating range -- whether
// "low bit width" means <= 3, <= 5 or <= 7, and where an 8-bit specialisation would
// have to start to be worth building.
//
// Denominator (CLAUDE.md): cv::pyrDown on CV_8U, one thread, same content, same
// geometry. It is FLAT across the sweep on purpose -- OpenCV has no cheaper mode
// for a caller who only needs three bits, and that is exactly the asymmetry binCV
// exists to exploit.
//
// Two sweeps, because they answer different questions:
//   N -> N   what an 8-bit-style pipeline costs at each width
//   1 -> N   what binCV's OWN pipeline costs: binary in, N bits out, which is what
//            pyrDown does at level 0 of every ladder
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bincv-cpp/ops/pyramid.hpp"
#include "measure_util.hpp"

#if defined(BINCV_WITH_OPENCV)
#include <opencv2/opencv.hpp>
#endif

using W = uint32_t;
using bincv::PyrDownFilter;

namespace {
constexpr int kW = 640, kH = 480, kDW = 320, kDH = 240;

template <size_t NIn>
void fill(bincv::QuantMat<NIn, W>& src) {
    const unsigned maxV = (1u << NIn) - 1u;
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
            src.set(y, x, static_cast<unsigned>(x * 7 + y * 13) % (maxV + 1u));
}

constexpr auto G = PyrDownFilter::Gaussian5x5;
constexpr auto B = PyrDownFilter::Box2x2;
} // namespace

template <PyrDownFilter F, size_t NIn, size_t NOut>
void runArm(int arm, const char* name) {
    // Allocated INSIDE the measurement, so nothing else is resident and nothing
    // outlives its lambda. The first version of this file declared every width up
    // front -- 1 + 2 + ... + 8 planes of 640x480, ~1.4 MB against a 1 MB L2 -- and
    // measureInterleaved pumped that whole set between samples, inflating the cheap
    // arms threefold: it reported `box 1 -> 3` at 352.7 us where
    // benchmark/pyrfilter_benchmark.cpp measures the identical call at 112.1 us.
    // A second version moved them into scoped blocks and left the lambdas holding
    // dangling references, which aborted in malloc. Hence this shape.
    bincv::QuantMat<NIn, W> src(kW, kH);
    bincv::QuantMat<NOut, W> dst(kDW, kDH);
    fill<NIn>(src);
    std::vector<measure::Bench> b = {
        {name, [&](int) { bincv::pyrDownFiltered<F, NOut, NIn, W>(src, dst); }}};
    const auto t = measure::measureInterleaved(b, 9, 60.0);
    std::printf("ARM %2d  %-30s %10.1f us\n", arm, name, t[0].medianNs / 1000.0);
}

int main(int argc, char** argv) {
    // ONE ARM PER PROCESS, selected by argv[1]; the caller loops. See runArm.
    const int arm = argc > 1 ? std::atoi(argv[1]) : -1;
    switch (arm) {
        case 0:  runArm<G, 1, 1>(arm, "gauss 1 -> 1"); break;
        case 1:  runArm<G, 2, 2>(arm, "gauss 2 -> 2"); break;
        case 2:  runArm<G, 3, 3>(arm, "gauss 3 -> 3"); break;
        case 3:  runArm<G, 4, 4>(arm, "gauss 4 -> 4"); break;
        case 4:  runArm<G, 5, 5>(arm, "gauss 5 -> 5"); break;
        case 5:  runArm<G, 8, 8>(arm, "gauss 8 -> 8"); break;
        case 6:  runArm<G, 1, 3>(arm, "gauss 1 -> 3 (binary in)"); break;
        case 7:  runArm<G, 1, 5>(arm, "gauss 1 -> 5 (binary in)"); break;
        case 8:  runArm<B, 1, 1>(arm, "box   1 -> 1"); break;
        case 9:  runArm<B, 2, 2>(arm, "box   2 -> 2"); break;
        case 10: runArm<B, 3, 3>(arm, "box   3 -> 3"); break;
        case 11: runArm<B, 4, 4>(arm, "box   4 -> 4"); break;
        case 12: runArm<B, 5, 5>(arm, "box   5 -> 5"); break;
        case 13: runArm<B, 8, 8>(arm, "box   8 -> 8"); break;
        case 14: runArm<B, 1, 3>(arm, "box   1 -> 3 (SHIPPED shape)"); break;
#if defined(BINCV_WITH_OPENCV)
        case 15: {
            cv::Mat cvSrc(kH, kW, CV_8U), cvDst;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x)
                    cvSrc.at<uchar>(y, x) = static_cast<uchar>((x * 7 + y * 13) % 256);
            cv::setNumThreads(1);
            std::vector<measure::Bench> b = {
                {"cv::pyrDown 8U (denominator)", [&](int) { cv::pyrDown(cvSrc, cvDst); }}};
            const auto t = measure::measureInterleaved(b, 9, 60.0);
            std::printf("ARM %2d  %-30s %10.1f us\n", arm,
                        "cv::pyrDown 8U (denominator)", t[0].medianNs / 1000.0);
            break;
        }
#endif
        default: std::printf("usage: bitwidth_crossover <arm 0..15>\n"); return 2;
    }
    return 0;
}
