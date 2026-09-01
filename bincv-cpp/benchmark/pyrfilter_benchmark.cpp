// ===========================================================================
// earlier work -- THE SPEED AXIS of the pyramid design space.
//
// a measurement measured the accuracy frontier with reference implementations and found the
// two axes are NOT independent: BOX_2x2 saturates at 3 bits (+0.82 yield points
// from N=2 to N=7) where GAUSSIAN_5x5 keeps paying (+3.93). The points worth
// pricing are GAUSSIAN_5x5 @ N=3 (0.65 below the anchor) and BOX_3x3 @ N=3.
//
// This is the cost side. It matters because a measurement measured pyrDown at 25.8% of the
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

#if defined(BINCV_WITH_OPENCV)
#include <opencv2/opencv.hpp>
#endif

using W = uint32_t;
using bincv::PyrDownFilter;

int main() {
    const int w = 640, h = 480;
    bincv::QuantMat<1, W> src(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            src.set(y, x, ((x * 7 + y * 13) % 29 == 0 || (x + y) % 37 == 0) ? 1u : 0u);

    // The 8-bit source for the compatibility point: same content, 8 bpp.
    bincv::QuantMat<8, W> src8(w, h), d8(320, 240);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            src8.set(y, x, static_cast<unsigned>((x * 7 + y * 13) % 256));

#if defined(BINCV_WITH_OPENCV)
    cv::Mat cvSrc8(h, w, CV_8U), cvDst8;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            cvSrc8.at<uchar>(y, x) = static_cast<uchar>((x * 7 + y * 13) % 256);
    cv::setNumThreads(1);
#endif

    bincv::QuantMat<3, W> d3(320, 240);
    bincv::QuantMat<2, W> d2(320, 240);
    bincv::QuantMat<1, W> d1(320, 240);

    std::printf("=== speed axis: pyrDown filters, 640x480 -> 320x240, NIn=1 ===\n\n");
    std::vector<measure::Bench> b = {
        {"pyrDown (shipped BOX_2x2 route) N=2", [&](int) { bincv::pyrDownBox<2, 1, W>(src, d2); }},
        {"pyrDown (shipped BOX_2x2 route) N=3", [&](int) { bincv::pyrDownBox<3, 1, W>(src, d3); }},
        {"filtered DIRECT_SUBSAMPLE N=1", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::DirectSubsample, 1, 1, W>(src, d1); }},
        {"filtered BOX_2x2 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box2x2, 3, 1, W>(src, d3); }},
        {"filtered BOX_3x3 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box3x3, 3, 1, W>(src, d3); }},
        {"filtered GAUSSIAN_3x3 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian3x3, 3, 1, W>(src, d3); }},
        {"filtered GAUSSIAN_5x5 N=3", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 3, 1, W>(src, d3); }},
        // THE OPENCV-COMPATIBILITY POINT: 8 bits in, 8 bits out, cv::pyrDown's
        // filter. Neither binCV claim applies here -- footprint is 8 bpp on both
        // sides by construction, and this is the configuration where bit-slicing
        // is at its worst, since the accumulators are 12 and 16 planes wide.
        // Measured so the docs can state the cost rather than imply there is none.
        {"filtered GAUSSIAN_5x5 8 -> 8 (cv::pyrDown shape)", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 8, 8, W>(src8, d8); }},
        {"filtered BOX_2x2 8 -> 8", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box2x2, 8, 8, W>(src8, d8); }},
        // WHAT THE BORDER COSTS. Reflect101 is now the default because it is
        // what cv::pyrDown does; Zero is the cheaper deviation binCV shipped before
        // and every measurement up to this was taken on. The vertical axis is free
        // (a row pointer), so this pair prices the HORIZONTAL rim -- at most
        // ceil(Radius/2) output columns per side, recomputed per pixel.
        {"GAUSSIAN_5x5 1->3 border=Reflect101", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 3, 1, W,
                                    bincv::PyrDownBorder::Reflect101>(src, d3); }},
        {"GAUSSIAN_5x5 1->3 border=Zero", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 3, 1, W,
                                    bincv::PyrDownBorder::Zero>(src, d3); }},
        {"BOX_3x3 1->3 border=Reflect101", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box3x3, 3, 1, W,
                                    bincv::PyrDownBorder::Reflect101>(src, d3); }},
        {"BOX_2x2 1->3 border=Reflect101", [&](int) {
             bincv::pyrDownFiltered<PyrDownFilter::Box2x2, 3, 1, W,
                                    bincv::PyrDownBorder::Reflect101>(src, d3); }},
#if defined(BINCV_WITH_OPENCV)
        // THE DENOMINATOR (CLAUDE.md): OpenCV doing the same semantic operation on
        // the same content. At 8 -> 8 this is literally the same function, so it is
        // the only fair comparison for the compatibility point.
        {"cv::pyrDown 8U 640x480 (the denominator)", [&](int) {
             cv::pyrDown(cvSrc8, cvDst8); }},
#endif
    };
    const auto t = measure::measureInterleaved(b, 7, 60.0);
    std::printf(" %-38s %10s %9s\n", "arm", "us", "vs shipped");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf(" %-38s %10.1f %8.2fx\n", b[i].name.c_str(), t[i].medianNs / 1000.0,
                    t[i].medianNs / t[1].medianNs);
    }
    std::printf("\n 'vs shipped' is against the hand-written BOX_2x2 route at N=3.\n"
                " pyrDown is 25.8%% of the frontend, and a level-0 pass is the\n"
                " largest of the three the ladder runs.\n");
    return 0;
}
