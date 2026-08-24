// X-47 -- INTEROP OR SPECIALISATION above the bit-width crossover?
//
// X-46 measured binCV 2.5-14x slower than OpenCV above the (filter-dependent)
// crossover. The candidate answers: specialise wide-N cases internally to a byte
// representation -- a second storage layout and a second implementation of every
// kernel -- or make QuantMat<N> <-> cv::Mat conversion first-class and hand wide
// intermediates to OpenCV, which is already optimal at 8 bits.
//
// R = the 8->8 round trip toCVMatNormalized -> cv::pyrDown -> fromCVMat, against
// B = native pyrDownFiltered<Gaussian5x5, 8, 8> (X-46: 7094 us). Rule and bands
// are pre-registered in EXPERIMENTS.md X-47.
//
// The per-direction conversion cost is ALSO the general answer: any operation's
// interop decision is (native_binCV - native_OpenCV) against that tax, so the
// tax is timed at N = 8 and N = 3 rather than tabulating every operation.
//
// ONE ARM PER PROCESS, selected by argv[1] -- X-46's method note: its first
// version held every arm's working set resident at once, pumped ~1.4 MB through
// a 1 MB L2 between samples, and inflated the cheap arms threefold. The caller
// loops: ./scripts/run_on_pi.sh pi4 'bash ../benchmark/interop_sweep.sh'
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bincv-cpp/ops/pyramid.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "measure_util.hpp"

using W = uint32_t;
using bincv::PyrDownFilter;

namespace {
constexpr int kW = 640, kH = 480, kDW = 320, kDH = 240;

template <size_t N>
void fillQuant(bincv::QuantMat<N, W>& m) {
    constexpr unsigned maxV = (1u << N) - 1u;
    for (int y = 0; y < m.rows(); ++y)
        for (int x = 0; x < m.cols(); ++x)
            m.set(y, x, static_cast<unsigned>(x * 7 + y * 13) % (maxV + 1u));
}

void fillMat(cv::Mat& m) {
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.cols; ++x)
            m.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 7 + y * 13) % 256);
}

void report(int arm, const char* name, double ns) {
    std::printf("ARM %2d  %-38s %10.1f us\n", arm, name, ns / 1000.0);
}
} // namespace

int main(int argc, char** argv) {
    const int arm = argc > 1 ? std::atoi(argv[1]) : -1;
    cv::setNumThreads(1);
    switch (arm) {
        case 0: {  // export, the interop configuration
            bincv::QuantMat<8, W> q(kW, kH);
            fillQuant<8>(q);
            cv::Mat out;
            std::vector<measure::Bench> b = {{"toCVMatNormalized  N=8 640x480", [&](int) {
                                                  q.toCVMatNormalized(out);
                                                  measure::g_sink += out.data[0];
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 1: {  // import
            cv::Mat in(kH, kW, CV_8U);
            fillMat(in);
            bincv::QuantMat<8, W> q;
            std::vector<measure::Bench> b = {{"fromCVMat          N=8 640x480", [&](int) {
                                                  q.fromCVMat(in);
                                                  measure::g_sink += q.at(0, 0);
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 2: {  // R: the whole round trip X-47's bands are written on
            bincv::QuantMat<8, W> src(kW, kH), dst;
            fillQuant<8>(src);
            cv::Mat wide, down;
            std::vector<measure::Bench> b = {
                {"ROUND TRIP to+cv::pyrDown+from 8->8", [&](int) {
                     src.toCVMatNormalized(wide);
                     cv::pyrDown(wide, down);
                     dst.fromCVMat(down);
                     measure::g_sink += dst.at(0, 0);
                 }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 3: {  // B: the native bit-sliced arm the round trip is judged against
            bincv::QuantMat<8, W> src(kW, kH), dst(kDW, kDH);
            fillQuant<8>(src);
            std::vector<measure::Bench> b = {
                {"native GAUSSIAN_5x5 8->8 (X-46's B)", [&](int) {
                     bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 8, 8, W>(src, dst);
                     measure::g_sink += dst.at(0, 0);
                 }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 4: {  // the floor
            cv::Mat in(kH, kW, CV_8U), out;
            fillMat(in);
            std::vector<measure::Bench> b = {{"cv::pyrDown 8U (floor)", [&](int) {
                                                  cv::pyrDown(in, out);
                                                  measure::g_sink += out.data[0];
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 5: {  // the tax at a width binCV pipelines actually hold
            bincv::QuantMat<3, W> q(kW, kH);
            fillQuant<3>(q);
            cv::Mat out;
            std::vector<measure::Bench> b = {{"toCVMatNormalized  N=3 640x480", [&](int) {
                                                  q.toCVMatNormalized(out);
                                                  measure::g_sink += out.data[0];
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        case 6: {
            cv::Mat in(kH, kW, CV_8U);
            fillMat(in);
            bincv::QuantMat<3, W> q;
            std::vector<measure::Bench> b = {{"fromCVMat          N=3 640x480", [&](int) {
                                                  q.fromCVMat(in);
                                                  measure::g_sink += q.at(0, 0);
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0].medianNs);
            break;
        }
        default:
            std::printf("usage: interop_roundtrip <arm 0..6>\n");
            return 2;
    }
    return 0;
}
