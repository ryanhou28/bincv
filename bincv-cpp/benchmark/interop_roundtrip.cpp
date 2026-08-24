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

/// Reports the SPREAD next to the central value -- measure_util.hpp's hazard 3,
/// which an earlier version of this file violated by printing only the median. With
/// one arm per process there is no interleaving left to absorb drift, so the printed
/// spread is the only within-run error bar this design can produce.
void report(int arm, const char* name, const measure::Timing& t) {
    std::printf("ARM %2d  %-38s %9.1f us  [%9.1f .. %9.1f, spread %4.1f%%]\n", arm, name,
                t.medianNs / 1000.0, t.minNs / 1000.0, t.maxNs / 1000.0, t.spreadPct());
}

/// Peak live bytes an arm holds, computed rather than measured -- CLAUDE.md requires
/// memory and speed together, and this experiment settles a memory/speed trade.
void reportBytes(const char* name, size_t bytes) {
    std::printf("BYTES   %-38s %9zu B\n", name, bytes);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
            break;
        }
        case 4: {  // the floor
            cv::Mat in(kH, kW, CV_8U), out;
            fillMat(in);
            std::vector<measure::Bench> b = {{"cv::pyrDown 8U (floor)", [&](int) {
                                                  cv::pyrDown(in, out);
                                                  measure::g_sink += out.data[0];
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
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
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
            break;
        }
        case 7: {  // T's decimated half: the import R actually runs, at 320x240.
            cv::Mat in(kDH, kDW, CV_8U);
            fillMat(in);
            bincv::QuantMat<8, W> q;
            std::vector<measure::Bench> b = {{"fromCVMat          N=8 320x240", [&](int) {
                                                  q.fromCVMat(in);
                                                  measure::g_sink += q.at(0, 0);
                                              }}};
            report(arm, b[0].name.c_str(), measure::measureInterleaved(b, 9, 60.0)[0]);
            break;
        }
        case 8: {
            // AGREEMENT, and the honest answer is that the two arms DISAGREE.
            // measure_util.hpp hazard 4 requires whatever is compared to agree
            // before it is timed, and this file did not check. R runs cv::pyrDown
            // (BORDER_REFLECT_101); B runs pyrDownFiltered, whose route reads
            // outside the frame as ZERO (ops/pyramid.hpp). They therefore differ on
            // a 2-pixel rim BY CONSTRUCTION. Quantified here rather than asserted,
            // because "3.7x faster" means nothing without knowing the substitute
            // computes a different answer -- and where.
            bincv::QuantMat<8, W> src(kW, kH), nat(kDW, kDH), viaCv;
            fillQuant<8>(src);
            bincv::pyrDownFiltered<PyrDownFilter::Gaussian5x5, 8, 8, W>(src, nat);
            cv::Mat wide, down;
            src.toCVMatNormalized(wide);
            cv::pyrDown(wide, down);
            viaCv.fromCVMat(down);
            size_t diffAll = 0, diffInterior = 0, maxDelta = 0;
            for (int y = 0; y < kDH; ++y) {
                for (int x = 0; x < kDW; ++x) {
                    const unsigned a = nat.at(y, x), c = viaCv.at(y, x);
                    if (a == c) continue;
                    ++diffAll;
                    const size_t d = a > c ? a - c : c - a;
                    if (d > maxDelta) maxDelta = d;
                    if (y >= 2 && y < kDH - 2 && x >= 2 && x < kDW - 2) ++diffInterior;
                }
            }
            std::printf("AGREE   native vs round trip: %zu of %d pixels differ "
                        "(%zu of them interior), max |delta| %zu of 255\n",
                        diffAll, kDW * kDH, diffInterior, maxDelta);
            break;
        }
        case 9: {
            // PEAK WORKING SET, both paths, computed exactly. An earlier version of
            // this experiment reported speed only -- the same defect X-44's rule
            // carried -- while settling a trade whose whole point is that the interop
            // path MATERIALISES A BYTE-PER-PIXEL FRAME, which is what binCV exists
            // to avoid.
            const size_t srcQ = static_cast<size_t>(kW) * kH;       // 8 planes, 1 bit each
            const size_t dstQ = static_cast<size_t>(kDW) * kDH;
            reportBytes("native: src QuantMat<8> + dst", srcQ + dstQ);
            reportBytes("interop: + cv 8U in, cv 8U out", srcQ + dstQ + srcQ + dstQ);
            reportBytes("interop: + fromCVMat transient", srcQ + dstQ + srcQ + dstQ + dstQ);
            break;
        }
        default:
            std::printf("usage: interop_roundtrip <arm 0..9>\n");
            return 2;
    }
    return 0;
}
