// The five-point essential matrix against cv::findEssentialMat(RANSAC).
//
// THE DENOMINATOR (CLAUDE.md): `cv::findEssentialMat(pts1, pts2, focal=1,
// pp=(0,0), cv::RANSAC, confidence, threshold, mask)` on the SAME correspondences
// with the SAME threshold, in the same normalised coordinates. That is what a
// caller runs today without binCV, and it is the same minimal solver -- OpenCV's
// five-point core -- so this is like against like rather than one algorithm
// against another.
//
// WHAT IS BEING COMPARED, WRITTEN BEFORE MEASURING
//
// Both sides run a five-point solver inside a RANSAC loop. The costs that differ
// are the elimination (this file's degree-10 determinant against OpenCV's
// formulation) and the memory contract. The inlier count is checked before
// anything is timed, because two randomised searches that disagree about the
// consensus set are not comparable on speed.
//
// THE HEAP COLUMN IS PEAK LIVE, AND IT IS THE SMALLER HALF OF THE STORY --
// essential_stack_benchmark measures the stack, which is where both solvers keep
// their working arrays and where the real comparison is.
//
// The figure comes from heap_probe, which interposes the C allocator. It has to:
// a replaced `operator new` -- which this file used to use -- cannot see
// `cv::fastMalloc`, so it missed the matrix data and reported OpenCV at a flat
// 2 744 B at every input size. The true figure grows with the input, from 16 568 B
// at 200 correspondences to 84 952 B at 2 000. That error was 17x at 1 000 points
// and it ran AGAINST binCV, which allocates nothing at all.
//
// An earlier version summed every allocation instead of tracking the high-water of
// live bytes, and reported OpenCV at 323 088 B. Peak live and cumulative traffic are
// two different quantities and this file reports them as two rows, never as one.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv-cpp/ops/essential.hpp"
#include "heap_probe.hpp"
#include "measure_util.hpp"

namespace {

using bincv::EssentialMatrix;
using bincv::Point2f;

constexpr double kThreshold = 0.002;

struct Scene {
    std::vector<Point2f> from, to;
    std::vector<cv::Point2f> cvFrom, cvTo;
    size_t inliers = 0;
};

Scene makeScene(size_t count, int outlierPct, uint64_t seed) {
    uint64_t s = seed;
    auto uni = [&s]() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((s >> 33) % 1000001) / 1000000.0) - 1.0;
    };
    const double ax = uni() * 0.4, ay = uni() * 0.4, az = uni() * 0.4;
    const double ca = std::cos(ax), sa = std::sin(ax), cb = std::cos(ay), sb = std::sin(ay),
                 cc = std::cos(az), sc = std::sin(az);
    const double R[3][3] = {{cb * cc, -cb * sc, sb},
                            {sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb},
                            {-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb}};
    double t[3] = {uni(), uni(), 1.0};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) t[i] /= tn;

    Scene sc2;
    for (size_t i = 0; i < count; ++i) {
        const double X[3] = {uni() * 2.0, uni() * 2.0, 4.0 + uni()};
        Point2f a{static_cast<float>(X[0] / X[2]), static_cast<float>(X[1] / X[2])};
        double Xc[3];
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += R[k][j] * X[j];
            Xc[k] = acc + t[k];
        }
        Point2f b{static_cast<float>(Xc[0] / Xc[2]), static_cast<float>(Xc[1] / Xc[2])};
        if (static_cast<int>(i % 100) < outlierPct) {
            b.x += static_cast<float>(uni() * 0.4);
            b.y += static_cast<float>(uni() * 0.4);
        } else {
            ++sc2.inliers;
        }
        sc2.from.push_back(a);
        sc2.to.push_back(b);
        sc2.cvFrom.push_back(cv::Point2f(a.x, a.y));
        sc2.cvTo.push_back(cv::Point2f(b.x, b.y));
    }
    return sc2;
}

void runSize(size_t count, int outlierPct) {
    Scene s = makeScene(count, outlierPct, 0x5EED + count);
    std::vector<uint32_t> flags(2 * bincv::ransacScratchWords(count));
    std::vector<uint8_t> mask(count);
    const bincv::RansacScratch scratch{flags.data(), count};
    bincv::RansacParams p;
    p.threshold = kThreshold;
    p.maxIterations = 500;
    EssentialMatrix model;

    const bincv::RansacResult r = bincv::findEssentialMat(s.from.data(), s.to.data(), count, p,
                                                          scratch, &model, mask.data());
    cv::Mat cvMask;
    const cv::Mat cvE = cv::findEssentialMat(s.cvFrom, s.cvTo, 1.0, cv::Point2d(0, 0), cv::RANSAC,
                                             0.99, kThreshold, cvMask);
    size_t cvInliers = 0;
    for (int i = 0; i < cvMask.rows; ++i) cvInliers += cvMask.at<uint8_t>(i, 0) != 0 ? 1u : 0u;

    std::printf("\n================ %zu correspondences, %d%% outliers ================\n",
                count, outlierPct);
    std::printf(" planted inliers %zu | binCV %zu (%d iterations) | OpenCV %zu\n",
                s.inliers, r.inliers, r.iterations, cvInliers);
    if (!r.found || cvE.empty()) {
        std::printf(" ONE SIDE FOUND NO MODEL -- no ratio below is meaningful.\n");
        return;
    }

    const heapprobe::Reading binAlloc = heapprobe::around([&]() {
        EssentialMatrix m2;
        const bincv::RansacResult rr = bincv::findEssentialMat(s.from.data(), s.to.data(), count,
                                                               p, scratch, &m2, mask.data());
        measure::g_sink += rr.inliers;
    });
    const heapprobe::Reading cvAlloc = heapprobe::around([&]() {
        cv::Mat mk;
        const cv::Mat ee = cv::findEssentialMat(s.cvFrom, s.cvTo, 1.0, cv::Point2d(0, 0),
                                                cv::RANSAC, 0.99, kThreshold, mk);
        measure::g_sink += static_cast<size_t>(ee.rows) + static_cast<size_t>(mk.rows);
    });

    std::printf("\n PEAK SIMULTANEOUSLY-LIVE HEAP OF ONE CALL"
                " (allocator-level; sees cv::fastMalloc)\n");
    std::printf("   binCV  %9zu B   nothing at all; the caller's scratch is %zu B and is\n"
                "                     reused across frames\n",
                binAlloc.peakLive, bincv::ransacScratchBytes(count));
    std::printf("   OpenCV %9zu B   net %lld B at the end of the call\n",
                cvAlloc.peakLive, cvAlloc.net);
    std::printf("\n ALLOCATOR TRAFFIC DURING ONE CALL -- A SEPARATE CLAIM FROM THE ABOVE\n");
    std::printf("   binCV  %6zu calls   |  OpenCV %6zu calls, %zu under 128 B\n",
                binAlloc.calls, cvAlloc.calls, cvAlloc.smallCalls);
    std::printf("   Peak live says how many bytes must exist at once; traffic says how\n"
                "   often the allocator is entered. Neither substitutes for the other.\n");
    std::printf("\n   THE HEAP IS THE SMALLER HALF. Both solvers keep their working arrays\n");
    std::printf("   on the STACK; run benchmark/essential_stack_benchmark for that, where\n");
    std::printf("   the probe first has to recover two known answers before it reports.\n");

    std::vector<measure::Bench> benches;
    benches.push_back({"binCV five-point", [&](int) {
                           EssentialMatrix m2;
                           const bincv::RansacResult rr = bincv::findEssentialMat(
                               s.from.data(), s.to.data(), count, p, scratch, &m2, nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"cv::findEssentialMat", [&](int) {
                           cv::Mat mk;
                           const cv::Mat ee =
                               cv::findEssentialMat(s.cvFrom, s.cvTo, 1.0, cv::Point2d(0, 0),
                                                    cv::RANSAC, 0.99, kThreshold, mk);
                           measure::g_sink += static_cast<size_t>(ee.rows);
                       }});
    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 7, 150.0);
    std::printf("\n %-24s %12s %8s %11s\n", "variant", "us/call", "spread", "vs OpenCV");
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf(" %-24s %12.1f %7.1f%% %10.2fx\n", benches[i].name.c_str(),
                    t[i].medianNs / 1000.0, t[i].spreadPct(), t[1].medianNs / t[i].medianNs);
    }
}

} // namespace

int main() {
    cv::setNumThreads(1);
    std::printf("Five-point essential matrix -- binCV against cv::findEssentialMat(RANSAC)\n");
    std::printf("=========================================================================\n");
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n",
                CV_VERSION, cv::getNumThreads());
    std::printf("normalised coordinates, threshold %.4f, confidence 0.99, cap 500\n", kThreshold);

    // The instrument proves itself before any figure below is believed.
    std::printf("\n HEAP PROBE SELF-CHECK -- known answers it must recover\n");
    if (!heapprobe::selfCheck()) {
        std::printf("\n THE HEAP PROBE FAILED ITS OWN CHECKS. No figure below would be\n"
                    " evidence, so none is printed.\n");
        return 1;
    }

    runSize(200, 20);
    runSize(500, 30);
    runSize(1000, 40);
    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
