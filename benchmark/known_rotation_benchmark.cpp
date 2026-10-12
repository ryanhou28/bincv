// The essential matrix under a known rotation (two-point samples) against the five-point
// estimators a caller would otherwise run on the same correspondences.
//
// THE QUESTION: what does an essential matrix for this frame cost a caller who already
// knows the rotation -- from a gyroscope -- and what does it cost one who does not? The
// two estimators do not take the same input, so this is not like against like in
// algorithm. It is like against like in ROLE: the same correspondences, the same
// threshold and confidence, the same consensus rule, an `EssentialMatrix` out.
//
// THE DENOMINATORS: binCV's `findEssentialMat` and `findEssentialMatRealRoots`, and
// `cv::findEssentialMat(RANSAC)` -- the best existing option for that role on a host.
// (On M-profile the real-root solver is the best existing one; bincv_m7_essential's
// KNOWNROT command times the same comparison on the board.)
//
// WHAT DECIDES, WRITTEN BEFORE MEASURING
//
//   * ACCURACY FIRST, because a faster estimator that finds a different consensus is not
//     comparable on speed: each side's inlier count against the planted inliers, and the
//     translation's angle to the planted one, are printed before anything is timed.
//   * SPEED: the known-rotation estimator should be faster than every five-point side on
//     every case -- a hypothesis is cross products where theirs is a degree-10 solve, and
//     a two-point sample is clean more often. A case where it is not is a defect to find.
//   * MEMORY: nothing allocated (the heap probe below), and the same caller scratch,
//     `ransacScratchBytes(count)`, as the five-point estimators.
//
// The rotation is the planted one, EXACT. A real gyroscope's rotation carries error,
// which costs consensus rather than time; that cost is the caller's sensor's to measure,
// and nothing here pretends to.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv/ops/essential.hpp"
#include "bincv/ops/essentialKnownRotation.hpp"
#include "heap_probe.hpp"
#include "measure_util.hpp"

namespace {

using bincv::EssentialMatrix;
using bincv::Point2f;

constexpr double kThreshold = 0.002;

struct Scene {
    std::vector<Point2f> from, to;
    std::vector<cv::Point2f> cvFrom, cvTo;
    double r[9];
    double t[3];
    size_t inliers = 0;
};

/// essential_benchmark's scene, keeping the rotation and translation it plants.
Scene makeScene(size_t count, int outlierPct, uint64_t seed) {
    uint64_t s = seed;
    auto uni = [&s]() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((s >> 33) % 1000001) / 1000000.0) - 1.0;
    };
    Scene sc;
    const double ax = uni() * 0.4, ay = uni() * 0.4, az = uni() * 0.4;
    const double ca = std::cos(ax), sa = std::sin(ax), cb = std::cos(ay), sb = std::sin(ay),
                 cc = std::cos(az), ss = std::sin(az);
    const double R[9] = {cb * cc,
                         -cb * ss,
                         sb,
                         sa * sb * cc + ca * ss,
                         -sa * sb * ss + ca * cc,
                         -sa * cb,
                         -ca * sb * cc + sa * ss,
                         ca * sb * ss + sa * cc,
                         ca * cb};
    for (int i = 0; i < 9; ++i) sc.r[i] = R[i];
    double t[3] = {uni(), uni(), 1.0};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) sc.t[i] = t[i] / tn;

    for (size_t i = 0; i < count; ++i) {
        const double X[3] = {uni() * 2.0, uni() * 2.0, 4.0 + uni()};
        Point2f a{static_cast<float>(X[0] / X[2]), static_cast<float>(X[1] / X[2])};
        double Xc[3];
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += R[k * 3 + j] * X[j];
            Xc[k] = acc + sc.t[k];
        }
        Point2f b{static_cast<float>(Xc[0] / Xc[2]), static_cast<float>(Xc[1] / Xc[2])};
        if (static_cast<int>(i % 100) < outlierPct) {
            b.x += static_cast<float>(uni() * 0.4);
            b.y += static_cast<float>(uni() * 0.4);
        } else {
            ++sc.inliers;
        }
        sc.from.push_back(a);
        sc.to.push_back(b);
        sc.cvFrom.push_back(cv::Point2f(a.x, a.y));
        sc.cvTo.push_back(cv::Point2f(b.x, b.y));
    }
    return sc;
}

/// The angle, in degrees, between the planted translation and the one `e` carries
/// under the planted rotation (`E R^T` is proportional to `[t]x`), sign ignored.
double translationErrorDeg(const double* e, const Scene& s) {
    double m[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += e[i * 3 + k] * s.r[j * 3 + k];
            m[i * 3 + j] = acc;
        }
    }
    const double t[3] = {m[7] - m[5], m[2] - m[6], m[3] - m[1]};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    if (!(tn > 0.0)) return 180.0;
    const double c = std::fabs(t[0] * s.t[0] + t[1] * s.t[1] + t[2] * s.t[2]) / tn;
    return std::acos(c > 1.0 ? 1.0 : c) * 57.29577951308232;
}

void runSize(size_t count, int outlierPct) {
    const Scene s = makeScene(count, outlierPct, 0x5EED + count);
    std::vector<uint32_t> flags(2 * bincv::ransacScratchWords(count));
    std::vector<uint8_t> mask(count);
    const bincv::RansacScratch scratch{flags.data(), count};
    bincv::RansacParams p;
    p.threshold = kThreshold;
    p.maxIterations = 500;

    std::printf("\n================ %zu correspondences, %d%% outliers ================\n",
                count, outlierPct);
    EssentialMatrix eK, eKmin, e5, eRR;
    const bincv::RansacResult rK = bincv::essentialFromKnownRotation(
        s.from.data(), s.to.data(), count, s.r, p, scratch, &eK, mask.data());
    bincv::RansacParams pMin = p;
    pMin.refine = false;
    const bincv::RansacResult rKmin = bincv::essentialFromKnownRotation(
        s.from.data(), s.to.data(), count, s.r, pMin, scratch, &eKmin, nullptr);
    const bincv::RansacResult r5 =
        bincv::findEssentialMat(s.from.data(), s.to.data(), count, p, scratch, &e5, nullptr);
    const bincv::RansacResult rRR = bincv::findEssentialMatRealRoots(
        s.from.data(), s.to.data(), count, p, scratch, &eRR, nullptr);
    cv::Mat cvMask;
    const cv::Mat cvE = cv::findEssentialMat(s.cvFrom, s.cvTo, 1.0, cv::Point2d(0, 0), cv::RANSAC,
                                             0.99, kThreshold, cvMask);
    size_t cvInliers = 0;
    for (int i = 0; i < cvMask.rows; ++i) cvInliers += cvMask.at<uint8_t>(i, 0) != 0 ? 1u : 0u;
    double cvAngle = 180.0;
    if (!cvE.empty()) {
        double e[9];
        for (int i = 0; i < 9; ++i) e[i] = cvE.at<double>(i / 3, i % 3);
        cvAngle = translationErrorDeg(e, s);
    }

    std::printf(" planted inliers %zu\n", s.inliers);
    std::printf(" %-30s %8s %11s %16s\n", "estimator", "inliers", "iterations",
                "translation, deg");
    std::printf(" %-30s %8zu %11d %16.4f\n", "known rotation, refit", rK.inliers,
                rK.iterations, translationErrorDeg(eK.m, s));
    std::printf(" %-30s %8zu %11d %16.4f\n", "known rotation, no refit", rKmin.inliers,
                rKmin.iterations, translationErrorDeg(eKmin.m, s));
    std::printf(" %-30s %8zu %11d %16.4f\n", "binCV five-point", r5.inliers, r5.iterations,
                translationErrorDeg(e5.m, s));
    std::printf(" %-30s %8zu %11d %16.4f\n", "binCV five-point, real roots", rRR.inliers,
                rRR.iterations, translationErrorDeg(eRR.m, s));
    std::printf(" %-30s %8zu %11s %16.4f\n", "cv::findEssentialMat", cvInliers, "--", cvAngle);
    if (!rK.found || !r5.found || cvE.empty()) {
        std::printf(" A SIDE FOUND NO MODEL -- no ratio below is meaningful.\n");
        return;
    }

    const heapprobe::Reading heap = heapprobe::around([&]() {
        EssentialMatrix m2;
        const bincv::RansacResult rr = bincv::essentialFromKnownRotation(
            s.from.data(), s.to.data(), count, s.r, p, scratch, &m2, mask.data());
        measure::g_sink += rr.inliers;
    });
    std::printf("\n heap during one known-rotation call: %zu B peak live, %zu calls; caller "
                "scratch %zu B, as the five-point's\n",
                heap.peakLive, heap.calls, bincv::ransacScratchBytes(count));

    std::vector<measure::Bench> benches;
    benches.push_back({"known rotation, refit", [&](int) {
                           EssentialMatrix m2;
                           const bincv::RansacResult rr = bincv::essentialFromKnownRotation(
                               s.from.data(), s.to.data(), count, s.r, p, scratch, &m2, nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"known rotation, no refit", [&](int) {
                           EssentialMatrix m2;
                           const bincv::RansacResult rr = bincv::essentialFromKnownRotation(
                               s.from.data(), s.to.data(), count, s.r, pMin, scratch, &m2,
                               nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"binCV five-point", [&](int) {
                           EssentialMatrix m2;
                           const bincv::RansacResult rr = bincv::findEssentialMat(
                               s.from.data(), s.to.data(), count, p, scratch, &m2, nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"binCV five-point, real roots", [&](int) {
                           EssentialMatrix m2;
                           const bincv::RansacResult rr = bincv::findEssentialMatRealRoots(
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
    std::printf("\n %-30s %12s %8s %18s\n", "variant", "us/call", "spread",
                "known rotation is");
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf(" %-30s %12.2f %7.1f%% %17.1fx\n", benches[i].name.c_str(),
                    t[i].medianNs / 1000.0, t[i].spreadPct(), t[i].medianNs / t[0].medianNs);
    }
}

} // namespace

int main() {
    cv::setNumThreads(1);
    std::printf("Essential matrix under a known rotation -- two-point samples against the "
                "five-point estimators\n");
    std::printf("====================================================================="
                "=================\n");
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n",
                CV_VERSION, cv::getNumThreads());
    std::printf("normalised coordinates, threshold %.4f, confidence 0.99, cap 500; the "
                "rotation is the planted one, exact\n",
                kThreshold);
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
