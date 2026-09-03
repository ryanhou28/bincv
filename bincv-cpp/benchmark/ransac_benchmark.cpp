// RANSAC against cv::estimateAffine2D -- and the column that matters is memory.
//
// THE DENOMINATOR (CLAUDE.md): `cv::estimateAffine2D(src, dst, mask, cv::RANSAC,
// threshold)` on the SAME correspondences with the SAME threshold. That is what a
// caller runs today without binCV.
//
// WHAT THIS FILE EXPECTS TO SHOW, WRITTEN BEFORE MEASURING
//
// Not a speedup. RANSAC's cost is the minimal solver -- dense floating-point work on
// a 3x3 -- and bit packing has nothing to say about it. The one structurally
// bit-parallel step is counting inliers, over a few hundred flags: 25-250 bytes,
// resident in L1 either way, which this project has already measured to be worth
// nothing at that scale. That is stated here so the timing column is read as the
// secondary one it is.
//
// BOTH SIDES NOW REFIT, so the timing column is a like-for-like one. binCV used to
// return the minimal-set fit while cv::estimateAffine2D refined over its consensus
// set, which made it faster partly by doing less -- and, more seriously, left its
// model 13x further from a planted transform at 0.5 px of inlier noise. The driver
// refits by default now, the arm is switchable through RansacParams::refine, and
// this benchmark times the default.
//
// AND THE HEADLINE RATIO IS NOT ABOUT THE RANSAC. Measured with the arms below, the
// two search loops are within about 1.07x of each other at 1 000 correspondences.
// Effectively the whole difference is the REFIT: OpenCV runs up to `refineIters`
// Levenberg-Marquardt steps, and binCV solves the same least-squares problem in
// closed form because an affine transform is linear in its six parameters. The
// same split governs the memory -- 80% of OpenCV's peak live heap here is the
// refinement, not the search.
//
// This decomposition is printed rather than described, because "binCV's RANSAC is
// 10x faster" is a false reading of the default-versus-default number and would be
// the natural one to take from a single ratio.
//
// The primary columns are WORKING SET and ALLOCATOR TRAFFIC, and they are two claims
// rather than one. binCV's working set is `ransacScratchBytes(n)` -- two flags per
// correspondence -- which is smaller than OpenCV's and, more usefully, knowable before
// the call rather than during it. Separately, binCV makes no allocator calls at all,
// because that buffer is the caller's and is reused across frames. Reporting the second
// as though it were the first would say binCV uses no memory, which is false: the
// memory moved to the caller, it did not disappear.
//
// HOW THE MEMORY COLUMN IS MEASURED, NOT ACCOUNTED
//
// heap_probe interposes the C allocator, so it sees every path into the heap --
// including `cv::fastMalloc`, which `cv::Mat` uses and which a replaced
// `operator new` never observes. This file used to replace `operator new` and so
// under-reported OpenCV badly; see heap_probe.hpp for the size of that error.
//
// The figure is the HIGH-WATER of simultaneously-live bytes, not the sum of every
// allocation. The sum counts a buffer that was already handed back, and OpenCV
// allocates and releases repeatedly inside its loop, so summing overstates it.
// Peak live and allocator traffic are reported as two rows because they are two
// claims.
//
// VALIDITY
//
// measure_util.hpp's protocol: volatile sink, calibrated batches, interleaved
// variants, spread beside the median. The two sides are compared for agreement
// BEFORE anything is timed -- not model against model, which would compare two random
// draws, but inlier count against inlier count on the same data.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv-cpp/ops/ransac.hpp"
#include "heap_probe.hpp"
#include "measure_util.hpp"

namespace {

using bincv::Affine2D;
using bincv::Point2f;

constexpr double kThreshold = 3.0;

struct Scene {
    std::vector<Point2f> from, to;
    std::vector<cv::Point2f> cvFrom, cvTo;
    size_t inliers = 0;
};

/// @brief `count` correspondences under a fixed affine, with `outlierPct` displaced
/// far enough to be unambiguous. The same content feeds both sides.
Scene makeScene(size_t count, int outlierPct, uint64_t seed) {
    const Affine2D t{{1.10f, -0.20f, 12.0f, 0.15f, 0.95f, -7.0f}};
    Scene s;
    uint64_t st = seed;
    auto nextf = [&st]() {
        st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return static_cast<float>((st >> 33) % 100000) / 100000.0f;
    };
    for (size_t i = 0; i < count; ++i) {
        const Point2f a{nextf() * 640.0f, nextf() * 480.0f};
        Point2f b{t.m[0] * a.x + t.m[1] * a.y + t.m[2], t.m[3] * a.x + t.m[4] * a.y + t.m[5]};
        const bool outlier = static_cast<int>(i % 100) < outlierPct;
        if (outlier) {
            b.x += 60.0f + nextf() * 120.0f;
            b.y -= 55.0f + nextf() * 110.0f;
        } else {
            ++s.inliers;
        }
        s.from.push_back(a);
        s.to.push_back(b);
        s.cvFrom.push_back(cv::Point2f(a.x, a.y));
        s.cvTo.push_back(cv::Point2f(b.x, b.y));
    }
    return s;
}

void runSize(size_t count, int outlierPct) {
    Scene s = makeScene(count, outlierPct, 0xA11CE + count);

    std::vector<uint32_t> flags(2 * bincv::ransacScratchWords(count));
    std::vector<uint8_t> mask(count);
    const bincv::RansacScratch scratch{flags.data(), count};
    bincv::RansacParams p;
    p.threshold = kThreshold;
    Affine2D model;

    std::vector<uint8_t> cvMask;

    // --- agreement, before anything is timed ---------------------------------
    const bincv::RansacResult r =
        bincv::estimateAffine2D(s.from.data(), s.to.data(), count, p, scratch, &model,
                                mask.data());
    const cv::Mat cvModel = cv::estimateAffine2D(s.cvFrom, s.cvTo, cvMask, cv::RANSAC, kThreshold);
    size_t cvInliers = 0;
    for (uint8_t v : cvMask) cvInliers += (v != 0) ? 1u : 0u;

    std::printf("\n================ %zu correspondences, %d%% outliers ================\n",
                count, outlierPct);
    std::printf(" planted inliers %zu | binCV %zu (%d iterations) | OpenCV %zu\n",
                s.inliers, r.inliers, r.iterations, cvInliers);
    if (!r.found || cvModel.empty()) {
        std::printf(" ONE SIDE FOUND NO MODEL -- no ratio below is meaningful.\n");
        return;
    }

    // --- allocation, read from a replaced operator new ------------------------
    const heapprobe::Reading binAlloc = heapprobe::around([&]() {
        bincv::Affine2D m2;
        const bincv::RansacResult rr = bincv::estimateAffine2D(s.from.data(), s.to.data(), count,
                                                               p, scratch, &m2, mask.data());
        measure::g_sink += rr.inliers;
    });
    // OpenCV with its refinement disabled, so the heap figure can be attributed the
    // same way the timing is: how much of it is the search, and how much the refit.
    const heapprobe::Reading cvSearchOnly = heapprobe::around([&]() {
        std::vector<uint8_t> mk;
        const cv::Mat mm = cv::estimateAffine2D(s.cvFrom, s.cvTo, mk, cv::RANSAC, kThreshold,
                                                2000, 0.99, 0);
        measure::g_sink += static_cast<size_t>(mm.rows) + mk.size();
    });
    const heapprobe::Reading cvAlloc = heapprobe::around([&]() {
        std::vector<uint8_t> mk;
        const cv::Mat mm = cv::estimateAffine2D(s.cvFrom, s.cvTo, mk, cv::RANSAC, kThreshold);
        measure::g_sink += static_cast<size_t>(mm.rows) + mk.size();
    });

    // TWO DIFFERENT QUANTITIES, AND CONFLATING THEM OVERSTATES THE RESULT. binCV's
    // allocator traffic is zero; its MEMORY is not. The working set moved to the
    // caller, it did not disappear, and the comparison worth making is working set
    // against working set.
    const size_t binSet = bincv::ransacScratchBytes(count);
    std::printf("\n PEAK SIMULTANEOUSLY-LIVE HEAP OF ONE CALL"
                " (allocator-level; sees cv::fastMalloc)\n");
    std::printf("   binCV  %9zu B   plus %zu B of caller-owned scratch, which is known\n"
                "                     before the call via ransacScratchBytes()\n",
                binAlloc.peakLive, binSet);
    std::printf("   OpenCV %9zu B   allocated internally; not visible from its signature\n",
                cvAlloc.peakLive);
    std::printf("   -> %.1fx smaller, counting binCV's scratch against OpenCV's peak live\n",
                static_cast<double>(cvAlloc.peakLive) / static_cast<double>(binSet));
    std::printf("   ATTRIBUTED: %zu B of OpenCV's peak is the SEARCH and %zu B is the REFIT\n"
                "   (its refineIters=0 arm, on the same counter). Most of the ratio above is\n"
                "   the refinement, which binCV does in closed form and on the stack -- it is\n"
                "   not a leaner RANSAC.\n",
                cvSearchOnly.peakLive,
                cvAlloc.peakLive > cvSearchOnly.peakLive
                    ? cvAlloc.peakLive - cvSearchOnly.peakLive : size_t{0});
    std::printf("   binCV's own stack is measured by essential_stack_benchmark, not here:\n"
                "   192 B on x86-64 against OpenCV's 11 440 B. A heap-only figure would\n"
                "   flatter a library whose rule is that kernels do not allocate.\n");
    std::printf("\n ALLOCATOR TRAFFIC DURING ONE CALL -- A SEPARATE CLAIM FROM THE ABOVE\n");
    std::printf("   binCV  %6zu calls   the caller's buffer is reused across frames\n",
                binAlloc.calls);
    std::printf("   OpenCV %6zu calls   per call, so ~%zu/second in a 20 Hz frame loop\n",
                cvAlloc.calls, cvAlloc.calls * 20);
    std::printf("   Peak live says how many bytes must exist at once; traffic says how\n"
                "   often the allocator is entered. Neither substitutes for the other.\n");

    // --- time ----------------------------------------------------------------
    // Four arms, so the ratio can be attributed rather than just reported: each
    // side with its refit on (the default a caller gets) and off (the search alone).
    std::vector<measure::Bench> benches;
    benches.push_back({"binCV", [&](int) {
                           Affine2D m2;
                           const bincv::RansacResult rr =
                               bincv::estimateAffine2D(s.from.data(), s.to.data(), count, p,
                                                       scratch, &m2, nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"cv::estimateAffine2D", [&](int) {
                           std::vector<uint8_t> mk;
                           const cv::Mat mm =
                               cv::estimateAffine2D(s.cvFrom, s.cvTo, mk, cv::RANSAC, kThreshold);
                           measure::g_sink += static_cast<size_t>(mm.rows);
                       }});
    benches.push_back({"binCV, search only", [&](int) {
                           bincv::RansacParams q = p;
                           q.refine = false;
                           Affine2D m2;
                           const bincv::RansacResult rr =
                               bincv::estimateAffine2D(s.from.data(), s.to.data(), count, q,
                                                       scratch, &m2, nullptr);
                           measure::g_sink += rr.inliers;
                       }});
    benches.push_back({"cv::, search only", [&](int) {
                           std::vector<uint8_t> mk;
                           const cv::Mat mm =
                               cv::estimateAffine2D(s.cvFrom, s.cvTo, mk, cv::RANSAC, kThreshold,
                                                    2000, 0.99, 0);
                           measure::g_sink += static_cast<size_t>(mm.rows);
                       }});

    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 7, 120.0);

    std::printf("\n %-24s %12s %8s %11s\n", "variant", "us/call", "spread", "vs OpenCV");
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf(" %-24s %12.2f %7.1f%% %10.2fx\n", benches[i].name.c_str(),
                    t[i].medianNs / 1000.0, t[i].spreadPct(), t[1].medianNs / t[i].medianNs);
    }
    std::printf(" READ THE LAST TWO ROWS BEFORE QUOTING THE FIRST TWO. Both sides refit, so\n"
                " this is like for like -- but the two SEARCH loops are close, and nearly all\n"
                " of the difference is the refit: OpenCV iterates Levenberg-Marquardt where\n"
                " binCV solves the same least squares in closed form, the model being linear\n"
                " in its parameters. \"binCV's RANSAC is faster\" is not what this shows.\n"
                " Accuracy against a planted transform is asserted in tests/test_ransac.cpp.\n");
}

} // namespace

int main() {
    cv::setNumThreads(1);
    std::printf("RANSAC -- bincv::estimateAffine2D against cv::estimateAffine2D(RANSAC)\n");
    std::printf("======================================================================\n");
    std::printf("OpenCV %s, cv::getNumThreads() = %d; binCV is single-threaded\n",
                CV_VERSION, cv::getNumThreads());
    std::printf("threshold %.1f px, confidence 0.99, iteration cap 2000, seeded and"
                " deterministic\n", kThreshold);

    // The instrument proves itself before any figure below is believed.
    std::printf("\n HEAP PROBE SELF-CHECK -- known answers it must recover\n");
    if (!heapprobe::selfCheck()) {
        std::printf("\n THE HEAP PROBE FAILED ITS OWN CHECKS. No figure below would be\n"
                    " evidence, so none is printed.\n");
        return 1;
    }

    runSize(200, 30);
    runSize(1000, 30);
    runSize(2000, 50);

    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}
