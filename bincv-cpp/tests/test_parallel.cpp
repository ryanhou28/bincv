// The parallel-for customisation point and the optional pool.
//
// BIT-EXACTNESS IS THE PRECONDITION, NOT A GOAL. Keypoints are independent -- each
// iteration writes only nextPts[p], status[p] and err[p], and reads only const views
// -- so splitting the point array cannot move a flow vector. If it ever does, the
// split has a data race and any timing measured through it is meaningless. made
// that a precondition of the experiment; this file is where it is enforced.
//
// The pool itself is NOT part of bincv_core: core is allocation-free and builds
// -fno-exceptions, and std::thread is usable under neither. This test therefore lives
// on the OpenCV/hosted side of the build, and the serial default is what the three
// core-only configurations exercise.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "bincv-cpp/threads/pool.hpp"
#include "test_util.hpp"

namespace {
using namespace bincv;
}

BINCV_TEST(Parallel, SerialIsTheDefault) {
    // A core-only build has no backend, so parallelFor must run straight through.
    // This is not a formality: every kernel using it has to be correct that way,
    // because on an embedded target it is the only way it ever runs.
    BINCV_CHECK(getNumThreads() == 1);
    size_t seen = 0;
    parallelFor(37, [&](size_t) { ++seen; });
    BINCV_CHECK(seen == 37);
}

BINCV_TEST(Parallel, PoolRunsEveryIndexExactlyOnce) {
    constexpr size_t kN = 5000;
    std::vector<int> hits(kN, 0);
    {
        ThreadPool pool(4);
        pool.install();
        BINCV_CHECK(getNumThreads() > 1);
        parallelFor(kN, [&](size_t i) { hits[i] += 1; });
    }
    // The pool un-installs itself on destruction, so a later call is serial again --
    // otherwise a dangling backend pointer would outlive the threads it dispatches to.
    BINCV_CHECK(getNumThreads() == 1);
    size_t wrong = 0;
    for (int h : hits) if (h != 1) ++wrong;
    std::printf(" every index once: %zu of %zu wrong\n", wrong, kN);
    BINCV_CHECK(wrong == 0);
}

BINCV_TEST(Parallel, TrackerIsBitExactWithAPoolInstalled) {
    // The claim that matters. Same frames, same points, same parameters; the only
    // difference is whether a backend is installed.
    //
    // Self-contained on purpose -- test_opticalflow.cpp's frontend helpers live in its
    // anonymous namespace, and a copy of them here would drift. What this needs is a
    // level with real structure and points inside it, which is a dozen lines.
    constexpr int kW = 200, kH = 160;
    QuantMat<1, uint32_t> prev(kW, kH), next(kW, kH);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            // Textured enough that the gradient covariance is non-degenerate, and
            // shifted by one column between frames so there is a flow to find.
            const unsigned ux = static_cast<unsigned>(x), uy = static_cast<unsigned>(y);
            const unsigned v = ((ux / 3u + uy / 5u) % 2u) ^ ((ux * uy) % 7u == 0u ? 1u : 0u);
            prev.set(y, x, v);
            const int xs = x + 1 < kW ? x + 1 : x;
            const unsigned uxs = static_cast<unsigned>(xs);
            const unsigned w = ((uxs / 3u + uy / 5u) % 2u) ^ ((uxs * uy) % 7u == 0u ? 1u : 0u);
            next.set(y, x, w);
        }
    SignedQuantMat<1, uint32_t> dx(kW, kH), dy(kW, kH);
    derivativeX(prev, dx);
    derivativeY(prev, dy);
    const auto level = lkLevel<1>(prev, next, dx, dy);

    std::vector<Point2f> pts;
    for (int y = 40; y < kH - 40; y += 7)
        for (int x = 40; x < kW - 40; x += 9)
            pts.push_back(Point2f{static_cast<float>(x), static_cast<float>(y)});
    BINCV_CHECK(pts.size() >= 32);

    LKParams params;
    std::vector<Point2f> serial(pts.size()), threaded(pts.size());
    std::vector<uint8_t> sStat(pts.size()), tStat(pts.size());
    std::vector<float> sErr(pts.size()), tErr(pts.size());

    calcOpticalFlowPyrLK<1, uint32_t>(&level, 1, pts.data(), serial.data(), sStat.data(),
                                      sErr.data(), pts.size(), params);
    {
        ThreadPool pool(4);
        pool.install();
        BINCV_CHECK(getNumThreads() > 1);
        calcOpticalFlowPyrLK<1, uint32_t>(&level, 1, pts.data(), threaded.data(),
                                          tStat.data(), tErr.data(), pts.size(), params);
    }

    size_t diff = 0, tracked = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (sStat[i]) ++tracked;
        // EXACT equality on floats, deliberately. Every point takes the same
        // arithmetic in the same order whichever thread runs it, so a tolerance here
        // would hide precisely the reordering this test exists to forbid.
        if (serial[i].x != threaded[i].x || serial[i].y != threaded[i].y ||
            sStat[i] != tStat[i] || sErr[i] != tErr[i]) ++diff;
    }
    std::printf(" %zu points (%zu tracked), %zu differ between serial and 4 threads\n",
                pts.size(), tracked, diff);
    // A test where nothing tracked would compare zeros and pass for the wrong reason.
    BINCV_CHECK(tracked > 0);
    BINCV_CHECK(diff == 0);
}

BINCV_TEST_MAIN("test_parallel")
