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
// nothing at that scale. If binCV comes out ahead on time it will be because it does
// not refit over the consensus set and OpenCV does, which is a difference in what the
// two compute rather than in how fast they compute it. That is stated here so the
// timing column is read as the secondary one it is.
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
// `operator new` is replaced in this translation unit. That replacement is global --
// it binds for the whole program, OpenCV's own allocations included -- so the counts
// below are readings rather than estimates from what a call is known to materialize.
// Both sides are measured the same way, with the same counter, around one call.
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
#include <new>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv-cpp/ops/ransac.hpp"
#include "measure_util.hpp"

namespace {

std::size_t g_newCount = 0;
std::size_t g_newBytes = 0;

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

struct Counted {
    std::size_t calls = 0;
    std::size_t bytes = 0;
};

/// @brief Allocation performed by one call, read rather than accounted.
template <typename F>
Counted countOneCall(F&& f) {
    const std::size_t c0 = g_newCount, b0 = g_newBytes;
    f();
    return Counted{g_newCount - c0, g_newBytes - b0};
}

void runSize(size_t count, int outlierPct) {
    Scene s = makeScene(count, outlierPct, 0xA11CE + count);

    std::vector<uint8_t> best(count), cand(count), mask(count);
    const bincv::RansacScratch scratch{best.data(), cand.data(), count};
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
    const Counted binAlloc = countOneCall([&]() {
        bincv::Affine2D m2;
        const bincv::RansacResult rr = bincv::estimateAffine2D(s.from.data(), s.to.data(), count,
                                                               p, scratch, &m2, mask.data());
        measure::g_sink += rr.inliers;
    });
    const Counted cvAlloc = countOneCall([&]() {
        std::vector<uint8_t> mk;
        const cv::Mat mm = cv::estimateAffine2D(s.cvFrom, s.cvTo, mk, cv::RANSAC, kThreshold);
        measure::g_sink += static_cast<size_t>(mm.rows) + mk.size();
    });

    // TWO DIFFERENT QUANTITIES, AND CONFLATING THEM OVERSTATES THE RESULT. binCV's
    // allocator traffic is zero; its MEMORY is not. The working set moved to the
    // caller, it did not disappear, and the comparison worth making is working set
    // against working set.
    const size_t binSet = bincv::ransacScratchBytes(count);
    std::printf("\n WORKING SET OF ONE CALL\n");
    std::printf("   binCV  %9zu B   caller-owned, known before the call via"
                " ransacScratchBytes()\n", binSet);
    std::printf("   OpenCV %9zu B   allocated internally; not visible from its signature\n",
                cvAlloc.bytes);
    std::printf("   -> %.1fx smaller, and it is a number a caller can budget rather than"
                " discover\n", static_cast<double>(cvAlloc.bytes) / static_cast<double>(binSet));
    std::printf("\n ALLOCATOR TRAFFIC DURING ONE CALL (operator new replaced; both sides,"
                " same counter)\n");
    std::printf("   binCV  %6zu calls   the caller's buffer is reused across frames\n",
                binAlloc.calls);
    std::printf("   OpenCV %6zu calls   per call, so ~%zu/second in a 20 Hz frame loop\n",
                cvAlloc.calls, cvAlloc.calls * 20);
    std::printf("   A SEPARATE CLAIM from the one above: this is about jitter and allocator\n"
                "   pressure, not about how many bytes are live.\n");

    // --- time ----------------------------------------------------------------
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
    const std::vector<measure::Timing> t = measure::measureInterleaved(benches, 7, 120.0);

    std::printf("\n %-24s %12s %8s %11s\n", "variant", "us/call", "spread", "vs OpenCV");
    for (size_t i = 0; i < benches.size(); ++i) {
        std::printf(" %-24s %12.2f %7.1f%% %10.2fx\n", benches[i].name.c_str(),
                    t[i].medianNs / 1000.0, t[i].spreadPct(), t[1].medianNs / t[i].medianNs);
    }
    std::printf(" NOTE: binCV returns the minimal-set fit; OpenCV refits over its consensus\n"
                " set. The two do not compute the same last step, which is a difference in\n"
                " WHAT is computed and is why the timing column is the secondary one here.\n");
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

    runSize(200, 30);
    runSize(1000, 30);
    runSize(2000, 50);

    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}

// ---------------------------------------------------------------------------
// Replaced globally, so OpenCV's own allocations are counted too. Every form,
// including the nothrow ones -- a partial replacement would leave a mismatch.
// ---------------------------------------------------------------------------
namespace {
void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    g_newBytes += bytes;
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}
void* countedAllocateNoThrow(std::size_t bytes) noexcept {
    ++g_newCount;
    g_newBytes += bytes;
    return std::malloc(bytes == 0 ? 1 : bytes);
}
void countedFree(void* p) noexcept { std::free(p); }
} // namespace

void* operator new(std::size_t bytes) { return countedAllocate(bytes); }
void* operator new[](std::size_t bytes) { return countedAllocate(bytes); }
void operator delete(void* p) noexcept { countedFree(p); }
void operator delete[](void* p) noexcept { countedFree(p); }
void operator delete(void* p, std::size_t) noexcept { countedFree(p); }
void operator delete[](void* p, std::size_t) noexcept { countedFree(p); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    return countedAllocateNoThrow(bytes);
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    return countedAllocateNoThrow(bytes);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { countedFree(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { countedFree(p); }
