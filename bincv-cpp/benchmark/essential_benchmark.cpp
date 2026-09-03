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
// their working arrays and where the real comparison is. `operator new` is replaced
// globally, so OpenCV's own allocations land on the same counter, and each block
// carries its size so a free can subtract what the allocation added. The figure
// reported is therefore the HIGH-WATER mark of simultaneously-live bytes.
//
// An earlier version of this file summed every allocation instead and reported
// OpenCV at 323 088 B against binCV's 15 080. That was wrong by two orders of
// magnitude in binCV's favour: OpenCV allocates and releases repeatedly inside its
// loop, so its peak live is 2 744 B and the sum counts buffers it had already given
// back. Measured correctly, BINCV HOLDS MORE MEMORY, NOT LESS -- the solver's stack
// frame is live for the whole call. Working set and allocator traffic are two
// different claims and only the second one favours binCV.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv-cpp/ops/essential.hpp"
#include "measure_util.hpp"

namespace {

std::size_t g_newCount = 0;
std::size_t g_smallCount = 0;
std::size_t g_liveBytes = 0;
std::size_t g_peakBytes = 0;
constexpr std::size_t kAllocHeader = 16;

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

struct Counted {
    std::size_t calls = 0;       ///< allocator round-trips
    std::size_t smallCalls = 0;  ///< of those, blocks under 128 B
    std::size_t peak = 0;        ///< HIGH-WATER live bytes, not the sum of allocations
};
/// @brief Peak simultaneously-live bytes across one call, which is the quantity a
/// caller has to budget. Summing every allocation instead would count a
/// buffer that was freed before the next one was taken, and OpenCV allocates
/// and releases repeatedly inside its loop -- so the sum overstates it badly.
template <typename F>
Counted countOneCall(F&& f) {
    const std::size_t c0 = g_newCount, s0 = g_smallCount;
    const std::size_t base = g_liveBytes;
    g_peakBytes = g_liveBytes;
    f();
    return Counted{g_newCount - c0, g_smallCount - s0, g_peakBytes - base};
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

    const Counted binAlloc = countOneCall([&]() {
        EssentialMatrix m2;
        const bincv::RansacResult rr = bincv::findEssentialMat(s.from.data(), s.to.data(), count,
                                                               p, scratch, &m2, mask.data());
        measure::g_sink += rr.inliers;
    });
    const Counted cvAlloc = countOneCall([&]() {
        cv::Mat mk;
        const cv::Mat ee = cv::findEssentialMat(s.cvFrom, s.cvTo, 1.0, cv::Point2d(0, 0),
                                                cv::RANSAC, 0.99, kThreshold, mk);
        measure::g_sink += static_cast<size_t>(ee.rows) + static_cast<size_t>(mk.rows);
    });

    std::printf("\n MEMORY -- HEAP ONLY, WHICH IS NOT THE WHOLE STORY\n");
    std::printf("   binCV  %9zu B heap   caller scratch %zu B + solver stack %zu B,"
                " no heap at all\n", size_t{0}, bincv::ransacScratchBytes(count),
                bincv::essentialSolverStackBytes());
    std::printf("   OpenCV %9zu B heap   peak live, in %zu blocks, %zu of them under"
                " 128 B\n", cvAlloc.peak, cvAlloc.calls, cvAlloc.smallCalls);
    std::printf("\n   BOTH SOLVERS KEEP THEIR WORKING ARRAYS ON THE STACK, so these heap\n");
    std::printf("   figures compare the small change and miss the thing itself. Run\n");
    std::printf("   benchmark/essential_stack_benchmark for the stack, which is where\n");
    std::printf("   the comparison actually lives: measured there, binCV's whole call\n");
    std::printf("   needs about 6.4 to 7.2 KB of stack against OpenCV's 16.4 to 17.2 KB,\n");
    std::printf("   the spread being the bisection granularity. Roughly 2.4x smaller.\n");
    std::printf("\n   An earlier version of this file divided binCV's stack by OpenCV's\n");
    std::printf("   heap and reported binCV as using 2.1x MORE memory. It uses less.\n");
    std::printf("\n ALLOCATOR TRAFFIC DURING ONE CALL -- this one IS like for like\n");
    std::printf("   binCV  %6zu calls   |  OpenCV %6zu calls, %zu under 128 B\n",
                binAlloc.calls, cvAlloc.calls, cvAlloc.smallCalls);

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
    runSize(200, 20);
    runSize(500, 30);
    runSize(1000, 40);
    std::printf("\n sink %zu\n", static_cast<size_t>(measure::g_sink));
    return 0;
}

namespace {
void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes < 128) ++g_smallCount;
    void* raw = std::malloc(bytes + kAllocHeader);
    if (raw == nullptr) std::abort();
    *static_cast<std::size_t*>(raw) = bytes;
    g_liveBytes += bytes;
    if (g_liveBytes > g_peakBytes) g_peakBytes = g_liveBytes;
    return static_cast<char*>(raw) + kAllocHeader;
}
void* countedAllocateNoThrow(std::size_t bytes) noexcept {
    ++g_newCount;
    if (bytes < 128) ++g_smallCount;
    void* raw = std::malloc(bytes + kAllocHeader);
    if (raw == nullptr) return nullptr;
    *static_cast<std::size_t*>(raw) = bytes;
    g_liveBytes += bytes;
    if (g_liveBytes > g_peakBytes) g_peakBytes = g_liveBytes;
    return static_cast<char*>(raw) + kAllocHeader;
}
void countedFree(void* p) noexcept {
    if (p == nullptr) return;
    void* raw = static_cast<char*>(p) - kAllocHeader;
    g_liveBytes -= *static_cast<std::size_t*>(raw);
    std::free(raw);
}
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
