// HOW MUCH STACK DOES ONE CALL NEED? binCV against cv::findEssentialMat.
//
// WHY THIS FILE EXISTS. The memory comparison in essential_benchmark.cpp can only
// see the heap, because that is all a replaced operator new can see. Both solvers
// keep their working arrays on the STACK, so heap figures compare the small change
// and miss the thing itself -- and comparing binCV's stack against OpenCV's heap,
// which an earlier version of that file did, is not a measurement at all.
//
// -fstack-usage answers this for binCV and cannot answer it for a prebuilt
// libopencv. So: run the call on a thread with a bounded stack and find the
// smallest stack it survives. Each trial is a forked child, so an overflow is a
// failed trial rather than the end of the run.
//
// TWO PASSES, BECAUSE PTHREAD_STACK_MIN IS 16 KB AND binCV FITS UNDER IT. Bisecting
// the stack SIZE therefore cannot resolve binCV at all. The second pass fixes a
// generous stack and bisects the PADDING eaten before the call instead: the largest
// padding a workload tolerates is headroom it did not need, so the difference
// between two workloads' tolerated padding is the difference in what they used.
// The baseline row is an empty workload and is what the thread machinery itself
// costs; every figure is reported net of it.
//
// RUN IT MORE THAN ONCE. Consecutive runs move by about 512 B -- the bisection
// granularity plus scheduling -- so the figures are a range and not a point. Two
// runs here gave binCV 6 144 / 6 656 B and OpenCV 16 384 / 17 152 B.
//
// Resolution is the bisection granularity, 256 B. Cross-checked against
// -fstack-usage, which puts fivePointEssential's own frame at 5 136 B where this
// reports 6 656 B for the solver including its caller -- consistent.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include "bincv-cpp/ops/essential.hpp"
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {
const int N = 1000;
std::vector<cv::Point2f> g_ca, g_cb;
std::vector<bincv::Point2f> g_ba, g_bb;
std::vector<uint32_t> g_flags;
volatile int g_guard = 0;

void buildScene() {
    uint64_t s = 99;
    auto uni = [&]() { s = s*6364136223846793005ULL + 1442695040888963407ULL;
        return 2.0*(double((s>>33)%1000001)/1000000.0) - 1.0; };
    double t[3] = {0.3, 0.1, 1.0};
    const double tn = std::sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);
    for (int i = 0; i < 3; ++i) t[i] /= tn;
    for (int i = 0; i < N; ++i) {
        double X[3] = {uni()*2, uni()*2, 4+uni()};
        const float ax = float(X[0]/X[2]), ay = float(X[1]/X[2]);
        double Xc[3] = {X[0]+t[0], X[1]+t[1], X[2]+t[2]};
        float bx = float(Xc[0]/Xc[2]), by = float(Xc[1]/Xc[2]);
        if (i % 5 == 0) { bx += float(uni()*0.4); by += float(uni()*0.4); }
        g_ca.push_back(cv::Point2f(ax, ay)); g_cb.push_back(cv::Point2f(bx, by));
        g_ba.push_back({ax, ay}); g_bb.push_back({bx, by});
    }
    g_flags.assign(2 * bincv::ransacScratchWords(size_t(N)), 0);
}

void* wBaseline(void*) { g_guard += 1; return nullptr; }

void* wFivePoint(void*) {
    bincv::EssentialMatrix e[10];
    g_guard += bincv::fivePointEssential(g_ba.data(), g_bb.data(), e);
    return nullptr;
}

void* wBincvRansac(void*) {
    bincv::RansacScratch sc{g_flags.data(), size_t(N)};
    bincv::RansacParams p; p.threshold = 0.002; p.maxIterations = 500;
    bincv::EssentialMatrix e;
    g_guard += int(bincv::findEssentialMat(g_ba.data(), g_bb.data(), size_t(N),
                                           p, sc, &e).inliers);
    return nullptr;
}

void* wOpenCv(void*) {
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(g_ca, g_cb, 1.0, cv::Point2d(0,0),
                                     cv::RANSAC, 0.99, 0.002, mask);
    g_guard += E.rows;
    return nullptr;
}

// PTHREAD_STACK_MIN is 16 KB, and binCV's whole call fits under it -- so bisecting
// the STACK SIZE cannot resolve it. Bisect the PADDING consumed before the call
// instead, on a fixed generous stack: the largest padding a workload survives is
// the headroom it did not need, and the difference between two workloads' padding
// is the difference in what they used. That resolves below the floor.
constexpr size_t kFixedStack = 256u * 1024u;
size_t g_pad = 0;
void* (*g_inner)(void*) = nullptr;

void* wPadded(void* a) {
    volatile char* pad = static_cast<volatile char*>(__builtin_alloca(g_pad));
    for (size_t i = 0; i < g_pad; i += 512) pad[i] = 1;  // commit it
    void* r = g_inner(a);
    for (size_t i = 0; i < g_pad; i += 512) g_guard += pad[i];
    return r;
}

/// @brief true when the workload completes with `pad` bytes eaten before the call.
bool survivesPad(void* (*fn)(void*), size_t pad) {
    const pid_t pid = fork();
    if (pid == 0) {
        g_inner = fn;
        g_pad = pad;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        if (pthread_attr_setstacksize(&attr, kFixedStack) != 0) _exit(2);
        pthread_t th;
        if (pthread_create(&th, &attr, wPadded, nullptr) != 0) _exit(3);
        pthread_join(th, nullptr);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/// @brief Largest padding the workload tolerates. More padding tolerated means less
/// stack used by the workload itself.
size_t maxPad(void* (*fn)(void*)) {
    size_t lo = 0, hi = kFixedStack;
    while (hi - lo > 256) {
        const size_t mid = lo + (hi - lo) / 2;
        if (survivesPad(fn, mid)) lo = mid; else hi = mid;
    }
    return lo;
}

/// @brief true when the workload completes on a thread with `bytes` of stack.
bool survives(void* (*fn)(void*), size_t bytes) {
    const pid_t pid = fork();
    if (pid == 0) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        if (pthread_attr_setstacksize(&attr, bytes) != 0) _exit(2);
        pthread_t th;
        if (pthread_create(&th, &attr, fn, nullptr) != 0) _exit(3);
        pthread_join(th, nullptr);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

size_t bisect(void* (*fn)(void*), const char* name) {
    size_t lo = 16u * 1024u, hi = 16u * 1024u;
    while (hi <= 8u * 1024u * 1024u && !survives(fn, hi)) hi *= 2;
    if (hi > 8u * 1024u * 1024u) { std::printf("  %-34s  did not survive 8 MB\n", name); return 0; }
    while (hi - lo > 512) {
        const size_t mid = lo + (hi - lo) / 2;
        if (survives(fn, mid)) hi = mid; else lo = mid;
    }
    std::printf("  %-34s %8zu B\n", name, hi);
    return hi;
}
} // namespace

int main() {
    buildScene();
    cv::setNumThreads(1);
    // Warm both so lazy one-time initialisation is not charged to a trial.
    { cv::Mat m; cv::findEssentialMat(g_ca, g_cb, 1.0, cv::Point2d(0,0),
                                      cv::RANSAC, 0.99, 0.002, m); }
    { bincv::EssentialMatrix e[10]; bincv::fivePointEssential(g_ba.data(), g_bb.data(), e); }

    std::printf("Smallest thread stack the call survives, 1000 correspondences\n\n");
    const size_t base = bisect(wBaseline, "baseline (empty workload)");
    const size_t fp   = bisect(wFivePoint, "binCV fivePointEssential alone");
    const size_t br   = bisect(wBincvRansac, "binCV findEssentialMat (whole)");
    const size_t cv2  = bisect(wOpenCv, "cv::findEssentialMat (whole)");
    std::printf("\n  Net of baseline (limited by the 16 KB floor):\n");
    if (fp)  std::printf("    binCV five-point solver        %8zu B\n", fp - base);
    if (br)  std::printf("    binCV whole RANSAC call        %8zu B\n", br - base);
    if (cv2) std::printf("    OpenCV whole call              %8zu B\n", cv2 - base);

    std::printf("\nResolving below the floor: largest padding tolerated on a 256 KB stack\n");
    std::printf("(more padding tolerated == less stack used by the call itself)\n\n");
    const size_t pBase = maxPad(wBaseline);
    const size_t pFive = maxPad(wFivePoint);
    const size_t pBin  = maxPad(wBincvRansac);
    const size_t pCv   = maxPad(wOpenCv);
    std::printf("  %-34s %8zu B tolerated\n", "baseline (empty workload)", pBase);
    std::printf("  %-34s %8zu B tolerated\n", "binCV fivePointEssential alone", pFive);
    std::printf("  %-34s %8zu B tolerated\n", "binCV findEssentialMat (whole)", pBin);
    std::printf("  %-34s %8zu B tolerated\n", "cv::findEssentialMat (whole)", pCv);
    std::printf("\n  STACK USED, net of baseline:\n");
    std::printf("    binCV five-point solver        %8zu B\n", pBase - pFive);
    std::printf("    binCV whole RANSAC call        %8zu B\n", pBase - pBin);
    std::printf("    OpenCV whole call              %8zu B\n", pBase - pCv);
    return 0;
}
