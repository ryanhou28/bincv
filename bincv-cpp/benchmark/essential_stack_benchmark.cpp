// HOW MUCH STACK MUST A CALLER RESERVE FOR ONE CALL? Both estimators this library
// ships, against the OpenCV call each stands in for.
//
// WHY THIS FILE EXISTS. The heap comparisons in essential_benchmark.cpp and
// ransac_benchmark.cpp cannot see this: the solvers keep their working arrays on
// the STACK. Reporting only the heap therefore compares the small change and misses
// the thing itself, and comparing binCV's stack against OpenCV's heap -- which an
// earlier version of the first of those did -- is not a measurement at all.
//
// The affine arms are here for the same reason. binCV's no-allocation rule means a
// heap-only instrument credits it with a saving it did not make; the only way that
// claim is honest is if both storage classes are measured on both sides. See
// docs/reports/methodology-memory.md, which puts a number on how wrong heap-only
// gets: 64x, from storage class alone.
//
// WHAT IS MEASURED, AND WHY THIS QUANTITY. The smallest thread stack on which the
// call completes without hitting the guard page. That is what a caller sizing a
// thread has to know, and it is not the same as "bytes written": a frame that is
// allocated and only partly written still has to be reserved. Instruments that scan
// for written bytes -- stack painting, watermarking -- report the smaller quantity
// and would understate this one.
//
// THE INSTRUMENT PROVES ITSELF FIRST. Two calibration workloads consume a stack
// amount this file chooses, 4 KiB and 16 KiB. If the probe does not recover those
// within tolerance the run FAILS and reports nothing, because a measuring device
// that has never returned a known answer is not evidence. This is the same reasoning
// as verify.sh's gate self-check.
//
// WHY NOT VALGRIND MASSIF --stacks=yes. It was tried and it fails the calibration
// above: measured here it reported the whole binCV call using LESS stack than an
// empty workload -- 8 216 B against 10 088 B -- which is impossible, and it still
// did so with snapshot counts between 766 and 980.
//
// The reason is not that its tracking is coarse. Massif hooks Valgrind's
// new_mem_stack / die_mem_stack, which fire on every stack-pointer change, so it
// tracks exactly. What is periodic is the RECORDING: stack size is written out only
// at snapshots, and the peak-snapshot logic is driven by heap size, not stack size.
// A stack peak that rises and falls inside one call is tracked and never written
// down. That is why massif works for a large long-lived frame and fails for a
// transient few-kilobyte one, which is precisely this case. Its heap figures are
// sound; its stack figures are not usable at this granularity.
//
// TWO PASSES, BECAUSE PTHREAD_STACK_MIN IS 16 KiB AND binCV FITS UNDER IT. Bisecting
// the stack SIZE cannot resolve anything smaller than that floor. The second pass
// fixes a generous stack and bisects the PADDING consumed before the call instead:
// the largest padding a workload tolerates is headroom it did not need, so the
// difference between two workloads' tolerated padding is the difference in what they
// used. The baseline row is an empty workload and every figure is net of it.
//
// A MEASURED FIGURE IS A LOWER BOUND ON THE WORST CASE, NOT THE WORST CASE. It is the
// deepest stack reached by the paths that ran. Each workload therefore solves several
// different scenes and the figure is the deepest of them, which widens the coverage
// without changing what is being claimed. A caller budgeting against these numbers
// should keep margin.
//
// A GUARD PAGE CATCHES A STACK POINTER THAT WALKS INTO IT, NOT ONE THAT STEPS OVER
// IT, so a large frame that is reserved and left unwritten can be missed. The sparse
// row below measures exactly that, and the two architectures differ: x86-64 catches a
// 65 536 B sparse frame, aarch64 reads it as 16 B and the row prints UNDER-READ. It
// does not reach the figures here -- the DENSE calibration passes to the byte on both,
// in the 4-16 KiB range the measured frames occupy, and on the device -fstack-usage
// puts the deepest five-point path near 6 432 B where this probe reads 6 928 B, which
// is above the static bound rather than below it.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "bincv-cpp/ops/essential.hpp"
#include "bincv-cpp/ops/ransac.hpp"

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kPoints = 1000;
constexpr int kScenes = 6;      ///< distinct scenes per workload; the figure is their max
constexpr size_t kGranularity = 16;
constexpr size_t kFixedStack = 512u * 1024u;
constexpr int kRepeats = 3;

struct Scene {
    std::vector<cv::Point2f> ca, cb;
    std::vector<bincv::Point2f> ba, bb;
};
Scene g_scene[kScenes];
std::vector<uint32_t> g_flags;
volatile int g_guard = 0;

/// @brief Pixel-coordinate correspondences under a fixed affine, for the affine
/// arms. The essential scenes are in normalised coordinates and are not reusable
/// here.
struct AffineScene {
    std::vector<bincv::Point2f> from, to;
    std::vector<cv::Point2f> cf, ct;
};
AffineScene g_affine[kScenes];

void buildAffineScene(AffineScene& s, uint64_t seed) {
    const bincv::Affine2D t{{1.10f, -0.20f, 12.0f, 0.15f, 0.95f, -7.0f}};
    uint64_t st = seed;
    auto nextf = [&st]() {
        st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return static_cast<float>((st >> 33) % 100000) / 100000.0f;
    };
    for (int i = 0; i < kPoints; ++i) {
        const bincv::Point2f a{nextf() * 640.0f, nextf() * 480.0f};
        bincv::Point2f b{t.m[0] * a.x + t.m[1] * a.y + t.m[2],
                         t.m[3] * a.x + t.m[4] * a.y + t.m[5]};
        if (i % 4 == 0) {
            b.x += 60.0f + nextf() * 120.0f;
            b.y -= 55.0f + nextf() * 110.0f;
        }
        s.from.push_back(a);
        s.to.push_back(b);
        s.cf.push_back(cv::Point2f(a.x, a.y));
        s.ct.push_back(cv::Point2f(b.x, b.y));
    }
}

void buildScene(Scene& s, uint64_t seed) {
    uint64_t st = seed;
    auto uni = [&st]() {
        st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((st >> 33) % 1000001) / 1000000.0) - 1.0;
    };
    double t[3] = {0.3 + 0.1 * uni(), 0.1 * uni(), 1.0};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    for (int i = 0; i < 3; ++i) t[i] /= tn;
    for (int i = 0; i < kPoints; ++i) {
        const double X[3] = {uni() * 2, uni() * 2, 4 + uni()};
        const float ax = static_cast<float>(X[0] / X[2]), ay = static_cast<float>(X[1] / X[2]);
        const double Xc[3] = {X[0] + t[0], X[1] + t[1], X[2] + t[2]};
        float bx = static_cast<float>(Xc[0] / Xc[2]), by = static_cast<float>(Xc[1] / Xc[2]);
        if (i % 5 == 0) {
            bx += static_cast<float>(uni() * 0.4);
            by += static_cast<float>(uni() * 0.4);
        }
        s.ca.push_back(cv::Point2f(ax, ay));
        s.cb.push_back(cv::Point2f(bx, by));
        s.ba.push_back({ax, ay});
        s.bb.push_back({bx, by});
    }
}

// --- workloads -------------------------------------------------------------
// Each runs over every scene so the reported figure is the deepest of several paths.

void* wBaseline(void*) {
    for (int i = 0; i < kScenes; ++i) g_guard += 1;
    return nullptr;
}

/// @brief Consumes a stack amount this file chooses, so the probe can be checked
/// against a known answer. `volatile` and the touch loop keep it from being elided.
template <size_t Bytes>
void* wCalibration(void*) {
    for (int i = 0; i < kScenes; ++i) {
        volatile char buf[Bytes];
        for (size_t k = 0; k < Bytes; k += 64) buf[k] = static_cast<char>(k);
        g_guard += buf[Bytes - 1];
    }
    return nullptr;
}

/// @brief A frame the same size as the calibration above but barely touched, which
/// is the case a guard page can fail to catch: if the stack pointer moves past the
/// guard in one step and nothing is written inside it, no fault is raised. Reported
/// as a diagnostic rather than a gate -- it measures the probe, not the library.
template <size_t Bytes>
void* wSparse(void*) {
    for (int i = 0; i < kScenes; ++i) {
        volatile char buf[Bytes];
        // Only the SHALLOW end -- the byte nearest the caller. The deep end, which
        // is what sits next to the guard page, is never written.
        buf[Bytes - 1] = 2;
        g_guard += buf[Bytes - 1];
    }
    return nullptr;
}

void* wFivePoint(void*) {
    for (int i = 0; i < kScenes; ++i) {
        bincv::EssentialMatrix e[10];
        g_guard += bincv::fivePointEssential(g_scene[i].ba.data(), g_scene[i].bb.data(), e);
    }
    return nullptr;
}

void* wBincvRansac(void*) {
    for (int i = 0; i < kScenes; ++i) {
        const bincv::RansacScratch sc{g_flags.data(), static_cast<size_t>(kPoints)};
        bincv::RansacParams p;
        p.threshold = 0.002;
        p.maxIterations = 500;
        bincv::EssentialMatrix e;
        g_guard += static_cast<int>(
            bincv::findEssentialMat(g_scene[i].ba.data(), g_scene[i].bb.data(),
                                    static_cast<size_t>(kPoints), p, sc, &e).inliers);
    }
    return nullptr;
}

void* wOpenCv(void*) {
    for (int i = 0; i < kScenes; ++i) {
        cv::Mat mask;
        const cv::Mat E = cv::findEssentialMat(g_scene[i].ca, g_scene[i].cb, 1.0,
                                               cv::Point2d(0, 0), cv::RANSAC, 0.99, 0.002, mask);
        g_guard += E.rows;
    }
    return nullptr;
}

void* wBincvAffine(void*) {
    for (int i = 0; i < kScenes; ++i) {
        const bincv::RansacScratch sc{g_flags.data(), static_cast<size_t>(kPoints)};
        bincv::RansacParams p;
        p.threshold = 3.0;
        bincv::Affine2D m;
        g_guard += static_cast<int>(
            bincv::estimateAffine2D(g_affine[i].from.data(), g_affine[i].to.data(),
                                    static_cast<size_t>(kPoints), p, sc, &m).inliers);
    }
    return nullptr;
}

void* wOpenCvAffine(void*) {
    for (int i = 0; i < kScenes; ++i) {
        std::vector<uint8_t> mask;
        const cv::Mat m = cv::estimateAffine2D(g_affine[i].cf, g_affine[i].ct, mask,
                                               cv::RANSAC, 3.0);
        g_guard += m.rows;
    }
    return nullptr;
}

// --- the probe -------------------------------------------------------------

size_t g_pad = 0;
void* (*g_inner)(void*) = nullptr;

void* wPadded(void* a) {
    volatile char* pad = static_cast<volatile char*>(__builtin_alloca(g_pad));
    for (size_t i = 0; i < g_pad; i += 512) pad[i] = 1;  // commit every page
    void* r = g_inner(a);
    for (size_t i = 0; i < g_pad; i += 512) g_guard += pad[i];
    return r;
}

/// @brief true when the workload completes with `pad` bytes eaten before the call.
/// Each trial is a forked child, so an overflow is a failed trial and not the end
/// of the run.
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
    while (hi - lo > kGranularity) {
        const size_t mid = lo + (hi - lo) / 2;
        if (survivesPad(fn, mid)) lo = mid; else hi = mid;
    }
    return lo;
}

struct Reading {
    size_t lo = 0, hi = 0;  ///< across repeats
};

/// @brief Stack used by `fn` beyond the empty workload, repeated so spread is visible.
Reading measure(void* (*fn)(void*), const size_t* padBaseline) {
    Reading r{SIZE_MAX, 0};
    for (int k = 0; k < kRepeats; ++k) {
        const size_t pad = maxPad(fn);
        const size_t used = padBaseline[k] > pad ? padBaseline[k] - pad : 0;
        if (used < r.lo) r.lo = used;
        if (used > r.hi) r.hi = used;
    }
    return r;
}

void printRow(const char* name, Reading r) {
    if (r.lo == r.hi) std::printf("  %-34s %8zu B\n", name, r.lo);
    else std::printf("  %-34s %8zu - %zu B\n", name, r.lo, r.hi);
}

} // namespace

int main() {
    for (int i = 0; i < kScenes; ++i) buildScene(g_scene[i], 99 + 7777 * static_cast<uint64_t>(i));
    for (int i = 0; i < kScenes; ++i) {
        buildAffineScene(g_affine[i], 5150 + 7919 * static_cast<uint64_t>(i));
    }
    g_flags.assign(2 * bincv::ransacScratchWords(static_cast<size_t>(kPoints)), 0);
    cv::setNumThreads(1);

    // Warm both so lazy one-time initialisation is not charged to a trial.
    {
        cv::Mat m;
        cv::findEssentialMat(g_scene[0].ca, g_scene[0].cb, 1.0, cv::Point2d(0, 0),
                             cv::RANSAC, 0.99, 0.002, m);
    }
    {
        bincv::EssentialMatrix e[10];
        bincv::fivePointEssential(g_scene[0].ba.data(), g_scene[0].bb.data(), e);
    }
    {
        std::vector<uint8_t> mk;
        cv::estimateAffine2D(g_affine[0].cf, g_affine[0].ct, mk, cv::RANSAC, 3.0);
    }

    std::printf("Stack a caller must reserve for one call, %d correspondences,"
                " deepest of %d scenes\n", kPoints, kScenes);
    std::printf("Probe: guard-page bisection on a %zu KiB thread stack,"
                " %zu B granularity, %d repeats\n\n", kFixedStack / 1024, kGranularity, kRepeats);

    size_t padBaseline[kRepeats];
    for (int k = 0; k < kRepeats; ++k) padBaseline[k] = maxPad(wBaseline);

    // --- the instrument proves itself before any figure below is believed -----
    std::printf("CALIBRATION -- known answers the probe must recover\n");
    const Reading c4 = measure(wCalibration<4096>, padBaseline);
    const Reading c16 = measure(wCalibration<16384>, padBaseline);
    printRow("workload consuming 4 096 B", c4);
    printRow("workload consuming 16 384 B", c16);

    bool ok = true;
    struct Check { const char* name; size_t expect; Reading got; };
    const Check checks[] = {{"4 096 B", 4096, c4}, {"16 384 B", 16384, c16}};
    for (const Check& c : checks) {
        // Tolerance: the bisection granularity plus 5%, which covers the frame the
        // calibration workload itself needs around its buffer.
        const size_t tol = kGranularity + c.expect / 20;
        if (c.got.lo + tol < c.expect || c.got.hi > c.expect + tol) {
            std::printf("  FAIL: expected %s +/- %zu, measured %zu - %zu\n",
                        c.name, tol, c.got.lo, c.got.hi);
            ok = false;
        }
    }
    if (!ok) {
        std::printf("\nTHE PROBE DID NOT RECOVER ITS KNOWN ANSWERS. No figure below would\n"
                    "be evidence, so none is printed.\n");
        return 1;
    }
    std::printf("  both recovered within tolerance -- the figures below are readings\n");

    // Diagnostic, not a gate: what the probe does with a frame that is reserved but
    // barely written. A guard page catches a stack pointer that walks into it, not
    // one that steps over it, so this is where the technique is weakest. Both sides
    // below are measured with the same probe, so a shared limitation does not tilt
    // the comparison -- but it does bound what a single figure means.
    const Reading sparse = measure(wSparse<65536>, padBaseline);
    std::printf("\n  probe limit -- 65 536 B frame, two bytes written: reads %zu B\n",
                sparse.hi);
    std::printf("  %s\n", sparse.hi + 65536 / 20 >= 65536
                    ? "reserved space is caught even when unwritten"
                    : "UNDER-READ: a large sparse frame steps over the guard page");
    std::printf("\n");

    // --- the measurement ------------------------------------------------------
    std::printf("STACK USED BY ONE CALL, net of the empty workload\n");
    printRow("binCV fivePointEssential alone", measure(wFivePoint, padBaseline));
    printRow("binCV findEssentialMat (whole)", measure(wBincvRansac, padBaseline));
    printRow("cv::findEssentialMat (whole)", measure(wOpenCv, padBaseline));
    std::printf("\n  and the affine pair, so its memory claim is not heap-only either:\n");
    printRow("binCV estimateAffine2D", measure(wBincvAffine, padBaseline));
    printRow("cv::estimateAffine2D", measure(wOpenCvAffine, padBaseline));
    std::printf("\n  A measured figure is a LOWER BOUND on worst-case stack: it is the\n"
                "  deepest of the paths that ran, over %d scenes. Budget with margin.\n",
                kScenes);
    return 0;
}
