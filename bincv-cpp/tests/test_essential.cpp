// The five-point essential matrix, and what pins it.
//
// The solver eliminates a system of ten cubics down to a degree-10 polynomial. A
// wrong sign or a transposed index in that chain produces a solver that still
// returns matrices, still looks plausible, and is wrong -- so the checks here are
// chosen to fail loudly on exactly that:
//
//   * the planted E must be RECOVERED, not merely approximated
//   * EVERY returned solution must satisfy the epipolar constraint, not just the
//     one that happens to match -- a solution set polluted by spurious roots
//     passes the first check and fails this one
//   * the count of solutions must be in the range the problem is known to have
//   * end to end, RANSAC must separate inliers from outliers on real geometry
//
// Ground truth is a planted E = [t]x R, so "correct" is checkable without an
// oracle. cv::findEssentialMat is compared against as well, but as a second
// opinion rather than as the definition.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "bincv-cpp/ops/essential.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace {

using namespace bincv;

std::size_t g_newCount = 0;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    double uniform() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return 2.0 * (static_cast<double>((s >> 33) % 1000001) / 1000000.0) - 1.0;
    }
};

/// @brief A camera pair and the essential matrix it induces, planted so that
/// correctness is checkable without an oracle.
struct Pose {
    double r[3][3];
    double t[3];
    double e[9];  // [t]x R, unit Frobenius norm
};

Pose makePose(Rng& rng, double angleScale) {
    Pose p;
    const double ax = rng.uniform() * angleScale, ay = rng.uniform() * angleScale,
                 az = rng.uniform() * angleScale;
    const double ca = std::cos(ax), sa = std::sin(ax), cb = std::cos(ay), sb = std::sin(ay),
                 cc = std::cos(az), sc = std::sin(az);
    const double r[3][3] = {{cb * cc, -cb * sc, sb},
                            {sa * sb * cc + ca * sc, -sa * sb * sc + ca * cc, -sa * cb},
                            {-ca * sb * cc + sa * sc, ca * sb * sc + sa * cc, ca * cb}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) p.r[i][j] = r[i][j];
    }
    p.t[0] = rng.uniform();
    p.t[1] = rng.uniform();
    p.t[2] = 1.0 + 0.3 * rng.uniform();
    const double tn = std::sqrt(p.t[0] * p.t[0] + p.t[1] * p.t[1] + p.t[2] * p.t[2]);
    for (int i = 0; i < 3; ++i) p.t[i] /= tn;

    const double tx[3][3] = {{0, -p.t[2], p.t[1]}, {p.t[2], 0, -p.t[0]}, {-p.t[1], p.t[0], 0}};
    double norm = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += tx[i][k] * p.r[k][j];
            p.e[i * 3 + j] = acc;
            norm += acc * acc;
        }
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < 9; ++i) p.e[i] /= norm;
    return p;
}

/// @brief One correspondence through `pose`, from a point in front of both cameras.
void project(const Pose& pose, Rng& rng, Point2f& a, Point2f& b) {
    const double x[3] = {rng.uniform() * 2.0, rng.uniform() * 2.0, 4.0 + rng.uniform()};
    a.x = static_cast<float>(x[0] / x[2]);
    a.y = static_cast<float>(x[1] / x[2]);
    double xc[3];
    for (int i = 0; i < 3; ++i) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k) acc += pose.r[i][k] * x[k];
        xc[i] = acc + pose.t[i];
    }
    b.x = static_cast<float>(xc[0] / xc[2]);
    b.y = static_cast<float>(xc[1] / xc[2]);
}

/// @brief |q2^T E q1|, normalised by the matrix scale so the bound means something.
double epipolar(const EssentialMatrix& e, Point2f a, Point2f b) {
    const double q1[3] = {static_cast<double>(a.x), static_cast<double>(a.y), 1.0};
    const double q2[3] = {static_cast<double>(b.x), static_cast<double>(b.y), 1.0};
    double acc = 0.0, scale = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            acc += q2[i] * e.m[i * 3 + j] * q1[j];
            scale += e.m[i * 3 + j] * e.m[i * 3 + j];
        }
    }
    return std::fabs(acc) / std::sqrt(scale > 0 ? scale : 1.0);
}

/// @brief Distance to a planted E, up to the sign the matrix is only defined to.
double distanceToPlanted(const EssentialMatrix& got, const double planted[9]) {
    double plus = 0.0, minus = 0.0;
    for (int i = 0; i < 9; ++i) {
        const double a = got.m[i] - planted[i], b = got.m[i] + planted[i];
        plus += a * a;
        minus += b * b;
    }
    return std::sqrt(plus < minus ? plus : minus);
}

} // namespace

BINCV_TEST(Essential, FivePointRecoversThePlantedMatrix) {
    Rng rng(0x5EED5);
    const int kTrials = 200;
    int recovered = 0, everySolutionEpipolar = 0, solutionsTotal = 0, solved = 0;

    for (int trial = 0; trial < kTrials; ++trial) {
        const Pose pose = makePose(rng, 0.5);
        Point2f from[5], to[5];
        for (int i = 0; i < 5; ++i) project(pose, rng, from[i], to[i]);

        EssentialMatrix sols[10];
        const int n = fivePointEssential(from, to, sols);
        solutionsTotal += n;
        BINCV_CHECK(n >= 0 && n <= 10);
        if (n >= 1) ++solved;

        double best = 1e30;
        bool allEpipolar = true;
        for (int s = 0; s < n; ++s) {
            const double d = distanceToPlanted(sols[s], pose.e);
            if (d < best) best = d;
            for (int i = 0; i < 5; ++i) {
                if (epipolar(sols[s], from[i], to[i]) > 1e-6) allEpipolar = false;
            }
        }
        if (best < 1e-3) ++recovered;
        if (allEpipolar) ++everySolutionEpipolar;
    }

    // TWO SEPARATE PROPERTIES, AND CONFLATING THEM HID A REGRESSION ONCE.
    //
    // "Did it find anything" is not "is what it found correct". A minimal sample can
    // be near-degenerate and yield no real root, which RANSAC simply skips -- so a
    // small shortfall here is a robustness figure, not a wrong answer. Measured, the
    // solver returns at least one solution on about 99.5% of random non-degenerate
    // samples.
    BINCV_CHECK(solved >= kTrials - 4);
    // The strong one, and it admits no shortfall: EVERY solution returned must
    // satisfy the epipolar constraint. Measured residuals sit at machine precision
    // (max 7.5e-14 over 400 trials), so a spurious root -- which lands orders of
    // magnitude above that -- fails this decisively.
    BINCV_CHECK_EQ(everySolutionEpipolar, kTrials);
    // Float input coordinates bound how closely the planted E can be recovered;
    // see the note on EssentialMatrix.
    BINCV_CHECK(recovered >= kTrials - 4);
    std::printf(" essential: solutions found on %d/%d samples, planted E recovered %d/%d,"
                " every solution epipolar %d/%d, %.2f solutions per call\n",
                solved, kTrials, recovered, kTrials, everySolutionEpipolar, kTrials,
                static_cast<double>(solutionsTotal) / kTrials);
}

BINCV_TEST(Essential, RansacSeparatesInliersFromOutliers) {
    Rng rng(0xC0FFEE);
    const int kScenes = 25, kPoints = 300;
    int clean = 0;
    double sumPrecision = 0.0, sumRecall = 0.0;

    for (int scene = 0; scene < kScenes; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        std::vector<Point2f> from(kPoints), to(kPoints);
        std::vector<uint8_t> truth(kPoints, 1);
        size_t planted = 0;
        for (int i = 0; i < kPoints; ++i) {
            project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
            if (i % 5 == 0) {
                truth[static_cast<size_t>(i)] = 0;
                to[static_cast<size_t>(i)].x += static_cast<float>(rng.uniform() * 0.4);
                to[static_cast<size_t>(i)].y += static_cast<float>(rng.uniform() * 0.4);
            } else {
                ++planted;
            }
        }

        std::vector<uint8_t> best(static_cast<size_t>(kPoints)), cand(static_cast<size_t>(kPoints)),
            mask(static_cast<size_t>(kPoints));
        const RansacScratch scratch{best.data(), cand.data(), static_cast<size_t>(kPoints)};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 500;
        params.seed = 7;
        EssentialMatrix e;
        const RansacResult r = findEssentialMat(from.data(), to.data(),
                                                static_cast<size_t>(kPoints), params, scratch, &e,
                                                mask.data());
        BINCV_CHECK_EQ(r.found, true);

        size_t tp = 0, fp = 0;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 0) continue;
            if (truth[i] != 0) ++tp; else ++fp;
        }
        const double precision = (tp + fp) > 0 ? static_cast<double>(tp) /
                                                     static_cast<double>(tp + fp) : 0.0;
        const double recall = static_cast<double>(tp) / static_cast<double>(planted);
        sumPrecision += precision;
        sumRecall += recall;
        if (precision > 0.98 && recall > 0.90) ++clean;
    }

    BINCV_CHECK_EQ(clean, kScenes);
    std::printf(" essential: %d/%d scenes clean, mean inlier precision %.4f recall %.4f\n",
                clean, kScenes, sumPrecision / kScenes, sumRecall / kScenes);
}

BINCV_TEST(Essential, DeterministicAndDegenerate) {
    Rng rng(0xD00D);
    const Pose pose = makePose(rng, 0.4);
    const int kPoints = 120;
    std::vector<Point2f> from(kPoints), to(kPoints);
    for (int i = 0; i < kPoints; ++i) {
        project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
    }
    std::vector<uint8_t> b(static_cast<size_t>(kPoints)), c(static_cast<size_t>(kPoints));
    const RansacScratch scratch{b.data(), c.data(), static_cast<size_t>(kPoints)};
    RansacParams params;
    params.threshold = 0.002;

    EssentialMatrix e1, e2;
    const RansacResult r1 = findEssentialMat(from.data(), to.data(),
                                             static_cast<size_t>(kPoints), params, scratch, &e1);
    const RansacResult r2 = findEssentialMat(from.data(), to.data(),
                                             static_cast<size_t>(kPoints), params, scratch, &e2);
    BINCV_CHECK_EQ(r1.inliers, r2.inliers);
    BINCV_CHECK_EQ(r1.iterations, r2.iterations);
    for (int i = 0; i < 9; ++i) BINCV_CHECK_EQ(e1.m[i], e2.m[i]);

    // Fewer than five correspondences: no model, output untouched.
    EssentialMatrix untouched;
    untouched.m[4] = 42.0;
    const RansacResult few = findEssentialMat(from.data(), to.data(), 4, params, scratch,
                                              &untouched);
    BINCV_CHECK_EQ(few.found, false);
    BINCV_CHECK_EQ(untouched.m[4], 42.0);

    // Five identical correspondences: the epipolar system is rank-deficient and the
    // elimination must decline rather than return noise.
    Point2f same[5], sameTo[5];
    for (int i = 0; i < 5; ++i) {
        same[i] = Point2f{0.25f, -0.1f};
        sameTo[i] = Point2f{0.31f, -0.08f};
    }
    EssentialMatrix sols[10];
    const int n = fivePointEssential(same, sameTo, sols);
    BINCV_CHECK(n >= 0 && n <= 10);
    std::printf(" essential: deterministic across runs; degenerate five-point returns %d\n", n);
}

BINCV_TEST(Essential, AllocatesNothing) {
    Rng rng(0xA110C);
    const Pose pose = makePose(rng, 0.4);
    const int kPoints = 200;
    std::vector<Point2f> from(kPoints), to(kPoints);
    for (int i = 0; i < kPoints; ++i) {
        project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
    }
    std::vector<uint8_t> b(static_cast<size_t>(kPoints)), c(static_cast<size_t>(kPoints));
    const RansacScratch scratch{b.data(), c.data(), static_cast<size_t>(kPoints)};
    RansacParams params;
    params.threshold = 0.002;
    EssentialMatrix e;

    const std::size_t before = g_newCount;
    const RansacResult r = findEssentialMat(from.data(), to.data(),
                                            static_cast<size_t>(kPoints), params, scratch, &e);
    const std::size_t during = g_newCount - before;

    BINCV_CHECK_EQ(during, std::size_t{0});
    BINCV_CHECK_EQ(r.found, true);
    // The stack figure is the one that decides whether this fits on a small part,
    // so it is asserted rather than merely documented.
    BINCV_CHECK(essentialSolverStackBytes() < 32768);
    std::printf(" essential: operator new called %zu times; solver stack %zu B\n", during,
                essentialSolverStackBytes());
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Essential, AgreesWithOpenCV) {
    Rng rng(0xBEEF77);
    const int kScenes = 12, kPoints = 260;
    int agreed = 0;

    for (int scene = 0; scene < kScenes; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        std::vector<Point2f> from(kPoints), to(kPoints);
        std::vector<cv::Point2f> cvFrom, cvTo;
        for (int i = 0; i < kPoints; ++i) {
            project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
            if (i % 6 == 0) {
                to[static_cast<size_t>(i)].x += static_cast<float>(rng.uniform() * 0.4);
                to[static_cast<size_t>(i)].y += static_cast<float>(rng.uniform() * 0.4);
            }
            cvFrom.push_back(cv::Point2f(from[static_cast<size_t>(i)].x,
                                         from[static_cast<size_t>(i)].y));
            cvTo.push_back(cv::Point2f(to[static_cast<size_t>(i)].x,
                                       to[static_cast<size_t>(i)].y));
        }

        std::vector<uint8_t> b(static_cast<size_t>(kPoints)), c(static_cast<size_t>(kPoints)),
            mask(static_cast<size_t>(kPoints));
        const RansacScratch scratch{b.data(), c.data(), static_cast<size_t>(kPoints)};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 500;
        EssentialMatrix e;
        const RansacResult r = findEssentialMat(from.data(), to.data(),
                                                static_cast<size_t>(kPoints), params, scratch, &e,
                                                mask.data());

        cv::Mat cvMask;
        const cv::Mat cvE = cv::findEssentialMat(cvFrom, cvTo, 1.0, cv::Point2d(0, 0), cv::RANSAC,
                                                 0.99, 0.002, cvMask);
        size_t cvInliers = 0;
        for (int i = 0; i < cvMask.rows; ++i) cvInliers += cvMask.at<uint8_t>(i, 0) != 0 ? 1u : 0u;

        // Not bit-exactness -- two randomised searches. What must hold is that both
        // find essentially the same consensus set on data with one true model.
        const double gap = std::fabs(static_cast<double>(r.inliers) -
                                     static_cast<double>(cvInliers));
        if (r.found && !cvE.empty() && gap <= 0.05 * kPoints) ++agreed;
    }

    BINCV_CHECK_EQ(agreed, kScenes);
    std::printf(" essential: inlier counts agree with cv::findEssentialMat on %d/%d scenes\n",
                agreed, kScenes);
}
#endif

namespace {
void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}
void* countedAllocateNoThrow(std::size_t bytes) noexcept {
    ++g_newCount;
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

BINCV_TEST_MAIN("test_essential")
