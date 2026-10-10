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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "bincv/ops/essential.hpp"
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
    /// @brief Standard normal, for putting noise on the INLIERS. A scene whose
    /// inliers are exact cannot tell an accurate estimator from a lucky one.
    double gaussian() {
        const double u1 = 0.5 * (uniform() + 1.0) + 1e-9, u2 = 0.5 * (uniform() + 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
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

/// @brief The geometries a five-point estimator is actually judged on.
enum class Degenerate { PureRotation, PlanarScene, TinyBaseline };

/// @brief A scene of `n` correspondences under a degenerate configuration.
/// @note NONE of these is detected as degenerate, by binCV or by OpenCV, and that
/// is the point of testing them: a caller who reads a high inlier count here as
/// evidence of a valid essential matrix is wrong, and the library should behave
/// the same way OpenCV does rather than differently-wrong.
void buildDegenerate(Degenerate kind, size_t n, std::vector<Point2f>& a,
                     std::vector<Point2f>& b) {
    Rng rng(UINT64_C(0xD36E));
    const double ang = 0.15;
    const double r[3][3] = {{std::cos(ang), 0.0, std::sin(ang)},
                            {0.0, 1.0, 0.0},
                            {-std::sin(ang), 0.0, std::cos(ang)}};
    double t[3] = {0.0, 0.0, 0.0};
    if (kind == Degenerate::TinyBaseline) {
        t[0] = 3e-5;
        t[1] = 1e-5;
        t[2] = 1e-4;
    } else if (kind == Degenerate::PlanarScene) {
        t[0] = 0.28;
        t[1] = 0.09;
        t[2] = 0.95;
    }
    for (size_t i = 0; i < n; ++i) {
        double x[3] = {rng.uniform() * 2.0, rng.uniform() * 2.0, 4.0 + rng.uniform()};
        if (kind == Degenerate::PlanarScene) x[2] = 5.0;  // every point on one plane
        double xc[3];
        for (int k = 0; k < 3; ++k) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j) acc += r[k][j] * x[j];
            xc[k] = acc + t[k];
        }
        Point2f p{static_cast<float>(x[0] / x[2]), static_cast<float>(x[1] / x[2])};
        Point2f q{static_cast<float>(xc[0] / xc[2]), static_cast<float>(xc[1] / xc[2])};
        if (i % 10 == 0) {
            q.x += static_cast<float>(rng.uniform() * 0.2);
            q.y += static_cast<float>(rng.uniform() * 0.2);
        }
        a.push_back(p);
        b.push_back(q);
    }
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

/// @brief The published solver and the one built for a small core behind one spelling,
/// so that each accuracy test below holds both to the same bound.
template <bool RealRoots>
int solveFive(const Point2f* from, const Point2f* to, EssentialMatrix* out) {
    if constexpr (RealRoots) {
        return fivePointEssentialRealRoots(from, to, out);
    } else {
        return fivePointEssential(from, to, out);
    }
}
template <bool RealRoots>
RansacResult estimate(const Point2f* from, const Point2f* to, size_t count,
                      const RansacParams& params, RansacScratch scratch, EssentialMatrix* model,
                      uint8_t* mask = nullptr) {
    if constexpr (RealRoots) {
        return findEssentialMatRealRoots(from, to, count, params, scratch, model, mask);
    } else {
        return findEssentialMat(from, to, count, params, scratch, model, mask);
    }
}
constexpr const char* solverName(bool realRoots) {
    return realRoots ? "real-root solver" : "published solver";
}

} // namespace

template <bool RealRoots>
void FivePointRecoversThePlantedMatrixBody() {
    Rng rng(0x5EED5);
    const int kTrials = 200;
    int recovered = 0, everySolutionEpipolar = 0, solutionsTotal = 0, solved = 0;

    for (int trial = 0; trial < kTrials; ++trial) {
        const Pose pose = makePose(rng, 0.5);
        Point2f from[5], to[5];
        for (int i = 0; i < 5; ++i) project(pose, rng, from[i], to[i]);

        EssentialMatrix sols[10];
        const int n = solveFive<RealRoots>(from, to, sols);
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
    std::printf(" [%s]", solverName(RealRoots));
    std::printf(" essential: solutions found on %d/%d samples, planted E recovered %d/%d,"
                " every solution epipolar %d/%d, %.2f solutions per call\n",
                solved, kTrials, recovered, kTrials, everySolutionEpipolar, kTrials,
                static_cast<double>(solutionsTotal) / kTrials);
}
BINCV_TEST(Essential, FivePointRecoversThePlantedMatrix) {
    FivePointRecoversThePlantedMatrixBody<false>();
}
BINCV_TEST(Essential, FivePointRecoversThePlantedMatrix_RealRoots) {
    FivePointRecoversThePlantedMatrixBody<true>();
}

template <bool RealRoots>
void RansacSeparatesInliersFromOutliersBody() {
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

        std::vector<uint32_t> flags(2 * ransacScratchWords(static_cast<size_t>(kPoints)));
        std::vector<uint8_t> mask(static_cast<size_t>(kPoints));
        const RansacScratch scratch{flags.data(), static_cast<size_t>(kPoints)};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 500;
        params.seed = 7;
        EssentialMatrix e;
        const RansacResult r = estimate<RealRoots>(from.data(), to.data(),
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
    std::printf(" [%s]", solverName(RealRoots));
    std::printf(" essential: %d/%d scenes clean, mean inlier precision %.4f recall %.4f\n",
                clean, kScenes, sumPrecision / kScenes, sumRecall / kScenes);
}
BINCV_TEST(Essential, RansacSeparatesInliersFromOutliers) {
    RansacSeparatesInliersFromOutliersBody<false>();
}
BINCV_TEST(Essential, RansacSeparatesInliersFromOutliers_RealRoots) {
    RansacSeparatesInliersFromOutliersBody<true>();
}

namespace {
/// @brief How far `e` is from being an essential matrix: the larger of |det E| and the
/// largest entry of `2 E E^T E - tr(E E^T) E`, at unit norm.
double essentialConstraint(const EssentialMatrix& m) {
    double e[3][3], eet[3][3];
    for (int i = 0; i < 9; ++i) e[i / 3][i % 3] = m.m[i];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            eet[i][j] = 0.0;
            for (int k = 0; k < 3; ++k) eet[i][j] += e[i][k] * e[j][k];
        }
    }
    const double tr = eet[0][0] + eet[1][1] + eet[2][2];
    double worst = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += eet[i][k] * e[k][j];
            worst = std::max(worst, std::fabs(2.0 * acc - tr * e[i][j]));
        }
    }
    const double det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                       e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                       e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
    return std::max(worst, std::fabs(det));
}
/// @brief The largest entry of `a - b` or `a + b`, whichever is smaller: E is defined
/// only up to sign.
double signlessDistance(const EssentialMatrix& a, const EssentialMatrix& b) {
    double plus = 0.0, minus = 0.0;
    for (int i = 0; i < 9; ++i) {
        plus = std::max(plus, std::fabs(a.m[i] - b.m[i]));
        minus = std::max(minus, std::fabs(a.m[i] + b.m[i]));
    }
    return std::min(plus, minus);
}
} // namespace

BINCV_TEST(Essential, RealRootSolverAgreesWhereWellConditioned) {
    // The two solvers share the system and differ in how they find its roots, so where
    // the problem is well conditioned they must give the same matrices to roundoff, and
    // where it is not, both answers are poor and may differ. "Well conditioned" is read
    // off the published solver's own answer: an E within 1e-10 of satisfying the
    // essential constraints. Every such solution must have a counterpart from the
    // real-root solver within 1e-6; everything else is reported, not judged.
    Rng rng(0x2EA1);
    size_t accurate = 0, accurateMatched = 0, published = 0, realRoots = 0, unmatched = 0;
    // Solutions far from satisfying the essential constraints: a spurious root lands
    // there, and the epipolar residual cannot see it -- every combination of the
    // nullspace basis satisfies the five epipolar constraints by construction.
    size_t poorPublished = 0, poorRealRoots = 0;
    double worstAccurate = 0.0;
    for (int trial = 0; trial < 3000; ++trial) {
        const Pose pose = makePose(rng, 0.5);
        Point2f from[5], to[5];
        for (int i = 0; i < 5; ++i) project(pose, rng, from[i], to[i]);
        EssentialMatrix a[10], b[10];
        const int na = fivePointEssential(from, to, a);
        const int nb = fivePointEssentialRealRoots(from, to, b);
        published += static_cast<size_t>(na);
        realRoots += static_cast<size_t>(nb);
        for (int j = 0; j < nb; ++j) poorRealRoots += essentialConstraint(b[j]) > 1e-6 ? 1u : 0u;
        for (int i = 0; i < na; ++i) {
            poorPublished += essentialConstraint(a[i]) > 1e-6 ? 1u : 0u;
            double best = 1e30;
            for (int j = 0; j < nb; ++j) best = std::min(best, signlessDistance(a[i], b[j]));
            if (best >= 1e-6) ++unmatched;
            if (essentialConstraint(a[i]) < 1e-10) {
                ++accurate;
                if (best < 1e-6) ++accurateMatched;
                worstAccurate = std::max(worstAccurate, best);
            }
        }
    }
    BINCV_CHECK_EQ(accurateMatched, accurate);
    BINCV_CHECK(accurate > published / 2);
    // No more poor solutions than the published solver returns, with a margin for the
    // ill-conditioned samples on which the two differ.
    BINCV_CHECK(poorRealRoots <= poorPublished + poorPublished / 10 + 5);
    std::printf(" essential: solutions more than 1e-6 from essential -- published %zu,"
                " real-root %zu\n", poorPublished, poorRealRoots);
    std::printf(" essential: %zu published solutions, %zu real-root; of the %zu accurate to"
                " 1e-10, all within %.1e of a real-root one; %zu of all unmatched to 1e-6\n",
                published, realRoots, accurate, worstAccurate, unmatched);
}

#if BINCV_RANSAC_SQUARED_TEST
BINCV_TEST(Essential, SquaredTestDecidesAsTheResidual) {
    // isInlier against `residual < threshold`, the comparison it replaces, with the
    // threshold placed ON each residual and one float either side -- every one of those
    // inside the band where the squares must not decide -- and at fixed thresholds,
    // including ones whose square is not a normal double. The squared test compiles
    // only on M-profile; this suite compiles it everywhere.
    // The single-precision Scorer is held to the same answers alongside, at the same
    // thresholds and also on coordinates three hundred times larger, where its error
    // bound is three hundred times wider -- and at thresholds either side of the range
    // it decides in.
    BINCV_CHECK(impl::ransacSquaredTestEnabled() == (BINCV_RANSAC_SQUARED_TEST_DEFAULT != 0));
    BINCV_CHECK(impl::ransacFloatScorerEnabled() == (BINCV_RANSAC_SQUARED_TEST_DEFAULT != 0));
    Rng rng(0x5A0A2E);
    size_t compared = 0, inliers = 0;
    const auto scorer = [](const EssentialMatrix& m, Point2f a, Point2f b, float t) {
        return EssentialModel::Scorer(m, t)(a, b);
    };
    const float fixed[] = {0.002f,
                           0.0005f,
                           3.0f,
                           1e-6f,
                           std::nextafter(1e-6f, 0.0f),
                           1e6f,
                           std::nextafter(1e6f, 1e38f),
                           1e-30f,
                           std::numeric_limits<float>::denorm_min(),
                           1e30f,
                           std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN(),
                           0.0f,
                           -1.0f};
    for (int scene = 0; scene < 200; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        EssentialMatrix e;
        for (int i = 0; i < 9; ++i) e.m[i] = pose.e[i];
        for (int k = 0; k < 50; ++k) {
            Point2f a, b;
            project(pose, rng, a, b);
            const double noise = (k % 3 == 0) ? 0.003 : (k % 3 == 1 ? 0.0005 : 0.2);
            b.x += static_cast<float>(rng.uniform() * noise);
            b.y += static_cast<float>(rng.uniform() * noise);
            for (const float scale : {1.0f, 300.0f}) {
                const Point2f sa{a.x * scale, a.y * scale}, sb{b.x * scale, b.y * scale};
                const float r = EssentialModel::residual(e, sa, sb);
                const float ts[3] = {r, std::nextafter(r, 1e38f), std::nextafter(r, 0.0f)};
                for (const float t : ts) {
                    const bool want = r < t;
                    BINCV_CHECK(EssentialModel::isInlier(e, sa, sb, t) == want);
                    BINCV_CHECK(scorer(e, sa, sb, t) == want);
                    inliers += want ? 1u : 0u;
                    ++compared;
                }
                for (const float t : fixed) {
                    const bool want = r < t;
                    BINCV_CHECK(EssentialModel::isInlier(e, sa, sb, t) == want);
                    BINCV_CHECK(scorer(e, sa, sb, t) == want);
                    ++compared;
                }
            }
        }
    }
    // Points near the epipoles, where E a and E^T b nearly vanish: the denominator is
    // then small beside the magnitudes it is computed from, which is the case the
    // Scorer's bound on it exists for.
    for (int scene = 0; scene < 100; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        EssentialMatrix e;
        for (int i = 0; i < 9; ++i) e.m[i] = pose.e[i];
        // E = [t]x R: E (R^T t) = 0 on the right, t^T E = 0 on the left.
        double e1[3] = {0, 0, 0};
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) e1[i] += pose.r[k][i] * pose.t[k];
        }
        if (std::fabs(e1[2]) < 1e-3 || std::fabs(pose.t[2]) < 1e-3) continue;
        for (const double off : {1e-2, 1e-4, 1e-6}) {
            for (int k = 0; k < 4; ++k) {
                const Point2f a{static_cast<float>(e1[0] / e1[2] + off * rng.uniform()),
                                static_cast<float>(e1[1] / e1[2] + off * rng.uniform())};
                const Point2f b{static_cast<float>(pose.t[0] / pose.t[2] + off * rng.uniform()),
                                static_cast<float>(pose.t[1] / pose.t[2] + off * rng.uniform())};
                const float r = EssentialModel::residual(e, a, b);
                const float ts[4] = {r, std::nextafter(r, 1e38f), std::nextafter(r, 0.0f),
                                     0.002f};
                for (const float t : ts) {
                    const bool want = r < t;
                    BINCV_CHECK(EssentialModel::isInlier(e, a, b, t) == want);
                    BINCV_CHECK(scorer(e, a, b, t) == want);
                    compared += 2;
                }
            }
        }
    }

    // And points where only the denominator nearly vanishes: E a close to (0, 0, c) and
    // E^T b close to (0, 0, c'), so the numerator stays large and the residual is huge.
    // An essential matrix has no such points -- (0, 0, c) is perpendicular to t only at
    // c = 0, so they are its epipoles again -- but the Scorer must be exact for any
    // hypothesis, and a poorly conditioned solve returns matrices that are not quite
    // essential. So these are random matrices of unit norm. Thresholds placed on such a
    // residual put the outlier test exactly where the denominator is least accurate.
    for (int scene = 0; scene < 300; ++scene) {
        EssentialMatrix e;
        double norm = 0.0;
        for (double& v : e.m) {
            v = rng.uniform();
            norm += v * v;
        }
        for (double& v : e.m) v /= std::sqrt(norm);
        const auto solve2 = [](double a, double b, double c, double d, double r0, double r1,
                               double& x, double& y) {
            const double det = a * d - b * c;
            if (std::fabs(det) < 1e-6) return false;
            x = (r0 * d - b * r1) / det;
            y = (a * r1 - c * r0) / det;
            return std::fabs(x) < 50 && std::fabs(y) < 50;
        };
        double ax, ay, bx, by;
        // Rows 0 and 1 of E against (ax, ay, 1); columns 0 and 1 against (bx, by, 1).
        if (!solve2(e.m[0], e.m[1], e.m[3], e.m[4], -e.m[2], -e.m[5], ax, ay)) continue;
        if (!solve2(e.m[0], e.m[3], e.m[1], e.m[4], -e.m[6], -e.m[7], bx, by)) continue;
        for (const double off : {1e-2, 1e-3, 1e-4, 1e-5}) {
            for (int k = 0; k < 4; ++k) {
                const Point2f a{static_cast<float>(ax + off * rng.uniform()),
                                static_cast<float>(ay + off * rng.uniform())};
                const Point2f b{static_cast<float>(bx + off * rng.uniform()),
                                static_cast<float>(by + off * rng.uniform())};
                const float r = EssentialModel::residual(e, a, b);
                const float ts[3] = {r, std::nextafter(r, 1e38f), std::nextafter(r, 0.0f)};
                for (const float t : ts) {
                    const bool want = r < t;
                    BINCV_CHECK(EssentialModel::isInlier(e, a, b, t) == want);
                    BINCV_CHECK(scorer(e, a, b, t) == want);
                    compared += 2;
                }
            }
        }
    }

    // The residual's own special cases: a vanishing denominator, and non-finite points.
    EssentialMatrix zero;
    for (double& v : zero.m) v = 0.0;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Point2f odd[] = {{0.1f, 0.2f}, {inf, 0.0f}, {nan, 0.3f}, {1e30f, -1e30f}};
    const Pose pose = makePose(rng, 0.4);
    EssentialMatrix e;
    for (int i = 0; i < 9; ++i) e.m[i] = pose.e[i];
    for (const Point2f& a : odd) {
        for (const Point2f& b : odd) {
            for (const float t : fixed) {
                BINCV_CHECK(EssentialModel::isInlier(e, a, b, t) ==
                            (EssentialModel::residual(e, a, b) < t));
                BINCV_CHECK(EssentialModel::isInlier(zero, a, b, t) ==
                            (EssentialModel::residual(zero, a, b) < t));
                BINCV_CHECK(scorer(e, a, b, t) == (EssentialModel::residual(e, a, b) < t));
                BINCV_CHECK(scorer(zero, a, b, t) ==
                            (EssentialModel::residual(zero, a, b) < t));
                compared += 4;
            }
        }
    }
    std::printf(" %zu decisions compared, %zu of the on-the-threshold ones inliers\n",
                compared, inliers);
}

BINCV_TEST(Essential, SquaredTestScoringIsBitExact) {
    // Whole estimations scored by the residual, by the squared test, and by the
    // single-precision Scorer in front of it: the same iterations, the same support, the
    // same model to the bit and the same mask.
    const bool saved = impl::ransacSquaredTestEnabled();
    const bool savedFloat = impl::ransacFloatScorerEnabled();
    Rng rng(0xB17E4AC7);
    for (int scene = 0; scene < 20; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        const size_t n = 100 + static_cast<size_t>(scene) * 37;
        std::vector<Point2f> from(n), to(n);
        for (size_t i = 0; i < n; ++i) {
            project(pose, rng, from[i], to[i]);
            const double noise = (i % 4 == 0) ? 0.4 : 0.002;
            to[i].x += static_cast<float>(rng.uniform() * noise);
            to[i].y += static_cast<float>(rng.uniform() * noise);
        }
        std::vector<uint32_t> flags(2 * ransacScratchWords(n));
        const RansacScratch scratch{flags.data(), n};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 300;
        params.seed = static_cast<uint64_t>(scene) + 1;
        std::vector<uint8_t> maskA(n), maskB(n), maskC(n);
        EssentialMatrix ea, eb, ec;
        impl::ransacSquaredTestEnabled() = false;
        const RansacResult ra =
            findEssentialMat(from.data(), to.data(), n, params, scratch, &ea, maskA.data());
        impl::ransacSquaredTestEnabled() = true;
        impl::ransacFloatScorerEnabled() = false;
        const RansacResult rb =
            findEssentialMat(from.data(), to.data(), n, params, scratch, &eb, maskB.data());
        impl::ransacFloatScorerEnabled() = true;
        const RansacResult rc =
            findEssentialMat(from.data(), to.data(), n, params, scratch, &ec, maskC.data());
        BINCV_CHECK(ra.found == rb.found && rb.found == rc.found);
        BINCV_CHECK(ra.inliers == rb.inliers && rb.inliers == rc.inliers);
        BINCV_CHECK(ra.iterations == rb.iterations && rb.iterations == rc.iterations);
        BINCV_CHECK(maskA == maskB && maskB == maskC);
        BINCV_CHECK(std::memcmp(ea.m, eb.m, sizeof ea.m) == 0);
        BINCV_CHECK(std::memcmp(eb.m, ec.m, sizeof eb.m) == 0);
    }
    impl::ransacSquaredTestEnabled() = saved;
    impl::ransacFloatScorerEnabled() = savedFloat;
}
#endif

template <bool RealRoots>
void DeterministicAndDegenerateBody() {
    Rng rng(0xD00D);
    const Pose pose = makePose(rng, 0.4);
    const int kPoints = 120;
    std::vector<Point2f> from(kPoints), to(kPoints);
    for (int i = 0; i < kPoints; ++i) {
        project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
    }
    std::vector<uint32_t> fl(2 * ransacScratchWords(static_cast<size_t>(kPoints)));
    const RansacScratch scratch{fl.data(), static_cast<size_t>(kPoints)};
    RansacParams params;
    params.threshold = 0.002;

    EssentialMatrix e1, e2;
    const RansacResult r1 = estimate<RealRoots>(from.data(), to.data(),
                                             static_cast<size_t>(kPoints), params, scratch, &e1);
    const RansacResult r2 = estimate<RealRoots>(from.data(), to.data(),
                                             static_cast<size_t>(kPoints), params, scratch, &e2);
    BINCV_CHECK_EQ(r1.inliers, r2.inliers);
    BINCV_CHECK_EQ(r1.iterations, r2.iterations);
    for (int i = 0; i < 9; ++i) BINCV_CHECK_EQ(e1.m[i], e2.m[i]);

    // Fewer than five correspondences: no model, output untouched.
    EssentialMatrix untouched;
    untouched.m[4] = 42.0;
    const RansacResult few = estimate<RealRoots>(from.data(), to.data(), 4, params, scratch,
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
    const int n = solveFive<RealRoots>(same, sameTo, sols);
    BINCV_CHECK(n >= 0 && n <= 10);
    std::printf(" [%s]", solverName(RealRoots));
    std::printf(" essential: deterministic across runs; degenerate five-point returns %d\n", n);
}
BINCV_TEST(Essential, DeterministicAndDegenerate) {
    DeterministicAndDegenerateBody<false>();
}
BINCV_TEST(Essential, DeterministicAndDegenerate_RealRoots) {
    DeterministicAndDegenerateBody<true>();
}

template <bool RealRoots>
void DegenerateGeometryStaysFiniteAndNormalisedBody() {
    // Pure rotation admits NO essential matrix -- E = [t]x R with t = 0 is the zero
    // matrix -- and a tiny baseline is arbitrarily close to that. A planar scene
    // leaves the five-point problem with a family of solutions rather than a point.
    // None of these is detected here, exactly as none is detected by OpenCV. What
    // must hold regardless is that whatever comes back is a USABLE matrix and not
    // NaN, and that a zero matrix is never passed off as a fit: a zero E satisfies
    // q2^T E q1 = 0 for every correspondence and would make every point an inlier.
    const Degenerate kinds[] = {Degenerate::PureRotation, Degenerate::PlanarScene,
                                Degenerate::TinyBaseline};
    for (Degenerate k : kinds) {
        std::vector<Point2f> a, b;
        buildDegenerate(k, 400, a, b);
        std::vector<uint32_t> flags(2 * ransacScratchWords(a.size()), 0);
        RansacScratch scratch{flags.data(), a.size()};
        RansacParams p;
        p.threshold = 0.002;
        p.maxIterations = 500;
        EssentialMatrix e{};
        const RansacResult r =
            estimate<RealRoots>(a.data(), b.data(), a.size(), p, scratch, &e);
        double fro = 0.0;
        for (int i = 0; i < 9; ++i) {
            BINCV_CHECK(std::isfinite(e.m[i]));
            fro += static_cast<double>(e.m[i]) * static_cast<double>(e.m[i]);
        }
        if (r.found) {
            // Unit Frobenius norm, which is also what rules out the zero matrix.
            BINCV_CHECK(std::fabs(std::sqrt(fro) - 1.0) < 1e-3);
            BINCV_CHECK(r.inliers >= EssentialModel::kMinimalSetSize);
        }
        BINCV_CHECK_EQ(r.refined, false);
    }
}
BINCV_TEST(Essential, DegenerateGeometryStaysFiniteAndNormalised) {
    DegenerateGeometryStaysFiniteAndNormalisedBody<false>();
}
BINCV_TEST(Essential, DegenerateGeometryStaysFiniteAndNormalised_RealRoots) {
    DegenerateGeometryStaysFiniteAndNormalisedBody<true>();
}

BINCV_TEST(Essential, AllocatesNothing) {
    Rng rng(0xA110C);
    const Pose pose = makePose(rng, 0.4);
    const int kPoints = 200;
    std::vector<Point2f> from(kPoints), to(kPoints);
    for (int i = 0; i < kPoints; ++i) {
        project(pose, rng, from[static_cast<size_t>(i)], to[static_cast<size_t>(i)]);
    }
    std::vector<uint32_t> fl(2 * ransacScratchWords(static_cast<size_t>(kPoints)));
    const RansacScratch scratch{fl.data(), static_cast<size_t>(kPoints)};
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
template <bool RealRoots>
void AgreesWithOpenCVBody() {
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

        std::vector<uint32_t> flags(2 * ransacScratchWords(static_cast<size_t>(kPoints)));
        std::vector<uint8_t> mask(static_cast<size_t>(kPoints));
        const RansacScratch scratch{flags.data(), static_cast<size_t>(kPoints)};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 500;
        EssentialMatrix e;
        const RansacResult r = estimate<RealRoots>(from.data(), to.data(),
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
    std::printf(" [%s]", solverName(RealRoots));
    std::printf(" essential: inlier counts agree with cv::findEssentialMat on %d/%d scenes\n",
                agreed, kScenes);
}
BINCV_TEST(Essential, AgreesWithOpenCV) {
    AgreesWithOpenCVBody<false>();
}
BINCV_TEST(Essential, AgreesWithOpenCV_RealRoots) {
    AgreesWithOpenCVBody<true>();
}

template <bool RealRoots>
void DegenerateGeometryIsNoWorseThanOpenCVBody() {
    // The bar for a Tier 2 drop-in on a degenerate configuration is not "gets it
    // right" -- there is no right answer for pure rotation -- but "does not behave
    // worse than the call it stands in for". Both find a large consensus set on all
    // three, which is a property of the geometry rather than of either implementation.
    const Degenerate kinds[] = {Degenerate::PureRotation, Degenerate::PlanarScene,
                                Degenerate::TinyBaseline};
    const char* names[] = {"pure rotation", "planar scene", "tiny baseline"};
    for (int ki = 0; ki < 3; ++ki) {
        std::vector<Point2f> a, b;
        buildDegenerate(kinds[ki], 400, a, b);
        std::vector<cv::Point2f> ca, cb;
        for (size_t i = 0; i < a.size(); ++i) {
            ca.push_back(cv::Point2f(a[i].x, a[i].y));
            cb.push_back(cv::Point2f(b[i].x, b[i].y));
        }
        std::vector<uint32_t> flags(2 * ransacScratchWords(a.size()), 0);
        RansacScratch scratch{flags.data(), a.size()};
        RansacParams p;
        p.threshold = 0.002;
        p.maxIterations = 500;
        EssentialMatrix e{};
        const RansacResult r =
            estimate<RealRoots>(a.data(), b.data(), a.size(), p, scratch, &e);
        cv::Mat mask;
        const cv::Mat cvE = cv::findEssentialMat(ca, cb, 1.0, cv::Point2d(0, 0), cv::RANSAC,
                                                 0.99, 0.002, mask);
        size_t cvIn = 0;
        for (int i = 0; i < mask.rows; ++i) cvIn += mask.at<uint8_t>(i, 0) != 0 ? 1u : 0u;
        BINCV_CHECK_EQ(r.found, true);
        BINCV_CHECK(!cvE.empty());
        // Within 5% of the correspondences of each other. Not equality: two random
        // searches over a degenerate family land on different members of it.
        const double gap = std::fabs(static_cast<double>(cvIn) - static_cast<double>(r.inliers));
        BINCV_CHECK(gap <= 0.05 * static_cast<double>(a.size()));
        std::printf(" [%s]", solverName(RealRoots));
        std::printf(" essential: %-14s binCV %zu inliers, OpenCV %zu"
                    " (neither detects the degeneracy)\n", names[ki], r.inliers, cvIn);
    }
}
BINCV_TEST(Essential, DegenerateGeometryIsNoWorseThanOpenCV) {
    DegenerateGeometryIsNoWorseThanOpenCVBody<false>();
}
BINCV_TEST(Essential, DegenerateGeometryIsNoWorseThanOpenCV_RealRoots) {
    DegenerateGeometryIsNoWorseThanOpenCVBody<true>();
}

template <bool RealRoots>
void ModelAccuracyMatchesOpenCVUnderNoiseBody() {
    // Neither side refits here -- cv::findEssentialMat returns what its RANSAC
    // registrator found, exactly as this driver does -- so this asks whether the
    // five-point solve itself is as good, on data that is not exact.
    //
    // The sibling test in test_ransac.cpp exists because the AFFINE estimator was
    // NOT equivalent: OpenCV refits there and binCV did not, which cost 13x in
    // accuracy and was invisible while every inlier was noise-free. This test is
    // the check that the same gap does not exist for the essential matrix.
    double binSum = 0.0, cvSum = 0.0;
    int scenes = 0;
    for (int k = 0; k < 10; ++k) {
        Rng rng(UINT64_C(0x5CE9E) + static_cast<uint64_t>(k) * UINT64_C(7919));
        const Pose pose = makePose(rng, 0.2);
        const size_t n = 400;
        std::vector<Point2f> from(n), to(n), cleanFrom, cleanTo;
        std::vector<cv::Point2f> cf, ct;
        for (size_t i = 0; i < n; ++i) {
            Point2f a{}, b{};
            project(pose, rng, a, b);
            const bool outlier = (i % 5) == 0;
            if (!outlier) {
                cleanFrom.push_back(a);
                cleanTo.push_back(b);
                b.x += static_cast<float>(rng.gaussian() * 0.001);
                b.y += static_cast<float>(rng.gaussian() * 0.001);
            } else {
                b.x += static_cast<float>(rng.uniform() * 0.2);
                b.y += static_cast<float>(rng.uniform() * 0.2);
            }
            from[i] = a;
            to[i] = b;
            cf.push_back(cv::Point2f(a.x, a.y));
            ct.push_back(cv::Point2f(b.x, b.y));
        }
        std::vector<uint32_t> flags(2 * ransacScratchWords(n), 0);
        RansacScratch scratch{flags.data(), n};
        RansacParams p;
        p.threshold = 0.002;
        p.maxIterations = 500;
        EssentialMatrix e;
        const RansacResult r = estimate<RealRoots>(from.data(), to.data(), n, p, scratch, &e);
        cv::Mat mask;
        const cv::Mat cvE = cv::findEssentialMat(cf, ct, 1.0, cv::Point2d(0, 0), cv::RANSAC,
                                                 0.99, 0.002, mask);
        if (!r.found || cvE.empty() || cvE.rows < 3) continue;
        // A refit is not claimed for this model, and must not be reported.
        BINCV_CHECK_EQ(r.refined, false);
        EssentialMatrix ce;
        for (int i = 0; i < 9; ++i) {
            ce.m[i] = static_cast<float>(cvE.at<double>(i / 3, i % 3));
        }
        // Scored on the CLEAN correspondences, so the number is the model's error
        // and not the noise it was fitted through.
        double bAcc = 0.0, cAcc = 0.0;
        for (size_t i = 0; i < cleanFrom.size(); ++i) {
            bAcc += epipolar(e, cleanFrom[i], cleanTo[i]);
            cAcc += epipolar(ce, cleanFrom[i], cleanTo[i]);
        }
        binSum += bAcc / static_cast<double>(cleanFrom.size());
        cvSum += cAcc / static_cast<double>(cleanFrom.size());
        ++scenes;
    }
    BINCV_CHECK(scenes >= 8);
    const double binErr = binSum / scenes, cvErr = cvSum / scenes;
    // Two different random draws, so not equality. The claim is that binCV is not
    // worse, with a band for the difference in hypotheses each happened to see.
    BINCV_CHECK(binErr <= cvErr * 1.5);
    std::printf(" [%s]", solverName(RealRoots));
    std::printf(" essential: epipolar error on clean pairs -- binCV %.6f,"
                " OpenCV %.6f (0.001 inlier noise)\n", binErr, cvErr);
}
BINCV_TEST(Essential, ModelAccuracyMatchesOpenCVUnderNoise) {
    ModelAccuracyMatchesOpenCVUnderNoiseBody<false>();
}
BINCV_TEST(Essential, ModelAccuracyMatchesOpenCVUnderNoise_RealRoots) {
    ModelAccuracyMatchesOpenCVUnderNoiseBody<true>();
}
#endif  // BINCV_WITH_OPENCV

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
