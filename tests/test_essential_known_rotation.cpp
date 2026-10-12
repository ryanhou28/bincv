// The essential matrix under a known rotation, by two-point samples, and what pins it.
//
// Ground truth is planted: a rotation R, a unit translation t, and points seen by both
// cameras, so "correct" needs no oracle. OpenCV has no counterpart to compare against.
// The checks are chosen to fail on the mistakes this solver can make and still return
// plausible matrices:
//
//   * the planted translation must be RECOVERED from any two exact correspondences,
//     and every correspondence of the pose must then satisfy q2^T E q1 = 0
//   * the rotation convention: the transpose of the planted rotation must NOT explain
//     the scene, so a caller who hands over R^T is not silently served
//   * end to end, RANSAC must separate inliers from outliers, and the refit must earn
//     its place by landing closer to the planted translation than the minimal sample
//   * a sample that cannot fix a direction is declined rather than returned
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "bincv/ops/essentialKnownRotation.hpp"
#include "test_util.hpp"

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
    double gaussian() {
        const double u1 = 0.5 * (uniform() + 1.0) + 1e-9, u2 = 0.5 * (uniform() + 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

/// @brief A camera pair: `r` row-major, `t` unit, and `e = [t]x R / sqrt(2)`.
struct Pose {
    double r[9];
    double t[3];
    double e[9];
};

Pose makePose(Rng& rng, double angleScale) {
    Pose p;
    const double ax = rng.uniform() * angleScale, ay = rng.uniform() * angleScale,
                 az = rng.uniform() * angleScale;
    const double ca = std::cos(ax), sa = std::sin(ax), cb = std::cos(ay), sb = std::sin(ay),
                 cc = std::cos(az), sc = std::sin(az);
    const double r[9] = {cb * cc,
                         -cb * sc,
                         sb,
                         sa * sb * cc + ca * sc,
                         -sa * sb * sc + ca * cc,
                         -sa * cb,
                         -ca * sb * cc + sa * sc,
                         ca * sb * sc + sa * cc,
                         ca * cb};
    for (int i = 0; i < 9; ++i) p.r[i] = r[i];
    p.t[0] = rng.uniform();
    p.t[1] = rng.uniform();
    p.t[2] = 1.0 + 0.3 * rng.uniform();
    const double tn = std::sqrt(p.t[0] * p.t[0] + p.t[1] * p.t[1] + p.t[2] * p.t[2]);
    for (double& v : p.t) v /= tn;
    const double tx[9] = {0, -p.t[2], p.t[1], p.t[2], 0, -p.t[0], -p.t[1], p.t[0], 0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += tx[i * 3 + k] * p.r[k * 3 + j];
            p.e[i * 3 + j] = acc / std::sqrt(2.0);
        }
    }
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
        for (int k = 0; k < 3; ++k) acc += pose.r[i * 3 + k] * x[k];
        xc[i] = acc + pose.t[i];
    }
    b.x = static_cast<float>(xc[0] / xc[2]);
    b.y = static_cast<float>(xc[1] / xc[2]);
}

/// @brief A scene of `n` correspondences, every `outlierEvery`-th displaced (0: none),
/// the rest carrying Gaussian noise of `sigma` on both points. `inlier[i]` says which.
void buildScene(const Pose& pose, Rng& rng, size_t n, size_t outlierEvery, double sigma,
                std::vector<Point2f>& from, std::vector<Point2f>& to,
                std::vector<uint8_t>& inlier) {
    from.resize(n);
    to.resize(n);
    inlier.assign(n, 1);
    for (size_t i = 0; i < n; ++i) {
        project(pose, rng, from[i], to[i]);
        if (outlierEvery != 0 && i % outlierEvery == 0) {
            to[i].x += static_cast<float>(rng.uniform() * 0.3);
            to[i].y += static_cast<float>(rng.uniform() * 0.3);
            inlier[i] = 0;
        } else {
            from[i].x += static_cast<float>(sigma * rng.gaussian());
            from[i].y += static_cast<float>(sigma * rng.gaussian());
            to[i].x += static_cast<float>(sigma * rng.gaussian());
            to[i].y += static_cast<float>(sigma * rng.gaussian());
        }
    }
}

/// @brief `|q2^T E q1|` for one correspondence.
double algebraic(const EssentialMatrix& e, Point2f a, Point2f b) {
    const double q1[3] = {static_cast<double>(a.x), static_cast<double>(a.y), 1.0};
    const double q2[3] = {static_cast<double>(b.x), static_cast<double>(b.y), 1.0};
    double acc = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) acc += q2[i] * e.m[i * 3 + j] * q1[j];
    }
    return std::fabs(acc);
}

/// @brief The largest entry of `e - planted`, taking whichever sign of `e` is closer.
double distanceToPlanted(const EssentialMatrix& e, const Pose& pose) {
    double plus = 0.0, minus = 0.0;
    for (int i = 0; i < 9; ++i) {
        plus = std::max(plus, std::fabs(e.m[i] - pose.e[i]));
        minus = std::max(minus, std::fabs(e.m[i] + pose.e[i]));
    }
    return std::min(plus, minus);
}

/// @brief The angle, in degrees, between the planted translation and the one `e`
/// carries: `E R^T = [t]x / sqrt(2)`, sign ignored.
double translationErrorDeg(const EssentialMatrix& e, const Pose& pose) {
    double m[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) acc += e.m[i * 3 + k] * pose.r[j * 3 + k];
            m[i * 3 + j] = acc;
        }
    }
    const double t[3] = {m[7] - m[5], m[2] - m[6], m[3] - m[1]};
    const double tn = std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
    const double c = std::fabs(t[0] * pose.t[0] + t[1] * pose.t[1] + t[2] * pose.t[2]) / tn;
    return std::acos(std::min(1.0, c)) * 57.29577951308232;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

BINCV_TEST(KnownRotation, TwoPointRecoversThePlantedTranslation) {
    // Exact correspondences (float positions, which is what bounds the precision): any
    // two that are not degenerate give back the planted matrix, and that matrix holds
    // every other correspondence of the pose on its epipolar line.
    Rng rng(0x2B0C);
    size_t poses = 0, recovered = 0, declined = 0;
    double worstMatch = 0.0, worstAlgebraic = 0.0;
    for (int trial = 0; trial < 300; ++trial) {
        const Pose pose = makePose(rng, 0.4);
        const KnownRotationEssentialModel model(pose.r);
        Point2f from[12], to[12];
        for (int i = 0; i < 12; ++i) project(pose, rng, from[i], to[i]);
        for (size_t s = 0; s + 1 < 12; s += 2) {
            const size_t idx[2] = {s, s + 1};
            EssentialMatrix e;
            const size_t n = model.estimate(from, to, idx, &e);
            BINCV_CHECK(n <= 1);
            if (n == 0) {
                ++declined;
                continue;
            }
            ++poses;
            double fro = 0.0;
            for (double v : e.m) fro += v * v;
            BINCV_CHECK(std::fabs(std::sqrt(fro) - 1.0) < 1e-12);
            const double d = distanceToPlanted(e, pose);
            worstMatch = std::max(worstMatch, d);
            if (d < 1e-3) ++recovered;
            for (int i = 0; i < 12; ++i) {
                worstAlgebraic = std::max(worstAlgebraic, algebraic(e, from[i], to[i]));
            }
        }
    }
    std::printf(" known rotation: %zu of %zu samples recover the planted E to 1e-3 "
                "(worst %.2e), %zu declined; worst |q2^T E q1| %.2e\n",
                recovered, poses, worstMatch, declined, worstAlgebraic);
    BINCV_CHECK_EQ(declined, static_cast<size_t>(0));
    BINCV_CHECK_EQ(recovered, poses);
    BINCV_CHECK(worstAlgebraic < 1e-5);
}

BINCV_TEST(KnownRotation, TheTransposeDoesNotExplainTheScene) {
    // The convention is the caller's easiest mistake: a column-major library's raw data
    // is R^T. Under the planted rotation every exact correspondence is an inlier; under
    // its transpose almost none may be, or a flipped rotation would pass unnoticed.
    Rng rng(0x7A45);
    for (int trial = 0; trial < 20; ++trial) {
        const Pose pose = makePose(rng, 0.4);
        std::vector<Point2f> from, to;
        std::vector<uint8_t> inlier;
        buildScene(pose, rng, 200, 0, 0.0, from, to, inlier);
        double rt[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) rt[i * 3 + j] = pose.r[j * 3 + i];
        std::vector<uint32_t> flags(2 * ransacScratchWords(from.size()));
        const RansacScratch scratch{flags.data(), from.size()};
        RansacParams params;
        params.threshold = 0.002;
        params.maxIterations = 200;
        EssentialMatrix e;
        const RansacResult right = essentialFromKnownRotation(
            from.data(), to.data(), from.size(), pose.r, params, scratch, &e);
        const RansacResult wrong = essentialFromKnownRotation(
            from.data(), to.data(), from.size(), rt, params, scratch, &e);
        BINCV_CHECK(right.found);
        BINCV_CHECK_EQ(right.inliers, from.size());
        BINCV_CHECK(wrong.inliers < from.size() / 4);
    }
}

BINCV_TEST(KnownRotation, RansacSeparatesInliersFromOutliers) {
    // A third and a half of the correspondences displaced, the inliers noisy: the mask
    // must find the planted inliers, the translation must land near the planted one, and
    // the stopping rule must need fewer hypotheses than the five-point estimator does on
    // the same scene.
    Rng rng(0x5E9A);
    size_t missed = 0, admitted = 0, planted = 0, outliers = 0;
    double worstAngle = 0.0;
    int iterations2 = 0, iterations5 = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const Pose pose = makePose(rng, 0.3);
        std::vector<Point2f> from, to;
        std::vector<uint8_t> inlier;
        buildScene(pose, rng, 300, trial % 2 == 0 ? 3 : 2, 0.0005, from, to, inlier);
        std::vector<uint32_t> flags(2 * ransacScratchWords(from.size()));
        const RansacScratch scratch{flags.data(), from.size()};
        RansacParams params;
        params.threshold = 0.003;
        params.confidence = 0.999;
        params.maxIterations = 1000;
        params.seed = static_cast<uint64_t>(trial) + 1;
        std::vector<uint8_t> mask(from.size());
        EssentialMatrix e, e5;
        const RansacResult r = essentialFromKnownRotation(
            from.data(), to.data(), from.size(), pose.r, params, scratch, &e, mask.data());
        const RansacResult r5 =
            findEssentialMat(from.data(), to.data(), from.size(), params, scratch, &e5);
        BINCV_CHECK(r.found);
        BINCV_CHECK(r.refined);
        iterations2 += r.iterations;
        iterations5 += r5.iterations;
        worstAngle = std::max(worstAngle, translationErrorDeg(e, pose));
        for (size_t i = 0; i < from.size(); ++i) {
            if (inlier[i]) {
                ++planted;
                missed += mask[i] == 0;
            } else {
                ++outliers;
                admitted += mask[i] != 0;
            }
        }
    }
    std::printf(" known rotation: missed %zu of %zu planted inliers, admitted %zu of %zu "
                "outliers; worst translation %.3f deg; %d hypotheses against the "
                "five-point's %d\n",
                missed, planted, admitted, outliers, worstAngle, iterations2, iterations5);
    BINCV_CHECK(missed * 100 < planted);       // under 1% of the inliers lost
    BINCV_CHECK(admitted * 20 < outliers);     // under 5% of the outliers let in
    BINCV_CHECK(worstAngle < 2.0);
    BINCV_CHECK(iterations2 * 4 < iterations5);
}

BINCV_TEST(KnownRotation, TheRefitLandsCloserThanTheMinimalSample) {
    // The refit is on by default, so it has to be worth its cost: over noisy scenes the
    // refitted translation must sit closer to the planted one than the minimal sample's
    // does, by a clear margin in the median.
    Rng rng(0x4EF1);
    std::vector<double> minimal, refitted;
    for (int trial = 0; trial < 100; ++trial) {
        const Pose pose = makePose(rng, 0.3);
        std::vector<Point2f> from, to;
        std::vector<uint8_t> inlier;
        buildScene(pose, rng, 160, 3, 0.001, from, to, inlier);
        std::vector<uint32_t> flags(2 * ransacScratchWords(from.size()));
        const RansacScratch scratch{flags.data(), from.size()};
        RansacParams params;
        params.threshold = 0.005;
        params.seed = static_cast<uint64_t>(trial) + 7;
        EssentialMatrix a, b;
        params.refine = false;
        const RansacResult ra = essentialFromKnownRotation(
            from.data(), to.data(), from.size(), pose.r, params, scratch, &a);
        params.refine = true;
        const RansacResult rb = essentialFromKnownRotation(
            from.data(), to.data(), from.size(), pose.r, params, scratch, &b);
        BINCV_CHECK(ra.found && rb.found);
        BINCV_CHECK_EQ(ra.refined, false);
        BINCV_CHECK_EQ(rb.refined, true);
        // The refit never moves the consensus the stopping rule used.
        BINCV_CHECK_EQ(ra.inliers, rb.inliers);
        minimal.push_back(translationErrorDeg(a, pose));
        refitted.push_back(translationErrorDeg(b, pose));
    }
    const double mMin = median(minimal), mRef = median(refitted);
    std::printf(" known rotation: median translation error %.3f deg minimal, %.3f deg "
                "refitted\n",
                mMin, mRef);
    BINCV_CHECK(mRef * 2.0 < mMin);
}

BINCV_TEST(KnownRotation, DeterministicAndDegenerate) {
    Rng rng(0xD0D0);
    const Pose pose = makePose(rng, 0.4);
    std::vector<Point2f> from, to;
    std::vector<uint8_t> inlier;
    buildScene(pose, rng, 120, 4, 0.0005, from, to, inlier);
    std::vector<uint32_t> fl(2 * ransacScratchWords(from.size()));
    const RansacScratch scratch{fl.data(), from.size()};
    RansacParams params;
    params.threshold = 0.003;

    EssentialMatrix e1, e2;
    const RansacResult r1 = essentialFromKnownRotation(from.data(), to.data(), from.size(),
                                                       pose.r, params, scratch, &e1);
    const RansacResult r2 = essentialFromKnownRotation(from.data(), to.data(), from.size(),
                                                       pose.r, params, scratch, &e2);
    BINCV_CHECK_EQ(r1.inliers, r2.inliers);
    BINCV_CHECK_EQ(r1.iterations, r2.iterations);
    BINCV_CHECK(std::memcmp(e1.m, e2.m, sizeof e1.m) == 0);

    // One correspondence: no model, output untouched.
    EssentialMatrix untouched;
    untouched.m[4] = 42.0;
    const RansacResult few = essentialFromKnownRotation(from.data(), to.data(), 1, pose.r,
                                                        params, scratch, &untouched);
    BINCV_CHECK_EQ(few.found, false);
    BINCV_CHECK_EQ(untouched.m[4], 42.0);

    const KnownRotationEssentialModel model(pose.r);
    const size_t pair[2] = {0, 1};
    EssentialMatrix e;
    // The same correspondence twice: one plane, no direction.
    const Point2f same[2] = {from[1], from[1]}, sameTo[2] = {to[1], to[1]};
    BINCV_CHECK_EQ(model.estimate(same, sameTo, pair, &e), static_cast<size_t>(0));
    // No parallax: q2 is exactly R q1, so n vanishes.
    const double id[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const KnownRotationEssentialModel identity(id);
    const Point2f still[2] = {Point2f{0.1f, -0.2f}, Point2f{-0.3f, 0.05f}};
    BINCV_CHECK_EQ(identity.estimate(still, still, pair, &e), static_cast<size_t>(0));

    // A scene with no translation at all. In exact arithmetic every sample would be
    // declined; with float positions the constraints are rounding noise, a direction
    // comes back, and every correspondence fits it -- the five-point estimator cannot
    // tell pure rotation either. What must hold is that a zero matrix, which every
    // correspondence would satisfy, is never returned: a model is unit-norm or absent.
    std::vector<Point2f> rotated(from.size());
    for (size_t i = 0; i < from.size(); ++i) {
        const double x = static_cast<double>(from[i].x), y = static_cast<double>(from[i].y);
        const double p[3] = {pose.r[0] * x + pose.r[1] * y + pose.r[2],
                             pose.r[3] * x + pose.r[4] * y + pose.r[5],
                             pose.r[6] * x + pose.r[7] * y + pose.r[8]};
        rotated[i] = Point2f{static_cast<float>(p[0] / p[2]), static_cast<float>(p[1] / p[2])};
    }
    EssentialMatrix none;
    none.m[0] = 7.0;
    const RansacResult pure = essentialFromKnownRotation(from.data(), rotated.data(),
                                                         from.size(), pose.r, params, scratch,
                                                         &none);
    std::printf(" known rotation: deterministic; a translation-free scene finds %s\n",
                pure.found ? "a model" : "nothing");
    if (pure.found) {
        double fro = 0.0;
        for (double v : none.m) fro += v * v;
        BINCV_CHECK(std::fabs(std::sqrt(fro) - 1.0) < 1e-9);
    } else {
        BINCV_CHECK_EQ(none.m[0], 7.0);
    }
}

#if BINCV_RANSAC_SQUARED_TEST
BINCV_TEST(KnownRotation, SquaredTestScoringIsBitExact) {
    // Inherited from EssentialModel, so the three scorings must agree here too: the same
    // iterations, support, model and mask.
    const bool saved = impl::ransacSquaredTestEnabled();
    const bool savedFloat = impl::ransacFloatScorerEnabled();
    Rng rng(0xB17E);
    for (int scene = 0; scene < 20; ++scene) {
        const Pose pose = makePose(rng, 0.4);
        std::vector<Point2f> from, to;
        std::vector<uint8_t> inlier;
        buildScene(pose, rng, 100 + static_cast<size_t>(scene) * 37, 4, 0.001, from, to,
                   inlier);
        const size_t n = from.size();
        std::vector<uint32_t> flags(2 * ransacScratchWords(n));
        const RansacScratch scratch{flags.data(), n};
        RansacParams params;
        params.threshold = 0.002;
        params.seed = static_cast<uint64_t>(scene) + 1;
        std::vector<uint8_t> maskA(n), maskB(n), maskC(n);
        EssentialMatrix ea, eb, ec;
        impl::ransacSquaredTestEnabled() = false;
        const RansacResult ra = essentialFromKnownRotation(from.data(), to.data(), n, pose.r,
                                                           params, scratch, &ea, maskA.data());
        impl::ransacSquaredTestEnabled() = true;
        impl::ransacFloatScorerEnabled() = false;
        const RansacResult rb = essentialFromKnownRotation(from.data(), to.data(), n, pose.r,
                                                           params, scratch, &eb, maskB.data());
        impl::ransacFloatScorerEnabled() = true;
        const RansacResult rc = essentialFromKnownRotation(from.data(), to.data(), n, pose.r,
                                                           params, scratch, &ec, maskC.data());
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

BINCV_TEST(KnownRotation, AllocatesNothing) {
    Rng rng(0xA110);
    const Pose pose = makePose(rng, 0.3);
    std::vector<Point2f> from, to;
    std::vector<uint8_t> inlier;
    buildScene(pose, rng, 200, 3, 0.0005, from, to, inlier);
    std::vector<uint32_t> flags(2 * ransacScratchWords(from.size()));
    std::vector<uint8_t> mask(from.size());
    RansacParams params;
    params.threshold = 0.003;
    EssentialMatrix e;
    const std::size_t before = g_newCount;
    const RansacResult r = essentialFromKnownRotation(
        from.data(), to.data(), from.size(), pose.r, params,
        RansacScratch{flags.data(), from.size()}, &e, mask.data());
    BINCV_CHECK(r.found);
    BINCV_CHECK_EQ(g_newCount, before);
}

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

BINCV_TEST_MAIN("test_essential_known_rotation")
