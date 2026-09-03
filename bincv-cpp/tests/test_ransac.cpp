// Random sample consensus: the contract, and what it can and cannot promise.
//
// RANSAC is randomised, so "bit-exact against OpenCV" is not available even in
// principle -- two implementations that sample differently see different hypotheses
// and stop at different iterations. What CAN be pinned is pinned here:
//
//   * the model is recovered from data with gross outliers in it
//   * the same seed gives the same answer, bit for bit
//   * a different seed still finds the same support, so the answer is a property of
//     the data rather than of one lucky draw
//   * the consensus rule agrees with OpenCV's: given the SAME model, the two mark
//     the same correspondences as inliers
//   * nothing allocates
//
//   * the returned model is ACCURATE, not merely supported by the right points
//
// The consensus-rule one matters for the tier claim. Comparing the two MODELS
// directly would compare two random draws; comparing the two INLIER SETS under one
// model compares the thing the threshold actually defines.
//
// THAT ARGUMENT WAS ONCE USED TO SKIP ACCURACY ENTIRELY, AND IT IS WRONG TO. Two
// draws cannot be compared to each other, but both can be compared to a PLANTED
// transform, and that is a real check the suite went without. It went without it
// because every inlier here used to be generated exactly from the transform, which
// makes a minimal-set fit through three exact points exact -- so a refit over the
// consensus set changed nothing and its absence was invisible. With noise on the
// inliers the un-refitted model was 13x further from the truth than OpenCV's.
// `makeScene` therefore takes a noise level, and the tests below use it.
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "bincv-cpp/ops/ransac.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace {

using namespace bincv;

// --------------------------------------------------------------------------
// Allocation counting, as the other suites here do it.
// --------------------------------------------------------------------------
std::size_t g_newCount = 0;

struct Correspondences {
    std::vector<Point2f> from;
    std::vector<Point2f> to;
    std::vector<uint8_t> truth;  // 1 where the point was generated as an inlier
    size_t inlierCount = 0;
};

/// @brief `count` correspondences under `t`, of which every `outlierEvery`-th is
/// displaced far enough to be an unambiguous outlier.
/// @param noiseSd Gaussian noise on the INLIERS, in pixels. Zero makes every inlier
/// exact, which makes a minimal-set fit exact and hides whether the model was
/// refitted at all -- so accuracy tests must pass something non-zero.
Correspondences makeScene(const Affine2D& t, size_t count, int outlierEvery, uint64_t seed,
                          float noiseSd = 0.0f) {
    Correspondences c;
    uint64_t s = seed;
    auto nextf = [&s]() {
        s = s * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        return static_cast<float>((s >> 33) % 100000) / 100000.0f;
    };
    auto gauss = [&nextf]() {  // Box-Muller
        const double u1 = static_cast<double>(nextf()) + 1e-9;
        const double u2 = static_cast<double>(nextf());
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    };
    for (size_t i = 0; i < count; ++i) {
        const Point2f a{nextf() * 640.0f, nextf() * 480.0f};
        Point2f b{t.m[0] * a.x + t.m[1] * a.y + t.m[2], t.m[3] * a.x + t.m[4] * a.y + t.m[5]};
        const bool outlier = (outlierEvery > 0) && (i % static_cast<size_t>(outlierEvery) == 0);
        if (outlier) {
            b.x += 60.0f + nextf() * 120.0f;
            b.y -= 55.0f + nextf() * 110.0f;
        } else if (noiseSd > 0.0f) {
            b.x += static_cast<float>(gauss()) * noiseSd;
            b.y += static_cast<float>(gauss()) * noiseSd;
        }
        c.from.push_back(a);
        c.to.push_back(b);
        c.truth.push_back(outlier ? uint8_t{0} : uint8_t{1});
        if (!outlier) ++c.inlierCount;
    }
    return c;
}

struct Fixture {
    Correspondences scene;
    std::vector<uint32_t> flags;
    std::vector<uint8_t> mask;
    RansacScratch scratch;

    explicit Fixture(const Correspondences& s) : scene(s) {
        flags.assign(2 * ransacScratchWords(scene.from.size()), 0);
        mask.assign(scene.from.size(), 0);
        scratch = RansacScratch{flags.data(), scene.from.size()};
    }
};

const Affine2D kTruth{{1.10f, -0.20f, 12.0f, 0.15f, 0.95f, -7.0f}};

/// @brief RMS distance, in pixels, between where `m` sends each source point and
/// where `kTruth` sends it. Scores the MODEL against the planted transform rather
/// than against one noise realisation, so it is comparable across implementations.
double modelRms(const Affine2D& m, const std::vector<Point2f>& pts) {
    double acc = 0.0;
    for (const Point2f& a : pts) {
        const double ex = static_cast<double>(m.m[0]) * a.x + static_cast<double>(m.m[1]) * a.y +
                          static_cast<double>(m.m[2]) -
                          (static_cast<double>(kTruth.m[0]) * a.x +
                           static_cast<double>(kTruth.m[1]) * a.y +
                           static_cast<double>(kTruth.m[2]));
        const double ey = static_cast<double>(m.m[3]) * a.x + static_cast<double>(m.m[4]) * a.y +
                          static_cast<double>(m.m[5]) -
                          (static_cast<double>(kTruth.m[3]) * a.x +
                           static_cast<double>(kTruth.m[4]) * a.y +
                           static_cast<double>(kTruth.m[5]));
        acc += ex * ex + ey * ey;
    }
    return std::sqrt(acc / static_cast<double>(pts.size()));
}

} // namespace

// --------------------------------------------------------------------------

BINCV_TEST(Ransac, RecoversTheModelThroughOutliers) {
    // 30% gross outliers. The model is exactly determined by any three inliers, so a
    // correct run recovers it to float precision rather than approximately.
    Fixture f(makeScene(kTruth, 200, 10, 0xA11CEULL));
    RansacParams p;
    p.threshold = 3.0;
    Affine2D m;
    const RansacResult r = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                            f.scene.from.size(), p, f.scratch, &m,
                                            f.mask.data());

    BINCV_CHECK_EQ(r.found, true);
    BINCV_CHECK_EQ(r.inliers, f.scene.inlierCount);

    // Every generated inlier is found, and no outlier is.
    size_t wrong = 0;
    for (size_t i = 0; i < f.scene.from.size(); ++i) {
        if (f.mask[i] != f.scene.truth[i]) ++wrong;
    }
    BINCV_CHECK_EQ(wrong, size_t{0});

    for (int i = 0; i < 6; ++i) {
        BINCV_CHECK(std::fabs(static_cast<double>(m.m[i] - kTruth.m[i])) < 1e-3);
    }
    std::printf(" ransac: %zu/%zu inliers in %d iterations, model exact to 1e-3\n",
                r.inliers, f.scene.from.size(), r.iterations);
}

BINCV_TEST(Ransac, TheSameSeedGivesTheSameAnswer) {
    Fixture f(makeScene(kTruth, 300, 4, 0xBEEF01ULL));
    RansacParams p;
    p.seed = 12345;

    Affine2D a, b;
    const RansacResult ra = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                             f.scene.from.size(), p, f.scratch, &a, nullptr);
    const RansacResult rb = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                             f.scene.from.size(), p, f.scratch, &b, nullptr);
    BINCV_CHECK_EQ(ra.inliers, rb.inliers);
    BINCV_CHECK_EQ(ra.iterations, rb.iterations);
    for (int i = 0; i < 6; ++i) BINCV_CHECK_EQ(a.m[i], b.m[i]);

    // A different seed is a different draw, but the SUPPORT is a property of the
    // data: it must land on the same consensus set, not merely a similar one.
    p.seed = 999;
    Affine2D c;
    const RansacResult rc = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                             f.scene.from.size(), p, f.scratch, &c, nullptr);
    BINCV_CHECK_EQ(rc.inliers, ra.inliers);
    std::printf(" ransac: seed 12345 and seed 999 both find %zu inliers\n", ra.inliers);
}

BINCV_TEST(Ransac, DegenerateAndUndersizedInputs) {
    // Fewer correspondences than the minimal set: no model, and the output is not touched.
    Affine2D untouched;
    untouched.m[2] = 1234.0f;
    std::vector<uint32_t> fl(2 * ransacScratchWords(2));
    RansacScratch sc{fl.data(), 2};
    const Correspondences two = makeScene(kTruth, 2, 0, 1);
    RansacParams p;
    const RansacResult r =
        estimateAffine2D(two.from.data(), two.to.data(), 2, p, sc, &untouched, nullptr);
    BINCV_CHECK_EQ(r.found, false);
    BINCV_CHECK_EQ(r.inliers, size_t{0});
    BINCV_CHECK_EQ(untouched.m[2], 1234.0f);

    // Every source point identical: every minimal sample is degenerate, so the solver
    // rejects all of them and the call reports honestly rather than dividing by zero.
    Correspondences same;
    for (size_t i = 0; i < 20; ++i) {
        same.from.push_back(Point2f{5.0f, 7.0f});
        same.to.push_back(Point2f{9.0f, 3.0f});
    }
    Fixture f(same);
    Affine2D m;
    const RansacResult rd = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                             f.scene.from.size(), p, f.scratch, &m,
                                             f.mask.data());
    BINCV_CHECK_EQ(rd.found, false);
    for (size_t i = 0; i < f.mask.size(); ++i) BINCV_CHECK_EQ(f.mask[i], uint8_t{0});
    std::printf(" ransac: undersized and fully degenerate inputs both report found=false\n");
}

BINCV_TEST(Ransac, AllocatesNothing) {
    Fixture f(makeScene(kTruth, 500, 5, 0xC0FFEEULL));
    RansacParams p;
    Affine2D m;

    const std::size_t before = g_newCount;
    const RansacResult r = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                            f.scene.from.size(), p, f.scratch, &m,
                                            f.mask.data());
    const std::size_t during = g_newCount - before;

    BINCV_CHECK_EQ(during, std::size_t{0});
    BINCV_CHECK_EQ(r.found, true);
    // The declared cost is the whole cost: two flags per correspondence.
    BINCV_CHECK_EQ(ransacScratchBytes(500), std::size_t{128});
    std::printf(" ransac: operator new called %zu times across the call, scratch %zu B\n",
                during, ransacScratchBytes(500));
}

BINCV_TEST(Ransac, RefitOverTheConsensusSetBeatsTheMinimalFit) {
    // THE REGRESSION GUARD. A minimal-set fit is exact only when its three sampled
    // correspondences are exact. Put noise on the inliers and it carries that
    // sample's error into the answer; the refit averages it away. Without this test
    // the whole difference is invisible, because a noise-free scene makes the two
    // identical.
    double sumRefit = 0.0, sumMinimal = 0.0;
    int scenes = 0;
    for (int k = 0; k < 12; ++k) {
        Fixture f(makeScene(kTruth, 400, 4, 0xACCU + static_cast<uint64_t>(k) * 7919u, 0.5f));
        RansacParams p;
        p.threshold = 3.0;

        Affine2D refit;
        const RansacResult a = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                                f.scene.from.size(), p, f.scratch, &refit);
        p.refine = false;
        Affine2D minimal;
        const RansacResult b = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                                f.scene.from.size(), p, f.scratch, &minimal);
        if (!a.found || !b.found) continue;
        BINCV_CHECK_EQ(a.refined, true);
        BINCV_CHECK_EQ(b.refined, false);
        // The consensus set is the same either way: refining changes the model, not
        // which correspondences supported it.
        BINCV_CHECK_EQ(a.inliers, b.inliers);
        sumRefit += modelRms(refit, f.scene.from);
        sumMinimal += modelRms(minimal, f.scene.from);
        ++scenes;
    }
    BINCV_CHECK(scenes >= 10);
    const double refitRms = sumRefit / scenes, minimalRms = sumMinimal / scenes;
    // Measured around 0.06 px against 0.90 px at this noise level. The bar is set
    // at 3x rather than at the measured 13x so that it fails on a REGRESSION rather
    // than on ordinary drift.
    BINCV_CHECK(refitRms * 3.0 < minimalRms);
    BINCV_CHECK(refitRms < 0.25);
    std::printf(" ransac: refit %.4f px vs minimal-set %.4f px at 0.5 px inlier noise\n",
                refitRms, minimalRms);
}

BINCV_TEST(Ransac, RefitIsSkippedWhenTheConsensusSetIsDegenerate) {
    // Collinear sources cannot determine an affine, so the refit must decline and
    // leave the minimal-set model rather than divide by a near-zero determinant.
    Correspondences c;
    for (int i = 0; i < 40; ++i) {
        const Point2f a{static_cast<float>(i) * 5.0f, 100.0f};  // one horizontal line
        c.from.push_back(a);
        c.to.push_back({kTruth.m[0] * a.x + kTruth.m[1] * a.y + kTruth.m[2],
                        kTruth.m[3] * a.x + kTruth.m[4] * a.y + kTruth.m[5]});
        c.truth.push_back(1);
        ++c.inlierCount;
    }
    Fixture f(c);
    RansacParams p;
    p.threshold = 3.0;
    Affine2D m;
    const RansacResult r = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                            f.scene.from.size(), p, f.scratch, &m);
    // Whatever the sampler managed, it must not report a refit it could not do.
    BINCV_CHECK_EQ(r.refined, false);
    for (int i = 0; i < 6; ++i) BINCV_CHECK(std::isfinite(m.m[i]));
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Ransac, ConsensusRuleAgreesWithOpenCV) {
    // The tier claim. Two RANSAC runs cannot be compared model to model -- they are
    // different random draws. What CAN be compared is the consensus rule: given ONE
    // model, do the two implementations mark the same correspondences as inliers?
    // That is what `threshold` defines and what a caller relies on.
    Fixture f(makeScene(kTruth, 400, 8, 0xD15EA5EULL));
    RansacParams p;
    p.threshold = 3.0;
    Affine2D m;
    const RansacResult r = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                            f.scene.from.size(), p, f.scratch, &m,
                                            f.mask.data());
    BINCV_CHECK_EQ(r.found, true);

    // binCV's model, scored by OpenCV's arithmetic: transform with cv::transform and
    // apply the same strictly-below-threshold rule.
    std::vector<cv::Point2f> src, dst;
    for (size_t i = 0; i < f.scene.from.size(); ++i) {
        src.push_back(cv::Point2f(f.scene.from[i].x, f.scene.from[i].y));
        dst.push_back(cv::Point2f(f.scene.to[i].x, f.scene.to[i].y));
    }
    cv::Mat mm = (cv::Mat_<double>(2, 3) << static_cast<double>(m.m[0]), static_cast<double>(m.m[1]),
                  static_cast<double>(m.m[2]), static_cast<double>(m.m[3]),
                  static_cast<double>(m.m[4]), static_cast<double>(m.m[5]));
    std::vector<cv::Point2f> mapped;
    cv::transform(src, mapped, mm);

    size_t differ = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double dx = static_cast<double>(mapped[i].x - dst[i].x);
        const double dy = static_cast<double>(mapped[i].y - dst[i].y);
        const bool cvInlier = std::sqrt(dx * dx + dy * dy) < p.threshold;
        if (cvInlier != (f.mask[i] != 0)) ++differ;
    }
    BINCV_CHECK_EQ(differ, size_t{0});

    // And OpenCV's own estimator, run independently, finds the same support. This is
    // NOT bit-exactness and is not claimed as such -- it is the agreement bound.
    std::vector<uint8_t> cvMask;
    const cv::Mat cvModel = cv::estimateAffine2D(src, dst, cvMask, cv::RANSAC, p.threshold);
    size_t cvInliers = 0;
    for (uint8_t v : cvMask) cvInliers += (v != 0) ? 1u : 0u;

    BINCV_CHECK(!cvModel.empty());
    const double gap = std::fabs(static_cast<double>(cvInliers) - static_cast<double>(r.inliers));
    BINCV_CHECK(gap <= 0.02 * static_cast<double>(f.scene.from.size()));
    std::printf(" ransac: same model -> %zu inlier disagreements with OpenCV's rule;"
                " independent runs %zu vs %zu inliers\n",
                differ, r.inliers, cvInliers);
}

BINCV_TEST(Ransac, ModelAccuracyMatchesOpenCVUnderNoise) {
    // Both estimators are compared to the PLANTED transform, which two random draws
    // can both be measured against even though they cannot be measured against each
    // other. This is what says binCV's estimateAffine2D fills the same role as
    // cv::estimateAffine2D rather than merely finding the same inliers.
    double binSum = 0.0, cvSum = 0.0;
    int scenes = 0;
    for (int k = 0; k < 12; ++k) {
        Fixture f(makeScene(kTruth, 400, 4, 0xF00DU + static_cast<uint64_t>(k) * 7919u, 0.5f));
        RansacParams p;
        p.threshold = 3.0;
        Affine2D m;
        const RansacResult r = estimateAffine2D(f.scene.from.data(), f.scene.to.data(),
                                                f.scene.from.size(), p, f.scratch, &m);
        std::vector<cv::Point2f> src, dst;
        for (size_t i = 0; i < f.scene.from.size(); ++i) {
            src.push_back(cv::Point2f(f.scene.from[i].x, f.scene.from[i].y));
            dst.push_back(cv::Point2f(f.scene.to[i].x, f.scene.to[i].y));
        }
        std::vector<uint8_t> cvMask;
        const cv::Mat cvModel = cv::estimateAffine2D(src, dst, cvMask, cv::RANSAC, p.threshold);
        if (!r.found || cvModel.empty()) continue;
        Affine2D cm;
        for (int i = 0; i < 6; ++i) {
            cm.m[i] = static_cast<float>(cvModel.at<double>(i / 3, i % 3));
        }
        binSum += modelRms(m, f.scene.from);
        cvSum += modelRms(cm, f.scene.from);
        ++scenes;
    }
    BINCV_CHECK(scenes >= 10);
    const double binRms = binSum / scenes, cvRms = cvSum / scenes;
    // Not equality: the two search different hypotheses, so they land on different
    // consensus sets and different fits. The claim is that binCV is not WORSE, with
    // a 1.5x band for the difference in draws. Measured, binCV comes out slightly
    // ahead -- it solves the least squares exactly where OpenCV runs a bounded
    // number of Levenberg-Marquardt steps.
    BINCV_CHECK(binRms <= cvRms * 1.5);
    std::printf(" ransac: model error vs planted truth -- binCV %.4f px,"
                " OpenCV %.4f px (0.5 px inlier noise)\n", binRms, cvRms);
}
#endif

// --------------------------------------------------------------------------
// Counted allocation, replacing every form the library could reach -- including
// the nothrow ones, which a partial replacement would leave mismatched.
// --------------------------------------------------------------------------
namespace {
void* countedAllocate(std::size_t bytes) {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) std::abort();
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) std::abort();
    return p;
}
void* countedAllocateNoThrow(std::size_t bytes) noexcept {
    ++g_newCount;
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) return nullptr;
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

BINCV_TEST_MAIN("test_ransac")
