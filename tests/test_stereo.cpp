// Sparse rectified stereo matching.
//
// Core-only, with one OpenCV-gated cross-check. The correctness story runs on
// SYNTHETIC disparity, where a constructed pair is exact ground truth for the
// whole path: constant shift, a TWO-BAND scene with a real occlusion (the case
// that catches a sign or off-by-one a global shift cancels), and a genuinely
// FRACTIONAL disparity that the parabola must recover off the integer grid.
// Where OpenCV is present, cv::StereoBM closes the reference triangle: both
// implementations against the truth and therefore against each other -- the
// same left-reference convention and the same d = xL - xR sign, which is
// exactly what that test would catch being wrong. The real-pair arm belongs to
// the stereo pipeline example and waits on a rectified pair sequence.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/shift.hpp"
#include "bincv/ops/stereo.hpp"
#include "bincv/ops/pack.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace {
using namespace bincv;

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;

std::vector<uint8_t> smoothImage(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = seed;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    std::vector<uint8_t> tmp(w * h);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 1; x + 1 < w; ++x)
                tmp[y * w + x] = static_cast<uint8_t>(
                    (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
        for (size_t y = 1; y + 1 < h; ++y)
            for (size_t x = 0; x < w; ++x)
                img[y * w + x] = static_cast<uint8_t>(
                    (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) /
                    4u);
    }
    return img;
}
} // namespace

BINCV_TEST(Stereo, RefinementFindsAnExactSyntheticDisparity) {
    // R(x) = L(x + d): every pixel's true disparity is d. Refinement is handed a
    // deliberately wrong initial guess inside its radius and must land on d --
    // exactly with the parabola off, within half a pixel with it on (at the exact
    // optimum the fit's vertex is pulled by whichever neighbour is cheaper, and
    // that is the fit working, not failing).
    constexpr size_t kW = 320, kH = 200;
    constexpr int kDisp = 13;
    BinMat<uint32_t> left(kW, kH);
    uint64_t st = 42;
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            left.set(static_cast<int>(y), static_cast<int>(x),
                     ((st >> 40) & 7u) < 3u ? 1u : 0u);
        }
    BinMat<uint32_t> right(kW, kH);
    shiftLeft(left.constView(), right.view(), kDisp);

    std::vector<float> kp;
    for (int y = 30; y < 170; y += 20)
        for (int x = 60; x < 240; x += 20) {
            kp.push_back(static_cast<float>(x));
            kp.push_back(static_cast<float>(y));
        }
    const size_t n = kp.size() / 2;

    for (int subPixel = 0; subPixel <= 1; ++subPixel) {
        std::vector<StereoMatch> m(n);
        for (auto& e : m) {
            e.valid = 1;
            e.disparity = kDisp + 2.0f;   // wrong on purpose, inside refineRadius
        }
        StereoMatchParams p;
        p.maxDisparity = 40;
        p.subPixel = subPixel != 0;
        stereoRefineDisparity<uint32_t>(left.constView(), right.constView(), kp.data(), n,
                                        m.data(), p);
        size_t good = 0;
        for (size_t i = 0; i < n; ++i) {
            BINCV_CHECK(m[i].valid == 1);
            if (p.subPixel) {
                if (std::abs(m[i].disparity - static_cast<float>(kDisp)) < 0.5f) ++good;
            } else {
                if (m[i].disparity == static_cast<float>(kDisp)) ++good;
            }
        }
        std::printf(" subPixel=%d: %zu of %zu at the true disparity\n", subPixel, good, n);
        BINCV_CHECK(good == n);
    }
}

BINCV_TEST(Stereo, DescriptorStageGatesRowAndRange) {
    // One left keypoint, three right candidates: the twin (right row, right
    // disparity), one 3 rows off, one at an impossible disparity. Only the twin
    // may survive, whatever its descriptor distance is.
    constexpr size_t kW = 200, kH = 100;
    const std::vector<uint8_t> img = smoothImage(kW, kH, 7);

    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float leftKp[2] = {150.0f, 50.0f};
    const float rightKp[6] = {130.0f, 50.0f,     // disparity 20, same row: the twin
                              130.0f, 53.0f,     // 3 rows off
                              30.0f, 50.0f};     // disparity 120, out of range
    std::vector<uint32_t> dl(kWords), dr(3 * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, leftKp, 1, pat, dl.data());
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, rightKp, 3, pat,
                                           dr.data());
    // Make every candidate's descriptor IDENTICAL to the query's, so only the
    // geometric gates can decide -- which is exactly what is under test.
    for (size_t j = 0; j < 3; ++j)
        for (size_t w = 0; w < kWords; ++w) dr[j * kWords + w] = dl[w];

    StereoMatchParams p;
    p.maxDisparity = 64;
    p.rowTolerance = 2;
    StereoMatch m;
    stereoDescriptorMatch<uint32_t>(leftKp, 1, dl.data(), rightKp, 3, dr.data(), kWords,
                                    &m, p);
    BINCV_CHECK(m.valid == 1);
    BINCV_CHECK(m.rightIndex == 0);
    BINCV_CHECK(m.distance == 0);
    BINCV_CHECK(m.disparity == 20.0f);

    // Widen the tolerance and the 3-rows-off candidate becomes admissible; the
    // twin still wins on distance only if it differs -- here they tie, and the
    // FIRST candidate in scan order is kept, which is the twin. Then remove the
    // twin and the row-tolerant candidate is the answer.
    p.rowTolerance = 3;
    stereoDescriptorMatch<uint32_t>(leftKp, 1, dl.data(), rightKp + 2, 2, dr.data() + kWords,
                                    kWords, &m, p);
    BINCV_CHECK(m.valid == 1);
    BINCV_CHECK(m.rightIndex == 0);   // the 3-rows-off one, now index 0 of the pair
    p.rowTolerance = 2;
    stereoDescriptorMatch<uint32_t>(leftKp, 1, dl.data(), rightKp + 2, 2, dr.data() + kWords,
                                    kWords, &m, p);
    BINCV_CHECK(m.valid == 0);
}

BINCV_TEST(Stereo, EndToEndOnASyntheticPair) {
    // The whole path on a shifted wide pair: describe both sides, match, refine.
    // Every keypoint must land on its twin and the refined disparity must be the
    // shift, within the sub-pixel fit's half-pixel.
    constexpr size_t kW = 320, kH = 200;
    constexpr int kDisp = 17;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 55);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kDisp];

    BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
    packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

    std::vector<float> kpL, kpR;
    for (int y = 40; y < 160; y += 15)
        for (int x = 80; x < 260; x += 15) {
            kpL.push_back(static_cast<float>(x));
            kpL.push_back(static_cast<float>(y));
            kpR.push_back(static_cast<float>(x - kDisp));
            kpR.push_back(static_cast<float>(y));
        }
    const size_t n = kpL.size() / 2;
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> dl(n * kWords), dr(n * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(lw.data(), kW, kH, kW, kpL.data(), n, pat,
                                           dl.data());
    computeBrief<kBits, uint8_t, uint32_t>(rw.data(), kW, kH, kW, kpR.data(), n, pat,
                                           dr.data());

    StereoMatchParams p;
    p.maxDisparity = 40;
    std::vector<StereoMatch> m(n);
    stereoMatchRectified<uint32_t>(lb.constView(), rb.constView(), kpL.data(), n, dl.data(),
                                   kpR.data(), n, dr.data(), kWords, m.data(), p);
    size_t valid = 0, twin = 0, close = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!m[i].valid) continue;
        ++valid;
        if (m[i].rightIndex == i) ++twin;
        if (std::abs(m[i].disparity - static_cast<float>(kDisp)) < 0.5f) ++close;
    }
    std::printf(" %zu keypoints: %zu valid, %zu on their twin, %zu within 0.5 px of %d\n",
                n, valid, twin, close, kDisp);
    BINCV_CHECK(valid == n);
    BINCV_CHECK(twin == valid);
    BINCV_CHECK(close == valid);
}

BINCV_TEST(Stereo, AWindowOffTheFrameReportsInvalidNotClamped) {
    constexpr size_t kW = 64, kH = 64;
    BinMat<uint32_t> l(kW, kH), r(kW, kH);
    const float kp[2] = {-30.0f, -30.0f};
    StereoMatch m;
    m.valid = 1;
    m.disparity = 5.0f;
    StereoMatchParams p;
    p.maxDisparity = 16;
    stereoRefineDisparity<uint32_t>(l.constView(), r.constView(), kp, 1, &m, p);
    BINCV_CHECK(m.valid == 0);
}

BINCV_TEST(Stereo, PiecewiseDisparityLandsPerBand) {
    // A constant shift cannot catch a sign error or an off-by-one that happens to
    // cancel; a scene with TWO disparities can. The right image is built the way
    // a real rig sees one: each left column band lands at its own disparity, and
    // the near band overwrites the far one in the contested zone -- a real
    // occlusion. Keypoints keep clear of the band boundary and the occlusion, and
    // every one must land on ITS band's disparity, not a blend.
    constexpr size_t kW = 400, kH = 200;
    constexpr int kDNear = 24, kDFar = 10, kSplit = 200;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 91);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y) {
        for (int xL = 0; xL < static_cast<int>(kW); ++xL) {
            const int d = xL < kSplit ? kDFar : kDNear;
            const int xR = xL - d;
            if (xR >= 0) rw[y * kW + static_cast<size_t>(xR)] = lw[y * kW + static_cast<size_t>(xL)];
        }
    }
    BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
    packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

    std::vector<float> kpL, kpR;
    std::vector<int> truth;
    for (int y = 30; y < 170; y += 15) {
        for (int x = 60; x <= 340; x += 20) {
            if (x > 150 && x < 260) continue;   // boundary + occlusion margin
            const int d = x < kSplit ? kDFar : kDNear;
            kpL.push_back(static_cast<float>(x));
            kpL.push_back(static_cast<float>(y));
            kpR.push_back(static_cast<float>(x - d));
            kpR.push_back(static_cast<float>(y));
            truth.push_back(d);
        }
    }
    const size_t n = truth.size();
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> dl(n * kWords), dr(n * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(lw.data(), kW, kH, kW, kpL.data(), n, pat, dl.data());
    computeBrief<kBits, uint8_t, uint32_t>(rw.data(), kW, kH, kW, kpR.data(), n, pat, dr.data());

    StereoMatchParams p;
    p.maxDisparity = 40;
    std::vector<StereoMatch> m(n);
    stereoMatchRectified<uint32_t>(lb.constView(), rb.constView(), kpL.data(), n, dl.data(),
                                   kpR.data(), n, dr.data(), kWords, m.data(), p);
    size_t good = 0;
    for (size_t i = 0; i < n; ++i) {
        if (m[i].valid &&
            std::abs(m[i].disparity - static_cast<float>(truth[i])) < 0.5f) ++good;
    }
    std::printf(" two-band scene: %zu of %zu keypoints on their own band's disparity\n",
                good, n);
    BINCV_CHECK_EQ(good, n);
}

BINCV_TEST(Stereo, TheParabolaRecoversAFractionalDisparity) {
    // Depth is 1/disparity, so the sub-pixel fit is the whole reason the refine
    // stage carries any float at all. Ground truth at 12.5 px: the right image is
    // the left sampled at x + 12.5 by exact 2-tap averaging, THEN binarized -- so
    // the packed pair genuinely carries a half-pixel offset and an integer-only
    // matcher is wrong by 0.5 on every keypoint. The fit must beat that clearly,
    // not marginally.
    constexpr size_t kW = 320, kH = 160;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 123);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + 13 < kW; ++x)
            rw[y * kW + x] = static_cast<uint8_t>(
                (static_cast<unsigned>(lw[y * kW + x + 12]) +
                 static_cast<unsigned>(lw[y * kW + x + 13]) + 1u) / 2u);
    BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
    packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

    std::vector<float> kp;
    for (int y = 30; y < 130; y += 12)
        for (int x = 60; x < 280; x += 16) {
            kp.push_back(static_cast<float>(x));
            kp.push_back(static_cast<float>(y));
        }
    const size_t n = kp.size() / 2;
    std::vector<StereoMatch> m(n);
    for (auto& e : m) {
        e.valid = 1;
        e.disparity = 12.0f;
    }
    StereoMatchParams p;
    p.maxDisparity = 40;
    stereoRefineDisparity<uint32_t>(lb.constView(), rb.constView(), kp.data(), n, m.data(), p);
    double sum = 0.0;
    size_t inside = 0;
    for (size_t i = 0; i < n; ++i) {
        BINCV_CHECK(m[i].valid == 1);
        sum += static_cast<double>(m[i].disparity);
        if (m[i].disparity > 12.0f && m[i].disparity < 13.0f) ++inside;
    }
    const double mean = sum / static_cast<double>(n);
    std::printf(" fractional truth 12.5: mean refined %.3f, %zu of %zu strictly inside"
                " (12, 13)\n", mean, inside, n);
    // The mean must sit near the true half-pixel -- an integer-only answer means
    // 12.0 or 13.0 -- and most individual fits must leave the integer grid.
    BINCV_CHECK(std::abs(mean - 12.5) < 0.25);
    BINCV_CHECK(inside * 3 >= n * 2);
}

#if BINCV_STEREO_WINDOW_COUNT
BINCV_TEST(Stereo, StagedArmIsBitExact) {
    // The staged refinement against the per-row path, every field of every match, over
    // window shapes up to the strip's 32 bits and the arm's 31 rows, radii either side
    // of what fits a strip, and keypoints scattered to the borders so that both the
    // windows it stages and the ones it must decline are exercised. The arm compiles
    // only where a population count is software; this suite compiles it everywhere.
    BINCV_CHECK(impl::stereoWindowCountEnabled() == (BINCV_STEREO_WINDOW_COUNT_DEFAULT != 0));
    const bool saved = impl::stereoWindowCountEnabled();

    constexpr size_t kW = 203, kH = 97;   // neither a whole number of words
    BinMat<uint32_t> left(kW, kH), right(kW, kH);
    uint64_t st = 0x57E7E0ull;
    const auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return st >> 33;
    };
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            left.set(static_cast<int>(y), static_cast<int>(x), next() % 5u < 2u ? 1u : 0u);
            right.set(static_cast<int>(y), static_cast<int>(x), next() % 7u < 3u ? 1u : 0u);
        }

    struct Shape {
        int winW, winH, radius, minD, maxD;
    };
    const Shape shapes[] = {{3, 3, 1, 0, 64},   {11, 11, 4, 0, 64},  {11, 11, 9, 0, 64},
                            {11, 11, 10, 0, 64}, {15, 9, 4, 5, 30},   {25, 31, 2, 0, 64},
                            {25, 32, 2, 0, 64},  {30, 5, 1, 0, 9}, {31, 3, 1, 0, 64},
                            {7, 13, 3, 20, 21}};
    size_t staged = 0, declined = 0, compared = 0;
    for (const Shape& sh : shapes) {
        constexpr size_t kPoints = 400;
        std::vector<float> kp(2 * kPoints);
        std::vector<StereoMatch> in(kPoints);
        for (size_t i = 0; i < kPoints; ++i) {
            kp[2 * i] = static_cast<float>(next() % (kW * 8)) / 8.0f;
            kp[2 * i + 1] = static_cast<float>(next() % (kH * 8)) / 8.0f;
            in[i].valid = (next() % 10u) != 0u ? 1 : 0;
            in[i].disparity = static_cast<float>(static_cast<int>(next() % 80u) - 8) / 2.0f;
        }
        StereoMatchParams p;
        p.winWidth = sh.winW;
        p.winHeight = sh.winH;
        p.refineRadius = sh.radius;
        p.minDisparity = sh.minD;
        p.maxDisparity = sh.maxD;
        for (int subPixel = 0; subPixel <= 1; ++subPixel) {
            p.subPixel = subPixel != 0;
            std::vector<StereoMatch> a = in, b = in;
            impl::stereoWindowCountEnabled() = false;
            stereoRefineDisparity<uint32_t>(left.constView(), right.constView(), kp.data(),
                                            kPoints, a.data(), p);
            impl::stereoWindowCountEnabled() = true;
            stereoRefineDisparity<uint32_t>(left.constView(), right.constView(), kp.data(),
                                            kPoints, b.data(), p);
            for (size_t i = 0; i < kPoints; ++i) {
                uint32_t da, db;
                std::memcpy(&da, &a[i].disparity, sizeof da);
                std::memcpy(&db, &b[i].disparity, sizeof db);
                BINCV_CHECK(da == db);
                BINCV_CHECK(a[i].valid == b[i].valid);
                ++compared;
            }
        }
        // Which windows the arm took: the kernel's own anchor and range arithmetic.
        for (size_t i = 0; i < kPoints; ++i) {
            if (!in[i].valid) continue;
            const long long ax =
                impl::floorToLL(kp[2 * i] - static_cast<float>(sh.winW - 1) * 0.5f);
            const long long ay =
                impl::floorToLL(kp[2 * i + 1] - static_cast<float>(sh.winH - 1) * 0.5f);
            const long long d0 = impl::floorToLL(in[i].disparity + 0.5f);
            const long long lo = std::max<long long>(d0 - sh.radius, sh.minD);
            const long long hi = std::min<long long>(d0 + sh.radius, sh.maxD);
            if (lo > hi) continue;
            const bool ok = impl::stereoStagedPlan<uint32_t>(kW, kH, ax, ay, sh.winW, sh.winH,
                                                             lo, hi).ok;
            ok ? ++staged : ++declined;
        }
    }
    impl::stereoWindowCountEnabled() = saved;
    std::printf(" %zu matches compared; %zu windows staged, %zu declined\n", compared, staged,
                declined);
    BINCV_CHECK(staged > 1000);
    BINCV_CHECK(declined > 1000);
}
#endif

#if BINCV_STEREO_INTEGER_ROW_GATE
BINCV_TEST(Stereo, IntegerRowGateIsBitExact) {
    // The row test on bit patterns against the published float test. First the interval
    // itself: for each left y, every probe y -- the interval's own ends and one float
    // either side of them, and values across the whole range -- is inside the keys
    // exactly when the published comparison passes it. Then whole calls, every field of
    // every match, gate off against gate on, with NaN, infinite, signed-zero and huge
    // coordinates in the mix. Compiled only on M-profile; this suite compiles it
    // everywhere.
    BINCV_CHECK(impl::stereoIntegerRowGateEnabled() ==
                (BINCV_STEREO_INTEGER_ROW_GATE_DEFAULT != 0));
    const bool saved = impl::stereoIntegerRowGateEnabled();
    uint64_t st = 0x80A7E5ull;
    const auto next = [&st]() {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        return st >> 33;
    };
    const float inf = std::numeric_limits<float>::infinity();
    const auto drawY = [&]() -> float {
        switch (next() % 8u) {
        case 0: return static_cast<float>(next() % 480u);
        case 1: return static_cast<float>(next() % 3840u) / 8.0f;
        case 2: return static_cast<float>(next() % 100000u) / 7.0f - 5000.0f;
        case 3: return (next() & 1u) ? 0.0f : -0.0f;
        case 4: return static_cast<float>(next() % 1000u) * 1e5f;
        case 5: return static_cast<float>(next()) * 1e-12f;
        default: return static_cast<float>(next() % 480u) + 0.5f;
        }
    };

    size_t probes = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const float yL = drawY();
        const float tol = static_cast<float>(next() % 8u);
        int32_t lo = 0, hi = 0;
        impl::stereoRowKeys(yL, tol, lo, hi);
        const float ends[] = {impl::stereoKeyFloat(lo), impl::stereoKeyFloat(hi)};
        std::vector<float> ys = {yL, inf, -inf, 0.0f, -0.0f, yL + tol, yL - tol};
        for (const float e : ends) {
            ys.push_back(e);
            ys.push_back(std::nextafter(e, inf));
            ys.push_back(std::nextafter(e, -inf));
        }
        for (int k = 0; k < 20; ++k) ys.push_back(drawY());
        for (const float y : ys) {
            const float dy = y - yL;
            const bool want = !(dy > tol || dy < -tol);
            const int32_t key = impl::stereoOrderedKey(y);
            BINCV_CHECK((key >= lo && key <= hi) == want);
            ++probes;
        }
    }

    // A draw below `n`, as an index -- `next()` is 64 bits and `size_t` may be 32.
    const auto pick = [&next](size_t n) { return static_cast<size_t>(next() % n); };
    size_t calls = 0, gated = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const size_t nl = 1 + pick(60), nr = pick(80);
        std::vector<float> l(2 * nl), r(2 * nr);
        for (size_t i = 0; i < nl; ++i) {
            l[2 * i] = static_cast<float>(next() % 400u) + 64.0f;
            l[2 * i + 1] = drawY();
        }
        for (size_t j = 0; j < nr; ++j) {
            // Mostly near some left keypoint's row, so the gate passes often enough to
            // matter; the rest anywhere.
            const size_t i = pick(nl);
            r[2 * j] = l[2 * i] - static_cast<float>(next() % 70u);
            const int row = static_cast<int>(next() % 9u) - 4;
            r[2 * j + 1] =
                (next() % 4u) != 0u ? l[2 * i + 1] + static_cast<float>(row) : drawY();
        }
        if (trial % 10 == 3 && nr > 0) r[2 * pick(nr) + 1] = std::nanf("");
        if (trial % 10 == 5) l[2 * pick(nl) + 1] = std::nanf("");
        if (trial % 10 == 7) l[2 * pick(nl) + 1] = (next() & 1u) ? inf : -inf;
        if (trial % 10 == 8 && nr > 0) r[2 * pick(nr) + 1] = (next() & 1u) ? inf : -inf;
        std::vector<uint32_t> dl(nl * kWords), dr(nr * kWords);
        for (auto& w : dl) w = static_cast<uint32_t>(next()) & 0x0F0F0F0Fu;
        for (auto& w : dr) w = static_cast<uint32_t>(next()) & 0x0F0F0F0Fu;
        StereoMatchParams p;
        p.rowTolerance = static_cast<int>(next() % 4u);
        p.minDisparity = static_cast<int>(next() % 5u);
        p.maxDisparity = p.minDisparity + 1 + static_cast<int>(next() % 64u);
        p.maxHamming = 256;
        std::vector<StereoMatch> a(nl), b(nl);
        impl::stereoIntegerRowGateEnabled() = false;
        stereoDescriptorMatch<uint32_t>(l.data(), nl, dl.data(), r.data(), nr, dr.data(),
                                        kWords, a.data(), p);
        impl::stereoIntegerRowGateEnabled() = true;
        stereoDescriptorMatch<uint32_t>(l.data(), nl, dl.data(), r.data(), nr, dr.data(),
                                        kWords, b.data(), p);
        for (size_t i = 0; i < nl; ++i) {
            BINCV_CHECK(std::memcmp(&a[i].disparity, &b[i].disparity, sizeof(float)) == 0);
            BINCV_CHECK(a[i].distance == b[i].distance);
            BINCV_CHECK(a[i].rightIndex == b[i].rightIndex);
            BINCV_CHECK(a[i].valid == b[i].valid);
            gated += a[i].valid;
        }
        ++calls;
    }
    impl::stereoIntegerRowGateEnabled() = saved;
    std::printf(" %zu row probes; %zu calls, %zu matches compared valid\n", probes, calls,
                gated);
    BINCV_CHECK(gated > 1000);
}
#endif

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(Stereo, AgreesWithCvStereoBMOnCleanGroundTruth) {
    // The reference-implementation cross-check. cv::StereoBM prices a DIFFERENT
    // operation (dense, SAD on prefiltered bytes) so agreement is a sanity
    // triangle, not an equality: on a clean constant-disparity pair, binCV's
    // sparse matcher and StereoBM must both land on the truth, and therefore
    // near each other -- same left-reference convention, same disparity sign
    // (d = xL - xR), which is exactly what this test would catch being wrong.
    constexpr size_t kW = 400, kH = 200;
    constexpr int kDisp = 16;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 777);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kDisp];

    cv::Mat lcv(static_cast<int>(kH), static_cast<int>(kW), CV_8U), rcv = lcv.clone();
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            lcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = lw[y * kW + x];
            rcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = rw[y * kW + x];
        }
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(64, 21);
    cv::Mat disp16;
    bm->compute(lcv, rcv, disp16);

    BinMat<uint32_t> lb(kW, kH), rb(kW, kH);
    packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
    packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

    // Keypoints inside StereoBM's valid region: it blanks x < numDisparities and
    // a blockSize/2 rim, so start well right of 64 + 10.
    std::vector<float> kpL, kpR;
    for (int y = 30; y < 170; y += 15)
        for (int x = 100; x < 340; x += 20) {
            kpL.push_back(static_cast<float>(x));
            kpL.push_back(static_cast<float>(y));
            kpR.push_back(static_cast<float>(x - kDisp));
            kpR.push_back(static_cast<float>(y));
        }
    const size_t n = kpL.size() / 2;
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> dl(n * kWords), dr(n * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(lw.data(), kW, kH, kW, kpL.data(), n, pat, dl.data());
    computeBrief<kBits, uint8_t, uint32_t>(rw.data(), kW, kH, kW, kpR.data(), n, pat, dr.data());
    StereoMatchParams p;
    p.maxDisparity = 40;
    std::vector<StereoMatch> m(n);
    stereoMatchRectified<uint32_t>(lb.constView(), rb.constView(), kpL.data(), n, dl.data(),
                                   kpR.data(), n, dr.data(), kWords, m.data(), p);

    size_t oursGood = 0, bmValid = 0, bmGood = 0, agree = 0;
    for (size_t i = 0; i < n; ++i) {
        BINCV_CHECK(m[i].valid == 1);
        const float ours = m[i].disparity;
        if (std::abs(ours - static_cast<float>(kDisp)) <= 0.5f) ++oursGood;
        const short raw = disp16.at<short>(static_cast<int>(kpL[2 * i + 1]),
                                           static_cast<int>(kpL[2 * i]));
        if (raw < 0) continue;   // StereoBM's own invalid marker
        ++bmValid;
        const float bmD = static_cast<float>(raw) / 16.0f;
        if (std::abs(bmD - static_cast<float>(kDisp)) <= 1.0f) ++bmGood;
        if (std::abs(bmD - ours) <= 1.5f) ++agree;
    }
    std::printf(" truth %d: ours within 0.5 px on %zu/%zu; StereoBM valid at %zu, within"
                " 1 px on %zu; the two within 1.5 px of each other on %zu\n",
                kDisp, oursGood, n, bmValid, bmGood, agree);
    BINCV_CHECK_EQ(oursGood, n);
    BINCV_CHECK(bmValid > n / 2);        // BM validity is its own business
    BINCV_CHECK_EQ(bmGood, bmValid);     // where BM speaks, it agrees with truth
    BINCV_CHECK_EQ(agree, bmValid);      // and therefore with us
}
#endif // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_stereo")
