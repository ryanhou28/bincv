// Sparse rectified stereo matching.
//
// Core-only. The correctness story runs on SYNTHETIC disparity: a shifted frame is
// exact ground truth for the whole path -- the descriptor stage's candidate gating,
// the Hamming refinement's integer optimum, and the parabola's behavior at it --
// with no dataset involved. The real-pair arm belongs to the stereo frontend
// example and waits on a rectified pair sequence.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/shift.hpp"
#include "bincv/ops/stereo.hpp"
#include "bincv/ops/pack.hpp"
#include "test_util.hpp"

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

BINCV_TEST_MAIN("test_stereo")
