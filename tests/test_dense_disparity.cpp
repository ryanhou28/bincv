// Dense disparity from census matching.
//
// Core-only, with one OpenCV-gated cross-check. Same doctrine as the sparse
// matcher's suite: synthetic pairs are EXACT ground truth, so the assertions are
// equalities in the supported region, not similarity scores -- a constant shift
// must come back as that constant, a two-band scene must respect its own depth
// edge, and everything the kernel cannot evaluate must say so with the invalid
// marker rather than a plausible number.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/denseDisparity.hpp"
#include "bincv/ops/pack.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#endif

namespace {
using namespace bincv;

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

template <typename W>
void runConstantShift(uint64_t seed) {
    constexpr size_t kW = 200, kH = 80;
    constexpr int kDisp = 16;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, seed);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kDisp];

    DenseDisparityParams p;
    p.maxDisparity = 32;
    std::vector<W> sw(denseDisparityScratchWords<24, W>(kW, p));
    std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(kW * kH, 0);
    denseDisparity<24, uint8_t, W>(lw.data(), rw.data(), kW, kH, kW, kW, kCensus5x5, p,
                                   sw.data(), sw.size(), sr.data(), sr.size(),
                                   disp.data(), kW);

    const size_t hw = static_cast<size_t>(p.winWidth / 2);
    const size_t hh = static_cast<size_t>(p.winHeight / 2);
    size_t exact = 0, total = 0, rimBad = 0;
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            const bool rim = y < hh || y + hh >= kH || x < hw || x + hw >= kW;
            if (rim) {
                if (disp[y * kW + x] != kDenseDisparityInvalid) ++rimBad;
                continue;
            }
            // The fully supported region: the window's right-image support
            // exists for the TRUE disparity and the whole search range.
            if (x < static_cast<size_t>(p.maxDisparity) + hw) continue;
            if (x + kDisp + hw >= kW) continue;   // right image is dark past the shift
            ++total;
            if (disp[y * kW + x] == kDisp) ++exact;
        }
    std::printf(" W=%zu-bit constant shift: %zu of %zu supported pixels exact,"
                " %zu rim pixels not invalid\n",
                sizeof(W) * 8, exact, total, rimBad);
    BINCV_CHECK_EQ(exact, total);
    BINCV_CHECK_EQ(rimBad, size_t{0});
}
} // namespace

BINCV_TEST(DenseDisparity, ConstantShiftIsExactAtEveryWordType) {
    // A perfect pair has a zero-cost window at the true disparity and nowhere
    // else on textured content, so the supported region must be EXACT -- and the
    // word type must not matter, which is where the shifted-word gather would
    // show a seam.
    runConstantShift<uint8_t>(11);
    runConstantShift<uint16_t>(22);
    runConstantShift<uint32_t>(33);
    runConstantShift<uint64_t>(44);
}

BINCV_TEST(DenseDisparity, ATwoBandSceneRespectsItsOwnDepthEdge) {
    constexpr size_t kW = 260, kH = 80;
    constexpr int kDFar = 10, kDNear = 24, kSplit = 130;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 91);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (int xL = 0; xL < static_cast<int>(kW); ++xL) {
            const int d = xL < kSplit ? kDFar : kDNear;
            const int xR = xL - d;
            if (xR >= 0)
                rw[y * kW + static_cast<size_t>(xR)] = lw[y * kW + static_cast<size_t>(xL)];
        }

    DenseDisparityParams p;
    p.maxDisparity = 32;
    std::vector<uint32_t> sw(denseDisparityScratchWords<24, uint32_t>(kW, p));
    std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(kW * kH, 0);
    denseDisparity<24, uint8_t, uint32_t>(lw.data(), rw.data(), kW, kH, kW, kW,
                                          kCensus5x5, p, sw.data(), sw.size(), sr.data(),
                                          sr.size(), disp.data(), kW);

    // Away from the depth edge and the OCCLUSION, each side answers with ITS
    // disparity. The occlusion is one-sided and wider than the edge: the near
    // band overwrites the far band's right-image support for the last
    // (kDNear - kDFar) far columns before the split, so the far bound backs off
    // by the occlusion width plus the window's and the census's reach -- pixels
    // in that strip have no true match to find, which is physics, not a defect.
    const int farMargin = (kDNear - kDFar) + p.winWidth + 2;
    const int margin = p.winWidth / 2 + 4;
    size_t farOk = 0, farN = 0, nearOk = 0, nearN = 0;
    for (size_t y = 8; y + 8 < kH; ++y) {
        for (int x = p.maxDisparity + p.winWidth; x < kSplit - farMargin; ++x) {
            ++farN;
            if (disp[y * kW + static_cast<size_t>(x)] == kDFar) ++farOk;
        }
        for (int x = kSplit + margin + kDNear; x + p.winWidth < static_cast<int>(kW);
             ++x) {
            ++nearN;
            if (disp[y * kW + static_cast<size_t>(x)] == kDNear) ++nearOk;
        }
    }
    std::printf(" two-band map: far %zu/%zu, near %zu/%zu at their band's truth\n", farOk,
                farN, nearOk, nearN);
    BINCV_CHECK_EQ(farOk, farN);
    BINCV_CHECK_EQ(nearOk, nearN);
}

BINCV_TEST(DenseDisparity, DegenerateGeometryIsAllInvalidNotPlausible) {
    // An image narrower than the window, or a range no column supports, has no
    // answer -- every pixel must say so.
    constexpr size_t kW = 7, kH = 12;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 3), rw = lw;
    DenseDisparityParams p;
    p.maxDisparity = 32;
    std::vector<uint32_t> sw(denseDisparityScratchWords<8, uint32_t>(kW, p));
    std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(kW * kH, 0);
    denseDisparity<8, uint8_t, uint32_t>(lw.data(), rw.data(), kW, kH, kW, kW, kCensus3x3,
                                         p, sw.data(), sw.size(), sr.data(), sr.size(),
                                         disp.data(), kW);
    size_t bad = 0;
    for (uint8_t v : disp)
        if (v != kDenseDisparityInvalid) ++bad;
    BINCV_CHECK_EQ(bad, size_t{0});
}

BINCV_TEST(DenseDisparity, TheTwoVerticalArmsAreBitIdentical) {
    // The sliding accumulator exists for speed and pays scratch linear in D;
    // the recompute arm keeps scratch independent of D. They are ONE answer
    // with two costs, and this holds them to identical output maps -- byte for
    // byte, on a scene with a real depth edge, at two word types -- so neither
    // arm can drift into being "the fast one that answers differently".
    constexpr size_t kW = 260, kH = 60;
    constexpr int kSplit = 130;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 4242);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (int xL = 0; xL < static_cast<int>(kW); ++xL) {
            const int d = xL < kSplit ? 8 : 19;
            if (xL - d >= 0)
                rw[y * kW + static_cast<size_t>(xL - d)] =
                    lw[y * kW + static_cast<size_t>(xL)];
        }

    const auto runBoth = [&](auto wordTag) {
        using W = decltype(wordTag);
        DenseDisparityParams p;
        p.maxDisparity = 32;
        std::vector<uint8_t> slide(kW * kH, 0), recomp(kW * kH, 1);
        {
            std::vector<W> sw(denseDisparityScratchWords<24, W>(kW, p));
            std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
            denseDisparity<24, uint8_t, W>(lw.data(), rw.data(), kW, kH, kW, kW,
                                           kCensus5x5, p, sw.data(), sw.size(),
                                           sr.data(), sr.size(), slide.data(), kW);
        }
        p.recomputeVertical = true;
        {
            std::vector<W> sw(denseDisparityScratchWords<24, W>(kW, p));
            std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
            denseDisparity<24, uint8_t, W>(lw.data(), rw.data(), kW, kH, kW, kW,
                                           kCensus5x5, p, sw.data(), sw.size(),
                                           sr.data(), sr.size(), recomp.data(), kW);
        }
        size_t differ = 0;
        for (size_t i = 0; i < slide.size(); ++i)
            if (slide[i] != recomp[i]) ++differ;
        std::printf(" W=%zu-bit: %zu of %zu map bytes differ between the arms\n",
                    sizeof(W) * 8, differ, slide.size());
        BINCV_CHECK_EQ(differ, size_t{0});
    };
    runBoth(uint32_t{});
    runBoth(uint64_t{});
}

BINCV_TEST(DenseDisparity, TheBinaryNativePathIsExactOnItsOwnRepresentation) {
    // The premise-native spelling: packed frames in, no census -- the cost is
    // the tracker's own window Hamming, densely. Same equalities as the wide
    // path: constant shift exact at every word type in the supported region,
    // both vertical arms bit-identical, the two-band depth edge respected.
    constexpr size_t kW = 200, kH = 80;
    constexpr int kDisp = 16;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 606);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kDisp];

    const auto runOne = [&](auto wordTag) {
        using W = decltype(wordTag);
        BinMat<W> lb(kW, kH), rb(kW, kH);
        packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
        packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

        DenseDisparityParams p;
        p.maxDisparity = 32;
        std::vector<W> sw(denseDisparityBinaryScratchWords<W>(kW, p));
        std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
        std::vector<uint8_t> slide(kW * kH, 0);
        denseDisparityBinary<W>(lb.constView(), rb.constView(), p, sw.data(), sw.size(),
                                sr.data(), sr.size(), slide.data(), kW);

        const size_t hw = static_cast<size_t>(p.winWidth / 2);
        const size_t hh = static_cast<size_t>(p.winHeight / 2);
        size_t exact = 0, total = 0;
        for (size_t y = hh; y + hh < kH; ++y)
            for (size_t x = static_cast<size_t>(p.maxDisparity) + hw;
                 x + kDisp + hw < kW; ++x) {
                ++total;
                if (slide[y * kW + x] == kDisp) ++exact;
            }
        std::printf(" binary W=%zu-bit: %zu of %zu supported pixels exact\n",
                    sizeof(W) * 8, exact, total);
        BINCV_CHECK_EQ(exact, total);

        p.recomputeVertical = true;
        std::vector<W> sw2(denseDisparityBinaryScratchWords<W>(kW, p));
        std::vector<uint8_t> recomp(kW * kH, 1);
        denseDisparityBinary<W>(lb.constView(), rb.constView(), p, sw2.data(), sw2.size(),
                                sr.data(), sr.size(), recomp.data(), kW);
        size_t differ = 0;
        for (size_t i = 0; i < slide.size(); ++i)
            if (slide[i] != recomp[i]) ++differ;
        BINCV_CHECK_EQ(differ, size_t{0});

#if defined(BINCV_DENSE_SIMD)
        // The vector arm against the portable arm in ONE binary, via the
        // runtime switch -- the same contract the packer's arm carries.
        if (sizeof(W) == 8 && impl::hasDenseSimd()) {
            p.recomputeVertical = false;
            std::vector<uint8_t> scalarMap(kW * kH, 2);
            impl::denseSimdEnabled() = false;
            denseDisparityBinary<W>(lb.constView(), rb.constView(), p, sw.data(),
                                    sw.size(), sr.data(), sr.size(), scalarMap.data(),
                                    kW);
            impl::denseSimdEnabled() = true;
            size_t armDiffer = 0;
            for (size_t i = 0; i < slide.size(); ++i)
                if (slide[i] != scalarMap[i]) ++armDiffer;
            BINCV_CHECK_EQ(armDiffer, size_t{0});
        }
#endif
    };
    runOne(uint8_t{});
    runOne(uint16_t{});
    runOne(uint32_t{});
    runOne(uint64_t{});
}

BINCV_TEST(DenseDisparity, TheBinaryPathRespectsADepthEdge) {
    constexpr size_t kW = 260, kH = 80;
    constexpr int kDFar = 10, kDNear = 24, kSplit = 130;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 91);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (int xL = 0; xL < static_cast<int>(kW); ++xL) {
            const int d = xL < kSplit ? kDFar : kDNear;
            if (xL - d >= 0)
                rw[y * kW + static_cast<size_t>(xL - d)] =
                    lw[y * kW + static_cast<size_t>(xL)];
        }
    BinMat<uint64_t> lb(kW, kH), rb(kW, kH);
    packBits<PackRule::GreaterThan>(lw.data(), kW, kH, kW, lb.view(), uint8_t{127});
    packBits<PackRule::GreaterThan>(rw.data(), kW, kH, kW, rb.view(), uint8_t{127});

    DenseDisparityParams p;
    p.maxDisparity = 32;
    std::vector<uint64_t> sw(denseDisparityBinaryScratchWords<uint64_t>(kW, p));
    std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
    std::vector<uint8_t> disp(kW * kH, 0);
    denseDisparityBinary<uint64_t>(lb.constView(), rb.constView(), p, sw.data(),
                                   sw.size(), sr.data(), sr.size(), disp.data(), kW);
    const int farMargin = (kDNear - kDFar) + p.winWidth + 2;
    const int margin = p.winWidth / 2 + 4;
    size_t farOk = 0, farN = 0, nearOk = 0, nearN = 0;
    for (size_t y = 8; y + 8 < kH; ++y) {
        for (int x = p.maxDisparity + p.winWidth; x < kSplit - farMargin; ++x) {
            ++farN;
            if (disp[y * kW + static_cast<size_t>(x)] == kDFar) ++farOk;
        }
        for (int x = kSplit + margin + kDNear; x + p.winWidth < static_cast<int>(kW);
             ++x) {
            ++nearN;
            if (disp[y * kW + static_cast<size_t>(x)] == kDNear) ++nearOk;
        }
    }
    std::printf(" binary two-band: far %zu/%zu, near %zu/%zu\n", farOk, farN, nearOk,
                nearN);
    BINCV_CHECK_EQ(farOk, farN);
    BINCV_CHECK_EQ(nearOk, nearN);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(DenseDisparity, TheStereoBMTriangleClosesOnCleanGroundTruth) {
    // The reference-implementation cross-check, exactly as the sparse matcher
    // has it: cv::StereoBM prices the same ROLE with different numerics, so on
    // clean constant-disparity ground truth both must land on the truth and
    // therefore near each other -- the sanity triangle, not an equality.
    constexpr size_t kW = 320, kH = 120;
    constexpr int kDisp = 16;
    const std::vector<uint8_t> lw = smoothImage(kW, kH, 777);
    std::vector<uint8_t> rw(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kDisp < kW; ++x) rw[y * kW + x] = lw[y * kW + x + kDisp];

    DenseDisparityParams p;
    p.maxDisparity = 48;
    std::vector<uint32_t> sw(denseDisparityScratchWords<24, uint32_t>(kW, p));
    std::vector<uint16_t> sr(denseDisparityScratchRows(kW));
    std::vector<uint8_t> ours(kW * kH, 0);
    denseDisparity<24, uint8_t, uint32_t>(lw.data(), rw.data(), kW, kH, kW, kW,
                                          kCensus5x5, p, sw.data(), sw.size(), sr.data(),
                                          sr.size(), ours.data(), kW);

    cv::Mat lcv(static_cast<int>(kH), static_cast<int>(kW), CV_8U);
    cv::Mat rcv = lcv.clone();
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) {
            lcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = lw[y * kW + x];
            rcv.at<uint8_t>(static_cast<int>(y), static_cast<int>(x)) = rw[y * kW + x];
        }
    cv::Mat disp16;
    cv::StereoBM::create(64, 21)->compute(lcv, rcv, disp16);

    size_t oursGood = 0, oursN = 0, bmGood = 0, bmN = 0, agree = 0, both = 0;
    for (size_t y = 12; y + 12 < kH; y += 3)
        for (size_t x = 100; x + 24 + kDisp < kW; x += 5) {
            const uint8_t o = ours[y * kW + x];
            if (o != kDenseDisparityInvalid) {
                ++oursN;
                if (o == kDisp) ++oursGood;
            }
            const short raw =
                disp16.at<short>(static_cast<int>(y), static_cast<int>(x));
            if (raw >= 0) {
                ++bmN;
                const float bm = static_cast<float>(raw) / 16.0f;
                if (bm > kDisp - 1.0f && bm < kDisp + 1.0f) ++bmGood;
                if (o != kDenseDisparityInvalid) {
                    ++both;
                    if (bm > static_cast<float>(o) - 1.5f &&
                        bm < static_cast<float>(o) + 1.5f) ++agree;
                }
            }
        }
    std::printf(" ours exact %zu/%zu; StereoBM within 1 px %zu/%zu; the two within"
                " 1.5 px on %zu/%zu shared pixels\n",
                oursGood, oursN, bmGood, bmN, agree, both);
    BINCV_CHECK_EQ(oursGood, oursN);
    BINCV_CHECK(bmN > 0 && bmGood == bmN);
    BINCV_CHECK_EQ(agree, both);
}
#endif // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_dense_disparity")
