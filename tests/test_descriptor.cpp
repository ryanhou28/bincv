// Binary descriptors and Hamming matching.
//
// Core-only: descriptors need no OpenCV, and the point of the family is that matching
// is popcount(a ^ b) -- the operation binCV is built out of.
//
// WHAT IS AND IS NOT CLAIMED. Descriptors from two different patterns are
// incomparable -- true of BRIEF generally, and why the pattern is an argument
// rather than a hidden constant. With the DEFAULT pattern these are not cv::ORB
// descriptors; with the vendored table in ops/orbPattern.hpp they are, byte for
// byte at angle zero over the same blurred pixels, and the OrbPattern suite at
// the bottom of this file is what holds that to "are" rather than "should be".
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/ops/descriptor.hpp"
#include "bincv/ops/orbPattern.hpp"
#include "bincv/ops/orientation.hpp"
#include "test_util.hpp"

#ifdef BINCV_WITH_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {
using namespace bincv;

constexpr size_t kBits = 256;
constexpr size_t kWords = kBits / 32;

std::vector<uint8_t> texturedImage(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = seed;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    return img;
}
} // namespace

BINCV_TEST(Descriptor, PatternIsDeterministic) {
    // Two builds, two runs, two machines must agree -- a descriptor computed today has
    // to match one computed tomorrow. A pattern that silently varied would be a
    // CORRECTNESS bug, not a quality one, because descriptors from different patterns
    // are incomparable.
    BriefPattern<kBits> a, b;
    makeBriefPattern<kBits>(a);
    makeBriefPattern<kBits>(b);
    size_t diff = 0;
    for (size_t i = 0; i < kBits; ++i)
        if (a.pair[i].ax != b.pair[i].ax || a.pair[i].ay != b.pair[i].ay ||
            a.pair[i].bx != b.pair[i].bx || a.pair[i].by != b.pair[i].by) ++diff;
    BINCV_CHECK(diff == 0);
    // A different seed must give a different pattern, or the seed does nothing.
    BriefPattern<kBits> c;
    makeBriefPattern<kBits>(c, 31, 0xABCDEF01ull);
    size_t moved = 0;
    for (size_t i = 0; i < kBits; ++i) if (a.pair[i].ax != c.pair[i].ax) ++moved;
    std::printf(" same seed: %zu pairs differ; other seed: %zu of %zu moved\n", diff,
                moved, kBits);
    BINCV_CHECK(moved > kBits / 4);
    // Every offset must sit inside the patch, or computeBrief rejects keypoints that
    // are nowhere near a border.
    int maxAbs = 0;
    for (size_t i = 0; i < kBits; ++i) {
        maxAbs = std::max(maxAbs, std::abs(static_cast<int>(a.pair[i].ax)));
        maxAbs = std::max(maxAbs, std::abs(static_cast<int>(a.pair[i].by)));
    }
    std::printf(" widest offset %d (patch half is 15)\n", maxAbs);
    BINCV_CHECK(maxAbs <= 15);
}

BINCV_TEST(Descriptor, IdenticalPatchesGiveDistanceZero) {
    constexpr size_t kW = 120, kH = 90;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 7);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float kp[4] = {50.0f, 40.0f, 50.0f, 40.0f};   // the SAME point, twice
    std::vector<uint32_t> d(2 * kWords);
    std::vector<uint8_t> keep(2);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, kp, 2, pat, d.data(),
                                           keep.data());
    BINCV_CHECK(keep[0] == 1 && keep[1] == 1);
    const unsigned dist = hammingDistance<uint32_t>(d.data(), d.data() + kWords, kWords);
    std::printf(" same patch twice: distance %u of %zu bits\n", dist, kBits);
    BINCV_CHECK(dist == 0);
}

BINCV_TEST(Descriptor, ATranslatedImageMatchesBackToItsOwnKeypoints) {
    // The claim that makes descriptors useful: shift the image, describe the shifted
    // keypoints, and each must match its own original -- not a neighbour.
    constexpr size_t kW = 160, kH = 120;
    const int shift = 5;
    const std::vector<uint8_t> a = texturedImage(kW, kH, 99);
    std::vector<uint8_t> b(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + static_cast<size_t>(shift) < kW; ++x)
            b[y * kW + x] = a[y * kW + x + static_cast<size_t>(shift)];

    std::vector<float> kpA, kpB;
    for (int y = 30; y < 90; y += 11)
        for (int x = 30; x < 110; x += 13) {
            kpA.push_back(static_cast<float>(x));
            kpA.push_back(static_cast<float>(y));
            kpB.push_back(static_cast<float>(x - shift));
            kpB.push_back(static_cast<float>(y));
        }
    const size_t n = kpA.size() / 2;
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    std::vector<uint32_t> da(n * kWords), db(n * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(a.data(), kW, kH, kW, kpA.data(), n, pat, da.data());
    computeBrief<kBits, uint8_t, uint32_t>(b.data(), kW, kH, kW, kpB.data(), n, pat, db.data());

    std::vector<DescriptorMatch> m(n);
    matchDescriptors<uint32_t>(db.data(), n, da.data(), n, kWords, m.data(), 80);
    size_t correct = 0, accepted = 0;
    for (size_t i = 0; i < n; ++i) {
        if (m[i].valid) ++accepted;
        if (m[i].valid && m[i].trainIndex == i) ++correct;
    }
    std::printf(" %zu keypoints, %zu accepted by the ratio test, %zu correct\n", n,
                accepted, correct);
    // Not 100%: a random-texture image has genuinely ambiguous patches, and the ratio
    // test is SUPPOSED to reject those. What must hold is that nearly everything it
    // ACCEPTS is right -- a matcher that accepts confidently and wrongly is worse than
    // one that abstains.
    BINCV_CHECK(accepted > n / 2);
    BINCV_CHECK(correct == accepted);
}

BINCV_TEST(Descriptor, BorderKeypointsAreRejectedNotInvented) {
    // A keypoint whose patch falls outside the image has no descriptor. Clamping would
    // produce a confident match against nothing, which is worse than no match.
    constexpr size_t kW = 60, kH = 60;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 3);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float kp[6] = {1.0f, 1.0f, 30.0f, 30.0f, 58.0f, 58.0f};
    std::vector<uint32_t> d(3 * kWords);
    std::vector<uint8_t> keep(3);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, kp, 3, pat, d.data(),
                                           keep.data());
    std::printf(" keep flags: corner=%u center=%u corner=%u\n", keep[0], keep[1], keep[2]);
    BINCV_CHECK(keep[0] == 0);
    BINCV_CHECK(keep[1] == 1);
    BINCV_CHECK(keep[2] == 0);
    // A rejected keypoint's descriptor is zeroed rather than left as whatever the
    // partial loop wrote -- otherwise a caller ignoring `keep` gets garbage that looks
    // like data.
    unsigned bits = 0;
    for (size_t w = 0; w < kWords; ++w) bits += static_cast<unsigned>(__builtin_popcount(d[w]));
    BINCV_CHECK(bits == 0);
}

BINCV_TEST(Descriptor, RatioTestNeedsTwoCandidates) {
    // With one train descriptor there is no second-best, so there is no ratio to test.
    // Accepting unconditionally would make a single-keypoint train set match anything.
    constexpr size_t kW = 80, kH = 80;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 11);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float one[2] = {40.0f, 40.0f};
    const float other[2] = {20.0f, 60.0f};
    std::vector<uint32_t> t(kWords), q(kWords);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, one, 1, pat, t.data());
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, other, 1, pat, q.data());
    DescriptorMatch m;
    matchDescriptors<uint32_t>(q.data(), 1, t.data(), 1, kWords, &m, 80);
    std::printf(" single-candidate train set: valid=%d\n", m.valid ? 1 : 0);
    BINCV_CHECK(!m.valid);
}

BINCV_TEST(Descriptor, GatedWithAnUnboundedWindowIsBruteForceExactly) {
    // The gated matcher's contract: the gate ADMITS, the ratio test decides, and
    // a window nothing escapes reproduces matchDescriptors field for field --
    // which is what makes the two arms comparable in a benchmark.
    constexpr size_t kW = 160, kH = 120, kN = 60;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 314);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    std::vector<float> xyA, xyB;
    uint64_t st = 99;
    for (size_t i = 0; i < kN; ++i) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        xyA.push_back(static_cast<float>(20 + (st >> 33) % (kW - 40)));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        xyA.push_back(static_cast<float>(20 + (st >> 33) % (kH - 40)));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        xyB.push_back(static_cast<float>(20 + (st >> 33) % (kW - 40)));
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        xyB.push_back(static_cast<float>(20 + (st >> 33) % (kH - 40)));
    }
    std::vector<uint32_t> da(kN * kWords), db(kN * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, xyA.data(), kN, pat,
                                           da.data());
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, xyB.data(), kN, pat,
                                           db.data());
    std::vector<DescriptorMatch> brute(kN), gated(kN);
    matchDescriptors<uint32_t>(da.data(), kN, db.data(), kN, kWords, brute.data(), 80);
    matchDescriptorsGated<uint32_t>(da.data(), xyA.data(), kN, db.data(), xyB.data(), kN,
                                    kWords, 1e9f, 1e9f, gated.data(), 80);
    size_t differ = 0;
    for (size_t i = 0; i < kN; ++i) {
        if (brute[i].valid != gated[i].valid || brute[i].trainIndex != gated[i].trainIndex ||
            brute[i].distance != gated[i].distance ||
            brute[i].secondDistance != gated[i].secondDistance) ++differ;
    }
    std::printf(" unbounded gate vs brute force: %zu of %zu results differ\n", differ, kN);
    BINCV_CHECK_EQ(differ, size_t{0});
}

BINCV_TEST(Descriptor, TheWindowExcludesADistantImpostor) {
    // The failure mode the gate exists to prevent measuring as a success: an
    // IDENTICAL descriptor far across the frame (repeated texture) would win a
    // brute-force match; the window keeps association local, so the near, merely
    // similar candidate must win instead.
    constexpr size_t kW = 200, kH = 100;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 21);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float q[2] = {50.0f, 50.0f};
    // Train: the impostor at the query's own patch but 120 px away by claimed
    // position; a near candidate from a neighbouring patch; and a second near
    // candidate so the ratio test has its rival.
    const float t[6] = {170.0f, 50.0f, 53.0f, 50.0f, 47.0f, 53.0f};
    std::vector<uint32_t> dq(kWords), dt(3 * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, q, 1, pat, dq.data());
    const float near2[4] = {53.0f, 50.0f, 47.0f, 53.0f};
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, near2, 2, pat,
                                           dt.data() + kWords);
    for (size_t w = 0; w < kWords; ++w) dt[w] = dq[w];   // the impostor: distance 0

    DescriptorMatch brute, gated;
    matchDescriptors<uint32_t>(dq.data(), 1, dt.data(), 3, kWords, &brute, 100);
    matchDescriptorsGated<uint32_t>(dq.data(), q, 1, dt.data(), t, 3, kWords, 20.0f, 20.0f,
                                    &gated, 100);
    std::printf(" brute picks train %zu (d=%u); gated picks train %zu (d=%u)\n",
                brute.trainIndex, brute.distance, gated.trainIndex, gated.distance);
    BINCV_CHECK_EQ(brute.trainIndex, size_t{0});   // the impostor, at distance 0
    BINCV_CHECK(gated.trainIndex != 0);            // excluded by geometry
}

BINCV_TEST(Descriptor, TheOctaveBandAndTheTwoAdmittedRuleHold) {
    constexpr size_t kW = 120, kH = 90;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 8);
    BriefPattern<kBits> pat;
    makeBriefPattern<kBits>(pat);
    const float q[2] = {60.0f, 45.0f};
    const float t[6] = {62.0f, 45.0f, 58.0f, 44.0f, 60.0f, 47.0f};
    std::vector<uint32_t> dq(kWords), dt(3 * kWords);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, q, 1, pat, dq.data());
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, t, 3, pat, dt.data());
    const int qo[1] = {0};

    // Octave band: candidates at octaves {0, 3, 1} under delta 1 admit only
    // indices 0 and 2 -- and the winner must come from the admitted set.
    const int to[3] = {0, 3, 1};
    DescriptorMatch m;
    matchDescriptorsGated<uint32_t>(dq.data(), q, 1, dt.data(), t, 3, kWords, 50.0f, 50.0f,
                                    &m, 100, qo, to, 1);
    BINCV_CHECK(m.trainIndex != 1);

    // One admitted candidate is no match, even at distance 0: with no rival
    // inside the gate there is no ratio to test, exactly as matchDescriptors
    // treats a one-element train set.
    const int toFar[3] = {0, 3, 3};
    matchDescriptorsGated<uint32_t>(dq.data(), q, 1, dt.data(), t, 3, kWords, 50.0f, 50.0f,
                                    &m, 100, qo, toFar, 1);
    std::printf(" one admitted candidate: valid=%d\n", m.valid ? 1 : 0);
    BINCV_CHECK(!m.valid);
}

namespace {

/// A SMOOTH texture, not raw noise: the rotation-invariance claim is about content
/// where a one-pixel sampling error keeps most of the signal, which raw noise (zero
/// spatial correlation) deliberately does not. Two box-blur passes per axis.
std::vector<uint8_t> smoothImage(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img = texturedImage(w, h, seed);
    std::vector<uint8_t> tmp(w * h);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 2; x + 2 < w; ++x) {
                unsigned s = 0;
                for (int d = -2; d <= 2; ++d)
                    s += img[y * w + x + static_cast<size_t>(d + 2) - 2u];
                tmp[y * w + x] = static_cast<uint8_t>(s / 5);
            }
        for (size_t y = 2; y + 2 < h; ++y)
            for (size_t x = 0; x < w; ++x) {
                unsigned s = 0;
                for (int d = -2; d <= 2; ++d)
                    s += tmp[(y + static_cast<size_t>(d + 2) - 2u) * w + x];
                img[y * w + x] = static_cast<uint8_t>(s / 5);
            }
    }
    return img;
}
} // namespace

BINCV_TEST(Steered, BinZeroIsTheBasePatternExactly) {
    // The Q16 identity rotation is exact -- cos = 65536, sin = 0 rounds every offset
    // to itself -- so bin 0 must BE the base pattern, byte for byte.
    BriefPattern<kBits> base;
    makeBriefPattern<kBits>(base);
    SteeredBriefPattern<kBits> steered;
    makeSteeredBriefPattern<kBits>(steered, base);
    size_t diff = 0;
    for (size_t i = 0; i < kBits; ++i)
        if (steered.bin[0].pair[i].ax != base.pair[i].ax ||
            steered.bin[0].pair[i].ay != base.pair[i].ay ||
            steered.bin[0].pair[i].bx != base.pair[i].bx ||
            steered.bin[0].pair[i].by != base.pair[i].by) ++diff;
    BINCV_CHECK(diff == 0);
    // And a half-turn is exact too: (x, y) -> (-x, -y), no rounding involved.
    size_t diffHalf = 0;
    for (size_t i = 0; i < kBits; ++i)
        if (steered.bin[15].pair[i].ax != -base.pair[i].ax ||
            steered.bin[15].pair[i].ay != -base.pair[i].ay) ++diffHalf;
    BINCV_CHECK(diffHalf == 0);
}

BINCV_TEST(Steered, AngleBinsQuantizeToNearestStep) {
    const float deg = 0.017453292519943295f;
    BINCV_CHECK(briefAngleBin(0.0f) == 0);
    BINCV_CHECK(briefAngleBin(3.0f * deg) == 0);     // inside bin 0's half-width
    BINCV_CHECK(briefAngleBin(12.0f * deg) == 1);    // a bin center
    BINCV_CHECK(briefAngleBin(-12.0f * deg) == 29);  // wraps, does not clamp
    BINCV_CHECK(briefAngleBin(180.0f * deg) == 15);
    BINCV_CHECK(briefAngleBin(-180.0f * deg) == 15);
}

BINCV_TEST(Steered, ZeroAnglesReproduceComputeBriefBitForBit) {
    // Steering is an EXTENSION, not a reinterpretation: with every angle in bin 0 the
    // steered kernel must produce computeBrief's exact output, keep flags included.
    constexpr size_t kW = 160, kH = 120;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 21);
    BriefPattern<kBits> base;
    makeBriefPattern<kBits>(base);
    SteeredBriefPattern<kBits> steered;
    makeSteeredBriefPattern<kBits>(steered, base);

    std::vector<float> kp;
    for (int y = 5; y < static_cast<int>(kH) - 5; y += 13)
        for (int x = 5; x < static_cast<int>(kW) - 5; x += 17) {
            kp.push_back(static_cast<float>(x));
            kp.push_back(static_cast<float>(y));
        }
    const size_t n = kp.size() / 2;
    const std::vector<float> angles(n, 0.0f);
    std::vector<uint32_t> plain(n * kWords), rot(n * kWords);
    std::vector<uint8_t> keepP(n), keepR(n);
    computeBrief<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, kp.data(), n, base,
                                           plain.data(), keepP.data());
    computeBriefSteered<kBits, uint8_t, uint32_t>(img.data(), kW, kH, kW, kp.data(), n,
                                                  angles.data(), steered, rot.data(),
                                                  keepR.data());
    size_t same = 0;
    for (size_t i = 0; i < n * kWords; ++i)
        if (plain[i] == rot[i]) ++same;
    std::printf(" zero-angle: %zu of %zu words identical\n", same, n * kWords);
    BINCV_CHECK(same == n * kWords);
    for (size_t i = 0; i < n; ++i) BINCV_CHECK(keepP[i] == keepR[i]);
}

BINCV_TEST(Steered, AQuarterTurnIsRecoveredAndUnsteeredBriefLosesIt) {
    // The ORB paper's own figure, as a test: rotate the content 90 degrees, and
    // unsteered BRIEF's matches die while steered ones survive. The margin is the
    // whole reason steering exists, so it is asserted, not just printed.
    constexpr size_t kW = 240, kH = 240;
    const std::vector<uint8_t> a = smoothImage(kW, kH, 77);
    std::vector<uint8_t> b(kW * kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x)
            b[x * kW + (kH - 1 - y)] = a[y * kW + x];   // (x, y) -> (kH-1-y, x)

    std::vector<float> kpA, kpB;
    for (int y = 40; y < 200; y += 12)
        for (int x = 40; x < 200; x += 12) {
            kpA.push_back(static_cast<float>(x));
            kpA.push_back(static_cast<float>(y));
            kpB.push_back(static_cast<float>(static_cast<int>(kH) - 1 - y));
            kpB.push_back(static_cast<float>(x));
        }
    const size_t n = kpA.size() / 2;

    BriefPattern<kBits> base;
    makeBriefPattern<kBits>(base);
    SteeredBriefPattern<kBits> steered;
    makeSteeredBriefPattern<kBits>(steered, base);

    std::vector<float> angA(n), angB(n);
    keypointOrientation<uint8_t>(a.data(), kW, kH, kW, kpA.data(), n, angA.data());
    keypointOrientation<uint8_t>(b.data(), kW, kH, kW, kpB.data(), n, angB.data());

    std::vector<uint32_t> dA(n * kWords), dB(n * kWords), uA(n * kWords), uB(n * kWords);
    computeBriefSteered<kBits, uint8_t, uint32_t>(a.data(), kW, kH, kW, kpA.data(), n,
                                                  angA.data(), steered, dA.data());
    computeBriefSteered<kBits, uint8_t, uint32_t>(b.data(), kW, kH, kW, kpB.data(), n,
                                                  angB.data(), steered, dB.data());
    computeBrief<kBits, uint8_t, uint32_t>(a.data(), kW, kH, kW, kpA.data(), n, base,
                                           uA.data());
    computeBrief<kBits, uint8_t, uint32_t>(b.data(), kW, kH, kW, kpB.data(), n, base,
                                           uB.data());

    unsigned long long steeredDist = 0, unsteeredDist = 0;
    for (size_t i = 0; i < n; ++i) {
        steeredDist += hammingDistance<uint32_t>(dA.data() + i * kWords,
                                                 dB.data() + i * kWords, kWords);
        unsteeredDist += hammingDistance<uint32_t>(uA.data() + i * kWords,
                                                   uB.data() + i * kWords, kWords);
    }
    std::vector<DescriptorMatch> mS(n), mU(n);
    matchDescriptors<uint32_t>(dB.data(), n, dA.data(), n, kWords, mS.data(), 80);
    matchDescriptors<uint32_t>(uB.data(), n, uA.data(), n, kWords, mU.data(), 80);
    size_t okS = 0, okU = 0;
    for (size_t i = 0; i < n; ++i) {
        if (mS[i].valid && mS[i].trainIndex == i) ++okS;
        if (mU[i].valid && mU[i].trainIndex == i) ++okU;
    }
    std::printf(" quarter turn, %zu keypoints: mean distance steered %.1f unsteered %.1f"
                " (of %zu bits); correct matches steered %zu unsteered %zu\n",
                n, static_cast<double>(steeredDist) / static_cast<double>(n),
                static_cast<double>(unsteeredDist) / static_cast<double>(n), kBits, okS,
                okU);
    // A same-point steered pair must be far below chance (128); unsteered at a quarter
    // turn IS chance. The match-rate margin is the practical statement of the same.
    BINCV_CHECK(steeredDist * 2 < unsteeredDist);
    BINCV_CHECK(okS > (n * 3) / 5);
    BINCV_CHECK(okU < n / 5);
}

BINCV_TEST(OrbPattern, TheVendoredTableStaysInsideTheDisc) {
    // The table's exact geometric envelope, PINNED: coordinates within [-13, 12]
    // per axis, widest sample at norm sqrt(338). A mis-transcribed entry cannot
    // hide behind "the descriptors still mostly match" -- it moves this number
    // or the byte-equality test below.
    int maxNormSq = 0, lo = 127, hi = -128;
    for (size_t i = 0; i < 256; ++i) {
        const BriefPair& q = kOrbBriefPattern.pair[i];
        const int c[4] = {q.ax, q.ay, q.bx, q.by};
        for (int j = 0; j < 4; ++j) {
            lo = std::min(lo, c[j]);
            hi = std::max(hi, c[j]);
        }
        maxNormSq = std::max(maxNormSq,
                             std::max(c[0] * c[0] + c[1] * c[1], c[2] * c[2] + c[3] * c[3]));
    }
    std::printf(" ORB table: coords [%d, %d], widest sample norm^2 %d\n", lo, hi, maxNormSq);
    BINCV_CHECK_EQ(lo, -13);
    BINCV_CHECK_EQ(hi, 12);
    BINCV_CHECK_EQ(maxNormSq, 338);
    // And rotating it must fit int8 at every bin -- the assert inside the
    // builder fires in debug builds; building it at all is the release check.
    SteeredBriefPattern<256> steered;
    makeSteeredBriefPattern<256>(steered, kOrbBriefPattern);
    BINCV_CHECK(steered.bin[0].pair[0].ax == kOrbBriefPattern.pair[0].ax);
}

#ifdef BINCV_WITH_OPENCV
BINCV_TEST(OrbPattern, BitExactAgainstCvOrbAtAngleZero) {
    // THE PROVENANCE PROOF. A wrong value anywhere in the 1024 vendored numbers
    // flips descriptor bits, so byte equality against cv::ORB::compute over real
    // sampling is what shows the table is OpenCV's, not one like it. The
    // controlled variables: angles are zero (bin 0 is the identity rotation on
    // both sides); keypoints sit on integer coordinates (cv::ORB rounds,
    // computeBrief truncates -- equal only on integers, which is a property of
    // this test, not of the kernels) and clear of cv::ORB's 31-pixel border
    // filter; and binCV's arm reads the SAME blurred pixels -- which requires
    // reproducing ORB's exact dance: copy into a border-extended buffer and
    // GaussianBlur the interior SUB-VIEW in place. A plain out-of-place blur of
    // the same image with the same parameters differs by +/-1 on ~20% of pixels
    // (a different rounding path inside OpenCV), and near-tie comparisons then
    // flip ~7% of descriptor bytes -- measured here before this test settled on
    // the dance. Interop callers should know that: "the same blur" means the
    // same code path, not the same parameters.
    constexpr int kW = 320, kH = 240;
    const std::vector<uint8_t> img = texturedImage(static_cast<size_t>(kW),
                                                   static_cast<size_t>(kH), 4242);
    cv::Mat gray(kH, kW, CV_8U);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
            gray.at<uint8_t>(y, x) = img[static_cast<size_t>(y) * kW +
                                         static_cast<size_t>(x)];
    // ORB's own preprocessing, reproduced exactly (see the header comment).
    cv::Mat ext;
    cv::copyMakeBorder(gray, ext, 31, 31, 31, 31, cv::BORDER_REFLECT_101);
    cv::Mat sub = ext(cv::Rect(31, 31, kW, kH));
    cv::GaussianBlur(sub, sub, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
    const cv::Mat blurred = sub.clone();

    std::vector<cv::KeyPoint> kps;
    std::vector<float> xy;
    for (int y = 40; y < kH - 40; y += 16) {
        for (int x = 40; x < kW - 40; x += 16) {
            cv::KeyPoint k(static_cast<float>(x), static_cast<float>(y), 31.0f);
            k.angle = 0.0f;
            k.octave = 0;
            kps.push_back(k);
            xy.push_back(static_cast<float>(x));
            xy.push_back(static_cast<float>(y));
        }
    }
    const size_t n = xy.size() / 2;

    cv::Mat cvDesc;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(gray, kps, cvDesc);
    BINCV_CHECK_EQ(static_cast<size_t>(cvDesc.rows), n);   // nothing filtered

    // uint8 words so the byte layout question never arises: bit b of byte w is
    // pair 8w+b on both sides.
    std::vector<uint8_t> ours(n * 32);
    computeBrief<256, uint8_t, uint8_t>(blurred.ptr<uint8_t>(0),
                                        static_cast<size_t>(kW),
                                        static_cast<size_t>(kH), blurred.step,
                                        xy.data(), n, kOrbBriefPattern, ours.data());
    size_t badBytes = 0;
    for (size_t i = 0; i < n; ++i)
        for (int j = 0; j < 32; ++j)
            if (ours[i * 32 + static_cast<size_t>(j)] !=
                cvDesc.at<uint8_t>(static_cast<int>(i), j)) ++badBytes;
    std::printf(" %zu keypoints, %zu of %zu descriptor bytes differ from cv::ORB\n", n,
                badBytes, n * 32);
    BINCV_CHECK_EQ(badBytes, size_t{0});
}
#endif // BINCV_WITH_OPENCV

BINCV_TEST_MAIN("test_descriptor")
