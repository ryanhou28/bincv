// Binary descriptors and Hamming matching.
//
// Core-only: descriptors need no OpenCV, and the point of the family is that matching
// is popcount(a ^ b) -- the operation binCV is built out of.
//
// WHAT IS DELIBERATELY NOT CLAIMED. These are not cv::ORB descriptors. OpenCV uses a
// specific learned 256-pair table from the ORB paper plus an orientation from the
// intensity centroid; binCV reproduces neither, and BriefPattern exists so a caller
// who needs OpenCV-comparable descriptors can supply OpenCV's table. Descriptors from
// two different patterns are incomparable -- that is true of BRIEF generally, and it
// is why the pattern is an argument rather than a hidden constant.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/descriptor.hpp"
#include "test_util.hpp"

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
    std::printf(" keep flags: corner=%u centre=%u corner=%u\n", keep[0], keep[1], keep[2]);
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

BINCV_TEST_MAIN("test_descriptor")
