// Keypoint orientation from the intensity centroid.
//
// Core-only. The load-bearing claim is EXACTNESS, not plausibility: the bit-plane
// spelling's masked-popcount moments must equal the per-pixel spelling's integer
// moments -- same numbers, not close numbers -- because the two are one operation
// with two implementations, and a drift between them would make a pipeline's
// angles depend on which input representation it happened to hold.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/quantMat.hpp"
#include "bincv/ops/orientation.hpp"
#include "test_util.hpp"

namespace {
using namespace bincv;

std::vector<uint8_t> texturedImage(size_t w, size_t h, uint64_t seed, unsigned levels) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = seed;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>((st >> 40) % levels);
    }
    return img;
}

/// The reference: per-pixel integer moments over the same disc, then the same
/// atan2 call. Written independently of the kernel's loops on purpose.
float referenceAngle(const std::vector<uint8_t>& img, size_t w, int cx, int cy,
                     int radius) {
    long long m10 = 0, m01 = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            const long long v = img[static_cast<size_t>(cy + dy) * w +
                                    static_cast<size_t>(cx + dx)];
            m10 += dx * v;
            m01 += dy * v;
        }
    }
    if (m10 == 0 && m01 == 0) return 0.0f;
    return std::atan2(static_cast<float>(m01), static_cast<float>(m10));
}

template <typename W>
void checkAllSpellingsAgree(uint64_t seed) {
    constexpr size_t kW = 160, kH = 120;
    const std::vector<uint8_t> bytes = texturedImage(kW, kH, seed, 2);   // 0/1 pixels
    BinMat<W> packed(kW, kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x) packed.set(static_cast<int>(y), static_cast<int>(x), bytes[y * kW + x]);

    std::vector<float> kp;
    for (int y = 16; y < static_cast<int>(kH) - 16; y += 7)
        for (int x = 16; x < static_cast<int>(kW) - 16; x += 9) {
            kp.push_back(static_cast<float>(x));
            kp.push_back(static_cast<float>(y));
        }
    const size_t n = kp.size() / 2;
    std::vector<float> aWide(n), aBits(n);
    std::vector<uint8_t> keepW(n), keepB(n);
    keypointOrientation<uint8_t>(bytes.data(), kW, kH, kW, kp.data(), n, aWide.data(),
                                 keepW.data());
    keypointOrientation<W>(packed.constView(), kp.data(), n, aBits.data(), keepB.data());

    size_t exact = 0, refExact = 0;
    for (size_t i = 0; i < n; ++i) {
        BINCV_CHECK(keepW[i] == keepB[i]);
        if (aWide[i] == aBits[i]) ++exact;
        const float ref = referenceAngle(bytes, kW, static_cast<int>(kp[2 * i]),
                                         static_cast<int>(kp[2 * i + 1]), 15);
        if (aBits[i] == ref) ++refExact;
    }
    std::printf(" W=%zu-bit: %zu keypoints, wide==bits %zu, bits==reference %zu\n",
                sizeof(W) * 8, n, exact, refExact);
    BINCV_CHECK(exact == n);
    BINCV_CHECK(refExact == n);
}
} // namespace

BINCV_TEST(Orientation, BitPlaneMatchesWideAndReferenceExactly) {
    // Three implementations -- per-pixel reference, wide-sum kernel, masked-popcount
    // kernel -- and one answer. At all four word types, because the segment gather
    // crosses word boundaries differently at each width.
    checkAllSpellingsAgree<uint8_t>(101);
    checkAllSpellingsAgree<uint16_t>(202);
    checkAllSpellingsAgree<uint32_t>(303);
    checkAllSpellingsAgree<uint64_t>(404);
}

BINCV_TEST(Orientation, NBitPlanesWeightExactly) {
    // A 2-bit image's moments must equal the wide spelling's on the same VALUES --
    // plane weighting is part of the contract, not an approximation.
    constexpr size_t kW = 96, kH = 96;
    const std::vector<uint8_t> vals = texturedImage(kW, kH, 55, 4);   // 0..3
    QuantMat<2, uint32_t> q(kW, kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x)
            q.set(static_cast<int>(y), static_cast<int>(x), vals[y * kW + x]);

    std::vector<float> kp;
    for (int y = 20; y < 76; y += 9)
        for (int x = 20; x < 76; x += 9) {
            kp.push_back(static_cast<float>(x));
            kp.push_back(static_cast<float>(y));
        }
    const size_t n = kp.size() / 2;
    std::vector<float> aWide(n), aBits(n);
    keypointOrientation<uint8_t>(vals.data(), kW, kH, kW, kp.data(), n, aWide.data());
    const BinMatConstView<uint32_t> planes[2] = {q.constPlane(0), q.constPlane(1)};
    keypointOrientation<uint32_t>(planes, 2, kp.data(), n, aBits.data());
    size_t exact = 0;
    for (size_t i = 0; i < n; ++i)
        if (aWide[i] == aBits[i]) ++exact;
    std::printf(" N=2: %zu of %zu angles identical\n", exact, n);
    BINCV_CHECK(exact == n);
}

BINCV_TEST(Orientation, KnownMomentsGiveKnownAngles) {
    // Hand-checkable cases: one set pixel at a known offset IS the centroid.
    constexpr size_t kW = 64, kH = 64;
    std::vector<uint8_t> img(kW * kH, 0);
    img[32 * kW + 32 + 5] = 1;    // +x from (32, 32): angle 0
    img[(40 + 6) * kW + 20] = 1;  // +y from (20, 40): straight "down", +pi/2

    const float kp[4] = {32.0f, 32.0f, 20.0f, 40.0f};
    float angle[2];
    keypointOrientation<uint8_t>(img.data(), kW, kH, kW, kp, 2, angle);
    std::printf(" +x pixel: %.6f (want 0), +y pixel: %.6f (want %.6f)\n",
                static_cast<double>(angle[0]), static_cast<double>(angle[1]),
                static_cast<double>(std::atan2(1.0f, 0.0f)));
    BINCV_CHECK(angle[0] == 0.0f);
    BINCV_CHECK(angle[1] == std::atan2(1.0f, 0.0f));
}

BINCV_TEST(Orientation, FlatPatchReportsZeroAndIsKept) {
    // No structure means no orientation; 0 is the deterministic answer, and the
    // keypoint is NOT rejected -- it is inside the image, it is just boring.
    constexpr size_t kW = 64, kH = 64;
    const std::vector<uint8_t> img(kW * kH, 77);
    const float kp[2] = {32.0f, 32.0f};
    float angle = -1.0f;
    uint8_t keep = 0;
    keypointOrientation<uint8_t>(img.data(), kW, kH, kW, kp, 1, &angle, &keep);
    BINCV_CHECK(angle == 0.0f);
    BINCV_CHECK(keep == 1);
}

BINCV_TEST(Orientation, BorderKeypointsAreRejectedNotInvented) {
    constexpr size_t kW = 64, kH = 64;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 9, 256);
    const float kp[6] = {5.0f, 32.0f, 32.0f, 32.0f, 32.0f, 60.0f};
    float angle[3];
    uint8_t keep[3];
    keypointOrientation<uint8_t>(img.data(), kW, kH, kW, kp, 3, angle, keep);
    std::printf(" keep flags: near-left=%u center=%u near-bottom=%u\n", keep[0], keep[1],
                keep[2]);
    BINCV_CHECK(keep[0] == 0 && angle[0] == 0.0f);
    BINCV_CHECK(keep[1] == 1);
    BINCV_CHECK(keep[2] == 0 && angle[2] == 0.0f);
}

BINCV_TEST(Orientation, QuarterTurnShiftsTheAngleAQuarterTurn) {
    // Rotate the CONTENT by 90 degrees and the centroid must follow it. The disc is
    // integer-symmetric under quarter turns, so the moments transform exactly; the
    // angles differ by pi/2 up to atan2's own rounding.
    constexpr size_t kW = 96, kH = 96;
    std::vector<uint8_t> a = texturedImage(kW, kH, 31, 256);
    std::vector<uint8_t> b(kW * kH);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x < kW; ++x)
            b[x * kW + (kH - 1 - y)] = a[y * kW + x];   // (x, y) -> (kH-1-y, x)

    size_t checked = 0;
    for (int yc = 24; yc < 72; yc += 8) {
        for (int xc = 24; xc < 72; xc += 8) {
            const float kpA[2] = {static_cast<float>(xc), static_cast<float>(yc)};
            const float kpB[2] = {static_cast<float>(static_cast<int>(kH) - 1 - yc),
                                  static_cast<float>(xc)};
            float aa, ab;
            keypointOrientation<uint8_t>(a.data(), kW, kH, kW, kpA, 1, &aa);
            keypointOrientation<uint8_t>(b.data(), kW, kH, kW, kpB, 1, &ab);
            const double twoPi = 6.283185307179586;
            double d = static_cast<double>(ab) - static_cast<double>(aa) - twoPi / 4.0;
            while (d > 3.15) d -= twoPi;
            while (d < -3.15) d += twoPi;
            BINCV_CHECK(std::abs(d) < 1e-5);
            ++checked;
        }
    }
    std::printf(" %zu quarter-turn pairs within 1e-5\n", checked);
}

BINCV_TEST_MAIN("test_orientation")
