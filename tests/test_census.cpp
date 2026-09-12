// The census transform.
//
// Core-only. The contract is exactness against a per-pixel reference written
// independently of the kernel's loops, at all four word types -- plus the two
// structural properties the op is used FOR: order-dependence only (an order
// reversal flips every strict comparison, the testable face of the monotone
// invariance stereo costs borrow census for) and shift-equivariance away from
// borders (the property the disparity cost leans on against a shifted census).
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv/binMat.hpp"
#include "bincv/ops/census.hpp"
#include "bincv/ops/reduce.hpp"
#include "test_util.hpp"

namespace {
using namespace bincv;

std::vector<uint8_t> texturedImage(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    uint64_t st = seed;
    for (auto& v : img) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<uint8_t>(st >> 40);
    }
    return img;
}

/// The reference: the definition, one pixel at a time.
template <size_t K>
unsigned referenceBit(const std::vector<uint8_t>& img, size_t w, size_t h,
                      const CensusPattern<K>& pat, size_t k, long long x, long long y) {
    const long long xn = x + pat.at[k].dx, yn = y + pat.at[k].dy;
    if (xn < 0 || yn < 0 || xn >= static_cast<long long>(w) ||
        yn >= static_cast<long long>(h))
        return 0;
    return img[static_cast<size_t>(yn) * w + static_cast<size_t>(xn)] >
                   img[static_cast<size_t>(y) * w + static_cast<size_t>(x)]
               ? 1u
               : 0u;
}

template <typename W>
void checkAgainstReference(uint64_t seed) {
    constexpr size_t kW = 90, kH = 45;   // not word-aligned on purpose
    const std::vector<uint8_t> img = texturedImage(kW, kH, seed);
    std::vector<BinMat<W>> store;
    store.reserve(24);
    std::vector<BinMatView<W>> planes;
    for (size_t k = 0; k < 24; ++k) {
        store.emplace_back(kW, kH);
        planes.push_back(store.back().view());
    }
    censusTransform<24, uint8_t, W>(img.data(), kW, kH, kW, kCensus5x5, planes.data());
    size_t bad = 0;
    for (size_t k = 0; k < 24; ++k)
        for (size_t y = 0; y < kH; ++y)
            for (size_t x = 0; x < kW; ++x) {
                const unsigned got =
                    store[k].at(static_cast<int>(y), static_cast<int>(x)) ? 1u : 0u;
                if (got != referenceBit<24>(img, kW, kH, kCensus5x5, k,
                                            static_cast<long long>(x),
                                            static_cast<long long>(y)))
                    ++bad;
            }
    std::printf(" W=%zu-bit: %zu of %zu bits differ from the reference\n", sizeof(W) * 8,
                bad, 24 * kW * kH);
    BINCV_CHECK_EQ(bad, size_t{0});
    // The padding invariant, through the counter that would over-count it.
    for (size_t k = 0; k < 24; ++k) {
        size_t ref = 0;
        for (size_t y = 0; y < kH; ++y)
            for (size_t x = 0; x < kW; ++x)
                ref += referenceBit<24>(img, kW, kH, kCensus5x5, k,
                                        static_cast<long long>(x),
                                        static_cast<long long>(y));
        BINCV_CHECK_EQ(countNonZero(store[k].constView()), ref);
    }
}
} // namespace

BINCV_TEST(Census, MatchesThePerPixelReferenceAtEveryWordType) {
    checkAgainstReference<uint8_t>(11);
    checkAgainstReference<uint16_t>(22);
    checkAgainstReference<uint32_t>(33);
    checkAgainstReference<uint64_t>(44);
}

BINCV_TEST(Census, OrderReversalFlipsEveryStrictComparison) {
    // The structural property stereo borrows census FOR is that only the ORDER of
    // gray levels matters. On full-range uint8 content the only strictly monotone
    // self-map is the identity, so the testable form is the reversal: inverting
    // the image (255 - v) reverses every strict comparison, so a plane of the
    // original and the same plane of the inverted image can never share a set
    // bit -- I(n) > I(c) and I(n) < I(c) cannot both hold. Ties and border
    // comparisons are 0 on both sides, so zero overlap is exact, not approximate.
    constexpr size_t kW = 64, kH = 48;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 5);
    std::vector<uint8_t> inv(img.size());
    for (size_t i = 0; i < img.size(); ++i) inv[i] = static_cast<uint8_t>(255u - img[i]);

    BinMat<uint32_t> a[8] = {{kW, kH}, {kW, kH}, {kW, kH}, {kW, kH},
                             {kW, kH}, {kW, kH}, {kW, kH}, {kW, kH}};
    BinMat<uint32_t> b[8] = {{kW, kH}, {kW, kH}, {kW, kH}, {kW, kH},
                             {kW, kH}, {kW, kH}, {kW, kH}, {kW, kH}};
    BinMatView<uint32_t> va[8], vb[8];
    for (int k = 0; k < 8; ++k) {
        va[k] = a[k].view();
        vb[k] = b[k].view();
    }
    censusTransform<8, uint8_t, uint32_t>(img.data(), kW, kH, kW, kCensus3x3, va);
    censusTransform<8, uint8_t, uint32_t>(inv.data(), kW, kH, kW, kCensus3x3, vb);
    size_t overlap = 0;
    for (int k = 0; k < 8; ++k) {
        const uint32_t* pa = a[k].data();
        const uint32_t* pb = b[k].data();
        for (size_t w = 0; w < a[k].sizeInWords(); ++w)
            overlap += static_cast<size_t>(
                __builtin_popcountll(static_cast<unsigned long long>(pa[w] & pb[w])));
    }
    std::printf(" inverted-image planes overlap original on %zu bits (must be 0)\n",
                overlap);
    BINCV_CHECK_EQ(overlap, size_t{0});
}

BINCV_TEST(Census, ShiftEquivariantAwayFromBorders) {
    // The property the disparity cost leans on: census(shift(img)) equals
    // shift(census(img)) wherever neither side touched a border. Checked at a
    // 7-pixel shift over the interior.
    constexpr size_t kW = 96, kH = 40;
    constexpr int kShift = 7;
    const std::vector<uint8_t> img = texturedImage(kW, kH, 77);
    std::vector<uint8_t> shifted(kW * kH, 0);
    for (size_t y = 0; y < kH; ++y)
        for (size_t x = 0; x + kShift < kW; ++x)
            shifted[y * kW + x] = img[y * kW + x + kShift];

    BinMat<uint32_t> a[8] = {{kW, kH}, {kW, kH}, {kW, kH}, {kW, kH},
                             {kW, kH}, {kW, kH}, {kW, kH}, {kW, kH}};
    BinMat<uint32_t> b[8] = {{kW, kH}, {kW, kH}, {kW, kH}, {kW, kH},
                             {kW, kH}, {kW, kH}, {kW, kH}, {kW, kH}};
    BinMatView<uint32_t> va[8], vb[8];
    for (int k = 0; k < 8; ++k) {
        va[k] = a[k].view();
        vb[k] = b[k].view();
    }
    censusTransform<8, uint8_t, uint32_t>(img.data(), kW, kH, kW, kCensus3x3, va);
    censusTransform<8, uint8_t, uint32_t>(shifted.data(), kW, kH, kW, kCensus3x3, vb);
    size_t bad = 0, compared = 0;
    for (int k = 0; k < 8; ++k)
        for (size_t y = 1; y + 1 < kH; ++y)
            for (size_t x = 1; x + kShift + 1 < kW; ++x) {
                const unsigned s = b[k].at(static_cast<int>(y), static_cast<int>(x));
                const unsigned o =
                    a[k].at(static_cast<int>(y), static_cast<int>(x + kShift));
                if (s != o) ++bad;
                ++compared;
            }
    std::printf(" shift-equivariance: %zu of %zu interior bits differ\n", bad, compared);
    BINCV_CHECK_EQ(bad, size_t{0});
}

BINCV_TEST_MAIN("test_census")
