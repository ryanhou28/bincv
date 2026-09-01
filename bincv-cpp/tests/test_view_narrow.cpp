// ===========================================================================
// / narrowLevel -- a 64-bit level tracked through the 32-bit vector kernels.
//
// binCV's vectorized tracking kernels are gated on 32-bit words, because an LK window
// is 31 pixels and a wider word is more than half idle. A caller who wants 64-bit words
// elsewhere -- where they genuinely halve the work -- used to face a choice between that
// and a tracker running 8.6x slow. `narrowLevel` removes the choice, and it is a VIEW:
// no copy, no allocation.
//
// It is exact only if a 64-bit plane and a 32-bit plane with twice the stride are the
// same bytes. They are, on little-endian, and this file checks that pixel by pixel
// rather than asserting it -- and then checks the thing that actually matters, which is
// that the TRACKER agrees.
// ===========================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/ops/opticalFlow.hpp"
#include "test_util.hpp"

namespace {

/// The same content in both word widths, so every downstream comparison is of the
/// KERNELS rather than of the data.
template <typename W>
void fill(bincv::BinMat<W>& m, uint64_t seed) {
    uint64_t st = seed;
    for (int y = 0; y < m.rows(); ++y) {
        for (int x = 0; x < m.cols(); ++x) {
            st = st * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            m.set(y, x, ((st >> 42) % 5u == 0u) ? 1u : 0u);
        }
    }
}

}  // namespace

BINCV_TEST(Narrow, PlaneReadsIdenticallyAtBothWidths) {
    constexpr int W = 200, H = 9;
    bincv::BinMat<uint64_t> m(W, H);
    fill(m, UINT64_C(4242));
    const bincv::BinMatConstView<uint32_t> v32 = bincv::narrowPlane<uint64_t>(m.constPlane(0));
    size_t bad = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const unsigned a = m.at(y, x) ? 1u : 0u;
            const unsigned b =
                (v32.row(static_cast<size_t>(y))[static_cast<size_t>(x) / 32] >>
                 (static_cast<size_t>(x) % 32)) & 1u;
            if (a != b) ++bad;
        }
    }
    std::printf(" 64-bit plane read as 32-bit: %zu of %d pixels differ\n", bad, W * H);
    BINCV_CHECK_EQ(bad, size_t{0});
}

BINCV_TEST(Narrow, PaddingBitsStayZero) {
    // The invariant every word-wise reduction in the library depends on. A width that is
    // not a multiple of 64 leaves padding in the 64-bit word; the derived 32-bit words
    // must inherit it as zero, including a high half that is ENTIRELY padding.
    for (int W : {94, 100, 127, 129}) {
        bincv::BinMat<uint64_t> m(W, 4);
        fill(m, static_cast<uint64_t>(W));
        const bincv::BinMatConstView<uint32_t> v = bincv::narrowPlane<uint64_t>(m.constPlane(0));
        size_t bad = 0;
        const size_t words = bincv::impl::minRowWords<uint32_t>(static_cast<size_t>(W));
        for (size_t y = 0; y < v.height; ++y) {
            for (size_t i = 0; i < words; ++i) {
                const size_t firstPixel = i * 32;
                if (firstPixel + 32 <= static_cast<size_t>(W)) continue;
                const size_t valid = static_cast<size_t>(W) - firstPixel;
                const uint32_t tail = v.row(y)[i] >> valid;
                if (valid < 32 && tail != 0) ++bad;
            }
        }
        if (bad) std::printf(" width %d: %zu rows carry non-zero padding\n", W, bad);
        BINCV_CHECK_EQ(bad, size_t{0});
    }
}

BINCV_TEST(Narrow, TrackingA64BitLevelMatches32BitExactly) {
    // THE CLAIM THAT MATTERS. A caller with 64-bit storage narrows at the call and gets
    // the vector kernels; the flow vectors must be identical to a native 32-bit build,
    // not merely close, because the kernels compute the same integers from the same bits.
    constexpr int W = 160, H = 120;
    bincv::BinMat<uint64_t> p64(W, H), n64(W, H);
    bincv::BinMat<uint32_t> p32(W, H), n32(W, H);
    fill(p64, UINT64_C(7));
    fill(n64, UINT64_C(9));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            p32.set(y, x, p64.at(y, x));
            n32.set(y, x, n64.at(y, x));
        }
    }

    bincv::SignedQuantMat<1, uint64_t> dx64(W, H), dy64(W, H);
    bincv::SignedQuantMat<1, uint32_t> dx32(W, H), dy32(W, H);
    bincv::derivativeX(p64, dx64);
    bincv::derivativeY(p64, dy64);
    bincv::derivativeX(p32, dx32);
    bincv::derivativeY(p32, dy32);

    const bincv::LKLevelN<1, uint64_t> lv64 = bincv::lkLevel<1>(p64, n64, dx64, dy64);
    const bincv::LKLevelN<1, uint32_t> lv32 = bincv::lkLevel<1>(p32, n32, dx32, dy32);
    const bincv::LKLevelN<1, uint32_t> narrowed = bincv::narrowLevel(lv64);

    std::vector<bincv::Point2f> pts;
    for (int y = 20; y < H - 20; y += 7) {
        for (int x = 20; x < W - 20; x += 9) {
            pts.push_back(bincv::Point2f{static_cast<float>(x), static_cast<float>(y)});
        }
    }
    const size_t n = pts.size();
    std::vector<bincv::Point2f> outN(n), outR(n);
    std::vector<uint8_t> stN(n), stR(n);
    std::vector<float> errN(n), errR(n);
    bincv::LKParams params;

    bincv::calcOpticalFlowPyrLK<1, uint32_t>(&narrowed, 1, pts.data(), outN.data(),
                                             stN.data(), errN.data(), n, params);
    bincv::calcOpticalFlowPyrLK<1, uint32_t>(&lv32, 1, pts.data(), outR.data(), stR.data(),
                                             errR.data(), n, params);

    size_t posDiff = 0, statusDiff = 0, errDiff = 0, tracked = 0;
    for (size_t i = 0; i < n; ++i) {
        if (stN[i] != stR[i]) ++statusDiff;
        if (stN[i]) ++tracked;
        if (outN[i].x != outR[i].x || outN[i].y != outR[i].y) ++posDiff;
        if (errN[i] != errR[i]) ++errDiff;
    }
    std::printf(" %zu points (%zu tracked): narrowed-64 vs native-32 -- %zu positions,"
                " %zu status, %zu err differ\n", n, tracked, posDiff, statusDiff, errDiff);
    std::printf(" kernel for the narrowed level: %s\n",
                bincv::lkPathName<bincv::LKLevelN<1, uint32_t>>());
    BINCV_CHECK(n > 100);
    BINCV_CHECK(tracked > 20);
    BINCV_CHECK_EQ(posDiff, size_t{0});
    BINCV_CHECK_EQ(statusDiff, size_t{0});
    BINCV_CHECK_EQ(errDiff, size_t{0});
}

BINCV_TEST_MAIN("test_view_narrow")
