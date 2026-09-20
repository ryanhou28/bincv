// The ternary/N-bit derivative and the gradient covariance, device against
// host. A CUDA translation unit because two of its claims can only be asked of
// a kernel: that nvcc's device pass gives the HOST's answer for the host's own
// `impl::signedDifference` and `impl::combineBitSlicedPairs`, which this
// backend calls rather than forking.
//
// WHAT IS PROVEN HERE, AND IN WHAT ORDER
//
//  1. THE DERIVATIVE, WORD FOR WORD against the host kernel over a shape sweep,
//     a border sweep and a depth sweep -- widths that do and do not end on a
//     word boundary, the width == 1 and height == 1 degenerates, every
//     BorderType, both constant fills, and N = 1..4. Both axes.
//  2. THE THREE DERIVATIVE ARMS AGREE IN ONE BINARY: derivativeX, derivativeY,
//     and derivativeXY with impl::derivativeFusedArmEnabled() true AND false.
//     The fused arm has no host twin, so its correctness is DEFINED as "equals
//     host derivativeX and host derivativeY run separately" -- a definition,
//     not something inherited, which is why it is checked against both host
//     kernels rather than against itself.
//  3. THE PADDING INVARIANT AND THE CANONICAL-ZERO RULE, read off the
//     downloaded words rather than inferred from equality: no bit at or past
//     `width` is set in ANY destination plane, sign included, and no pixel
//     anywhere has magnitude 0 with its sign bit set.
//  4. DIRTY SOURCE PADDING. A view of width W over an all-ones buffer of width
//     W2 > W at the same stride, on BOTH sides. The device result must equal
//     the host result and its own padding must come back clean. Without this
//     the "the source mask is dead BECAUSE the right-border fixup is there"
//     coupling is untested on this backend.
//  5. THE COVARIANCE, FOUR WAYS, because a kernel that agrees with only one of
//     them is a plausible-looking wrong answer: against the host N-bit kernel,
//     against the host FIVE-ARGUMENT ternary kernel at N = 1, against this
//     backend's existing countCovarianceBatchAsync at N = 1 recombined through
//     CovarianceCount::crossTerm, and batch against single-region.
//  6. BOTH COVARIANCE BATCH ARMS IN ONE BINARY -- the funnel-aligned run and
//     the word-by-word one, which divide DIFFERENT units across the block --
//     held to one answer over the whole rect set, and both to the
//     single-region form, whose traversal is different again.
//  7. THE WINDOW CONTRACT: clipped not rejected, negative origins legal, wholly
//     outside gives {0, 0, 0}, a bit at or past width never counted, and the
//     sign planes read only where both magnitudes are set.
//  8. THE NAMED DEVICE DOMAINS are asserted in a checked build and reported as
//     cudaErrorInvalidValue in an unchecked one.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/core/types.hpp"
#include "bincv/cuda/covariance.hpp"
#include "bincv/cuda/derivative.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/reduce.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/covariance.hpp"
#include "bincv/ops/derivative.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace {

using bincv::BinMatConstView;
using bincv::BinMatView;
using bincv::BorderType;
using bincv::GradientCovariance;
using bincv::QuantMat;
using bincv::Rect;
using bincv::SignedQuantMat;
using bincv::cuda::DeviceBinMat;
using bincv::cuda::DeviceGradientCovariance;
using bincv::cuda::DevicePlaneBlockConstView;
using bincv::cuda::DevicePlaneBlockView;

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Fills an N-bit level from a pseudo-random frame through the host packer, so
/// the source is a level a real caller could hold rather than hand-set words.
template <size_t N>
void fillQuant(QuantMat<N, uint32_t>& m, uint64_t seed) {
    const size_t w = m.getWidth();
    const size_t h = m.getHeight();
    if (w == 0 || h == 0) return;
    std::vector<uint8_t> frame(w * h);
    for (auto& v : frame) v = static_cast<uint8_t>(splitmix(seed));
    BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = m.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(frame.data(), w, h, w,
                                                                    planes);
}

/// The whole plane stack as ONE matrix -- the shape a raw transfer copies.
template <size_t N>
BinMatConstView<uint32_t> stackOf(const QuantMat<N, uint32_t>& m) {
    return BinMatConstView<uint32_t>(m.data(), m.getWidth(), N * m.getHeight(),
                                     m.getAlignedWidth());
}
template <size_t N>
BinMatView<uint32_t> mutableStackOf(QuantMat<N, uint32_t>& m) {
    return BinMatView<uint32_t>(m.data(), m.getWidth(), N * m.getHeight(),
                                m.getAlignedWidth());
}

/// Words differing between two plane stacks, over the PIXEL words only.
size_t wordsDiffering(BinMatConstView<uint32_t> a, BinMatConstView<uint32_t> b) {
    if (a.width == 0 || a.height == 0) return 0;
    const size_t words = bincv::impl::minRowWords<uint32_t>(a.width);
    size_t bad = 0;
    for (size_t y = 0; y < a.height; ++y)
        for (size_t i = 0; i < words; ++i)
            if (a.row(y)[i] != b.row(y)[i]) ++bad;
    return bad;
}

/// Padding bits set at or past `width`, across every row of a stack.
size_t paddingBitsSet(BinMatConstView<uint32_t> v) {
    if (v.width == 0 || v.height == 0) return 0;
    const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(v.width);
    if (tail == 0xFFFFFFFFu) return 0;
    const size_t words = bincv::impl::minRowWords<uint32_t>(v.width);
    size_t bad = 0;
    for (size_t y = 0; y < v.height; ++y)
        if ((v.row(y)[words - 1] & ~tail) != 0u) ++bad;
    return bad;
}

/// Pixels with magnitude 0 and the sign bit SET -- a "negative zero", which
/// SignedQuantMat's canonical-zero rule forbids. Word-wise, and the padding is
/// masked out so a clean padding bit is not mistaken for a violation.
template <size_t N>
size_t canonicalZeroViolations(const QuantMat<N + 1, uint32_t>& signedBlock, size_t width,
                               size_t height) {
    if (width == 0 || height == 0) return 0;
    const size_t words = bincv::impl::minRowWords<uint32_t>(width);
    const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(width);
    size_t bad = 0;
    for (size_t y = 0; y < height; ++y) {
        for (size_t i = 0; i < words; ++i) {
            uint32_t magOr = 0u;
            for (size_t p = 0; p < N; ++p) magOr |= signedBlock.plane(p).row(y)[i];
            uint32_t sign = signedBlock.plane(N).row(y)[i];
            if (i + 1 == words) sign &= tail;
            if ((sign & ~magOr) != 0u) ++bad;
        }
    }
    return bad;
}

// ---------------------------------------------------------------------------
// One derivative configuration, run on both sides and compared
// ---------------------------------------------------------------------------

/// Counters a sweep accumulates, so a 2,000-configuration sweep is a handful of
/// checks rather than 8,000 -- the shape the other suites here use.
struct DerivTally {
    size_t configs = 0;
    size_t badWordsX = 0;
    size_t badWordsY = 0;
    size_t badWordsFusedX = 0;
    size_t badWordsFusedY = 0;
    size_t badWordsFusedOffX = 0;
    size_t badWordsFusedOffY = 0;
    size_t padding = 0;
    size_t canonical = 0;
    size_t launchErrors = 0;
};

/// @param srcOverride When non-empty, the DEVICE source view to use instead of
/// a freshly uploaded copy of `src` -- the dirty-padding case hands a narrow
/// view over a wider all-ones block here.
template <size_t N>
void runDerivativeConfig(const QuantMat<N, uint32_t>& src, BorderType borderType,
                         bool borderValue, DerivTally& tally,
                         const DevicePlaneBlockConstView* srcOverride = nullptr) {
    const size_t w = src.getWidth();
    const size_t h = src.getHeight();
    ++tally.configs;

    // --- the host answer, both axes, through the shipped entry points ---
    SignedQuantMat<N, uint32_t> hostDx(static_cast<int>(w), static_cast<int>(h));
    SignedQuantMat<N, uint32_t> hostDy(static_cast<int>(w), static_cast<int>(h));
    bincv::derivativeX<N, uint32_t>(src, hostDx, borderType, borderValue);
    bincv::derivativeY<N, uint32_t>(src, hostDy, borderType, borderValue);

    // --- the device source ---
    DeviceBinMat devSrcOwned(static_cast<int>(w), static_cast<int>(N * h));
    DevicePlaneBlockConstView devSrc;
    if (srcOverride != nullptr) {
        devSrc = *srcOverride;
    } else {
        if (bincv::cuda::upload(stackOf(src), devSrcOwned.view()) != cudaSuccess)
            ++tally.launchErrors;
        devSrc = bincv::cuda::planeBlock(devSrcOwned.constView(), N);
    }

    // --- four device destinations: two single-axis, two fused ---
    DeviceBinMat devDx(static_cast<int>(w), static_cast<int>((N + 1) * h));
    DeviceBinMat devDy(static_cast<int>(w), static_cast<int>((N + 1) * h));
    DeviceBinMat devFx(static_cast<int>(w), static_cast<int>((N + 1) * h));
    DeviceBinMat devFy(static_cast<int>(w), static_cast<int>((N + 1) * h));
    const auto blockOf = [](DeviceBinMat& m) {
        return bincv::cuda::planeBlock(m.view(), N + 1);
    };

    if (bincv::cuda::derivativeX(devSrc, blockOf(devDx), borderType, borderValue) !=
        cudaSuccess)
        ++tally.launchErrors;
    if (bincv::cuda::derivativeY(devSrc, blockOf(devDy), borderType, borderValue) !=
        cudaSuccess)
        ++tally.launchErrors;
    if (bincv::cuda::derivativeXY(devSrc, blockOf(devFx), blockOf(devFy), borderType,
                                  borderValue) != cudaSuccess)
        ++tally.launchErrors;
    if (cudaDeviceSynchronize() != cudaSuccess) ++tally.launchErrors;

    // --- download and compare, plane for plane, padding words included ---
    QuantMat<N + 1, uint32_t> gotDx(static_cast<int>(w), static_cast<int>(h));
    QuantMat<N + 1, uint32_t> gotDy(static_cast<int>(w), static_cast<int>(h));
    QuantMat<N + 1, uint32_t> gotFx(static_cast<int>(w), static_cast<int>(h));
    QuantMat<N + 1, uint32_t> gotFy(static_cast<int>(w), static_cast<int>(h));
    const auto fetch = [&](DeviceBinMat& d, QuantMat<N + 1, uint32_t>& out) {
        if (bincv::cuda::download(d.constView(), mutableStackOf(out)) != cudaSuccess)
            ++tally.launchErrors;
    };
    fetch(devDx, gotDx);
    fetch(devDy, gotDy);
    fetch(devFx, gotFx);
    fetch(devFy, gotFy);
    if (cudaDeviceSynchronize() != cudaSuccess) ++tally.launchErrors;

    const BinMatConstView<uint32_t> hx = stackOf<N + 1>(hostDx.planes());
    const BinMatConstView<uint32_t> hy = stackOf<N + 1>(hostDy.planes());
    tally.badWordsX += wordsDiffering(hx, stackOf(gotDx));
    tally.badWordsY += wordsDiffering(hy, stackOf(gotDy));
    tally.badWordsFusedX += wordsDiffering(hx, stackOf(gotFx));
    tally.badWordsFusedY += wordsDiffering(hy, stackOf(gotFy));

    tally.padding += paddingBitsSet(stackOf(gotDx)) + paddingBitsSet(stackOf(gotDy)) +
                     paddingBitsSet(stackOf(gotFx)) + paddingBitsSet(stackOf(gotFy));
    tally.canonical += canonicalZeroViolations<N>(gotDx, w, h) +
                       canonicalZeroViolations<N>(gotDy, w, h) +
                       canonicalZeroViolations<N>(gotFx, w, h) +
                       canonicalZeroViolations<N>(gotFy, w, h);

    // --- THE SAME FUSED CALL WITH THE ARM SWITCHED OFF, IN THIS BINARY ---
    bincv::cuda::impl::derivativeFusedArmEnabled() = false;
    DeviceBinMat offX(static_cast<int>(w), static_cast<int>((N + 1) * h));
    DeviceBinMat offY(static_cast<int>(w), static_cast<int>((N + 1) * h));
    if (bincv::cuda::derivativeXY(devSrc, blockOf(offX), blockOf(offY), borderType,
                                  borderValue) != cudaSuccess)
        ++tally.launchErrors;
    if (cudaDeviceSynchronize() != cudaSuccess) ++tally.launchErrors;
    bincv::cuda::impl::derivativeFusedArmEnabled() = true;

    QuantMat<N + 1, uint32_t> gotOffX(static_cast<int>(w), static_cast<int>(h));
    QuantMat<N + 1, uint32_t> gotOffY(static_cast<int>(w), static_cast<int>(h));
    fetch(offX, gotOffX);
    fetch(offY, gotOffY);
    if (cudaDeviceSynchronize() != cudaSuccess) ++tally.launchErrors;
    tally.badWordsFusedOffX += wordsDiffering(hx, stackOf(gotOffX));
    tally.badWordsFusedOffY += wordsDiffering(hy, stackOf(gotOffY));
}

/// @note Checked per SHAPE rather than per sweep. A sweep-wide tally folds a
/// thousand configurations into one number, and the failure it reports then
/// names nothing a reader can reproduce; checking per width and per depth
/// costs a few hundred assertions and makes a failure say which extent
/// broke.
void checkTally(const char* what, const DerivTally& t) {
    std::printf("  %s: %zu configurations\n", what, t.configs);
    BINCV_CHECK(t.configs > 0);
    BINCV_CHECK_EQ(t.launchErrors, size_t{0});
    BINCV_CHECK_EQ(t.badWordsX, size_t{0});
    BINCV_CHECK_EQ(t.badWordsY, size_t{0});
    BINCV_CHECK_EQ(t.badWordsFusedX, size_t{0});
    BINCV_CHECK_EQ(t.badWordsFusedY, size_t{0});
    BINCV_CHECK_EQ(t.badWordsFusedOffX, size_t{0});
    BINCV_CHECK_EQ(t.badWordsFusedOffY, size_t{0});
    BINCV_CHECK_EQ(t.padding, size_t{0});
    BINCV_CHECK_EQ(t.canonical, size_t{0});
}

const BorderType kBorders[5] = {bincv::BORDER_CONSTANT, bincv::BORDER_REPLICATE,
                                bincv::BORDER_REFLECT, bincv::BORDER_WRAP,
                                bincv::BORDER_REFLECT_101};

} // namespace

// ---------------------------------------------------------------------------
// 1. The derivative's SHAPE sweep
//
// The widths are the ones that do and do not end on a word boundary, either
// side of 32, 64 and 96, plus the width == 1 degenerate the host header calls
// out (both reflects give column 0, so the derivative is identically zero). The
// heights include 1, the vertical degenerate.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivative, ShapeSweepMatchesHost) {
    const size_t widths[] = {1, 2, 3, 31, 32, 33, 63, 64, 65, 97, 200};
    const size_t heights[] = {1, 2, 3, 13, 40};
    uint64_t seed = 0xC0FFEEu;
    for (size_t w : widths) {
        DerivTally t1;
        DerivTally t2;
        for (size_t h : heights) {
            {
                QuantMat<1, uint32_t> src(static_cast<int>(w), static_cast<int>(h));
                fillQuant<1>(src, ++seed);
                runDerivativeConfig<1>(src, bincv::BORDER_REFLECT_101, false, t1);
                runDerivativeConfig<1>(src, bincv::BORDER_CONSTANT, true, t1);
            }
            {
                QuantMat<2, uint32_t> src(static_cast<int>(w), static_cast<int>(h));
                fillQuant<2>(src, ++seed);
                runDerivativeConfig<2>(src, bincv::BORDER_REFLECT_101, false, t2);
                runDerivativeConfig<2>(src, bincv::BORDER_REPLICATE, false, t2);
            }
        }
        char label[96];
        std::snprintf(label, sizeof label, "shape sweep width %zu, N = 1", w);
        checkTally(label, t1);
        std::snprintf(label, sizeof label, "shape sweep width %zu, N = 2", w);
        checkTally(label, t2);
    }
}

// ---------------------------------------------------------------------------
// 2. The derivative's BORDER sweep -- every BorderType, both constant fills,
// at a width that ends on a word boundary and one that does not.
//
// Nothing about the border rule is restated on the device: impl::borderIndex
// runs on the host and its answers arrive as kernel arguments. So what this
// sweep pins is that the LAUNCHER resolves the right four coordinates, not that
// a device copy of cv::borderInterpolate agrees with the host one -- there is
// no such copy to disagree.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivative, BorderSweepMatchesHost) {
    const size_t widths[] = {64, 97};
    uint64_t seed = 0x5EEDu;
    for (size_t w : widths) {
        for (size_t b = 0; b < 5; ++b) {
            const BorderType bt = kBorders[b];
            DerivTally t1;
            DerivTally t3;
            for (int bv = 0; bv < 2; ++bv) {
                QuantMat<1, uint32_t> s1(static_cast<int>(w), 13);
                fillQuant<1>(s1, ++seed);
                runDerivativeConfig<1>(s1, bt, bv != 0, t1);
                QuantMat<3, uint32_t> s3(static_cast<int>(w), 9);
                fillQuant<3>(s3, ++seed);
                runDerivativeConfig<3>(s3, bt, bv != 0, t3);
            }
            char label[96];
            std::snprintf(label, sizeof label, "border sweep width %zu, type %d, N = 1", w,
                          static_cast<int>(bt));
            checkTally(label, t1);
            std::snprintf(label, sizeof label, "border sweep width %zu, type %d, N = 3", w,
                          static_cast<int>(bt));
            checkTally(label, t3);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The derivative's DEPTH sweep -- every N the device domain admits.
//
// N == 1 takes the ternary three-op spelling inside impl::signedDifference and
// N > 1 takes the ripple-borrow subtract plus the conditional negate, so this
// sweep is also what checks that nvcc's device pass gives the host's answer for
// BOTH branches of the host's own route selector.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivative, DepthSweepMatchesHost) {
    DerivTally t;
    uint64_t seed = 0xD3B7u;
    QuantMat<1, uint32_t> s1(200, 40);
    fillQuant<1>(s1, ++seed);
    runDerivativeConfig<1>(s1, bincv::BORDER_REFLECT_101, false, t);
    QuantMat<2, uint32_t> s2(200, 40);
    fillQuant<2>(s2, ++seed);
    runDerivativeConfig<2>(s2, bincv::BORDER_REFLECT_101, false, t);
    QuantMat<3, uint32_t> s3(200, 40);
    fillQuant<3>(s3, ++seed);
    runDerivativeConfig<3>(s3, bincv::BORDER_REFLECT_101, false, t);
    QuantMat<4, uint32_t> s4(200, 40);
    fillQuant<4>(s4, ++seed);
    runDerivativeConfig<4>(s4, bincv::BORDER_REFLECT_101, false, t);
    checkTally("depth sweep, N = 1..4", t);
}

// ---------------------------------------------------------------------------
// 4. A REAL FRAME, both shipped ladder depths.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivative, FrameMatchesHost) {
    DerivTally t;
    uint64_t seed = 0x11FEu;
    QuantMat<1, uint32_t> s1(752, 120);
    fillQuant<1>(s1, ++seed);
    runDerivativeConfig<1>(s1, bincv::BORDER_REFLECT_101, false, t);
    QuantMat<2, uint32_t> s2(752, 120);
    fillQuant<2>(s2, ++seed);
    runDerivativeConfig<2>(s2, bincv::BORDER_REFLECT_101, false, t);
    checkTally("frame 752 x 120, N = 1 and 2", t);
}

// ---------------------------------------------------------------------------
// 5. DIRTY SOURCE PADDING, on both sides.
//
// A view of width W over an ALL-ONES buffer of width W2 > W at the same stride.
// The host header records that its source mask is dead BECAUSE the right-border
// fixup overwrites exactly the bit `cur >> 1` leaks into; this is the device
// statement of that coupling. Without it, deleting the device fixup would still
// pass every other case here.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivative, DirtySourcePaddingMatchesHost) {
    constexpr size_t N = 2;
    const size_t narrow = 97;
    const size_t wide = 128;
    const size_t h = 11;

    // Host: an all-ones wide block, then a NARROW view over it at the wide
    // stride -- so the tail word's bits past 97 are set.
    QuantMat<N, uint32_t> wideSrc(static_cast<int>(wide), static_cast<int>(h));
    for (size_t p = 0; p < N; ++p) {
        BinMatView<uint32_t> plane = wideSrc.plane(p);
        for (size_t y = 0; y < h; ++y)
            for (size_t i = 0; i < wideSrc.getAlignedWidth(); ++i) plane.row(y)[i] = 0xFFFFFFFFu;
    }
    const size_t stride = wideSrc.getAlignedWidth();
    QuantMat<N, uint32_t> narrowSrc(wideSrc.data(), static_cast<int>(narrow),
                                    static_cast<int>(h), stride);

    // Device: upload the WIDE block, then name a narrow plane block over it at
    // the same stride. Same bytes, same dirty padding.
    DeviceBinMat devWide(static_cast<int>(wide), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf(wideSrc), devWide.view()), cudaSuccess);
    const DevicePlaneBlockConstView devNarrow{devWide.constView().ptr, narrow, h, stride, N};

    DerivTally t;
    runDerivativeConfig<N>(narrowSrc, bincv::BORDER_REFLECT_101, false, t, &devNarrow);
    runDerivativeConfig<N>(narrowSrc, bincv::BORDER_CONSTANT, false, t, &devNarrow);
    checkTally("dirty source padding, N = 2", t);
}

// ---------------------------------------------------------------------------
// 6. The derivative's NAMED DEVICE DOMAIN
// ---------------------------------------------------------------------------
// Deliberate domain violations, run unchecked and reported checked by the
// shared harness's BINCV_CHECK_EQ_UNLESS_CHECKED. See its docstring for why.
#define BINCV_DC_EXPECT_REJECTED(call) \
    BINCV_CHECK_EQ_UNLESS_CHECKED(call, cudaErrorInvalidValue)

BINCV_TEST(CudaDerivative, ReportsAnErrorOutsideItsNamedDomain) {
    DeviceBinMat src(64, 8);        // N = 1
    DeviceBinMat right(64, 16);     // N + 1 = 2 planes
    DeviceBinMat tooFewPlanes(64, 8);
    DeviceBinMat wrongWidth(63, 16);
    DeviceBinMat deep(64, 5 * 8);   // N = 5, above derivativeMaxPlanes()

    BINCV_CHECK_EQ(bincv::cuda::derivativeMaxPlanes(), size_t{4});

    BINCV_DC_EXPECT_REJECTED(
        bincv::cuda::derivativeX(bincv::cuda::planeBlock(src.constView(), 1),
                                 bincv::cuda::planeBlock(tooFewPlanes.view(), 1)));
    BINCV_DC_EXPECT_REJECTED(
        bincv::cuda::derivativeY(bincv::cuda::planeBlock(src.constView(), 1),
                                 bincv::cuda::planeBlock(wrongWidth.view(), 2)));
    BINCV_DC_EXPECT_REJECTED(
        bincv::cuda::derivativeX(bincv::cuda::planeBlock(src.constView(), 1),
                                 bincv::cuda::planeBlock(right.view(), 2),
                                 static_cast<BorderType>(7)));
    // N = 5 is outside the DEVICE domain even though SignedQuantMat admits it.
    {
        DeviceBinMat deepDst(64, 6 * 8);
        BINCV_DC_EXPECT_REJECTED(
            bincv::cuda::derivativeX(bincv::cuda::planeBlock(deep.constView(), 5),
                                     bincv::cuda::planeBlock(deepDst.view(), 6)));
    }
    // IN PLACE IS NOT SUPPORTED: a destination that shares a word with the
    // source is a silent cross-block race on the device, so it is rejected.
    BINCV_DC_EXPECT_REJECTED(
        bincv::cuda::derivativeX(bincv::cuda::planeBlock(right.constView(), 1),
                                 bincv::cuda::planeBlock(right.view(), 2)));
    // ...and in the fused form the two destinations must differ from each other.
    BINCV_DC_EXPECT_REJECTED(
        bincv::cuda::derivativeXY(bincv::cuda::planeBlock(src.constView(), 1),
                                  bincv::cuda::planeBlock(right.view(), 2),
                                  bincv::cuda::planeBlock(right.view(), 2)));

    BINCV_CHECK_EQ(bincv::cuda::derivativeX(bincv::cuda::planeBlock(src.constView(), 1),
                                            bincv::cuda::planeBlock(right.view(), 2)),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// THE COVARIANCE
// ---------------------------------------------------------------------------
namespace {

/// The rect set: whole frame, interior, negative origins, wholly outside, zero
/// extents, single column, single row, edge-touching, the tracker's 31x31 at
/// several positions, and two windows WIDER than 32 pixels -- which is the case
/// the aligned-run arm's own gate excludes.
const Rect kRects[] = {
    Rect{0, 0, 200, 90},    // the whole frame
    Rect{10, 10, 31, 31},   // the tracker's window, interior
    Rect{0, 0, 31, 31},     // touching the top-left corner
    Rect{169, 59, 31, 31},  // touching the bottom-right corner
    Rect{-5, -5, 31, 31},   // a negative origin -- legal, clipped
    Rect{185, 75, 31, 31},  // running off the right and bottom edges
    Rect{-40, 10, 20, 20},  // wholly outside to the left
    Rect{400, 10, 20, 20},  // wholly outside to the right
    Rect{10, 200, 20, 20},  // wholly outside below
    Rect{10, 10, 0, 20},    // zero width
    Rect{10, 10, 20, 0},    // zero height
    Rect{10, 10, -4, 20},   // negative width
    Rect{31, 10, 1, 40},    // a single column, mid-word
    Rect{32, 10, 1, 40},    // a single column, on a word boundary
    Rect{5, 20, 60, 1},     // a single row
    Rect{7, 3, 32, 32},     // exactly 32 wide, mid-word -- the funnel arm's edge
    Rect{0, 3, 32, 32},     // exactly 32 wide, aligned
    Rect{5, 5, 33, 40},     // 33 wide: the aligned-run gate EXCLUDES this
    Rect{5, 5, 70, 40},     // comfortably wider than one word
};
constexpr size_t kRectCount = sizeof(kRects) / sizeof(kRects[0]);

/// A derivative pair on both sides, from one random N-bit level -- the input a
/// covariance caller actually holds.
template <size_t N>
struct DerivPair {
    SignedQuantMat<N, uint32_t> hostDx;
    SignedQuantMat<N, uint32_t> hostDy;
    DeviceBinMat devDx;
    DeviceBinMat devDy;

    DerivPair(size_t w, size_t h, uint64_t seed)
        : hostDx(static_cast<int>(w), static_cast<int>(h)),
          hostDy(static_cast<int>(w), static_cast<int>(h)),
          devDx(static_cast<int>(w), static_cast<int>((N + 1) * h)),
          devDy(static_cast<int>(w), static_cast<int>((N + 1) * h)) {
        QuantMat<N, uint32_t> src(static_cast<int>(w), static_cast<int>(h));
        fillQuant<N>(src, seed);
        bincv::derivativeX<N, uint32_t>(src, hostDx);
        bincv::derivativeY<N, uint32_t>(src, hostDy);
        // The device side gets the HOST's derivative uploaded rather than
        // recomputed, so a covariance failure here cannot be a derivative
        // failure wearing a covariance's name.
        bincv::cuda::upload(stackOf<N + 1>(hostDx.planes()), devDx.view());
        bincv::cuda::upload(stackOf<N + 1>(hostDy.planes()), devDy.view());
        cudaDeviceSynchronize();
    }

    DevicePlaneBlockConstView dx() const {
        return bincv::cuda::planeBlock(devDx.constView(), N + 1);
    }
    DevicePlaneBlockConstView dy() const {
        return bincv::cuda::planeBlock(devDy.constView(), N + 1);
    }
    GradientCovariance host(Rect r) const {
        return bincv::gradientCovariance<N, uint32_t>(hostDx, hostDy, r);
    }
};

/// Runs the whole rect set through the batch, once, and hands back the results.
std::vector<DeviceGradientCovariance> runBatch(DevicePlaneBlockConstView dx,
                                               DevicePlaneBlockConstView dy,
                                               const Rect* rects, size_t count,
                                               size_t& errors) {
    bincv::cuda::DeviceArray<Rect> dRects(count);
    bincv::cuda::DeviceArray<DeviceGradientCovariance> dOut(count);
    if (cudaMemcpy(dRects.data(), rects, count * sizeof(Rect), cudaMemcpyHostToDevice) !=
        cudaSuccess)
        ++errors;
    if (bincv::cuda::gradientCovarianceBatchAsync(dx, dy, dRects.data(), count,
                                                  dOut.data()) != cudaSuccess)
        ++errors;
    std::vector<DeviceGradientCovariance> out(count);
    if (cudaMemcpy(out.data(), dOut.data(), count * sizeof(DeviceGradientCovariance),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        ++errors;
    return out;
}

bool sameCovariance(const DeviceGradientCovariance& d, const GradientCovariance& h) {
    return d.sumXX == h.sumXX && d.sumYY == h.sumYY && d.sumXY == h.sumXY;
}

/// The whole four-way comparison at one N: every arm against the host and
/// against each other, over the whole rect set.
template <size_t N>
void checkCovarianceAt(size_t w, size_t h, uint64_t seed) {
    DerivPair<N> p(w, h, seed);
    size_t errors = 0;

    bincv::cuda::impl::covarianceAlignedRunEnabled() = true;
    const auto aligned = runBatch(p.dx(), p.dy(), kRects, kRectCount, errors);
    bincv::cuda::impl::covarianceAlignedRunEnabled() = false;
    const auto perWord = runBatch(p.dx(), p.dy(), kRects, kRectCount, errors);
    bincv::cuda::impl::covarianceAlignedRunEnabled() = true;

    std::printf("  covariance N = %zu at %zu x %zu: %zu rects, both arms\n", N, w, h,
                kRectCount);
    BINCV_CHECK_EQ(errors, size_t{0});
    // PER RECT, not per sweep: the rect set is chosen so that each entry is a
    // different claim -- a negative origin, a wholly-outside window, a run
    // exactly 32 wide, a run the aligned arm's gate excludes -- and a single
    // folded counter would report "one of nineteen" and name none of them.
    for (size_t i = 0; i < kRectCount; ++i) {
        const GradientCovariance expected = p.host(kRects[i]);
        BINCV_CHECK(sameCovariance(aligned[i], expected));
        // Integer addition over the same masked words: the two arms cannot
        // disagree, so any difference is a bug and not a reordering. They do
        // NOT divide the same unit across the block -- aligned, a thread takes
        // a row; unaligned, a (row, word) pair -- so this is a real comparison
        // and not two names for one traversal.
        BINCV_CHECK(sameCovariance(perWord[i], expected));
        // ...and the single-region form, whose traversal is different again,
        // is the batch's oracle.
        BINCV_CHECK(sameCovariance(
            aligned[i], bincv::cuda::gradientCovariance(p.dx(), p.dy(), kRects[i])));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// 7. The covariance against the HOST N-bit kernel, and all three arms against
// each other and against the single-region form, at every supported N.
//
// The frame is 200 x 90: a width that is NOT a multiple of 32, so every rect in
// the set exercises head and tail masks.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, MatchesHostAtEveryDepth) {
    checkCovarianceAt<1>(200, 90, 0xA1u);
    checkCovarianceAt<2>(200, 90, 0xB2u);
    checkCovarianceAt<3>(200, 90, 0xC3u);
    checkCovarianceAt<4>(200, 90, 0xD4u);
}

// ---------------------------------------------------------------------------
// 8. N == 1 THREE WAYS.
//
// The device N-bit kernel runs its GENERIC loop at N = 1 -- it does not
// delegate to a ternary special case, exactly as the host refuses to -- so all
// three of these must be the same three integers:
//
//   * the host's FIVE-ARGUMENT ternary gradientCovariance,
//   * this file's device N-bit kernel at N = 1,
//   * this backend's existing countCovarianceBatchAsync, recombined through
//     CovarianceCount::crossTerm.
//
// Delegating at N = 1 would make the first two vacuous, which is why it does
// not; the third is what keeps the two DEVICE routes from disagreeing.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, TernaryRouteAgreesThreeWays) {
    DerivPair<1> p(200, 90, 0xE5u);
    size_t errors = 0;
    const auto got = runBatch(p.dx(), p.dy(), kRects, kRectCount, errors);

    // The existing device ternary batch, on the same four planes.
    bincv::cuda::DeviceArray<Rect> dRects(kRectCount);
    bincv::cuda::DeviceArray<bincv::cuda::DeviceCovarianceCount> dCounts(kRectCount);
    BINCV_CHECK_EQ(cudaMemcpy(dRects.data(), kRects, kRectCount * sizeof(Rect),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);
    const DevicePlaneBlockConstView dxb = p.dx();
    const DevicePlaneBlockConstView dyb = p.dy();
    BINCV_CHECK_EQ(bincv::cuda::countCovarianceBatchAsync(
                       dxb.plane(0), dyb.plane(0), dxb.plane(1), dyb.plane(1),
                       dRects.data(), kRectCount, dCounts.data()),
                   cudaSuccess);
    std::vector<bincv::cuda::DeviceCovarianceCount> counts(kRectCount);
    BINCV_CHECK_EQ(cudaMemcpy(counts.data(), dCounts.data(),
                              kRectCount * sizeof(bincv::cuda::DeviceCovarianceCount),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);

    BINCV_CHECK_EQ(errors, size_t{0});
    for (size_t i = 0; i < kRectCount; ++i) {
        // The host's FIVE-ARGUMENT ternary overload, not the N-bit one.
        BINCV_CHECK(sameCovariance(
            got[i], bincv::gradientCovariance<uint32_t>(
                        p.hostDx.constMagnitude(0), p.hostDy.constMagnitude(0),
                        p.hostDx.constSign(), p.hostDy.constSign(), kRects[i])));

        const bincv::CovarianceCount c = bincv::cuda::toHost(counts[i]);
        GradientCovariance recombined;
        recombined.sumXX = static_cast<int64_t>(c.xx);
        recombined.sumYY = static_cast<int64_t>(c.yy);
        recombined.sumXY = static_cast<int64_t>(c.crossTerm());
        BINCV_CHECK(sameCovariance(got[i], recombined));
    }
}

// ---------------------------------------------------------------------------
// 9. A BIT AT OR PAST `width` IS NEVER COUNTED, whatever it holds.
//
// Narrow views over wide all-ones derivative blocks, on both sides -- the
// covariance's statement of the same padding contract the derivative's dirty
// case makes. An arm that dropped a head or tail mask would pass every rect in
// the interior and fail here.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, DirtyPaddingIsNeverCounted) {
    constexpr size_t N = 2;
    const size_t narrow = 97;
    const size_t wide = 128;
    const size_t h = 40;

    QuantMat<N + 1, uint32_t> wx(static_cast<int>(wide), static_cast<int>(h));
    QuantMat<N + 1, uint32_t> wy(static_cast<int>(wide), static_cast<int>(h));
    uint64_t seed = 0xF00Du;
    for (size_t p = 0; p < N + 1; ++p) {
        BinMatView<uint32_t> px = wx.plane(p);
        BinMatView<uint32_t> py = wy.plane(p);
        for (size_t y = 0; y < h; ++y) {
            for (size_t i = 0; i < wx.getAlignedWidth(); ++i) {
                px.row(y)[i] = static_cast<uint32_t>(splitmix(seed));
                py.row(y)[i] = static_cast<uint32_t>(splitmix(seed));
            }
        }
    }
    const size_t stride = wx.getAlignedWidth();

    DeviceBinMat dwx(static_cast<int>(wide), static_cast<int>((N + 1) * h));
    DeviceBinMat dwy(static_cast<int>(wide), static_cast<int>((N + 1) * h));
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf<N + 1>(wx), dwx.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf<N + 1>(wy), dwy.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const DevicePlaneBlockConstView ndx{dwx.constView().ptr, narrow, h, stride, N + 1};
    const DevicePlaneBlockConstView ndy{dwy.constView().ptr, narrow, h, stride, N + 1};

    BinMatConstView<uint32_t> hmx[N];
    BinMatConstView<uint32_t> hmy[N];
    for (size_t p = 0; p < N; ++p) {
        hmx[p] = BinMatConstView<uint32_t>(wx.data() + p * h * stride, narrow, h, stride);
        hmy[p] = BinMatConstView<uint32_t>(wy.data() + p * h * stride, narrow, h, stride);
    }
    const BinMatConstView<uint32_t> hsx(wx.data() + N * h * stride, narrow, h, stride);
    const BinMatConstView<uint32_t> hsy(wy.data() + N * h * stride, narrow, h, stride);

    size_t errors = 0;
    const auto got = runBatch(ndx, ndy, kRects, kRectCount, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    for (size_t i = 0; i < kRectCount; ++i)
        BINCV_CHECK(sameCovariance(
            got[i], bincv::gradientCovariance<N, uint32_t>(hmx, hmy, hsx, hsy, kRects[i])));
}

// ---------------------------------------------------------------------------
// 10. THE SIGN PLANES ARE READ ONLY WHERE BOTH MAGNITUDES ARE SET.
//
// `m_x[i] & m_y[j]` can only be set where both magnitudes are non-zero, so the
// canonical-zero rule stays irrelevant to the answer rather than becoming a
// precondition on the caller -- the host's promise 5. Dirtying both sign planes
// wherever the magnitudes are clear must not move any of the three numbers.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, DirtySignPlanesCannotMoveTheAnswer) {
    constexpr size_t N = 2;
    DerivPair<N> p(200, 90, 0x1234u);
    size_t errors = 0;
    const auto clean = runBatch(p.dx(), p.dy(), kRects, kRectCount, errors);

    // Set every sign bit whose pixel has magnitude zero, on the host copy, and
    // re-upload. That is a "negative zero" everywhere the derivative wrote a
    // canonical one -- legal input to this operation, by promise 5.
    const auto dirty = [&](SignedQuantMat<N, uint32_t>& m) {
        const size_t words = bincv::impl::minRowWords<uint32_t>(m.getWidth());
        for (size_t y = 0; y < m.getHeight(); ++y) {
            for (size_t i = 0; i < words; ++i) {
                uint32_t magOr = 0u;
                for (size_t q = 0; q < N; ++q) magOr |= m.magnitude(q).row(y)[i];
                m.sign().row(y)[i] |= ~magOr;
            }
        }
    };
    dirty(p.hostDx);
    dirty(p.hostDy);
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf<N + 1>(p.hostDx.planes()), p.devDx.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf<N + 1>(p.hostDy.planes()), p.devDy.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto dirtied = runBatch(p.dx(), p.dy(), kRects, kRectCount, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    for (size_t i = 0; i < kRectCount; ++i) {
        BINCV_CHECK_EQ(dirtied[i].sumXX, clean[i].sumXX);
        BINCV_CHECK_EQ(dirtied[i].sumYY, clean[i].sumYY);
        BINCV_CHECK_EQ(dirtied[i].sumXY, clean[i].sumXY);
    }
}

// ---------------------------------------------------------------------------
// 11. DEGENERATE BATCH SIZES -- a tracker's first and last frames.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, DegenerateBatchSizes) {
    DerivPair<1> p(200, 90, 0x99u);
    // count == 0 writes nothing and is not an error.
    BINCV_CHECK_EQ(bincv::cuda::gradientCovarianceBatchAsync(p.dx(), p.dy(), nullptr, 0,
                                                             nullptr),
                   cudaSuccess);
    size_t errors = 0;
    const Rect one[] = {Rect{10, 10, 31, 31}};
    const auto got = runBatch(p.dx(), p.dy(), one, 1, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    BINCV_CHECK(sameCovariance(got[0], p.host(one[0])));
    // A wholly-outside window is a VALUE, not an error.
    const Rect outside[] = {Rect{9000, 9000, 31, 31}};
    const auto none = runBatch(p.dx(), p.dy(), outside, 1, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    BINCV_CHECK(none[0].sumXX == 0 && none[0].sumYY == 0 && none[0].sumXY == 0);
}

// ---------------------------------------------------------------------------
// 12. READ-ONLY, SO ALIASING IS UNRESTRICTED (the host's promise 4).
//
// gradientCovariance(dx, dx, w) is the SumIx^2 case, and its cross term equals
// SumIx^2 because a plane never disagrees in sign with itself.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, SelfPairIsWellDefined) {
    DerivPair<2> p(200, 90, 0x77u);
    size_t errors = 0;
    const auto got = runBatch(p.dx(), p.dx(), kRects, kRectCount, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    for (size_t i = 0; i < kRectCount; ++i) {
        BINCV_CHECK(sameCovariance(
            got[i], bincv::gradientCovariance<2, uint32_t>(p.hostDx, p.hostDx, kRects[i])));
        BINCV_CHECK_EQ(got[i].sumYY, got[i].sumXX);
        BINCV_CHECK_EQ(got[i].sumXY, got[i].sumXX);
    }
}

// ---------------------------------------------------------------------------
// 13. The covariance's NAMED DEVICE DOMAIN
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCovariance, ReportsAnErrorOutsideItsNamedDomain) {
    BINCV_CHECK_EQ(bincv::cuda::covarianceMaxPlanes(), size_t{4});
    DeviceBinMat good(64, 2 * 8);    // N = 1: 2 planes
    DeviceBinMat mismatch(63, 2 * 8);
    DeviceBinMat deep(64, 6 * 8);    // N = 5, above covarianceMaxPlanes()
    bincv::cuda::DeviceArray<Rect> dRects(1);
    bincv::cuda::DeviceArray<DeviceGradientCovariance> dOut(1);
    const Rect r{0, 0, 16, 4};
    BINCV_CHECK_EQ(cudaMemcpy(dRects.data(), &r, sizeof r, cudaMemcpyHostToDevice),
                   cudaSuccess);

    BINCV_DC_EXPECT_REJECTED(bincv::cuda::gradientCovarianceBatchAsync(
        bincv::cuda::planeBlock(good.constView(), 2),
        bincv::cuda::planeBlock(mismatch.constView(), 2), dRects.data(), 1, dOut.data()));
    BINCV_DC_EXPECT_REJECTED(bincv::cuda::gradientCovarianceBatchAsync(
        bincv::cuda::planeBlock(deep.constView(), 6),
        bincv::cuda::planeBlock(deep.constView(), 6), dRects.data(), 1, dOut.data()));
    // One plane is a magnitude stack with no sign plane -- not a signed block.
    BINCV_DC_EXPECT_REJECTED(bincv::cuda::gradientCovarianceBatchAsync(
        bincv::cuda::planeBlock(good.constView(), 1),
        bincv::cuda::planeBlock(good.constView(), 1), dRects.data(), 1, dOut.data()));

    BINCV_CHECK_EQ(bincv::cuda::gradientCovarianceBatchAsync(
                       bincv::cuda::planeBlock(good.constView(), 2),
                       bincv::cuda::planeBlock(good.constView(), 2), dRects.data(), 1,
                       dOut.data()),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// 14. THE WHOLE CHAIN, DEVICE-RESIDENT: derivativeXY then the batched
// covariance, with nothing crossing the bus in between except the 24 bytes per
// window. This is the shape a resident tracker runs, and it is what makes the
// two halves of this family one family.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDerivCov, ResidentChainMatchesHost) {
    constexpr size_t N = 2;
    const size_t w = 200, h = 90;
    QuantMat<N, uint32_t> src(static_cast<int>(w), static_cast<int>(h));
    fillQuant<N>(src, 0xBEEFu);

    SignedQuantMat<N, uint32_t> hostDx(static_cast<int>(w), static_cast<int>(h));
    SignedQuantMat<N, uint32_t> hostDy(static_cast<int>(w), static_cast<int>(h));
    bincv::derivativeX<N, uint32_t>(src, hostDx);
    bincv::derivativeY<N, uint32_t>(src, hostDy);

    DeviceBinMat devSrc(static_cast<int>(w), static_cast<int>(N * h));
    DeviceBinMat devDx(static_cast<int>(w), static_cast<int>((N + 1) * h));
    DeviceBinMat devDy(static_cast<int>(w), static_cast<int>((N + 1) * h));
    BINCV_CHECK_EQ(bincv::cuda::upload(stackOf(src), devSrc.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::derivativeXY(
                       bincv::cuda::planeBlock(devSrc.constView(), N),
                       bincv::cuda::planeBlock(devDx.view(), N + 1),
                       bincv::cuda::planeBlock(devDy.view(), N + 1)),
                   cudaSuccess);

    size_t errors = 0;
    const auto got = runBatch(bincv::cuda::planeBlock(devDx.constView(), N + 1),
                              bincv::cuda::planeBlock(devDy.constView(), N + 1), kRects,
                              kRectCount, errors);
    BINCV_CHECK_EQ(errors, size_t{0});
    for (size_t i = 0; i < kRectCount; ++i)
        BINCV_CHECK(sameCovariance(
            got[i], bincv::gradientCovariance<N, uint32_t>(hostDx, hostDy, kRects[i])));
}

namespace {
bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return false;
    }
    return true;
}
} // namespace

#if BINCV_TEST_WITH_GTEST
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    const int summaryRc = ::bincv::test::summarize("CUDA derivative and covariance tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA derivative and covariance tests", argc, argv);
}
#endif
