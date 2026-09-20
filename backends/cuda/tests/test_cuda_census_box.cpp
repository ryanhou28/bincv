// The packed census matcher's WARP-BOX arm against the host library, and
// against the kernel it replaces, in one binary.
//
// WHY THIS SUITE EXISTS SEPARATELY from the packed case already in
// test_cuda_backend.cpp: that case runs ONE arm. `denseDisparityCensusPacked`
// now has two kernels behind it, and the project's rule for a fast arm is that
// both answer to the same map in one binary through the runtime switch. Every
// shape below is therefore run twice -- `densePackedBoxEnabled()` true and
// false -- and compared to the host map AND to each other.
//
// THE SHAPES ARE NOT A GENERIC SWEEP. Each one is a structural edge THIS
// kernel introduces, and the comment on each says which:
//
//   * a warp owns `33 - winWidth` anchor columns, so an anchor count that is
//     not a multiple of that span leaves a partial last warp;
//   * fewer anchors than one span means the whole grid is one partial warp;
//   * the horizontal aggregation is a binary decomposition of `winWidth`, so
//     every distinct shape of that decomposition's bit pattern is exercised;
//   * the vertical slide walks a strip, so an output-row count that is not a
//     multiple of the strip exercises the tail;
//   * lanes whose column has run off the row, and lanes whose column is left
//     of the disparity, take predicated loads that must contribute zero.
//
// TWO THINGS A MAP COMPARISON CANNOT CATCH, stated because neither tool that
// would catch them runs on this machine: `compute-sanitizer --tool synccheck`
// (divergent shuffle participation) and `--tool memcheck` (the predicated
// out-of-row read) are both unavailable here. What stands in for them is
// structural: every early exit in the kernel is warp-uniform by construction --
// it depends only on blockIdx, threadIdx.y and kernel arguments -- and the
// shuffle deltas are template parameters, so no lane can reach a shuffle its
// neighbours do not. The narrow-width and minDisparity > 0 cases below are what
// make the predicated-load lanes exist at all.
//
// Exits 77 when no CUDA device is present, so a GPU-less configure still builds
// and reports honestly.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/cuda/census.hpp"
#include "bincv/cuda/denseCensusBox.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/census.hpp"
#include "bincv/ops/denseDisparity.hpp"
#include "test_util.hpp"

namespace {

constexpr size_t kK = 24;  // census 5x5, the reference descriptor

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// A rectified pair with real structure: the right image is the left shifted by
/// a known disparity, so the map has content rather than noise.
std::vector<uint8_t> smoothFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = static_cast<uint8_t>(splitmix(seed) >> 40);
    std::vector<uint8_t> tmp(w * h, 0);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            tmp[y * w + x] = static_cast<uint8_t>(
                (img[y * w + x - 1] + 2u * img[y * w + x] + img[y * w + x + 1]) / 4u);
    for (size_t y = 1; y + 1 < h; ++y)
        for (size_t x = 1; x + 1 < w; ++x)
            img[y * w + x] = static_cast<uint8_t>(
                (tmp[(y - 1) * w + x] + 2u * tmp[y * w + x] + tmp[(y + 1) * w + x]) / 4u);
    return img;
}

/// @brief Runs one shape through the host wide path and BOTH device arms, and
/// holds all three to the same bytes -- on TWO kinds of content.
///
/// WHY TWO, and this is the load-bearing part of the suite: on a pair where the
/// right image is an exact shift of the left, the correct disparity's window
/// cost is ZERO and every other candidate's is hundreds, so the winner-take-all
/// is insensitive to arithmetic errors of a few hundred. A deliberately broken
/// halo -- `33 - winWidth` output lanes widened to `34 - winWidth`, which makes
/// the top lane of every warp aggregate one column twice instead of reaching a
/// column that is not in its warp -- produced a BYTE-IDENTICAL map on the
/// shifted pair across every shape here. It was caught only by running the same
/// shapes on an UNCORRELATED pair, where every candidate's cost sits within a
/// few counts of every other and the argmin is decided by the exact sum. A
/// suite that only ever sees a clean shift is testing the disparity search, not
/// the window arithmetic.
///
/// @param why Printed on a mismatch, because a shape list whose failures say
/// only "index 4" costs the next reader the derivation again.
void checkShapeOn(size_t w, size_t h, const bincv::DenseDisparityParams& p,
                  const std::vector<uint8_t>& lw, const std::vector<uint8_t>& rw,
                  const char* why, const char* content);

void checkShape(size_t w, size_t h, const bincv::DenseDisparityParams& p, size_t shift,
                uint64_t seed, const char* why) {
    const auto lw = smoothFrame(w, h, seed);
    std::vector<uint8_t> shifted(w * h, 0);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x + shift < w; ++x) shifted[y * w + x] = lw[y * w + x + shift];
    checkShapeOn(w, h, p, lw, shifted, why, "right = left shifted");
    // An INDEPENDENT right frame: no true match anywhere, so the costs across
    // candidates are near-ties and the map is a direct readout of the window
    // sums and of the smallest-disparity tie rule.
    checkShapeOn(w, h, p, lw, smoothFrame(w, h, seed ^ 0x5DEECE66DULL), why,
                 "right uncorrelated");
}

void checkShapeOn(size_t w, size_t h, const bincv::DenseDisparityParams& p,
                  const std::vector<uint8_t>& lw, const std::vector<uint8_t>& rw,
                  const char* why, const char* content) {
    // The host wide path is the truth.
    std::vector<uint32_t> sw(bincv::denseDisparityScratchWords<kK, uint32_t>(w, p));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(w));
    std::vector<uint8_t> expect(w * h, 0xAA);
    bincv::denseDisparity<kK, uint8_t, uint32_t>(lw.data(), rw.data(), w, h, w, w,
                                                 bincv::kCensus5x5, p, sw.data(),
                                                 sw.size(), sr.data(), sr.size(),
                                                 expect.data(), w);

    bincv::cuda::DeviceImage<uint8_t> dL(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint8_t> dR(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(lw.data(), w, h, w, dL.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(rw.data(), w, h, w, dR.view()),
                   cudaSuccess);
    bincv::cuda::DeviceImage<uint32_t> descL(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint32_t> descR(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::censusTransformPacked<kK>(dL.constView(),
                                                          bincv::kCensus5x5,
                                                          descL.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::censusTransformPacked<kK>(dR.constView(),
                                                          bincv::kCensus5x5,
                                                          descR.view()),
                   cudaSuccess);

    bincv::cuda::DeviceImage<uint8_t> dDisp(static_cast<int>(w), static_cast<int>(h));
    std::vector<uint8_t> arm[2];
    for (int a = 0; a < 2; ++a) {
        bincv::cuda::impl::densePackedBoxEnabled() = (a == 0);
        BINCV_CHECK_EQ(bincv::cuda::denseDisparityCensusPacked(descL.constView(),
                                                               descR.constView(), p,
                                                               dDisp.view()),
                       cudaSuccess);
        arm[a].assign(w * h, 0x55);
        BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(),
                                                           arm[a].data(), w),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        size_t bad = 0;
        for (size_t i = 0; i < w * h; ++i)
            if (expect[i] != arm[a][i]) ++bad;
        if (bad != 0)
            std::printf("    shape %zux%zu win %dx%d d[%d,%d] (%s / %s): arm %s\n", w, h,
                        p.winWidth, p.winHeight, p.minDisparity, p.maxDisparity, why,
                        content, a == 0 ? "BOX" : "reference");
        BINCV_CHECK_EQ(bad, 0u);
    }
    bincv::cuda::impl::densePackedBoxEnabled() = true;
    // The two arms against each other, which is the switchable-arm rule's own
    // statement and is not implied by both matching the host on THIS frame if
    // one of them were to depend on something the frame does not vary.
    BINCV_CHECK(arm[0] == arm[1]);

    // THE RIM INVARIANT. This op has no padding BITS -- its inputs are wide
    // uint32 descriptor images and its output is a byte map, so there is no
    // word tail past `width` to keep zero. What the same rule becomes here is
    // that every pixel no candidate can serve reads exactly the invalid
    // marker, and that no pixel is left unwritten: the rim is the only place a
    // kernel that writes past its anchors would show up.
    const size_t hw = static_cast<size_t>(p.winWidth / 2);
    const size_t hh = static_cast<size_t>(p.winHeight / 2);
    size_t rimBad = 0;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            const bool inside = y >= hh && y + hh < h && x >= hw && x + hw < w;
            if (!inside && arm[0][y * w + x] != bincv::kDenseDisparityInvalid) ++rimBad;
        }
    BINCV_CHECK_EQ(rimBad, 0u);
}

} // namespace

// ---------------------------------------------------------------------------
// The warp span: a warp produces 33 - winWidth anchors, so the anchor count's
// remainder against that span is where an off-by-one in grid.x or in the output
// gate lands. At winWidth 9 the span is 24.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, PartialAndExactWarpSpans) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    // 131 -> 123 anchors = 5*24 + 3; 157 -> 149 = 6*24 + 5; 200 -> 192 = 8*24.
    checkShape(131, 47, p, 9, 0xACE1, "anchor span leaves 3 of a warp");
    checkShape(157, 41, p, 9, 0xBD22, "anchor span leaves 5 of a warp");
    checkShape(200, 37, p, 9, 0xCE33, "anchor span divides exactly -- the control");
}

// ---------------------------------------------------------------------------
// Fewer anchors than ONE warp span: the whole grid is a single partial warp, so
// a kernel that assumes a full span has nowhere to hide. 30 - 9 + 1 = 22 < 24.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, FewerAnchorsThanOneWarpSpan) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 12;
    checkShape(30, 29, p, 5, 0xD144, "22 anchors, one partial warp for the whole frame");
    checkShape(11, 19, p, 3, 0xD255, "3 anchors at winWidth 9 -- the narrowest that runs");
}

// ---------------------------------------------------------------------------
// Widths that are not a multiple of 32, independently of the span arithmetic:
// the halo lanes whose column has run off the row are what these exercise.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, WidthsNotAMultipleOf32) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 24;
    checkShape(97, 33, p, 7, 0xE166, "width 97");
    checkShape(131, 35, p, 7, 0xE277, "width 131");
    checkShape(157, 31, p, 7, 0xE388, "width 157");
    checkShape(128, 33, p, 7, 0xE499, "width 128 -- a multiple of 32, the control");
}

// ---------------------------------------------------------------------------
// The horizontal aggregation is a binary decomposition of winWidth, so the
// SHAPE of that decomposition changes with the width's bit pattern: a single
// trailing one (3), alternating (5), all ones (7, 15), sparse (9, 17). 19 is
// outside the arm's gate and must fall back to the reference kernel, which is
// what makes the benchmark's gate-exclusion line meaningful.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, EveryShapeOfTheWidthDecomposition) {
    for (const int winW : {3, 5, 7, 9, 11, 13, 15, 17, 19}) {
        bincv::DenseDisparityParams p;
        p.maxDisparity = 20;
        p.winWidth = winW;
        p.winHeight = 5;
        checkShape(101, 29, p, 7, 0xF100u + static_cast<uint64_t>(winW),
                   "one winWidth of the decomposition sweep");
    }
}

// ---------------------------------------------------------------------------
// The vertical slide walks a strip of output rows; an output-row count that is
// not a multiple of it is where `rowsHere` and the write loop disagree if
// either is wrong. Several heights, so the tail is every possible remainder.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, StripTails) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 16;
    for (const size_t h : {11u, 12u, 13u, 14u, 15u, 47u})
        checkShape(83, h, p, 6, 0x9100u + h, "output rows against the strip");
}

// ---------------------------------------------------------------------------
// Window HEIGHTS: the seed/leave/enter indices are the half of the shape shared
// with the arm being replaced, and the only part of it that winHeight moves.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, WindowHeights) {
    for (const int winH : {3, 5, 7, 9, 15}) {
        bincv::DenseDisparityParams p;
        p.maxDisparity = 18;
        p.winWidth = 9;
        p.winHeight = winH;
        checkShape(89, 41, p, 8, 0xA100u + static_cast<uint64_t>(winH),
                   "window height sweep");
    }
}

// ---------------------------------------------------------------------------
// minDisparity > 0 and a disparity range clamped by the width. Together these
// are what make the predicated loads exist: a lane whose column is left of the
// disparity would read BEFORE its row -- before the allocation on row 0 -- so
// its load is predicated off and contributes zero.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, DisparityRangeEdges) {
    {
        bincv::DenseDisparityParams p;
        p.minDisparity = 5;
        p.maxDisparity = 24;
        checkShape(120, 33, p, 11, 0xB111, "minDisparity > 0");
    }
    {
        // maxDisparity far past what the width can support: dEnd clamps to
        // width - winWidth, the host's own clamp.
        bincv::DenseDisparityParams p;
        p.maxDisparity = 200;
        checkShape(48, 27, p, 7, 0xB222, "dEnd clamped by the width");
    }
    {
        bincv::DenseDisparityParams p;
        p.minDisparity = 0;
        p.maxDisparity = 0;
        checkShape(77, 25, p, 4, 0xB333, "a single candidate disparity");
    }
}

// ---------------------------------------------------------------------------
// The early-outs, both taken before any launch: no output row at all, and a
// window wider than the image. The whole map must be the invalid marker on
// both arms, which is also the host's answer.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, DegenerateShapesTakeTheEarlyOut) {
    {
        bincv::DenseDisparityParams p;
        p.maxDisparity = 8;
        p.winWidth = 9;
        p.winHeight = 9;
        checkShape(64, 7, p, 3, 0xC111, "height below the window");
    }
    {
        bincv::DenseDisparityParams p;
        p.maxDisparity = 8;
        p.winWidth = 9;
        p.winHeight = 3;
        checkShape(8, 21, p, 2, 0xC222, "window wider than the image");
    }
}

// ---------------------------------------------------------------------------
// The GATE ITSELF, as a function rather than through a kernel. The two
// conditions are documented in denseCensusBox.hpp and both are derived, so both
// are pinned here: the width bound comes from `33 - winWidth` output lanes per
// warp, and the cost bound comes from folding `(cost << 8) | disparity` into
// one 32-bit register -- stated in general, `winWidth * winHeight * 32`, not as
// the reference frame's 9*9*24 = 1944.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, TheGateAcceptsExactlyWhatItDocuments) {
    bincv::DenseDisparityParams p;
    for (const int winW : {3, 5, 7, 9, 11, 13, 15, 17}) {
        p.winWidth = winW;
        p.winHeight = 9;
        BINCV_CHECK(bincv::cuda::impl::densePackedBoxAccepts(p));
    }
    for (const int winW : {1, 19, 21, 33}) {
        p.winWidth = winW;
        p.winHeight = 9;
        BINCV_CHECK(!bincv::cuda::impl::densePackedBoxAccepts(p));
    }
    // Even widths never reach here from the public op, which asserts oddness --
    // but the gate is what the launcher trusts, so it states the condition
    // itself rather than inheriting it.
    p.winWidth = 8;
    p.winHeight = 9;
    BINCV_CHECK(!bincv::cuda::impl::densePackedBoxAccepts(p));

    // The packing bound, at the only place it can bite: a window tall enough
    // that winWidth * winHeight * 32 leaves 24 bits.
    p.winWidth = 17;
    p.winHeight = 30841;  // 17 * 30841 * 32 = 16,777,504 > 0xFFFFFE
    BINCV_CHECK(!bincv::cuda::impl::densePackedBoxAccepts(p));
    p.winHeight = 30839;  // 17 * 30839 * 32 = 16,776,416 <= 0xFFFFFE
    BINCV_CHECK(bincv::cuda::impl::densePackedBoxAccepts(p));
}

// ---------------------------------------------------------------------------
// A caller that skips the gate gets an ERROR, not a wrong map. The launcher is
// internal and its only caller checks the gate first, so this is the standing
// proof that the guard exists at all -- the alternative is a ninth
// instantiation aggregating the wrong number of columns and returning a
// fully-formed, wrong answer.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, TheLauncherRefusesWhatTheGateRejects) {
    const size_t w = 64, h = 32;
    bincv::cuda::DeviceImage<uint32_t> descL(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint32_t> descR(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint8_t> dDisp(static_cast<int>(w), static_cast<int>(h));
    bincv::DenseDisparityParams p;
    p.winWidth = 19;
    p.winHeight = 9;
    p.maxDisparity = 8;
    BINCV_CHECK_EQ(bincv::cuda::impl::launchDensePackedWarpBox(
                       descL.constView(), descR.constView(), 0, 8, p, dDisp.view(),
                       h - 8, nullptr),
                   cudaErrorInvalidValue);
    // And a width the gate accepts but the IMAGE does not: winWidth > width.
    p.winWidth = 17;
    bincv::cuda::DeviceImage<uint32_t> narrowL(8, 16);
    bincv::cuda::DeviceImage<uint32_t> narrowR(8, 16);
    bincv::cuda::DeviceImage<uint8_t> narrowD(8, 16);
    BINCV_CHECK_EQ(bincv::cuda::impl::launchDensePackedWarpBox(
                       narrowL.constView(), narrowR.constView(), 0, 4, p,
                       narrowD.view(), 8, nullptr),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

// ---------------------------------------------------------------------------
// THE TIE RULE, on content that actually produces ties.
//
// A near-tie is not a tie. On smoothed content a 9x9x24 window cost lands
// somewhere in [0, 1944] and two candidates almost never land on the SAME
// integer, so "ties keep the smallest disparity" is a rule the rest of this
// suite never reaches: inverting the kernel's strict `<` to `<=` -- which makes
// the LARGEST disparity win a tie -- passed every case above. What forces the
// rule is content whose costs are equal by construction:
//
//   * A CONSTANT frame. Census compares each neighbour against the centre with
//     `>`, so a flat image gives a descriptor of all zeros everywhere, every
//     candidate's Hamming cost is 0, and every candidate ties. The map must
//     then be minDisparity at every pixel with support -- which is the tie rule
//     and nothing else.
//   * A HORIZONTALLY PERIODIC frame. Disparities differing by the period match
//     exactly as well as each other, so the ties are spread through the
//     disparity range rather than covering all of it.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, ContentThatTiesPinsTheSmallestDisparityRule) {
    for (const int minD : {0, 3, 7}) {
        bincv::DenseDisparityParams p;
        p.minDisparity = minD;
        p.maxDisparity = minD + 17;
        const size_t w = 113, h = 37;
        const std::vector<uint8_t> flat(w * h, 128);
        checkShapeOn(w, h, p, flat, flat, "a flat frame -- every candidate ties",
                     "constant");
    }
    for (const size_t period : {4u, 8u, 13u}) {
        bincv::DenseDisparityParams p;
        p.maxDisparity = 40;
        const size_t w = 137, h = 33;
        std::vector<uint8_t> img(w * h);
        uint64_t seed = 0x7E51ULL + period;
        std::vector<uint8_t> col(period);
        for (auto& v : col) v = static_cast<uint8_t>(splitmix(seed) >> 40);
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                img[y * w + x] = static_cast<uint8_t>(col[x % period] +
                                                      static_cast<uint8_t>(y % 3));
        checkShapeOn(w, h, p, img, img, "a periodic frame -- ties every `period`",
                     "periodic");
    }
}

// ---------------------------------------------------------------------------
// The reference frame's own shape, once, so the suite is not made only of
// small cases. 256x128 is the largest that keeps the suite quick while still
// spanning several warps in x and several strips in y.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCensusBox, ReferenceShapeAtTheReferenceParameters) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    p.winWidth = 9;
    p.winHeight = 9;
    checkShape(256, 128, p, 21, 0x5EED, "the shipped parameters at a frame-like size");
}

namespace {
bool cudaDevicePresent() {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("SKIP: no CUDA device available (%s)\n",
                    err == cudaSuccess ? "zero devices" : cudaGetErrorString(err));
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
    const int summaryRc = ::bincv::test::summarize("CUDA census box-matcher tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA census box-matcher tests", argc, argv);
}
#endif
