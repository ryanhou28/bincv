// Device-versus-host bit-exactness for the sensor stage: threshold, binarize,
// edgeThreshold.
//
// The shared representation makes every case a raw comparison: run the host
// kernel and the device kernel on the same bytes, download, compare word for
// word. The host library is the truth.
//
// TWO THINGS HERE THAT THE OTHER BACKEND SUITES DO NOT DO, both because they
// close a real hole rather than add coverage for its own sake:
//
//   * EVERY destination is checked for CLEAN PADDING -- `row[words-1]` AND-ed
//     with the complement of the tail mask must be zero. `mismatchWords` compares
//     only `rowWords` words per row and therefore cannot see a set bit past
//     `width`; without this the padding invariant is untested for these ops, and
//     a phantom bit there makes every later word-wise reduction over-count.
//
//   * The TIER 1 CHAIN for `threshold` is closed against the HOST's OWN
//     reduction. This gate configures -DBINCV_USE_OPENCV=OFF, so
//     `bincv::threshold(const cv::Mat&, ...)` does not exist here -- but
//     `impl::thresholdCutoff` does, because it was hoisted out of that #ifdef,
//     and so does `bincv::packBits`. The oracle below is those two composed,
//     which is the host entry point's own body rather than a restatement of it.
//     A test-local copy of the cutoff rule would agree on 127.5 and disagree on
//     NaN, and nothing would say so.
//
// The three BINCV_HOST_DEVICE helpers this family calls from device code --
// impl::reflect101Edge, thresholdGE and impl::thresholdCutoff -- are already
// pinned host-against-device by test_cuda_shared_helpers.cu. They are SHARED,
// not forked, so there is no twin here to hold in step; what that suite proves
// is that nvcc's device compilation of them agrees with the host's, which is the
// only failure a shared helper still has.
//
// Exits 77 when no CUDA device is present, like the other suites here.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/edge.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/threshold.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/edge.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/threshold.hpp"
#include "bincv/quantMat.hpp"
#include "test_util.hpp"

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// @brief The size matrix every op is swept over. Seven of the ten have a width
/// that is not a multiple of 32, which is what exercises the tail mask; the
/// three degenerate extents are where reflect-101 folds onto itself; and
/// {128, 3} sits exactly on the byte-lane arm's width gate, which is the one
/// width where "at least one full 128-pixel group" is true by nothing to
/// spare. 752 and 1000 are the two that put the byte-lane arm through its
/// grid-stride loop with a partial trailing group.
struct Size {
    size_t w, h;
};
const Size kSizes[] = {{752, 480}, {1000, 40}, {128, 3}, {97, 13}, {64, 5},
                       {33, 2},    {31, 7},    {1, 1},   {1, 9},   {9, 1}};

/// @brief A source containing EVERY value 0..255 at least once, so that for every
/// threshold swept the values `thresh - 1`, `thresh` and `thresh + 1` are all
/// present. A sampled fill misses exactly the boundary this operation is
/// most often wrong about.
std::vector<uint8_t> everyValueFrame(size_t w, size_t h) {
    std::vector<uint8_t> img(w * h);
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<uint8_t>(i % 256);
    return img;
}

template <typename T>
std::vector<T> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (auto& v : img) v = static_cast<T>(splitmix(seed));
    return img;
}

/// @brief A linear ramp: the central difference is constant, so a Wide /
/// Forward / Backward mix-up is a WHOLE-IMAGE difference rather than a
/// few pixels at the border.
template <typename T>
std::vector<T> rampFrame(size_t w, size_t h, unsigned step) {
    std::vector<T> img(w * h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
            img[y * w + x] = static_cast<T>((x + y) * step);
    return img;
}

/// @brief A single bright pixel on black: a sign error in the border reflection
/// shows as a misplaced pair of edges.
template <typename T>
std::vector<T> impulseFrame(size_t w, size_t h, T value) {
    std::vector<T> img(w * h, T{0});
    img[(h / 2) * w + (w / 2)] = value;
    return img;
}

size_t mismatchWords(const bincv::BinMat<uint32_t>& expect,
                     const bincv::BinMat<uint32_t>& got) {
    size_t bad = 0;
    const size_t words = bincv::cuda::rowWords(expect.getWidth());
    for (size_t y = 0; y < expect.getHeight(); ++y) {
        const uint32_t* a = expect.constView().row(y);
        const uint32_t* b = got.constView().row(y);
        for (size_t i = 0; i < words; ++i)
            if (a[i] != b[i]) ++bad;
    }
    return bad;
}

/// @brief Set bits PAST `width` in any row's trailing word. Must be zero.
/// @note mismatchWords cannot see these: it compares the same words on both
/// sides, and both sides being dirty in the same way reads as a match.
size_t dirtyPaddingBits(const bincv::BinMat<uint32_t>& m) {
    const size_t words = bincv::cuda::rowWords(m.getWidth());
    const uint32_t tail = bincv::cuda::rowTailMask(m.getWidth());
    if (tail == 0xFFFFFFFFu) return 0;  // the row ends on a word boundary
    size_t bad = 0;
    for (size_t y = 0; y < m.getHeight(); ++y) {
        const uint32_t v = m.constView().row(y)[words - 1] & ~tail;
        bad += static_cast<size_t>(__builtin_popcount(v));
    }
    return bad;
}

/// @brief Runs a device op into a fresh device matrix and downloads the result.
template <typename Launch>
bincv::BinMat<uint32_t> runDevice(size_t w, size_t h, Launch&& launch) {
    bincv::cuda::DeviceBinMat d(static_cast<int>(w), static_cast<int>(h));
    // Poisoned first: an op that forgets to write the trailing word, or a row,
    // then fails loudly instead of inheriting a zeroed allocation's answer.
    bincv::BinMat<uint32_t> poison(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        uint32_t* row = poison.view().row(y);
        for (size_t i = 0; i < bincv::cuda::rowWords(w); ++i) row[i] = 0xA5A5A5A5u;
    }
    bincv::cuda::upload(poison.constView(), d.view());
    const cudaError_t rc = launch(d.view());
    BINCV_CHECK_EQ(rc, cudaSuccess);
    bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::download(d.constView(), got.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return got;
}

/// @brief The host oracle for `cuda::threshold`: the host's OWN cutoff reduction
/// composed with the host's OWN packer -- the body of
/// `bincv::threshold(const cv::Mat&, ...)`, reachable with no OpenCV.
bincv::BinMat<uint32_t> hostThreshold(const std::vector<uint8_t>& src, size_t w, size_t h,
                                      double thresh) {
    bincv::BinMat<uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    const size_t words = bincv::cuda::rowWords(w);
    const int cutoff = bincv::impl::thresholdCutoff(thresh);
    if (cutoff <= 0) {
        const uint32_t tail = bincv::cuda::rowTailMask(w);
        for (size_t y = 0; y < h; ++y) {
            uint32_t* row = out.view().row(y);
            for (size_t i = 0; i < words; ++i) row[i] = (i + 1 == words) ? tail : ~0u;
        }
        return out;
    }
    if (cutoff > 255) {
        for (size_t y = 0; y < h; ++y) {
            uint32_t* row = out.view().row(y);
            for (size_t i = 0; i < words; ++i) row[i] = 0;
        }
        return out;
    }
    bincv::packBits<bincv::PackRule::GreaterEqual, uint8_t, uint32_t>(
        src.data(), w, h, w, out.view(), static_cast<uint8_t>(cutoff));
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// threshold -- TIER 1. The sweep is over the boundary, not the middle.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaThreshold, MatchesHostOverTheWholeThresholdRange) {
    const size_t w = 97, h = 13;  // width not a multiple of 32
    const auto frame = everyValueFrame(w, h);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);

    size_t bad = 0, dirty = 0;
    for (int t = -2; t <= 257; ++t) {
        const double thresh = static_cast<double>(t);
        const auto expect = hostThreshold(frame, w, h, thresh);
        const auto got = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
            return bincv::cuda::threshold(dImg.constView(), dst, thresh);
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

BINCV_TEST(CudaThreshold, MatchesHostOutsideCvThresholdsDomain) {
    const size_t w = 97, h = 13;
    const auto frame = everyValueFrame(w, h);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);

    const double kInf = std::numeric_limits<double>::infinity();
    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    const double thresholds[] = {127.5,  0.5,   254.5, -0.5,   1e300, -1e300,
                                 kInf,   -kInf, kNaN,  2147483648.0, -2147483648.0};
    size_t bad = 0, dirty = 0;
    for (double thresh : thresholds) {
        const auto expect = hostThreshold(frame, w, h, thresh);
        const auto got = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
            return bincv::cuda::threshold(dImg.constView(), dst, thresh);
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);

    // The named ends, checked as whole-image answers so an inverted relation is
    // not a few pixels but the entire map.
    const auto zero = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
        return bincv::cuda::threshold(dImg.constView(), dst, 0.0);
    });
    size_t setAtZero = 0, nonZeroPixels = 0;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            if (frame[y * w + x] != 0) ++nonZeroPixels;
            if ((zero.constView().row(y)[x / 32] >> (x % 32)) & 1u) ++setAtZero;
        }
    // thresh 0 sets every NON-ZERO pixel, not every pixel: the `>=` bug.
    BINCV_CHECK_EQ(setAtZero, nonZeroPixels);
    BINCV_CHECK(nonZeroPixels < w * h);

    const auto nan = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
        return bincv::cuda::threshold(dImg.constView(), dst, kNaN);
    });
    const auto negInf = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
        return bincv::cuda::threshold(dImg.constView(), dst, -kInf);
    });
    size_t setAtNaN = 0, setAtNegInf = 0;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) {
            setAtNaN += (nan.constView().row(y)[x / 32] >> (x % 32)) & 1u;
            setAtNegInf += (negInf.constView().row(y)[x / 32] >> (x % 32)) & 1u;
        }
    BINCV_CHECK_EQ(setAtNaN, 0u);        // p > NaN is false for every p
    BINCV_CHECK_EQ(setAtNegInf, w * h);  // everything exceeds -inf
}

BINCV_TEST(CudaThreshold, MatchesHostAcrossTheSizeMatrix) {
    const double thresholds[] = {0.0, 1.0, 127.0, 254.0, 255.0, 300.0, -1.0};
    size_t bad = 0, dirty = 0;
    for (const auto& s : kSizes) {
        const auto frame = everyValueFrame(s.w, s.h);
        bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(s.w),
                                               static_cast<int>(s.h));
        BINCV_CHECK_EQ(
            bincv::cuda::uploadImage<uint8_t>(frame.data(), s.w, s.h, s.w, dImg.view()),
            cudaSuccess);
        for (double thresh : thresholds) {
            const auto expect = hostThreshold(frame, s.w, s.h, thresh);
            const auto got = runDevice(s.w, s.h, [&](bincv::cuda::DeviceBinMatView dst) {
                return bincv::cuda::threshold(dImg.constView(), dst, thresh);
            });
            bad += mismatchWords(expect, got);
            dirty += dirtyPaddingBits(got);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

#ifdef BINCV_WITH_OPENCV
// The other half of the Tier 1 chain, where OpenCV is available: the device arm
// against the host ENTRY POINT rather than against its recomposed body. The gate
// configures -DBINCV_USE_OPENCV=OFF, so this case does not run there -- which is
// exactly why the recomposed oracle above exists.
BINCV_TEST(CudaThreshold, MatchesHostEntryPointThroughOpenCV) {
    const size_t w = 97, h = 13;
    const auto frame = everyValueFrame(w, h);
    cv::Mat src(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                const_cast<uint8_t*>(frame.data()));
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    size_t bad = 0;
    for (int t = -1; t <= 256; ++t) {
        const double thresh = static_cast<double>(t);
        bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
        bincv::threshold<uint32_t>(src, expect.view(), thresh);
        const auto got = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
            return bincv::cuda::threshold(dImg.constView(), dst, thresh);
        });
        bad += mismatchWords(expect, got);
    }
    BINCV_CHECK_EQ(bad, 0u);
}
#endif

// ---------------------------------------------------------------------------
// binarize -- TIER 3. Plane blocks from the real packer where one exists, and
// synthesized above it: packQuant caps at 8 on BOTH sides, while the plane-view
// entry point does not, so n = 16 and n = 32 have to build their own planes.
// ---------------------------------------------------------------------------
namespace {

/// @brief Host `binarize<N>` over a plane block laid out as the device expects.
template <size_t N>
bincv::BinMat<uint32_t> hostBinarize(const bincv::BinMat<uint32_t>& block, size_t w,
                                     size_t h, unsigned thresh) {
    bincv::BinMatConstView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) {
        planes[p] = bincv::BinMatConstView<uint32_t>{
            block.constView().ptr + p * h * block.constView().stride, w, h,
            block.constView().stride};
    }
    bincv::BinMat<uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    // Deduced rather than spelled: naming <N, uint32_t> explicitly also forms
    // QuantMat<N, uint32_t> for the sibling overload, and that type static-asserts
    // above 8 planes. Deduction discards that candidate before it is instantiated.
    bincv::binarize(planes, out.view(), thresh);
    return out;
}

/// @brief Pseudo-random plane words, with the padding bits of every plane row
/// deliberately left DIRTY. thresholdGE answers every lane including those,
/// so this is what proves the destination's trailing word is masked rather
/// than inheriting the source's junk.
bincv::BinMat<uint32_t> dirtyPlaneBlock(size_t w, size_t h, size_t n, uint64_t seed) {
    bincv::BinMat<uint32_t> block(static_cast<int>(w), static_cast<int>(n * h));
    const size_t words = bincv::cuda::rowWords(w);
    for (size_t y = 0; y < n * h; ++y) {
        uint32_t* row = block.view().row(y);
        for (size_t i = 0; i < words; ++i) row[i] = static_cast<uint32_t>(splitmix(seed));
    }
    return block;
}

template <size_t N>
void binarizeCase(size_t w, size_t h, size_t& bad, size_t& dirty) {
    const auto block = dirtyPlaneBlock(w, h, N, 0x5EED0000u + N * 131 + w);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::upload(block.constView(), dBlock.view()), cudaSuccess);
    const bincv::cuda::DeviceBinMatConstView blockView = dBlock.constView();
    const bincv::cuda::DevicePlaneBlockConstView planes{blockView.ptr, w, h,
                                                        blockView.stride, N};

    // Every threshold from 0 to the "selects nothing" end, plus the two values a
    // caller reaches by arithmetic rather than by choice.
    const unsigned maxValue = static_cast<unsigned>((1ull << N) - 1ull);
    std::vector<unsigned> thresholds;
    if (N <= 8) {
        for (unsigned t = 0; t <= maxValue; ++t) thresholds.push_back(t);
    } else {
        for (unsigned t = 0; t < 8; ++t) thresholds.push_back(t);
        thresholds.push_back(maxValue / 2);
        thresholds.push_back(maxValue - 1);
        thresholds.push_back(maxValue);
    }
    thresholds.push_back(~0u);

    for (unsigned t : thresholds) {
        const auto expect = hostBinarize<N>(block, w, h, t);
        const auto got = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
            return bincv::cuda::binarize(planes, dst, t);
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
}

} // namespace

BINCV_TEST(CudaBinarize, MatchesHostAcrossPlaneCountsAndThresholds) {
    size_t bad = 0, dirty = 0;
    for (const auto& s : kSizes) {
        binarizeCase<1>(s.w, s.h, bad, dirty);
        binarizeCase<2>(s.w, s.h, bad, dirty);  // the shipped tracking depth
        binarizeCase<3>(s.w, s.h, bad, dirty);
        binarizeCase<5>(s.w, s.h, bad, dirty);
        binarizeCase<8>(s.w, s.h, bad, dirty);  // QuantMat's cap
    }
    // The plane-view entry point goes past QuantMat's cap; packQuant does not, so
    // these two cannot be produced by it and are synthesized above.
    binarizeCase<16>(97, 13, bad, dirty);
    binarizeCase<32>(97, 13, bad, dirty);  // the widest `unsigned` can express
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

BINCV_TEST(CudaBinarize, ConsumesPackQuantOutputUnchanged) {
    // The pipeline shape the op exists for: a frame ingested by the DEVICE
    // packer, thresholded by the device binarizer, against the same two host
    // kernels. No repacking between them -- one allocation, one stride.
    const size_t w = 752, h = 480, n = 4;
    const auto frame = randomFrame<uint8_t>(w, h, 0xC0FFEEu);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(n * h));
    BINCV_CHECK_EQ(bincv::cuda::packQuant(dImg.constView(), dBlock.view(), n),
                   cudaSuccess);

    bincv::QuantMat<4, uint32_t> hostQ(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMatView<uint32_t> hostPlanes[4];
    for (size_t p = 0; p < 4; ++p) hostPlanes[p] = hostQ.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, 4, uint8_t, uint32_t>(frame.data(), w, h, w,
                                                                    hostPlanes);

    size_t bad = 0, dirty = 0;
    const bincv::cuda::DeviceBinMatConstView blockView = dBlock.constView();
    const bincv::cuda::DevicePlaneBlockConstView planes{blockView.ptr, w, h,
                                                        blockView.stride, n};
    for (unsigned t = 0; t <= 16u; ++t) {
        bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
        bincv::binarize<4, uint32_t>(hostQ, expect.view(), t);
        const auto got = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView dst) {
            return bincv::cuda::binarize(planes, dst, t);
        });
        bad += mismatchWords(expect, got);
        dirty += dirtyPaddingBits(got);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
}

// The device domain is narrower than what the view type can express, and the
// rule for such a domain is that the op NAMES it, ASSERTS it, and RETURNS an
// error outside it. Those two halves cannot be exercised in one build: where the
// assertion is live it aborts before the return is reached. So the case exists
// in both configurations and checks the half that configuration has, rather than
// disappearing from one of them -- a check that silently stops running is worse
// than one that says which half it is.
#if BINCV_DEBUG_CHECKS
BINCV_TEST(CudaBinarize, RejectsAPlaneCountOutsideItsDomain) {
    BINCV_CHECK_EQ(BINCV_DEBUG_CHECKS, 1);
    std::printf("        [half] assertions are LIVE here, so cuda::binarize aborts on an\n"
                "               out-of-domain plane count before its error return is\n"
                "               reached. The RETURN half runs in the release build.\n");
}
#else
BINCV_TEST(CudaBinarize, RejectsAPlaneCountOutsideItsDomain) {
    BINCV_CHECK_EQ(BINCV_DEBUG_CHECKS, 0);
    const size_t w = 64, h = 4;
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceBinMat dOut(static_cast<int>(w), static_cast<int>(h));
    const bincv::cuda::DeviceBinMatConstView bv = dBlock.constView();
    const bincv::cuda::DevicePlaneBlockConstView tooMany{bv.ptr, w, h, bv.stride, 33};
    const bincv::cuda::DevicePlaneBlockConstView none{bv.ptr, w, h, bv.stride, 0};
    BINCV_CHECK_EQ(bincv::cuda::binarize(tooMany, dOut.view(), 0u), cudaErrorInvalidValue);
    BINCV_CHECK_EQ(bincv::cuda::binarize(none, dOut.view(), 0u), cudaErrorInvalidValue);
}
#endif

// ---------------------------------------------------------------------------
// edgeThreshold -- TIER 3. Twelve parameter combinations, two source types, four
// source shapes, and BOTH ARMS HELD TO THE SAME MAP IN ONE BINARY.
// ---------------------------------------------------------------------------
namespace {

template <typename SrcT>
bincv::BinMat<uint32_t> hostEdge(const std::vector<SrcT>& frame, size_t w, size_t h,
                                 SrcT t, bincv::EdgeCombine c, bincv::EdgeRelation r,
                                 bincv::EdgeSpatial s) {
    bincv::BinMat<uint32_t> out(static_cast<int>(w), static_cast<int>(h));
    using bincv::EdgeCombine;
    using bincv::EdgeRelation;
    using bincv::EdgeSpatial;
#define BINCV_EDGE_CASE(CC, RR, SS)                                                  \
    if (c == EdgeCombine::CC && r == EdgeRelation::RR && s == EdgeSpatial::SS) {      \
        bincv::edgeThreshold<EdgeCombine::CC, EdgeRelation::RR, EdgeSpatial::SS,      \
                             SrcT, uint32_t>(frame.data(), w, h, w, out.view(), t);   \
        return out;                                                                  \
    }
    BINCV_EDGE_CASE(Or, Ge, Wide)
    BINCV_EDGE_CASE(Or, Ge, Forward)
    BINCV_EDGE_CASE(Or, Ge, Backward)
    BINCV_EDGE_CASE(Or, Gt, Wide)
    BINCV_EDGE_CASE(Or, Gt, Forward)
    BINCV_EDGE_CASE(Or, Gt, Backward)
    BINCV_EDGE_CASE(And, Ge, Wide)
    BINCV_EDGE_CASE(And, Ge, Forward)
    BINCV_EDGE_CASE(And, Ge, Backward)
    BINCV_EDGE_CASE(And, Gt, Wide)
    BINCV_EDGE_CASE(And, Gt, Forward)
    BINCV_EDGE_CASE(And, Gt, Backward)
#undef BINCV_EDGE_CASE
    return out;
}

const bincv::EdgeCombine kCombines[] = {bincv::EdgeCombine::Or, bincv::EdgeCombine::And};
const bincv::EdgeRelation kRelations[] = {bincv::EdgeRelation::Ge,
                                          bincv::EdgeRelation::Gt};
const bincv::EdgeSpatial kSpatials[] = {bincv::EdgeSpatial::Wide,
                                        bincv::EdgeSpatial::Forward,
                                        bincv::EdgeSpatial::Backward};

/// @brief One (frame, size, threshold) against every parameter combination, on
/// BOTH arms, comparing each to the host and the two to each other.
template <typename SrcT>
void edgeSweep(const std::vector<SrcT>& frame, size_t w, size_t h, SrcT t, size_t& bad,
               size_t& dirty, size_t& armDiff) {
    bincv::cuda::DeviceImage<SrcT> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<SrcT>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    for (auto c : kCombines) {
        for (auto r : kRelations) {
            for (auto s : kSpatials) {
                const auto expect = hostEdge<SrcT>(frame, w, h, t, c, r, s);
                bincv::cuda::impl::edgeVectorEnabled() = true;
                const auto onArm = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
                    return bincv::cuda::edgeThreshold(dImg.constView(), d, t, c, r, s);
                });
                bincv::cuda::impl::edgeVectorEnabled() = false;
                const auto offArm = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
                    return bincv::cuda::edgeThreshold(dImg.constView(), d, t, c, r, s);
                });
                bincv::cuda::impl::edgeVectorEnabled() = true;
                bad += mismatchWords(expect, onArm);
                bad += mismatchWords(expect, offArm);
                armDiff += mismatchWords(onArm, offArm);
                dirty += dirtyPaddingBits(onArm) + dirtyPaddingBits(offArm);
            }
        }
    }
}

} // namespace

BINCV_TEST(CudaEdgeThreshold, MatchesHostAcrossEveryCombinationAndSize) {
    size_t bad = 0, dirty = 0, armDiff = 0;
    for (const auto& s : kSizes) {
        const auto rnd = randomFrame<uint8_t>(s.w, s.h, 0xED6E0000u + s.w);
        const auto ramp = rampFrame<uint8_t>(s.w, s.h, 7);
        const auto imp = impulseFrame<uint8_t>(s.w, s.h, uint8_t{255});
        const std::vector<uint8_t> flat(s.w * s.h, uint8_t{119});
        for (uint8_t t : {uint8_t{0}, uint8_t{1}, uint8_t{17}, uint8_t{24}, uint8_t{128},
                          uint8_t{254}, uint8_t{255}}) {
            edgeSweep<uint8_t>(rnd, s.w, s.h, t, bad, dirty, armDiff);
        }
        edgeSweep<uint8_t>(ramp, s.w, s.h, uint8_t{7}, bad, dirty, armDiff);
        edgeSweep<uint8_t>(ramp, s.w, s.h, uint8_t{14}, bad, dirty, armDiff);
        edgeSweep<uint8_t>(imp, s.w, s.h, uint8_t{200}, bad, dirty, armDiff);
        edgeSweep<uint8_t>(flat, s.w, s.h, uint8_t{0}, bad, dirty, armDiff);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    // The two arms are the same operation. If they ever disagree the faster one
    // is not an optimization.
    BINCV_CHECK_EQ(armDiff, 0u);
}

BINCV_TEST(CudaEdgeThreshold, MatchesHostForA16BitSource) {
    // Why SrcT is not just uint8_t: downconverting 12 -> 8 first truncates the
    // OPERANDS, and a genuine 12-bit gradient of 15 counts becomes exactly zero.
    size_t bad = 0, dirty = 0, armDiff = 0;
    for (const auto& s : kSizes) {
        const auto rnd = randomFrame<uint16_t>(s.w, s.h, 0x16B00000u + s.w);
        const auto ramp = rampFrame<uint16_t>(s.w, s.h, 137);
        for (uint16_t t : {uint16_t{0}, uint16_t{1}, uint16_t{17}, uint16_t{4095},
                           uint16_t{65535}}) {
            edgeSweep<uint16_t>(rnd, s.w, s.h, t, bad, dirty, armDiff);
        }
        edgeSweep<uint16_t>(ramp, s.w, s.h, uint16_t{137}, bad, dirty, armDiff);
        edgeSweep<uint16_t>(ramp, s.w, s.h, uint16_t{274}, bad, dirty, armDiff);
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    BINCV_CHECK_EQ(armDiff, 0u);
}

BINCV_TEST(CudaEdgeThreshold, NamedWholeImageAnswersAtTheEnds) {
    // Two cases where getting the relation or the fold wrong is not a few pixels
    // but the entire map, which is what makes them worth naming.
    const size_t w = 200, h = 9;
    const auto frame = randomFrame<uint8_t>(w, h, 0x9911u);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);

    // t = 0 with Ge: every difference is >= 0, so EVERY pixel is an edge.
    const auto all = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
        return bincv::cuda::edgeThreshold(dImg.constView(), d, uint8_t{0},
                                          bincv::EdgeCombine::And,
                                          bincv::EdgeRelation::Ge);
    });
    size_t set = 0;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) set += (all.constView().row(y)[x / 32] >> (x % 32)) & 1u;
    BINCV_CHECK_EQ(set, w * h);
    BINCV_CHECK_EQ(dirtyPaddingBits(all), 0u);

    // t = 255 with Gt folds the threshold to 256, which no unsigned byte
    // difference reaches: nothing is an edge, and no special case says so.
    const auto none = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
        return bincv::cuda::edgeThreshold(dImg.constView(), d, uint8_t{255},
                                          bincv::EdgeCombine::Or,
                                          bincv::EdgeRelation::Gt);
    });
    set = 0;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
            set += (none.constView().row(y)[x / 32] >> (x % 32)) & 1u;
    BINCV_CHECK_EQ(set, 0u);
}

BINCV_TEST(CudaEdgeThreshold, MatchesHostOnASubWidthViewOfAWiderFrame) {
    // THE ONLY SHAPE WHERE A QUAD STRADDLES `width`. The byte-lane arm's gate is
    // on the STRIDE being a multiple of four, not the width, so a sub-width
    // window onto a wider frame can have a width that is not -- and then the
    // four-pixel quad holding the last pixel runs past it and has to fall to the
    // per-pixel predicate mid-warp. A tight-stride DeviceImage can never produce
    // that shape (its stride IS its width, so the gate only ever accepts widths
    // that are multiples of four), which is exactly why it is built explicitly
    // here rather than swept for.
    const size_t full = 752, h = 40;
    const auto frame = randomFrame<uint8_t>(full, h, 0x5B0B0u);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(full), static_cast<int>(h));
    BINCV_CHECK_EQ(
        bincv::cuda::uploadImage<uint8_t>(frame.data(), full, h, full, dImg.view()),
        cudaSuccess);
    const bincv::cuda::DeviceImageView<uint8_t> fullView = dImg.view();

    size_t bad = 0, dirty = 0, armDiff = 0;
    for (size_t w : {size_t{749}, size_t{750}, size_t{751}, size_t{752}}) {
        const bincv::cuda::DeviceImageConstView<uint8_t> sub{fullView.ptr, w, h,
                                                             fullView.stride};
        BINCV_CHECK(bincv::cuda::impl::edgeVectorApplies(
            w, fullView.stride, fullView.ptr, 1, bincv::EdgeSpatial::Wide, 17));
        for (auto c : kCombines) {
            bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
            if (c == bincv::EdgeCombine::Or) {
                bincv::edgeThreshold<bincv::EdgeCombine::Or, bincv::EdgeRelation::Ge,
                                     bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
                    frame.data(), w, h, full, expect.view(), uint8_t{17});
            } else {
                bincv::edgeThreshold<bincv::EdgeCombine::And, bincv::EdgeRelation::Ge,
                                     bincv::EdgeSpatial::Wide, uint8_t, uint32_t>(
                    frame.data(), w, h, full, expect.view(), uint8_t{17});
            }
            bincv::cuda::impl::edgeVectorEnabled() = true;
            const auto on = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
                return bincv::cuda::edgeThreshold(sub, d, uint8_t{17}, c);
            });
            bincv::cuda::impl::edgeVectorEnabled() = false;
            const auto off = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
                return bincv::cuda::edgeThreshold(sub, d, uint8_t{17}, c);
            });
            bincv::cuda::impl::edgeVectorEnabled() = true;
            bad += mismatchWords(expect, on) + mismatchWords(expect, off);
            armDiff += mismatchWords(on, off);
            dirty += dirtyPaddingBits(on) + dirtyPaddingBits(off);
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(dirty, 0u);
    BINCV_CHECK_EQ(armDiff, 0u);
}

BINCV_TEST(CudaEdgeThreshold, TheVectorArmsGateIsWhatItSaysItIs) {
    // The runtime switch alone is half the rule. The other half is a shape the
    // fast arm's own gate REJECTS -- which the benchmark then has to time at
    // ~1.00x. Both halves need the gate to be one predicate, not a restatement.
    using bincv::cuda::impl::edgeVectorApplies;
    alignas(4) uint8_t base[8] = {0};

    // The shipped shape, on an aligned tight row: in.
    BINCV_CHECK(edgeVectorApplies(752, 752, base, 1, bincv::EdgeSpatial::Wide, 17));
    // uint16 source: the instructions are byte-lane.
    BINCV_CHECK(!edgeVectorApplies(752, 752, base, 2, bincv::EdgeSpatial::Wide, 17));
    // Forward and Backward: the host's own vector arm covers only Wide too.
    BINCV_CHECK(!edgeVectorApplies(752, 752, base, 1, bincv::EdgeSpatial::Forward, 17));
    BINCV_CHECK(!edgeVectorApplies(752, 752, base, 1, bincv::EdgeSpatial::Backward, 17));
    // tp == 256, which t = 255 with Gt folds to: a byte lane cannot hold it.
    BINCV_CHECK(!edgeVectorApplies(752, 752, base, 1, bincv::EdgeSpatial::Wide, 256));
    // A tight stride that is not a multiple of 4: the arm reads the row as
    // uint32, and the stride clause is also what keeps the last word-load of a
    // row inside that row.
    BINCV_CHECK(!edgeVectorApplies(97, 97, base, 1, bincv::EdgeSpatial::Wide, 17));
    BINCV_CHECK(edgeVectorApplies(752, 752, base, 1, bincv::EdgeSpatial::Wide, 255));
    // Narrower than one warp's 128 pixels.
    BINCV_CHECK(!edgeVectorApplies(64, 64, base, 1, bincv::EdgeSpatial::Wide, 17));
    // An unaligned row base.
    BINCV_CHECK(!edgeVectorApplies(752, 752, base + 1, 1, bincv::EdgeSpatial::Wide, 17));

    // And the switch must actually switch: with it off, a shape the gate ACCEPTS
    // still has to give the identical map.
    const size_t w = 752, h = 33;
    const auto frame = randomFrame<uint8_t>(w, h, 0x5117u);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    BINCV_CHECK(edgeVectorApplies(w, dImg.getStride(), nullptr, 1,
                                  bincv::EdgeSpatial::Wide, 17));
    bincv::cuda::impl::edgeVectorEnabled() = true;
    const auto on = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
        return bincv::cuda::edgeThreshold(dImg.constView(), d, uint8_t{17});
    });
    bincv::cuda::impl::edgeVectorEnabled() = false;
    const auto off = runDevice(w, h, [&](bincv::cuda::DeviceBinMatView d) {
        return bincv::cuda::edgeThreshold(dImg.constView(), d, uint8_t{17});
    });
    bincv::cuda::impl::edgeVectorEnabled() = true;
    BINCV_CHECK_EQ(mismatchWords(on, off), 0u);
    BINCV_CHECK_EQ(mismatchWords(hostEdge<uint8_t>(frame, w, h, uint8_t{17},
                                                   bincv::EdgeCombine::Or,
                                                   bincv::EdgeRelation::Ge,
                                                   bincv::EdgeSpatial::Wide),
                                 on),
                   0u);
}

// ---------------------------------------------------------------------------
// Entry point: probe the device first, and report "not performed" as 77 --
// a pass this binary did not earn is worse than a skip it announces.
// ---------------------------------------------------------------------------
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
    const int summaryRc = ::bincv::test::summarize("CUDA sensor-stage tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA sensor-stage tests", argc, argv);
}
#endif
