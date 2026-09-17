// Device-versus-host bit-exactness for the CUDA backend.
//
// The shared representation makes every one of these a raw comparison: run the
// host kernel and the device kernel on the same input, download, and the maps
// must match word for word (bit matrices) or byte for byte (disparity). The
// host library is the truth; a device kernel that is faster and different is
// not an optimization.
//
// Exits 77 -- the "not performed" code verify_cross.sh and verify_cortex_m.sh
// already use -- when no CUDA device is present, so a GPU-less configure of
// the backend still builds this suite and reports honestly.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/census.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/logic.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/reduce.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/census.hpp"
#include "bincv/ops/denseDisparity.hpp"
#include "bincv/ops/logic.hpp"
#include "bincv/ops/pack.hpp"
#include "bincv/ops/reduce.hpp"
#include "test_util.hpp"

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template <typename T>
std::vector<T> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (auto& v : img) v = static_cast<T>(splitmix(seed));
    return img;
}

/// A synthetic rectified pair: right is left shifted by a known disparity, so
/// the dense tests exercise a map with real structure rather than noise.
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

template <typename W>
bincv::BinMat<W> randomBits(size_t w, size_t h, uint64_t seed) {
    const auto frame = randomFrame<uint8_t>(w, h, seed);
    bincv::BinMat<W> m(static_cast<int>(w), static_cast<int>(h));
    bincv::packBits<bincv::PackRule::GreaterThan>(frame.data(), w, h, w, m.view(),
                                                  uint8_t{127});
    return m;
}

/// Words that differ between a host matrix and a downloaded device result.
template <typename W>
size_t mismatchWords(const bincv::BinMat<W>& expect, const bincv::BinMat<W>& got) {
    size_t bad = 0;
    const size_t words = (expect.getWidth() * 1 + expect.WordBits - 1) / expect.WordBits;
    for (size_t y = 0; y < expect.getHeight(); ++y) {
        const W* a = expect.constView().row(y);
        const W* b = got.constView().row(y);
        for (size_t i = 0; i < words; ++i)
            if (a[i] != b[i]) ++bad;
    }
    return bad;
}

template <typename W>
bincv::BinMat<W> roundTrip(const bincv::BinMat<W>& src) {
    bincv::cuda::DeviceBinMat d(static_cast<int>(src.getWidth()),
                                static_cast<int>(src.getHeight()));
    BINCV_CHECK_EQ(bincv::cuda::upload(src.constView(), d.view()), cudaSuccess);
    bincv::BinMat<W> back(static_cast<int>(src.getWidth()),
                          static_cast<int>(src.getHeight()));
    BINCV_CHECK_EQ(bincv::cuda::download(d.constView(), back.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return back;
}

} // namespace

// ---------------------------------------------------------------------------
// The twin helpers cannot drift: the device copies of minRowWords and
// rowTailMask are asserted equal to the host originals across widths.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaCore, TwinHelpersAgreeWithHost) {
    size_t badWords = 0, badMasks = 0;
    for (size_t w = 1; w <= 4100; ++w) {
        if (bincv::cuda::rowWords(w) != bincv::impl::minRowWords<uint32_t>(w)) ++badWords;
        if (bincv::cuda::rowTailMask(w) != bincv::impl::rowTailMask<uint32_t>(w))
            ++badMasks;
    }
    BINCV_CHECK_EQ(badWords, 0u);
    BINCV_CHECK_EQ(badMasks, 0u);
}

// ---------------------------------------------------------------------------
// Transfers: a raw byte copy in both directions, at every host word width,
// including widths that do not end on any word boundary.
// ---------------------------------------------------------------------------
namespace {
template <typename W>
void testRoundTrip(const char* label) {
    const size_t sizes[][2] = {{97, 13}, {752, 480}, {64, 5}, {33, 2}, {1, 1}};
    for (const auto& s : sizes) {
        const auto m = randomBits<W>(s[0], s[1], 0xABCD0000u + s[0]);
        const auto back = roundTrip<W>(m);
        BINCV_CHECK_EQ(mismatchWords<W>(m, back), 0u);
    }
    (void)label;
}
} // namespace

BINCV_TEST(CudaTransfer, RoundTrip_uint8_t) { testRoundTrip<uint8_t>("uint8_t"); }
BINCV_TEST(CudaTransfer, RoundTrip_uint16_t) { testRoundTrip<uint16_t>("uint16_t"); }
BINCV_TEST(CudaTransfer, RoundTrip_uint32_t) { testRoundTrip<uint32_t>("uint32_t"); }
BINCV_TEST(CudaTransfer, RoundTrip_uint64_t) { testRoundTrip<uint64_t>("uint64_t"); }

// ---------------------------------------------------------------------------
// Logic: device twins against the host Tier 1 kernels, including the in-place
// case and the padding invariant after NOT.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaLogic, MatchesHost) {
    const size_t sizes[][2] = {{97, 13}, {640, 480}, {31, 7}};
    for (const auto& s : sizes) {
        const size_t w = s[0], h = s[1];
        const auto a = randomBits<uint32_t>(w, h, 0x1111 + w);
        const auto b = randomBits<uint32_t>(w, h, 0x2222 + w);

        bincv::cuda::DeviceBinMat da(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceBinMat db(static_cast<int>(w), static_cast<int>(h));
        bincv::cuda::DeviceBinMat dd(static_cast<int>(w), static_cast<int>(h));
        BINCV_CHECK_EQ(bincv::cuda::upload(a.constView(), da.view()), cudaSuccess);
        BINCV_CHECK_EQ(bincv::cuda::upload(b.constView(), db.view()), cudaSuccess);

        bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
        bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));

        struct OpCase {
            const char* name;
            int op;
        } cases[] = {{"and", 0}, {"or", 1}, {"xor", 2}, {"not", 3}};
        for (const auto& c : cases) {
            switch (c.op) {
                case 0:
                    bincv::bitwiseAnd<uint32_t>(a.constView(), b.constView(), expect.view());
                    BINCV_CHECK_EQ(bincv::cuda::bitwiseAnd(da.constView(), db.constView(),
                                                           dd.view()),
                                   cudaSuccess);
                    break;
                case 1:
                    bincv::bitwiseOr<uint32_t>(a.constView(), b.constView(), expect.view());
                    BINCV_CHECK_EQ(bincv::cuda::bitwiseOr(da.constView(), db.constView(),
                                                          dd.view()),
                                   cudaSuccess);
                    break;
                case 2:
                    bincv::bitwiseXor<uint32_t>(a.constView(), b.constView(), expect.view());
                    BINCV_CHECK_EQ(bincv::cuda::bitwiseXor(da.constView(), db.constView(),
                                                           dd.view()),
                                   cudaSuccess);
                    break;
                case 3:
                    bincv::bitwiseNot<uint32_t>(a.constView(), expect.view());
                    BINCV_CHECK_EQ(bincv::cuda::bitwiseNot(da.constView(), dd.view()),
                                   cudaSuccess);
                    break;
            }
            BINCV_CHECK_EQ(bincv::cuda::download(dd.constView(), got.view()), cudaSuccess);
            BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
            BINCV_CHECK_EQ(mismatchWords<uint32_t>(expect, got), 0u);
        }

        // In-place, the supported exact-alias case: db &= da.
        bincv::BinMat<uint32_t> bCopy(b);
        bincv::bitwiseAnd<uint32_t>(bCopy.constView(), a.constView(), bCopy.view());
        BINCV_CHECK_EQ(bincv::cuda::bitwiseAnd(db.constView(), da.constView(), db.view()),
                       cudaSuccess);
        BINCV_CHECK_EQ(bincv::cuda::download(db.constView(), got.view()), cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        BINCV_CHECK_EQ(mismatchWords<uint32_t>(bCopy, got), 0u);
    }
}

// ---------------------------------------------------------------------------
// Reductions: whole view and clipped regions, negative origins included --
// the clip is the host's own clipRegion, so what is being tested is the
// traversal and the counting.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaReduce, CountNonZeroMatchesHost) {
    const auto m = randomBits<uint32_t>(200, 90, 0x5150);
    bincv::cuda::DeviceBinMat d(200, 90);
    BINCV_CHECK_EQ(bincv::cuda::upload(m.constView(), d.view()), cudaSuccess);

    BINCV_CHECK_EQ(bincv::cuda::countNonZero(d.constView()),
                   bincv::countNonZero<uint32_t>(m.constView()));

    const bincv::Rect rects[] = {
        {0, 0, 200, 90},   {3, 5, 60, 40},    {-10, -10, 50, 50},
        {150, 60, 500, 500}, {31, 0, 34, 90}, {0, 89, 200, 1},
        {199, 0, 1, 90},   {5, 5, 0, 10},     {-300, 4, 20, 20},
    };
    for (const auto& r : rects) {
        BINCV_CHECK_EQ(bincv::cuda::countNonZero(d.constView(), r),
                       bincv::countNonZero<uint32_t>(m.constView(), r));
    }
}

// ---------------------------------------------------------------------------
// The sensor stage: device packBits against host packBits, all three rules,
// both source widths.
// ---------------------------------------------------------------------------
namespace {
template <typename SrcT>
void testPack(const char* label) {
    const size_t w = 197, h = 61;
    const auto frame = randomFrame<SrcT>(w, h, 0xBEEF);
    bincv::cuda::DeviceImage<SrcT> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<SrcT>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
    const SrcT threshold = static_cast<SrcT>(sizeof(SrcT) == 1 ? 127 : 31000);

    const bincv::PackRule rules[] = {bincv::PackRule::NonZero,
                                     bincv::PackRule::GreaterThan,
                                     bincv::PackRule::GreaterEqual};
    for (const auto rule : rules) {
        switch (rule) {
            case bincv::PackRule::NonZero:
                bincv::packBits<bincv::PackRule::NonZero>(frame.data(), w, h, w,
                                                          expect.view(), threshold);
                break;
            case bincv::PackRule::GreaterThan:
                bincv::packBits<bincv::PackRule::GreaterThan>(frame.data(), w, h, w,
                                                              expect.view(), threshold);
                break;
            case bincv::PackRule::GreaterEqual:
                bincv::packBits<bincv::PackRule::GreaterEqual>(frame.data(), w, h, w,
                                                               expect.view(), threshold);
                break;
        }
        BINCV_CHECK_EQ(bincv::cuda::packBits(dImg.constView(), dBits.view(), rule,
                                             threshold),
                       cudaSuccess);
        BINCV_CHECK_EQ(bincv::cuda::download(dBits.constView(), got.view()), cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        BINCV_CHECK_EQ(mismatchWords<uint32_t>(expect, got), 0u);
    }
    (void)label;
}
} // namespace

BINCV_TEST(CudaPack, MatchesHost_uint8_t) { testPack<uint8_t>("uint8_t"); }
BINCV_TEST(CudaPack, MatchesHost_uint16_t) { testPack<uint16_t>("uint16_t"); }

// ---------------------------------------------------------------------------
// Census: every plane of the device block against the host transform, both
// shipped patterns, both source widths.
// ---------------------------------------------------------------------------
namespace {
template <size_t K, typename SrcT>
void testCensus(const bincv::CensusPattern<K>& pattern) {
    const size_t w = 101, h = 37;
    const auto frame = randomFrame<SrcT>(w, h, 0xCE9505 + K);

    // Host: K standalone planes.
    std::vector<bincv::BinMat<uint32_t>> planes;
    std::vector<bincv::BinMatView<uint32_t>> views;
    planes.reserve(K);
    for (size_t k = 0; k < K; ++k)
        planes.emplace_back(static_cast<int>(w), static_cast<int>(h));
    for (size_t k = 0; k < K; ++k) views.push_back(planes[k].view());
    bincv::censusTransform<K, SrcT, uint32_t>(frame.data(), w, h, w, pattern,
                                              views.data());

    // Device: one K-plane block.
    bincv::cuda::DeviceImage<SrcT> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<SrcT>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(K * h));
    BINCV_CHECK_EQ(bincv::cuda::censusTransform<K>(dImg.constView(), pattern,
                                                   dBlock.view()),
                   cudaSuccess);
    bincv::BinMat<uint32_t> gotBlock(static_cast<int>(w), static_cast<int>(K * h));
    BINCV_CHECK_EQ(bincv::cuda::download(dBlock.constView(), gotBlock.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t badPlanes = 0;
    const size_t words = bincv::impl::minRowWords<uint32_t>(w);
    for (size_t k = 0; k < K; ++k) {
        size_t bad = 0;
        for (size_t y = 0; y < h; ++y) {
            const uint32_t* a = planes[k].constView().row(y);
            const uint32_t* b = gotBlock.constView().row(k * h + y);
            for (size_t i = 0; i < words; ++i)
                if (a[i] != b[i]) ++bad;
        }
        if (bad != 0) ++badPlanes;
    }
    BINCV_CHECK_EQ(badPlanes, 0u);
}
} // namespace

BINCV_TEST(CudaCensus, Census5x5_uint8_t) { testCensus<24, uint8_t>(bincv::kCensus5x5); }
BINCV_TEST(CudaCensus, Census3x3_uint8_t) { testCensus<8, uint8_t>(bincv::kCensus3x3); }
BINCV_TEST(CudaCensus, Census5x5_uint16_t) { testCensus<24, uint16_t>(bincv::kCensus5x5); }

// ---------------------------------------------------------------------------
// Dense disparity, binary entry: the device map against the host map, byte for
// byte, across parameter shapes including the degenerate ones.
// ---------------------------------------------------------------------------
namespace {
void testDenseBinary(size_t w, size_t h, const bincv::DenseDisparityParams& p,
                     int shift) {
    const auto lw = smoothFrame(w, h, 0xD15B + w);
    std::vector<uint8_t> rw(w * h, 0);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x + static_cast<size_t>(shift) < w; ++x)
            rw[y * w + x] = lw[y * w + x + static_cast<size_t>(shift)];

    bincv::BinMat<uint64_t> lb(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMat<uint64_t> rb(static_cast<int>(w), static_cast<int>(h));
    bincv::packBits<bincv::PackRule::GreaterThan>(lw.data(), w, h, w, lb.view(),
                                                  uint8_t{127});
    bincv::packBits<bincv::PackRule::GreaterThan>(rw.data(), w, h, w, rb.view(),
                                                  uint8_t{127});

    std::vector<uint64_t> sw(bincv::denseDisparityBinaryScratchWords<uint64_t>(w, p));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(w));
    std::vector<uint8_t> expect(w * h, 0xAA);
    bincv::denseDisparityBinary<uint64_t>(lb.constView(), rb.constView(), p, sw.data(),
                                          sw.size(), sr.data(), sr.size(), expect.data(),
                                          w);

    bincv::cuda::DeviceBinMat dl(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceBinMat dr(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::upload(lb.constView(), dl.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(rb.constView(), dr.view()), cudaSuccess);
    bincv::cuda::DeviceImage<uint8_t> dDisp(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(), p,
                                                     dDisp.view()),
                   cudaSuccess);
    std::vector<uint8_t> got(w * h, 0x55);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), got.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t bad = 0;
    for (size_t i = 0; i < w * h; ++i)
        if (expect[i] != got[i]) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
}
} // namespace

BINCV_TEST(CudaDense, Binary_D32_9x9) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    testDenseBinary(160, 120, p, 11);
}

BINCV_TEST(CudaDense, Binary_D64_5x5) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    p.winWidth = 5;
    p.winHeight = 5;
    testDenseBinary(200, 60, p, 21);
}

BINCV_TEST(CudaDense, Binary_MinDisparity) {
    bincv::DenseDisparityParams p;
    p.minDisparity = 5;
    p.maxDisparity = 24;
    testDenseBinary(128, 40, p, 9);
}

BINCV_TEST(CudaDense, Binary_RangeClampedByWidth) {
    // maxDisparity larger than the width can support: dEnd clamps, exactly as
    // the host clamps it.
    bincv::DenseDisparityParams p;
    p.maxDisparity = 200;
    testDenseBinary(48, 32, p, 3);
}

BINCV_TEST(CudaDense, Binary_DegenerateAllInvalid) {
    // Height below the window: every pixel is the invalid marker on both sides.
    bincv::DenseDisparityParams p;
    p.maxDisparity = 16;
    testDenseBinary(64, 7, p, 3);
}

BINCV_TEST(CudaDense, Binary_UnevenWidth) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    p.winWidth = 11;
    p.winHeight = 5;
    testDenseBinary(157, 83, p, 13);
}

// ---------------------------------------------------------------------------
// Dense disparity, census entry: device censusTransform feeding the device
// matcher against the host wide-input path, byte for byte.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDense, CensusEntryMatchesHostWidePath) {
    const size_t w = 160, h = 96;
    constexpr size_t K = 24;
    const auto lw = smoothFrame(w, h, 0xCE2255);
    std::vector<uint8_t> rw(w * h, 0);
    const size_t shift = 9;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x + shift < w; ++x) rw[y * w + x] = lw[y * w + x + shift];

    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;

    std::vector<uint32_t> sw(bincv::denseDisparityScratchWords<K, uint32_t>(w, p));
    std::vector<uint16_t> sr(bincv::denseDisparityScratchRows(w));
    std::vector<uint8_t> expect(w * h, 0xAA);
    bincv::denseDisparity<K, uint8_t, uint32_t>(lw.data(), rw.data(), w, h, w, w,
                                                bincv::kCensus5x5, p, sw.data(),
                                                sw.size(), sr.data(), sr.size(),
                                                expect.data(), w);

    bincv::cuda::DeviceImage<uint8_t> dL(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint8_t> dR(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(lw.data(), w, h, w, dL.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(rw.data(), w, h, w, dR.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat cenL(static_cast<int>(w), static_cast<int>(K * h));
    bincv::cuda::DeviceBinMat cenR(static_cast<int>(w), static_cast<int>(K * h));
    BINCV_CHECK_EQ(bincv::cuda::censusTransform<K>(dL.constView(), bincv::kCensus5x5,
                                                   cenL.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::censusTransform<K>(dR.constView(), bincv::kCensus5x5,
                                                   cenR.view()),
                   cudaSuccess);
    bincv::cuda::DeviceImage<uint8_t> dDisp(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::denseDisparityCensus(cenL.constView(), cenR.constView(),
                                                     K, h, p, dDisp.view()),
                   cudaSuccess);
    std::vector<uint8_t> got(w * h, 0x55);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), got.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t bad = 0;
    for (size_t i = 0; i < w * h; ++i)
        if (expect[i] != got[i]) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
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
    const int summaryRc = ::bincv::test::summarize("CUDA backend tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA backend tests", argc, argv);
}
#endif
