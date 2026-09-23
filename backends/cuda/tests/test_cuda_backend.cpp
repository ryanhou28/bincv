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
#include "bincv/cuda/compaction.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/denseDisparity.hpp"
#include "bincv/cuda/features.hpp"
#include "bincv/cuda/logic.hpp"
#include "bincv/cuda/pack.hpp"
#include "bincv/cuda/reduce.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/quantMat.hpp"
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
// The masked reductions and the covariance: every form against the host, over
// the same rectangle set, including negative origins and empty clips.
// ---------------------------------------------------------------------------
namespace {
const bincv::Rect kRects[] = {
    {0, 0, 200, 90},     {3, 5, 60, 40},   {-10, -10, 50, 50}, {150, 60, 500, 500},
    {31, 0, 34, 90},     {0, 89, 200, 1},  {199, 0, 1, 90},    {5, 5, 0, 10},
    {-300, 4, 20, 20},   {64, 32, 31, 31}, {1, 1, 63, 63},     {170, 70, 40, 40},
};
} // namespace

BINCV_TEST(CudaReduce, MaskedFormsMatchHost) {
    const size_t w = 200, h = 90;
    const auto a = randomBits<uint32_t>(w, h, 0xAA01);
    const auto b = randomBits<uint32_t>(w, h, 0xBB02);
    const auto c0 = randomBits<uint32_t>(w, h, 0xCC03);
    const auto c1 = randomBits<uint32_t>(w, h, 0xDD04);

    bincv::cuda::DeviceBinMat da(w, h), db(w, h), dc0(w, h), dc1(w, h);
    BINCV_CHECK_EQ(bincv::cuda::upload(a.constView(), da.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(b.constView(), db.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(c0.constView(), dc0.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::upload(c1.constView(), dc1.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    for (const auto& r : kRects) {
        // countAnd
        BINCV_CHECK_EQ(bincv::cuda::countAnd(da.constView(), db.constView(), r),
                       bincv::countAnd<uint32_t>(a.constView(), b.constView(), r));

        // countAndSplit, selector-plane form
        const auto hSplit =
            bincv::countAndSplit<uint32_t>(a.constView(), b.constView(), c0.constView(), r);
        const auto dSplit = bincv::cuda::countAndSplit(da.constView(), db.constView(),
                                                       dc0.constView(), r);
        BINCV_CHECK_EQ(dSplit.whenClear, hSplit.whenClear);
        BINCV_CHECK_EQ(dSplit.whenSet, hSplit.whenSet);
        BINCV_CHECK_EQ(dSplit.crossTerm(), hSplit.crossTerm());

        // countAndSplit, the no-plane (c0 ^ c1) form the covariance calls
        const auto hSplitX = bincv::countAndSplit<uint32_t>(
            a.constView(), b.constView(), c0.constView(), c1.constView(), r);
        const auto dSplitX = bincv::cuda::countAndSplit(
            da.constView(), db.constView(), dc0.constView(), dc1.constView(), r);
        BINCV_CHECK_EQ(dSplitX.whenClear, hSplitX.whenClear);
        BINCV_CHECK_EQ(dSplitX.whenSet, hSplitX.whenSet);

        // countCovariance, both forms
        const auto hCov = bincv::countCovariance<uint32_t>(a.constView(), b.constView(),
                                                           c0.constView(), r);
        const auto dCov = bincv::cuda::countCovariance(da.constView(), db.constView(),
                                                       dc0.constView(), r);
        BINCV_CHECK_EQ(dCov.xx, hCov.xx);
        BINCV_CHECK_EQ(dCov.yy, hCov.yy);
        BINCV_CHECK_EQ(dCov.xy.whenClear, hCov.xy.whenClear);
        BINCV_CHECK_EQ(dCov.xy.whenSet, hCov.xy.whenSet);
        BINCV_CHECK_EQ(dCov.crossTerm(), hCov.crossTerm());

        const auto hCovX = bincv::countCovariance<uint32_t>(
            a.constView(), b.constView(), c0.constView(), c1.constView(), r);
        const auto dCovX = bincv::cuda::countCovariance(
            da.constView(), db.constView(), dc0.constView(), dc1.constView(), r);
        BINCV_CHECK_EQ(dCovX.xx, hCovX.xx);
        BINCV_CHECK_EQ(dCovX.yy, hCovX.yy);
        BINCV_CHECK_EQ(dCovX.crossTerm(), hCovX.crossTerm());
    }
}

// The batched form is the entry point a tracker uses, so it is held to BOTH
// the host and the single-region device form -- a batch that agreed with
// neither would be a plausible-looking wrong answer.
BINCV_TEST(CudaReduce, BatchMatchesHostAndSingleRegion) {
    const size_t w = 200, h = 90;
    const auto a = randomBits<uint32_t>(w, h, 0x1A01);
    const auto b = randomBits<uint32_t>(w, h, 0x2B02);
    const auto c0 = randomBits<uint32_t>(w, h, 0x3C03);
    const auto c1 = randomBits<uint32_t>(w, h, 0x4D04);

    bincv::cuda::DeviceBinMat da(w, h), db(w, h), dc0(w, h), dc1(w, h);
    bincv::cuda::upload(a.constView(), da.view());
    bincv::cuda::upload(b.constView(), db.view());
    bincv::cuda::upload(c0.constView(), dc0.view());
    bincv::cuda::upload(c1.constView(), dc1.view());
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const size_t n = sizeof(kRects) / sizeof(kRects[0]);
    bincv::Rect* dRegions = nullptr;
    bincv::cuda::DeviceCovarianceCount* dOut = nullptr;
    BINCV_CHECK_EQ(cudaMalloc(&dRegions, n * sizeof(bincv::Rect)), cudaSuccess);
    BINCV_CHECK_EQ(cudaMalloc(&dOut, n * sizeof(bincv::cuda::DeviceCovarianceCount)),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(dRegions, kRects, n * sizeof(bincv::Rect),
                              cudaMemcpyHostToDevice),
                   cudaSuccess);

    // The XOR form.
    BINCV_CHECK_EQ(bincv::cuda::countCovarianceBatchAsync(da.constView(), db.constView(),
                                                          dc0.constView(),
                                                          dc1.constView(), dRegions, n,
                                                          dOut),
                   cudaSuccess);
    std::vector<bincv::cuda::DeviceCovarianceCount> got(n);
    BINCV_CHECK_EQ(cudaMemcpy(got.data(), dOut,
                              n * sizeof(bincv::cuda::DeviceCovarianceCount),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t badVsHost = 0, badVsSingle = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto host = bincv::countCovariance<uint32_t>(
            a.constView(), b.constView(), c0.constView(), c1.constView(), kRects[i]);
        const auto batch = bincv::cuda::toHost(got[i]);
        if (batch.xx != host.xx || batch.yy != host.yy ||
            batch.xy.whenClear != host.xy.whenClear ||
            batch.xy.whenSet != host.xy.whenSet)
            ++badVsHost;
        const auto single = bincv::cuda::countCovariance(
            da.constView(), db.constView(), dc0.constView(), dc1.constView(), kRects[i]);
        if (batch.xx != single.xx || batch.yy != single.yy ||
            batch.crossTerm() != single.crossTerm())
            ++badVsSingle;
    }
    BINCV_CHECK_EQ(badVsHost, 0u);
    BINCV_CHECK_EQ(badVsSingle, 0u);

    // The selector-plane form of the batch.
    BINCV_CHECK_EQ(bincv::cuda::countCovarianceBatchAsync(da.constView(), db.constView(),
                                                          dc0.constView(), dRegions, n,
                                                          dOut),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaMemcpy(got.data(), dOut,
                              n * sizeof(bincv::cuda::DeviceCovarianceCount),
                              cudaMemcpyDeviceToHost),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    size_t badPlaneForm = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto host = bincv::countCovariance<uint32_t>(
            a.constView(), b.constView(), c0.constView(), kRects[i]);
        const auto batch = bincv::cuda::toHost(got[i]);
        if (batch.xx != host.xx || batch.yy != host.yy ||
            batch.crossTerm() != host.crossTerm())
            ++badPlaneForm;
    }
    BINCV_CHECK_EQ(badPlaneForm, 0u);

    // A batch of one and a batch of zero: the degenerate counts a tracker
    // reaches on its first and last frames.
    BINCV_CHECK_EQ(bincv::cuda::countCovarianceBatchAsync(da.constView(), db.constView(),
                                                          dc0.constView(),
                                                          dc1.constView(), dRegions, 1,
                                                          dOut),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::countCovarianceBatchAsync(da.constView(), db.constView(),
                                                          dc0.constView(),
                                                          dc1.constView(), dRegions, 0,
                                                          dOut),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaFree(dRegions);
    cudaFree(dOut);
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

// packQuant: the N-bit ingestion path, every supported depth, against the host
// packer -- which is what pins "the device computes the same integer
// expression" rather than leaving it as a claim in a comment.
namespace {
template <size_t N, typename SrcT>
void testPackQuant() {
    const size_t w = 133, h = 41;
    const auto frame = randomFrame<SrcT>(w, h, 0x9000 + N);

    bincv::BinMatView<uint32_t> views[N];
    std::vector<bincv::BinMat<uint32_t>> planes;
    planes.reserve(N);
    for (size_t p = 0; p < N; ++p)
        planes.emplace_back(static_cast<int>(w), static_cast<int>(h));
    for (size_t p = 0; p < N; ++p) views[p] = planes[p].view();
    bincv::packQuant<bincv::QuantRule::Scale, N, SrcT, uint32_t>(frame.data(), w, h, w,
                                                                 views);

    bincv::cuda::DeviceImage<SrcT> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<SrcT>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::packQuant(dImg.constView(), dBlock.view(), N),
                   cudaSuccess);
    bincv::BinMat<uint32_t> gotBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::download(dBlock.constView(), gotBlock.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const size_t words = bincv::impl::minRowWords<uint32_t>(w);
    size_t badPlanes = 0;
    for (size_t p = 0; p < N; ++p) {
        size_t bad = 0;
        for (size_t y = 0; y < h; ++y) {
            const uint32_t* x = planes[p].constView().row(y);
            const uint32_t* g = gotBlock.constView().row(p * h + y);
            for (size_t i = 0; i < words; ++i)
                if (x[i] != g[i]) ++bad;
        }
        if (bad != 0) ++badPlanes;
    }
    BINCV_CHECK_EQ(badPlanes, 0u);
}
} // namespace

BINCV_TEST(CudaPack, Quant_N1_uint8_t) { testPackQuant<1, uint8_t>(); }
BINCV_TEST(CudaPack, Quant_N2_uint8_t) { testPackQuant<2, uint8_t>(); }
BINCV_TEST(CudaPack, Quant_N4_uint8_t) { testPackQuant<4, uint8_t>(); }
BINCV_TEST(CudaPack, Quant_N8_uint8_t) { testPackQuant<8, uint8_t>(); }
BINCV_TEST(CudaPack, Quant_N2_uint16_t) { testPackQuant<2, uint16_t>(); }
BINCV_TEST(CudaPack, Quant_N5_uint16_t) { testPackQuant<5, uint16_t>(); }

// packRows: a chunked fill must be identical to the whole-frame one, which is
// the property that makes banded packing exact rather than approximate.
BINCV_TEST(CudaPack, RowsChunkedEqualsWhole) {
    const size_t w = 197, h = 60;
    const auto frame = randomFrame<uint8_t>(w, h, 0x7070);
    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view());

    bincv::cuda::DeviceBinMat whole(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::packBits(dImg.constView(), whole.view(),
                                         bincv::PackRule::GreaterThan, uint8_t{127}),
                   cudaSuccess);

    // The same frame in three bands, through packRows.
    bincv::cuda::DeviceBinMat banded(static_cast<int>(w), static_cast<int>(h));
    const size_t bands[][2] = {{0, 17}, {17, 23}, {40, 20}};
    for (const auto& band : bands) {
        bincv::cuda::DeviceImageConstView<uint8_t> chunk(
            dImg.constView().ptr + band[0] * dImg.getStride(), w, band[1],
            dImg.getStride());
        BINCV_CHECK_EQ(bincv::cuda::packRows(chunk, banded.view(), band[0],
                                             bincv::PackRule::GreaterThan, uint8_t{127}),
                       cudaSuccess);
    }
    bincv::BinMat<uint32_t> wholeHost(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMat<uint32_t> bandedHost(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::download(whole.constView(), wholeHost.view());
    bincv::cuda::download(banded.constView(), bandedHost.view());
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BINCV_CHECK_EQ(mismatchWords<uint32_t>(wholeHost, bandedHost), 0u);
}

// unpackTo8Bit: the reverse, against the host's.
BINCV_TEST(CudaPack, UnpackMatchesHost) {
    const size_t w = 157, h = 33;
    const auto bits = randomBits<uint32_t>(w, h, 0x5A5A);
    std::vector<uint8_t> expect(w * h, 0xCC);
    bincv::unpackTo8Bit<uint32_t>(bits.constView(), expect.data(), w, 255, 0);

    bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::upload(bits.constView(), dBits.view());
    bincv::cuda::DeviceImage<uint8_t> dOut(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::unpackTo8Bit(dBits.constView(), dOut.view(), 255, 0),
                   cudaSuccess);
    std::vector<uint8_t> got(w * h, 0x33);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dOut.constView(), got.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    size_t bad = 0;
    for (size_t i = 0; i < w * h; ++i)
        if (expect[i] != got[i]) ++bad;
    BINCV_CHECK_EQ(bad, 0u);

    // A non-default on/off pair, since those are separate parameters.
    std::vector<uint8_t> expect2(w * h, 0);
    bincv::unpackTo8Bit<uint32_t>(bits.constView(), expect2.data(), w, 7, 3);
    BINCV_CHECK_EQ(bincv::cuda::unpackTo8Bit(dBits.constView(), dOut.view(), 7, 3),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dOut.constView(), got.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    bad = 0;
    for (size_t i = 0; i < w * h; ++i)
        if (expect2[i] != got[i]) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
}

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
    for (const bool tiled : {true, false}) {
        bincv::cuda::impl::censusTiledEnabled() = tiled;
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
    bincv::cuda::impl::censusTiledEnabled() = true;
}
} // namespace

BINCV_TEST(CudaCensus, Census5x5_uint8_t) { testCensus<24, uint8_t>(bincv::kCensus5x5); }
BINCV_TEST(CudaCensus, Census3x3_uint8_t) { testCensus<8, uint8_t>(bincv::kCensus3x3); }
BINCV_TEST(CudaCensus, Census5x5_uint16_t) { testCensus<24, uint16_t>(bincv::kCensus5x5); }

namespace {
// Thirty-two distinct offsets -- the pattern POD's whole capacity, so every slot
// the kernels unroll over is read, and a slot that picked up its neighbour's
// offset would compare against the wrong pixel. A 7x5 neighbourhood less its
// centre and the two bottom corners.
constexpr bincv::CensusPattern<32> kCensusFullPod = {{
    {-3, -2}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {3, -2},
    {-3, -1}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, {3, -1},
    {-3, 0},  {-2, 0},  {-1, 0},  {1, 0},  {2, 0},  {3, 0},
    {-3, 1},  {-2, 1},  {-1, 1},  {0, 1},  {1, 1},  {2, 1},  {3, 1},
    {-2, 2},  {-1, 2},  {0, 2},   {1, 2},  {2, 2},
}};

/// Bit k of every packed descriptor against plane k of the host transform. The
/// dense suite reaches the packed layout only through a disparity map, which a
/// permutation of the bits cannot change; this holds each bit to its own plane.
template <size_t K>
void testCensusPackedBits(const bincv::CensusPattern<K>& pattern) {
    const size_t w = 101, h = 37;
    const auto frame = randomFrame<uint8_t>(w, h, 0xCE9A00 + K);
    std::vector<bincv::BinMat<uint32_t>> planes;
    std::vector<bincv::BinMatView<uint32_t>> views;
    planes.reserve(K);
    for (size_t k = 0; k < K; ++k)
        planes.emplace_back(static_cast<int>(w), static_cast<int>(h));
    for (size_t k = 0; k < K; ++k) views.push_back(planes[k].view());
    bincv::censusTransform<K, uint8_t, uint32_t>(frame.data(), w, h, w, pattern,
                                                 views.data());

    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceImage<uint32_t> dDesc(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::censusTransformPacked<K>(dImg.constView(), pattern,
                                                         dDesc.view()),
                   cudaSuccess);
    std::vector<uint32_t> desc(w * h, 0xDEADBEEFu);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint32_t>(dDesc.constView(), desc.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t bad = 0;
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            uint32_t expect = 0;
            for (size_t k = 0; k < K; ++k)
                expect |= ((planes[k].constView().row(y)[x / 32] >> (x % 32)) & 1u) << k;
            if (desc[y * w + x] != expect) ++bad;
        }
    }
    BINCV_CHECK_EQ(bad, 0u);
}
} // namespace

BINCV_TEST(CudaCensus, FullPodPattern_uint8_t) {
    testCensus<32, uint8_t>(kCensusFullPod);
}
BINCV_TEST(CudaCensus, FullPodPattern_uint16_t) {
    testCensus<32, uint16_t>(kCensusFullPod);
}
BINCV_TEST(CudaCensus, PackedBitsArePlanes) {
    testCensusPackedBits<24>(bincv::kCensus5x5);
    testCensusPackedBits<32>(kCensusFullPod);
}

// ---------------------------------------------------------------------------
// Dense disparity, binary entry: the device map against the host map, byte for
// byte, across parameter shapes including the degenerate ones.
// ---------------------------------------------------------------------------
namespace {
/// @brief Breaks the exact-shift relationship between a stereo pair.
///
/// A right image built as an EXACT shift of the left makes the correct
/// disparity's window cost zero and every other candidate's cost hundreds, so
/// the argmin is insensitive to window-sum errors of a few hundred. A suite
/// built only that way passes a matcher whose window aggregation is wrong: the
/// census box matcher's first suite did exactly that, accepting a halo one lane
/// too wide on every shape with a byte-identical map. Perturbing the right
/// image leaves a correct matcher's answer defined by the host and makes a
/// wrong window sum able to change it.
///
/// Deterministic and content-dependent, so the two backends see one image.
void breakExactShift(std::vector<uint8_t>& rw, size_t w, size_t h, uint32_t seed) {
    uint32_t st = seed | 1u;
    for (size_t i = 0; i < w * h; ++i) {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        const int delta = static_cast<int>(st % 61u) - 30;
        const int v = static_cast<int>(rw[i]) + delta;
        rw[i] = static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

void testDenseBinary(size_t w, size_t h, const bincv::DenseDisparityParams& p, int shift,
                     bool perturb = false) {
    const auto lw = smoothFrame(w, h, 0xD15B + w);
    std::vector<uint8_t> rw(w * h, 0);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x + static_cast<size_t>(shift) < w; ++x)
            rw[y * w + x] = lw[y * w + x + static_cast<size_t>(shift)];
    if (perturb) breakExactShift(rw, w, h, 0x5EED0001u + static_cast<uint32_t>(w));

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

    // All three device arms answer to the same host map: the word-parallel
    // bit-sliced arm the launcher prefers, the per-pixel sliding arm behind the
    // first switch, and the straightforward reference kernel behind the second.
    for (int arm = 0; arm < 3; ++arm) {
        bincv::cuda::impl::denseFastArmEnabled() = arm < 2;
        bincv::cuda::impl::denseBitSlicedEnabled() = arm == 0;
        BINCV_CHECK_EQ(bincv::cuda::denseDisparityBinary(dl.constView(), dr.constView(),
                                                         p, dDisp.view()),
                       cudaSuccess);
        std::vector<uint8_t> got(w * h, 0x55);
        BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), got.data(),
                                                           w),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        size_t bad = 0;
        for (size_t i = 0; i < w * h; ++i)
            if (expect[i] != got[i]) ++bad;
        BINCV_CHECK_EQ(bad, 0u);
    }
    bincv::cuda::impl::denseFastArmEnabled() = true;
    bincv::cuda::impl::denseBitSlicedEnabled() = true;
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

// The same shapes on content that is NOT an exact shift. This is what makes the
// byte-for-byte comparison above able to see a window-aggregation error at all;
// see breakExactShift.
BINCV_TEST(CudaDense, Binary_D32_9x9_NotAnExactShift) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 32;
    testDenseBinary(160, 120, p, 11, true);
}

BINCV_TEST(CudaDense, Binary_D64_5x5_NotAnExactShift) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    p.winWidth = 5;
    p.winHeight = 5;
    testDenseBinary(200, 60, p, 21, true);
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

// A word-parallel kernel makes the row's TAIL WORD dangerous in a way the
// per-pixel arms are not: its padding lanes past `width` take part in the same
// arithmetic as the real ones, and a shifted read runs off the end of the row.
// These two widths put the last anchor inside a partly-padded word -- 97 leaves
// one pixel in the tail word, 67 leaves three -- so a padding lane that is not
// zero, or a neighbouring word read past the row, changes the map.
BINCV_TEST(CudaDense, Binary_TailWordOnePixel) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 24;
    testDenseBinary(97, 40, p, 7);
}

// The reference frame and search the benchmark prices, held to the host map
// here: a speed claim about a configuration nothing checks is a claim about an
// unverified kernel.
BINCV_TEST(CudaDense, Binary_ReferenceFrame) {
    bincv::DenseDisparityParams p;
    p.maxDisparity = 64;
    testDenseBinary(752, 480, p, 21);
}

BINCV_TEST(CudaDense, Binary_TailWordThreePixels) {
    bincv::DenseDisparityParams p;
    p.minDisparity = 2;
    p.maxDisparity = 20;
    p.winWidth = 7;
    p.winHeight = 7;
    testDenseBinary(67, 29, p, 5);
}

// ---------------------------------------------------------------------------
// Dense disparity, census entry: device censusTransform feeding the device
// matcher against the host wide-input path, byte for byte.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaDense, CensusEntryMatchesHostWidePath) {
    const size_t w = 160, h = 96;
    constexpr size_t K = 24;
    // Two contents: an exact shift, and one that is not. Only the second can see
    // a window-aggregation error at all -- see breakExactShift.
    for (int perturb = 0; perturb < 2; ++perturb) {
    const auto lw = smoothFrame(w, h, 0xCE2255);
    std::vector<uint8_t> rw(w * h, 0);
    const size_t shift = 9;
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x + shift < w; ++x) rw[y * w + x] = lw[y * w + x + shift];
    if (perturb) breakExactShift(rw, w, h, 0x5EED0002u);

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
    for (const bool fastArm : {true, false}) {
        bincv::cuda::impl::denseFastArmEnabled() = fastArm;
        BINCV_CHECK_EQ(bincv::cuda::denseDisparityCensus(cenL.constView(),
                                                         cenR.constView(), K, h, p,
                                                         dDisp.view()),
                       cudaSuccess);
        std::vector<uint8_t> got(w * h, 0x55);
        BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), got.data(),
                                                           w),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        size_t bad = 0;
        for (size_t i = 0; i < w * h; ++i)
            if (expect[i] != got[i]) ++bad;
        BINCV_CHECK_EQ(bad, 0u);
    }
    bincv::cuda::impl::denseFastArmEnabled() = true;
    }
}

// The packed-descriptor census path: a different intermediate LAYOUT for the
// same comparisons, so it must land on the same map as the host wide path and
// as the plane-block device path. Hamming distance is invariant under a
// permutation of a descriptor's bits, and this is what holds that to account.
BINCV_TEST(CudaDense, PackedCensusMatchesHostAndPlaneForm) {
    constexpr size_t K = 24;
    // Third column: whether the right image stays an EXACT shift of the left.
    // Every shape runs both ways, because only the perturbed content can see a
    // window-aggregation error at all -- see breakExactShift.
    const size_t sizes[][3] = {{160, 96, 0}, {131, 47, 0}, {160, 96, 1}, {131, 47, 1}};
    for (const auto& sz : sizes) {
        const size_t w = sz[0], h = sz[1];
        const auto lw = smoothFrame(w, h, 0xACE1 + w);
        std::vector<uint8_t> rw(w * h, 0);
        const size_t shift = 9;
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x + shift < w; ++x) rw[y * w + x] = lw[y * w + x + shift];
        if (sz[2]) breakExactShift(rw, w, h, 0x5EED0003u + static_cast<uint32_t>(w));

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
        bincv::cuda::uploadImage<uint8_t>(lw.data(), w, h, w, dL.view());
        bincv::cuda::uploadImage<uint8_t>(rw.data(), w, h, w, dR.view());
        bincv::cuda::DeviceImage<uint32_t> descL(static_cast<int>(w),
                                                 static_cast<int>(h));
        bincv::cuda::DeviceImage<uint32_t> descR(static_cast<int>(w),
                                                 static_cast<int>(h));
        BINCV_CHECK_EQ(bincv::cuda::censusTransformPacked<K>(dL.constView(),
                                                             bincv::kCensus5x5,
                                                             descL.view()),
                       cudaSuccess);
        BINCV_CHECK_EQ(bincv::cuda::censusTransformPacked<K>(dR.constView(),
                                                             bincv::kCensus5x5,
                                                             descR.view()),
                       cudaSuccess);
        bincv::cuda::DeviceImage<uint8_t> dDisp(static_cast<int>(w),
                                                static_cast<int>(h));
        BINCV_CHECK_EQ(bincv::cuda::denseDisparityCensusPacked(descL.constView(),
                                                               descR.constView(), p,
                                                               dDisp.view()),
                       cudaSuccess);
        std::vector<uint8_t> got(w * h, 0x55);
        BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDisp.constView(), got.data(),
                                                           w),
                       cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        size_t bad = 0;
        for (size_t i = 0; i < w * h; ++i)
            if (expect[i] != got[i]) ++bad;
        BINCV_CHECK_EQ(bad, 0u);
    }
}

// ---------------------------------------------------------------------------
// THE N-BIT PLANE BLOCK ADDRESSES THE HOST'S WORDS AND NO OTHERS
//
// QuantMat<N, WordType> is one BinMat of N*height rows: plane p is rows
// [p*height, (p+1)*height), and plane(p) hands back data() + p * planeWords()
// with planeWords() == height * alignedWidth. A device plane block over the
// same geometry must name the same word offsets. This is swept rather than
// argued because a plane offset that drifts does not crash -- it returns a
// fully-formed view of the WRONG plane, which reads as a correct answer.
// ---------------------------------------------------------------------------
namespace {
template <size_t N>
size_t planeBlockOffsetMismatches(size_t w, size_t h, size_t rowAlignment) {
    bincv::QuantMat<N, uint32_t> host(static_cast<int>(w), static_cast<int>(h),
                                      rowAlignment);
    // The device view is built over the HOST pointer on purpose: this case is
    // about address arithmetic, and pointing it at a device allocation would
    // only add a transfer to a comparison of offsets.
    bincv::cuda::DevicePlaneBlockView dev(host.data(), host.getWidth(), host.getHeight(),
                                          host.getAlignedWidth(), N);

    size_t bad = 0;
    if (dev.planeWords() != host.planeWords()) ++bad;
    if (dev.block().ptr != host.data()) ++bad;
    if (dev.block().height != N * host.getHeight()) ++bad;
    if (dev.block().stride != host.getAlignedWidth()) ++bad;
    if (dev.planes != N) ++bad;
    for (size_t p = 0; p < N; ++p) {
        const bincv::BinMatConstView<uint32_t> hostPlane = host.constPlane(p);
        if (dev.planeData(p) != hostPlane.ptr) ++bad;
        if (dev.plane(p).ptr != hostPlane.ptr) ++bad;
        if (dev.plane(p).width != hostPlane.width) ++bad;
        if (dev.plane(p).height != hostPlane.height) ++bad;
        if (dev.plane(p).stride != hostPlane.stride) ++bad;
        for (size_t y = 0; y < host.getHeight(); ++y)
            if (dev.row(p, y) != hostPlane.row(y)) ++bad;
    }
    // The const twin must name the same words; a read-only kernel that read a
    // different plane than the writing one would be the worst version of this.
    const bincv::cuda::DevicePlaneBlockConstView cdev = dev;
    for (size_t p = 0; p < N; ++p)
        if (cdev.planeData(p) != dev.planeData(p)) ++bad;
    return bad;
}
} // namespace

BINCV_TEST(CudaPlaneBlock, MirrorsHostQuantMatLayout) {
    // Tight rows, aligned rows, a width that ends mid-word, and a single row --
    // the stride cases that make p * height * stride disagree with anything else.
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<1>(64, 16, 4), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<2>(65, 17, 4), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<3>(133, 41, 4), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<4>(1, 1, 4), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<5>(97, 3, 32), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<8>(640, 480, 4), 0u);
    BINCV_CHECK_EQ(planeBlockOffsetMismatches<8>(31, 5, 128), 0u);
}

// The same claim end to end: the device packer writes the plane block, the
// plane block view is asked for each plane, and what comes back is the host
// container's plane. Offsets agreeing is one thing; the bytes agreeing after a
// real kernel wrote them is the claim that matters.
namespace {
template <size_t N>
void testPlaneBlockAgainstHostQuantMat(size_t w, size_t h) {
    const auto frame = randomFrame<uint8_t>(w, h, 0xB10C + N);

    bincv::QuantMat<N, uint32_t> host(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMatView<uint32_t> planes[N];
    for (size_t p = 0; p < N; ++p) planes[p] = host.plane(p);
    bincv::packQuant<bincv::QuantRule::Scale, N, uint8_t, uint32_t>(frame.data(), w, h, w,
                                                                    planes);

    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::packQuant(dImg.constView(), dBlock.view(), N),
                   cudaSuccess);

    const bincv::cuda::DevicePlaneBlockView block =
        bincv::cuda::planeBlock(dBlock.view(), N);
    BINCV_CHECK_EQ(block.height, h);
    BINCV_CHECK_EQ(block.planes, N);

    size_t badPlanes = 0;
    for (size_t p = 0; p < N; ++p) {
        bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
        // Downloading THROUGH plane(p) is the point: a wrong plane offset lands
        // here as the neighbouring plane's bits, not as an error.
        BINCV_CHECK_EQ(bincv::cuda::download(block.plane(p), got.view()), cudaSuccess);
        BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const size_t words = bincv::impl::minRowWords<uint32_t>(w);
        size_t bad = 0;
        for (size_t y = 0; y < h; ++y) {
            const uint32_t* a = host.constPlane(p).row(y);
            const uint32_t* b = got.constView().row(y);
            for (size_t i = 0; i < words; ++i)
                if (a[i] != b[i]) ++bad;
        }
        if (bad != 0) ++badPlanes;
    }
    BINCV_CHECK_EQ(badPlanes, 0u);
}
} // namespace

BINCV_TEST(CudaPlaneBlock, PlanesHoldDevicePackQuantOutput_N2) {
    testPlaneBlockAgainstHostQuantMat<2>(133, 41);
}
BINCV_TEST(CudaPlaneBlock, PlanesHoldDevicePackQuantOutput_N5) {
    testPlaneBlockAgainstHostQuantMat<5>(97, 23);
}

// ---------------------------------------------------------------------------
// THE RESULT PODs AND THE HOST TYPES THEY CONVERT TO
//
// ONE of them is byte-identical to the host's, which is what makes its download
// a raw copy; the static_asserts in features.hpp hold the layout and this case
// holds the VALUES, including through a reinterpretation of the bytes. The other
// three narrow a field under the documented domain, so the conversion is the
// thing under test.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaFeatures, CornerPodsAreTheHostBytes) {
    // DeviceFastCorner is 12 bytes where the host type is 16: the score narrows to
    // int32 and `toHost` widens it. What is checked here is the whole width of the
    // narrowed field, because a record that is right for a score of 13 and wrong at
    // the field's edge is a record that fails on the one frame nobody tested.
    // The values a DETECTOR can actually emit are swept exhaustively in the FAST
    // suite; this is the type's own range.
    BINCV_CHECK_EQ(sizeof(bincv::cuda::DeviceFastCorner), size_t{12});
    bincv::cuda::DeviceFastCorner df;
    df.x = 41;
    df.y = -7;
    df.score = 13;
    const bincv::FastCorner hf = df.toHost();
    BINCV_CHECK_EQ(hf.x, 41);
    BINCV_CHECK_EQ(hf.y, -7);
    BINCV_CHECK_EQ(hf.score, 13);
    const int edges[] = {0, 1, -1, 16, 2147483647, -2147483647 - 1};
    size_t badScores = 0;
    for (int e : edges) {
        bincv::cuda::DeviceFastCorner d;
        d.x = e;
        d.y = e;
        d.score = e;
        const bincv::FastCorner h = d.toHost();
        if (h.score != static_cast<long long>(e)) ++badScores;
        if (h.x != e || h.y != e) ++badScores;
    }
    BINCV_CHECK_EQ(badScores, size_t{0});

    bincv::cuda::DeviceCorner dc;
    dc.x = 5;
    dc.y = 600;
    dc.response = 0.25f;
    const bincv::Corner hc = dc.toHost();
    BINCV_CHECK_EQ(hc.x, 5);
    BINCV_CHECK_EQ(hc.y, 600);
    BINCV_CHECK_EQ(hc.response, 0.25f);
    bincv::Corner rawC{};
    std::memcpy(static_cast<void*>(&rawC), &dc, sizeof(rawC));
    BINCV_CHECK_EQ(rawC.x, hc.x);
    BINCV_CHECK_EQ(rawC.y, hc.y);
    BINCV_CHECK_EQ(rawC.response, hc.response);
}

BINCV_TEST(CudaFeatures, MatchPodsWidenTheirNarrowedIndex) {
    bincv::cuda::DeviceDescriptorMatch dm;
    dm.trainIndex = 4000000000u;  // a full-range uint32 index, not a small one
    dm.distance = 31;
    dm.secondDistance = 90;
    dm.valid = 1;
    const bincv::DescriptorMatch hm = dm.toHost();
    BINCV_CHECK_EQ(hm.trainIndex, size_t{4000000000u});
    BINCV_CHECK_EQ(hm.distance, 31u);
    BINCV_CHECK_EQ(hm.secondDistance, 90u);
    BINCV_CHECK(hm.valid);

    bincv::cuda::DeviceStereoMatch ds;
    ds.disparity = -3.5f;
    ds.distance = 77;
    ds.rightIndex = 4000000000u;
    ds.valid = 0;
    const bincv::StereoMatch hs = ds.toHost();
    BINCV_CHECK_EQ(hs.disparity, -3.5f);
    BINCV_CHECK_EQ(hs.distance, 77u);
    BINCV_CHECK_EQ(hs.rightIndex, size_t{4000000000u});
    BINCV_CHECK_EQ(hs.valid, uint8_t{0});

    // The batched spelling, which is the only one the families produce.
    bincv::cuda::DeviceStereoMatch batch[3];
    for (uint32_t i = 0; i < 3; ++i) {
        batch[i].disparity = static_cast<float>(i);
        batch[i].distance = i;
        batch[i].rightIndex = i * 10u;
        batch[i].valid = 1;
    }
    bincv::StereoMatch hostBatch[3];
    bincv::cuda::toHost(batch, 3, hostBatch);
    size_t bad = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (hostBatch[i].rightIndex != i * 10u) ++bad;
        if (hostBatch[i].disparity != static_cast<float>(i)) ++bad;
        if (hostBatch[i].valid != 1) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
}

// ---------------------------------------------------------------------------
// THE SET VIEWS ADDRESS THE HOST FAMILY'S OWN ARRAYS
//
// computeBrief writes `out + k * words` and matchDescriptors reads
// `query + q * words`; the keypoint arrays are interleaved (x, y). The device
// views index the same way, so an upload is a raw copy -- and a descriptor
// computed at ANY host word width is the same bytes, which is what lets the
// uint32-only device accept all four.
// ---------------------------------------------------------------------------
BINCV_TEST(CudaFeatures, SetViewsMatchHostRawArrayLayout) {
    constexpr size_t kBits = 256;
    const size_t count = 37;
    const size_t words = bincv::descriptorWords<kBits, uint32_t>();
    BINCV_CHECK_EQ(bincv::cuda::descriptorWords<kBits>(), static_cast<uint32_t>(words));

    std::vector<float> xy(2 * count);
    std::vector<int32_t> octave(count);
    for (size_t i = 0; i < count; ++i) {
        xy[2 * i] = static_cast<float>(i) + 0.5f;
        xy[2 * i + 1] = static_cast<float>(i) * 2.0f;
        octave[i] = static_cast<int32_t>(i % 4);
    }
    const bincv::cuda::DeviceKeypointSetConstView kps =
        bincv::cuda::keypointSet(xy.data(), count, octave.data());
    size_t bad = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (kps.x(i) != xy[2 * i]) ++bad;
        if (kps.y(i) != xy[2 * i + 1]) ++bad;
    }
    BINCV_CHECK_EQ(bad, 0u);
    BINCV_CHECK_EQ(kps.count, static_cast<uint32_t>(count));
    BINCV_CHECK(kps.hasOctave());
    BINCV_CHECK(!bincv::cuda::keypointSet(xy.data(), count).hasOctave());

    std::vector<uint32_t> desc(count * words, 0u);
    const bincv::cuda::DeviceDescriptorSetConstView set =
        bincv::cuda::descriptorSet(static_cast<const uint32_t*>(desc.data()), count, words);
    size_t badPitch = 0;
    for (uint32_t i = 0; i < count; ++i)
        if (set.descriptor(i) != desc.data() + static_cast<size_t>(i) * words) ++badPitch;
    BINCV_CHECK_EQ(badPitch, 0u);
    BINCV_CHECK_EQ(set.sizeInWords(), count * words);

    // The word-width claim: BRIEF at uint64 host words is byte-identical to
    // BRIEF at uint32, so the device's uint32-only set accepts either.
    const size_t w = 64, h = 64;
    const auto frame = randomFrame<uint8_t>(w, h, 0xD35C);
    bincv::BriefPattern<kBits> pattern;
    bincv::makeBriefPattern<kBits>(pattern);
    std::vector<float> kxy(2 * count);
    for (size_t i = 0; i < count; ++i) {
        kxy[2 * i] = static_cast<float>(20 + (i % 20));
        kxy[2 * i + 1] = static_cast<float>(20 + (i % 17));
    }
    std::vector<uint32_t> at32(count * words);
    std::vector<uint64_t> at64(count * (kBits / 64));
    bincv::computeBrief<kBits, uint8_t, uint32_t>(frame.data(), w, h, w, kxy.data(), count,
                                                  pattern, at32.data());
    bincv::computeBrief<kBits, uint8_t, uint64_t>(frame.data(), w, h, w, kxy.data(), count,
                                                  pattern, at64.data());
    BINCV_CHECK_EQ(std::memcmp(at32.data(), at64.data(), count * kBits / 8), 0);
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
