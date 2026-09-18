// The custom-predicate packers, which only a CUDA translation unit can
// instantiate -- so this suite IS one, and its existence is the proof that
// packCustom.cuh's contract is usable rather than merely documented.
//
// Held to the host's packBitsIf / packQuantWith on the same frame with the
// same rule, which is the only way "arbitrary rule, same answer" is a claim
// rather than a hope.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/packCustom.cuh"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/pack.hpp"
#include "test_util.hpp"

namespace {

uint64_t splitmix(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::vector<uint8_t> randomFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<uint8_t> img(w * h);
    for (auto& v : img) v = static_cast<uint8_t>(splitmix(seed));
    return img;
}

/// A rule none of the shipped enums expresses: a band, not a threshold. Both
/// halves are the same functor type so the two sides cannot drift.
struct BandRule {
    uint8_t lo, hi;
    __host__ __device__ bool operator()(uint8_t v) const { return v >= lo && v <= hi; }
};

/// A non-monotonic map, likewise beyond QuantRule::Scale.
struct FoldMap {
    __host__ __device__ unsigned operator()(uint8_t v) const {
        return static_cast<unsigned>(v ^ static_cast<uint8_t>(v >> 3)) & 0x7u;
    }
};

size_t mismatchWords(const bincv::BinMat<uint32_t>& a, const bincv::BinMat<uint32_t>& b) {
    size_t bad = 0;
    const size_t words = bincv::impl::minRowWords<uint32_t>(a.getWidth());
    for (size_t y = 0; y < a.getHeight(); ++y) {
        const uint32_t* ra = a.constView().row(y);
        const uint32_t* rb = b.constView().row(y);
        for (size_t i = 0; i < words; ++i)
            if (ra[i] != rb[i]) ++bad;
    }
    return bad;
}

} // namespace

BINCV_TEST(CudaPackCustom, PackBitsIfMatchesHost) {
    const size_t w = 171, h = 29;
    const auto frame = randomFrame(w, h, 0xFACE01);
    const BandRule rule{60, 190};

    bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
    bincv::packBitsIf<uint8_t, uint32_t>(frame.data(), w, h, w, expect.view(), rule);

    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBits(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::packBitsIf(dImg.constView(), dBits.view(), rule),
                   cudaSuccess);
    bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::download(dBits.constView(), got.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BINCV_CHECK_EQ(mismatchWords(expect, got), 0u);
}

BINCV_TEST(CudaPackCustom, PackQuantWithMatchesHost) {
    const size_t w = 149, h = 31;
    constexpr size_t N = 3;
    const auto frame = randomFrame(w, h, 0xFACE02);
    const FoldMap map;

    std::vector<bincv::BinMat<uint32_t>> planes;
    planes.reserve(N);
    for (size_t p = 0; p < N; ++p)
        planes.emplace_back(static_cast<int>(w), static_cast<int>(h));
    bincv::BinMatView<uint32_t> views[N];
    for (size_t p = 0; p < N; ++p) views[p] = planes[p].view();
    bincv::packQuantWith<N, uint8_t, uint32_t>(frame.data(), w, h, w, views, map);

    bincv::cuda::DeviceImage<uint8_t> dImg(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dImg.view()),
                   cudaSuccess);
    bincv::cuda::DeviceBinMat dBlock(static_cast<int>(w), static_cast<int>(N * h));
    BINCV_CHECK_EQ(bincv::cuda::packQuantWith(dImg.constView(), dBlock.view(), N, map),
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
            const uint32_t* a = planes[p].constView().row(y);
            const uint32_t* b = gotBlock.constView().row(p * h + y);
            for (size_t i = 0; i < words; ++i)
                if (a[i] != b[i]) ++bad;
        }
        if (bad != 0) ++badPlanes;
    }
    BINCV_CHECK_EQ(badPlanes, 0u);
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
    const int summaryRc = ::bincv::test::summarize("CUDA custom-packer tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA custom-packer tests", argc, argv);
}
#endif
