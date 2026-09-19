// Device-versus-host bit-exactness for the median family.
//
// The host library is the truth and the format is shared, so every case here is
// a raw comparison: run both, download, compare. A device kernel that is faster
// and different is not an optimization.
//
// A CUDA translation unit for two reasons. One is that `maj3` now carries
// BINCV_HOST_DEVICE and the binary kernel CALLS IT rather than restating it, so
// the thing worth pinning is that nvcc's device pass gives the host's answer for
// the host's own function -- and that can only be asked from a kernel. The other
// is the padding case below, which builds a deliberately dirty device matrix
// with a raw pitched copy rather than through `upload`, which by contract moves
// pixel bytes only.
//
// ONE THING THIS SUITE CANNOT SEE, STATED RATHER THAN LEFT IMPLICIT: inside a
// CUDA translation unit `ops/medianWide.hpp` takes its PORTABLE arm, because its
// AVX2 gate excludes `__CUDACC__` on purpose. So the comparison here is against
// the host library's portable kernel. What closes the gap is that
// tests/test_median_wide.cpp already holds the host's vector arm and its
// portable arm to each other on the host side; the two halves compose into
// "the device matches the host library", which is the claim.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "bincv/binMat.hpp"
#include "bincv/cuda/deviceBinMat.hpp"
#include "bincv/cuda/median.hpp"
#include "bincv/cuda/transfer.hpp"
#include "bincv/ops/bitslice.hpp"
#include "bincv/ops/denoise.hpp"
#include "bincv/ops/medianWide.hpp"
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

/// A frame with long runs of equal values as well as noise: ties between equal
/// samples are where a selection network could differ from a sort if the median
/// were not a unique value.
template <typename T>
std::vector<T> tiedFrame(size_t w, size_t h, uint64_t seed) {
    std::vector<T> img(w * h);
    for (size_t i = 0; i < img.size(); ++i)
        img[i] = static_cast<T>((splitmix(seed) % 4u) * 17u);
    return img;
}

/// @note The packing is spelled out here rather than taken from
/// `bincv::packBits`, and the reason is a gate hazard rather than a
/// preference: `packBits<PackRule>` is a chain of `if constexpr ... else if
/// constexpr ... else`, and nvcc's front end reports "missing return statement"
/// for such a chain in a DEBUG build. Instantiating it from this CUDA
/// translation unit therefore puts two warnings from a HOST header into the
/// gate's log, where -Werror does not catch them but the log scan does. The
/// three lines below are the same rule and owe nothing to a header.
bincv::BinMat<uint32_t> randomBits(size_t w, size_t h, uint64_t seed) {
    const auto frame = randomFrame<uint8_t>(w, h, seed);
    bincv::BinMat<uint32_t> m(static_cast<int>(w), static_cast<int>(h));
    for (size_t y = 0; y < h; ++y) {
        uint32_t* row = m.view().row(y);
        for (size_t x = 0; x < w; ++x)
            if (frame[y * w + x] > 127u) row[x / 32] |= (uint32_t{1} << (x % 32));
    }
    return m;
}

/// Words that differ, trailing partial word included.
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

/// Rows whose trailing word carries a bit past `width`. The invariant every
/// word-wise reduction depends on.
size_t dirtyPaddingRows(const bincv::BinMat<uint32_t>& m) {
    const size_t words = bincv::impl::minRowWords<uint32_t>(m.getWidth());
    const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(m.getWidth());
    size_t bad = 0;
    for (size_t y = 0; y < m.getHeight(); ++y)
        if ((m.constView().row(y)[words - 1] & ~tail) != 0u) ++bad;
    return bad;
}

/// Runs the device binary median on a host matrix and brings the map back.
bincv::BinMat<uint32_t> deviceDenoise(const bincv::BinMat<uint32_t>& src) {
    const int w = static_cast<int>(src.getWidth());
    const int h = static_cast<int>(src.getHeight());
    bincv::cuda::DeviceBinMat dSrc(w, h), dDst(w, h);
    BINCV_CHECK_EQ(bincv::cuda::upload(src.constView(), dSrc.view()), cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::denoiseMedian3(dSrc.constView(), dDst.view()),
                   cudaSuccess);
    bincv::BinMat<uint32_t> got(w, h);
    BINCV_CHECK_EQ(bincv::cuda::download(dDst.constView(), got.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return got;
}

// ---------------------------------------------------------------------------
// maj3 on the device: the shared helper, evaluated where it is used
// ---------------------------------------------------------------------------

__global__ void maj3TruthKernel(uint32_t* out) {
    // Bit i of a, b, c enumerates i as a three-bit input, so one call covers
    // all eight combinations at once and the result is a fixed word.
    const uint32_t a = 0xAAAAAAAAu, b = 0xCCCCCCCCu, c = 0xF0F0F0F0u;
    out[0] = ::bincv::maj3<uint32_t>(a, b, c);
    // And the two identities the binary kernel leans on.
    out[1] = ::bincv::maj3<uint32_t>(0u, b, c);
    out[2] = static_cast<uint32_t>(b & c);
}

} // namespace

BINCV_TEST(CudaMedian, Maj3OnDeviceMatchesHost) {
    uint32_t* d = nullptr;
    BINCV_CHECK_EQ(cudaMalloc(&d, 3 * sizeof(uint32_t)), cudaSuccess);
    maj3TruthKernel<<<1, 1>>>(d);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
    uint32_t got[3] = {0, 0, 0};
    BINCV_CHECK_EQ(cudaMemcpy(got, d, sizeof(got), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d);

    const uint32_t a = 0xAAAAAAAAu, b = 0xCCCCCCCCu, c = 0xF0F0F0F0u;
    BINCV_CHECK_EQ(got[0], bincv::maj3<uint32_t>(a, b, c));
    // maj3(0, b, c) == b & c is what collapses row 0 to a single AND.
    BINCV_CHECK_EQ(got[1], got[2]);
    BINCV_CHECK_EQ(got[1], bincv::maj3<uint32_t>(0u, b, c));
}

// ---------------------------------------------------------------------------
// denoiseMedian3 against the host, both arms, over the shape matrix
// ---------------------------------------------------------------------------

BINCV_TEST(CudaMedian, DenoiseMatchesHost) {
    // Widths 31/33/63/97/157 are the point: they are the only ones where the
    // tail mask and the last column's zero right-neighbour are observable.
    // 32/64/128 pin the all-ones tail mask. Heights 1 and 2 pin the row-0
    // collapse, which is a different expression.
    const size_t sizes[][2] = {{1, 1},   {2, 1},    {31, 7},   {32, 4},   {33, 2},
                               {63, 3},  {64, 5},   {65, 6},   {96, 3},   {97, 13},
                               {127, 9}, {128, 64}, {129, 11}, {157, 83}, {752, 480},
                               // Around the eighth word boundary: the tail mask
                               // at a multi-word width behaves differently from
                               // the one- and two-word cases above only if the
                               // trailing-word index is computed wrongly.
                               {255, 3}, {256, 3},  {257, 3}};
    for (const auto& s : sizes) {
        const auto src = randomBits(s[0], s[1], 0xD00D0000u + s[0]);
        bincv::BinMat<uint32_t> expect(static_cast<int>(s[0]), static_cast<int>(s[1]));
        bincv::denoiseMedian3<uint32_t>(src.constView(), expect.view());
        const auto got = deviceDenoise(src);
        BINCV_CHECK_EQ(mismatchWords(expect, got), 0u);
        BINCV_CHECK_EQ(dirtyPaddingRows(got), 0u);
    }
}

BINCV_TEST(CudaMedian, DenoiseDirtyPaddingIsTheBorder) {
    // The case that catches the whole class of port errors. Every row's trailing
    // word gets its PADDING bits set on BOTH sides, so the last column's right
    // neighbour has something other than zero to find. The host masks `c`
    // before the shift; a port that masks the store instead passes every
    // interior test and fails here, on exactly the widths that do not end on a
    // word boundary.
    const size_t sizes[][2] = {{33, 5}, {97, 7}, {63, 4}, {157, 9}, {129, 3}, {1, 4}};
    {
        for (const auto& s : sizes) {
            const int w = static_cast<int>(s[0]), h = static_cast<int>(s[1]);
            auto src = randomBits(s[0], s[1], 0xBAD00000u + s[0]);
            const size_t words = bincv::impl::minRowWords<uint32_t>(s[0]);
            const uint32_t tail = bincv::impl::rowTailMask<uint32_t>(s[0]);
            for (size_t y = 0; y < s[1]; ++y) src.view().row(y)[words - 1] |= ~tail;

            bincv::BinMat<uint32_t> expect(w, h);
            bincv::denoiseMedian3<uint32_t>(src.constView(), expect.view());

            // A RAW pitched copy of whole WORDS, not `upload`: upload moves
            // pixel bytes by contract, so it would leave the device's padding
            // clean and the case would not exist.
            bincv::cuda::DeviceBinMat dSrc(w, h), dDst(w, h);
            BINCV_CHECK_EQ(
                cudaMemcpy2D(dSrc.view().ptr, dSrc.getAlignedWidth() * sizeof(uint32_t),
                             src.constView().ptr,
                             src.constView().stride * sizeof(uint32_t),
                             words * sizeof(uint32_t), s[1], cudaMemcpyHostToDevice),
                cudaSuccess);
            BINCV_CHECK_EQ(bincv::cuda::denoiseMedian3(dSrc.constView(), dDst.view()),
                           cudaSuccess);
            bincv::BinMat<uint32_t> got(w, h);
            BINCV_CHECK_EQ(bincv::cuda::download(dDst.constView(), got.view()),
                           cudaSuccess);
            BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
            BINCV_CHECK_EQ(mismatchWords(expect, got), 0u);

            // And the destination's padding must be clean on the DEVICE, read
            // back as whole words rather than through download's pixel bytes.
            std::vector<uint32_t> raw(words * s[1]);
            BINCV_CHECK_EQ(
                cudaMemcpy2D(raw.data(), words * sizeof(uint32_t), dDst.view().ptr,
                             dDst.getAlignedWidth() * sizeof(uint32_t),
                             words * sizeof(uint32_t), s[1], cudaMemcpyDeviceToHost),
                cudaSuccess);
            size_t dirty = 0;
            for (size_t y = 0; y < s[1]; ++y)
                if ((raw[y * words + words - 1] & ~tail) != 0u) ++dirty;
            BINCV_CHECK_EQ(dirty, 0u);
        }
    }
}

BINCV_TEST(CudaMedian, DenoiseEmptyAndDomain) {
    // Empty views are a no-op, not an error -- the host's rule.
    bincv::cuda::DeviceBinMatConstView emptySrc{};
    bincv::cuda::DeviceBinMatView emptyDst{};
    BINCV_CHECK_EQ(bincv::cuda::denoiseMedian3(emptySrc, emptyDst), cudaSuccess);

    // In place is refused rather than silently wrong. THE REFUSAL IS A RETURN
    // VALUE AND AN ASSERTION, and a debug build takes the assertion -- which
    // aborts, by design, because passing an overlapping destination is a
    // programming error and not a runtime condition. So the return value is
    // exercised where the assertion is compiled out; the DEBUG gate's job here
    // is that the assertion BUILDS for nvcc's device pass, which it does by
    // being in this translation unit's launcher at all.
#if !BINCV_DEBUG_CHECKS
    bincv::cuda::DeviceBinMat m(64, 8);
    BINCV_CHECK_EQ(bincv::cuda::denoiseMedian3(m.constView(), m.view()),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
#endif
}

BINCV_TEST(CudaMedian, DenoiseSubWindowViewIsStillCorrect) {
    // A view that names a sub-rectangle of a wider matrix: its rows are a
    // partial width at the parent's stride, so the trailing word carries the
    // NEIGHBOURS' live pixels rather than padding. The host's precondition is
    // that such a view must end on a word boundary, and this case is the
    // device's proof that it honours the same one.
    const size_t parentW = 256, w = 128, h = 9;
    const auto parent = randomBits(parentW, h, 0x5AFE01);
    bincv::BinMat<uint32_t> expect(static_cast<int>(w), static_cast<int>(h));
    const bincv::BinMatConstView<uint32_t> subHost{parent.constView().ptr, w, h,
                                                   parent.constView().stride};
    bincv::denoiseMedian3<uint32_t>(subHost, expect.view());

    bincv::cuda::DeviceBinMat dParent(static_cast<int>(parentW), static_cast<int>(h));
    bincv::cuda::DeviceBinMat dDst(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::upload(parent.constView(), dParent.view()), cudaSuccess);
    const bincv::cuda::DeviceBinMatConstView sub{dParent.view().ptr, w, h,
                                                 dParent.getAlignedWidth()};
    BINCV_CHECK_EQ(bincv::cuda::denoiseMedian3(sub, dDst.view()), cudaSuccess);
    bincv::BinMat<uint32_t> got(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::download(dDst.constView(), got.view()), cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    BINCV_CHECK_EQ(mismatchWords(expect, got), 0u);
    BINCV_CHECK_EQ(dirtyPaddingRows(got), 0u);
}

// ---------------------------------------------------------------------------
// medianWide against the host, both arms, both types, five patterns
// ---------------------------------------------------------------------------

namespace {

/// One (size, pattern, type) case: host and device on identical input, every
/// pixel compared, borders included -- the border is where zero fill lives and
/// an interior-only comparison cannot see it.
template <size_t K, typename T>
size_t wideMismatch(size_t w, size_t h, const bincv::MedianPattern<K>& pattern,
                    const std::vector<T>& frame) {
    std::vector<T> expect(w * h, T{0});
    bincv::medianWide<K, T>(frame.data(), w, h, w, expect.data(), w, pattern);

    bincv::cuda::DeviceImage<T> dSrc(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<T> dDst(static_cast<int>(w), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<T>(frame.data(), w, h, w, dSrc.view()),
                   cudaSuccess);
    BINCV_CHECK_EQ(bincv::cuda::medianWide<K>(dSrc.constView(), dDst.view(), pattern),
                   cudaSuccess);
    std::vector<T> got(w * h, T{0});
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<T>(dDst.constView(), got.data(), w),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);

    size_t bad = 0;
    for (size_t i = 0; i < w * h; ++i)
        if (expect[i] != got[i]) ++bad;
    return bad;
}

// The two shipped neighbourhoods and three adversarial ones.
constexpr bincv::MedianPattern<3> kWideL = bincv::kMedianReferenceL;
constexpr bincv::MedianPattern<5> kWidePlus = bincv::kMedianReferencePlus;
/// K = 3 with a reach of three columns: still inside the fast arm's shape, but
/// every sample lands in a different byte lane of a different word.
constexpr bincv::MedianPattern<3> kWideSpread{{{0, 3}, {0, 0}, {0, -3}}};
/// K = 5 reaching five columns each way, so a four-pixel run's border test is
/// the thing doing the work.
constexpr bincv::MedianPattern<5> kWideFar{{{0, -5}, {-1, 0}, {0, 0}, {1, 0}, {0, 5}}};
/// The 3x3 square: K at its cap, and the pattern with the most comparators.
constexpr bincv::MedianPattern<9> kWideSquare{{{-1, -1}, {-1, 0}, {-1, 1},
                                               {0, -1},  {0, 0},  {0, 1},
                                               {1, -1},  {1, 0},  {1, 1}}};
/// K = 1: the median of one sample is that sample, shifted.
constexpr bincv::MedianPattern<1> kWideShift{{{0, 1}}};

template <typename T>
void sweepWide(uint64_t seed) {
    // 1/3/31/33/97/157 are where a four-pixel run straddles the image edge;
    // 128, 752 and 64 are 4-aligned strides that ADMIT the fast arm; 97, 33,
    // 31 and 157 are strides that are not multiples of 4 and must be refused
    // by the alignment gate and still be correct.
    const size_t sizes[][2] = {{1, 1},   {3, 3},   {31, 7},   {33, 2},
                               {64, 9},  {97, 13}, {128, 64}, {157, 5},
                               {255, 4}, {256, 4}, {752, 64}};
    for (const bool fast : {true, false}) {
        bincv::cuda::impl::medianWideFastArmEnabled() = fast;
        for (const auto& s : sizes) {
            const auto noisy = randomFrame<T>(s[0], s[1], seed + s[0]);
            const auto tied = tiedFrame<T>(s[0], s[1], seed + 7777 + s[0]);
            for (const auto* f : {&noisy, &tied}) {
                BINCV_CHECK_EQ((wideMismatch<3, T>(s[0], s[1], kWideL, *f)), 0u);
                BINCV_CHECK_EQ((wideMismatch<5, T>(s[0], s[1], kWidePlus, *f)), 0u);
                BINCV_CHECK_EQ((wideMismatch<3, T>(s[0], s[1], kWideSpread, *f)), 0u);
                BINCV_CHECK_EQ((wideMismatch<5, T>(s[0], s[1], kWideFar, *f)), 0u);
                BINCV_CHECK_EQ((wideMismatch<9, T>(s[0], s[1], kWideSquare, *f)), 0u);
                BINCV_CHECK_EQ((wideMismatch<1, T>(s[0], s[1], kWideShift, *f)), 0u);
            }
        }
    }
    bincv::cuda::impl::medianWideFastArmEnabled() = true;
}

} // namespace

BINCV_TEST(CudaMedian, WideU8MatchesHostBothArms) { sweepWide<uint8_t>(0x1234000); }

BINCV_TEST(CudaMedian, WideU16MatchesHostBothArms) { sweepWide<uint16_t>(0x5678000); }

BINCV_TEST(CudaMedian, WideSubWindowViewIsStillCorrect) {
    // THE GATE MUST READ THE VIEW, NOT THE CONTAINER. A caller may legally build
    // a sub-window view one byte into a cudaMalloc'd image; its rows are then
    // odd-addressed and a 32-bit load on them is illegal, not merely slow. The
    // stride-only cases above cannot see this -- the container's stride is
    // still a multiple of four.
    const size_t w = 128, h = 17;
    const auto frame = randomFrame<uint8_t>(w, h, 0xA11A5);
    const size_t subW = w - 1;

    std::vector<uint8_t> expect(subW * h, 0);
    bincv::medianWide<3, uint8_t>(frame.data() + 1, subW, h, w, expect.data(), subW,
                                  kWideL);

    bincv::cuda::DeviceImage<uint8_t> dSrc(static_cast<int>(w), static_cast<int>(h));
    bincv::cuda::DeviceImage<uint8_t> dDst(static_cast<int>(subW), static_cast<int>(h));
    BINCV_CHECK_EQ(bincv::cuda::uploadImage<uint8_t>(frame.data(), w, h, w, dSrc.view()),
                   cudaSuccess);
    const bincv::cuda::DeviceImageConstView<uint8_t> sub{dSrc.view().ptr + 1, subW, h, w};
    BINCV_CHECK_EQ(bincv::cuda::medianWide<3>(sub, dDst.view(), kWideL), cudaSuccess);
    std::vector<uint8_t> got(subW * h, 0);
    BINCV_CHECK_EQ(bincv::cuda::downloadImage<uint8_t>(dDst.constView(), got.data(), subW),
                   cudaSuccess);
    BINCV_CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    size_t bad = 0;
    for (size_t i = 0; i < subW * h; ++i)
        if (expect[i] != got[i]) ++bad;
    BINCV_CHECK_EQ(bad, 0u);
}

BINCV_TEST(CudaMedian, WideEmptyAndDomain) {
    bincv::cuda::DeviceImageConstView<uint8_t> emptySrc{};
    bincv::cuda::DeviceImageView<uint8_t> emptyDst{};
    BINCV_CHECK_EQ(bincv::cuda::medianWide<3>(emptySrc, emptyDst, kWideL), cudaSuccess);

    // Aliasing and an out-of-domain offset are refused, not computed. The
    // device's aliasing check is a bounding box and therefore stricter than the
    // host's per-row predicate; the header says so. Both refusals are an
    // assertion AND a return value, and a debug build takes the assertion,
    // which aborts -- so the return value is exercised in the configuration
    // where the assertion is compiled out.
#if !BINCV_DEBUG_CHECKS
    bincv::cuda::DeviceImage<uint8_t> img(64, 8);
    BINCV_CHECK_EQ(bincv::cuda::medianWide<3>(img.constView(), img.view(), kWideL),
                   cudaErrorInvalidValue);

    const bincv::MedianPattern<3> tooFar{{{0, 200}, {0, 0}, {0, -1}}};
    bincv::cuda::DeviceImage<uint8_t> dst(64, 8);
    BINCV_CHECK_EQ(bincv::cuda::medianWide<3>(img.constView(), dst.view(), tooFar),
                   cudaErrorInvalidValue);
    BINCV_CHECK_EQ(cudaGetLastError(), cudaSuccess);
#endif
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
    const int summaryRc = ::bincv::test::summarize("CUDA median tests");
    return (rc != 0 || summaryRc != 0) ? 1 : 0;
}
#else
int main(int argc, char** argv) {
    if (!cudaDevicePresent()) return 77;
    return ::bincv::test::runAll("CUDA median tests", argc, argv);
}
#endif
