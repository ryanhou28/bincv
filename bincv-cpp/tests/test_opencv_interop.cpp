// OpenCV interop tests. Only built when OpenCV is available; the core test suite
// (test_binMat.cpp) covers everything that must work without it.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/ops/corner.hpp"
#include "bincv-cpp/ops/derivative.hpp"
#include "bincv-cpp/quantMat.hpp"
#include "bincv-cpp/util.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// One-shot allocation failure
//
// fromCVMat() must build its new buffer before it commits the new dimensions;
// otherwise a failed allocation leaves the matrix describing storage it does not
// have, and every later read trusts those dimensions. Proving that needs an
// allocation that fails on demand, so the global operators are replaced here.
// Same shape as tests/test_binMat.cpp.
// ---------------------------------------------------------------------------

namespace {
bool g_failNextAlloc = false;

// Both replacement forms allocate through this rather than forwarding to each
// other -- see tests/test_storage.cpp for why (readability of the malloc/free
// pairing; not, as this comment once claimed, a compiler diagnostic).
void* failableAllocate(std::size_t bytes) {
    if (g_failNextAlloc) {
        g_failNextAlloc = false;   // one shot: arm it immediately before the call
        throw std::bad_alloc();
    }
    // array-new passes SIZE_MAX as its overflow sentinel and expects the allocator
    // to fail rather than to forward it to malloc.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) throw std::bad_alloc();
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
} // namespace

void* operator new(std::size_t bytes)   { return failableAllocate(bytes); }
void* operator new[](std::size_t bytes) { return failableAllocate(bytes); }

void operator delete(void* p) noexcept                { std::free(p); }
void operator delete[](void* p) noexcept              { std::free(p); }
void operator delete(void* p, std::size_t) noexcept   { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

// A synthetic pattern round-trips through BinMat without loss.
void testRoundTrip() {
    std::cout << "\n--- cv::Mat round trip ---\n";

    const int width = 70;   // deliberately not a multiple of the word size
    const int height = 9;

    cv::Mat src = cv::Mat::zeros(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if ((x + y) % 3 == 0) src.at<uint8_t>(y, x) = 255;

    bincv::BinMat<> bin;
    bin.fromCVMat(src);
    BINCV_CHECK_EQ(bin.cols(), width);
    BINCV_CHECK_EQ(bin.rows(), height);

    cv::Mat out;
    bin.toCVMat(out);
    BINCV_CHECK_EQ(out.cols, width);
    BINCV_CHECK_EQ(out.rows, height);
    BINCV_CHECK_EQ(out.type(), CV_8UC1);

    int mismatches = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            if ((src.at<uint8_t>(y, x) != 0) != (out.at<uint8_t>(y, x) != 0)) ++mismatches;
    BINCV_CHECK_EQ(mismatches, 0);

    // Non-zero pixel count must agree with OpenCV's own
    BINCV_CHECK_EQ(bin.countNonZero(), cv::countNonZero(src));

    // Normalized conversion maps set bits to 255
    cv::Mat norm;
    bin.toCVMatNormalized(norm);
    double minVal, maxVal;
    cv::minMaxLoc(norm, &minVal, &maxVal);
    BINCV_CHECK(maxVal == 255.0);
    BINCV_CHECK(minVal == 0.0);
    BINCV_CHECK_EQ(cv::countNonZero(norm), cv::countNonZero(src));
}

// Every word width must agree with OpenCV on the same input.
void testAllWordTypes() {
    std::cout << "\n--- Interop across word types ---\n";

    cv::Mat src = cv::Mat::zeros(17, 45, CV_8UC1);
    cv::randu(src, 0, 2);  // 0/1 values

    const int expected = cv::countNonZero(src);

    bincv::BinMat8 m8;   m8.fromCVMat(src);   BINCV_CHECK_EQ(m8.countNonZero(), expected);
    bincv::BinMat16 m16; m16.fromCVMat(src);  BINCV_CHECK_EQ(m16.countNonZero(), expected);
    bincv::BinMat32 m32; m32.fromCVMat(src);  BINCV_CHECK_EQ(m32.countNonZero(), expected);
    bincv::BinMat64 m64; m64.fromCVMat(src);  BINCV_CHECK_EQ(m64.countNonZero(), expected);
}

void testInvalidInput() {
    std::cout << "\n--- Interop argument validation ---\n";

    bincv::BinMat<> bin;
    cv::Mat emptyMat;
    BINCV_CHECK_THROWS(bin.fromCVMat(emptyMat), std::invalid_argument);

    cv::Mat wrongType = cv::Mat::zeros(4, 4, CV_32FC1);
    BINCV_CHECK_THROWS(bin.fromCVMat(wrongType), std::invalid_argument);

    // Converting an empty BinMat yields an empty cv::Mat rather than throwing
    bincv::BinMat<> emptyBin;
    cv::Mat out;
    emptyBin.toCVMat(out);
    BINCV_CHECK(out.empty());
}

// A failed conversion must leave the matrix exactly as it was.
void testFromCVMatAllocationFailure() {
    std::cout << "\n--- fromCVMat is atomic across allocation failure ---\n";

    bincv::BinMat<> m(8, 2);
    m.set(1, 7, true);
    const size_t wordsBefore = m.sizeInWords();
    const bincv::BinMat<>::WordType* const dataBefore = m.data();

    // Built before the allocator is armed, so the sentinel hits fromCVMat's own
    // allocation and nothing else: fromCVMat allocates exactly once, and only
    // argument checks and stride arithmetic run before it.
    cv::Mat big = cv::Mat::zeros(64, 64, CV_8UC1);
    big.at<uint8_t>(0, 0) = 255;

    bool threw = false;
    g_failNextAlloc = true;
    try {
        m.fromCVMat(big);
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    g_failNextAlloc = false;
    BINCV_CHECK(threw);

    // Dimensions, storage, and contents all still describe the ORIGINAL buffer.
    // If the dimensions had been committed first, m would claim to be 64x64 over
    // 4 words, and every subsequent read would trust that claim -- more so since
    // T1.4, because at() no longer bounds-checks in release and so cannot catch
    // it. The word count is the check that matters: sizeInWords() must equal
    // height * alignedWidth for the ORIGINAL shape, not the attempted one --
    // which pins the same property the deleted BINCV_CHECK_THROWS(m.at(63, 63))
    // used to pin, and pins it directly rather than through an accessor. That
    // accessor's own bounds check is covered by tests/test_assert_abort.cpp.
    BINCV_CHECK_EQ(m.getWidth(), size_t(8));
    BINCV_CHECK_EQ(m.getHeight(), size_t(2));
    BINCV_CHECK_EQ(m.sizeInWords(), wordsBefore);
    BINCV_CHECK_EQ(m.sizeInWords(), m.getHeight() * m.getAlignedWidth());
    BINCV_CHECK(m.data() == dataBefore);
    BINCV_CHECK(m.at(1, 7));
    BINCV_CHECK_EQ(m.countNonZero(), 1);

    // The matrix is still usable, and the same call succeeds when memory does.
    m.fromCVMat(big);
    BINCV_CHECK_EQ(m.getWidth(), size_t(64));
    BINCV_CHECK_EQ(m.getHeight(), size_t(64));
    BINCV_CHECK_EQ(m.countNonZero(), 1);
}

// Exercises the real sample image if it is present.
void testSampleImage() {
    std::cout << "\n--- Sample image conversion ---\n";

    const std::string imagePath = std::filesystem::path(__FILE__).parent_path().string()
        + "/images/1403715887284058112_bin_normalized.png";

    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cout << "  (skipped: sample image not found at " << imagePath << ")\n";
        return;
    }

    bincv::BinMat<> bin;
    bin.fromCVMat(input);
    BINCV_CHECK_EQ(bin.cols(), input.cols);
    BINCV_CHECK_EQ(bin.rows(), input.rows);
    BINCV_CHECK_EQ(bin.countNonZero(), cv::countNonZero(input));

    cv::Mat output, outputNormalized;
    bin.toCVMat(output);
    bin.toCVMatNormalized(outputNormalized);

    bincv::util::save_test_image("test_opencv_conv_output.png",
                                 output.data, input.cols, input.rows);
    bincv::util::save_test_image("test_opencv_conv_output_normalized.png",
                                 outputNormalized.data, input.cols, input.rows);
    std::cout << "  saved test_opencv_conv_output{,_normalized}.png\n";

    // Memory: 1 bit per pixel versus OpenCV's 8
    const size_t binBytes = bin.sizeInWords() * sizeof(bincv::BinMat<>::WordType);
    const size_t cvBytes = input.total() * input.elemSize();
    std::cout << "  memory: binCV " << binBytes << " B vs OpenCV " << cvBytes
              << " B (" << (static_cast<double>(cvBytes) / static_cast<double>(binBytes))
              << "x reduction)\n";
    BINCV_CHECK(binBytes < cvBytes);
}

// ---------------------------------------------------------------------------
// T3.7 ON A REAL FRAME -- the "Done when" bullet tests/test_corner.cpp cannot reach
//
// test_corner.cpp is a CORE suite: it must build without OpenCV, so it cannot
// decode a PNG and every frame in it is synthesised. T3.7's first Done-when bullet
// asks for detected corners matched against the reference on real content, and
// that check has to live where an image decoder does. It lives here.
//
// THE REFERENCE, EXPRESSED IN STOCK OPENCV. gftt.cpp with
// `gftt_corner_derivative_type: BINARIZED` is: two [-1, 0, 1] filter2D taps, the
// three product planes, a boxFilter SUM over the block, the min eigenvalue, then
// minMaxLoc / THRESH_TOZERO / dilate / the `val != 0 && val == tmp[x]` scan /
// greaterThanPtr / the greedy spacing filter. Written out below with OpenCV
// primitives, sharing NO code with ops/corner.hpp -- not the response, not the
// comparator, not the selection.
//
// TWO DELIBERATE ALIGNMENTS, so that a disagreement means something:
//   * the box filter uses BORDER_CONSTANT, because a SUM with a zero fill is
//     exactly T3.6's clipped window (D-13). The reference's BORDER_REPLICATE is a
//     separate, documented deviation (ops/corner.hpp, "THE BORDER").
//   * the derivative uses BORDER_REFLECT_101, which is D-19's choice and
//     filter2D's default.
// Everything else is the reference's own arithmetic in the reference's own order.
// ---------------------------------------------------------------------------

/// @brief gftt.cpp's comparator, from SEAL/opencv_internal/include/gftt.hpp.
struct GreaterThanPtr {
    bool operator()(const float* a, const float* b) const

    // Ensure a fully deterministic result of the sort
    { return (*a > *b) ? true : (*a < *b) ? false : (a > b); }
};

void testRealFrameCorners() {
    std::cout << "\n--- T3.7 corners on a real frame, against the gftt.cpp pipeline ---\n";

    const std::string imagePath = std::filesystem::path(__FILE__).parent_path().string()
        + "/images/1403715887284058112_bin_normalized.png";
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cout << "  (skipped: sample image not found at " << imagePath << ")\n";
        return;
    }
    const int w = input.cols, h = input.rows;
    BINCV_CHECK(w > 32 && h > 32);

    // The SAME binary content on both sides: 1 bit per pixel for binCV, CV_8U
    // holding {0, 1} for OpenCV. That is CLAUDE.md's denominator rule.
    cv::Mat bytes(h, w, CV_8U);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            bytes.at<uint8_t>(y, x) = input.at<uint8_t>(y, x) ? uint8_t{1} : uint8_t{0};

    bincv::BinMat<uint32_t> bin;
    bin.fromCVMat(input);
    BINCV_CHECK_EQ(bin.countNonZero(), cv::countNonZero(input));

    bincv::GoodFeaturesParams params;  // SEAL/seal_params.yaml verbatim
    bincv::TernaryMat<uint32_t> dx(static_cast<size_t>(w), static_cast<size_t>(h));
    bincv::TernaryMat<uint32_t> dy(static_cast<size_t>(w), static_cast<size_t>(h));
    bincv::derivativeX(bin, dx);
    bincv::derivativeY(bin, dy);

    std::vector<float> mapStorage(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    bincv::ResponseMap map{mapStorage.data(), static_cast<size_t>(w), static_cast<size_t>(h),
                           static_cast<size_t>(w)};
    // CAPACITY IS NOT maxCorners: the array is also the candidate buffer, so it is
    // sized to the worst case and `candidatesTruncated` must come back false.
    std::vector<bincv::Corner> got(static_cast<size_t>(w - 2) * static_cast<size_t>(h - 2));
    const bincv::CornerResult r =
        bincv::goodFeaturesToTrack(dx, dy, params, map, got.data(), got.size());
    BINCV_CHECK_EQ(r.candidatesTruncated, false);
    BINCV_CHECK(r.count > 32);

    // ---- the reference pipeline, in stock OpenCV -------------------------
    const cv::Mat kx = (cv::Mat_<float>(1, 3) << -1.0f, 0.0f, 1.0f);
    const cv::Mat ky = (cv::Mat_<float>(3, 1) << -1.0f, 0.0f, 1.0f);
    cv::Mat cvDx, cvDy, xx, yy, xy, eig, dilated;
    cv::filter2D(bytes, cvDx, CV_32F, kx, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
    cv::filter2D(bytes, cvDy, CV_32F, ky, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT_101);
    cv::multiply(cvDx, cvDx, xx);
    cv::multiply(cvDy, cvDy, yy);
    cv::multiply(cvDx, cvDy, xy);
    const cv::Size ksz(params.blockSize, params.blockSize);
    const int bd = cv::BORDER_CONSTANT | cv::BORDER_ISOLATED;
    cv::boxFilter(xx, xx, CV_32F, ksz, cv::Point(-1, -1), false, bd);
    cv::boxFilter(yy, yy, CV_32F, ksz, cv::Point(-1, -1), false, bd);
    cv::boxFilter(xy, xy, CV_32F, ksz, cv::Point(-1, -1), false, bd);

    eig.create(h, w, CV_32F);
    size_t mapDiffering = 0;
    for (int y = 0; y < h; ++y) {
        const float* a = xx.ptr<float>(y);
        const float* b = yy.ptr<float>(y);
        const float* c = xy.ptr<float>(y);
        float* e = eig.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            const double s = static_cast<double>(a[x]) + static_cast<double>(b[x]);
            const double d = static_cast<double>(a[x]) - static_cast<double>(b[x]);
            const double cc = static_cast<double>(c[x]);
            e[x] = static_cast<float>(0.5 * (s - std::sqrt(d * d + 4.0 * cc * cc)));
            if (e[x] != mapStorage[static_cast<size_t>(y) * static_cast<size_t>(w) +
                                   static_cast<size_t>(x)])
                ++mapDiffering;
        }
    }
    // The three covariance sums are integers on both sides -- popcounts here, exact
    // float products and an exact float box-filter sum there -- so the maps agree
    // BIT FOR BIT and the documented tolerance on a real frame is ZERO. This is a
    // stronger statement than tier 2 owes and it is asserted as measured, not as a
    // promise: the tier 2 caveat is about cv::cornerMinEigenVal's SOBEL path, which
    // is a different operation.
    BINCV_CHECK_EQ(mapDiffering, static_cast<size_t>(0));

    double maxVal = 0.0;
    cv::minMaxLoc(eig, nullptr, &maxVal, nullptr, nullptr);
    cv::threshold(eig, eig, maxVal * params.qualityLevel, 0.0, cv::THRESH_TOZERO);
    cv::dilate(eig, dilated, cv::Mat());
    std::vector<const float*> candidates;
    for (int y = 1; y + 1 < h; ++y) {
        const float* e = eig.ptr<float>(y);
        const float* t = dilated.ptr<float>(y);
        for (int x = 1; x + 1 < w; ++x)
            if (e[x] != 0.0f && e[x] == t[x]) candidates.push_back(e + x);
    }
    BINCV_CHECK_EQ(r.candidatesRanked, candidates.size());
    std::sort(candidates.begin(), candidates.end(), GreaterThanPtr());

    std::vector<cv::Point> want;
    const double minDistSq = params.minDistance * params.minDistance;
    const float* base = eig.ptr<float>(0);
    const size_t stride = eig.step / sizeof(float);
    for (size_t i = 0; i < candidates.size(); ++i) {
        const size_t ofs = static_cast<size_t>(candidates[i] - base);
        const int y = static_cast<int>(ofs / stride);
        const int x = static_cast<int>(ofs % stride);
        bool good = true;
        for (size_t j = 0; j < want.size(); ++j) {
            const double ddx = static_cast<double>(x) - static_cast<double>(want[j].x);
            const double ddy = static_cast<double>(y) - static_cast<double>(want[j].y);
            if (ddx * ddx + ddy * ddy < minDistSq) {
                good = false;
                break;
            }
        }
        if (!good) continue;
        want.push_back(cv::Point(x, y));
        if (want.size() == static_cast<size_t>(params.maxCorners)) break;
    }

    // SAME CORNERS, SAME ORDER. The tie rule is doing real work here: a binarized
    // 3x3 min-eigenvalue map takes few distinct values, so most of these positions
    // are decided by it, and the port breaks ties by ADDRESS. An implementation
    // whose ties ran the other way could not produce this number.
    BINCV_CHECK_EQ(r.count, want.size());
    size_t differing = 0;
    for (size_t i = 0; i < r.count && i < want.size(); ++i)
        if (got[i].x != want[i].x || got[i].y != want[i].y) ++differing;
    BINCV_CHECK_EQ(differing, static_cast<size_t>(0));

    std::cout << "  " << w << "x" << h << " real frame: " << r.candidatesRanked
              << " NMS survivors, " << r.count << " corners, ALL at the same positions as the "
              << "gftt.cpp pipeline; response map bit-identical over " << (w * h)
              << " pixels\n";
}

} // namespace

BINCV_TEST(OpenCVInterop, RoundTrip)                 { testRoundTrip(); }
BINCV_TEST(OpenCVInterop, AllWordTypes)              { testAllWordTypes(); }
BINCV_TEST(OpenCVInterop, InvalidInput)              { testInvalidInput(); }
BINCV_TEST(OpenCVInterop, FromCVMatAllocationFailure) { testFromCVMatAllocationFailure(); }
BINCV_TEST(OpenCVInterop, SampleImage)               { testSampleImage(); }
BINCV_TEST(OpenCVInterop, RealFrameCorners)          { testRealFrameCorners(); }

BINCV_TEST_MAIN("BinMat OpenCV interop tests")
