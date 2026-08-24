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

// ===========================================================================
// QuantMat<N> conversions (X-47) -- the wide-intermediate bridge
//
// Above the (filter-dependent) bit-width crossover X-46 measured, the fast
// implementation of an 8-bit operation is OpenCV's, and these conversions are
// the way there. Three properties, each load-bearing:
//   1. The transpose-based loops equal a per-pixel reference -- any bit-order
//      slip in transpose8x8's wiring shows here.
//   2. fromCVMat(toCVMatNormalized(m)) == m, exactly, at every N. X-47's rule
//      derives this (255 and MaxValue odd, so no rounding ties); the test is
//      what makes the derivation checkable rather than trusted.
//   3. Padding bits are zero after fromCVMat (D-13) -- a conversion that set
//      them would make every later word-wise reduction over-count.
// ===========================================================================

template <size_t N, typename WordType>
void quantConvertOne(int w, int h) {
    constexpr unsigned maxV = (1u << N) - 1u;
    bincv::QuantMat<N, WordType> m(w, h);
    uint64_t st = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(N) << 32) ^ sizeof(WordType);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            m.set(y, x, static_cast<unsigned>(st >> 33) % (maxV + 1u));
        }
    }

    // 1. Both exports against the per-pixel definition.
    cv::Mat raw, norm;
    m.toCVMat(raw);
    m.toCVMatNormalized(norm);
    BINCV_CHECK_EQ(raw.cols, w);
    BINCV_CHECK_EQ(raw.rows, h);
    BINCV_CHECK_EQ(raw.type(), CV_8UC1);
    size_t badRaw = 0, badNorm = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned v = m.at(y, x);
            if (raw.at<uint8_t>(y, x) != v) ++badRaw;
            const unsigned want = (v * 255u + (maxV >> 1)) / maxV;
            if (norm.at<uint8_t>(y, x) != want) ++badNorm;
        }
    }
    BINCV_CHECK_EQ(badRaw, static_cast<size_t>(0));
    BINCV_CHECK_EQ(badNorm, static_cast<size_t>(0));

    // 2. The exact round trip. `back` carries a NON-DEFAULT row alignment, which
    //    does double duty: it is the only way the padding check below sees any
    //    alignment words past the used ones (with a default-aligned destination
    //    that half of the check is vacuous), and it is what catches a fromCVMat
    //    that rebuilds at word granularity instead of preserving the caller's
    //    stride. An `empty() ? DefaultRowAlignment : getRowAlignment()` guard in
    //    fromCVMat did exactly that, and no test saw it.
    bincv::QuantMat<N, WordType> back(0, 0, 64);
    BINCV_CHECK_EQ(back.getRowAlignment(), static_cast<size_t>(64));
    back.fromCVMat(norm);
    BINCV_CHECK_EQ(back.getRowAlignment(), static_cast<size_t>(64));
    BINCV_CHECK_EQ(back.cols(), w);
    BINCV_CHECK_EQ(back.rows(), h);
    size_t badTrip = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (back.at(y, x) != m.at(y, x)) ++badTrip;
    BINCV_CHECK_EQ(badTrip, static_cast<size_t>(0));

    // 3. Padding bits zero after fromCVMat, in every plane of every row --
    //    both the tail bits of the last used word and any alignment words past it.
    const WordType tail = bincv::impl::rowTailMask<WordType>(static_cast<size_t>(w));
    const size_t used = bincv::impl::minRowWords<WordType>(static_cast<size_t>(w));
    size_t badPad = 0;
    for (size_t p = 0; p < N; ++p) {
        const auto plane = back.constPlane(p);
        for (int y = 0; y < h; ++y) {
            const WordType* row = plane.row(static_cast<size_t>(y));
            if ((row[used - 1] & static_cast<WordType>(~tail)) != 0) ++badPad;
            for (size_t k = used; k < plane.stride; ++k)
                if (row[k] != 0) ++badPad;
        }
    }
    BINCV_CHECK_EQ(badPad, static_cast<size_t>(0));

    // 4. Empty in, empty out -- the branch both exports carry and which no test
    //    reached before.
    bincv::QuantMat<N, WordType> none;
    cv::Mat emptyOut(3, 3, CV_8U);
    none.toCVMat(emptyOut);
    BINCV_CHECK(emptyOut.empty());
    none.toCVMatNormalized(emptyOut);
    BINCV_CHECK(emptyOut.empty());
}

void testQuantMatConversions() {
    // Odd widths so the tail-group path runs; widths past one word so the group
    // loop crosses word boundaries at every word type.
    quantConvertOne<2, uint8_t>(37, 11);
    quantConvertOne<2, uint32_t>(37, 11);
    quantConvertOne<3, uint32_t>(130, 7);
    quantConvertOne<3, uint64_t>(130, 7);
    quantConvertOne<5, uint16_t>(51, 9);
    quantConvertOne<5, uint64_t>(64, 5);
    quantConvertOne<7, uint32_t>(96, 4);
    quantConvertOne<8, uint32_t>(33, 9);
    quantConvertOne<8, uint64_t>(257, 3);
    std::cout << "  QuantMat<N> conversions: transpose == per-pixel reference, "
                 "round trip exact, padding clear, at N in {2,3,5,7,8}\n";
}

void testQuantMatFromCVMatQuantizes() {
    // Every byte value, so the LUT is exercised end to end: 256 columns hit
    // 0..255 exactly once per row.
    cv::Mat all(3, 256, CV_8U);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 256; ++x) all.at<uint8_t>(y, x) = static_cast<uint8_t>(x);

    bincv::QuantMat<3, uint32_t> q3;
    q3.fromCVMat(all);
    size_t bad = 0;
    for (int x = 0; x < 256; ++x) {
        const unsigned want = (static_cast<unsigned>(x) * 7u + 127u) / 255u;
        if (q3.at(0, x) != want) ++bad;
    }
    BINCV_CHECK_EQ(bad, static_cast<size_t>(0));

    // N == 8 is the identity in BOTH directions: fromCVMat then toCVMat must
    // reproduce the input byte for byte -- the interop configuration X-47 times.
    bincv::QuantMat<8, uint32_t> q8;
    q8.fromCVMat(all);
    cv::Mat out;
    q8.toCVMat(out);
    size_t badId = 0;
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 256; ++x)
            if (out.at<uint8_t>(y, x) != all.at<uint8_t>(y, x)) ++badId;
    BINCV_CHECK_EQ(badId, static_cast<size_t>(0));
    std::cout << "  fromCVMat quantizes all 256 byte values correctly; N=8 is the identity\n";
}

void testQuantMatAlignmentSurvivesReuse() {
    // THE REALISTIC TRIGGER for the alignment bug, which is buffer reuse rather
    // than the degenerate constructor: a moved-from matrix is empty but keeps its
    // alignment, so a fromCVMat that consulted empty() rebuilt it at word
    // granularity and silently dropped an opt-in Tier 2 / DMA stride.
    cv::Mat m(4, 100, CV_8U);
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 100; ++x) m.at<uint8_t>(y, x) = static_cast<uint8_t>(x);

    bincv::QuantMat<4, uint32_t> src(100, 4, 64);
    const size_t wantAlign = src.getRowAlignment();
    const size_t wantStride = src.getAlignedWidth();
    BINCV_CHECK_EQ(wantAlign, static_cast<size_t>(64));

    bincv::QuantMat<4, uint32_t> moved = std::move(src);
    BINCV_CHECK(src.empty());
    BINCV_CHECK_EQ(src.getRowAlignment(), wantAlign);
    src.fromCVMat(m);
    BINCV_CHECK_EQ(src.getRowAlignment(), wantAlign);
    BINCV_CHECK_EQ(src.getAlignedWidth(), wantStride);

    // And the values still land correctly at the wider stride.
    size_t bad = 0;
    for (int x = 0; x < 100; ++x)
        if (src.at(0, x) != (static_cast<unsigned>(x) * 15u + 127u) / 255u) ++bad;
    BINCV_CHECK_EQ(bad, static_cast<size_t>(0));
    std::cout << "  fromCVMat preserves an opt-in row alignment across buffer reuse\n";
}

void testQuantMatConversionErrors() {
    bincv::QuantMat<4, uint32_t> q;
    BINCV_CHECK_THROWS(q.fromCVMat(cv::Mat()), std::invalid_argument);
    cv::Mat wrongType(4, 4, CV_32F);
    BINCV_CHECK_THROWS(q.fromCVMat(wrongType), std::invalid_argument);
    // A rejected fromCVMat leaves the matrix untouched. NOTE WHAT THIS DOES AND
    // DOES NOT COVER: the throw comes from the CV_8UC1 type check, BEFORE any
    // allocation, so this exercises argument validation rather than the
    // commit-last property proper. Commit-last is about a FAILED ALLOCATION
    // mid-conversion, which needs an injected bad_alloc -- see
    // testFromCVMatAllocationFailure for the N == 1 shape of that test. Claiming
    // this case proved commit-last would be claiming coverage that is not here.
    bincv::QuantMat<4, uint32_t> keep(5, 5);
    keep.set(2, 2, 9u);
    BINCV_CHECK_THROWS(keep.fromCVMat(wrongType), std::invalid_argument);
    BINCV_CHECK_EQ(keep.at(2, 2), 9u);
}

} // namespace

BINCV_TEST(OpenCVInterop, RoundTrip)                 { testRoundTrip(); }
BINCV_TEST(OpenCVInterop, AllWordTypes)              { testAllWordTypes(); }
BINCV_TEST(OpenCVInterop, InvalidInput)              { testInvalidInput(); }
BINCV_TEST(OpenCVInterop, FromCVMatAllocationFailure) { testFromCVMatAllocationFailure(); }
BINCV_TEST(OpenCVInterop, SampleImage)               { testSampleImage(); }
BINCV_TEST(OpenCVInterop, RealFrameCorners)          { testRealFrameCorners(); }
BINCV_TEST(OpenCVInterop, QuantMatConversions)       { testQuantMatConversions(); }
BINCV_TEST(OpenCVInterop, QuantMatQuantizeLaw)       { testQuantMatFromCVMatQuantizes(); }
BINCV_TEST(OpenCVInterop, QuantMatConversionErrors)  { testQuantMatConversionErrors(); }
BINCV_TEST(OpenCVInterop, QuantMatAlignmentReuse)    { testQuantMatAlignmentSurvivesReuse(); }

BINCV_TEST_MAIN("BinMat OpenCV interop tests")
