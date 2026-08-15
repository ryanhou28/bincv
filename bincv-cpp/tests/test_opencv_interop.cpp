// OpenCV interop tests. Only built when OpenCV is available; the core test suite
// (test_binMat.cpp) covers everything that must work without it.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>

#include <opencv2/opencv.hpp>

#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/util.hpp"
#include "test_util.hpp"

// ---------------------------------------------------------------------------
// One-shot allocation failure
//
// fromCVMat() must build its new buffer before it commits the new dimensions;
// otherwise a failed allocation leaves the matrix describing storage it does not
// have, and at()'s bounds check then validates against dimensions the buffer
// cannot back. Proving that needs an allocation that fails on demand, so the
// global operators are replaced here. Same shape as tests/test_binMat.cpp.
// ---------------------------------------------------------------------------

namespace {
bool g_failNextAlloc = false;
}

void* operator new(std::size_t bytes) {
    if (g_failNextAlloc) {
        g_failNextAlloc = false;   // one shot: arm it immediately before the call
        throw std::bad_alloc();
    }
    // array-new passes SIZE_MAX as its overflow sentinel and expects the allocator
    // to fail; forwarding it to malloc also makes GCC warn once this inlines.
    if (bytes > static_cast<std::size_t>(PTRDIFF_MAX)) throw std::bad_alloc();
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

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
    // 4 words and at(63, 63) would pass its bounds check and read past the end.
    BINCV_CHECK_EQ(m.getWidth(), size_t(8));
    BINCV_CHECK_EQ(m.getHeight(), size_t(2));
    BINCV_CHECK_EQ(m.sizeInWords(), wordsBefore);
    BINCV_CHECK_EQ(m.sizeInWords(), m.getHeight() * m.getAlignedWidth());
    BINCV_CHECK(m.data() == dataBefore);
    BINCV_CHECK(m.at(1, 7));
    BINCV_CHECK_EQ(m.countNonZero(), 1);
    BINCV_CHECK_THROWS(m.at(63, 63), std::out_of_range);

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

} // namespace

int main() {
    std::cout << "=== BinMat OpenCV interop tests ===\n";

    testRoundTrip();
    testAllWordTypes();
    testInvalidInput();
    testFromCVMatAllocationFailure();
    testSampleImage();

    return bincv::test::summarize("BinMat OpenCV interop tests");
}
