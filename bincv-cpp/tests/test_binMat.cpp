#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/util.hpp"

// @todo: use Catch2 or GTest for testing

int main() {

    // -- Test OpenCV conversion with BinMat --
    // Functions covered:
    // - fromCVMat
    // - toCVMat
    // - toCVMatNormalized
    // - BinMat(width, height)
    std::string imagePath = std::filesystem::path(__FILE__).parent_path().string()
        + "/images/1403715887284058112_bin_normalized.png";

    // Load image in grayscale
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    bincv::BinMat binMat(width, height);

    binMat.fromCVMat(input);

    cv::Mat output;
    cv::Mat outputNormalized;

    binMat.toCVMat(output);
    binMat.toCVMatNormalized(outputNormalized);

    // Save or show result
    bincv::util::save_test_image("test_opencv_conv_output.png", output.data, width, height);
    bincv::util::save_test_image("test_opencv_conv_output_normalized.png", outputNormalized.data, width, height);
    std::cout << "Saved output image: test_opencv_conv_output.png" << std::endl;
    std::cout << "Saved output image: test_opencv_conv_output_normalized.png" << std::endl;

    std::cout << "\n--- BinMat Functionality Tests ---\n";

    // Constructor test
    bincv::BinMat testMat(8, 4);
    std::cout << "Initial matrix (8x4):\n";
    testMat.printMatrix();
    std::cout << "Internal data (packed bytes):\n";
    testMat.printInternalData();

    // Set and get
    testMat.set(0, 0, true);
    testMat.set(3, 7, true);
    if (!testMat.at(0, 0) || !testMat.at(3, 7)) {
        std::cerr << "Set/at/operator() test failed\n";
    } else {
        std::cout << "Set/at/operator() test passed\n";
    }

    // Pointer access
    uint8_t* row1 = testMat.ptr(1);
    row1[0] |= 0xF0;
    if (!testMat.at(1, 4)) {
        std::cerr << "ptr() test failed\n";
    } else {
        std::cout << "ptr() test passed\n";
    }

    // Resize to new shape
    testMat.resize(16, 2);
    std::cout << "Resized matrix (16x2):\n";
    testMat.printMatrix();
    testMat.printMatrix();

    // @todo: test that resize() maintains original data where possible

    // @todo: test resizing to 0, 0

    // Padding
    testMat.set(0, 0, true);
    testMat.set(1, 15, true);
    testMat.pad(1, 2, 2, 1);
    std::cout << "Padded matrix (with 1s in corners):\n";
    testMat.printMatrix();

    // Transpose (returning new object)
    bincv::BinMat tmat = testMat.transposed();
    std::cout << "Transposed matrix:\n";
    tmat.printMatrix();

    // In-place transpose
    tmat.transpose();
    std::cout << "After in-place transpose (should match previous padded matrix):\n";
    tmat.printMatrix();

    // forEachNonZero
    std::cout << "Non-zero pixel coordinates:\n";
    tmat.forEachNonZero([](int y, int x) {
        std::cout << "  (" << y << ", " << x << ")\n";
    });

    std::cout << "\n--- Additional BinMat Case Tests ---\n";

    // Empty BinMat
    bincv::BinMat empty;
    empty.printMatrix(); // should print nothing

    // Invalid dimensions
    try {
        bincv::BinMat invalid(-1, 10);
        std::cerr << "FAILED: negative dimension constructor should throw or assert\n";
    } catch (...) {
        std::cout << "Caught error on invalid constructor\n";
    }

    // Out-of-bounds access
    try {
        testMat.at(999, 0);
        std::cerr << "FAILED: at() did not throw on out-of-bounds\n";
    } catch (const std::out_of_range&) {
        std::cout << "Correctly threw on out-of-bounds at()\n";
    }

    try {
        testMat.set(-1, 5, true);
        std::cerr << "FAILED: set() did not throw on out-of-bounds\n";
    } catch (const std::out_of_range&) {
        std::cout << "Correctly threw on out-of-bounds set()\n";
    }

    // Transposing empty matrix
    bincv::BinMat emptyT = empty.transposed();
    std::cout << "Transposed empty matrix (should be empty):\n";
    emptyT.printMatrix();

    // Transposing non-square
    bincv::BinMat nonsquare(2, 5);
    nonsquare.set(0, 0, true);
    nonsquare.set(4, 1, true);
    std::cout << "Original non-square matrix:\n";
    nonsquare.printMatrix();
    nonsquare.transpose();
    std::cout << "Transposed non-square matrix:\n";
    nonsquare.printMatrix();

    // forEachNonZero with empty matrix
    std::cout << "forEachNonZero with empty matrix:\n";
    try {
        empty.forEachNonZero([](int y, int x) {
            std::cerr << "FAILED: should not enter callback for empty matrix\n";
        });
    } catch (const std::runtime_error& e) {
        std::cout << "Correctly caught error on forEachNonZero with empty matrix: " << e.what() << "\n";
    }

    // @todo: add padding with value of 1

    std::cout << "\nAll basic BinMat tests completed.\n";


    return 0;
}
