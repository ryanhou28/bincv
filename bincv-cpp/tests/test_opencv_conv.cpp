#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "bincv-cpp/binMat.hpp"
#include "bincv-cpp/util.hpp"

int main() {
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

    return 0;
}
