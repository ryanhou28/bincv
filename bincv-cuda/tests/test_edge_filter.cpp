#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "bincv-cuda/edge_filter.hpp"

int main() {
    std::string imagePath = std::filesystem::path(__FILE__).parent_path().string()
        + "/images/1403715887284058112.png";
    int threshold = 30;

    // Load image in grayscale
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    cv::Mat output(height, width, CV_8UC1, cv::Scalar(0));

    // Run CUDA horizontal edge filter
    bincv::runHorizontalEdgeFilter(input.data, output.data, width, height, threshold);

    // Save or show result
    std::string outputDir = std::filesystem::current_path().string() + "/tests/output";
    std::filesystem::create_directories(outputDir);

    std::string outputPath = outputDir + "/output_edge.png";
    cv::imwrite(outputPath, output);
    std::cout << "Saved output image to " << outputPath << std::endl;

    return 0;
}
