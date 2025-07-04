#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "bincv-cuda/edge_filter.hpp"
#include "bincv-cuda/util.hpp"

int main() {
    std::string imagePath = std::filesystem::path(__FILE__).parent_path().string()
        + "/images/1403715887284058112.png";
    int threshold = 17;

    // Load image in grayscale
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;

    cv::Mat horEdges(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat verEdges(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat SEALEdges(height, width, CV_8UC1, cv::Scalar(0));

    // Run CUDA edge filters
    bincv::horizontalEdgeFilter8b(input.data, horEdges.data, width, height, threshold);
    bincv::verticalEdgeFilter8b(input.data, verEdges.data, width, height, threshold);
    bincv::SEALEdgeFilter8b(input.data, SEALEdges.data, width, height, threshold);

    // Normalize the output to 8-bit
    cv::Mat normHorEdges(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat normVerEdges(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat normSEALEdges(height, width, CV_8UC1, cv::Scalar(0));
    bincv::util::norm_1b_to_uint8(horEdges.data, normHorEdges.data, width, height);
    bincv::util::norm_1b_to_uint8(verEdges.data, normVerEdges.data, width, height);
    bincv::util::norm_1b_to_uint8(SEALEdges.data, normSEALEdges.data, width, height);

    // Save or show result
    bincv::util::save_test_image("horizontal_edges.png", normHorEdges.data, width, height);
    bincv::util::save_test_image("vertical_edges.png", normVerEdges.data, width, height);
    bincv::util::save_test_image("SEAL_edges.png", normSEALEdges.data, width, height);
    std::cout << "Saved output image: horizontal_edges.png" << std::endl;
    std::cout << "Saved output image: vertical_edges.png" << std::endl;
    std::cout << "Saved output image: SEAL_edges.png" << std::endl;

    return 0;
}
