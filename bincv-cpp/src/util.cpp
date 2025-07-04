#include <stdint.h>
#include "bincv-cpp/util.hpp"

namespace bincv {
namespace util {

void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height) {
    // Create a directory for test outputs if it doesn't exist
    std::string outputDir = std::filesystem::current_path().string() + "/tests/output";
    std::filesystem::create_directories(outputDir);

    // Construct the full path for the output image
    std::string outputPath = outputDir + "/" + imageName;

    // Create an OpenCV Mat from the input data
    cv::Mat outputImage(height, width, CV_8UC1, const_cast<uint8_t*>(h_input));

    // Save the image using OpenCV
    cv::imwrite(outputPath, outputImage);
}

} // namespace util
} // namespace bincv