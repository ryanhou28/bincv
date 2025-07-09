#pragma once

#include <cstdint>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace bincv {
namespace util {

// save_test_image
// 
// @brief Saves a test image to the specified path in the test/output directory.
void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height);

} // namespace util
} // namespace bincv