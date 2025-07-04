#pragma once

#include <cstdint>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace bincv {
namespace util {

// norm_1b_to_uint8
// 
// @brief Normalizes a 1-bit image to an 8-bit image.
void norm_1b_to_uint8(const uint8_t* h_input, uint8_t* h_output, int width, int height);

// save_test_image
// 
// @brief Saves a test image to the specified path in the test/output directory.
void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height);

} // namespace util
} // namespace bincv