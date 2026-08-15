#pragma once

// Image I/O helpers for tests and examples. These genuinely need OpenCV, so they
// are not part of the dependency-free core.

#ifdef BINCV_WITH_OPENCV

#include <cstdint>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace bincv {
namespace util {

// @brief Saves a test image to the specified path in the test/output directory.
void save_test_image(const std::string& imageName, const uint8_t* h_input, int width, int height);

} // namespace util
} // namespace bincv

#endif // BINCV_WITH_OPENCV